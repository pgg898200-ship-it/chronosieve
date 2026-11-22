#include "cs_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct cs_fuser {
    cs_fuser_config_t cfg;
    size_t K;
    size_t K_pad;

    /* 64-byte aligned SoA arrays for NEON/AVX2 vectorization across K agents */
    double *weights;      /* Normalized BOA weights w_{t,k}, sum = 1 */
    double *var_acc;      /* Cumulative squared excess loss V_{t,k} */
    double *excess;       /* Scratch excess loss r_{t,k} */
    double *exponents;    /* Scratch BOA exponent z_k */
    double *rogue_stats;  /* Latest per-agent Catoni E-SR statistic */
    uint8_t *rogue_flags; /* 1 if agent k triggered Catoni drift attribution */

    /* Per-agent residual Catoni E-detectors + optional auto-calibration buffers */
    cs_edetector_t **rogue_dets;
    double *cal_bufs;     /* [K x cal_window] when auto-calibrating */
    uint32_t *cal_counts; /* Active samples collected per agent */

    double loss_scale;    /* Running max |r_{t,k}| (E_t in Bernstein bound) */
    double last_pred;
    uint64_t step_count;
};

void cs_fuser_config_default(cs_fuser_config_t *cfg, size_t n_agents) {
    if (!cfg) {
        return;
    }
    cfg->n_agents = n_agents;
    cfg->share_base = 0.01;
    cfg->share_shift = 0.35;
    cfg->shift_var_decay = 0.25;
    cfg->rogue_threshold = 200.0;
    cfg->rogue_mu0 = NAN;
    cfg->rogue_sigma = NAN;
    cfg->cal_window = 48U;
    cfg->use_abs_loss = false;
    cfg->quarantine_rogue = true;
}

static bool cs_fuser_validate_cfg(const cs_fuser_config_t *cfg) {
    if (!cfg || cfg->n_agents == 0) {
        return false;
    }
    if (!isfinite(cfg->share_base) || cfg->share_base < 0.0 || cfg->share_base > 1.0) {
        return false;
    }
    if (!isfinite(cfg->share_shift) || cfg->share_shift < 0.0 || cfg->share_shift > 1.0) {
        return false;
    }
    if (!isfinite(cfg->shift_var_decay) || cfg->shift_var_decay < 0.0 || cfg->shift_var_decay > 1.0) {
        return false;
    }
    if (!isfinite(cfg->rogue_threshold) || cfg->rogue_threshold <= 1.0) {
        return false;
    }
    return true;
}

cs_fuser_t *cs_fuser_create(const cs_fuser_config_t *cfg) {
    if (!cs_fuser_validate_cfg(cfg)) {
        return NULL;
    }

    cs_fuser_t *fus = (cs_fuser_t *)calloc(1U, sizeof(cs_fuser_t));
    if (!fus) {
        return NULL;
    }
    fus->cfg = *cfg;
    if (fus->cfg.cal_window < 16U) {
        fus->cfg.cal_window = 16U;
    }
    fus->K = cfg->n_agents;
    fus->K_pad = ((fus->K + 7U) / 8U) * 8U;

    fus->weights = (double *)cs_aligned_calloc(64U, fus->K_pad, sizeof(double));
    fus->var_acc = (double *)cs_aligned_calloc(64U, fus->K_pad, sizeof(double));
    fus->excess = (double *)cs_aligned_calloc(64U, fus->K_pad, sizeof(double));
    fus->exponents = (double *)cs_aligned_calloc(64U, fus->K_pad, sizeof(double));
    fus->rogue_stats = (double *)calloc(fus->K, sizeof(double));
    fus->rogue_flags = (uint8_t *)calloc(fus->K, sizeof(uint8_t));
    fus->rogue_dets = (cs_edetector_t **)calloc(fus->K, sizeof(cs_edetector_t *));
    fus->cal_counts = (uint32_t *)calloc(fus->K, sizeof(uint32_t));

    if (!fus->weights || !fus->var_acc || !fus->excess || !fus->exponents ||
        !fus->rogue_stats || !fus->rogue_flags || !fus->rogue_dets || !fus->cal_counts) {
        cs_fuser_destroy(fus);
        return NULL;
    }

    bool known_baseline = isfinite(cfg->rogue_mu0) && isfinite(cfg->rogue_sigma) &&
                          cfg->rogue_sigma > 0.0;
    /* Family-wise Bonferroni-safe threshold across K simultaneous agents (FW-ARL >= rogue_threshold) */
    double eff_rogue_thr = cfg->rogue_threshold * (double)fus->K;
    if (known_baseline) {
        for (size_t k = 0; k < fus->K; ++k) {
            cs_edet_config_t ecfg;
            cs_edet_config_default(&ecfg);
            ecfg.family = CS_FAMILY_CATONI;
            ecfg.accum = CS_ACCUM_ESR;
            ecfg.side = CS_SIDE_UPPER;
            ecfg.threshold = eff_rogue_thr;
            ecfg.mu0 = cfg->rogue_mu0;
            ecfg.sigma = cfg->rogue_sigma;
            ecfg.auto_reset = false;
            fus->rogue_dets[k] = cs_edet_create(&ecfg);
            if (!fus->rogue_dets[k]) {
                cs_fuser_destroy(fus);
                return NULL;
            }
        }
    } else {
        fus->cal_bufs = (double *)calloc(fus->K * (size_t)fus->cfg.cal_window, sizeof(double));
        if (!fus->cal_bufs) {
            cs_fuser_destroy(fus);
            return NULL;
        }
    }

    cs_fuser_reset(fus);
    return fus;
}

void cs_fuser_destroy(cs_fuser_t *fus) {
    if (!fus) {
        return;
    }
    if (fus->rogue_dets) {
        for (size_t k = 0; k < fus->K; ++k) {
            cs_edet_destroy(fus->rogue_dets[k]);
        }
        free(fus->rogue_dets);
    }
    cs_aligned_free(fus->weights);
    cs_aligned_free(fus->var_acc);
    cs_aligned_free(fus->excess);
    cs_aligned_free(fus->exponents);
    free(fus->rogue_stats);
    free(fus->rogue_flags);
    free(fus->cal_bufs);
    free(fus->cal_counts);
    free(fus);
}

void cs_fuser_reset(cs_fuser_t *fus) {
    if (!fus) {
        return;
    }
    double inv_k = 1.0 / (double)fus->K;
    for (size_t k = 0; k < fus->K; ++k) {
        fus->weights[k] = inv_k;
        fus->var_acc[k] = 0.0;
        fus->excess[k] = 0.0;
        fus->exponents[k] = 0.0;
        fus->rogue_stats[k] = 0.0;
        fus->rogue_flags[k] = 0U;
        fus->cal_counts[k] = 0U;
        if (fus->cal_bufs) {
            /* Auto-calibrated detectors will be rebuilt from fresh calibration */
            cs_edet_destroy(fus->rogue_dets[k]);
            fus->rogue_dets[k] = NULL;
        } else if (fus->rogue_dets[k]) {
            cs_edet_reset(fus->rogue_dets[k]);
        }
    }
    fus->loss_scale = 1.0e-12;
    fus->last_pred = 0.0;
    fus->step_count = 0ULL;
}

void cs_fuser_clear_rogue(cs_fuser_t *fus, int32_t agent_idx) {
    if (!fus) {
        return;
    }
    if (agent_idx < 0) {
        for (size_t k = 0; k < fus->K; ++k) {
            fus->rogue_flags[k] = 0U;
            fus->rogue_stats[k] = 0.0;
            if (fus->rogue_dets[k]) {
                cs_edet_reset(fus->rogue_dets[k]);
            }
        }
    } else if ((size_t)agent_idx < fus->K) {
        size_t k = (size_t)agent_idx;
        fus->rogue_flags[k] = 0U;
        fus->rogue_stats[k] = 0.0;
        if (fus->rogue_dets[k]) {
            cs_edet_reset(fus->rogue_dets[k]);
        }
    }
}

/* Compute effective active prediction weights p_{t,k} respecting sleeping mask & quarantine. */
static size_t cs_fuser_compute_pred_internal(
    const cs_fuser_t *fus,
    const double *preds,
    const uint8_t *active,
    double *out_pred,
    double *out_wsum_eff,
    bool *out_used_quarantine)
{
    const size_t K = fus->K;
    size_t active_cnt = 0;
    size_t healthy_active_cnt = 0;

    for (size_t k = 0; k < K; ++k) {
        if (!active || active[k]) {
            active_cnt++;
            if (!fus->rogue_flags[k]) {
                healthy_active_cnt++;
            }
        }
    }

    if (active_cnt == 0) {
        *out_pred = fus->last_pred;
        if (out_wsum_eff) {
            *out_wsum_eff = 0.0;
        }
        if (out_used_quarantine) {
            *out_used_quarantine = false;
        }
        return 0;
    }

    bool apply_q = (fus->cfg.quarantine_rogue && healthy_active_cnt > 0);
    double wsum = 0.0;
    double ysum = 0.0;
    size_t used_cnt = 0;

    for (size_t k = 0; k < K; ++k) {
        if (active && !active[k]) {
            continue;
        }
        if (apply_q && fus->rogue_flags[k]) {
            continue;
        }
        double wk = fus->weights[k];
        wsum += wk;
        ysum += wk * preds[k];
        used_cnt++;
    }

    if (wsum > 0.0) {
        *out_pred = ysum / wsum;
    } else {
        /* Uniform fallback over eligible active agents */
        double avg = 0.0;
        for (size_t k = 0; k < K; ++k) {
            if (active && !active[k]) {
                continue;
            }
            if (apply_q && fus->rogue_flags[k]) {
                continue;
            }
            avg += preds[k];
        }
        *out_pred = avg / (double)used_cnt;
    }

    if (out_wsum_eff) {
        *out_wsum_eff = wsum;
    }
    if (out_used_quarantine) {
        *out_used_quarantine = apply_q;
    }
    return active_cnt;
}

cs_status_t cs_fuser_predict(
    const cs_fuser_t *fus,
    const double *preds,
    const uint8_t *active,
    double *out_pred)
{
    if (!fus || !preds || !out_pred) {
        return CS_ERR_INVALID;
    }
    for (size_t k = 0; k < fus->K; ++k) {
        if ((!active || active[k]) && !isfinite(preds[k])) {
            return CS_ERR_NONFINITE;
        }
    }
    cs_fuser_compute_pred_internal(fus, preds, active, out_pred, NULL, NULL);
    return CS_OK;
}

static inline double cs_fuser_loss_safe(double diff, bool use_abs) {
    double e = fabs(diff);
    if (use_abs) {
        return (e > 1.0e140) ? 1.0e140 : e;
    }
    if (e > 1.0e70) {
        e = 1.0e70;
    }
    return e * e;
}

cs_status_t cs_fuser_step(
    cs_fuser_t *fus,
    const double *preds,
    const uint8_t *active,
    double y_true,
    bool shift_detected,
    cs_fuser_result_t *out)
{
    if (!fus || !preds) {
        return CS_ERR_INVALID;
    }
    if (!isfinite(y_true)) {
        return CS_ERR_NONFINITE;
    }
    const size_t K = fus->K;
    for (size_t k = 0; k < K; ++k) {
        if ((!active || active[k]) && !isfinite(preds[k])) {
            return CS_ERR_NONFINITE;
        }
    }

    double y_hat = fus->last_pred;
    size_t active_cnt = cs_fuser_compute_pred_internal(
        fus, preds, active, &y_hat, NULL, NULL);

    fus->step_count++;

    if (active_cnt == 0) {
        if (out) {
            double max_rs = 0.0;
            int32_t worst = -1;
            size_t rcnt = 0;
            for (size_t j = 0; j < K; ++j) {
                if (fus->rogue_stats[j] > max_rs) {
                    max_rs = fus->rogue_stats[j];
                    worst = (int32_t)j;
                }
                if (fus->rogue_flags[j]) {
                    rcnt++;
                }
            }
            memset(out, 0, sizeof(*out));
            out->prediction = y_hat;
            out->max_rogue_stat = max_rs;
            out->worst_agent = worst;
            out->rogue_count = rcnt;
            out->any_rogue = (rcnt > 0);
            out->step_count = fus->step_count;
        }
        return CS_OK;
    }

    fus->last_pred = y_hat;
    double fused_loss = cs_fuser_loss_safe(y_hat - y_true, fus->cfg.use_abs_loss);

    /* Update per-agent residual Catoni E-detectors BEFORE BOA scale update so a sudden
     * extreme hallucination triggers quarantine immediately without poisoning E_t forever. */
    double max_rogue_stat = 0.0;
    int32_t worst_agent = -1;
    size_t rogue_count = 0;
    size_t healthy_active_cnt = 0;

    for (size_t j = 0; j < K; ++j) {
        if (!active || active[j]) {
            double abs_res = fabs(y_true - preds[j]);
            if (isfinite(abs_res)) {
                if (fus->rogue_dets[j]) {
                    cs_edet_result_t eres;
                    if (cs_edet_step(fus->rogue_dets[j], abs_res, &eres) == CS_OK) {
                        fus->rogue_stats[j] = eres.stat;
                        if (eres.alarmed) {
                            fus->rogue_flags[j] = 1U;
                        }
                    }
                } else if (fus->cal_bufs) {
                    uint32_t cnt = fus->cal_counts[j];
                    double *cbuf = fus->cal_bufs + j * (size_t)fus->cfg.cal_window;
                    if (cnt < fus->cfg.cal_window) {
                        cbuf[cnt] = abs_res;
                        cnt++;
                        fus->cal_counts[j] = cnt;
                    }
                    if (cnt >= fus->cfg.cal_window) {
                        cs_edet_config_t ecfg;
                        if (cs_edet_calibrate_catoni(
                                cbuf, cnt, fus->cfg.rogue_threshold * (double)K,
                                CS_SIDE_UPPER, 2.5, &ecfg) == CS_OK) {
                            fus->rogue_dets[j] = cs_edet_create(&ecfg);
                        }
                    }
                }
            }
            if (!fus->rogue_flags[j]) {
                healthy_active_cnt++;
            }
        }

        if (fus->rogue_stats[j] > max_rogue_stat) {
            max_rogue_stat = fus->rogue_stats[j];
            worst_agent = (int32_t)j;
        }
        if (fus->rogue_flags[j]) {
            rogue_count++;
        }
    }

    bool apply_q_upd = (fus->cfg.quarantine_rogue && healthy_active_cnt > 0);

    /* Compute baseline weighted loss l_bar across active (non-quarantined) agents */
    double wsum_upd = 0.0;
    for (size_t k = 0; k < K; ++k) {
        if (active && !active[k]) {
            continue;
        }
        if (apply_q_upd && fus->rogue_flags[k]) {
            continue;
        }
        wsum_upd += fus->weights[k];
    }

    double l_bar = 0.0;
    size_t elig_cnt = 0;
    for (size_t k = 0; k < K; ++k) {
        if (active && !active[k]) {
            continue;
        }
        if (apply_q_upd && fus->rogue_flags[k]) {
            continue;
        }
        double lk = cs_fuser_loss_safe(preds[k] - y_true, fus->cfg.use_abs_loss);
        if (wsum_upd > 0.0) {
            l_bar += (fus->weights[k] / wsum_upd) * lk;
        } else {
            l_bar += lk;
        }
        elig_cnt++;
    }
    if (wsum_upd <= 0.0 && elig_cnt > 0) {
        l_bar /= (double)elig_cnt;
    }

    /* Sleeping-expert zero-regret imputation: r_{t,k} = 0 for sleeping agents.
     * Quarantined rogue agents are excluded from inflating loss_scale E_t, and their
     * excess loss is clamped to [-E_t, +E_t] so they are penalized without poisoning E_t. */
    double step_max_r = 0.0;
    for (size_t k = 0; k < K; ++k) {
        if ((active && !active[k]) || (apply_q_upd && fus->rogue_flags[k])) {
            fus->excess[k] = 0.0;
            continue;
        }
        double lk = cs_fuser_loss_safe(preds[k] - y_true, fus->cfg.use_abs_loss);
        double rk = lk - l_bar;
        fus->excess[k] = rk;
        double ar = fabs(rk);
        if (ar > step_max_r) {
            step_max_r = ar;
        }
    }
    double old_loss_scale = fus->loss_scale;
    if (shift_detected) {
        /* Regime shift: re-anchor loss_scale E_t to the new regime's empirical excess scale */
        fus->loss_scale = (step_max_r > 1.0e-12) ? step_max_r : 1.0e-12;
    } else if (step_max_r > fus->loss_scale) {
        fus->loss_scale = step_max_r;
    }
    if (apply_q_upd) {
        const double q_cap = fus->loss_scale;
        for (size_t k = 0; k < K; ++k) {
            if ((!active || active[k]) && fus->rogue_flags[k]) {
                double lk = cs_fuser_loss_safe(preds[k] - y_true, fus->cfg.use_abs_loss);
                double rk = lk - l_bar;
                if (rk > q_cap) {
                    rk = q_cap;
                } else if (rk < -q_cap) {
                    rk = -q_cap;
                }
                fus->excess[k] = rk;
            }
        }
    }

    /* Shift-coupled variance decay (dimensionally scaled with E_t^2) so BOA learning rates
     * immediately re-open on regime change even after an extreme scale drop. */
    double var_decay = 1.0;
    if (shift_detected) {
        var_decay = fus->cfg.shift_var_decay;
        if (old_loss_scale > fus->loss_scale) {
            double ratio = fus->loss_scale / old_loss_scale;
            var_decay *= (ratio * ratio);
        }
    }
    const double gamma_eff = shift_detected ? fus->cfg.share_shift : fus->cfg.share_base;
    const double E_t = fus->loss_scale;
    const double E_t2 = E_t * E_t;
    const double eta_cap = 0.5 / E_t;
    const double log_K = log((double)(K > 2U ? K : 2U));

    /* Vectorized BOA second-order exponent computation: z_k = -eta_k * r_k - (eta_k * r_k)^2 */
    double z_max = -DBL_MAX;
    size_t k = 0;
#if defined(CS_SIMD_NEON)
    const float64x2_t vdecay = vdupq_n_f64(var_decay);
    const float64x2_t vcap = vdupq_n_f64(eta_cap);
    const float64x2_t vlogk = vdupq_n_f64(log_K);
    const float64x2_t ve2 = vdupq_n_f64(E_t2);
    float64x2_t vzmax = vdupq_n_f64(-DBL_MAX);

    for (; k + 2U <= K; k += 2U) {
        float64x2_t r = vld1q_f64(fus->excess + k);
        float64x2_t r2 = vmulq_f64(r, r);
        float64x2_t v = vfmaq_f64(r2, vld1q_f64(fus->var_acc + k), vdecay);
        vst1q_f64(fus->var_acc + k, v);

        float64x2_t eta = vminq_f64(vcap, vsqrtq_f64(vdivq_f64(vlogk, vaddq_f64(v, ve2))));
        float64x2_t er = vmulq_f64(eta, r);
        float64x2_t z = vnegq_f64(vfmaq_f64(er, er, er));
        vst1q_f64(fus->exponents + k, z);
        vzmax = vmaxq_f64(vzmax, z);
    }
    if (k > 0U) {
        z_max = vmaxvq_f64(vzmax);
    }
#elif defined(CS_SIMD_AVX2)
    const __m256d vdecay = _mm256_set1_pd(var_decay);
    const __m256d vcap = _mm256_set1_pd(eta_cap);
    const __m256d vlogk = _mm256_set1_pd(log_K);
    const __m256d ve2 = _mm256_set1_pd(E_t2);
    const __m256d vzero = _mm256_setzero_pd();
    __m256d vzmax = _mm256_set1_pd(-DBL_MAX);

    for (; k + 4U <= K; k += 4U) {
        __m256d r = _mm256_loadu_pd(fus->excess + k);
        __m256d r2 = _mm256_mul_pd(r, r);
        __m256d v = cs_avx_fmadd(_mm256_loadu_pd(fus->var_acc + k), vdecay, r2);
        _mm256_storeu_pd(fus->var_acc + k, v);

        __m256d eta = _mm256_min_pd(vcap, _mm256_sqrt_pd(_mm256_div_pd(vlogk, _mm256_add_pd(v, ve2))));
        __m256d er = _mm256_mul_pd(eta, r);
        __m256d z = _mm256_sub_pd(vzero, cs_avx_fmadd(er, er, er));
        _mm256_storeu_pd(fus->exponents + k, z);
        vzmax = _mm256_max_pd(vzmax, z);
    }
    if (k > 0U) {
        double zt[4];
        _mm256_storeu_pd(zt, vzmax);
        for (int j = 0; j < 4; ++j) {
            if (zt[j] > z_max) {
                z_max = zt[j];
            }
        }
    }
#endif
    for (; k < K; ++k) {
        double r = fus->excess[k];
        double v = fus->var_acc[k] * var_decay + r * r;
        fus->var_acc[k] = v;
        double eta = sqrt(log_K / (v + E_t2));
        if (eta > eta_cap) {
            eta = eta_cap;
        }
        double er = eta * r;
        double z = -(er + er * er);
        fus->exponents[k] = z;
        if (z > z_max) {
            z_max = z;
        }
    }

    /* Vectorized SIMD Estrin/Cephes exp() update & weight sum */
    double new_wsum = 0.0;
    k = 0;
#if defined(CS_SIMD_NEON)
    const float64x2_t vzmax_b = vdupq_n_f64(z_max);
    float64x2_t vwsum = vdupq_n_f64(0.0);
    for (; k + 2U <= K; k += 2U) {
        float64x2_t z = vsubq_f64(vld1q_f64(fus->exponents + k), vzmax_b);
        float64x2_t w = vmulq_f64(vld1q_f64(fus->weights + k), cs_vexp_f64(z));
        vst1q_f64(fus->weights + k, w);
        vwsum = vaddq_f64(vwsum, w);
    }
    new_wsum = vaddvq_f64(vwsum);
#elif defined(CS_SIMD_AVX2)
    const __m256d vzmax_b = _mm256_set1_pd(z_max);
    __m256d vwsum = _mm256_setzero_pd();
    for (; k + 4U <= K; k += 4U) {
        __m256d z = _mm256_sub_pd(_mm256_loadu_pd(fus->exponents + k), vzmax_b);
        __m256d w = _mm256_mul_pd(_mm256_loadu_pd(fus->weights + k), cs_vexp_f64(z));
        _mm256_storeu_pd(fus->weights + k, w);
        vwsum = _mm256_add_pd(vwsum, w);
    }
    double wt[4];
    _mm256_storeu_pd(wt, vwsum);
    new_wsum = (wt[0] + wt[1]) + (wt[2] + wt[3]);
#endif
    for (; k < K; ++k) {
        double w = fus->weights[k] * cs_scalar_exp(fus->exponents[k] - z_max);
        fus->weights[k] = w;
        new_wsum += w;
    }

    /* Vectorized normalization + shift-coupled Fixed-Share mixing */
    const double inv_K = 1.0 / (double)K;
    if (new_wsum > 0.0 && isfinite(new_wsum)) {
        const double scale = (1.0 - gamma_eff) / new_wsum;
        const double bias = gamma_eff * inv_K;
        k = 0;
#if defined(CS_SIMD_NEON)
        const float64x2_t vscale = vdupq_n_f64(scale);
        const float64x2_t vbias = vdupq_n_f64(bias);
        for (; k + 2U <= K; k += 2U) {
            float64x2_t w = vfmaq_f64(vbias, vscale, vld1q_f64(fus->weights + k));
            vst1q_f64(fus->weights + k, w);
        }
#elif defined(CS_SIMD_AVX2)
        const __m256d vscale = _mm256_set1_pd(scale);
        const __m256d vbias = _mm256_set1_pd(bias);
        for (; k + 4U <= K; k += 4U) {
            __m256d w = cs_avx_fmadd(vscale, _mm256_loadu_pd(fus->weights + k), vbias);
            _mm256_storeu_pd(fus->weights + k, w);
        }
#endif
        for (; k < K; ++k) {
            fus->weights[k] = bias + scale * fus->weights[k];
        }
    } else {
        for (size_t j = 0; j < K; ++j) {
            fus->weights[j] = inv_K;
        }
    }

    if (out) {
        out->prediction = y_hat;
        out->fused_loss = fused_loss;
        out->max_rogue_stat = max_rogue_stat;
        out->worst_agent = worst_agent;
        out->active_count = active_cnt;
        out->rogue_count = rogue_count;
        out->step_count = fus->step_count;
        out->any_rogue = (rogue_count > 0);
    }

    return CS_OK;
}

cs_status_t cs_fuser_step_batch(
    cs_fuser_t *fus,
    const double *preds_mat,
    const uint8_t *active_mat,
    const double *y_true,
    const uint8_t *shifts,
    size_t n_steps,
    double *out_preds,
    double *out_weights_final,
    uint8_t *out_rogue_any,
    cs_fuser_result_t *last_out)
{
    if (!fus || (n_steps > 0 && (!preds_mat || !y_true))) {
        return CS_ERR_INVALID;
    }
    const size_t K = fus->K;
    cs_fuser_result_t res;
    memset(&res, 0, sizeof(res));
    res.worst_agent = -1;

    for (size_t t = 0; t < n_steps; ++t) {
        const double *p_row = preds_mat + t * K;
        const uint8_t *a_row = active_mat ? (active_mat + t * K) : NULL;
        bool sh = shifts ? (shifts[t] != 0U) : false;
        cs_status_t st = cs_fuser_step(fus, p_row, a_row, y_true[t], sh, &res);
        if (st != CS_OK) {
            return st;
        }
        if (out_preds) {
            out_preds[t] = res.prediction;
        }
        if (out_rogue_any) {
            out_rogue_any[t] = res.any_rogue ? 1U : 0U;
        }
    }

    if (out_weights_final) {
        memcpy(out_weights_final, fus->weights, K * sizeof(double));
    }
    if (last_out) {
        *last_out = res;
    }
    return CS_OK;
}

cs_status_t cs_fuser_get_weights(const cs_fuser_t *fus, double *out_w, size_t k) {
    if (!fus || !out_w || k < fus->K) {
        return CS_ERR_INVALID;
    }
    memcpy(out_w, fus->weights, fus->K * sizeof(double));
    return CS_OK;
}

cs_status_t cs_fuser_get_rogue_stats(
    const cs_fuser_t *fus,
    double *out_stats,
    uint8_t *out_flags,
    size_t k)
{
    if (!fus || k < fus->K) {
        return CS_ERR_INVALID;
    }
    if (out_stats) {
        memcpy(out_stats, fus->rogue_stats, fus->K * sizeof(double));
    }
    if (out_flags) {
        memcpy(out_flags, fus->rogue_flags, fus->K * sizeof(uint8_t));
    }
    return CS_OK;
}
