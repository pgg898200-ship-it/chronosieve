#include "cs_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Helper: Iterative O(n) Quickselect for Empirical Conformal Quantile
 * ========================================================================= */

static double cs_quickselect(double *arr, size_t n, size_t k) {
    size_t left = 0;
    size_t right = n - 1U;
    while (left < right) {
        size_t mid = left + (right - left) / 2U;
        /* Median-of-three pivot */
        if (arr[mid] < arr[left]) {
            double tmp = arr[left]; arr[left] = arr[mid]; arr[mid] = tmp;
        }
        if (arr[right] < arr[left]) {
            double tmp = arr[left]; arr[left] = arr[right]; arr[right] = tmp;
        }
        if (arr[right] < arr[mid]) {
            double tmp = arr[mid]; arr[mid] = arr[right]; arr[right] = tmp;
        }
        double pivot = arr[mid];
        size_t i = left;
        size_t j = right;
        while (i <= j) {
            while (arr[i] < pivot) {
                ++i;
            }
            while (arr[j] > pivot) {
                if (j == 0) {
                    break;
                }
                --j;
            }
            if (i <= j) {
                double tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                ++i;
                if (j == 0) {
                    break;
                }
                --j;
            }
        }
        if (k <= j) {
            right = j;
        } else if (k >= i) {
            left = i;
        } else {
            break;
        }
    }
    return arr[k];
}

static double cs_empirical_conformal_quantile(
    const double *ring,
    size_t cap,
    size_t head,
    size_t count,
    double alpha,
    double *scratch)
{
    if (count == 0) {
        return 0.0;
    }
    /* Copy the `count` most recent elements from circular buffer ending at head - 1 */
    for (size_t i = 0; i < count; ++i) {
        size_t idx = (head + cap - count + i) % cap;
        scratch[i] = ring[idx];
    }
    /* Finite-sample conformal level: ceil((1 - alpha) * (n + 1)) / n */
    double pos = ceil((1.0 - alpha) * (double)(count + 1U)) - 1.0;
    size_t k = (pos <= 0.0) ? 0U : (size_t)pos;
    if (k >= count) {
        k = count - 1U;
    }
    return cs_quickselect(scratch, count, k);
}

/* =========================================================================
 * Module 2A: Pre-Control E-Reset Conformal PID (Single Channel)
 * ========================================================================= */

struct cs_conformal_pid {
    cs_conformal_config_t cfg;
    double q_next;
    double q_d;
    double err_sum;          /* Exact sum_{i=1}^t (err_i - alpha), NEVER reset */
    uint64_t step_count;
    uint64_t miscov_count;
    uint64_t reset_count;
    uint32_t warmup_left;    /* Mandatory post-reset calibration warmup W_cal >= 32 */

    cs_edetector_t *edet;    /* Strictly receives pre-control s_t = |y - y_hat|, NEVER err_t */
    double *score_ring;
    double *scratch;
    size_t score_cap;
    size_t score_head;
    size_t regime_count;
};

void cs_conformal_config_default(cs_conformal_config_t *cfg) {
    if (!cfg) {
        return;
    }
    cfg->alpha = 0.10;
    cfg->eta = 0.05;
    cfg->ki = 1.0;
    cfg->c_sat = 1.0;
    cfg->score_bound = 1.0e6;
    cfg->init_q = 0.0;
    cfg->edet_threshold = 500.0;
    cfg->warmup_cal = 32U;
    cfg->score_window = 128U;
    cfg->edet_max_prefix = 65536U;
    cfg->seed = 0xa0761d6478bd642fULL;
    cfg->enable_ereset = true;
}

static bool cs_conformal_validate_cfg(const cs_conformal_config_t *cfg) {
    if (!cfg) {
        return false;
    }
    if (!isfinite(cfg->alpha) || cfg->alpha <= 0.0 || cfg->alpha >= 1.0) {
        return false;
    }
    if (!isfinite(cfg->eta) || cfg->eta <= 0.0) {
        return false;
    }
    if (!isfinite(cfg->ki) || cfg->ki < 0.0) {
        return false;
    }
    if (!isfinite(cfg->c_sat) || cfg->c_sat <= 0.0) {
        return false;
    }
    if (!isfinite(cfg->score_bound) || cfg->score_bound <= 0.0) {
        return false;
    }
    if (!isfinite(cfg->init_q) || cfg->init_q < 0.0) {
        return false;
    }
    if (cfg->enable_ereset && (!isfinite(cfg->edet_threshold) || cfg->edet_threshold <= 1.0)) {
        return false;
    }
    return true;
}

cs_conformal_pid_t *cs_conformal_create(const cs_conformal_config_t *cfg) {
    cs_conformal_config_t local_cfg;
    if (cfg) {
        local_cfg = *cfg;
    } else {
        cs_conformal_config_default(&local_cfg);
    }
    if (!cs_conformal_validate_cfg(&local_cfg)) {
        return NULL;
    }
    /* Round 2 invariant: enforce mandatory W_cal >= 32 whenever E-reset is enabled. */
    if (local_cfg.enable_ereset && local_cfg.warmup_cal < 32U) {
        local_cfg.warmup_cal = 32U;
    }
    if (local_cfg.enable_ereset && local_cfg.score_window < 16U) {
        local_cfg.score_window = 128U;
    }

    cs_conformal_pid_t *cp = (cs_conformal_pid_t *)calloc(1U, sizeof(cs_conformal_pid_t));
    if (!cp) {
        return NULL;
    }
    cp->cfg = local_cfg;
    cp->q_d = fmin(fmax(local_cfg.init_q, 0.0), local_cfg.score_bound);
    cp->q_next = cp->q_d;
    cp->warmup_left = local_cfg.enable_ereset ? local_cfg.warmup_cal : 0U;

    if (local_cfg.score_window > 0U) {
        cp->score_cap = local_cfg.score_window;
        cp->score_ring = (double *)calloc(cp->score_cap, sizeof(double));
        cp->scratch = (double *)calloc(cp->score_cap, sizeof(double));
        if (!cp->score_ring || !cp->scratch) {
            cs_conformal_destroy(cp);
            return NULL;
        }
    }

    if (local_cfg.enable_ereset) {
        cs_edet_config_t ecfg;
        cs_edet_config_default(&ecfg);
        ecfg.family = CS_FAMILY_VOVK_RANK;
        ecfg.accum = CS_ACCUM_ESR;
        ecfg.side = CS_SIDE_TWO_SIDED;
        ecfg.threshold = local_cfg.edet_threshold;
        ecfg.max_prefix = (local_cfg.edet_max_prefix > 0U) ? local_cfg.edet_max_prefix : 65536U;
        ecfg.seed = local_cfg.seed;
        ecfg.auto_reset = false;
        cp->edet = cs_edet_create(&ecfg);
        if (!cp->edet) {
            cs_conformal_destroy(cp);
            return NULL;
        }
    }

    return cp;
}

void cs_conformal_destroy(cs_conformal_pid_t *cp) {
    if (!cp) {
        return;
    }
    cs_edet_destroy(cp->edet);
    free(cp->score_ring);
    free(cp->scratch);
    free(cp);
}

void cs_conformal_reset(cs_conformal_pid_t *cp) {
    if (!cp) {
        return;
    }
    cp->q_d = fmin(fmax(cp->cfg.init_q, 0.0), cp->cfg.score_bound);
    cp->q_next = cp->q_d;
    cp->err_sum = 0.0;
    cp->step_count = 0ULL;
    cp->miscov_count = 0ULL;
    cp->reset_count = 0ULL;
    cp->warmup_left = cp->cfg.enable_ereset ? cp->cfg.warmup_cal : 0U;
    cp->score_head = 0U;
    cp->regime_count = 0U;
    if (cp->edet) {
        cs_edet_reset(cp->edet);
    }
}

cs_status_t cs_conformal_step(
    cs_conformal_pid_t *cp,
    double y_true,
    double y_pred,
    cs_conformal_result_t *out)
{
    if (!cp) {
        return CS_ERR_INVALID;
    }
    if (!isfinite(y_true) || !isfinite(y_pred)) {
        return CS_ERR_NONFINITE;
    }

    const double q_used = cp->q_next;
    const double score = fabs(y_true - y_pred);
    if (!isfinite(score)) {
        return CS_ERR_NONFINITE;
    }
    const bool miscov = (score > q_used);
    const double err_t = miscov ? 1.0 : 0.0;

    cp->step_count++;
    if (miscov) {
        cp->miscov_count++;
    }

    /* EXACT Angelopoulos et al. (NeurIPS 2023) P and saturating-tangent I terms.
     * INVARIANT: err_sum is NEVER reset and eta is NEVER boosted. */
    cp->err_sum += (err_t - cp->cfg.alpha);
    const double q_p = cp->cfg.eta * cp->err_sum;

    const double t_f = (double)cp->step_count;
    const double theta = (cp->err_sum * log(t_f + 2.0)) / (cp->cfg.c_sat * (t_f + 1.0));
    const double q_i = cp->cfg.ki * cs_scalar_tan_clamped(theta);

    /* Push exogenous pre-control score s_t into scorecaster ring buffer. */
    if (cp->score_cap > 0U) {
        cp->score_ring[cp->score_head] = score;
        cp->score_head = (cp->score_head + 1U) % cp->score_cap;
        if (cp->regime_count < cp->score_cap) {
            cp->regime_count++;
        }
    }

    bool shift_reset = false;
    bool calibrating = false;
    double edet_stat = 0.0;

    if (cp->edet) {
        cs_edet_result_t eres;
        cs_status_t est = cs_edet_step(cp->edet, score, &eres);
        if (est != CS_OK) {
            return est;
        }
        edet_stat = eres.stat;

        if (cp->warmup_left > 0U) {
            /* Mandatory post-reset calibration warmup W_cal >= 32:
             * populate B-tree prefix and refine bounded D-term without firing alarms. */
            calibrating = true;
            cp->warmup_left--;
            if (cp->score_cap > 0U && cp->regime_count > 0U) {
                double q_emp = cs_empirical_conformal_quantile(
                    cp->score_ring, cp->score_cap, cp->score_head,
                    cp->regime_count, cp->cfg.alpha, cp->scratch);
                cp->q_d = fmin(fmax(q_emp, 0.0), cp->cfg.score_bound);
            }
            if (cp->warmup_left == 0U) {
                /* Arm cleanly with populated W_cal-point B-tree baseline and zero accumulator. */
                cs_edet_reset_accum_only(cp->edet);
                edet_stat = 0.0;
            }
        } else if (eres.alarmed) {
            /* Certified shift on pre-control score s_t:
             * Reset ONLY the bounded D-term scorecaster q_D in [0, B] to the post-shift
             * empirical (1 - alpha) quantile, then enforce mandatory W_cal >= 32 warmup. */
            shift_reset = true;
            cp->reset_count++;

            size_t post_seed = (cp->regime_count < 8U) ? cp->regime_count : 8U;
            if (post_seed == 0U) {
                post_seed = 1U;
            }
            cp->regime_count = post_seed;
            if (cp->score_cap > 0U) {
                double q_emp = cs_empirical_conformal_quantile(
                    cp->score_ring, cp->score_cap, cp->score_head,
                    cp->regime_count, cp->cfg.alpha, cp->scratch);
                cp->q_d = fmin(fmax(q_emp, 0.0), cp->cfg.score_bound);
            }
            cs_edet_reset(cp->edet);
            cp->warmup_left = cp->cfg.warmup_cal;
            calibrating = true;
        } else {
            /* Normal armed stationary tracking of bounded D-term in [0, B] (amortized O(1)). */
            if (cp->score_cap > 0U && cp->regime_count > 0U && (cp->step_count & 15ULL) == 0ULL) {
                double q_emp = cs_empirical_conformal_quantile(
                    cp->score_ring, cp->score_cap, cp->score_head,
                    cp->regime_count, cp->cfg.alpha, cp->scratch);
                cp->q_d = fmin(fmax(q_emp, 0.0), cp->cfg.score_bound);
            }
        }
    } else if (cp->score_cap > 0U && cp->regime_count > 0U && (cp->step_count & 15ULL) == 0ULL) {
        double q_emp = cs_empirical_conformal_quantile(
            cp->score_ring, cp->score_cap, cp->score_head,
            cp->regime_count, cp->cfg.alpha, cp->scratch);
        cp->q_d = fmin(fmax(q_emp, 0.0), cp->cfg.score_bound);
    }

    double q_raw = q_p + q_i + cp->q_d;
    cp->q_next = (q_raw > 0.0) ? q_raw : 0.0;

    if (out) {
        out->q_used = q_used;
        out->q_next = cp->q_next;
        out->lower = y_pred - q_used;
        out->upper = y_pred + q_used;
        out->score = score;
        out->q_p = q_p;
        out->q_i = q_i;
        out->q_d = cp->q_d;
        out->emp_coverage = 1.0 - (double)cp->miscov_count / (double)cp->step_count;
        out->edet_stat = edet_stat;
        out->step_count = cp->step_count;
        out->reset_count = cp->reset_count;
        out->miscovered = miscov;
        out->shift_reset = shift_reset;
        out->calibrating = calibrating;
    }

    return CS_OK;
}

cs_status_t cs_conformal_step_batch(
    cs_conformal_pid_t *cp,
    const double *y_true,
    const double *y_pred,
    size_t n,
    double *out_q_used,
    double *out_q_next,
    uint8_t *out_miscov,
    uint8_t *out_resets,
    cs_conformal_result_t *last_out)
{
    if (!cp || (n > 0 && (!y_true || !y_pred))) {
        return CS_ERR_INVALID;
    }
    if (n == 0) {
        if (last_out) {
            memset(last_out, 0, sizeof(*last_out));
            last_out->q_used = cp->q_next;
            last_out->q_next = cp->q_next;
            last_out->q_d = cp->q_d;
            last_out->step_count = cp->step_count;
            last_out->reset_count = cp->reset_count;
            last_out->emp_coverage = (cp->step_count > 0ULL)
                ? (1.0 - (double)cp->miscov_count / (double)cp->step_count) : 1.0;
        }
        return CS_OK;
    }

    cs_conformal_result_t res;
    memset(&res, 0, sizeof(res));
    for (size_t i = 0; i < n; ++i) {
        cs_status_t st = cs_conformal_step(cp, y_true[i], y_pred[i], &res);
        if (st != CS_OK) {
            return st;
        }
        if (out_q_used) {
            out_q_used[i] = res.q_used;
        }
        if (out_q_next) {
            out_q_next[i] = res.q_next;
        }
        if (out_miscov) {
            out_miscov[i] = res.miscovered ? 1U : 0U;
        }
        if (out_resets) {
            out_resets[i] = res.shift_reset ? 1U : 0U;
        }
    }
    if (last_out) {
        *last_out = res;
    }
    return CS_OK;
}

double cs_conformal_current_q(const cs_conformal_pid_t *cp) {
    return cp ? cp->q_next : 0.0;
}

/* =========================================================================
 * Module 2B: Multi-Channel SoA SIMD Conformal PID Kernel (NEON / AVX2 / Scalar)
 * ========================================================================= */

struct cs_conformal_soa {
    size_t n_channels;
    cs_conformal_config_t cfg;
    double *q_cur;
    double *err_sum;
    double *q_d;
    uint64_t step_count;
};

cs_conformal_soa_t *cs_conformal_soa_create(size_t n_channels, const cs_conformal_config_t *cfg) {
    if (n_channels == 0) {
        return NULL;
    }
    cs_conformal_config_t local_cfg;
    if (cfg) {
        local_cfg = *cfg;
    } else {
        cs_conformal_config_default(&local_cfg);
    }
    if (!cs_conformal_validate_cfg(&local_cfg)) {
        return NULL;
    }

    cs_conformal_soa_t *soa = (cs_conformal_soa_t *)calloc(1U, sizeof(cs_conformal_soa_t));
    if (!soa) {
        return NULL;
    }
    soa->n_channels = n_channels;
    soa->cfg = local_cfg;

    size_t pad = ((n_channels + 7U) / 8U) * 8U;
    soa->q_cur = (double *)cs_aligned_calloc(64U, pad, sizeof(double));
    soa->err_sum = (double *)cs_aligned_calloc(64U, pad, sizeof(double));
    soa->q_d = (double *)cs_aligned_calloc(64U, pad, sizeof(double));
    if (!soa->q_cur || !soa->err_sum || !soa->q_d) {
        cs_conformal_soa_destroy(soa);
        return NULL;
    }

    double init_qd = fmin(fmax(local_cfg.init_q, 0.0), local_cfg.score_bound);
    for (size_t c = 0; c < n_channels; ++c) {
        soa->q_d[c] = init_qd;
        soa->q_cur[c] = init_qd;
    }
    return soa;
}

void cs_conformal_soa_destroy(cs_conformal_soa_t *soa) {
    if (!soa) {
        return;
    }
    cs_aligned_free(soa->q_cur);
    cs_aligned_free(soa->err_sum);
    cs_aligned_free(soa->q_d);
    free(soa);
}

void cs_conformal_soa_reset(cs_conformal_soa_t *soa) {
    if (!soa) {
        return;
    }
    double init_qd = fmin(fmax(soa->cfg.init_q, 0.0), soa->cfg.score_bound);
    for (size_t c = 0; c < soa->n_channels; ++c) {
        soa->q_d[c] = init_qd;
        soa->q_cur[c] = init_qd;
        soa->err_sum[c] = 0.0;
    }
    soa->step_count = 0ULL;
}

cs_status_t cs_conformal_soa_step(
    cs_conformal_soa_t *soa,
    const double *y_true,
    const double *y_pred,
    double *out_q_used,
    double *out_q_next,
    uint8_t *out_miscov)
{
    if (!soa || !y_true || !y_pred) {
        return CS_ERR_INVALID;
    }
    const size_t C = soa->n_channels;
    for (size_t c = 0; c < C; ++c) {
        if (!isfinite(y_true[c]) || !isfinite(y_pred[c]) || !isfinite(y_true[c] - y_pred[c])) {
            return CS_ERR_NONFINITE;
        }
    }

    soa->step_count++;
    const double alpha = soa->cfg.alpha;
    const double eta = soa->cfg.eta;
    const double ki = soa->cfg.ki;
    const double t_f = (double)soa->step_count;
    const double scale_t = log(t_f + 2.0) / (soa->cfg.c_sat * (t_f + 1.0));

    size_t c = 0;
#if defined(CS_SIMD_NEON)
    const float64x2_t valpha = vdupq_n_f64(alpha);
    const float64x2_t veta = vdupq_n_f64(eta);
    const float64x2_t vki = vdupq_n_f64(ki);
    const float64x2_t vscale = vdupq_n_f64(scale_t);
    const float64x2_t vone = vdupq_n_f64(1.0);
    const float64x2_t vzero = vdupq_n_f64(0.0);

    for (; c + 2U <= C; c += 2U) {
        float64x2_t yt = vld1q_f64(y_true + c);
        float64x2_t yp = vld1q_f64(y_pred + c);
        float64x2_t qc = vld1q_f64(soa->q_cur + c);
        float64x2_t es = vld1q_f64(soa->err_sum + c);
        float64x2_t qd = vld1q_f64(soa->q_d + c);

        if (out_q_used) {
            vst1q_f64(out_q_used + c, qc);
        }

        float64x2_t sc = vabsq_f64(vsubq_f64(yt, yp));
        uint64x2_t mask = vcgtq_f64(sc, qc);
        float64x2_t err = vbslq_f64(mask, vone, vzero);
        es = vaddq_f64(es, vsubq_f64(err, valpha));

        float64x2_t qp = vmulq_f64(veta, es);
        float64x2_t theta = vmulq_f64(es, vscale);
        float64x2_t qi = vmulq_f64(vki, cs_vtan_clamped_f64(theta));
        float64x2_t qn = vmaxq_f64(vzero, vaddq_f64(vaddq_f64(qp, qi), qd));

        vst1q_f64(soa->err_sum + c, es);
        vst1q_f64(soa->q_cur + c, qn);
        if (out_q_next) {
            vst1q_f64(out_q_next + c, qn);
        }
        if (out_miscov) {
            out_miscov[c] = (vgetq_lane_u64(mask, 0) != 0ULL) ? 1U : 0U;
            out_miscov[c + 1U] = (vgetq_lane_u64(mask, 1) != 0ULL) ? 1U : 0U;
        }
    }
#elif defined(CS_SIMD_AVX2)
    const __m256d valpha = _mm256_set1_pd(alpha);
    const __m256d veta = _mm256_set1_pd(eta);
    const __m256d vki = _mm256_set1_pd(ki);
    const __m256d vscale = _mm256_set1_pd(scale_t);
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vzero = _mm256_setzero_pd();
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    for (; c + 4U <= C; c += 4U) {
        __m256d yt = _mm256_loadu_pd(y_true + c);
        __m256d yp = _mm256_loadu_pd(y_pred + c);
        __m256d qc = _mm256_loadu_pd(soa->q_cur + c);
        __m256d es = _mm256_loadu_pd(soa->err_sum + c);
        __m256d qd = _mm256_loadu_pd(soa->q_d + c);

        if (out_q_used) {
            _mm256_storeu_pd(out_q_used + c, qc);
        }

        __m256d sc = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(yt, yp));
        __m256d mask = _mm256_cmp_pd(sc, qc, _CMP_GT_OQ);
        __m256d err = _mm256_and_pd(mask, vone);
        es = _mm256_add_pd(es, _mm256_sub_pd(err, valpha));

        __m256d qp = _mm256_mul_pd(veta, es);
        __m256d theta = _mm256_mul_pd(es, vscale);
        __m256d qi = _mm256_mul_pd(vki, cs_vtan_clamped_f64(theta));
        __m256d qn = _mm256_max_pd(vzero, _mm256_add_pd(_mm256_add_pd(qp, qi), qd));

        _mm256_storeu_pd(soa->err_sum + c, es);
        _mm256_storeu_pd(soa->q_cur + c, qn);
        if (out_q_next) {
            _mm256_storeu_pd(out_q_next + c, qn);
        }
        if (out_miscov) {
            int mbits = _mm256_movemask_pd(mask);
            out_miscov[c + 0U] = (uint8_t)((mbits >> 0) & 1);
            out_miscov[c + 1U] = (uint8_t)((mbits >> 1) & 1);
            out_miscov[c + 2U] = (uint8_t)((mbits >> 2) & 1);
            out_miscov[c + 3U] = (uint8_t)((mbits >> 3) & 1);
        }
    }
#endif

    for (; c < C; ++c) {
        double qc = soa->q_cur[c];
        if (out_q_used) {
            out_q_used[c] = qc;
        }
        double sc = fabs(y_true[c] - y_pred[c]);
        bool mc = (sc > qc);
        double es = soa->err_sum[c] + ((mc ? 1.0 : 0.0) - alpha);
        double qp = eta * es;
        double qi = ki * cs_scalar_tan_clamped(es * scale_t);
        double qn = qp + qi + soa->q_d[c];
        if (qn < 0.0) {
            qn = 0.0;
        }
        soa->err_sum[c] = es;
        soa->q_cur[c] = qn;
        if (out_q_next) {
            out_q_next[c] = qn;
        }
        if (out_miscov) {
            out_miscov[c] = mc ? 1U : 0U;
        }
    }
    return CS_OK;
}

cs_status_t cs_conformal_soa_step_batch(
    cs_conformal_soa_t *soa,
    const double *y_true,
    const double *y_pred,
    size_t n_steps,
    double *out_q_next,
    uint8_t *out_miscov)
{
    if (!soa || (n_steps > 0 && (!y_true || !y_pred))) {
        return CS_ERR_INVALID;
    }
    const size_t C = soa->n_channels;
    for (size_t t = 0; t < n_steps; ++t) {
        cs_status_t st = cs_conformal_soa_step(
            soa,
            y_true + t * C,
            y_pred + t * C,
            NULL,
            out_q_next ? (out_q_next + t * C) : NULL,
            out_miscov ? (out_miscov + t * C) : NULL);
        if (st != CS_OK) {
            return st;
        }
    }
    return CS_OK;
}

/* =========================================================================
 * Module 2C: Online COMA — Conformal Majority Vote (Gasparin & Ramdas UAI 2024)
 * O(K log K) sweep-line producing exact disjoint union + convex hull + max agreement.
 * ========================================================================= */

typedef struct {
    double x;
    double w;
    int type; /* 0 = START (+w), 1 = END (-w) */
} cs_coma_event_t;

static int cs_coma_event_cmp(const void *a, const void *b) {
    const cs_coma_event_t *ea = (const cs_coma_event_t *)a;
    const cs_coma_event_t *eb = (const cs_coma_event_t *)b;
    if (ea->x < eb->x) {
        return -1;
    }
    if (ea->x > eb->x) {
        return 1;
    }
    return ea->type - eb->type;
}

static inline bool cs_coma_passes(double w, double tau) {
    if (tau >= 1.0) {
        return w >= 1.0 - 1.0e-12;
    }
    return w > tau + 1.0e-15;
}

static void cs_coma_emit_interval(
    cs_interval_t *out,
    size_t *n_out,
    size_t max_out,
    bool *overflowed,
    double lo,
    double hi)
{
    if (!(*overflowed) && *n_out > 0U && out[*n_out - 1U].hi >= lo) {
        if (hi > out[*n_out - 1U].hi) {
            out[*n_out - 1U].hi = hi;
        }
        return;
    }
    if (*n_out < max_out) {
        out[*n_out].lo = lo;
        out[*n_out].hi = hi;
        (*n_out)++;
    } else {
        *overflowed = true;
    }
}

cs_status_t cs_coma_merge(
    const double *lowers,
    const double *uppers,
    const double *weights,
    size_t k,
    double tau,
    bool randomized,
    uint64_t *rng_state,
    cs_interval_t *out_intervals,
    size_t max_intervals,
    cs_coma_result_t *out_res)
{
    if (!lowers || !uppers || !out_intervals || !out_res || k == 0 || max_intervals == 0) {
        return CS_ERR_INVALID;
    }

    double used_tau = tau;
    if (randomized) {
        if (!rng_state) {
            return CS_ERR_INVALID;
        }
        double u = cs_u64_to_uniform01(cs_splitmix64_next(rng_state));
        used_tau = 0.5 + 0.5 * u; /* Uniform[0.5, 1.0) */
    } else {
        if (!isfinite(used_tau) || used_tau < 0.0 || used_tau > 1.0) {
            return CS_ERR_INVALID;
        }
    }

    double wsum = 0.0;
    for (size_t i = 0; i < k; ++i) {
        if (!isfinite(lowers[i]) || !isfinite(uppers[i])) {
            return CS_ERR_NONFINITE;
        }
        if (lowers[i] > uppers[i]) {
            return CS_ERR_INVALID;
        }
        double wi = weights ? weights[i] : 1.0;
        if (!isfinite(wi) || wi < 0.0) {
            return CS_ERR_INVALID;
        }
        wsum += wi;
    }
    if (wsum <= 0.0) {
        return CS_ERR_INVALID;
    }
    double inv_wsum = 1.0 / wsum;

    if (k > SIZE_MAX / (2U * sizeof(cs_coma_event_t))) {
        return CS_ERR_NOMEM;
    }

    cs_coma_event_t stack_events[128];
    cs_coma_event_t *events = stack_events;
    if (2U * k > 128U) {
        events = (cs_coma_event_t *)malloc(2U * k * sizeof(cs_coma_event_t));
        if (!events) {
            return CS_ERR_NOMEM;
        }
    }

    size_t n_ev = 0;
    for (size_t i = 0; i < k; ++i) {
        double wi = (weights ? weights[i] : 1.0) * inv_wsum;
        if (wi <= 0.0) {
            continue;
        }
        events[n_ev].x = lowers[i];
        events[n_ev].w = wi;
        events[n_ev].type = 0;
        n_ev++;

        events[n_ev].x = uppers[i];
        events[n_ev].w = wi;
        events[n_ev].type = 1;
        n_ev++;
    }

    qsort(events, n_ev, sizeof(cs_coma_event_t), cs_coma_event_cmp);

    size_t n_out = 0;
    bool overflowed = false;
    double active_w = 0.0;
    double max_agree = 0.0;
    double prev_x = 0.0;
    bool has_prev = false;

    size_t idx = 0;
    while (idx < n_ev) {
        double x = events[idx].x;

        if (has_prev && x > prev_x && cs_coma_passes(active_w, used_tau)) {
            cs_coma_emit_interval(out_intervals, &n_out, max_intervals, &overflowed, prev_x, x);
        }

        double sum_start = 0.0;
        double sum_end = 0.0;
        while (idx < n_ev && events[idx].x == x) {
            if (events[idx].type == 0) {
                sum_start += events[idx].w;
            } else {
                sum_end += events[idx].w;
            }
            idx++;
        }

        double peak_w = active_w + sum_start;
        if (peak_w > max_agree) {
            max_agree = peak_w;
        }
        double next_w = peak_w - sum_end;
        if (next_w < 0.0 && next_w > -1.0e-12) {
            next_w = 0.0;
        }

        /* Handle isolated touching point [x, x] where peak passes but adjacent open intervals do not */
        if (cs_coma_passes(peak_w, used_tau) && !cs_coma_passes(next_w, used_tau)) {
            bool prev_covered = (has_prev && cs_coma_passes(active_w, used_tau));
            if (!prev_covered) {
                cs_coma_emit_interval(out_intervals, &n_out, max_intervals, &overflowed, x, x);
            }
        }

        active_w = next_w;
        prev_x = x;
        has_prev = true;
    }

    if (events != stack_events) {
        free(events);
    }

    out_res->n_intervals = n_out;
    out_res->max_agreement = max_agree;
    out_res->used_tau = used_tau;
    if (n_out > 0U) {
        out_res->hull_lo = out_intervals[0].lo;
        out_res->hull_hi = out_intervals[n_out - 1U].hi;
    } else {
        out_res->hull_lo = NAN;
        out_res->hull_hi = NAN;
    }

    return overflowed ? CS_ERR_BUFFER : CS_OK;
}
