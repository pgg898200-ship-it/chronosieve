#include "cs_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CS_ACC_CAP 1.0e300
#define CS_LOG_MIN -700.0
#define CS_LOG_MAX 690.0

const char *cs_version_string(void) {
    return "0.1.0";
}

const char *cs_simd_backend_name(void) {
    return CS_SIMD_NAME;
}

/* =========================================================================
 * Internal Detector Structure (128-byte aligned hot arrays for Apple M4)
 * ========================================================================= */

struct cs_edetector {
    double acc[CS_EDET_LANES];       /* E-SR R_t^(m) or E-CUSUM S_t^(m) */
    double lam[CS_EDET_LANES];       /* Betting grid lambda_m */
    double inv_denom[CS_EDET_LANES]; /* Precomputed 1 / (1 + 0.5 * lam^2 * sigma^2) for Catoni */

    cs_edet_config_t cfg;
    cs_edet_family_t eff_family;
    double d_clamp;                  /* Safe |x - mu0| clamp preventing lam*d / y^2 overflow */
    double cur_stat;
    double last_e_mean;
    double last_p_value;
    uint64_t step_count;
    uint64_t alarm_count;
    uint32_t epoch_count;

    cs_rng_t rng;
    cs_btree_t btree;
};

void cs_edet_config_default(cs_edet_config_t *cfg) {
    if (!cfg) {
        return;
    }
    cfg->family = CS_FAMILY_AUTO;
    cfg->accum = CS_ACCUM_ESR;
    cfg->side = CS_SIDE_TWO_SIDED;
    cfg->threshold = 1000.0;
    cfg->mu0 = NAN;
    cfg->sigma = NAN;
    cfg->max_prefix = 65536U;
    cfg->seed = 0x517cc1b727220a95ULL;
    cfg->auto_reset = false;
}

static void cs_fill_logspace(double *dst, size_t n, double lo, double hi, double sign) {
    if (n == 0) {
        return;
    }
    if (n == 1) {
        dst[0] = sign * sqrt(lo * hi);
        return;
    }
    double log_lo = log(lo);
    double log_hi = log(hi);
    for (size_t i = 0; i < n; ++i) {
        double frac = (double)i / (double)(n - 1U);
        dst[i] = sign * exp(log_lo + frac * (log_hi - log_lo));
    }
}

static cs_status_t cs_edet_init_grid(cs_edetector_t *det) {
    const cs_edet_config_t *cfg = &det->cfg;

    if (!isfinite(cfg->threshold) || cfg->threshold <= 1.0) {
        return CS_ERR_INVALID;
    }

    cs_edet_family_t fam = cfg->family;
    if (fam == CS_FAMILY_AUTO) {
        if (isfinite(cfg->mu0) && isfinite(cfg->sigma) && cfg->sigma > 0.0) {
            fam = CS_FAMILY_CATONI;
        } else {
            fam = CS_FAMILY_VOVK_RANK;
        }
    }
    det->eff_family = fam;

    for (size_t m = 0; m < CS_EDET_LANES; ++m) {
        det->inv_denom[m] = 1.0;
    }
    det->d_clamp = DBL_MAX;

    switch (fam) {
        case CS_FAMILY_BOUNDED: {
            if (!isfinite(cfg->mu0) || cfg->mu0 <= 1e-9 || cfg->mu0 >= 1.0 - 1e-9) {
                return CS_ERR_INVALID;
            }
            double max_pos = 0.90 / cfg->mu0;
            double min_pos = 0.02 / cfg->mu0;
            double max_neg = 0.90 / (1.0 - cfg->mu0);
            double min_neg = 0.02 / (1.0 - cfg->mu0);
            if (cfg->side == CS_SIDE_UPPER) {
                cs_fill_logspace(det->lam, 16U, min_pos, max_pos, 1.0);
            } else if (cfg->side == CS_SIDE_LOWER) {
                cs_fill_logspace(det->lam, 16U, min_neg, max_neg, -1.0);
            } else {
                cs_fill_logspace(det->lam, 8U, min_pos, max_pos, 1.0);
                cs_fill_logspace(det->lam + 8U, 8U, min_neg, max_neg, -1.0);
            }
            break;
        }
        case CS_FAMILY_CATONI: {
            if (!isfinite(cfg->mu0) || !isfinite(cfg->sigma) || cfg->sigma <= 0.0) {
                return CS_ERR_INVALID;
            }
            /* Build dimensionless grid c_m in [0.025, 1.40] first so inv_denom never overflows/underflows. */
            double c_grid[CS_EDET_LANES];
            const double c_lo = 0.025;
            const double c_hi = 1.40;
            if (cfg->side == CS_SIDE_UPPER) {
                cs_fill_logspace(c_grid, 16U, c_lo, c_hi, 1.0);
            } else if (cfg->side == CS_SIDE_LOWER) {
                cs_fill_logspace(c_grid, 16U, c_lo, c_hi, -1.0);
            } else {
                cs_fill_logspace(c_grid, 8U, c_lo, c_hi, 1.0);
                cs_fill_logspace(c_grid + 8U, 8U, c_lo, c_hi, -1.0);
            }
            double max_abs_lam = 0.0;
            for (size_t m = 0; m < CS_EDET_LANES; ++m) {
                double c = c_grid[m];
                double l = c / cfg->sigma;
                if (!isfinite(l) || l == 0.0) {
                    return CS_ERR_INVALID;
                }
                det->lam[m] = l;
                det->inv_denom[m] = 1.0 / (1.0 + 0.5 * c * c);
                double al = fabs(l);
                if (al > max_abs_lam) {
                    max_abs_lam = al;
                }
            }
            double dc = 1.0e150 / max_abs_lam;
            det->d_clamp = isfinite(dc) ? dc : DBL_MAX;
            break;
        }
        case CS_FAMILY_SYMMETRY: {
            if (!isfinite(cfg->mu0)) {
                return CS_ERR_INVALID;
            }
            double scale = (isfinite(cfg->sigma) && cfg->sigma > 0.0) ? cfg->sigma : 1.0;
            double c_grid[CS_EDET_LANES];
            if (cfg->side == CS_SIDE_UPPER) {
                cs_fill_logspace(c_grid, 16U, 0.01, 10.0, 1.0);
            } else if (cfg->side == CS_SIDE_LOWER) {
                cs_fill_logspace(c_grid, 16U, 0.01, 10.0, -1.0);
            } else {
                cs_fill_logspace(c_grid, 8U, 0.01, 10.0, 1.0);
                cs_fill_logspace(c_grid + 8U, 8U, 0.01, 10.0, -1.0);
            }
            double max_abs_lam = 0.0;
            for (size_t m = 0; m < CS_EDET_LANES; ++m) {
                double l = c_grid[m] / scale;
                if (!isfinite(l) || l == 0.0) {
                    return CS_ERR_INVALID;
                }
                det->lam[m] = l;
                double al = fabs(l);
                if (al > max_abs_lam) {
                    max_abs_lam = al;
                }
            }
            double dc = 1.0e150 / max_abs_lam;
            det->d_clamp = isfinite(dc) ? dc : DBL_MAX;
            break;
        }
        case CS_FAMILY_VOVK_RANK: {
            /* e_t(lam) = 1 + lam * (1 - 2 * p_t).
             * Upward shift => p_t -> 1 => (1 - 2*p_t) -> -1 => lam < 0 gives e_t > 1.
             * Downward shift => p_t -> 0 => (1 - 2*p_t) -> +1 => lam > 0 gives e_t > 1. */
            if (cfg->side == CS_SIDE_UPPER) {
                cs_fill_logspace(det->lam, 16U, 0.03, 0.92, -1.0);
            } else if (cfg->side == CS_SIDE_LOWER) {
                cs_fill_logspace(det->lam, 16U, 0.03, 0.92, 1.0);
            } else {
                cs_fill_logspace(det->lam, 8U, 0.03, 0.92, -1.0);
                cs_fill_logspace(det->lam + 8U, 8U, 0.03, 0.92, 1.0);
            }
            break;
        }
        default:
            return CS_ERR_INVALID;
    }

    return CS_OK;
}

cs_edetector_t *cs_edet_create(const cs_edet_config_t *cfg) {
    cs_edet_config_t local_cfg;
    if (cfg) {
        local_cfg = *cfg;
    } else {
        cs_edet_config_default(&local_cfg);
    }
    if (local_cfg.max_prefix == 0U) {
        local_cfg.max_prefix = 65536U;
    }

    cs_edetector_t *det = (cs_edetector_t *)cs_aligned_calloc(128U, 1U, sizeof(cs_edetector_t));
    if (!det) {
        return NULL;
    }
    det->cfg = local_cfg;
    cs_btree_init(&det->btree);
    cs_rng_seed(&det->rng, local_cfg.seed);

    if (cs_edet_init_grid(det) != CS_OK) {
        cs_aligned_free(det);
        return NULL;
    }
    det->last_p_value = NAN;
    return det;
}

void cs_edet_destroy(cs_edetector_t *det) {
    if (!det) {
        return;
    }
    cs_btree_free(&det->btree);
    cs_aligned_free(det);
}

void cs_edet_reset(cs_edetector_t *det) {
    if (!det) {
        return;
    }
    memset(det->acc, 0, sizeof(det->acc));
    det->cur_stat = 0.0;
    det->last_e_mean = 1.0;
    det->last_p_value = NAN;
    cs_btree_clear(&det->btree);
}

void cs_edet_reset_accum_only(cs_edetector_t *det) {
    if (!det) {
        return;
    }
    memset(det->acc, 0, sizeof(det->acc));
    det->cur_stat = 0.0;
    det->last_e_mean = 1.0;
}

cs_status_t cs_edet_calibrate_catoni(
    const double *warmup,
    size_t n,
    double threshold,
    cs_edet_side_t side,
    double inflation_z,
    cs_edet_config_t *out_cfg)
{
    if (!warmup || !out_cfg || n < 16U || !isfinite(threshold) || threshold <= 1.0) {
        return CS_ERR_INVALID;
    }
    /* Overflow-safe running mean */
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(warmup[i])) {
            return CS_ERR_NONFINITE;
        }
        mean += (warmup[i] - mean) / (double)(i + 1U);
    }

    /* Scale-invariant second & fourth central moments (immune to d^2 / d^4 overflow) */
    double max_abs_d = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double ad = fabs(warmup[i] - mean);
        if (!isfinite(ad)) {
            return CS_ERR_NONFINITE;
        }
        if (ad > max_abs_d) {
            max_abs_d = ad;
        }
    }

    double sd = 1.0e-12;
    double kurt = 3.0;
    if (max_abs_d > 0.0) {
        double inv_scale = 1.0 / max_abs_d;
        double sm2 = 0.0;
        double sm4 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double u = (warmup[i] - mean) * inv_scale;
            double u2 = u * u;
            sm2 += u2;
            sm4 += u2 * u2;
        }
        double svar = sm2 / (double)(n - 1U);
        if (svar > 0.0) {
            sd = max_abs_d * sqrt(svar);
            if (sd < 1.0e-12) {
                sd = 1.0e-12;
            }
            double k_est = (sm4 / (double)n) / (svar * svar);
            if (isfinite(k_est) && k_est > 3.0) {
                kurt = k_est;
            }
        }
    }

    double z = (isfinite(inflation_z) && inflation_z > 0.0) ? inflation_z : 2.5;
    double dn = (double)n;
    /* Conservative variance upper bound accounting for mean & kurtosis estimation error */
    double infl = 1.0 + (z / sqrt(dn)) * sqrt(kurt - 1.0) + (z * z) / dn;
    double sigma_eff = sd * sqrt(infl);
    double mu_eff = mean;
    if (side == CS_SIDE_UPPER) {
        /* Composite null E[X] <= mu_eff holds with high probability over finite warmup */
        mu_eff = mean + (z / sqrt(dn)) * sd;
    } else if (side == CS_SIDE_LOWER) {
        mu_eff = mean - (z / sqrt(dn)) * sd;
    }
    if (!isfinite(sigma_eff) || sigma_eff <= 0.0 || !isfinite(mu_eff)) {
        return CS_ERR_NONFINITE;
    }

    cs_edet_config_default(out_cfg);
    out_cfg->family = CS_FAMILY_CATONI;
    out_cfg->accum = CS_ACCUM_ESR;
    out_cfg->side = side;
    out_cfg->threshold = threshold;
    out_cfg->mu0 = mu_eff;
    out_cfg->sigma = sigma_eff;
    return CS_OK;
}

/* =========================================================================
 * Vectorized 16-Lane E-Value Evaluators (Zero-exp, NEON / AVX2 / Scalar)
 * ========================================================================= */

static inline void cs_eval_bounded_16(const double *lam, double mu0, double x, double out_e[16]) {
    double z = (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
    double d = z - mu0;
#if defined(CS_SIMD_NEON)
    float64x2_t vd = vdupq_n_f64(d);
    float64x2_t vone = vdupq_n_f64(1.0);
    for (size_t i = 0; i < 16U; i += 2U) {
        float64x2_t vl = vld1q_f64(lam + i);
        float64x2_t ve = vfmaq_f64(vone, vl, vd);
        vst1q_f64(out_e + i, ve);
    }
#elif defined(CS_SIMD_AVX2)
    __m256d vd = _mm256_set1_pd(d);
    __m256d vone = _mm256_set1_pd(1.0);
    for (size_t i = 0; i < 16U; i += 4U) {
        __m256d vl = _mm256_loadu_pd(lam + i);
        __m256d ve = cs_avx_fmadd(vl, vd, vone);
        _mm256_storeu_pd(out_e + i, ve);
    }
#else
    for (size_t i = 0; i < 16U; ++i) {
        out_e[i] = 1.0 + lam[i] * d;
    }
#endif
}

static inline double cs_clamp_diff(double x, double mu0, double d_clamp) {
    double d = x - mu0;
    return fmin(fmax(d, -d_clamp), d_clamp);
}

static inline void cs_eval_catoni_16(
    const double *lam,
    const double *inv_denom,
    double mu0,
    double d_clamp,
    double x,
    double out_e[16])
{
    double d = cs_clamp_diff(x, mu0, d_clamp);
#if defined(CS_SIMD_NEON)
    float64x2_t vd = vdupq_n_f64(d);
    float64x2_t vone = vdupq_n_f64(1.0);
    float64x2_t vhalf = vdupq_n_f64(0.5);
    float64x2_t vzero = vdupq_n_f64(0.0);
    for (size_t i = 0; i < 16U; i += 2U) {
        float64x2_t vl = vld1q_f64(lam + i);
        float64x2_t vinv = vld1q_f64(inv_denom + i);
        float64x2_t y = vmulq_f64(vl, vd);
        float64x2_t t = vfmaq_f64(vone, vhalf, vmulq_f64(y, y));
        float64x2_t pos = vaddq_f64(t, y);
        float64x2_t neg = vdivq_f64(vone, vsubq_f64(t, y));
        uint64x2_t mask = vcgeq_f64(y, vzero);
        float64x2_t psi = vbslq_f64(mask, pos, neg);
        vst1q_f64(out_e + i, vmulq_f64(psi, vinv));
    }
#elif defined(CS_SIMD_AVX2)
    __m256d vd = _mm256_set1_pd(d);
    __m256d vone = _mm256_set1_pd(1.0);
    __m256d vhalf = _mm256_set1_pd(0.5);
    __m256d vzero = _mm256_setzero_pd();
    for (size_t i = 0; i < 16U; i += 4U) {
        __m256d vl = _mm256_loadu_pd(lam + i);
        __m256d vinv = _mm256_loadu_pd(inv_denom + i);
        __m256d y = _mm256_mul_pd(vl, vd);
        __m256d t = cs_avx_fmadd(vhalf, _mm256_mul_pd(y, y), vone);
        __m256d pos = _mm256_add_pd(t, y);
        __m256d neg = _mm256_div_pd(vone, _mm256_sub_pd(t, y));
        __m256d mask = _mm256_cmp_pd(y, vzero, _CMP_GE_OQ);
        __m256d psi = _mm256_blendv_pd(neg, pos, mask);
        _mm256_storeu_pd(out_e + i, _mm256_mul_pd(psi, vinv));
    }
#else
    for (size_t i = 0; i < 16U; ++i) {
        double y = lam[i] * d;
        double t = 1.0 + 0.5 * y * y;
        double psi = (y >= 0.0) ? (t + y) : (1.0 / (t - y));
        out_e[i] = psi * inv_denom[i];
    }
#endif
}

static inline void cs_eval_symmetry_16(
    const double *lam,
    double mu0,
    double d_clamp,
    double x,
    double out_e[16])
{
    double d = cs_clamp_diff(x, mu0, d_clamp);
#if defined(CS_SIMD_NEON)
    float64x2_t vd = vdupq_n_f64(d);
    float64x2_t vone = vdupq_n_f64(1.0);
    for (size_t i = 0; i < 16U; i += 2U) {
        float64x2_t vl = vld1q_f64(lam + i);
        float64x2_t u = vmulq_f64(vl, vd);
        float64x2_t den = vaddq_f64(vone, vabsq_f64(u));
        float64x2_t ve = vaddq_f64(vone, vdivq_f64(u, den));
        vst1q_f64(out_e + i, ve);
    }
#elif defined(CS_SIMD_AVX2)
    __m256d vd = _mm256_set1_pd(d);
    __m256d vone = _mm256_set1_pd(1.0);
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    for (size_t i = 0; i < 16U; i += 4U) {
        __m256d vl = _mm256_loadu_pd(lam + i);
        __m256d u = _mm256_mul_pd(vl, vd);
        __m256d au = _mm256_andnot_pd(sign_mask, u);
        __m256d den = _mm256_add_pd(vone, au);
        __m256d ve = _mm256_add_pd(vone, _mm256_div_pd(u, den));
        _mm256_storeu_pd(out_e + i, ve);
    }
#else
    for (size_t i = 0; i < 16U; ++i) {
        double u = lam[i] * d;
        out_e[i] = 1.0 + u / (1.0 + fabs(u));
    }
#endif
}

static inline void cs_eval_rank_from_p_16(const double *lam, double p_val, double out_e[16]) {
    double s = 1.0 - 2.0 * p_val;
#if defined(CS_SIMD_NEON)
    float64x2_t vs = vdupq_n_f64(s);
    float64x2_t vone = vdupq_n_f64(1.0);
    for (size_t i = 0; i < 16U; i += 2U) {
        float64x2_t vl = vld1q_f64(lam + i);
        vst1q_f64(out_e + i, vfmaq_f64(vone, vl, vs));
    }
#elif defined(CS_SIMD_AVX2)
    __m256d vs = _mm256_set1_pd(s);
    __m256d vone = _mm256_set1_pd(1.0);
    for (size_t i = 0; i < 16U; i += 4U) {
        __m256d vl = _mm256_loadu_pd(lam + i);
        _mm256_storeu_pd(out_e + i, cs_avx_fmadd(vl, vs, vone));
    }
#else
    for (size_t i = 0; i < 16U; ++i) {
        out_e[i] = 1.0 + lam[i] * s;
    }
#endif
}

cs_status_t cs_edet_eval_lanes(cs_edetector_t *det, double x, double out_e[CS_EDET_LANES]) {
    if (!det || !out_e) {
        return CS_ERR_INVALID;
    }
    if (!isfinite(x)) {
        return CS_ERR_NONFINITE;
    }

    switch (det->eff_family) {
        case CS_FAMILY_BOUNDED:
            cs_eval_bounded_16(det->lam, det->cfg.mu0, x, out_e);
            det->last_p_value = NAN;
            break;
        case CS_FAMILY_CATONI:
            cs_eval_catoni_16(det->lam, det->inv_denom, det->cfg.mu0, det->d_clamp, x, out_e);
            det->last_p_value = NAN;
            break;
        case CS_FAMILY_SYMMETRY:
            cs_eval_symmetry_16(det->lam, det->cfg.mu0, det->d_clamp, x, out_e);
            det->last_p_value = NAN;
            break;
        case CS_FAMILY_VOVK_RANK: {
            /* Deterministic epoch reset when B-tree hits max_prefix (preserves exact iid U(0,1)) */
            if (det->btree.total_count >= det->cfg.max_prefix) {
                cs_btree_clear(&det->btree);
                det->epoch_count++;
            }
            uint32_t less = 0, equal = 0;
            cs_btree_rank_query(&det->btree, x, &less, &equal);
            cs_status_t st = cs_btree_insert(&det->btree, x);
            if (st != CS_OK) {
                return st;
            }
            double u = cs_rng_uniform01(&det->rng);
            double p_val = ((double)less + u * (1.0 + (double)equal)) / (double)det->btree.total_count;
            det->last_p_value = p_val;
            cs_eval_rank_from_p_16(det->lam, p_val, out_e);
            break;
        }
        default:
            return CS_ERR_INVALID;
    }
    return CS_OK;
}

/* =========================================================================
 * Vectorized Accumulator Updates (E-SR zero-exp & Log-Domain E-CUSUM)
 * ========================================================================= */

static inline double cs_accum_update_16(
    double acc[CS_EDET_LANES],
    const double e_lanes[CS_EDET_LANES],
    cs_edet_accum_t mode,
    double *out_e_mean)
{
    double e_sum = 0.0;
    double stat = 0.0;

    if (mode == CS_ACCUM_ESR) {
#if defined(CS_SIMD_NEON)
        float64x2_t vone = vdupq_n_f64(1.0);
        float64x2_t vcap = vdupq_n_f64(CS_ACC_CAP);
        float64x2_t rsum0 = vdupq_n_f64(0.0);
        float64x2_t rsum1 = vdupq_n_f64(0.0);
        float64x2_t esum0 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < 16U; i += 4U) {
            float64x2_t r0 = vld1q_f64(acc + i);
            float64x2_t e0 = vld1q_f64(e_lanes + i);
            float64x2_t r1 = vld1q_f64(acc + i + 2U);
            float64x2_t e1 = vld1q_f64(e_lanes + i + 2U);

            r0 = vminq_f64(vmulq_f64(vaddq_f64(r0, vone), e0), vcap);
            r1 = vminq_f64(vmulq_f64(vaddq_f64(r1, vone), e1), vcap);

            vst1q_f64(acc + i, r0);
            vst1q_f64(acc + i + 2U, r1);

            rsum0 = vaddq_f64(rsum0, r0);
            rsum1 = vaddq_f64(rsum1, r1);
            esum0 = vaddq_f64(esum0, vaddq_f64(e0, e1));
        }
        stat = vaddvq_f64(vaddq_f64(rsum0, rsum1)) * (1.0 / 16.0);
        e_sum = vaddvq_f64(esum0);
#elif defined(CS_SIMD_AVX2)
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vcap = _mm256_set1_pd(CS_ACC_CAP);
        __m256d rsum = _mm256_setzero_pd();
        __m256d esum = _mm256_setzero_pd();
        for (size_t i = 0; i < 16U; i += 4U) {
            __m256d r = _mm256_loadu_pd(acc + i);
            __m256d e = _mm256_loadu_pd(e_lanes + i);
            r = _mm256_min_pd(_mm256_mul_pd(_mm256_add_pd(r, vone), e), vcap);
            _mm256_storeu_pd(acc + i, r);
            rsum = _mm256_add_pd(rsum, r);
            esum = _mm256_add_pd(esum, e);
        }
        double rtmp[4], etmp[4];
        _mm256_storeu_pd(rtmp, rsum);
        _mm256_storeu_pd(etmp, esum);
        stat = ((rtmp[0] + rtmp[1]) + (rtmp[2] + rtmp[3])) * (1.0 / 16.0);
        e_sum = (etmp[0] + etmp[1]) + (etmp[2] + etmp[3]);
#else
        double r_sum = 0.0;
        for (size_t i = 0; i < 16U; ++i) {
            double r = (acc[i] + 1.0) * e_lanes[i];
            if (r > CS_ACC_CAP) {
                r = CS_ACC_CAP;
            }
            acc[i] = r;
            r_sum += r;
            e_sum += e_lanes[i];
        }
        stat = r_sum * (1.0 / 16.0);
#endif
    } else {
        /* Log-domain E-CUSUM: S_t^(m) = max(0, S_{t-1}^(m)) + log(e_t^(m)),
         * valid mixture C_t = (1/16) * sum_m exp(S_t^(m)). */
        for (size_t i = 0; i < 16U; ++i) {
            double e = e_lanes[i];
            e_sum += e;
            double base = (acc[i] > 0.0) ? acc[i] : 0.0;
            double le = log((e > 1.0e-300) ? e : 1.0e-300);
            double s = base + le;
            if (s < CS_LOG_MIN) {
                s = CS_LOG_MIN;
            } else if (s > CS_LOG_MAX) {
                s = CS_LOG_MAX;
            }
            acc[i] = s;
        }
#if defined(CS_SIMD_NEON)
        float64x2_t csum0 = vdupq_n_f64(0.0);
        float64x2_t csum1 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < 16U; i += 4U) {
            csum0 = vaddq_f64(csum0, cs_vexp_f64(vld1q_f64(acc + i)));
            csum1 = vaddq_f64(csum1, cs_vexp_f64(vld1q_f64(acc + i + 2U)));
        }
        stat = vaddvq_f64(vaddq_f64(csum0, csum1)) * (1.0 / 16.0);
#elif defined(CS_SIMD_AVX2)
        __m256d csum = _mm256_setzero_pd();
        for (size_t i = 0; i < 16U; i += 4U) {
            csum = _mm256_add_pd(csum, cs_vexp_f64(_mm256_loadu_pd(acc + i)));
        }
        double ctmp[4];
        _mm256_storeu_pd(ctmp, csum);
        stat = ((ctmp[0] + ctmp[1]) + (ctmp[2] + ctmp[3])) * (1.0 / 16.0);
#else
        double c_sum = 0.0;
        for (size_t i = 0; i < 16U; ++i) {
            c_sum += cs_scalar_exp(acc[i]);
        }
        stat = c_sum * (1.0 / 16.0);
#endif
    }

    if (out_e_mean) {
        *out_e_mean = e_sum * (1.0 / 16.0);
    }
    return stat;
}

cs_status_t cs_edet_step(cs_edetector_t *det, double x, cs_edet_result_t *out) {
    if (!det) {
        return CS_ERR_INVALID;
    }
    double e_lanes[CS_EDET_LANES];
    cs_status_t st = cs_edet_eval_lanes(det, x, e_lanes);
    if (st != CS_OK) {
        return st;
    }

    double e_mean = 1.0;
    double stat = cs_accum_update_16(det->acc, e_lanes, det->cfg.accum, &e_mean);

    det->step_count++;
    det->cur_stat = stat;
    det->last_e_mean = e_mean;
    bool alarmed = (stat >= det->cfg.threshold);
    if (alarmed) {
        det->alarm_count++;
    }

    if (out) {
        out->stat = stat;
        out->log_stat = log((stat > 1.0e-300) ? stat : 1.0e-300);
        out->last_e_mean = e_mean;
        out->last_p_value = det->last_p_value;
        out->step_count = det->step_count;
        out->alarm_count = det->alarm_count;
        out->epoch_count = det->epoch_count;
        out->alarmed = alarmed;
    }

    if (alarmed && det->cfg.auto_reset) {
        cs_edet_reset(det);
    }
    return CS_OK;
}

/* Register-resident NEON hot batch kernel for Catoni E-SR (zero-exp, zero state loads/stores) */
#if defined(CS_SIMD_NEON)
static cs_status_t cs_edet_batch_catoni_esr_neon(
    cs_edetector_t *det,
    const double *xs,
    size_t n,
    double *out_stats,
    uint8_t *out_alarms,
    cs_edet_result_t *last_out)
{
    const double mu0 = det->cfg.mu0;
    const double d_clamp = det->d_clamp;
    const double thr = det->cfg.threshold;
    const bool auto_reset = det->cfg.auto_reset;

    float64x2_t l0 = vld1q_f64(det->lam + 0);
    float64x2_t l1 = vld1q_f64(det->lam + 2);
    float64x2_t l2 = vld1q_f64(det->lam + 4);
    float64x2_t l3 = vld1q_f64(det->lam + 6);
    float64x2_t l4 = vld1q_f64(det->lam + 8);
    float64x2_t l5 = vld1q_f64(det->lam + 10);
    float64x2_t l6 = vld1q_f64(det->lam + 12);
    float64x2_t l7 = vld1q_f64(det->lam + 14);

    float64x2_t d0 = vld1q_f64(det->inv_denom + 0);
    float64x2_t d1 = vld1q_f64(det->inv_denom + 2);
    float64x2_t d2 = vld1q_f64(det->inv_denom + 4);
    float64x2_t d3 = vld1q_f64(det->inv_denom + 6);
    float64x2_t d4 = vld1q_f64(det->inv_denom + 8);
    float64x2_t d5 = vld1q_f64(det->inv_denom + 10);
    float64x2_t d6 = vld1q_f64(det->inv_denom + 12);
    float64x2_t d7 = vld1q_f64(det->inv_denom + 14);

    float64x2_t r0 = vld1q_f64(det->acc + 0);
    float64x2_t r1 = vld1q_f64(det->acc + 2);
    float64x2_t r2 = vld1q_f64(det->acc + 4);
    float64x2_t r3 = vld1q_f64(det->acc + 6);
    float64x2_t r4 = vld1q_f64(det->acc + 8);
    float64x2_t r5 = vld1q_f64(det->acc + 10);
    float64x2_t r6 = vld1q_f64(det->acc + 12);
    float64x2_t r7 = vld1q_f64(det->acc + 14);

    const float64x2_t vone = vdupq_n_f64(1.0);
    const float64x2_t vhalf = vdupq_n_f64(0.5);
    const float64x2_t vzero = vdupq_n_f64(0.0);
    const float64x2_t vcap = vdupq_n_f64(CS_ACC_CAP);

    double stat = det->cur_stat;
    bool alarmed = false;

#define CS_CATONI_STEP_LANE(r, l, invd)                                      \
    do {                                                                     \
        float64x2_t y_ = vmulq_f64((l), vdx);                                \
        float64x2_t t_ = vfmaq_f64(vone, vhalf, vmulq_f64(y_, y_));          \
        float64x2_t pos_ = vaddq_f64(t_, y_);                                \
        float64x2_t neg_ = vdivq_f64(vone, vsubq_f64(t_, y_));               \
        float64x2_t psi_ = vbslq_f64(vcgeq_f64(y_, vzero), pos_, neg_);      \
        float64x2_t e_ = vmulq_f64(psi_, (invd));                            \
        (r) = vminq_f64(vmulq_f64(vaddq_f64((r), vone), e_), vcap);          \
    } while (0)

    for (size_t t = 0; t < n; ++t) {
        double x = xs[t];
        if (!isfinite(x)) {
            vst1q_f64(det->acc + 0, r0);
            vst1q_f64(det->acc + 2, r1);
            vst1q_f64(det->acc + 4, r2);
            vst1q_f64(det->acc + 6, r3);
            vst1q_f64(det->acc + 8, r4);
            vst1q_f64(det->acc + 10, r5);
            vst1q_f64(det->acc + 12, r6);
            vst1q_f64(det->acc + 14, r7);
            if (t > 0U) {
                det->cur_stat = (alarmed && auto_reset) ? 0.0 : stat;
                if (alarmed && auto_reset) {
                    det->last_e_mean = 1.0;
                } else {
                    double e_prev[CS_EDET_LANES];
                    cs_eval_catoni_16(det->lam, det->inv_denom, mu0, d_clamp, xs[t - 1U], e_prev);
                    double esum = 0.0;
                    for (size_t m = 0; m < CS_EDET_LANES; ++m) {
                        esum += e_prev[m];
                    }
                    det->last_e_mean = esum * (1.0 / 16.0);
                }
            }
            return CS_ERR_NONFINITE;
        }
        float64x2_t vdx = vdupq_n_f64(cs_clamp_diff(x, mu0, d_clamp));

        CS_CATONI_STEP_LANE(r0, l0, d0);
        CS_CATONI_STEP_LANE(r1, l1, d1);
        CS_CATONI_STEP_LANE(r2, l2, d2);
        CS_CATONI_STEP_LANE(r3, l3, d3);
        CS_CATONI_STEP_LANE(r4, l4, d4);
        CS_CATONI_STEP_LANE(r5, l5, d5);
        CS_CATONI_STEP_LANE(r6, l6, d6);
        CS_CATONI_STEP_LANE(r7, l7, d7);

        float64x2_t s01 = vaddq_f64(r0, r1);
        float64x2_t s23 = vaddq_f64(r2, r3);
        float64x2_t s45 = vaddq_f64(r4, r5);
        float64x2_t s67 = vaddq_f64(r6, r7);
        float64x2_t s03 = vaddq_f64(s01, s23);
        float64x2_t s47 = vaddq_f64(s45, s67);
        stat = vaddvq_f64(vaddq_f64(s03, s47)) * (1.0 / 16.0);

        det->step_count++;
        alarmed = (stat >= thr);
        if (out_stats) {
            out_stats[t] = stat;
        }
        if (out_alarms) {
            out_alarms[t] = alarmed ? 1U : 0U;
        }
        if (alarmed) {
            det->alarm_count++;
            if (auto_reset) {
                r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = vzero;
            }
        }
    }
#undef CS_CATONI_STEP_LANE

    vst1q_f64(det->acc + 0, r0);
    vst1q_f64(det->acc + 2, r1);
    vst1q_f64(det->acc + 4, r2);
    vst1q_f64(det->acc + 6, r3);
    vst1q_f64(det->acc + 8, r4);
    vst1q_f64(det->acc + 10, r5);
    vst1q_f64(det->acc + 12, r6);
    vst1q_f64(det->acc + 14, r7);

    double e_last[CS_EDET_LANES];
    cs_eval_catoni_16(det->lam, det->inv_denom, mu0, d_clamp, xs[n - 1U], e_last);
    double last_esum = 0.0;
    for (size_t m = 0; m < CS_EDET_LANES; ++m) {
        last_esum += e_last[m];
    }
    double last_e_mean = last_esum * (1.0 / 16.0);

    det->cur_stat = (alarmed && auto_reset) ? 0.0 : stat;
    det->last_e_mean = (alarmed && auto_reset) ? 1.0 : last_e_mean;
    det->last_p_value = NAN;
    if (last_out) {
        last_out->stat = stat;
        last_out->log_stat = log((stat > 1.0e-300) ? stat : 1.0e-300);
        last_out->last_e_mean = last_e_mean;
        last_out->last_p_value = NAN;
        last_out->step_count = det->step_count;
        last_out->alarm_count = det->alarm_count;
        last_out->epoch_count = det->epoch_count;
        last_out->alarmed = alarmed;
    }
    return CS_OK;
}
#endif /* CS_SIMD_NEON */

cs_status_t cs_edet_step_batch(
    cs_edetector_t *det,
    const double *xs,
    size_t n,
    double *out_stats,
    uint8_t *out_alarms,
    cs_edet_result_t *last_out)
{
    if (!det || (n > 0 && !xs)) {
        return CS_ERR_INVALID;
    }
    if (n == 0) {
        if (last_out) {
            last_out->stat = det->cur_stat;
            last_out->log_stat = log((det->cur_stat > 1.0e-300) ? det->cur_stat : 1.0e-300);
            last_out->last_e_mean = det->last_e_mean;
            last_out->last_p_value = det->last_p_value;
            last_out->step_count = det->step_count;
            last_out->alarm_count = det->alarm_count;
            last_out->epoch_count = det->epoch_count;
            last_out->alarmed = false;
        }
        return CS_OK;
    }

#if defined(CS_SIMD_NEON)
    if (det->eff_family == CS_FAMILY_CATONI && det->cfg.accum == CS_ACCUM_ESR) {
        return cs_edet_batch_catoni_esr_neon(det, xs, n, out_stats, out_alarms, last_out);
    }
#endif

    cs_edet_result_t res;
    memset(&res, 0, sizeof(res));
    for (size_t i = 0; i < n; ++i) {
        cs_status_t st = cs_edet_step(det, xs[i], &res);
        if (st != CS_OK) {
            return st;
        }
        if (out_stats) {
            out_stats[i] = res.stat;
        }
        if (out_alarms) {
            out_alarms[i] = res.alarmed ? 1U : 0U;
        }
    }
    if (last_out) {
        *last_out = res;
    }
    return CS_OK;
}

cs_edet_family_t cs_edet_effective_family(const cs_edetector_t *det) {
    return det ? det->eff_family : CS_FAMILY_AUTO;
}

double cs_edet_current_stat(const cs_edetector_t *det) {
    return det ? det->cur_stat : 0.0;
}

uint64_t cs_edet_step_count(const cs_edetector_t *det) {
    return det ? det->step_count : 0ULL;
}

/* =========================================================================
 * Cross-Agent E-Fusion (Valid Under Arbitrary Unknown Cross-Agent Correlation)
 * ========================================================================= */

struct cs_efusion {
    double acc[CS_EDET_LANES]; /* Shared accumulator for CS_FUSION_STEP */
    size_t n_agents;
    cs_efusion_mode_t mode;
    cs_edet_accum_t accum;
    double threshold;
    bool auto_reset;

    cs_edetector_t **agents;
    double *weights;
    double cur_stat;
    uint64_t step_count;
    uint64_t alarm_count;
};

cs_efusion_t *cs_efusion_create(const cs_efusion_config_t *cfg) {
    if (!cfg || cfg->n_agents == 0 || !isfinite(cfg->threshold) || cfg->threshold <= 1.0) {
        return NULL;
    }

    cs_efusion_t *fus = (cs_efusion_t *)cs_aligned_calloc(64U, 1U, sizeof(cs_efusion_t));
    if (!fus) {
        return NULL;
    }
    fus->n_agents = cfg->n_agents;
    fus->mode = cfg->mode;
    fus->accum = cfg->accum;
    fus->threshold = cfg->threshold;
    fus->auto_reset = cfg->auto_reset;

    fus->agents = (cs_edetector_t **)calloc(cfg->n_agents, sizeof(cs_edetector_t *));
    fus->weights = (double *)calloc(cfg->n_agents, sizeof(double));
    if (!fus->agents || !fus->weights) {
        cs_efusion_destroy(fus);
        return NULL;
    }

    double wsum = 0.0;
    for (size_t k = 0; k < cfg->n_agents; ++k) {
        double w = cfg->weights ? cfg->weights[k] : (1.0 / (double)cfg->n_agents);
        if (!isfinite(w) || w < 0.0) {
            cs_efusion_destroy(fus);
            return NULL;
        }
        fus->weights[k] = w;
        wsum += w;
    }
    if (wsum <= 0.0) {
        cs_efusion_destroy(fus);
        return NULL;
    }
    for (size_t k = 0; k < cfg->n_agents; ++k) {
        fus->weights[k] /= wsum;
    }

    for (size_t k = 0; k < cfg->n_agents; ++k) {
        cs_edet_config_t acfg;
        if (cfg->agent_configs) {
            acfg = cfg->agent_configs[k];
        } else {
            cs_edet_config_default(&acfg);
            acfg.family = CS_FAMILY_CATONI;
            acfg.mu0 = 0.0;
            acfg.sigma = 1.0;
        }
        acfg.accum = cfg->accum;
        acfg.threshold = cfg->threshold;
        acfg.auto_reset = false; /* Fusion manages resets jointly */
        if (acfg.seed == 0x517cc1b727220a95ULL) {
            acfg.seed ^= (uint64_t)(k + 1U) * 0x9e3779b97f4a7c15ULL;
        }
        fus->agents[k] = cs_edet_create(&acfg);
        if (!fus->agents[k]) {
            cs_efusion_destroy(fus);
            return NULL;
        }
    }

    return fus;
}

void cs_efusion_destroy(cs_efusion_t *fus) {
    if (!fus) {
        return;
    }
    if (fus->agents) {
        for (size_t k = 0; k < fus->n_agents; ++k) {
            cs_edet_destroy(fus->agents[k]);
        }
        free(fus->agents);
    }
    free(fus->weights);
    cs_aligned_free(fus);
}

void cs_efusion_reset(cs_efusion_t *fus) {
    if (!fus) {
        return;
    }
    memset(fus->acc, 0, sizeof(fus->acc));
    for (size_t k = 0; k < fus->n_agents; ++k) {
        cs_edet_reset(fus->agents[k]);
    }
    fus->cur_stat = 0.0;
}

cs_status_t cs_efusion_step(
    cs_efusion_t *fus,
    const double *x_agents,
    const double *predictable_weights,
    cs_edet_result_t *out)
{
    if (!fus || !x_agents) {
        return CS_ERR_INVALID;
    }
    /* Process-level averaging R_bar_t = sum_k w_k R_t^(k) is ONLY a valid e-detector
     * for fixed prior weights w_k; time-varying predictable weights violate ARL >= gamma. */
    if (fus->mode == CS_FUSION_PROCESS && predictable_weights != NULL) {
        return CS_ERR_INVALID;
    }

    const size_t K = fus->n_agents;
    const double *w_src = predictable_weights ? predictable_weights : fus->weights;
    double wsum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        if (!isfinite(x_agents[k])) {
            return CS_ERR_NONFINITE;
        }
        if (!isfinite(w_src[k]) || w_src[k] < 0.0) {
            return CS_ERR_INVALID;
        }
        wsum += w_src[k];
    }
    if (wsum <= 0.0) {
        return CS_ERR_INVALID;
    }
    double inv_wsum = 1.0 / wsum;

    double stat = 0.0;
    double e_mean = 1.0;
    uint32_t max_epoch = 0U;

    if (fus->mode == CS_FUSION_STEP) {
        double e_bar[CS_EDET_LANES];
        memset(e_bar, 0, sizeof(e_bar));
        double e_k[CS_EDET_LANES];

        for (size_t k = 0; k < K; ++k) {
            cs_status_t st = cs_edet_eval_lanes(fus->agents[k], x_agents[k], e_k);
            if (st != CS_OK) {
                return st;
            }
            if (fus->agents[k]->epoch_count > max_epoch) {
                max_epoch = fus->agents[k]->epoch_count;
            }
            double wk = w_src[k] * inv_wsum;
            for (size_t m = 0; m < CS_EDET_LANES; ++m) {
                e_bar[m] += wk * e_k[m];
            }
        }
        stat = cs_accum_update_16(fus->acc, e_bar, fus->accum, &e_mean);
    } else {
        /* Process-level convex averaging: R_bar_t = sum_k w_k * R_t^(k).
         * Valid for heterogeneous families & Vovk Rank under arbitrary correlation. */
        double r_bar = 0.0;
        double em_bar = 0.0;
        for (size_t k = 0; k < K; ++k) {
            cs_edet_result_t r_k;
            cs_status_t st = cs_edet_step(fus->agents[k], x_agents[k], &r_k);
            if (st != CS_OK) {
                return st;
            }
            if (r_k.epoch_count > max_epoch) {
                max_epoch = r_k.epoch_count;
            }
            double wk = w_src[k] * inv_wsum;
            r_bar += wk * r_k.stat;
            em_bar += wk * r_k.last_e_mean;
        }
        stat = r_bar;
        e_mean = em_bar;
    }

    fus->step_count++;
    fus->cur_stat = stat;
    bool alarmed = (stat >= fus->threshold);
    if (alarmed) {
        fus->alarm_count++;
    }

    if (out) {
        out->stat = stat;
        out->log_stat = log((stat > 1.0e-300) ? stat : 1.0e-300);
        out->last_e_mean = e_mean;
        out->last_p_value = NAN;
        out->step_count = fus->step_count;
        out->alarm_count = fus->alarm_count;
        out->epoch_count = max_epoch;
        out->alarmed = alarmed;
    }

    if (alarmed && fus->auto_reset) {
        cs_efusion_reset(fus);
    }
    return CS_OK;
}

cs_status_t cs_efusion_step_batch(
    cs_efusion_t *fus,
    const double *x_matrix,
    size_t n_steps,
    double *out_stats,
    uint8_t *out_alarms,
    cs_edet_result_t *last_out)
{
    if (!fus || (n_steps > 0 && !x_matrix)) {
        return CS_ERR_INVALID;
    }
    if (n_steps == 0) {
        if (last_out) {
            memset(last_out, 0, sizeof(*last_out));
            last_out->stat = fus->cur_stat;
            last_out->log_stat = log((fus->cur_stat > 1.0e-300) ? fus->cur_stat : 1.0e-300);
            last_out->step_count = fus->step_count;
            last_out->alarm_count = fus->alarm_count;
        }
        return CS_OK;
    }

    cs_edet_result_t res;
    memset(&res, 0, sizeof(res));
    for (size_t t = 0; t < n_steps; ++t) {
        cs_status_t st = cs_efusion_step(fus, x_matrix + t * fus->n_agents, NULL, &res);
        if (st != CS_OK) {
            return st;
        }
        if (out_stats) {
            out_stats[t] = res.stat;
        }
        if (out_alarms) {
            out_alarms[t] = res.alarmed ? 1U : 0U;
        }
    }
    if (last_out) {
        *last_out = res;
    }
    return CS_OK;
}
