#include "cs_internal.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cs_digest {
    cs_digest_config_t cfg;
    double eps_eff;

    /* Circular ring buffer of exact raw ticks for lossless O(log N) zoom() */
    cs_ring_record_t *ring;
    size_t ring_cap;
    size_t ring_head;
    size_t ring_size;
    bool ring_evicted;

    /* Circular ring buffer of closed Swing PLA segments */
    cs_segment_t *segs;
    size_t seg_cap;
    size_t seg_head;
    size_t seg_size;

    /* Circular ring buffer of verbatim conformal anomalies */
    cs_anomaly_t *anoms;
    size_t anom_cap;
    size_t anom_head;
    size_t anom_size;

    /* Active open Swing PLA segment state */
    bool has_open;
    double open_t0;
    double open_y0;
    double open_t_last;
    double open_y_last;
    double m_min;
    double m_max;
    double s_dt2;
    double s_dtdy;
    uint32_t open_count;

    /* Monotonic timestamp & wake-up gate state */
    bool has_any;
    double last_t;
    uint64_t total_points;
    uint64_t total_segments;
    uint64_t total_anomalies;
    uint64_t wake_count;
    uint64_t last_wake_point;
    uint32_t consec_anoms;
};

void cs_digest_config_default(cs_digest_config_t *cfg) {
    if (!cfg) {
        return;
    }
    cfg->epsilon = 0.05;
    cfg->ring_capacity = 65536U;
    cfg->max_segments = 4096U;
    cfg->max_anomalies = 1024U;
    cfg->wake_cooldown = 0U;
    cfg->anomaly_burst = 1U;
}

cs_digest_t *cs_digest_create(const cs_digest_config_t *cfg) {
    cs_digest_config_t local_cfg;
    if (cfg) {
        local_cfg = *cfg;
    } else {
        cs_digest_config_default(&local_cfg);
    }
    if (!isfinite(local_cfg.epsilon) || local_cfg.epsilon < 0.0) {
        return NULL;
    }
    if (local_cfg.ring_capacity == 0U) {
        local_cfg.ring_capacity = 1024U;
    }
    if (local_cfg.max_segments == 0U) {
        local_cfg.max_segments = 1024U;
    }
    if (local_cfg.max_anomalies == 0U) {
        local_cfg.max_anomalies = 256U;
    }
    if (local_cfg.anomaly_burst == 0U) {
        local_cfg.anomaly_burst = 1U;
    }

    cs_digest_t *dig = (cs_digest_t *)calloc(1U, sizeof(cs_digest_t));
    if (!dig) {
        return NULL;
    }
    dig->cfg = local_cfg;
    /* Tiny machine-epsilon safety margin guarantees strict L_inf <= epsilon under float rounding */
    dig->eps_eff = (local_cfg.epsilon > 0.0) ? (local_cfg.epsilon * (1.0 - 1.0e-12)) : 0.0;

    dig->ring_cap = local_cfg.ring_capacity;
    dig->seg_cap = local_cfg.max_segments;
    dig->anom_cap = local_cfg.max_anomalies;

    dig->ring = (cs_ring_record_t *)calloc(dig->ring_cap, sizeof(cs_ring_record_t));
    dig->segs = (cs_segment_t *)calloc(dig->seg_cap, sizeof(cs_segment_t));
    dig->anoms = (cs_anomaly_t *)calloc(dig->anom_cap, sizeof(cs_anomaly_t));
    if (!dig->ring || !dig->segs || !dig->anoms) {
        cs_digest_destroy(dig);
        return NULL;
    }
    return dig;
}

void cs_digest_destroy(cs_digest_t *dig) {
    if (!dig) {
        return;
    }
    free(dig->ring);
    free(dig->segs);
    free(dig->anoms);
    free(dig);
}

void cs_digest_reset(cs_digest_t *dig) {
    if (!dig) {
        return;
    }
    dig->ring_head = 0U;
    dig->ring_size = 0U;
    dig->ring_evicted = false;
    dig->seg_head = 0U;
    dig->seg_size = 0U;
    dig->anom_head = 0U;
    dig->anom_size = 0U;
    dig->has_open = false;
    dig->open_count = 0U;
    dig->has_any = false;
    dig->last_t = 0.0;
    dig->total_points = 0ULL;
    dig->total_segments = 0ULL;
    dig->total_anomalies = 0ULL;
    dig->wake_count = 0ULL;
    dig->last_wake_point = 0ULL;
    dig->consec_anoms = 0U;
}

static inline void cs_digest_append_seg(cs_digest_t *dig, const cs_segment_t *seg) {
    dig->segs[dig->seg_head] = *seg;
    if (++dig->seg_head == dig->seg_cap) {
        dig->seg_head = 0U;
    }
    if (dig->seg_size < dig->seg_cap) {
        dig->seg_size++;
    }
    dig->total_segments++;
}

static inline void cs_digest_append_anom(cs_digest_t *dig, const cs_anomaly_t *anom) {
    dig->anoms[dig->anom_head] = *anom;
    if (++dig->anom_head == dig->anom_cap) {
        dig->anom_head = 0U;
    }
    if (dig->anom_size < dig->anom_cap) {
        dig->anom_size++;
    }
    dig->total_anomalies++;
}

static double cs_digest_open_slope(const cs_digest_t *dig) {
    if (dig->open_count <= 1U || dig->open_t_last <= dig->open_t0) {
        return 0.0;
    }
    double mid = dig->m_min + 0.5 * (dig->m_max - dig->m_min);
    if (!isfinite(mid)) {
        mid = 0.5 * dig->m_min + 0.5 * dig->m_max;
    }
    double m = (dig->s_dt2 > 0.0 && isfinite(dig->s_dt2) && isfinite(dig->s_dtdy))
        ? (dig->s_dtdy / dig->s_dt2)
        : mid;
    if (!isfinite(m)) {
        m = mid;
    }
    if (m < dig->m_min) {
        m = dig->m_min;
    }
    if (m > dig->m_max) {
        m = dig->m_max;
    }
    return m;
}

static cs_segment_t cs_digest_build_open_seg(const cs_digest_t *dig) {
    cs_segment_t seg;
    double slope = cs_digest_open_slope(dig);
    seg.t_start = dig->open_t0;
    seg.y_start = dig->open_y0;
    seg.t_end = dig->open_t_last;
    seg.y_end = dig->open_y0 + slope * (dig->open_t_last - dig->open_t0);
    seg.slope = slope;
    seg.count = dig->open_count;
    seg.flags = 0U;
    return seg;
}

static bool cs_digest_close_open(cs_digest_t *dig) {
    if (!dig->has_open) {
        return false;
    }
    cs_segment_t seg = cs_digest_build_open_seg(dig);
    cs_digest_append_seg(dig, &seg);
    dig->has_open = false;
    dig->open_count = 0U;
    return true;
}

static void cs_digest_start_open(cs_digest_t *dig, double t, double y) {
    dig->has_open = true;
    dig->open_t0 = t;
    dig->open_y0 = y;
    dig->open_t_last = t;
    dig->open_y_last = y;
    dig->m_min = -DBL_MAX;
    dig->m_max = DBL_MAX;
    dig->s_dt2 = 0.0;
    dig->s_dtdy = 0.0;
    dig->open_count = 1U;
}

cs_status_t cs_digest_push(
    cs_digest_t *dig,
    double t,
    double y,
    double y_hat,
    double q_t,
    double edet_stat,
    uint32_t flags,
    cs_digest_step_t *out)
{
    if (!dig) {
        return CS_ERR_INVALID;
    }
    if (!isfinite(t) || !isfinite(y)) {
        return CS_ERR_NONFINITE;
    }
    if (dig->has_any && t < dig->last_t) {
        return CS_ERR_INVALID;
    }

    /* 1. Store exact raw tick in circular ring buffer for lossless O(log N) zoom() */
    if (dig->ring_size == dig->ring_cap) {
        dig->ring_evicted = true;
    }
    cs_ring_record_t rec;
    rec.t = t;
    rec.y = y;
    rec.y_hat = y_hat;
    rec.q_t = q_t;
    rec.flags = flags;
    dig->ring[dig->ring_head] = rec;
    if (++dig->ring_head == dig->ring_cap) {
        dig->ring_head = 0U;
    }
    if (dig->ring_size < dig->ring_cap) {
        dig->ring_size++;
    }

    dig->has_any = true;
    dig->last_t = t;
    dig->total_points++;

    bool seg_emitted = false;
    const bool is_anom = (flags & CS_FLAG_ANOMALY) != 0U;
    const bool is_shift = (flags & CS_FLAG_SHIFT) != 0U;
    const bool is_rogue = (flags & CS_FLAG_ROGUE) != 0U;

    /* 2. Anomaly-Preserving Swing PLA update */
    if (is_anom) {
        /* Flush open PLA segment, then emit verbatim 0-error pinned segment + anomaly record */
        if (cs_digest_close_open(dig)) {
            seg_emitted = true;
        }
        cs_anomaly_t anom;
        anom.t = t;
        anom.y = y;
        anom.y_hat = y_hat;
        anom.residual = isfinite(y_hat) ? fabs(y - y_hat) : NAN;
        anom.q_t = q_t;
        anom.edet_stat = edet_stat;
        anom.flags = flags;
        cs_digest_append_anom(dig, &anom);

        cs_segment_t pinned;
        pinned.t_start = t;
        pinned.y_start = y;
        pinned.t_end = t;
        pinned.y_end = y;
        pinned.slope = 0.0;
        pinned.count = 1U;
        pinned.flags = flags | CS_FLAG_PINNED;
        cs_digest_append_seg(dig, &pinned);
        seg_emitted = true;
    } else {
        if (is_shift && dig->has_open) {
            /* Clean trajectory breakpoint right at certified regime change */
            cs_digest_close_open(dig);
            seg_emitted = true;
        }

        if (!dig->has_open) {
            cs_digest_start_open(dig, t, y);
        } else {
            double dt = t - dig->open_t0;
            double y_mag = fmax(fabs(y), fabs(dig->open_y0));
            double ulp_guard = 8.0 * DBL_EPSILON * y_mag;
            double eps_pt = (dig->eps_eff > ulp_guard) ? (dig->eps_eff - ulp_guard) : 0.0;

            if (!isfinite(dt)) {
                cs_digest_close_open(dig);
                seg_emitted = true;
                cs_digest_start_open(dig, t, y);
            } else if (dt <= 0.0) {
                if (fabs(y - dig->open_y0) <= eps_pt) {
                    dig->open_count++;
                } else {
                    cs_digest_close_open(dig);
                    seg_emitted = true;
                    cs_digest_start_open(dig, t, y);
                }
            } else {
                double dy = y - dig->open_y0;
                double inv_dt = 1.0 / dt;
                double u_i = isfinite(inv_dt) ? ((dy + eps_pt) * inv_dt) : ((dy + eps_pt) / dt);
                double l_i = isfinite(inv_dt) ? ((dy - eps_pt) * inv_dt) : ((dy - eps_pt) / dt);
                double new_min = (l_i > dig->m_min) ? l_i : dig->m_min;
                double new_max = (u_i < dig->m_max) ? u_i : dig->m_max;

                if (isfinite(u_i) && isfinite(l_i) && new_min <= new_max) {
                    dig->m_min = new_min;
                    dig->m_max = new_max;
                    dig->open_t_last = t;
                    dig->open_y_last = y;
                    dig->s_dt2 += dt * dt;
                    dig->s_dtdy += dt * dy;
                    dig->open_count++;
                } else {
                    /* Slope cone collapsed -> close segment & anchor new segment at (t, y) */
                    cs_digest_close_open(dig);
                    seg_emitted = true;
                    cs_digest_start_open(dig, t, y);
                }
            }
        }
    }

    /* 3. System-1 -> System-2 Wake-Up Gate */
    bool should_wake = false;
    if (is_anom) {
        dig->consec_anoms++;
    } else {
        dig->consec_anoms = 0U;
    }

    if (is_shift || is_rogue) {
        should_wake = true;
    } else if (is_anom && dig->consec_anoms >= dig->cfg.anomaly_burst) {
        if (dig->wake_count == 0ULL ||
            (dig->total_points - dig->last_wake_point) > (uint64_t)dig->cfg.wake_cooldown) {
            should_wake = true;
        }
    }

    if (should_wake) {
        dig->wake_count++;
        dig->last_wake_point = dig->total_points;
    }

    if (out) {
        out->total_points = dig->total_points;
        out->total_segments = dig->total_segments;
        out->total_anomalies = dig->total_anomalies;
        out->wake_count = dig->wake_count;
        out->should_wake = should_wake;
        out->segment_emitted = seg_emitted;
    }

    return CS_OK;
}

static inline void cs_digest_emit_reg_open(
    cs_digest_t *dig,
    double open_t0,
    double open_y0,
    double open_t_last,
    double m_min,
    double m_max,
    double s_dt2,
    double s_dtdy,
    uint32_t open_count)
{
    double slope = 0.0;
    if (open_count > 1U && open_t_last > open_t0) {
        double mid = m_min + 0.5 * (m_max - m_min);
        if (!isfinite(mid)) {
            mid = 0.5 * m_min + 0.5 * m_max;
        }
        double m = (s_dt2 > 0.0 && isfinite(s_dt2) && isfinite(s_dtdy))
            ? (s_dtdy / s_dt2)
            : mid;
        if (!isfinite(m)) m = mid;
        if (m < m_min) m = m_min;
        if (m > m_max) m = m_max;
        slope = m;
    }
    cs_segment_t seg;
    seg.t_start = open_t0;
    seg.y_start = open_y0;
    seg.t_end = open_t_last;
    seg.y_end = open_y0 + slope * (open_t_last - open_t0);
    seg.slope = slope;
    seg.count = open_count;
    seg.flags = 0U;
    cs_digest_append_seg(dig, &seg);
}

/* Tight register-resident + SIMD stationary fast path for flag-free batches.
 * Preserves exact ULP-guarded L_inf <= epsilon, exact segment boundaries, and lossless ring buffer. */
static cs_status_t cs_digest_push_batch_fast(
    cs_digest_t *dig,
    const double * CS_RESTRICT ts,
    const double * CS_RESTRICT ys,
    const double * CS_RESTRICT y_hats,
    const double * CS_RESTRICT qs,
    size_t n,
    uint8_t * CS_RESTRICT out_wakes,
    cs_digest_step_t *last_out)
{
    cs_ring_record_t * CS_RESTRICT ring = dig->ring;
    const size_t ring_cap = dig->ring_cap;
    const double eps_eff = dig->eps_eff;
    const double ulp8 = 8.0 * DBL_EPSILON;
    const bool simple_rec = (y_hats == NULL && qs == NULL);

    size_t ring_head = dig->ring_head;
    bool has_open = dig->has_open;
    double open_t0 = dig->open_t0;
    double open_y0 = dig->open_y0;
    double abs_y0 = has_open ? fabs(open_y0) : 0.0;
    double open_t_last = dig->open_t_last;
    double open_y_last = dig->open_y_last;
    double m_min = dig->m_min;
    double m_max = dig->m_max;
    double s_dt2 = dig->s_dt2;
    double s_dtdy = dig->s_dtdy;
    uint32_t open_count = dig->open_count;
    bool has_any = dig->has_any;
    double last_t = dig->last_t;
    bool seg_emitted = false;
    cs_status_t err = CS_OK;

#if defined(CS_SIMD_NEON)
    const float64x2_t vone = vdupq_n_f64(1.0);
    const float64x2_t vzero = vdupq_n_f64(0.0);
    const float64x2_t veps_eff = vdupq_n_f64(eps_eff);
    const float64x2_t vulp8 = vdupq_n_f64(ulp8);
#elif defined(CS_SIMD_AVX2)
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vzero = _mm256_setzero_pd();
    const __m256d veps_eff = _mm256_set1_pd(eps_eff);
    const __m256d vulp8 = _mm256_set1_pd(ulp8);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
#endif

    size_t i = 0U;
    while (i < n) {
#if defined(CS_SIMD_NEON)
        /* 2-wide NEON stationary fast path: if intersection of cone with BOTH points is non-empty,
         * monotonicity of m_min/m_max proves point i was also valid (exact mathematical equivalence). */
        if (CS_LIKELY(has_open && i + 2U <= n && abs_y0 < 1.0e150 && eps_eff < 1.0e150)) {
            double t0 = ts[i];
            double t1 = ts[i + 1U];
            double y0 = ys[i];
            double y1 = ys[i + 1U];
            double dt0 = t0 - open_t0;
            double dt1 = t1 - open_t0;
            if (CS_LIKELY(t0 >= last_t && t1 >= t0 && dt0 > 1.0e-140 && dt1 < 1.0e150 &&
                          fabs(y0) < 1.0e150 && fabs(y1) < 1.0e150)) {
                float64x2_t vt = vld1q_f64(ts + i);
                float64x2_t vy = vld1q_f64(ys + i);
                float64x2_t vdt = vsubq_f64(vt, vdupq_n_f64(open_t0));
                float64x2_t vdy = vsubq_f64(vy, vdupq_n_f64(open_y0));
                float64x2_t vymag = vmaxq_f64(vabsq_f64(vy), vdupq_n_f64(abs_y0));
                float64x2_t veps = vmaxq_f64(vzero, vfmsq_f64(veps_eff, vulp8, vymag));
                float64x2_t vinv = vdivq_f64(vone, vdt);
                float64x2_t vu = vmulq_f64(vaddq_f64(vdy, veps), vinv);
                float64x2_t vl = vmulq_f64(vsubq_f64(vdy, veps), vinv);
                double max_l = vmaxvq_f64(vl);
                double min_u = vminvq_f64(vu);
                double new_min = (max_l > m_min) ? max_l : m_min;
                double new_max = (min_u < m_max) ? min_u : m_max;
                if (CS_LIKELY(new_min <= new_max)) {
                    m_min = new_min;
                    m_max = new_max;
                    open_t_last = t1;
                    open_y_last = y1;
                    double dy0 = y0 - open_y0;
                    double dy1 = y1 - open_y0;
                    s_dt2 = (s_dt2 + dt0 * dt0) + dt1 * dt1;
                    s_dtdy = (s_dtdy + dt0 * dy0) + dt1 * dy1;
                    open_count += 2U;
                    last_t = t1;
                    seg_emitted = false;

                    if (CS_LIKELY(ring_head + 2U <= ring_cap)) {
                        cs_ring_record_t *r = ring + ring_head;
                        r[0].t = t0;
                        r[0].y = y0;
                        r[0].y_hat = simple_rec ? y0 : (y_hats ? y_hats[i] : y0);
                        r[0].q_t = simple_rec ? NAN : (qs ? qs[i] : NAN);
                        r[0].flags = 0U;
                        r[1].t = t1;
                        r[1].y = y1;
                        r[1].y_hat = simple_rec ? y1 : (y_hats ? y_hats[i + 1U] : y1);
                        r[1].q_t = simple_rec ? NAN : (qs ? qs[i + 1U] : NAN);
                        r[1].flags = 0U;
                        ring_head += 2U;
                        if (CS_UNLIKELY(ring_head == ring_cap)) {
                            ring_head = 0U;
                        }
                    } else {
                        for (size_t k = 0U; k < 2U; ++k) {
                            cs_ring_record_t *r = ring + ring_head;
                            double tk = ts[i + k];
                            double yk = ys[i + k];
                            r->t = tk;
                            r->y = yk;
                            r->y_hat = y_hats ? y_hats[i + k] : yk;
                            r->q_t = qs ? qs[i + k] : NAN;
                            r->flags = 0U;
                            if (++ring_head == ring_cap) {
                                ring_head = 0U;
                            }
                        }
                    }
                    i += 2U;
                    continue;
                }
            }
        }
#elif defined(CS_SIMD_AVX2)
        if (CS_LIKELY(has_open && i + 4U <= n && abs_y0 < 1.0e150 && eps_eff < 1.0e150)) {
            double t0 = ts[i];
            double t1 = ts[i + 1U];
            double t2 = ts[i + 2U];
            double t3 = ts[i + 3U];
            double y0 = ys[i];
            double y1 = ys[i + 1U];
            double y2 = ys[i + 2U];
            double y3 = ys[i + 3U];
            double dt0 = t0 - open_t0;
            double dt3 = t3 - open_t0;
            if (CS_LIKELY(t0 >= last_t && t1 >= t0 && t2 >= t1 && t3 >= t2 &&
                          dt0 > 1.0e-140 && dt3 < 1.0e150 &&
                          fabs(y0) < 1.0e150 && fabs(y1) < 1.0e150 &&
                          fabs(y2) < 1.0e150 && fabs(y3) < 1.0e150)) {
                __m256d vt = _mm256_loadu_pd(ts + i);
                __m256d vy = _mm256_loadu_pd(ys + i);
                __m256d vdt = _mm256_sub_pd(vt, _mm256_set1_pd(open_t0));
                __m256d vdy = _mm256_sub_pd(vy, _mm256_set1_pd(open_y0));
                __m256d vymag = _mm256_max_pd(_mm256_andnot_pd(sign_mask, vy), _mm256_set1_pd(abs_y0));
                __m256d veps = _mm256_max_pd(vzero, cs_avx_fnmadd(vulp8, vymag, veps_eff));
                __m256d vinv = _mm256_div_pd(vone, vdt);
                __m256d vu = _mm256_mul_pd(_mm256_add_pd(vdy, veps), vinv);
                __m256d vl = _mm256_mul_pd(_mm256_sub_pd(vdy, veps), vinv);
                double ua[4], la[4];
                _mm256_storeu_pd(ua, vu);
                _mm256_storeu_pd(la, vl);
                double max_l = fmax(fmax(la[0], la[1]), fmax(la[2], la[3]));
                double min_u = fmin(fmin(ua[0], ua[1]), fmin(ua[2], ua[3]));
                double new_min = (max_l > m_min) ? max_l : m_min;
                double new_max = (min_u < m_max) ? min_u : m_max;
                if (CS_LIKELY(new_min <= new_max)) {
                    m_min = new_min;
                    m_max = new_max;
                    open_t_last = t3;
                    open_y_last = y3;
                    double dt1 = t1 - open_t0;
                    double dt2 = t2 - open_t0;
                    s_dt2 = (((s_dt2 + dt0 * dt0) + dt1 * dt1) + dt2 * dt2) + dt3 * dt3;
                    s_dtdy = (((s_dtdy + dt0 * (y0 - open_y0)) + dt1 * (y1 - open_y0)) +
                              dt2 * (y2 - open_y0)) + dt3 * (y3 - open_y0);
                    open_count += 4U;
                    last_t = t3;
                    seg_emitted = false;

                    if (CS_LIKELY(ring_head + 4U <= ring_cap)) {
                        cs_ring_record_t *r = ring + ring_head;
                        for (size_t k = 0U; k < 4U; ++k) {
                            double yk = ys[i + k];
                            r[k].t = ts[i + k];
                            r[k].y = yk;
                            r[k].y_hat = simple_rec ? yk : (y_hats ? y_hats[i + k] : yk);
                            r[k].q_t = simple_rec ? NAN : (qs ? qs[i + k] : NAN);
                            r[k].flags = 0U;
                        }
                        ring_head += 4U;
                        if (CS_UNLIKELY(ring_head == ring_cap)) {
                            ring_head = 0U;
                        }
                    } else {
                        for (size_t k = 0U; k < 4U; ++k) {
                            cs_ring_record_t *r = ring + ring_head;
                            double yk = ys[i + k];
                            r->t = ts[i + k];
                            r->y = yk;
                            r->y_hat = y_hats ? y_hats[i + k] : yk;
                            r->q_t = qs ? qs[i + k] : NAN;
                            r->flags = 0U;
                            if (++ring_head == ring_cap) {
                                ring_head = 0U;
                            }
                        }
                    }
                    i += 4U;
                    continue;
                }
            }
        }
#endif

        /* Scalar exact path (segment start/close, tail, or non-stationary point) */
        double t = ts[i];
        double y = ys[i];
        if (CS_UNLIKELY(!isfinite(t) || !isfinite(y))) {
            err = CS_ERR_NONFINITE;
            break;
        }
        if (CS_UNLIKELY(has_any && t < last_t)) {
            err = CS_ERR_INVALID;
            break;
        }

        cs_ring_record_t *r = ring + ring_head;
        r->t = t;
        r->y = y;
        r->y_hat = simple_rec ? y : (y_hats ? y_hats[i] : y);
        r->q_t = simple_rec ? NAN : (qs ? qs[i] : NAN);
        r->flags = 0U;
        if (++ring_head == ring_cap) {
            ring_head = 0U;
        }

        has_any = true;
        last_t = t;
        seg_emitted = false;

        if (CS_UNLIKELY(!has_open)) {
            has_open = true;
            open_t0 = t;
            open_y0 = y;
            abs_y0 = fabs(y);
            open_t_last = t;
            open_y_last = y;
            m_min = -DBL_MAX;
            m_max = DBL_MAX;
            s_dt2 = 0.0;
            s_dtdy = 0.0;
            open_count = 1U;
        } else {
            double dt = t - open_t0;
            double ay = fabs(y);
            double y_mag = (ay > abs_y0) ? ay : abs_y0;
            double ulp_guard = ulp8 * y_mag;
            double eps_pt = (eps_eff > ulp_guard) ? (eps_eff - ulp_guard) : 0.0;

            if (CS_UNLIKELY(!isfinite(dt))) {
                cs_digest_emit_reg_open(dig, open_t0, open_y0, open_t_last, m_min, m_max, s_dt2, s_dtdy, open_count);
                seg_emitted = true;
                open_t0 = t; open_y0 = y; abs_y0 = ay;
                open_t_last = t; open_y_last = y;
                m_min = -DBL_MAX; m_max = DBL_MAX; s_dt2 = 0.0; s_dtdy = 0.0; open_count = 1U;
            } else if (CS_UNLIKELY(dt <= 0.0)) {
                if (fabs(y - open_y0) <= eps_pt) {
                    open_count++;
                } else {
                    cs_digest_emit_reg_open(dig, open_t0, open_y0, open_t_last, m_min, m_max, s_dt2, s_dtdy, open_count);
                    seg_emitted = true;
                    open_t0 = t; open_y0 = y; abs_y0 = ay;
                    open_t_last = t; open_y_last = y;
                    m_min = -DBL_MAX; m_max = DBL_MAX; s_dt2 = 0.0; s_dtdy = 0.0; open_count = 1U;
                }
            } else {
                double dy = y - open_y0;
                double inv_dt = 1.0 / dt;
                double u_i = CS_LIKELY(isfinite(inv_dt)) ? ((dy + eps_pt) * inv_dt) : ((dy + eps_pt) / dt);
                double l_i = CS_LIKELY(isfinite(inv_dt)) ? ((dy - eps_pt) * inv_dt) : ((dy - eps_pt) / dt);
                double new_min = (l_i > m_min) ? l_i : m_min;
                double new_max = (u_i < m_max) ? u_i : m_max;

                if (CS_LIKELY(isfinite(u_i) && isfinite(l_i) && new_min <= new_max)) {
                    m_min = new_min;
                    m_max = new_max;
                    open_t_last = t;
                    open_y_last = y;
                    s_dt2 += dt * dt;
                    s_dtdy += dt * dy;
                    open_count++;
                } else {
                    cs_digest_emit_reg_open(dig, open_t0, open_y0, open_t_last, m_min, m_max, s_dt2, s_dtdy, open_count);
                    seg_emitted = true;
                    open_t0 = t; open_y0 = y; abs_y0 = ay;
                    open_t_last = t; open_y_last = y;
                    m_min = -DBL_MAX; m_max = DBL_MAX; s_dt2 = 0.0; s_dtdy = 0.0; open_count = 1U;
                }
            }
        }
        i++;
    }

    /* Commit register-cached state back to dig */
    const size_t done = i;
    dig->ring_head = ring_head;
    if (done > ring_cap - dig->ring_size) {
        dig->ring_evicted = true;
        dig->ring_size = ring_cap;
    } else {
        dig->ring_size += done;
    }
    dig->has_open = has_open;
    dig->open_t0 = open_t0;
    dig->open_y0 = open_y0;
    dig->open_t_last = open_t_last;
    dig->open_y_last = open_y_last;
    dig->m_min = m_min;
    dig->m_max = m_max;
    dig->s_dt2 = s_dt2;
    dig->s_dtdy = s_dtdy;
    dig->open_count = open_count;
    dig->has_any = has_any;
    dig->last_t = last_t;
    dig->total_points += (uint64_t)done;
    if (done > 0U) {
        dig->consec_anoms = 0U;
        if (out_wakes) {
            memset(out_wakes, 0, done);
        }
    }

    if (err != CS_OK) {
        return err;
    }
    if (last_out) {
        last_out->total_points = dig->total_points;
        last_out->total_segments = dig->total_segments;
        last_out->total_anomalies = dig->total_anomalies;
        last_out->wake_count = dig->wake_count;
        last_out->should_wake = false;
        last_out->segment_emitted = seg_emitted;
    }
    return CS_OK;
}

cs_status_t cs_digest_push_batch(
    cs_digest_t *dig,
    const double *ts,
    const double *ys,
    const double *y_hats,
    const double *qs,
    const double *edet_stats,
    const uint32_t *flags,
    size_t n,
    uint8_t *out_wakes,
    cs_digest_step_t *last_out)
{
    if (!dig || (n > 0 && (!ts || !ys))) {
        return CS_ERR_INVALID;
    }
    if (n == 0U) {
        if (last_out) {
            last_out->total_points = dig->total_points;
            last_out->total_segments = dig->total_segments;
            last_out->total_anomalies = dig->total_anomalies;
            last_out->wake_count = dig->wake_count;
            last_out->should_wake = false;
            last_out->segment_emitted = false;
        }
        return CS_OK;
    }
    if (!flags) {
        return cs_digest_push_batch_fast(dig, ts, ys, y_hats, qs, n, out_wakes, last_out);
    }

    cs_digest_step_t step;
    memset(&step, 0, sizeof(step));
    for (size_t i = 0; i < n; ++i) {
        double yh = y_hats ? y_hats[i] : ys[i];
        double q = qs ? qs[i] : NAN;
        double es = edet_stats ? edet_stats[i] : 0.0;
        uint32_t fl = flags[i];
        cs_status_t st = cs_digest_push(dig, ts[i], ys[i], yh, q, es, fl, &step);
        if (st != CS_OK) {
            return st;
        }
        if (out_wakes) {
            out_wakes[i] = step.should_wake ? 1U : 0U;
        }
    }
    if (last_out) {
        *last_out = step;
    }
    return CS_OK;
}

cs_status_t cs_digest_flush(cs_digest_t *dig) {
    if (!dig) {
        return CS_ERR_INVALID;
    }
    cs_digest_close_open(dig);
    return CS_OK;
}

static size_t cs_digest_seg_logical_start(const cs_digest_t *dig) {
    return (dig->seg_head + dig->seg_cap - dig->seg_size) % dig->seg_cap;
}

static cs_segment_t cs_digest_seg_at(const cs_digest_t *dig, size_t idx) {
    if (idx < dig->seg_size) {
        size_t pos = (cs_digest_seg_logical_start(dig) + idx) % dig->seg_cap;
        return dig->segs[pos];
    }
    return cs_digest_build_open_seg(dig);
}

static size_t cs_digest_total_live_segs(const cs_digest_t *dig) {
    return dig->seg_size + (dig->has_open ? 1U : 0U);
}

cs_status_t cs_digest_get_segments(
    const cs_digest_t *dig,
    cs_segment_t *out_segs,
    size_t max_out,
    size_t *out_n)
{
    if (!dig || !out_n) {
        return CS_ERR_INVALID;
    }
    size_t total = cs_digest_total_live_segs(dig);
    if (!out_segs || max_out == 0U) {
        *out_n = total;
        return CS_OK;
    }
    size_t take = (total < max_out) ? total : max_out;
    size_t start = total - take;
    for (size_t i = 0; i < take; ++i) {
        out_segs[i] = cs_digest_seg_at(dig, start + i);
    }
    *out_n = take;
    return CS_OK;
}

cs_status_t cs_digest_get_anomalies(
    const cs_digest_t *dig,
    cs_anomaly_t *out_anoms,
    size_t max_out,
    size_t *out_n)
{
    if (!dig || !out_n) {
        return CS_ERR_INVALID;
    }
    size_t total = dig->anom_size;
    if (!out_anoms || max_out == 0U) {
        *out_n = total;
        return CS_OK;
    }
    size_t take = (total < max_out) ? total : max_out;
    size_t log_start = (dig->anom_head + dig->anom_cap - dig->anom_size) % dig->anom_cap;
    size_t skip = total - take;
    for (size_t i = 0; i < take; ++i) {
        size_t pos = (log_start + skip + i) % dig->anom_cap;
        out_anoms[i] = dig->anoms[pos];
    }
    *out_n = take;
    return CS_OK;
}

cs_status_t cs_digest_zoom(
    const cs_digest_t *dig,
    double t0,
    double t1,
    cs_ring_record_t *out_records,
    size_t max_out,
    size_t *out_n,
    bool *out_truncated)
{
    if (!dig || !out_n || !isfinite(t0) || !isfinite(t1) || t0 > t1) {
        return CS_ERR_INVALID;
    }
    if (out_truncated) {
        *out_truncated = false;
    }
    if (dig->ring_size == 0U) {
        *out_n = 0U;
        return CS_OK;
    }

    size_t start = (dig->ring_head + dig->ring_cap - dig->ring_size) % dig->ring_cap;
    double oldest_t = dig->ring[start].t;
    if (out_truncated && dig->ring_evicted && t0 < oldest_t) {
        *out_truncated = true;
    }

    /* Binary search for first record with rec.t >= t0 */
    size_t lo = 0U;
    size_t hi = dig->ring_size;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2U;
        double tm = dig->ring[(start + mid) % dig->ring_cap].t;
        if (tm < t0) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }

    /* Count matching records in [t0, t1] */
    size_t end = lo;
    while (end < dig->ring_size && dig->ring[(start + end) % dig->ring_cap].t <= t1) {
        end++;
    }
    size_t match_n = end - lo;

    if (!out_records) {
        *out_n = match_n;
        return CS_OK;
    }

    size_t copy_n = (match_n <= max_out) ? match_n : max_out;
    for (size_t i = 0; i < copy_n; ++i) {
        out_records[i] = dig->ring[(start + lo + i) % dig->ring_cap];
    }
    *out_n = copy_n;
    return (match_n > max_out) ? CS_ERR_BUFFER : CS_OK;
}

cs_status_t cs_digest_reconstruct(
    const cs_digest_t *dig,
    const double *ts,
    size_t n,
    double *out_y)
{
    if (!dig || (n > 0 && (!ts || !out_y))) {
        return CS_ERR_INVALID;
    }
    size_t total = cs_digest_total_live_segs(dig);
    if (n == 0U) {
        return CS_OK;
    }
    if (total == 0U) {
        return CS_ERR_EMPTY;
    }

    for (size_t i = 0; i < n; ++i) {
        double t = ts[i];
        if (!isfinite(t)) {
            return CS_ERR_NONFINITE;
        }

        /* Binary search last segment with seg.t_start <= t */
        size_t lo = 0U;
        size_t hi = total;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2U;
            cs_segment_t sm = cs_digest_seg_at(dig, mid);
            if (sm.t_start <= t) {
                lo = mid + 1U;
            } else {
                hi = mid;
            }
        }

        if (lo == 0U) {
            cs_segment_t s0 = cs_digest_seg_at(dig, 0U);
            out_y[i] = s0.y_start + s0.slope * (t - s0.t_start);
        } else {
            size_t idx = lo - 1U;
            cs_segment_t s = cs_digest_seg_at(dig, idx);
            if (t <= s.t_end || idx + 1U >= total) {
                out_y[i] = s.y_start + s.slope * (t - s.t_start);
            } else {
                /* Timestamp falls in gap between two segments: linear bridge */
                cs_segment_t snext = cs_digest_seg_at(dig, idx + 1U);
                double dt = snext.t_start - s.t_end;
                if (dt > 0.0) {
                    double frac = (t - s.t_end) / dt;
                    out_y[i] = s.y_end + frac * (snext.y_start - s.y_end);
                } else {
                    out_y[i] = s.y_end;
                }
            }
        }
    }
    return CS_OK;
}

static void cs_sb_append(char *buf, size_t cap, size_t *pos, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *dst = (buf && *pos < cap) ? (buf + *pos) : NULL;
    size_t rem = (buf && *pos < cap) ? (cap - *pos) : 0U;
    int written = vsnprintf(dst, rem, fmt, ap);
    va_end(ap);
    if (written > 0) {
        *pos += (size_t)written;
    }
}

cs_status_t cs_digest_format_markdown(
    const cs_digest_t *dig,
    const char *channel_name,
    double emp_coverage,
    double current_q,
    double edet_stat,
    double edet_threshold,
    int32_t rogue_agent,
    size_t max_segments_show,
    size_t max_anomalies_show,
    char *buf,
    size_t buf_cap,
    size_t *out_len)
{
    if (!dig || !out_len) {
        return CS_ERR_INVALID;
    }
    const char *ch = (channel_name && channel_name[0] != '\0') ? channel_name : "stream";
    if (max_segments_show == 0U) {
        max_segments_show = 6U;
    }
    if (max_anomalies_show == 0U) {
        max_anomalies_show = 5U;
    }

    size_t live_segs = cs_digest_total_live_segs(dig);
    double comp_ratio = (dig->total_segments + (dig->has_open ? 1ULL : 0ULL) > 0ULL)
        ? ((double)dig->total_points / (double)(dig->total_segments + (dig->has_open ? 1ULL : 0ULL)))
        : 1.0;

    size_t pos = 0U;
    cs_sb_append(buf, buf_cap, &pos, "## ChronoSieve Digest: `%s`\n", ch);
    cs_sb_append(buf, buf_cap, &pos,
        "- pts=%llu segs=%zu (%.1fx PLA, L_inf<=%.4g) anoms=%llu wakes=%llu\n",
        (unsigned long long)dig->total_points,
        live_segs,
        comp_ratio,
        dig->cfg.epsilon,
        (unsigned long long)dig->total_anomalies,
        (unsigned long long)dig->wake_count);
    cs_sb_append(buf, buf_cap, &pos,
        "- E=%.2f/%.1f (ARL>=%.0f) cov=%.3f q=%.4g",
        edet_stat, edet_threshold, edet_threshold, emp_coverage, current_q);
    if (rogue_agent >= 0) {
        cs_sb_append(buf, buf_cap, &pos, " rogue=agent_%d\n", (int)rogue_agent);
    } else {
        cs_sb_append(buf, buf_cap, &pos, " rogue=none\n");
    }

    size_t seg_show = (live_segs < max_segments_show) ? live_segs : max_segments_show;
    double zoom_t0 = dig->last_t;
    double zoom_t1 = dig->last_t;
    if (seg_show > 0U) {
        size_t s_start = live_segs - seg_show;
        cs_segment_t first_s = cs_digest_seg_at(dig, s_start);
        zoom_t0 = first_s.t_start;
        cs_sb_append(buf, buf_cap, &pos, "### Trajectory (t0..t1: y0->y1, m, n)\n");
        for (size_t i = 0; i < seg_show; ++i) {
            cs_segment_t s = cs_digest_seg_at(dig, s_start + i);
            zoom_t1 = s.t_end;
            cs_sb_append(buf, buf_cap, &pos, "- %.2f..%.2f: %.6g->%.6g m=%+.4g n=%u%s\n",
                s.t_start, s.t_end, s.y_start, s.y_end, s.slope,
                (unsigned)s.count, (s.flags & CS_FLAG_PINNED) ? "*" : "");
        }
    }

    size_t anom_show = (dig->anom_size < max_anomalies_show) ? dig->anom_size : max_anomalies_show;
    if (anom_show > 0U) {
        size_t log_start = (dig->anom_head + dig->anom_cap - dig->anom_size) % dig->anom_cap;
        size_t skip = dig->anom_size - anom_show;
        cs_sb_append(buf, buf_cap, &pos, "### Anomalies (t, y, resid, q)\n");
        for (size_t i = 0; i < anom_show; ++i) {
            cs_anomaly_t a = dig->anoms[(log_start + skip + i) % dig->anom_cap];
            cs_sb_append(buf, buf_cap, &pos, "- t=%.2f y=%.4g r=%.4g q=%.4g\n",
                a.t, a.y, a.residual, a.q_t);
        }
    }

    cs_sb_append(buf, buf_cap, &pos, "Drill-down: `zoom(\"%s\", %.2f, %.2f)`\n",
        ch, zoom_t0, zoom_t1);

    *out_len = pos;
    if (buf && buf_cap > 0U && pos >= buf_cap) {
        return CS_ERR_BUFFER;
    }
    return CS_OK;
}

size_t cs_digest_segment_count(const cs_digest_t *dig) {
    return dig ? cs_digest_total_live_segs(dig) : 0U;
}

size_t cs_digest_anomaly_count(const cs_digest_t *dig) {
    return dig ? dig->anom_size : 0U;
}

size_t cs_digest_ring_size(const cs_digest_t *dig) {
    return dig ? dig->ring_size : 0U;
}

double cs_digest_epsilon(const cs_digest_t *dig) {
    return dig ? dig->cfg.epsilon : 0.0;
}
