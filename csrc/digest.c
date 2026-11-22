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

static void cs_digest_append_seg(cs_digest_t *dig, const cs_segment_t *seg) {
    dig->segs[dig->seg_head] = *seg;
    dig->seg_head = (dig->seg_head + 1U) % dig->seg_cap;
    if (dig->seg_size < dig->seg_cap) {
        dig->seg_size++;
    }
    dig->total_segments++;
}

static void cs_digest_append_anom(cs_digest_t *dig, const cs_anomaly_t *anom) {
    dig->anoms[dig->anom_head] = *anom;
    dig->anom_head = (dig->anom_head + 1U) % dig->anom_cap;
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
    dig->ring_head = (dig->ring_head + 1U) % dig->ring_cap;
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
    cs_digest_step_t step;
    memset(&step, 0, sizeof(step));
    for (size_t i = 0; i < n; ++i) {
        double yh = y_hats ? y_hats[i] : ys[i];
        double q = qs ? qs[i] : NAN;
        double es = edet_stats ? edet_stats[i] : 0.0;
        uint32_t fl = flags ? flags[i] : 0U;
        cs_status_t st = cs_digest_push(dig, ts[i], ys[i], yh, q, es, fl, &step);
        if (st != CS_OK) {
            return st;
        }
        if (out_wakes) {
            out_wakes[i] = step.should_wake ? 1U : 0U;
        }
    }
    if (last_out) {
        if (n == 0) {
            step.total_points = dig->total_points;
            step.total_segments = dig->total_segments;
            step.total_anomalies = dig->total_anomalies;
            step.wake_count = dig->wake_count;
            step.should_wake = false;
            step.segment_emitted = false;
        }
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
