#ifndef CHRONOSIEVE_H
#define CHRONOSIEVE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version, Status Codes, and Runtime Introspection
 * ========================================================================= */

#define CS_VERSION_MAJOR 0
#define CS_VERSION_MINOR 1
#define CS_VERSION_PATCH 0

#define CS_EDET_LANES 16U

typedef enum {
    CS_OK = 0,
    CS_ERR_INVALID = -1,
    CS_ERR_NOMEM = -2,
    CS_ERR_NONFINITE = -3,
    CS_ERR_EMPTY = -4,
    CS_ERR_BUFFER = -5
} cs_status_t;

const char *cs_version_string(void);
const char *cs_simd_backend_name(void);

/* =========================================================================
 * Module 1: Zero-exp SIMD E-Detectors & Cross-Agent E-Fusion
 * ========================================================================= */

typedef enum {
    CS_FAMILY_BOUNDED = 0,   /* Z_t in [0, 1], known null mean mu0 */
    CS_FAMILY_CATONI = 1,    /* Wang-Ramdas algebraic Catoni, known (mu0, sigma) */
    CS_FAMILY_SYMMETRY = 2,  /* Odd algebraic symmetry, known mu0, unknown var (even Cauchy) */
    CS_FAMILY_VOVK_RANK = 3, /* Growing-prefix smoothed conformal rank B-tree, unknown (mu, sigma) */
    CS_FAMILY_AUTO = 4       /* Catoni if mu0 & sigma finite, else Vovk Rank */
} cs_edet_family_t;

typedef enum {
    CS_ACCUM_ESR = 0,    /* Shiryaev-Roberts: R_t^(m) = (R_{t-1}^(m) + 1) * e_t^(m), zero-exp */
    CS_ACCUM_ECUSUM = 1  /* Log-domain E-CUSUM, mixture (1/M) sum exp(S_t^(m)) */
} cs_edet_accum_t;

typedef enum {
    CS_SIDE_TWO_SIDED = 0, /* Detect both upward and downward shifts */
    CS_SIDE_UPPER = 1,     /* Detect upward shifts (X_t increases) */
    CS_SIDE_LOWER = 2      /* Detect downward shifts (X_t decreases) */
} cs_edet_side_t;

typedef struct {
    cs_edet_family_t family;
    cs_edet_accum_t accum;
    cs_edet_side_t side;
    double threshold;      /* Target ARL lower bound gamma (> 1.0, default 1000.0) */
    double mu0;            /* Null mean (NaN -> unknown, routes AUTO to Vovk Rank) */
    double sigma;          /* Null std dev for Catoni, or scale hint for Symmetry (NaN -> unknown) */
    uint32_t max_prefix;   /* Deterministic B-tree epoch reset for Vovk Rank (default 65536) */
    uint64_t seed;         /* Deterministic PRNG seed for Vovk Rank tie randomization */
    bool auto_reset;       /* If true, reset detector state automatically after alarm */
} cs_edet_config_t;

typedef struct {
    double stat;           /* Valid mixture statistic R_t or C_t */
    double log_stat;       /* log(max(stat, 1e-300)) for numerical inspection */
    double last_e_mean;    /* Mean e-value across lanes on this step */
    double last_p_value;   /* Smoothed conformal p-value (Vovk Rank only, else NaN) */
    uint64_t step_count;   /* Total steps observed */
    uint64_t alarm_count;  /* Cumulative certified alarms */
    uint32_t epoch_count;  /* Vovk Rank deterministic epoch rollovers */
    bool alarmed;          /* True if stat >= threshold on this tick */
} cs_edet_result_t;

typedef struct cs_edetector cs_edetector_t;

void cs_edet_config_default(cs_edet_config_t *cfg);
cs_edetector_t *cs_edet_create(const cs_edet_config_t *cfg);
void cs_edet_destroy(cs_edetector_t *det);
void cs_edet_reset(cs_edetector_t *det);
/* Reset only accumulator lanes (E-SR / E-CUSUM) while keeping the Vovk Rank B-tree baseline. */
void cs_edet_reset_accum_only(cs_edetector_t *det);

/* Calibrate a Catoni detector from a finite warmup sample with conservative variance inflation. */
cs_status_t cs_edet_calibrate_catoni(
    const double *warmup,
    size_t n,
    double threshold,
    cs_edet_side_t side,
    double inflation_z,
    cs_edet_config_t *out_cfg);

cs_status_t cs_edet_step(cs_edetector_t *det, double x, cs_edet_result_t *out);
cs_status_t cs_edet_step_batch(
    cs_edetector_t *det,
    const double *xs,
    size_t n,
    double *out_stats,      /* optional, length n */
    uint8_t *out_alarms,    /* optional, length n */
    cs_edet_result_t *last_out /* optional */);

/* Compute the 16-lane e-value vector for x without advancing accumulator (used by E-Fusion). */
cs_status_t cs_edet_eval_lanes(cs_edetector_t *det, double x, double out_e[CS_EDET_LANES]);

cs_edet_family_t cs_edet_effective_family(const cs_edetector_t *det);
double cs_edet_current_stat(const cs_edetector_t *det);
uint64_t cs_edet_step_count(const cs_edetector_t *det);

/* Cross-Agent E-Fusion (valid under arbitrary unknown cross-agent correlation) */
typedef enum {
    CS_FUSION_STEP = 0,    /* F_{t-1}-predictable convex combination e_bar_t = sum w_{t,k} e_{t,k} */
    CS_FUSION_PROCESS = 1  /* Process-level convex average R_bar_t = sum w_k R_t^(k) (rank-safe) */
} cs_efusion_mode_t;

typedef struct {
    size_t n_agents;
    cs_efusion_mode_t mode;
    cs_edet_accum_t accum;
    double threshold;
    bool auto_reset;
    const cs_edet_config_t *agent_configs; /* length n_agents, or NULL for default Catoni(0,1) */
    const double *weights;                 /* optional fixed prior weights (length n_agents) */
} cs_efusion_config_t;

typedef struct cs_efusion cs_efusion_t;

cs_efusion_t *cs_efusion_create(const cs_efusion_config_t *cfg);
void cs_efusion_destroy(cs_efusion_t *fus);
void cs_efusion_reset(cs_efusion_t *fus);

/* Step K agents simultaneously.
 * Optional F_{t-1}-predictable weights (length K, sum > 0) override priors ONLY in
 * CS_FUSION_STEP mode. Passing non-NULL predictable_weights in CS_FUSION_PROCESS mode
 * returns CS_ERR_INVALID because process-level averaging requires fixed prior weights
 * to guarantee ARL >= gamma. */
cs_status_t cs_efusion_step(
    cs_efusion_t *fus,
    const double *x_agents,
    const double *predictable_weights, /* NULL -> use configured weights */
    cs_edet_result_t *out);

cs_status_t cs_efusion_step_batch(
    cs_efusion_t *fus,
    const double *x_matrix,  /* row-major [n_steps x n_agents] */
    size_t n_steps,
    double *out_stats,       /* optional, length n_steps */
    uint8_t *out_alarms,     /* optional, length n_steps */
    cs_edet_result_t *last_out);

/* =========================================================================
 * Module 2: Pre-Control E-Reset Conformal PID + Multi-Agent COMA
 * ========================================================================= */

typedef struct {
    double alpha;             /* Target miscoverage rate in (0, 1), default 0.10 */
    double eta;               /* Fixed P-term step size (> 0), NEVER boosted */
    double ki;                /* Saturating tangent I-term gain (>= 0), default 1.0 */
    double c_sat;             /* Tangent saturation scale (> 0), default 1.0 */
    double score_bound;       /* Upper bound B > 0 for bounded D-term scorecaster q_D in [0, B] */
    double init_q;            /* Initial interval radius q_0 >= 0 */
    double edet_threshold;    /* Pre-control E-detector ARL threshold gamma (default 500.0) */
    uint32_t warmup_cal;      /* Mandatory post-reset calibration warmup W_cal >= 32 (default 32) */
    uint32_t score_window;    /* Rolling window for empirical (1-alpha) scorecaster (default 128) */
    uint32_t edet_max_prefix; /* Max B-tree prefix for pre-control Vovk Rank (default 65536) */
    uint64_t seed;            /* Deterministic PRNG seed */
    bool enable_ereset;       /* Enable pre-control E-reset of bounded D-term (default true) */
} cs_conformal_config_t;

typedef struct {
    double q_used;            /* Radius q_t applied to this tick */
    double q_next;            /* Updated radius q_{t+1} for the next tick */
    double lower;             /* y_hat - q_used */
    double upper;             /* y_hat + q_used */
    double score;             /* Pre-control nonconformity score s_t = |y_t - y_hat_t| */
    double q_p;               /* P-term component eta * E_t */
    double q_i;               /* Saturating tangent I-term component */
    double q_d;               /* Bounded D-term scorecaster in [0, B] */
    double emp_coverage;      /* Running empirical coverage 1 - (1/t) sum err_i */
    double edet_stat;         /* Pre-control E-detector wealth on s_t */
    uint64_t step_count;
    uint64_t reset_count;
    bool miscovered;          /* True if s_t > q_used */
    bool shift_reset;         /* True if pre-control E-detector triggered D-term reset */
    bool calibrating;         /* True if currently in mandatory W_cal warmup */
} cs_conformal_result_t;

typedef struct cs_conformal_pid cs_conformal_pid_t;

void cs_conformal_config_default(cs_conformal_config_t *cfg);
cs_conformal_pid_t *cs_conformal_create(const cs_conformal_config_t *cfg);
void cs_conformal_destroy(cs_conformal_pid_t *cp);
void cs_conformal_reset(cs_conformal_pid_t *cp);

/* Step with (y_true, y_pred); computes pre-control score s_t = |y_true - y_pred|. */
cs_status_t cs_conformal_step(
    cs_conformal_pid_t *cp,
    double y_true,
    double y_pred,
    cs_conformal_result_t *out);

/* Batch step over arrays of length n. Output pointers are optional. */
cs_status_t cs_conformal_step_batch(
    cs_conformal_pid_t *cp,
    const double *y_true,
    const double *y_pred,
    size_t n,
    double *out_q_used,
    double *out_q_next,
    uint8_t *out_miscov,
    uint8_t *out_resets,
    cs_conformal_result_t *last_out);

double cs_conformal_current_q(const cs_conformal_pid_t *cp);

/* Multi-channel SoA SIMD Conformal PID batch kernel (processes C channels in NEON/AVX2) */
typedef struct cs_conformal_soa cs_conformal_soa_t;

cs_conformal_soa_t *cs_conformal_soa_create(size_t n_channels, const cs_conformal_config_t *cfg);
void cs_conformal_soa_destroy(cs_conformal_soa_t *soa);
void cs_conformal_soa_reset(cs_conformal_soa_t *soa);

/* Step all C channels simultaneously for one timestamp. */
cs_status_t cs_conformal_soa_step(
    cs_conformal_soa_t *soa,
    const double *y_true,   /* length C */
    const double *y_pred,   /* length C */
    double *out_q_used,     /* optional, length C */
    double *out_q_next,     /* optional, length C */
    uint8_t *out_miscov     /* optional, length C */);

/* Step T timestamps across C channels (row-major [T x C]). */
cs_status_t cs_conformal_soa_step_batch(
    cs_conformal_soa_t *soa,
    const double *y_true,   /* [T x C] */
    const double *y_pred,   /* [T x C] */
    size_t n_steps,
    double *out_q_next,     /* optional, [T x C] */
    uint8_t *out_miscov     /* optional, [T x C] */);

/* Multi-Agent Online COMA (Gasparin & Ramdas, UAI 2024) */
typedef struct {
    double lo;
    double hi;
} cs_interval_t;

typedef struct {
    size_t n_intervals;       /* Number of disjoint intervals in exact majority-vote union */
    double hull_lo;           /* Convex hull lower bound (NaN if empty) */
    double hull_hi;           /* Convex hull upper bound (NaN if empty) */
    double max_agreement;     /* Maximum total weight covering any point y */
    double used_tau;          /* Threshold tau applied (or sampled U in [0.5, 1]) */
} cs_coma_result_t;

/* Compute exact O(K log K) COMA consensus intervals.
 * out_intervals must have capacity >= K (at most K disjoint intervals can result).
 * If randomized == true, draws U ~ Uniform[0.5, 1] using *rng_state (updated in place). */
cs_status_t cs_coma_merge(
    const double *lowers,
    const double *uppers,
    const double *weights,    /* optional: NULL -> equal weights 1/K */
    size_t k,
    double tau,               /* deterministic threshold in [0.5, 1.0], ignored if randomized */
    bool randomized,
    uint64_t *rng_state,      /* required if randomized == true */
    cs_interval_t *out_intervals,
    size_t max_intervals,
    cs_coma_result_t *out_res);

/* =========================================================================
 * Module 3: NEON Sleeping Fixed-Share BOA + Per-Agent Drift Attribution
 * ========================================================================= */

typedef struct {
    size_t n_agents;
    double share_base;        /* Stationary fixed-share mixing gamma_base (default 0.01) */
    double share_shift;       /* Shift-coupled mixing boost gamma_shift (default 0.35) */
    double shift_var_decay;   /* Decay multiplier on V_{t,k} on shift (default 0.25) */
    double rogue_threshold;   /* Per-agent Catoni ARL threshold for drift attribution (default 200.0) */
    double rogue_mu0;         /* Known baseline abs residual mean (NaN -> auto-calibrate) */
    double rogue_sigma;       /* Known baseline abs residual std (NaN -> auto-calibrate) */
    uint32_t cal_window;      /* Per-agent warmup ticks for auto-calibration (default 48) */
    bool use_abs_loss;        /* false -> squared loss (default), true -> absolute loss */
    bool quarantine_rogue;    /* If true, zero out active weight of flagged rogue agents if peers exist */
} cs_fuser_config_t;

typedef struct {
    double prediction;        /* Fused prediction y_hat_t */
    double fused_loss;        /* Loss of fused prediction against y_t */
    double max_rogue_stat;    /* Maximum per-agent Catoni wealth across active agents */
    int32_t worst_agent;      /* Index of agent with highest Catoni drift stat (-1 if none) */
    size_t active_count;      /* Number of awake agents on this tick */
    size_t rogue_count;       /* Number of agents currently flagged rogue */
    uint64_t step_count;
    bool any_rogue;           /* True if any agent triggered Catoni drift attribution */
} cs_fuser_result_t;

typedef struct cs_fuser cs_fuser_t;

void cs_fuser_config_default(cs_fuser_config_t *cfg, size_t n_agents);
cs_fuser_t *cs_fuser_create(const cs_fuser_config_t *cfg);
void cs_fuser_destroy(cs_fuser_t *fus);
void cs_fuser_reset(cs_fuser_t *fus);
void cs_fuser_clear_rogue(cs_fuser_t *fus, int32_t agent_idx); /* -1 clears all */

/* Predict only (before y_true arrives). active == NULL means all agents active. */
cs_status_t cs_fuser_predict(
    const cs_fuser_t *fus,
    const double *preds,
    const uint8_t *active,
    double *out_pred);

/* Predict + update weights + per-agent Catoni drift attribution. */
cs_status_t cs_fuser_step(
    cs_fuser_t *fus,
    const double *preds,      /* length K */
    const uint8_t *active,    /* optional length K (1=awake, 0=sleeping); NULL -> all awake */
    double y_true,
    bool shift_detected,      /* external E-detector shift signal for fixed-share boost */
    cs_fuser_result_t *out);

cs_status_t cs_fuser_step_batch(
    cs_fuser_t *fus,
    const double *preds_mat,  /* row-major [T x K] */
    const uint8_t *active_mat,/* optional row-major [T x K], NULL -> all awake */
    const double *y_true,     /* length T */
    const uint8_t *shifts,    /* optional length T, NULL -> no shifts */
    size_t n_steps,
    double *out_preds,        /* optional length T */
    double *out_weights_final,/* optional length K */
    uint8_t *out_rogue_any,   /* optional length T */
    cs_fuser_result_t *last_out);

cs_status_t cs_fuser_get_weights(const cs_fuser_t *fus, double *out_w, size_t k);
cs_status_t cs_fuser_get_rogue_stats(
    const cs_fuser_t *fus,
    double *out_stats,        /* optional length K */
    uint8_t *out_flags,       /* optional length K */
    size_t k);

/* =========================================================================
 * Module 4: Anomaly-Preserving Swing PLA + Wake-Up Gate + Ring-Buffer zoom()
 * ========================================================================= */

#define CS_FLAG_ANOMALY    (1U << 0)
#define CS_FLAG_SHIFT      (1U << 1)
#define CS_FLAG_ROGUE      (1U << 2)
#define CS_FLAG_PINNED     (1U << 3)

typedef struct {
    double t_start;
    double y_start;
    double t_end;
    double y_end;
    double slope;
    uint32_t count;
    uint32_t flags;           /* CS_FLAG_PINNED if verbatim anomaly point */
} cs_segment_t;

typedef struct {
    double t;
    double y;
    double y_hat;
    double residual;
    double q_t;
    double edet_stat;
    uint32_t flags;
} cs_anomaly_t;

typedef struct {
    double t;
    double y;
    double y_hat;
    double q_t;
    uint32_t flags;
} cs_ring_record_t;

typedef struct {
    double epsilon;           /* Guaranteed L_inf error bound >= 0 (default 0.05) */
    size_t ring_capacity;     /* Circular ring buffer capacity for lossless zoom() (default 65536) */
    size_t max_segments;      /* Retained Swing PLA segment ring capacity (default 4096) */
    size_t max_anomalies;     /* Retained verbatim anomaly ring capacity (default 1024) */
    uint32_t wake_cooldown;   /* Minimum ticks between non-shift anomaly wakeups (default 0) */
    uint32_t anomaly_burst;   /* Consecutive/recent anomalies to wake if no shift (default 1) */
} cs_digest_config_t;

typedef struct {
    uint64_t total_points;
    uint64_t total_segments;
    uint64_t total_anomalies;
    uint64_t wake_count;
    bool should_wake;         /* System-1 -> System-2 wake-up gate decision */
    bool segment_emitted;     /* True if a Swing PLA segment closed on this tick */
} cs_digest_step_t;

typedef struct cs_digest cs_digest_t;

void cs_digest_config_default(cs_digest_config_t *cfg);
cs_digest_t *cs_digest_create(const cs_digest_config_t *cfg);
void cs_digest_destroy(cs_digest_t *dig);
void cs_digest_reset(cs_digest_t *dig);

/* Push a single tick into Swing PLA + ring buffer + wake-up gate. */
cs_status_t cs_digest_push(
    cs_digest_t *dig,
    double t,
    double y,
    double y_hat,
    double q_t,
    double edet_stat,
    uint32_t flags,
    cs_digest_step_t *out);

/* Batch push ticks (zero-copy friendly). */
cs_status_t cs_digest_push_batch(
    cs_digest_t *dig,
    const double *ts,
    const double *ys,
    const double *y_hats,     /* optional, NULL -> uses ys (zero residual) */
    const double *qs,         /* optional, NULL -> NaN */
    const double *edet_stats, /* optional, NULL -> 0.0 */
    const uint32_t *flags,    /* optional, NULL -> 0 */
    size_t n,
    uint8_t *out_wakes,       /* optional, length n */
    cs_digest_step_t *last_out);

/* Flush any currently open Swing PLA segment so all pushed points are materialized. */
cs_status_t cs_digest_flush(cs_digest_t *dig);

/* Copy up to max_out most recent segments (chronological order). Returns count via *out_n. */
cs_status_t cs_digest_get_segments(
    const cs_digest_t *dig,
    cs_segment_t *out_segs,
    size_t max_out,
    size_t *out_n);

/* Copy up to max_out most recent anomalies (chronological order). Returns count via *out_n. */
cs_status_t cs_digest_get_anomalies(
    const cs_digest_t *dig,
    cs_anomaly_t *out_anoms,
    size_t max_out,
    size_t *out_n);

/* Lossless O(log N) zoom over [t0, t1] from circular ring buffer.
 * Sets *out_truncated = true if t0 precedes the oldest retained record. */
cs_status_t cs_digest_zoom(
    const cs_digest_t *dig,
    double t0,
    double t1,
    cs_ring_record_t *out_records,
    size_t max_out,
    size_t *out_n,
    bool *out_truncated);

/* Reconstruct y_hat(t) from closed + open Swing PLA segments (for verification / plotting). */
cs_status_t cs_digest_reconstruct(
    const cs_digest_t *dig,
    const double *ts,
    size_t n,
    double *out_y);

/* Format a compact ~100-250 token Markdown wake-up packet for System-2 LLM ingestion.
 * Returns required byte length (excluding NUL) via *out_len. */
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
    size_t *out_len);

size_t cs_digest_segment_count(const cs_digest_t *dig);
size_t cs_digest_anomaly_count(const cs_digest_t *dig);
size_t cs_digest_ring_size(const cs_digest_t *dig);
double cs_digest_epsilon(const cs_digest_t *dig);

#ifdef __cplusplus
}
#endif

#endif /* CHRONOSIEVE_H */
