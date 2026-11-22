#include "chronosieve.h"
#include "cs_internal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void test_simd_math_accuracy(void) {
    /* Verify Estrin/Cephes exp accuracy < 1e-12 across [-700, 709] */
    double max_exp_rel = 0.0;
    for (int i = 0; i <= 20000; ++i) {
        double x = -700.0 + (1409.0 * (double)i) / 20000.0;
        double ref = exp(x);
        double got = cs_scalar_exp(x);
#if defined(CS_SIMD_NEON)
        double vec_in[2] = {x, -x * 0.5};
        float64x2_t vr = cs_vexp_f64(vld1q_f64(vec_in));
        double got_v = vgetq_lane_f64(vr, 0);
        CHECK(fabs(got_v - got) <= 1.0e-12 * fmax(1.0, fabs(ref)));
        got = got_v;
#elif defined(CS_SIMD_AVX2)
        double vec_in[4] = {x, -x * 0.5, 0.0, 1.0};
        __m256d vr = cs_vexp_f64(_mm256_loadu_pd(vec_in));
        double vec_out[4];
        _mm256_storeu_pd(vec_out, vr);
        double got_v = vec_out[0];
        CHECK(fabs(got_v - got) <= 1.0e-12 * fmax(1.0, fabs(ref)));
        got = got_v;
#endif
        double rel = fabs(got - ref) / fmax(ref, 1.0e-300);
        if (rel > max_exp_rel) max_exp_rel = rel;
    }
    CHECK(max_exp_rel < 1.0e-12);
    CHECK(cs_scalar_exp(-800.0) == 0.0);
    CHECK(isinf(cs_scalar_exp(800.0)));

    /* Verify rational tan_clamped accuracy < 1e-12 across full clamped domain */
    double max_tan_rel = 0.0;
    for (int i = 0; i <= 20000; ++i) {
        double th = -CS_TAN_MAX_ARG + (2.0 * CS_TAN_MAX_ARG * (double)i) / 20000.0;
        double ref = tan(th);
        double got = cs_scalar_tan_clamped(th);
#if defined(CS_SIMD_NEON)
        double vec_in[2] = {th, -th};
        float64x2_t vr = cs_vtan_clamped_f64(vld1q_f64(vec_in));
        double got_v = vgetq_lane_f64(vr, 0);
        CHECK(fabs(got_v - got) <= 1.0e-12 * fmax(1.0, fabs(ref)));
        got = got_v;
#elif defined(CS_SIMD_AVX2)
        double vec_in[4] = {th, -th, 0.0, 0.5};
        __m256d vr = cs_vtan_clamped_f64(_mm256_loadu_pd(vec_in));
        double vec_out[4];
        _mm256_storeu_pd(vec_out, vr);
        double got_v = vec_out[0];
        CHECK(fabs(got_v - got) <= 1.0e-12 * fmax(1.0, fabs(ref)));
        got = got_v;
#endif
        double rel = fabs(got - ref) / fmax(1.0, fabs(ref));
        if (rel > max_tan_rel) max_tan_rel = rel;
    }
    CHECK(max_tan_rel < 1.0e-12);
    printf("[PASS] simd_math_accuracy (%s): max_exp_rel=%.3e max_tan_rel=%.3e\n",
           cs_simd_backend_name(), max_exp_rel, max_tan_rel);
}

static void test_btree_exact_order_stats(void) {
    cs_btree_t bt;
    cs_btree_init(&bt);
    const size_t N = 2500U;
    double *seen = (double *)malloc(N * sizeof(double));
    CHECK(seen != NULL);

    cs_rng_t rng;
    cs_rng_seed(&rng, 42ULL);

    for (size_t i = 0; i < N; ++i) {
        double x;
        if (i < 500U) {
            /* Heavy discrete ties */
            x = (double)((int)(cs_rng_u64(&rng) % 25ULL) - 12);
        } else if (i < 1000U) {
            /* Strictly increasing */
            x = (double)i * 0.1;
        } else if (i < 1500U) {
            /* Strictly decreasing */
            x = -(double)i * 0.1;
        } else {
            /* Continuous uniform */
            x = cs_rng_uniform01(&rng) * 100.0 - 50.0;
        }

        uint32_t brute_less = 0, brute_eq = 0;
        for (size_t j = 0; j < i; ++j) {
            if (seen[j] < x) brute_less++;
            else if (seen[j] == x) brute_eq++;
        }

        uint32_t bt_less = 0, bt_eq = 0;
        cs_btree_rank_query(&bt, x, &bt_less, &bt_eq);
        CHECK(bt_less == brute_less);
        CHECK(bt_eq == brute_eq);

        CHECK(cs_btree_insert(&bt, x) == CS_OK);
        seen[i] = x;
        CHECK(bt.total_count == (uint32_t)(i + 1U));
        CHECK(bt.nodes[bt.root].subtree_size == (uint32_t)(i + 1U));
    }

    /* Test O(1) epoch clear & fused cs_btree_rank_and_insert exactness */
    cs_btree_clear(&bt);
    CHECK(bt.total_count == 0U);
    uint32_t l = 99, e = 99;
    cs_btree_rank_query(&bt, 1.0, &l, &e);
    CHECK(l == 0U && e == 0U);
    CHECK(cs_btree_insert(&bt, 3.14) == CS_OK);
    cs_btree_rank_query(&bt, 3.14, &l, &e);
    CHECK(l == 0U && e == 1U);

    cs_btree_clear(&bt);
    cs_rng_seed(&rng, 98765ULL);
    for (size_t i = 0; i < N; ++i) {
        double x;
        if (i < 600U) {
            x = (double)((int)(cs_rng_u64(&rng) % 19ULL) - 9);
        } else if (i < 1200U) {
            x = (double)i * 0.25;
        } else if (i < 1800U) {
            x = -(double)i * 0.25;
        } else {
            x = cs_rng_uniform01(&rng) * 200.0 - 100.0;
        }
        uint32_t brute_less = 0, brute_eq = 0;
        for (size_t j = 0; j < i; ++j) {
            if (seen[j] < x) brute_less++;
            else if (seen[j] == x) brute_eq++;
        }
        uint32_t bt_less = 0, bt_eq = 0;
        CHECK(cs_btree_rank_and_insert(&bt, x, &bt_less, &bt_eq) == CS_OK);
        CHECK(bt_less == brute_less);
        CHECK(bt_eq == brute_eq);
        seen[i] = x;
        CHECK(bt.total_count == (uint32_t)(i + 1U));
        CHECK(bt.nodes[bt.root].subtree_size == (uint32_t)(i + 1U));
    }

    cs_btree_free(&bt);
    free(seen);
    printf("[PASS] btree_exact_order_stats (N=%zu, separate+fused)\n", N);
}

static void test_edetectors_and_efusion(void) {
    /* 1. AUTO routes to Vovk Rank when mu0/sigma are NaN, Catoni when finite */
    cs_edet_config_t cfg;
    cs_edet_config_default(&cfg);
    cs_edetector_t *d_auto1 = cs_edet_create(&cfg);
    CHECK(d_auto1 != NULL);
    CHECK(cs_edet_effective_family(d_auto1) == CS_FAMILY_VOVK_RANK);
    cs_edet_destroy(d_auto1);

    cfg.mu0 = 0.0;
    cfg.sigma = 1.0;
    cs_edetector_t *d_auto2 = cs_edet_create(&cfg);
    CHECK(d_auto2 != NULL);
    CHECK(cs_edet_effective_family(d_auto2) == CS_FAMILY_CATONI);
    /* Non-finite rejection */
    cs_edet_result_t r;
    CHECK(cs_edet_step(d_auto2, NAN, &r) == CS_ERR_NONFINITE);
    CHECK(cs_edet_step(d_auto2, INFINITY, &r) == CS_ERR_NONFINITE);
    /* Empty batch succeeds */
    CHECK(cs_edet_step_batch(d_auto2, NULL, 0U, NULL, NULL, &r) == CS_OK);
    /* Shift detection */
    bool detected = false;
    for (int i = 0; i < 60; ++i) {
        CHECK(cs_edet_step(d_auto2, 2.5, &r) == CS_OK);
        if (r.alarmed) { detected = true; break; }
    }
    CHECK(detected);
    cs_edet_destroy(d_auto2);

    /* 2. Bounded + E-CUSUM */
    cs_edet_config_default(&cfg);
    cfg.family = CS_FAMILY_BOUNDED;
    cfg.accum = CS_ACCUM_ECUSUM;
    cfg.mu0 = 0.2;
    cfg.threshold = 100.0;
    cs_edetector_t *d_bnd = cs_edet_create(&cfg);
    CHECK(d_bnd != NULL);
    detected = false;
    for (int i = 0; i < 60; ++i) {
        CHECK(cs_edet_step(d_bnd, 0.85, &r) == CS_OK);
        if (r.alarmed) { detected = true; break; }
    }
    CHECK(detected);
    cs_edet_destroy(d_bnd);

    /* 3. Symmetry detects shift */
    cs_edet_config_default(&cfg);
    cfg.family = CS_FAMILY_SYMMETRY;
    cfg.mu0 = 0.0;
    cfg.threshold = 100.0;
    cs_edetector_t *d_sym = cs_edet_create(&cfg);
    CHECK(d_sym != NULL);
    detected = false;
    for (int i = 0; i < 60; ++i) {
        CHECK(cs_edet_step(d_sym, 3.0, &r) == CS_OK);
        if (r.alarmed) { detected = true; break; }
    }
    CHECK(detected);
    cs_edet_destroy(d_sym);

    /* 4. Vovk Rank deterministic epoch rollover */
    cs_edet_config_default(&cfg);
    cfg.family = CS_FAMILY_VOVK_RANK;
    cfg.max_prefix = 64U;
    cfg.threshold = 1.0e9;
    cs_edetector_t *d_rk = cs_edet_create(&cfg);
    CHECK(d_rk != NULL);
    cs_rng_t rng;
    cs_rng_seed(&rng, 777ULL);
    for (int i = 0; i < 200; ++i) {
        CHECK(cs_edet_step(d_rk, cs_rng_uniform01(&rng), &r) == CS_OK);
        CHECK(r.last_p_value >= 0.0 && r.last_p_value <= 1.0);
    }
    CHECK(r.epoch_count == 3U);
    cs_edet_destroy(d_rk);

    /* 5. Cross-Agent E-Fusion (STEP and PROCESS) */
    for (int mode = 0; mode < 2; ++mode) {
        cs_efusion_config_t fcfg;
        memset(&fcfg, 0, sizeof(fcfg));
        fcfg.n_agents = 4U;
        fcfg.mode = (mode == 0) ? CS_FUSION_STEP : CS_FUSION_PROCESS;
        fcfg.accum = CS_ACCUM_ESR;
        fcfg.threshold = 100.0;
        cs_efusion_t *fus = cs_efusion_create(&fcfg);
        CHECK(fus != NULL);
        double x[4] = {2.0, 2.1, 1.9, 2.2};
        detected = false;
        for (int i = 0; i < 60; ++i) {
            CHECK(cs_efusion_step(fus, x, NULL, &r) == CS_OK);
            if (r.alarmed) { detected = true; break; }
        }
        CHECK(detected);
        if (mode == 1) {
            /* Process-level fusion MUST reject time-varying predictable_weights */
            double pw[4] = {1.0, 0.0, 0.0, 0.0};
            CHECK(cs_efusion_step(fus, x, pw, &r) == CS_ERR_INVALID);
        }
        cs_efusion_destroy(fus);
    }

    /* 6. Extreme finite scales (sigma=1e200 / 1e-200, x=1e308) must never produce NaN */
    {
        double sigmas[2] = {1.0e200, 1.0e-200};
        for (int s = 0; s < 2; ++s) {
            cs_edet_config_default(&cfg);
            cfg.family = CS_FAMILY_CATONI;
            cfg.mu0 = 0.0;
            cfg.sigma = sigmas[s];
            cfg.threshold = 100.0;
            cs_edetector_t *dc = cs_edet_create(&cfg);
            CHECK(dc != NULL);
            CHECK(cs_edet_step(dc, sigmas[s], &r) == CS_OK);
            CHECK(isfinite(r.stat) && isfinite(r.last_e_mean));
            cs_edet_destroy(dc);
        }

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_SYMMETRY;
        cfg.mu0 = 0.0;
        cfg.threshold = 100.0;
        cs_edetector_t *ds = cs_edet_create(&cfg);
        CHECK(ds != NULL);
        CHECK(cs_edet_step(ds, 1.0e308, &r) == CS_OK);
        CHECK(isfinite(r.stat) && isfinite(r.last_e_mean));
        CHECK(cs_edet_step(ds, 10.0, &r) == CS_OK);
        CHECK(isfinite(r.stat));
        cs_edet_destroy(ds);
    }

    printf("[PASS] edetectors_and_efusion\n");
}

static void test_conformal_and_coma(void) {
    cs_conformal_config_t cfg;
    cs_conformal_config_default(&cfg);
    cfg.warmup_cal = 5U; /* Must be enforced to >= 32! */
    cfg.score_bound = 50.0;
    cfg.edet_threshold = 100.0;
    cs_conformal_pid_t *cp = cs_conformal_create(&cfg);
    CHECK(cp != NULL);

    cs_conformal_result_t res;
    CHECK(cs_conformal_step(cp, NAN, 0.0, &res) == CS_ERR_NONFINITE);
    /* First 32 ticks must be calibrating */
    for (int i = 0; i < 32; ++i) {
        CHECK(cs_conformal_step(cp, 1.0, 0.0, &res) == CS_OK);
        CHECK(res.calibrating == true);
        CHECK(res.q_d >= 0.0 && res.q_d <= cfg.score_bound);
    }
    /* Tick 33 should be armed (not calibrating) */
    CHECK(cs_conformal_step(cp, 1.0, 0.0, &res) == CS_OK);
    CHECK(res.calibrating == false);
    /* Finite-input score overflow must fail atomically without corrupting state */
    CHECK(cs_conformal_step(cp, 1.0e308, -1.0e308, &res) == CS_ERR_NONFINITE);
    CHECK(cs_conformal_step(cp, 1.0, 0.0, &res) == CS_OK);
    CHECK(res.step_count == 34ULL);
    cs_conformal_destroy(cp);

    /* Multi-channel SoA */
    cs_conformal_config_default(&cfg);
    cfg.enable_ereset = false;
    cs_conformal_soa_t *soa = cs_conformal_soa_create(17U, &cfg);
    CHECK(soa != NULL);
    double yt[17], yp[17], qn[17];
    uint8_t mc[17];
    for (size_t c = 0; c < 17U; ++c) {
        yt[c] = (double)(c + 1U) * 0.2;
        yp[c] = 0.0;
    }
    for (int t = 0; t < 200; ++t) {
        CHECK(cs_conformal_soa_step(soa, yt, yp, NULL, qn, mc) == CS_OK);
    }
    for (size_t c = 0; c < 17U; ++c) {
        CHECK(isfinite(qn[c]) && qn[c] >= 0.0);
    }
    cs_conformal_soa_destroy(soa);

    /* COMA exact union + disjoint bimodal + singleton touching point */
    {
        double lo[3] = {0.0, 0.0, 5.0};
        double hi[3] = {6.0, 1.0, 6.0};
        double w[3]  = {0.4, 0.3, 0.3};
        cs_interval_t ivs[3];
        cs_coma_result_t cres;
        CHECK(cs_coma_merge(lo, hi, w, 3U, 0.5, false, NULL, ivs, 3U, &cres) == CS_OK);
        CHECK(cres.n_intervals == 2U);
        CHECK(fabs(ivs[0].lo - 0.0) < 1e-12 && fabs(ivs[0].hi - 1.0) < 1e-12);
        CHECK(fabs(ivs[1].lo - 5.0) < 1e-12 && fabs(ivs[1].hi - 6.0) < 1e-12);
        CHECK(fabs(cres.hull_lo - 0.0) < 1e-12 && fabs(cres.hull_hi - 6.0) < 1e-12);
        CHECK(fabs(cres.max_agreement - 0.7) < 1e-12);
        /* Buffer too small (1 < 2 disjoint intervals) must return CS_ERR_BUFFER */
        CHECK(cs_coma_merge(lo, hi, w, 3U, 0.5, false, NULL, ivs, 1U, &cres) == CS_ERR_BUFFER);
    }
    {
        /* Singleton touching point [1, 1] */
        double lo[2] = {0.0, 1.0};
        double hi[2] = {1.0, 2.0};
        double w[2]  = {0.5, 0.5};
        cs_interval_t ivs[2];
        cs_coma_result_t cres;
        CHECK(cs_coma_merge(lo, hi, w, 2U, 0.5, false, NULL, ivs, 2U, &cres) == CS_OK);
        CHECK(cres.n_intervals == 1U);
        CHECK(fabs(ivs[0].lo - 1.0) < 1e-12 && fabs(ivs[0].hi - 1.0) < 1e-12);
        CHECK(fabs(cres.max_agreement - 1.0) < 1e-12);
    }
    {
        /* Randomized U in [0.5, 1) */
        double lo[3] = {0.0, 0.5, 1.0};
        double hi[3] = {2.0, 2.5, 3.0};
        cs_interval_t ivs[3];
        cs_coma_result_t cres;
        uint64_t rstate = 12345ULL;
        CHECK(cs_coma_merge(lo, hi, NULL, 3U, 0.5, true, &rstate, ivs, 3U, &cres) == CS_OK);
        CHECK(cres.used_tau >= 0.5 && cres.used_tau < 1.0);
    }
    {
        /* cs_coma_merge_batch: empty batch, shared weights, per-step weights, and K=80 heap fallback */
        CHECK(cs_coma_merge_batch(NULL, NULL, NULL, false, 0U, 3U, 0.5, false, NULL, NULL, NULL, NULL, NULL, NULL) == CS_OK);

        double lo_b[6] = {0.0, 0.0, 5.0,  0.0, 1.0, 1.0};
        double hi_b[6] = {6.0, 1.0, 6.0,  1.0, 2.0, 2.0};
        double w_b[6]  = {0.4, 0.3, 0.3,  0.5, 0.25, 0.25};
        double hlo[2], hhi[2], agree[2];
        uint32_t nivs[2];
        cs_coma_result_t last_c;
        CHECK(cs_coma_merge_batch(lo_b, hi_b, w_b, true, 2U, 3U, 0.5, false, NULL, hlo, hhi, agree, nivs, &last_c) == CS_OK);
        CHECK(nivs[0] == 2U && fabs(hlo[0] - 0.0) < 1e-12 && fabs(hhi[0] - 6.0) < 1e-12 && fabs(agree[0] - 0.7) < 1e-12);
        /* Strict COMA (> tau=0.5): on [0,1) w=0.5, at x=1 w=1.0 > 0.5, on (1,2] w=0.5 => exact singleton [1.0, 1.0] */
        CHECK(nivs[1] == 1U && fabs(hlo[1] - 1.0) < 1e-12 && fabs(hhi[1] - 1.0) < 1e-12 && fabs(agree[1] - 1.0) < 1e-12);
        CHECK(last_c.n_intervals == 1U && fabs(last_c.hull_lo - 1.0) < 1e-12 && fabs(last_c.hull_hi - 1.0) < 1e-12);

        /* K = 80 (> 64 stack limit -> exercises heap path cleanly under ASan) */
        double lo80[160], hi80[160], w80[80];
        for (size_t k = 0; k < 80U; ++k) {
            lo80[k] = 1.0 + 0.01 * (double)k;
            hi80[k] = 5.0 + 0.01 * (double)k;
            lo80[80U + k] = -2.0;
            hi80[80U + k] = 2.0;
            w80[k] = 1.0;
        }
        CHECK(cs_coma_merge_batch(lo80, hi80, w80, false, 2U, 80U, 0.5, false, NULL, hlo, hhi, agree, nivs, NULL) == CS_OK);
        CHECK(nivs[0] == 1U && fabs(agree[0] - 1.0) < 1e-12);
        CHECK(nivs[1] == 1U && fabs(hlo[1] - (-2.0)) < 1e-12 && fabs(hhi[1] - 2.0) < 1e-12);

        /* Overflow/underflow-safe scale-invariant COMA weight normalization */
        double lo2[2] = {0.0, 0.0};
        double hi2[2] = {1.0, 1.0};
        double w_huge[2] = {1.0e308, 1.0e308};
        double w_tiny[2] = {1.0e-320, 1.0e-320};
        cs_interval_t iv2[2];
        cs_coma_result_t c2;
        CHECK(cs_coma_merge(lo2, hi2, w_huge, 2U, 0.5, false, NULL, iv2, 2U, &c2) == CS_OK);
        CHECK(c2.n_intervals == 1U && fabs(c2.max_agreement - 1.0) < 1e-12);
        CHECK(cs_coma_merge(lo2, hi2, w_tiny, 2U, 0.5, false, NULL, iv2, 2U, &c2) == CS_OK);
        CHECK(c2.n_intervals == 1U && fabs(c2.max_agreement - 1.0) < 1e-12);
    }

    printf("[PASS] conformal_and_coma\n");
}

static void test_fuser_and_digest(void) {
    /* Sleeping BOA + Rogue attribution */
    cs_fuser_config_t fcfg;
    cs_fuser_config_default(&fcfg, 4U);
    fcfg.rogue_mu0 = 0.2;
    fcfg.rogue_sigma = 0.3;
    fcfg.rogue_threshold = 50.0;
    cs_fuser_t *fus = cs_fuser_create(&fcfg);
    CHECK(fus != NULL);

    cs_fuser_result_t fres;
    uint8_t all_sleep[4] = {0, 0, 0, 0};
    double p_nan[4] = {NAN, NAN, NAN, NAN};
    /* Sleeping agents' NaNs must be safely ignored */
    CHECK(cs_fuser_step(fus, p_nan, all_sleep, 1.0, false, &fres) == CS_OK);
    CHECK(fres.active_count == 0U);

    uint8_t mask[4] = {1, 1, 1, 0};
    for (int t = 0; t < 80; ++t) {
        mask[3] = (t % 5 == 0) ? 1U : 0U;
        double preds[4] = {1.0, 1.05, 5.0, 1.02}; /* Agent 2 is rogue! */
        CHECK(cs_fuser_step(fus, preds, mask, 1.0, false, &fres) == CS_OK);
    }
    double rstats[4];
    uint8_t rflags[4];
    CHECK(cs_fuser_get_rogue_stats(fus, rstats, rflags, 4U) == CS_OK);
    CHECK(rflags[2] == 1U);
    CHECK(rflags[0] == 0U && rflags[1] == 0U);
    CHECK(fabs(fres.prediction - 1.0) < 0.2);
    cs_fuser_destroy(fus);

    /* Peer-relative rogue attribution immunity to 86-sigma exogenous spikes across K=3..8,
     * plus graceful raw-residual fallback when active_cnt < 3. */
    for (size_t K = 3U; K <= 8U; ++K) {
        cs_fuser_config_t pcfg;
        cs_fuser_config_default(&pcfg, K);
        CHECK(pcfg.peer_relative_rogue == true);
        pcfg.rogue_mu0 = 0.25;
        pcfg.rogue_sigma = 0.3;
        pcfg.rogue_threshold = 200.0;
        cs_fuser_t *pf = cs_fuser_create(&pcfg);
        CHECK(pf != NULL);
        double preds[8];
        for (int t = 0; t < 120; ++t) {
            double y_true = (t == 40 || t == 80) ? 95.5 : 10.0; /* 86-sigma common shock */
            for (size_t k = 0; k < K; ++k) {
                preds[k] = 10.0 + 0.05 * (double)((t + (int)k) % 3 - 1);
            }
            CHECK(cs_fuser_step(pf, preds, NULL, y_true, false, &fres) == CS_OK);
        }
        uint8_t pflags[8];
        CHECK(cs_fuser_get_rogue_stats(pf, NULL, pflags, K) == CS_OK);
        for (size_t k = 0; k < K; ++k) {
            CHECK(pflags[k] == 0U); /* Zero false quarantines on exogenous spikes! */
        }
        cs_fuser_destroy(pf);
    }
    /* Raw-residual fallback when active_cnt == 2 (< 3) */
    {
        cs_fuser_config_t pcfg;
        cs_fuser_config_default(&pcfg, 2U);
        pcfg.rogue_mu0 = 0.25;
        pcfg.rogue_sigma = 0.3;
        pcfg.rogue_threshold = 50.0;
        cs_fuser_t *pf = cs_fuser_create(&pcfg);
        CHECK(pf != NULL);
        for (int t = 0; t < 60; ++t) {
            double preds[2] = {1.0, 5.0};
            CHECK(cs_fuser_step(pf, preds, NULL, 1.0, false, &fres) == CS_OK);
        }
        uint8_t pflags[2];
        CHECK(cs_fuser_get_rogue_stats(pf, NULL, pflags, 2U) == CS_OK);
        CHECK(pflags[0] == 0U && pflags[1] == 1U);
        cs_fuser_destroy(pf);
    }
    /* Symmetric peer-median calibration floor at minimum cal_window=16 protects accurate agent 0 */
    {
        cs_fuser_config_t pcfg;
        cs_fuser_config_default(&pcfg, 4U);
        pcfg.cal_window = 16U;
        pcfg.rogue_threshold = 200.0;
        pcfg.peer_relative_rogue = true;
        cs_fuser_t *pf = cs_fuser_create(&pcfg);
        CHECK(pf != NULL);
        for (int t = 0; t < 160; ++t) {
            double sgn = (t % 2 == 0) ? 1.0 : -1.0;
            double p0 = (t == 90) ? 0.25 : (0.01 * sgn);
            double preds[4] = {p0, 0.45 * sgn, -0.50 * sgn, 0.55 * sgn};
            CHECK(cs_fuser_step(pf, preds, NULL, 0.0, false, &fres) == CS_OK);
        }
        uint8_t pflags[4];
        CHECK(cs_fuser_get_rogue_stats(pf, NULL, pflags, 4U) == CS_OK);
        CHECK(pflags[0] == 0U);
        cs_fuser_destroy(pf);
    }

    /* Swing PLA L_inf <= eps + verbatim anomalies + ring-buffer zoom() */
    cs_digest_config_t dcfg;
    cs_digest_config_default(&dcfg);
    dcfg.epsilon = 0.05;
    dcfg.ring_capacity = 256U;
    dcfg.max_segments = 4096U;
    cs_digest_t *dig = cs_digest_create(&dcfg);
    CHECK(dig != NULL);

    const size_t M = 400U;
    double ts[400], ys[400], rec[400];
    for (size_t i = 0; i < M; ++i) {
        ts[i] = (double)i;
        ys[i] = sin((double)i * 0.05) + ((i == 123U || i == 300U) ? 3.5 : 0.0);
        uint32_t fl = (i == 123U || i == 300U) ? CS_FLAG_ANOMALY : 0U;
        cs_digest_step_t st;
        CHECK(cs_digest_push(dig, ts[i], ys[i], ys[i], 0.5, 1.0, fl, &st) == CS_OK);
        if (fl & CS_FLAG_ANOMALY) {
            CHECK(st.should_wake == true);
        } else {
            CHECK(st.should_wake == false);
        }
    }
    /* Reject out-of-order timestamp */
    CHECK(cs_digest_push(dig, 10.0, 0.0, 0.0, 0.1, 0.0, 0U, NULL) == CS_ERR_INVALID);

    CHECK(cs_digest_reconstruct(dig, ts, M, rec) == CS_OK);
    for (size_t i = 0; i < M; ++i) {
        double err = fabs(rec[i] - ys[i]);
        if (i == 123U || i == 300U) {
            CHECK(err == 0.0);
        } else {
            CHECK(err <= dcfg.epsilon + 1.0e-12);
        }
    }

    /* Ring buffer holds last 256 records [144..399]; zooming [100, 200] must flag truncated=true */
    size_t zn = 0;
    bool trunc = false;
    cs_ring_record_t zbuf[128];
    CHECK(cs_digest_zoom(dig, 100.0, 200.0, zbuf, 128U, &zn, &trunc) == CS_OK);
    CHECK(trunc == true);
    CHECK(zn == 57U); /* 144..200 inclusive */
    CHECK(zbuf[0].t == 144.0 && zbuf[zn - 1U].t == 200.0);

    /* Zooming [295, 305] must be lossless (truncated=false) and exact */
    CHECK(cs_digest_zoom(dig, 295.0, 305.0, zbuf, 128U, &zn, &trunc) == CS_OK);
    CHECK(trunc == false);
    CHECK(zn == 11U);
    for (size_t i = 0; i < zn; ++i) {
        CHECK(zbuf[i].y == ys[295U + i]);
    }

    char md[2048];
    size_t md_len = 0;
    CHECK(cs_digest_format_markdown(dig, "sensor_0", 0.91, 0.42, 612.5, 500.0, 2, 6U, 5U, md, sizeof(md), &md_len) == CS_OK);
    CHECK(md_len > 100U && md_len < sizeof(md));
    CHECK(strstr(md, "zoom(\"sensor_0\"") != NULL);

    cs_digest_destroy(dig);

    /* Verify cs_digest_push_batch SIMD stationary fast path matches sequential cs_digest_push */
    {
        cs_digest_t *d_seq = cs_digest_create(&dcfg);
        cs_digest_t *d_bat = cs_digest_create(&dcfg);
        CHECK(d_seq != NULL && d_bat != NULL);
        for (size_t i = 0; i < M; ++i) {
            CHECK(cs_digest_push(d_seq, ts[i], ys[i], ys[i], 0.0, 0.0, 0U, NULL) == CS_OK);
        }
        CHECK(cs_digest_push_batch(d_bat, ts, ys, NULL, NULL, NULL, NULL, M, NULL, NULL) == CS_OK);
        CHECK(cs_digest_segment_count(d_seq) == cs_digest_segment_count(d_bat));
        double rec_seq[400], rec_bat[400];
        CHECK(cs_digest_reconstruct(d_seq, ts, M, rec_seq) == CS_OK);
        CHECK(cs_digest_reconstruct(d_bat, ts, M, rec_bat) == CS_OK);
        for (size_t i = 0; i < M; ++i) {
            CHECK(fabs(rec_bat[i] - ys[i]) <= dcfg.epsilon + 1.0e-12);
            CHECK(fabs(rec_bat[i] - rec_seq[i]) <= 1.0e-12);
        }
        /* Verify lossless zoom on batch-pushed digest */
        cs_ring_record_t zb[128];
        size_t zbn = 0U;
        bool zbt = true;
        CHECK(cs_digest_zoom(d_bat, 295.0, 305.0, zb, 128U, &zbn, &zbt) == CS_OK);
        CHECK(zbt == false && zbn == 11U);
        for (size_t i = 0; i < zbn; ++i) {
            CHECK(zb[i].y == ys[295U + i]);
        }
        cs_digest_destroy(d_seq);
        cs_digest_destroy(d_bat);

        /* Tiny dt=1e-290 + large eps=1e20 must match scalar (no +/-inf slope bypass in SIMD) */
        cs_digest_config_t tcfg;
        cs_digest_config_default(&tcfg);
        tcfg.epsilon = 1.0e20;
        cs_digest_t *td_seq = cs_digest_create(&tcfg);
        cs_digest_t *td_bat = cs_digest_create(&tcfg);
        CHECK(td_seq != NULL && td_bat != NULL);
        double tts[4] = {0.0, 1.0e-290, 2.0e-290, 3.0e-290};
        double tys[4] = {0.0, 0.0, 0.0, 0.0};
        for (size_t i = 0; i < 4U; ++i) {
            CHECK(cs_digest_push(td_seq, tts[i], tys[i], tys[i], 0.0, 0.0, 0U, NULL) == CS_OK);
        }
        CHECK(cs_digest_push_batch(td_bat, tts, tys, NULL, NULL, NULL, NULL, 4U, NULL, NULL) == CS_OK);
        CHECK(cs_digest_segment_count(td_seq) == 4U);
        CHECK(cs_digest_segment_count(td_bat) == cs_digest_segment_count(td_seq));
        cs_digest_destroy(td_seq);
        cs_digest_destroy(td_bat);
    }

    printf("[PASS] fuser_and_digest\n");
}

int main(void) {
    printf("=== ChronoSieve C11 Core Test Suite (SIMD=%s) ===\n", cs_simd_backend_name());
    test_simd_math_accuracy();
    test_btree_exact_order_stats();
    test_edetectors_and_efusion();
    test_conformal_and_coma();
    test_fuser_and_digest();
    printf("ALL C11 CORE TESTS PASSED!\n");
    return 0;
}
