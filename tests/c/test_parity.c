#include "chronosieve.h"
#include "cs_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("BACKEND %s\n", cs_simd_backend_name());

    cs_rng_t rng;
    cs_rng_seed(&rng, 0xCAFEBABE12345678ULL);

    const size_t N = 500U;
    double xs[500], zs[500], ys[500];
    for (size_t i = 0; i < N; ++i) {
        double u = cs_rng_uniform01(&rng);
        zs[i] = u;
        xs[i] = (u - 0.5) * 3.0 + ((i >= 350U) ? 0.6 : 0.0);
        ys[i] = sin((double)i * 0.03) + (u - 0.5) * 0.2;
    }

    /* 1. Catoni E-SR batch (register-resident NEON vs scalar) */
    {
        cs_edet_config_t cfg;
        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_CATONI;
        cfg.accum = CS_ACCUM_ESR;
        cfg.mu0 = 0.0;
        cfg.sigma = 1.0;
        cfg.threshold = 1.0e12;
        cs_edetector_t *d = cs_edet_create(&cfg);
        double stats[500];
        cs_edet_step_batch(d, xs, N, stats, NULL, NULL);
        for (size_t i = 0; i < N; i += 25U) {
            printf("catoni_esr_%zu %.17g\n", i, stats[i]);
        }
        cs_edet_destroy(d);
    }

    /* 2. Catoni E-CUSUM (uses SIMD vexpq_f64!) */
    {
        cs_edet_config_t cfg;
        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_CATONI;
        cfg.accum = CS_ACCUM_ECUSUM;
        cfg.mu0 = 0.0;
        cfg.sigma = 1.0;
        cfg.threshold = 1.0e12;
        cs_edetector_t *d = cs_edet_create(&cfg);
        double stats[500];
        cs_edet_step_batch(d, xs, N, stats, NULL, NULL);
        for (size_t i = 0; i < N; i += 50U) {
            printf("catoni_ecusum_%zu %.17g\n", i, stats[i]);
        }
        cs_edet_destroy(d);
    }

    /* 3. Bounded (1-sided & 2-sided), Symmetry (1-sided & 2-sided), & Vovk Rank E-SR */
    {
        cs_edet_config_t cfg;
        cs_edet_result_t r;
        double bstats[500];

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_BOUNDED;
        cfg.side = CS_SIDE_TWO_SIDED;
        cfg.mu0 = 0.5;
        cfg.threshold = 1.0e12;
        cs_edetector_t *d = cs_edet_create(&cfg);
        cs_edet_step_batch(d, zs, N, bstats, NULL, &r);
        printf("bounded_2s_mid %.17g\n", bstats[250]);
        printf("bounded_final %.17g\n", r.stat);
        cs_edet_destroy(d);

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_BOUNDED;
        cfg.side = CS_SIDE_UPPER;
        cfg.mu0 = 0.5;
        cfg.threshold = 1.0e12;
        d = cs_edet_create(&cfg);
        cs_edet_step_batch(d, zs, N, bstats, NULL, &r);
        printf("bounded_1s_final %.17g\n", r.stat);
        cs_edet_destroy(d);

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_SYMMETRY;
        cfg.side = CS_SIDE_TWO_SIDED;
        cfg.mu0 = 0.0;
        cfg.threshold = 1.0e12;
        d = cs_edet_create(&cfg);
        cs_edet_step_batch(d, xs, N, bstats, NULL, &r);
        printf("symmetry_2s_mid %.17g\n", bstats[250]);
        printf("symmetry_final %.17g\n", r.stat);
        cs_edet_destroy(d);

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_SYMMETRY;
        cfg.side = CS_SIDE_UPPER;
        cfg.mu0 = 0.0;
        cfg.threshold = 1.0e12;
        d = cs_edet_create(&cfg);
        cs_edet_step_batch(d, xs, N, bstats, NULL, &r);
        printf("symmetry_1s_final %.17g\n", r.stat);
        cs_edet_destroy(d);

        cs_edet_config_default(&cfg);
        cfg.family = CS_FAMILY_VOVK_RANK;
        cfg.threshold = 1.0e12;
        cfg.seed = 999ULL;
        d = cs_edet_create(&cfg);
        cs_edet_step_batch(d, xs, N, bstats, NULL, &r);
        printf("rank_mid %.17g\n", bstats[250]);
        printf("rank_final %.17g\n", r.stat);
        cs_edet_destroy(d);
    }

    /* 3b. Swing PLA batch & COMA batch parity */
    {
        cs_digest_config_t dcfg;
        cs_digest_config_default(&dcfg);
        dcfg.epsilon = 0.05;
        cs_digest_t *dig = cs_digest_create(&dcfg);
        double ts[500], rec[500];
        for (size_t i = 0; i < N; ++i) ts[i] = (double)i;
        cs_digest_push_batch(dig, ts, ys, NULL, NULL, NULL, NULL, N, NULL, NULL);
        cs_digest_reconstruct(dig, ts, N, rec);
        printf("digest_segs %.17g\n", (double)cs_digest_segment_count(dig));
        printf("digest_rec_250 %.17g\n", rec[250]);
        printf("digest_rec_499 %.17g\n", rec[499]);
        cs_digest_destroy(dig);

        double lo_m[80], hi_m[80], hlo[10], hhi[10], agree[10];
        for (size_t i = 0; i < 80U; ++i) {
            lo_m[i] = xs[i] - 1.5;
            hi_m[i] = xs[i] + 1.5 + 0.1 * zs[i];
        }
        cs_coma_merge_batch(lo_m, hi_m, NULL, false, 10U, 8U, 0.5, false, NULL, hlo, hhi, agree, NULL, NULL);
        printf("coma_hlo_9 %.17g\n", hlo[9]);
        printf("coma_hhi_9 %.17g\n", hhi[9]);
        printf("coma_agree_9 %.17g\n", agree[9]);
    }

    /* 4. Multi-channel Conformal PID SoA (19 channels: tests SIMD + scalar tail + vtan) */
    {
        const size_t C = 19U;
        cs_conformal_config_t cfg;
        cs_conformal_config_default(&cfg);
        cfg.enable_ereset = false;
        cfg.ki = 1.5;
        cfg.eta = 0.08;
        cs_conformal_soa_t *soa = cs_conformal_soa_create(C, &cfg);
        double yt[19], yp[19], qn[19];
        for (size_t t = 0; t < 150U; ++t) {
            for (size_t c = 0; c < C; ++c) {
                yt[c] = xs[(t * C + c) % N];
                yp[c] = ys[(t * C + c) % N];
            }
            cs_conformal_soa_step(soa, yt, yp, NULL, qn, NULL);
        }
        for (size_t c = 0; c < C; ++c) {
            printf("soa_q_%zu %.17g\n", c, qn[c]);
        }
        cs_conformal_soa_destroy(soa);
    }

    /* 5. Sleeping Fixed-Share BOA (K = 13 agents: tests SIMD exp + scalar tail) */
    {
        const size_t K = 13U;
        cs_fuser_config_t cfg;
        cs_fuser_config_default(&cfg, K);
        cfg.rogue_mu0 = 0.5;
        cfg.rogue_sigma = 0.8;
        cfg.rogue_threshold = 1.0e9;
        cs_fuser_t *fus = cs_fuser_create(&cfg);
        double preds[13];
        uint8_t active[13];
        cs_fuser_result_t fres;
        for (size_t t = 0; t < 200U; ++t) {
            for (size_t k = 0; k < K; ++k) {
                preds[k] = ys[t] + 0.1 * (double)(k + 1U) * xs[(t * 7U + k) % N];
                active[k] = ((t + k) % 4U == 0U && k > 1U) ? 0U : 1U;
            }
            bool sh = (t == 100U);
            cs_fuser_step(fus, preds, active, ys[t], sh, &fres);
        }
        printf("boa_pred %.17g\n", fres.prediction);
        double w[13], rs[13];
        cs_fuser_get_weights(fus, w, K);
        cs_fuser_get_rogue_stats(fus, rs, NULL, K);
        for (size_t k = 0; k < K; ++k) {
            printf("boa_w_%zu %.17g\n", k, w[k]);
            printf("boa_rs_%zu %.17g\n", k, rs[k]);
        }
        cs_fuser_destroy(fus);
    }

    return 0;
}
