#if !defined(_POSIX_C_SOURCE) && !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "chronosieve.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double cs_now_ns(void) {
    struct timespec ts;
#if defined(_WIN32)
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

static void print_usage(void) {
    printf(
        "chronosieve %s (%s SIMD) - Zero-dep C11 online inference & agent digest engine\n"
        "\n"
        "Usage:\n"
        "  chronosieve info\n"
        "  chronosieve detect [--family auto|catoni|bounded|symmetry|rank]\n"
        "                     [--accum esr|ecusum] [--side both|upper|lower]\n"
        "                     [--mu0 V] [--sigma S] [--threshold G] [--auto-reset] [file]\n"
        "  chronosieve conformal [--alpha A] [--eta E] [--threshold G] [--no-ereset] [file]\n"
        "  chronosieve digest [--eps E] [--channel NAME] [--threshold G] [--alpha A]\n"
        "                     [--zoom T0:T1] [file]\n"
        "  chronosieve bench [--n N]\n",
        cs_version_string(),
        cs_simd_backend_name());
}

static int parse_two_doubles(const char *line, double *a, double *b, int *count) {
    const char *p = line;
    while (*p && (isspace((unsigned char)*p) || *p == ',')) {
        ++p;
    }
    if (*p == '\0' || *p == '#') {
        *count = 0;
        return 0;
    }
    char *end1 = NULL;
    double v1 = strtod(p, &end1);
    if (end1 == p) {
        *count = 0;
        return -1;
    }
    *a = v1;
    p = end1;
    while (*p && (isspace((unsigned char)*p) || *p == ',' || *p == '\t' || *p == ';')) {
        ++p;
    }
    if (*p == '\0' || *p == '#') {
        *count = 1;
        return 0;
    }
    char *end2 = NULL;
    double v2 = strtod(p, &end2);
    if (end2 == p) {
        *count = 1;
        return 0;
    }
    *b = v2;
    *count = 2;
    return 0;
}

static int cmd_detect(int argc, char **argv) {
    cs_edet_config_t cfg;
    cs_edet_config_default(&cfg);
    const char *path = NULL;
    bool json = false;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--family") == 0 && i + 1 < argc) {
            const char *f = argv[++i];
            if (strcmp(f, "bounded") == 0) cfg.family = CS_FAMILY_BOUNDED;
            else if (strcmp(f, "catoni") == 0) cfg.family = CS_FAMILY_CATONI;
            else if (strcmp(f, "symmetry") == 0) cfg.family = CS_FAMILY_SYMMETRY;
            else if (strcmp(f, "rank") == 0 || strcmp(f, "vovk") == 0 || strcmp(f, "vovk_rank") == 0)
                cfg.family = CS_FAMILY_VOVK_RANK;
            else if (strcmp(f, "auto") == 0) cfg.family = CS_FAMILY_AUTO;
            else {
                fprintf(stderr, "chronosieve: unknown family '%s'\n", f);
                return 1;
            }
        } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
            const char *a = argv[++i];
            if (strcmp(a, "ecusum") == 0) cfg.accum = CS_ACCUM_ECUSUM;
            else if (strcmp(a, "esr") == 0) cfg.accum = CS_ACCUM_ESR;
            else {
                fprintf(stderr, "chronosieve: unknown accumulator '%s'\n", a);
                return 1;
            }
        } else if (strcmp(argv[i], "--side") == 0 && i + 1 < argc) {
            const char *s = argv[++i];
            if (strcmp(s, "upper") == 0) cfg.side = CS_SIDE_UPPER;
            else if (strcmp(s, "lower") == 0) cfg.side = CS_SIDE_LOWER;
            else if (strcmp(s, "both") == 0 || strcmp(s, "two_sided") == 0) cfg.side = CS_SIDE_TWO_SIDED;
            else {
                fprintf(stderr, "chronosieve: unknown side '%s'\n", s);
                return 1;
            }
        } else if (strcmp(argv[i], "--mu0") == 0 && i + 1 < argc) {
            cfg.mu0 = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            cfg.sigma = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            cfg.threshold = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--auto-reset") == 0) {
            cfg.auto_reset = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (argv[i][0] != '-') {
            path = argv[i];
        } else {
            fprintf(stderr, "chronosieve: unknown or incomplete option '%s'\n", argv[i]);
            return 1;
        }
    }

    FILE *fp = path ? fopen(path, "r") : stdin;
    if (!fp) {
        fprintf(stderr, "chronosieve: cannot open %s\n", path);
        return 1;
    }

    cs_edetector_t *det = cs_edet_create(&cfg);
    if (!det) {
        fprintf(stderr, "chronosieve: invalid detector configuration\n");
        if (path) fclose(fp);
        return 1;
    }

    char line[512];
    cs_edet_result_t res;
    memset(&res, 0, sizeof(res));
    uint64_t idx = 0;
    uint64_t first_alarm = 0;
    bool has_alarm = false;

    while (fgets(line, sizeof(line), fp)) {
        double a = 0.0, b = 0.0;
        int cnt = 0;
        if (parse_two_doubles(line, &a, &b, &cnt) != 0 || cnt == 0) {
            continue;
        }
        double val = (cnt == 2) ? b : a;
        if (cs_edet_step(det, val, &res) != CS_OK) {
            continue;
        }
        if (res.alarmed) {
            if (!has_alarm) {
                has_alarm = true;
                first_alarm = idx;
            }
            if (!json) {
                printf("ALARM tick=%llu stat=%.4f threshold=%.2f\n",
                       (unsigned long long)idx, res.stat, cfg.threshold);
            }
        }
        idx++;
    }

    if (json) {
        printf("{\"ticks\":%llu,\"alarms\":%llu,\"first_alarm\":%lld,\"final_stat\":%.6f,\"backend\":\"%s\"}\n",
               (unsigned long long)idx,
               (unsigned long long)res.alarm_count,
               has_alarm ? (long long)first_alarm : -1LL,
               res.stat,
               cs_simd_backend_name());
    } else {
        printf("SUMMARY ticks=%llu alarms=%llu final_stat=%.4f backend=%s\n",
               (unsigned long long)idx,
               (unsigned long long)res.alarm_count,
               res.stat,
               cs_simd_backend_name());
    }

    cs_edet_destroy(det);
    if (path) fclose(fp);
    return 0;
}

static int cmd_conformal(int argc, char **argv) {
    cs_conformal_config_t cfg;
    cs_conformal_config_default(&cfg);
    const char *path = NULL;
    bool json = false;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            cfg.alpha = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--eta") == 0 && i + 1 < argc) {
            cfg.eta = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            cfg.edet_threshold = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--no-ereset") == 0) {
            cfg.enable_ereset = false;
        } else if (strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (argv[i][0] != '-') {
            path = argv[i];
        } else {
            fprintf(stderr, "chronosieve: unknown or incomplete option '%s'\n", argv[i]);
            return 1;
        }
    }

    FILE *fp = path ? fopen(path, "r") : stdin;
    if (!fp) {
        fprintf(stderr, "chronosieve: cannot open %s\n", path);
        return 1;
    }

    cs_conformal_pid_t *cp = cs_conformal_create(&cfg);
    if (!cp) {
        fprintf(stderr, "chronosieve: invalid conformal configuration\n");
        if (path) fclose(fp);
        return 1;
    }

    char line[512];
    cs_conformal_result_t res;
    memset(&res, 0, sizeof(res));
    double q_sum = 0.0;
    uint64_t n = 0;

    while (fgets(line, sizeof(line), fp)) {
        double a = 0.0, b = 0.0;
        int cnt = 0;
        if (parse_two_doubles(line, &a, &b, &cnt) != 0 || cnt == 0) {
            continue;
        }
        double y_true = a;
        double y_pred = (cnt == 2) ? b : 0.0;
        if (cs_conformal_step(cp, y_true, y_pred, &res) == CS_OK) {
            q_sum += res.q_used;
            n++;
        }
    }

    double mean_q = (n > 0) ? (q_sum / (double)n) : 0.0;
    if (json) {
        printf("{\"ticks\":%llu,\"coverage\":%.6f,\"target_coverage\":%.6f,\"mean_q\":%.6f,\"final_q\":%.6f,\"resets\":%llu}\n",
               (unsigned long long)n, res.emp_coverage, 1.0 - cfg.alpha, mean_q, res.q_next,
               (unsigned long long)res.reset_count);
    } else {
        printf("CONFORMAL ticks=%llu coverage=%.4f target=%.4f mean_q=%.4f final_q=%.4f resets=%llu\n",
               (unsigned long long)n, res.emp_coverage, 1.0 - cfg.alpha, mean_q, res.q_next,
               (unsigned long long)res.reset_count);
    }

    cs_conformal_destroy(cp);
    if (path) fclose(fp);
    return 0;
}

static int cmd_digest(int argc, char **argv) {
    cs_digest_config_t dcfg;
    cs_digest_config_default(&dcfg);
    cs_conformal_config_t ccfg;
    cs_conformal_config_default(&ccfg);
    const char *channel = "stdin";
    const char *path = NULL;
    bool do_zoom = false;
    double zoom_t0 = 0.0, zoom_t1 = 0.0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            dcfg.epsilon = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--channel") == 0 && i + 1 < argc) {
            channel = argv[++i];
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            ccfg.alpha = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            ccfg.edet_threshold = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
            const char *z = argv[++i];
            const char *colon = strchr(z, ':');
            if (!colon) {
                fprintf(stderr, "chronosieve: --zoom expects T0:T1\n");
                return 1;
            }
            char *end0 = NULL;
            char *end1 = NULL;
            zoom_t0 = strtod(z, &end0);
            zoom_t1 = strtod(colon + 1, &end1);
            if (end0 != colon || end1 == colon + 1 || *end1 != '\0' ||
                !isfinite(zoom_t0) || !isfinite(zoom_t1) || zoom_t0 > zoom_t1) {
                fprintf(stderr, "chronosieve: invalid --zoom range '%s'\n", z);
                return 1;
            }
            do_zoom = true;
        } else if (argv[i][0] != '-') {
            path = argv[i];
        } else {
            fprintf(stderr, "chronosieve: unknown or incomplete option '%s'\n", argv[i]);
            return 1;
        }
    }

    FILE *fp = path ? fopen(path, "r") : stdin;
    if (!fp) {
        fprintf(stderr, "chronosieve: cannot open %s\n", path);
        return 1;
    }

    cs_digest_t *dig = cs_digest_create(&dcfg);
    cs_conformal_pid_t *cp = cs_conformal_create(&ccfg);
    if (!dig || !cp) {
        cs_digest_destroy(dig);
        cs_conformal_destroy(cp);
        if (path) fclose(fp);
        return 1;
    }

    char line[512];
    double prev_y = 0.0;
    bool has_prev = false;
    uint64_t idx = 0;
    cs_conformal_result_t cres;
    memset(&cres, 0, sizeof(cres));

    while (fgets(line, sizeof(line), fp)) {
        double a = 0.0, b = 0.0;
        int cnt = 0;
        if (parse_two_doubles(line, &a, &b, &cnt) != 0 || cnt == 0) {
            continue;
        }
        double t = (cnt == 2) ? a : (double)idx;
        double y = (cnt == 2) ? b : a;
        double y_hat = has_prev ? prev_y : y;
        has_prev = true;
        prev_y = y;

        cs_conformal_step(cp, y, y_hat, &cres);
        uint32_t fl = 0U;
        if (cres.miscovered && idx >= 16U) {
            fl |= CS_FLAG_ANOMALY;
        }
        if (cres.shift_reset) {
            fl |= CS_FLAG_SHIFT;
        }
        cs_digest_push(dig, t, y, y_hat, cres.q_used, cres.edet_stat, fl, NULL);
        idx++;
    }
    cs_digest_flush(dig);

    if (do_zoom) {
        size_t n_rec = 0;
        bool trunc = false;
        if (cs_digest_zoom(dig, zoom_t0, zoom_t1, NULL, 0U, &n_rec, &trunc) != CS_OK) {
            fprintf(stderr, "chronosieve: zoom failed\n");
            cs_digest_destroy(dig);
            cs_conformal_destroy(cp);
            if (path) fclose(fp);
            return 1;
        }
        printf("t,y,y_hat,q_t,flags\n");
        if (n_rec > 0U) {
            cs_ring_record_t *recs = (cs_ring_record_t *)malloc(n_rec * sizeof(*recs));
            if (recs) {
                cs_digest_zoom(dig, zoom_t0, zoom_t1, recs, n_rec, &n_rec, &trunc);
                for (size_t i = 0; i < n_rec; ++i) {
                    printf("%.6g,%.6g,%.6g,%.6g,%u\n",
                           recs[i].t, recs[i].y, recs[i].y_hat, recs[i].q_t, (unsigned)recs[i].flags);
                }
                free(recs);
            }
        }
    } else {
        size_t len = 0;
        cs_digest_format_markdown(
            dig, channel, cres.emp_coverage, cres.q_next,
            cres.edet_stat, ccfg.edet_threshold, -1,
            8U, 6U, NULL, 0U, &len);
        char *md = (char *)malloc(len + 2U);
        if (md) {
            cs_digest_format_markdown(
                dig, channel, cres.emp_coverage, cres.q_next,
                cres.edet_stat, ccfg.edet_threshold, -1,
                8U, 6U, md, len + 2U, &len);
            printf("%s", md);
            free(md);
        }
    }

    cs_digest_destroy(dig);
    cs_conformal_destroy(cp);
    if (path) fclose(fp);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    size_t N = 1000000U;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || v <= 0) {
                fprintf(stderr, "chronosieve: --n must be a positive integer\n");
                return 1;
            }
            N = (size_t)v;
        } else {
            fprintf(stderr, "chronosieve: unknown or incomplete option '%s'\n", argv[i]);
            return 1;
        }
    }

    const size_t C = 64U;
    const size_t K = 8U;
    size_t buf_n = (N < C) ? C : N;

    double *xs = (double *)malloc(buf_n * sizeof(double));
    double *ys = (double *)malloc(buf_n * sizeof(double));
    if (!xs || !ys) {
        free(xs);
        free(ys);
        return 1;
    }

    uint64_t state = 0x123456789abcdef0ULL;
    for (size_t i = 0; i < buf_n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = (double)(state >> 11) * 0x1.0p-53 - 0.5;
        xs[i] = u * 2.0;
        ys[i] = sin((double)i * 0.01) + u * 0.04;
    }

    printf("ChronoSieve Native C11 Benchmark (N=%zu, SIMD=%s)\n", N, cs_simd_backend_name());

    /* 1. Catoni E-SR batch */
    cs_edet_config_t ecfg;
    cs_edet_config_default(&ecfg);
    ecfg.family = CS_FAMILY_CATONI;
    ecfg.mu0 = 0.0;
    ecfg.sigma = 1.0;
    ecfg.threshold = 1.0e9;
    cs_edetector_t *cat = cs_edet_create(&ecfg);
    double t0 = cs_now_ns();
    cs_edet_result_t eres;
    cs_edet_step_batch(cat, xs, N, NULL, NULL, &eres);
    double dt_cat = cs_now_ns() - t0;
    printf("  Catoni E-SR (16-lane):      %6.2f ns/tick  (%7.1f M ticks/s)  stat=%.3f\n",
           dt_cat / (double)N, (double)N * 1.0e3 / dt_cat, eres.stat);
    cs_edet_destroy(cat);

    /* 2. Vovk Rank B-tree (N_rank = min(N, 200000)) */
    size_t N_rank = (N < 200000U) ? N : 200000U;
    cs_edet_config_default(&ecfg);
    ecfg.family = CS_FAMILY_VOVK_RANK;
    ecfg.threshold = 1.0e9;
    cs_edetector_t *rk = cs_edet_create(&ecfg);
    t0 = cs_now_ns();
    cs_edet_step_batch(rk, xs, N_rank, NULL, NULL, &eres);
    double dt_rk = cs_now_ns() - t0;
    printf("  Vovk Rank B-tree (65k cap): %6.2f ns/tick  (%7.2f M ticks/s)  stat=%.3f\n",
           dt_rk / (double)N_rank, (double)N_rank * 1.0e3 / dt_rk, eres.stat);
    cs_edet_destroy(rk);

    /* 3. Conformal PID SoA (64 channels) */
    size_t T_soa = buf_n / C;
    if (T_soa == 0U) T_soa = 1U;
    cs_conformal_config_t ccfg;
    cs_conformal_config_default(&ccfg);
    ccfg.enable_ereset = false;
    cs_conformal_soa_t *soa = cs_conformal_soa_create(C, &ccfg);
    t0 = cs_now_ns();
    cs_conformal_soa_step_batch(soa, ys, xs, T_soa, NULL, NULL);
    double dt_soa = cs_now_ns() - t0;
    double total_ch_ticks = (double)(T_soa * C);
    printf("  Conformal PID SoA (C=64):   %6.2f ns/ch-tk (%7.1f M ch-tk/s)\n",
           dt_soa / total_ch_ticks, total_ch_ticks * 1.0e3 / dt_soa);
    cs_conformal_soa_destroy(soa);

    /* 4. Sleeping Fixed-Share BOA (K=8 agents) */
    size_t T_boa = (buf_n / K < 100000U) ? (buf_n / K) : 100000U;
    if (T_boa == 0U) T_boa = 1U;
    cs_fuser_config_t fcfg;
    cs_fuser_config_default(&fcfg, K);
    fcfg.rogue_mu0 = 0.5;
    fcfg.rogue_sigma = 1.0;
    fcfg.rogue_threshold = 1.0e9;
    cs_fuser_t *fus = cs_fuser_create(&fcfg);
    cs_fuser_result_t fres;
    t0 = cs_now_ns();
    cs_fuser_step_batch(fus, xs, NULL, ys, NULL, T_boa, NULL, NULL, NULL, &fres);
    double dt_boa = cs_now_ns() - t0;
    printf("  Sleeping BOA + Catoni (K=8):%6.2f ns/step  (%7.2f M steps/s)  y_hat=%.3f\n",
           dt_boa / (double)T_boa, (double)T_boa * 1.0e3 / dt_boa, fres.prediction);
    cs_fuser_destroy(fus);

    /* 5. Swing PLA + Ring Buffer */
    double *ts = (double *)malloc(N * sizeof(double));
    if (ts) {
        for (size_t i = 0; i < N; ++i) ts[i] = (double)i;
        cs_digest_config_t dcfg;
        cs_digest_config_default(&dcfg);
        dcfg.epsilon = 0.05;
        cs_digest_t *dig = cs_digest_create(&dcfg);
        cs_digest_step_t dstep;
        t0 = cs_now_ns();
        cs_digest_push_batch(dig, ts, ys, NULL, NULL, NULL, NULL, N, NULL, &dstep);
        double dt_dig = cs_now_ns() - t0;
        printf("  Swing PLA + RingBuf:        %6.2f ns/tick  (%7.1f M ticks/s)  segs=%llu\n",
               dt_dig / (double)N, (double)N * 1.0e3 / dt_dig, (unsigned long long)dstep.total_segments);
        cs_digest_destroy(dig);
        free(ts);
    }

    free(xs);
    free(ys);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "info") == 0) {
        printf("chronosieve %s (SIMD=%s)\n", cs_version_string(), cs_simd_backend_name());
        return 0;
    }
    if (strcmp(argv[1], "detect") == 0) {
        return cmd_detect(argc, argv);
    }
    if (strcmp(argv[1], "conformal") == 0) {
        return cmd_conformal(argc, argv);
    }
    if (strcmp(argv[1], "digest") == 0) {
        return cmd_digest(argc, argv);
    }
    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc, argv);
    }

    fprintf(stderr, "chronosieve: unknown command '%s'\n", argv[1]);
    print_usage();
    return 1;
}
