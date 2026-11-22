#!/usr/bin/env python3
"""Zero-dependency example: SIMD batch E-Detectors, E-Reset Conformal PID, SoA, & COMA Batch.

Uses ONLY Python stdlib (`array`, `math`, `random`, `time`) + `chronosieve` (zero NumPy required!).
"""

from __future__ import annotations

import array
import math
import random
import time

import chronosieve as cs


def main() -> None:
    rng = random.Random(42)
    print(f"ChronoSieve v{cs.__version__} ({cs.simd_backend()} SIMD) — Zero-Dependency Stdlib Demo\n")

    # 1. Heavy-tailed Cauchy stream with zero-parameter Vovk Rank + Symmetry E-Detectors
    n = 1200
    cp = 800
    cauchy = array.array(
        "d",
        (math.tan(math.pi * (rng.random() - 0.5)) + (2.5 if i >= cp else 0.0) for i in range(n)),
    )
    alarms_sym = bytearray(n)
    alarms_rk = bytearray(n)
    d_sym = cs.EDetector(family="symmetry", mu0=0.0, sigma=1.0, threshold=5000.0)
    d_rk = cs.EDetector(family="vovk_rank", max_prefix=2048, threshold=5000.0, seed=7)
    d_sym.ingest_batch(cauchy, out_alarms=alarms_sym)
    d_rk.ingest_batch(cauchy, out_alarms=alarms_rk)
    first_sym = next((i for i, a in enumerate(alarms_sym) if a), -1)
    first_rk = next((i for i, a in enumerate(alarms_rk) if a), -1)
    print(f"[1] Cauchy(0,1) +2.5 shift at t={cp}:")
    print(f"    Symmetry E-SR first alarm:  t={first_sym} (delay={first_sym - cp} ticks)")
    print(f"    Vovk Rank E-SR first alarm: t={first_rk} (delay={first_rk - cp} ticks)")

    # 2. Pre-Control E-Reset Conformal PID vs Pure P+I PID across an abrupt 5x volatility shock
    n_vol = 2000
    shock_t = 1000
    ys = array.array(
        "d", (rng.gauss(0.0, 1.0 if i < shock_t else 5.0) for i in range(n_vol))
    )
    yp = array.array("d", [0.0] * n_vol)
    mc_ereset = bytearray(n_vol)
    mc_pure = bytearray(n_vol)
    cp_ereset = cs.ConformalPID(alpha=0.10, eta=0.05, enable_ereset=True)
    cp_pure = cs.ConformalPID(alpha=0.10, eta=0.05, enable_ereset=False)
    cp_ereset.ingest_batch(ys, yp, out_miscov=mc_ereset)
    cp_pure.ingest_batch(ys, yp, out_miscov=mc_pure)
    cov_er_shock = 1.0 - sum(mc_ereset[shock_t : shock_t + 100]) / 100.0
    cov_pu_shock = 1.0 - sum(mc_pure[shock_t : shock_t + 100]) / 100.0
    print("\n[2] Post-5x-volatility-shock 100-tick coverage (target = 90.0%):")
    print(f"    Pre-Control E-Reset PID: {cov_er_shock * 100:.1f}%")
    print(f"    Pure P+I Conformal PID:  {cov_pu_shock * 100:.1f}%")

    # 3. 64-Channel SoA ConformalBatchPID zero-copy throughput via stdlib array.array
    C, T = 64, 4000
    yt_soa = array.array("d", (rng.gauss(0.0, 1.0) for _ in range(T * C)))
    yp_soa = array.array("d", [0.0] * (T * C))
    soa = cs.ConformalBatchPID(n_channels=C, alpha=0.10, eta=0.05)
    t0 = time.perf_counter_ns()
    soa.ingest_batch(yt_soa, yp_soa)
    dt = time.perf_counter_ns() - t0
    print(f"\n[3] 64-channel SoA ConformalBatchPID ({T * C:,} ch-ticks):")
    print(f"    {dt / (T * C):.2f} ns/ch-tick ({(T * C) * 1e3 / dt:.1f} M ch-ticks/s)")

    # 4. Zero-copy COMA batch interval consensus (K=5 agents over T=1000 steps)
    K, T_c = 5, 1000
    lo_flat = array.array("d")
    hi_flat = array.array("d")
    for _ in range(T_c):
        for _k in range(K):
            c_val = rng.gauss(0.0, 0.4)
            lo_flat.append(c_val - 1.5)
            hi_flat.append(c_val + 1.5)
    out_lo = array.array("d", [0.0] * T_c)
    out_hi = array.array("d", [0.0] * T_c)
    out_ni = array.array("I", [0] * T_c)
    summary = cs.coma_merge_batch(
        lo_flat,
        hi_flat,
        n_agents=K,
        tau=0.5,
        out_hull_lo=out_lo,
        out_hull_hi=out_hi,
        out_n_intervals=out_ni,
    )
    avg_width = sum(out_hi[i] - out_lo[i] for i in range(T_c)) / T_c
    print(f"\n[4] Zero-copy COMA batch (T={T_c}, K={K}):")
    print(
        f"    avg hull width={avg_width:.3f}, last_hull={summary['last_hull']}, "
        f"last_n_ivs={summary['last_n_intervals']}"
    )


if __name__ == "__main__":
    main()
