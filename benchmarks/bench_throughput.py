#!/usr/bin/env python3
"""Reproducible throughput benchmark: Native C11, Zero-Copy Py_buffer, CPython scalar, & Pure Python."""

from __future__ import annotations

import array
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import chronosieve as cs
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _pure_py_catoni_esr(xs: list[float], mu0: float = 0.0, sigma: float = 1.0) -> float:
    lams = [0.025 * (1.40 / 0.025) ** (i / 7.0) / sigma for i in range(8)]
    lams += [-v for v in lams]
    inv_d = [1.0 / (1.0 + 0.5 * lam * lam * sigma * sigma) for lam in lams]
    acc = [0.0] * 16
    for x in xs:
        d = x - mu0
        for m in range(16):
            y = lams[m] * d
            t = 1.0 + 0.5 * y * y
            psi = (t + y) if y >= 0.0 else (1.0 / (t - y))
            acc[m] = (acc[m] + 1.0) * (psi * inv_d[m])
    return sum(acc) / 16.0


def _pure_py_conformal_pid(yt: list[float], yp: list[float], alpha: float = 0.1, eta: float = 0.05) -> float:
    q = 0.0
    err_sum = 0.0
    for idx, (y, yh) in enumerate(zip(yt, yp, strict=True), start=1):
        err = 1.0 if abs(y - yh) > q else 0.0
        err_sum += err - alpha
        theta = max(-1.5707, min(1.5707, err_sum * math.log(idx + 2.0) / (idx + 1.0)))
        q = max(0.0, eta * err_sum + math.tan(theta))
    return q


def run_benchmark(n_batch: int = 1_000_000, n_scalar: int = 100_000, n_py: int = 20_000) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    xs = np.ascontiguousarray(rng.standard_normal(n_batch), dtype=np.float64)
    ts = np.arange(n_batch, dtype=np.float64)
    ys = np.ascontiguousarray(np.sin(ts * 0.01) + rng.normal(0.0, 0.02, size=n_batch), dtype=np.float64)

    results: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "simd_backend": cs.simd_backend(),
        "n_batch": n_batch,
        "n_scalar": n_scalar,
        "kernels": {},
    }

    # 1. Zero-copy Py_buffer batch: Catoni E-SR (NumPy & stdlib array.array)
    det = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1e12)
    det.ingest_batch(xs[:1000])  # warmup
    det.reset()
    t0 = time.perf_counter_ns()
    det.ingest_batch(xs)
    dt_cat_np = time.perf_counter_ns() - t0
    results["kernels"]["catoni_esr_batch_numpy"] = {
        "ns_per_tick": dt_cat_np / n_batch,
        "m_ticks_per_sec": n_batch * 1e3 / dt_cat_np,
    }

    xs_arr = array.array("d", xs[:200_000])
    det.reset()
    t0 = time.perf_counter_ns()
    det.ingest_batch(xs_arr)
    dt_cat_arr = time.perf_counter_ns() - t0
    results["kernels"]["catoni_esr_batch_stdlib_array"] = {
        "ns_per_tick": dt_cat_arr / len(xs_arr),
        "m_ticks_per_sec": len(xs_arr) * 1e3 / dt_cat_arr,
    }

    # 2. CPython scalar step() dispatch overhead (Catoni E-SR)
    xs_list = xs[:n_scalar].tolist()
    det.reset()
    t0 = time.perf_counter_ns()
    for x in xs_list:
        det.step(x)
    dt_cat_scalar = time.perf_counter_ns() - t0
    results["kernels"]["catoni_esr_cpython_scalar_step"] = {
        "ns_per_tick": dt_cat_scalar / n_scalar,
        "m_ticks_per_sec": n_scalar * 1e3 / dt_cat_scalar,
    }

    # 3. Pure-Python reference Catoni E-SR
    t0 = time.perf_counter_ns()
    _pure_py_catoni_esr(xs_list[:n_py])
    dt_cat_py = time.perf_counter_ns() - t0
    results["kernels"]["catoni_esr_pure_python_ref"] = {
        "ns_per_tick": dt_cat_py / n_py,
        "m_ticks_per_sec": n_py * 1e3 / dt_cat_py,
    }

    # 4. Vovk Rank B-tree zero-copy batch
    n_rank = 200_000
    det_rk = cs.EDetector(family="vovk_rank", max_prefix=65536, threshold=1e12, seed=42)
    t0 = time.perf_counter_ns()
    det_rk.ingest_batch(xs[:n_rank])
    dt_rk = time.perf_counter_ns() - t0
    results["kernels"]["vovk_rank_btree_batch"] = {
        "ns_per_tick": dt_rk / n_rank,
        "m_ticks_per_sec": n_rank * 1e3 / dt_rk,
    }

    # 5. Conformal PID SoA 64-channel NEON batch vs single-channel E-Reset vs Pure Python
    C = 64
    T_soa = n_batch // C
    yt_soa = np.ascontiguousarray(ys[: T_soa * C].reshape(T_soa, C), dtype=np.float64)
    yp_soa = np.ascontiguousarray(xs[: T_soa * C].reshape(T_soa, C), dtype=np.float64)
    soa = cs.ConformalBatchPID(n_channels=C, alpha=0.10, eta=0.05)
    t0 = time.perf_counter_ns()
    soa.ingest_batch(yt_soa, yp_soa)
    dt_soa = time.perf_counter_ns() - t0
    total_ch_ticks = T_soa * C
    results["kernels"]["conformal_pid_soa_64ch_batch"] = {
        "ns_per_ch_tick": dt_soa / total_ch_ticks,
        "m_ch_ticks_per_sec": total_ch_ticks * 1e3 / dt_soa,
    }

    cp = cs.ConformalPID(alpha=0.10, eta=0.05, enable_ereset=True)
    n_cp = 50_000
    t0 = time.perf_counter_ns()
    cp.ingest_batch(ys[:n_cp], xs[:n_cp])
    dt_cp = time.perf_counter_ns() - t0
    results["kernels"]["conformal_pid_ereset_1ch_batch"] = {
        "ns_per_tick": dt_cp / n_cp,
        "m_ticks_per_sec": n_cp * 1e3 / dt_cp,
    }

    t0 = time.perf_counter_ns()
    _pure_py_conformal_pid(ys[:n_py].tolist(), xs[:n_py].tolist())
    dt_cp_py = time.perf_counter_ns() - t0
    results["kernels"]["conformal_pid_pure_python_ref"] = {
        "ns_per_tick": dt_cp_py / n_py,
        "m_ticks_per_sec": n_py * 1e3 / dt_cp_py,
    }

    # 6. Sleeping Fixed-Share BOA + Per-Agent Catoni (K=8 agents)
    K = 8
    T_boa = 100_000
    preds_mat = np.ascontiguousarray(rng.standard_normal((T_boa, K)), dtype=np.float64)
    y_boa = np.ascontiguousarray(ys[:T_boa], dtype=np.float64)
    fus = cs.SleepingBOA(n_agents=K, rogue_mu0=0.8, rogue_sigma=1.0, rogue_threshold=1e12)
    t0 = time.perf_counter_ns()
    fus.fuse_batch(preds_mat, y_boa)
    dt_boa = time.perf_counter_ns() - t0
    results["kernels"]["sleeping_boa_k8_batch"] = {
        "ns_per_step": dt_boa / T_boa,
        "m_steps_per_sec": T_boa * 1e3 / dt_boa,
    }

    # 7. Online COMA (K=8 agents)
    lo = np.linspace(-2.0, 0.0, K, dtype=np.float64)
    hi = lo + 3.0
    n_coma = 50_000
    t0 = time.perf_counter_ns()
    for _ in range(n_coma):
        cs.coma_merge(lo, hi, tau=0.5)
    dt_coma = time.perf_counter_ns() - t0
    results["kernels"]["coma_merge_k8_python_call"] = {
        "ns_per_call": dt_coma / n_coma,
        "m_calls_per_sec": n_coma * 1e3 / dt_coma,
    }

    # 8. Swing PLA + Ring Buffer zero-copy batch
    dig = cs.StreamDigest(epsilon=0.05, ring_capacity=65536)
    t0 = time.perf_counter_ns()
    dig.digest_batch(ts, ys)
    dt_dig = time.perf_counter_ns() - t0
    results["kernels"]["swing_pla_ringbuf_batch"] = {
        "ns_per_tick": dt_dig / n_batch,
        "m_ticks_per_sec": n_batch * 1e3 / dt_dig,
    }

    # 9. Native C11 CLI benchmark (if compiled)
    cli_bin = ROOT / "chronosieve"
    if cli_bin.exists():
        proc = subprocess.run(
            [str(cli_bin), "bench", "--n", "2000000"],
            capture_output=True,
            text=True,
            check=True,
        )
        results["native_cli_bytes"] = cli_bin.stat().st_size
        results["native_cli_bench_stdout"] = proc.stdout.strip()

    return results


def main() -> None:
    res = run_benchmark()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
