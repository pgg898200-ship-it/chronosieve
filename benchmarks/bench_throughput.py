#!/usr/bin/env python3
"""Reproducible median-of-5 throughput benchmark: Native C11, Py_buffer, scalar, & Pure Python."""

from __future__ import annotations

import array
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chronosieve as cs
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _median_ns(fn: Callable[[], Any], *, n_repeats: int = 5, warmup: bool = True) -> float:
    if warmup:
        fn()
    samples: list[int] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return float(statistics.median(samples))


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


def run_benchmark(
    n_batch: int = 1_000_000,
    n_scalar: int = 100_000,
    n_py: int = 20_000,
    n_repeats: int = 5,
) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    xs = np.ascontiguousarray(rng.standard_normal(n_batch), dtype=np.float64)
    us = np.ascontiguousarray(rng.uniform(0.0, 1.0, size=n_batch), dtype=np.float64)
    ts = np.arange(n_batch, dtype=np.float64)
    ys = np.ascontiguousarray(np.sin(ts * 0.01) + rng.normal(0.0, 0.02, size=n_batch), dtype=np.float64)

    results: dict[str, Any] = {
        "version": cs.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "simd_backend": cs.simd_backend(),
        "timing_method": f"median_of_{n_repeats}",
        "n_batch": n_batch,
        "n_scalar": n_scalar,
        "kernels": {},
    }

    # 1. Zero-copy Py_buffer batch E-Detectors: Bounded, Catoni, Symmetry (NumPy & stdlib array)
    det_bnd = cs.EDetector(family="bounded", mu0=0.5, threshold=1e12)
    dt_bnd = _median_ns(lambda: (det_bnd.reset(), det_bnd.ingest_batch(us)), n_repeats=n_repeats)
    results["kernels"]["bounded_esr_batch_numpy"] = {
        "ns_per_tick": dt_bnd / n_batch,
        "m_ticks_per_sec": n_batch * 1e3 / dt_bnd,
    }

    det = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1e12)
    dt_cat_np = _median_ns(lambda: (det.reset(), det.ingest_batch(xs)), n_repeats=n_repeats)
    results["kernels"]["catoni_esr_batch_numpy"] = {
        "ns_per_tick": dt_cat_np / n_batch,
        "m_ticks_per_sec": n_batch * 1e3 / dt_cat_np,
    }

    det_sym = cs.EDetector(family="symmetry", mu0=0.0, sigma=1.0, threshold=1e12)
    dt_sym = _median_ns(lambda: (det_sym.reset(), det_sym.ingest_batch(xs)), n_repeats=n_repeats)
    results["kernels"]["symmetry_esr_batch_numpy"] = {
        "ns_per_tick": dt_sym / n_batch,
        "m_ticks_per_sec": n_batch * 1e3 / dt_sym,
    }

    xs_arr = array.array("d", xs[:200_000])
    dt_cat_arr = _median_ns(lambda: (det.reset(), det.ingest_batch(xs_arr)), n_repeats=n_repeats)
    results["kernels"]["catoni_esr_batch_stdlib_array"] = {
        "ns_per_tick": dt_cat_arr / len(xs_arr),
        "m_ticks_per_sec": len(xs_arr) * 1e3 / dt_cat_arr,
    }

    # 2. CPython scalar step() dispatch overhead (Catoni E-SR)
    xs_list = xs[:n_scalar].tolist()

    def _run_scalar() -> None:
        det.reset()
        for x in xs_list:
            det.step(x)

    dt_cat_scalar = _median_ns(_run_scalar, n_repeats=n_repeats)
    results["kernels"]["catoni_esr_cpython_scalar_step"] = {
        "ns_per_tick": dt_cat_scalar / n_scalar,
        "m_ticks_per_sec": n_scalar * 1e3 / dt_cat_scalar,
    }

    # 3. Pure-Python reference Catoni E-SR
    xs_py = xs_list[:n_py]
    dt_cat_py = _median_ns(lambda: _pure_py_catoni_esr(xs_py), n_repeats=n_repeats)
    results["kernels"]["catoni_esr_pure_python_ref"] = {
        "ns_per_tick": dt_cat_py / n_py,
        "m_ticks_per_sec": n_py * 1e3 / dt_cat_py,
    }

    # 4. Vovk Rank B-tree zero-copy batch
    n_rank = 200_000
    xs_rank = xs[:n_rank]
    det_rk = cs.EDetector(family="vovk_rank", max_prefix=65536, threshold=1e12, seed=42)
    dt_rk = _median_ns(lambda: (det_rk.reset(), det_rk.ingest_batch(xs_rank)), n_repeats=n_repeats)
    results["kernels"]["vovk_rank_btree_batch"] = {
        "ns_per_tick": dt_rk / n_rank,
        "m_ticks_per_sec": n_rank * 1e3 / dt_rk,
    }

    # 5. Conformal PID SoA 64-channel SIMD batch vs single-channel E-Reset vs Pure Python
    C = 64
    T_soa = n_batch // C
    yt_soa = np.ascontiguousarray(ys[: T_soa * C].reshape(T_soa, C), dtype=np.float64)
    yp_soa = np.ascontiguousarray(xs[: T_soa * C].reshape(T_soa, C), dtype=np.float64)
    soa = cs.ConformalBatchPID(n_channels=C, alpha=0.10, eta=0.05)
    dt_soa = _median_ns(lambda: (soa.reset(), soa.ingest_batch(yt_soa, yp_soa)), n_repeats=n_repeats)
    total_ch_ticks = T_soa * C
    results["kernels"]["conformal_pid_soa_64ch_batch"] = {
        "ns_per_ch_tick": dt_soa / total_ch_ticks,
        "m_ch_ticks_per_sec": total_ch_ticks * 1e3 / dt_soa,
    }

    cp = cs.ConformalPID(alpha=0.10, eta=0.05, enable_ereset=True)
    n_cp = 50_000
    ys_cp, xs_cp = ys[:n_cp], xs[:n_cp]
    dt_cp = _median_ns(lambda: (cp.reset(), cp.ingest_batch(ys_cp, xs_cp)), n_repeats=n_repeats)
    results["kernels"]["conformal_pid_ereset_1ch_batch"] = {
        "ns_per_tick": dt_cp / n_cp,
        "m_ticks_per_sec": n_cp * 1e3 / dt_cp,
    }

    ys_py_list = ys[:n_py].tolist()
    xs_py_list = xs[:n_py].tolist()
    dt_cp_py = _median_ns(lambda: _pure_py_conformal_pid(ys_py_list, xs_py_list), n_repeats=n_repeats)
    results["kernels"]["conformal_pid_pure_python_ref"] = {
        "ns_per_tick": dt_cp_py / n_py,
        "m_ticks_per_sec": n_py * 1e3 / dt_cp_py,
    }

    # 6. Sleeping Fixed-Share BOA + Peer-Relative Per-Agent Catoni (K=8 agents)
    K = 8
    T_boa = 100_000
    preds_mat = np.ascontiguousarray(rng.standard_normal((T_boa, K)), dtype=np.float64)
    y_boa = np.ascontiguousarray(ys[:T_boa], dtype=np.float64)
    fus = cs.SleepingBOA(n_agents=K, rogue_mu0=0.8, rogue_sigma=1.0, rogue_threshold=1e12)
    dt_boa = _median_ns(lambda: (fus.reset(), fus.fuse_batch(preds_mat, y_boa)), n_repeats=n_repeats)
    results["kernels"]["sleeping_boa_k8_batch"] = {
        "ns_per_step": dt_boa / T_boa,
        "m_steps_per_sec": T_boa * 1e3 / dt_boa,
    }

    # 7. Online COMA (K=8 agents): zero-copy batch vs Python single-call
    T_coma = 100_000
    lo_mat = np.ascontiguousarray(rng.normal(0.0, 1.0, size=(T_coma, K)) - 1.5, dtype=np.float64)
    hi_mat = np.ascontiguousarray(lo_mat + 3.0, dtype=np.float64)
    out_hlo = np.empty(T_coma, dtype=np.float64)
    out_hhi = np.empty(T_coma, dtype=np.float64)
    dt_coma_batch = _median_ns(
        lambda: cs.coma_merge_batch(lo_mat, hi_mat, tau=0.5, out_hull_lo=out_hlo, out_hull_hi=out_hhi),
        n_repeats=n_repeats,
    )
    results["kernels"]["coma_merge_batch_k8_numpy"] = {
        "ns_per_step": dt_coma_batch / T_coma,
        "m_steps_per_sec": T_coma * 1e3 / dt_coma_batch,
    }

    lo = np.linspace(-2.0, 0.0, K, dtype=np.float64)
    hi = lo + 3.0
    n_coma = 50_000

    def _run_coma_py() -> None:
        for _ in range(n_coma):
            cs.coma_merge(lo, hi, tau=0.5)

    dt_coma = _median_ns(_run_coma_py, n_repeats=n_repeats)
    results["kernels"]["coma_merge_k8_python_call"] = {
        "ns_per_call": dt_coma / n_coma,
        "m_calls_per_sec": n_coma * 1e3 / dt_coma,
    }

    # 8. Swing PLA + Ring Buffer zero-copy batch
    dig = cs.StreamDigest(epsilon=0.05, ring_capacity=65536)
    dt_dig = _median_ns(lambda: (dig.reset(), dig.digest_batch(ts, ys)), n_repeats=n_repeats)
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
