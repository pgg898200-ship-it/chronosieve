#!/usr/bin/env python3
"""Fair, reproducible head-to-head benchmark against compiled incumbents (river, ruptures).

Measures BOTH:
  1. Median-of-5 throughput (`ns/tick`, `M ticks/s`) — separating zero-copy Py_buffer batch
     from per-tick CPython scalar `step()` (same Python loop cadence as `river`).
  2. Non-asymptotic statistical validity under heavy tails (Student-t_3 and Cauchy(0,1))
     vs. post-shift detection delay (+2.0 mean shift on Student-t_3).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chronosieve as cs
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent


def _try_import(mod_name: str) -> Any | None:
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        return None


def _pkg_version(pkg_name: str) -> str | None:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _median_ns(fn: Callable[[], Any], *, n_repeats: int = 5, warmup: bool = True) -> float:
    if warmup:
        fn()
    samples: list[int] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return float(statistics.median(samples))


def _eval_river_null_and_shift(
    factory: Callable[[], Any],
    t3_null: np.ndarray,
    cauchy_null: np.ndarray,
    t3_shift: np.ndarray,
    cp: int,
) -> dict[str, Any]:
    def _run_null(mat: np.ndarray) -> dict[str, Any]:
        n_trials, horizon = mat.shape
        stops: list[int] = []
        n_fa = 0
        for tr in range(n_trials):
            det = factory()
            hit_t = horizon
            for t, val in enumerate(mat[tr].tolist(), start=1):
                det.update(val)
                if det.drift_detected:
                    hit_t = t
                    n_fa += 1
                    break
            stops.append(hit_t)
        return {
            "n_trials": n_trials,
            "horizon": horizon,
            "false_alarm_rate": n_fa / n_trials,
            "empirical_arl_censored": float(np.mean(stops)),
        }

    def _run_shift(mat: np.ndarray) -> dict[str, Any]:
        n_trials, horizon = mat.shape
        pre_fa = 0
        delays: list[int] = []
        detected_post = 0
        for tr in range(n_trials):
            det = factory()
            early = False
            for t, val in enumerate(mat[tr].tolist()):
                det.update(val)
                if det.drift_detected:
                    if t < cp:
                        early = True
                        pre_fa += 1
                        break
                    detected_post += 1
                    delays.append(t - cp + 1)
                    break
            if early:
                continue
        n_clean = n_trials - pre_fa
        return {
            "n_trials": n_trials,
            "changepoint": cp,
            "pre_shift_false_alarm_rate": pre_fa / n_trials,
            "post_shift_detection_rate_given_clean": (detected_post / n_clean) if n_clean else 0.0,
            "mean_delay_ticks": float(np.mean(delays)) if delays else None,
            "median_delay_ticks": float(np.median(delays)) if delays else None,
        }

    return {
        "null_student_t3": _run_null(t3_null),
        "null_cauchy": _run_null(cauchy_null),
        "shift_student_t3_plus2": _run_shift(t3_shift),
    }


def _eval_cs_null_and_shift(
    factory: Callable[[int], cs.EDetector],
    t3_null: np.ndarray,
    cauchy_null: np.ndarray,
    t3_shift: np.ndarray,
    cp: int,
    *,
    run_cauchy: bool = True,
) -> dict[str, Any]:
    def _run_null(mat: np.ndarray) -> dict[str, Any]:
        n_trials, horizon = mat.shape
        stops: list[int] = []
        n_fa = 0
        alarms = np.empty(horizon, dtype=np.uint8)
        for tr in range(n_trials):
            det = factory(tr)
            det.ingest_batch(mat[tr], out_alarms=alarms)
            hits = np.flatnonzero(alarms)
            if hits.size > 0:
                stops.append(int(hits[0]) + 1)
                n_fa += 1
            else:
                stops.append(horizon)
        return {
            "n_trials": n_trials,
            "horizon": horizon,
            "false_alarm_rate": n_fa / n_trials,
            "empirical_arl_censored": float(np.mean(stops)),
        }

    def _run_shift(mat: np.ndarray) -> dict[str, Any]:
        n_trials, horizon = mat.shape
        pre_fa = 0
        delays: list[int] = []
        detected_post = 0
        alarms = np.empty(horizon, dtype=np.uint8)
        for tr in range(n_trials):
            det = factory(10_000 + tr)
            det.ingest_batch(mat[tr], out_alarms=alarms)
            hits = np.flatnonzero(alarms)
            if hits.size == 0:
                continue
            first = int(hits[0])
            if first < cp:
                pre_fa += 1
            else:
                detected_post += 1
                delays.append(first - cp + 1)
        n_clean = n_trials - pre_fa
        return {
            "n_trials": n_trials,
            "changepoint": cp,
            "pre_shift_false_alarm_rate": pre_fa / n_trials,
            "post_shift_detection_rate_given_clean": (detected_post / n_clean) if n_clean else 0.0,
            "mean_delay_ticks": float(np.mean(delays)) if delays else None,
            "median_delay_ticks": float(np.median(delays)) if delays else None,
        }

    out: dict[str, Any] = {
        "null_student_t3": _run_null(t3_null),
        "shift_student_t3_plus2": _run_shift(t3_shift),
    }
    if run_cauchy:
        out["null_cauchy"] = _run_null(cauchy_null)
    return out


def run_external_benchmark(n_ticks: int = 100_000, n_repeats: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    xs = np.ascontiguousarray(rng.standard_normal(n_ticks), dtype=np.float64)
    us = np.ascontiguousarray(rng.uniform(0.0, 1.0, size=n_ticks), dtype=np.float64)
    xs_list = xs.tolist()
    us_list = us.tolist()

    report: dict[str, Any] = {
        "chronosieve_version": cs.__version__,
        "chronosieve_simd": cs.simd_backend(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "n_ticks": n_ticks,
        "timing_method": f"median_of_{n_repeats}",
        "throughput": {},
        "statistical_validity": {},
    }

    # --- 1. ChronoSieve Throughput (zero-copy Py_buffer batch & CPython scalar step) ---
    det_bnd = cs.EDetector(family="bounded", mu0=0.5, threshold=1e12)
    dt_bnd_b = _median_ns(lambda: (det_bnd.reset(), det_bnd.ingest_batch(us)), n_repeats=n_repeats)

    det_cat = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1e12)
    dt_cat_b = _median_ns(lambda: (det_cat.reset(), det_cat.ingest_batch(xs)), n_repeats=n_repeats)

    det_sym = cs.EDetector(family="symmetry", mu0=0.0, sigma=1.0, threshold=1e12)
    dt_sym_b = _median_ns(lambda: (det_sym.reset(), det_sym.ingest_batch(xs)), n_repeats=n_repeats)

    det_rk = cs.EDetector(family="vovk_rank", max_prefix=65536, threshold=1e12, seed=42)
    dt_rk_b = _median_ns(lambda: (det_rk.reset(), det_rk.ingest_batch(xs)), n_repeats=n_repeats)

    def _cat_scalar() -> None:
        det_cat.reset()
        for x in xs_list:
            det_cat.step(x)

    def _bnd_scalar() -> None:
        det_bnd.reset()
        for u in us_list:
            det_bnd.step(u)

    dt_cat_s = _median_ns(_cat_scalar, n_repeats=n_repeats)
    dt_bnd_s = _median_ns(_bnd_scalar, n_repeats=n_repeats)

    report["throughput"]["chronosieve_bounded_batch"] = {
        "ns_per_tick": dt_bnd_b / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_bnd_b,
    }
    report["throughput"]["chronosieve_catoni_batch"] = {
        "ns_per_tick": dt_cat_b / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_cat_b,
    }
    report["throughput"]["chronosieve_symmetry_batch"] = {
        "ns_per_tick": dt_sym_b / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_sym_b,
    }
    report["throughput"]["chronosieve_vovk_rank_batch"] = {
        "ns_per_tick": dt_rk_b / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_rk_b,
    }
    report["throughput"]["chronosieve_catoni_scalar_step"] = {
        "ns_per_tick": dt_cat_s / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_cat_s,
    }
    report["throughput"]["chronosieve_bounded_scalar_step"] = {
        "ns_per_tick": dt_bnd_s / n_ticks,
        "m_ticks_per_sec": n_ticks * 1e3 / dt_bnd_s,
    }

    # --- 2. Optional: Rust-backed river (ADWIN, PageHinkley, KSWIN) ---
    river_drift = _try_import("river.drift")
    if river_drift is not None:
        river_ver = _pkg_version("river")

        def _run_adwin() -> None:
            d = river_drift.ADWIN(delta=0.002)
            for x in xs_list:
                d.update(x)

        def _run_ph() -> None:
            d = river_drift.PageHinkley()
            for x in xs_list:
                d.update(x)

        n_kswin = min(20_000, n_ticks)
        xs_ks = xs_list[:n_kswin]

        def _run_kswin() -> None:
            d = river_drift.KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=42)
            for x in xs_ks:
                d.update(x)

        dt_adwin = _median_ns(_run_adwin, n_repeats=n_repeats)
        dt_ph = _median_ns(_run_ph, n_repeats=n_repeats)
        dt_ks = _median_ns(_run_kswin, n_repeats=3)

        adwin_ns = dt_adwin / n_ticks
        ph_ns = dt_ph / n_ticks
        ks_ns = dt_ks / n_kswin
        cat_b_ns = dt_cat_b / n_ticks
        cat_s_ns = dt_cat_s / n_ticks
        rk_b_ns = dt_rk_b / n_ticks

        report["throughput"]["river_adwin"] = {
            "installed": True,
            "version": river_ver,
            "ns_per_tick": adwin_ns,
            "m_ticks_per_sec": 1e3 / adwin_ns,
            "speedup_chronosieve_catoni_batch": adwin_ns / cat_b_ns,
            "speedup_chronosieve_catoni_scalar": adwin_ns / cat_s_ns,
        }
        report["throughput"]["river_page_hinkley"] = {
            "installed": True,
            "version": river_ver,
            "ns_per_tick": ph_ns,
            "m_ticks_per_sec": 1e3 / ph_ns,
            "speedup_chronosieve_catoni_batch": ph_ns / cat_b_ns,
            "speedup_chronosieve_catoni_scalar": ph_ns / cat_s_ns,
        }
        report["throughput"]["river_kswin"] = {
            "installed": True,
            "version": river_ver,
            "n_ticks": n_kswin,
            "ns_per_tick": ks_ns,
            "m_ticks_per_sec": 1e3 / ks_ns,
            "speedup_chronosieve_vovk_rank_batch": ks_ns / rk_b_ns,
            "speedup_chronosieve_catoni_batch": ks_ns / cat_b_ns,
        }
    else:
        report["throughput"]["river_adwin"] = {"installed": False, "install_hint": "pip install river"}

    # --- 3. Optional: ruptures (offline changepoint Pelt l2 & rbf) ---
    ruptures = _try_import("ruptures")
    if ruptures is not None:
        rup_ver = _pkg_version("ruptures")
        xs_2k = np.ascontiguousarray(xs[:2000], dtype=np.float64)

        def _run_pelt_l2() -> None:
            ruptures.Pelt(model="l2").fit(xs_2k).predict(pen=10.0)

        def _run_pelt_rbf() -> None:
            ruptures.Pelt(model="rbf").fit(xs_2k).predict(pen=10.0)

        dt_l2 = _median_ns(_run_pelt_l2, n_repeats=5)
        dt_rbf = _median_ns(_run_pelt_rbf, n_repeats=3)
        report["throughput"]["ruptures_pelt_l2_2k_offline"] = {
            "installed": True,
            "version": rup_ver,
            "n_ticks": 2000,
            "ms_total": dt_l2 / 1e6,
            "ns_per_tick": dt_l2 / 2000,
        }
        report["throughput"]["ruptures_pelt_rbf_2k_offline"] = {
            "installed": True,
            "version": rup_ver,
            "n_ticks": 2000,
            "ms_total": dt_rbf / 1e6,
            "ns_per_tick": dt_rbf / 2000,
        }
    else:
        report["throughput"]["ruptures_pelt_rbf_2k_offline"] = {
            "installed": False,
            "install_hint": "pip install ruptures",
        }

    # --- 4. Heavy-Tail Statistical Validity & Shift Power (Student-t_3 & Cauchy(0, 1)) ---
    rng_stat = np.random.default_rng(20260222)
    n_trials = 60
    horizon = 1000
    shift_horizon = 600
    cp = 200

    t3_null = np.ascontiguousarray(rng_stat.standard_t(df=3.0, size=(n_trials, horizon)), dtype=np.float64)
    cauchy_null = np.ascontiguousarray(
        rng_stat.standard_cauchy(size=(n_trials, horizon)), dtype=np.float64
    )
    t3_shift = np.ascontiguousarray(
        rng_stat.standard_t(df=3.0, size=(n_trials, shift_horizon)), dtype=np.float64
    )
    t3_shift[:, cp:] += 2.0

    # Student-t_3 has true mean 0, variance 3 -> sigma = sqrt(3)
    sigma_t3 = math.sqrt(3.0)
    gamma = 1000.0
    report["statistical_validity"]["chronosieve_catoni_gamma1000"] = _eval_cs_null_and_shift(
        lambda tr: cs.EDetector(family="catoni", mu0=0.0, sigma=sigma_t3, threshold=gamma),
        t3_null,
        cauchy_null,
        t3_shift,
        cp,
        run_cauchy=False,  # Catoni requires finite variance (Cauchy has infinite variance)
    )
    report["statistical_validity"]["chronosieve_symmetry_gamma1000"] = _eval_cs_null_and_shift(
        lambda tr: cs.EDetector(family="symmetry", mu0=0.0, sigma=1.0, threshold=gamma),
        t3_null,
        cauchy_null,
        t3_shift,
        cp,
        run_cauchy=True,
    )
    report["statistical_validity"]["chronosieve_vovk_rank_gamma1000"] = _eval_cs_null_and_shift(
        lambda tr: cs.EDetector(
            family="vovk_rank", max_prefix=2048, threshold=gamma, seed=7000 + tr
        ),
        t3_null,
        cauchy_null,
        t3_shift,
        cp,
        run_cauchy=True,
    )

    if river_drift is not None:
        report["statistical_validity"]["river_adwin_delta002"] = _eval_river_null_and_shift(
            lambda: river_drift.ADWIN(delta=0.002), t3_null, cauchy_null, t3_shift, cp
        )
        report["statistical_validity"]["river_page_hinkley_default"] = _eval_river_null_and_shift(
            lambda: river_drift.PageHinkley(), t3_null, cauchy_null, t3_shift, cp
        )
        report["statistical_validity"]["river_kswin_alpha005"] = _eval_river_null_and_shift(
            lambda: river_drift.KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=42),
            t3_null,
            cauchy_null,
            t3_shift,
            cp,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ChronoSieve external competitor benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path (e.g. benchmarks/results_external_m4.json)",
    )
    parser.add_argument("--n", type=int, default=100_000, help="Ticks for throughput benchmarks")
    args = parser.parse_args()

    report = run_external_benchmark(n_ticks=args.n)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
