#!/usr/bin/env python3
"""Optional external competitor comparison harness (runs outside offline sandbox if libs installed).

Fairly credits compiled incumbents:
  - river (Rust-backed streaming drift detectors: ADWIN, PageHinkley, KSWIN)
  - conformal-prediction / risksieve (Rust conformal & online risk libraries)
  - stcpR6 (Rcpp sequential testing / e-detectors)
  - TerseTS (Zig piecewise linear time-series compression)
  - tsln-python (Time Series Language Notation for LLM serialization)
"""

from __future__ import annotations

import importlib
import json
import time
from typing import Any

import chronosieve as cs
import numpy as np


def _try_import(mod_name: str) -> Any | None:
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        return None


def main() -> None:
    rng = np.random.default_rng(2026)
    N = 100_000
    xs = np.ascontiguousarray(rng.standard_normal(N), dtype=np.float64)
    xs_list = xs.tolist()

    report: dict[str, Any] = {"n_ticks": N, "chronosieve_simd": cs.simd_backend(), "comparisons": {}}

    # ChronoSieve baselines
    det = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1e12)
    t0 = time.perf_counter_ns()
    det.ingest_batch(xs)
    dt_cs_batch = time.perf_counter_ns() - t0

    det.reset()
    t0 = time.perf_counter_ns()
    for x in xs_list:
        det.step(x)
    dt_cs_scalar = time.perf_counter_ns() - t0

    report["comparisons"]["chronosieve_catoni_batch"] = {"ns_per_tick": dt_cs_batch / N}
    report["comparisons"]["chronosieve_catoni_scalar"] = {"ns_per_tick": dt_cs_scalar / N}

    # Optional: river (Rust-backed ADWIN / PageHinkley)
    river_drift = _try_import("river.drift")
    if river_drift is not None:
        adwin = river_drift.ADWIN()
        t0 = time.perf_counter_ns()
        for x in xs_list:
            adwin.update(x)
        dt_adwin = time.perf_counter_ns() - t0
        report["comparisons"]["river_adwin"] = {
            "installed": True,
            "ns_per_tick": dt_adwin / N,
            "speedup_vs_batch": dt_adwin / dt_cs_batch,
        }
    else:
        report["comparisons"]["river_adwin"] = {
            "installed": False,
            "install_hint": "pip install river",
        }

    # Optional: tsln
    tsln = _try_import("tsln")
    if tsln is not None:
        report["comparisons"]["tsln_python"] = {"installed": True}
    else:
        report["comparisons"]["tsln_python"] = {
            "installed": False,
            "install_hint": "pip install tsln-python",
        }

    # Optional: ruptures (offline changepoint reference)
    ruptures = _try_import("ruptures")
    if ruptures is not None:
        algo = ruptures.Pelt(model="rbf").fit(xs[:2000])
        t0 = time.perf_counter_ns()
        algo.predict(pen=10)
        dt_rup = time.perf_counter_ns() - t0
        report["comparisons"]["ruptures_pelt_2k_offline"] = {"installed": True, "ms_total": dt_rup / 1e6}
    else:
        report["comparisons"]["ruptures_pelt_2k_offline"] = {
            "installed": False,
            "install_hint": "pip install ruptures",
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
