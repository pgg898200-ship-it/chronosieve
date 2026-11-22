#!/usr/bin/env python3
"""Master reproducible benchmark runner for ChronoSieve."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import bench_llm_eval  # noqa: E402
import bench_math_proofs  # noqa: E402
import bench_throughput  # noqa: E402
import bench_tokens  # noqa: E402


def main() -> None:
    print("[1/4] Running throughput benchmarks (Native C11, Py_buffer batch, CPython scalar, Pure Python)...")
    tp = bench_throughput.run_benchmark()

    print("[2/4] Running Monte Carlo mathematical verification proofs...")
    mp = bench_math_proofs.run_math_proofs()

    print("[3/4] Running LLM token reduction benchmarks (JSON vs CSV vs TSLN vs ChronoSieve)...")
    tk = bench_tokens.run_token_benchmark()

    print("[4/4] Running offline LLM eval scenario dry-run...")
    ev = bench_llm_eval.build_eval_scenario()

    edet_matches = re.findall(r"E=([0-9.]+)/([0-9.]+)", ev["digest_markdown"])
    digest_shows_shift = any(float(s) >= float(thr) for s, thr in edet_matches)

    llm_eval_bundle = {
        "ground_truth": ev["ground_truth"],
        "wake_count": ev["wake_count"],
        "wake_ticks": ev["wake_ticks"],
        "shift_ticks": ev["shift_ticks"],
        "rogue_flags": ev["rogue_flags"],
        "prompt_chars": ev["prompt_chars"],
        "prompt_est_tokens": ev["prompt_est_tokens"],
        "prompt_bpe_proxy_tokens": ev["prompt_bpe_proxy_tokens"],
        "raw_json_chars_div_4": ev["raw_json_chars_div_4"],
        "raw_json_bpe_proxy_tokens": ev["raw_json_bpe_proxy_tokens"],
        "prompt_vs_raw_json_bpe_reduction_pct": ev["prompt_vs_raw_json_bpe_reduction_pct"],
        "prompt_vs_raw_json_chars4_reduction_pct": ev["prompt_vs_raw_json_chars4_reduction_pct"],
        "digest_captures_ground_truth": (
            "t=180.00" in ev["digest_markdown"]
            and "95.5" in ev["digest_markdown"]
            and "180.00..180.00: 95.5->95.5" in ev["digest_markdown"]
            and "rogue=agent_2" in ev["digest_markdown"]
            and digest_shows_shift
            and any(150 <= s <= 170 for s in ev["shift_ticks"])
        ),
    }
    for opt_k in ("prompt_tiktoken", "raw_json_tiktoken", "prompt_vs_raw_json_o200k_reduction_pct"):
        if opt_k in ev:
            llm_eval_bundle[opt_k] = ev[opt_k]

    bundle = {
        "throughput": tp,
        "math_proofs": mp,
        "tokens": tk,
        "llm_eval_dry_run": llm_eval_bundle,
    }

    out_path = BENCH_DIR / "results_m4.json"
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved reproducible benchmark results to {out_path}")
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
