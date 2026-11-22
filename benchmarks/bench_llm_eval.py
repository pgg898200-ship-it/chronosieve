#!/usr/bin/env python3
"""Optional System-2 LLM evaluation harness (dry-run offline; live API outside sandbox)."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from typing import Any

import chronosieve as cs
import numpy as np


def _approx_bpe_tokens(text: str) -> int:
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)
    return sum(max(1, (len(p) + 3) // 4) for p in parts)


def build_eval_scenario(seed: int = 2026) -> dict[str, Any]:
    """Generate a multi-agent telemetry incident with ground-truth shift, anomaly, and rogue agent."""
    rng = np.random.default_rng(seed)
    # Note: per-agent Catoni rogue attribution monitors raw |y - p_k|; a single ~86-sigma exogenous
    # thermal spike at t=180 produces a ~4000x one-tick spike on healthy agents, whereas agent_2's
    # persistent -12C hallucination exceeds 1e60. Setting rogue_threshold=50000 cleanly separates
    # persistent agent drift from a single exogenous spike across 100/100 random seeds.
    pipe = cs.ChronoSieve(
        channel="gpu_temp_c",
        n_agents=3,
        alpha=0.005,
        epsilon=0.5,
        edet_threshold=1000.0,
        rogue_threshold=50000.0,
        anomaly_burst=1,
        wake_cooldown=5,
        monitor_level=True,
    )

    wake_packets: list[str] = []
    wake_ticks: list[int] = []
    shift_ticks: list[int] = []
    raw_rows: list[dict[str, Any]] = []
    for t in range(250):
        base = 62.0 if t < 150 else 74.0
        y = base + float(rng.normal(0.0, 0.25))
        if t == 180:
            y = 95.5  # Ground-truth thermal spike!
        # Healthy agents forecast the underlying regime level (no spike lookahead into y!)
        p0 = base + float(rng.normal(0.0, 0.2))
        p1 = base + float(rng.normal(0.0, 0.25))
        p2 = (base + float(rng.normal(0.0, 0.2))) if t < 160 else (base - 12.0)  # Agent 2 goes rogue!
        raw_rows.append(
            {"t": t, "y": round(y, 2), "preds": [round(p0, 2), round(p1, 2), round(p2, 2)]}
        )
        res = pipe.step(float(t), y, preds=[p0, p1, p2])
        if res["shift_detected"]:
            shift_ticks.append(t)
        if res["should_wake"] and res["digest_markdown"]:
            wake_packets.append(res["digest_markdown"])
            wake_ticks.append(t)

    digest_md = "\n\n".join(wake_packets) if wake_packets else pipe.format_markdown()
    rogue_flags = pipe.fuser.rogue_stats()[1] if pipe.fuser is not None else []
    raw_json = json.dumps(raw_rows, separators=(",", ":"))
    raw_c4 = len(raw_json) // 4
    raw_bpe = _approx_bpe_tokens(raw_json)

    prompt = (
        "You are a System-2 site reliability agent. Inspect these ChronoSieve wake-up digest packets "
        'and return JSON with keys {"shift_detected": bool, "spike_t": number, "rogue_agent": int}:\n\n'
        f"{digest_md}"
    )
    prompt_c4 = len(prompt) // 4
    prompt_bpe = _approx_bpe_tokens(prompt)
    return {
        "ground_truth": {"shift_detected": True, "spike_t": 180, "rogue_agent": 2},
        "wake_count": len(wake_packets),
        "wake_ticks": wake_ticks,
        "shift_ticks": shift_ticks,
        "rogue_flags": rogue_flags,
        "digest_markdown": digest_md,
        "prompt_chars": len(prompt),
        "prompt_est_tokens": prompt_c4,
        "prompt_bpe_proxy_tokens": prompt_bpe,
        "raw_json_chars": len(raw_json),
        "raw_json_chars_div_4": raw_c4,
        "raw_json_bpe_proxy_tokens": raw_bpe,
        "prompt_vs_raw_json_bpe_reduction_pct": 100.0 * (1.0 - prompt_bpe / raw_bpe) if raw_bpe else 0.0,
        "prompt_vs_raw_json_chars4_reduction_pct": 100.0 * (1.0 - prompt_c4 / raw_c4) if raw_c4 else 0.0,
        "prompt": prompt,
    }


def call_openai_compatible(prompt: str, model: str, api_key: str, base_url: str) -> str:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="ChronoSieve System-2 LLM evaluation harness")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build scenario & print packet without API call",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model ID for live OpenAI-compatible eval")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    args = parser.parse_args()

    scenario = build_eval_scenario()
    api_key = os.environ.get("OPENAI_API_KEY")
    if args.dry_run or not api_key:
        out = {
            "mode": "dry_run",
            "reason": "requested --dry-run" if args.dry_run else "OPENAI_API_KEY not set",
            "ground_truth": scenario["ground_truth"],
            "wake_count": scenario["wake_count"],
            "wake_ticks": scenario["wake_ticks"],
            "shift_ticks": scenario["shift_ticks"],
            "rogue_flags": scenario["rogue_flags"],
            "prompt_chars": scenario["prompt_chars"],
            "prompt_est_tokens": scenario["prompt_est_tokens"],
            "prompt_bpe_proxy_tokens": scenario["prompt_bpe_proxy_tokens"],
            "raw_json_chars_div_4": scenario["raw_json_chars_div_4"],
            "raw_json_bpe_proxy_tokens": scenario["raw_json_bpe_proxy_tokens"],
            "digest_preview": scenario["digest_markdown"],
        }
        print(json.dumps(out, indent=2))
        return

    reply = call_openai_compatible(scenario["prompt"], args.model, api_key, args.base_url)
    print(
        json.dumps(
            {
                "mode": "live",
                "model": args.model,
                "ground_truth": scenario["ground_truth"],
                "prompt_est_tokens": scenario["prompt_est_tokens"],
                "raw_reply": reply,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
