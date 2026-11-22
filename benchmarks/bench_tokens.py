#!/usr/bin/env python3
"""Reproducible LLM token reduction benchmark: Raw JSON vs CSV vs TSLN-style vs ChronoSieve."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any

import chronosieve as cs
import numpy as np


def approx_bpe_tokens(text: str) -> int:
    """Deterministic BPE-proxy token counter (subword chunks + punctuation, close to cl100k_base)."""
    if not text:
        return 0
    # Split into runs of alnum (chunked at <=4 chars) or single punctuation/symbols
    parts = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)
    total = 0
    for p in parts:
        total += max(1, (len(p) + 3) // 4)
    return total


def encode_tsln_style(ys: np.ndarray, window: int = 250) -> str:
    """Faithful TSLN-style compact delta-encoded text representation (per-window, no wake-up gate)."""
    chunks = []
    for start in range(0, len(ys), window):
        seg = ys[start : start + window]
        vals = [f"{seg[0]:.2f}"]
        for i in range(1, len(seg)):
            d = seg[i] - seg[i - 1]
            if abs(d) < 0.005:
                vals.append("=")
            else:
                vals.append(f"{d:+.2f}")
        chunks.append(f"W{start}:{','.join(vals)}")
    return "\n".join(chunks)


def _pct_reduction(cand: int, base: int) -> float:
    return 100.0 * (1.0 - (cand / base)) if base > 0 else 0.0


def _reconstruct_from_markdown_linf(pla_md: str, ys: np.ndarray) -> float:
    """Parse rendered Markdown segment lines and measure worst-case endpoint-interpolated L_inf error."""
    seg_re = re.compile(
        r"^- ([0-9.]+)\.\.([0-9.]+): ([-0-9.eE+]+)->([-0-9.eE+]+) m=([-+0-9.eE+]+) n=(\d+)"
    )
    covered = np.zeros(len(ys), dtype=np.bool_)
    max_err = 0.0
    for ln in pla_md.splitlines():
        m = seg_re.match(ln)
        if not m:
            continue
        i0 = int(round(float(m.group(1))))
        i1 = int(round(float(m.group(2))))
        y0 = float(m.group(3))
        y1 = float(m.group(4))
        for i in range(i0, i1 + 1):
            y_hat = y0 + (y1 - y0) * ((i - i0) / (i1 - i0)) if i1 > i0 else y0
            err = abs(y_hat - float(ys[i]))
            if err > max_err:
                max_err = err
            covered[i] = True
    if not bool(np.all(covered)):
        return float("inf")
    return float(max_err)


def run_mauna_loa_co2_benchmark(epsilon: float = 1.0) -> dict[str, Any]:
    """Real-world Mauna Loa CO2 (N=2,225 weekly obs, eps=1.0 ppm) full-series comparison.

    Gracefully skips if optional `statsmodels` is not installed (runtime stays 0-dependency).
    """
    try:
        sm = importlib.import_module("statsmodels.api")
    except ImportError:
        return {"available": False, "reason": "statsmodels not installed"}

    co2_df = sm.datasets.co2.load_pandas().data.dropna()
    ys = np.ascontiguousarray(co2_df["co2"].to_numpy(dtype=np.float64))
    n_obs = int(len(ys))
    ts = np.arange(n_obs, dtype=np.float64)

    raw_json = json.dumps(
        [{"t": int(t), "y": round(float(y), 2)} for t, y in zip(ts, ys, strict=True)],
        separators=(",", ":"),
    )
    raw_csv = "t,y\n" + "\n".join(f"{int(t)},{y:.2f}" for t, y in zip(ts, ys, strict=True))
    tsln_text = encode_tsln_style(ys, window=250)

    # Full-history native C11 Swing PLA Markdown digest (exact L_inf <= epsilon, no wake gating)
    dig = cs.StreamDigest(epsilon=epsilon, ring_capacity=4096, max_segments=4096)
    dig.digest_batch(ts, ys)
    dig.flush()
    rec = np.zeros(n_obs, dtype=np.float64)
    dig.reconstruct(ts, out_y=rec)
    max_linf_err = float(np.max(np.abs(rec - ys)))
    n_segs = int(dig.segment_count)
    pla_md = dig.format_markdown(channel="mauna_loa_co2", max_segments=n_segs)
    md_linf_err = _reconstruct_from_markdown_linf(pla_md, ys)

    c4_json, bpe_json = len(raw_json) // 4, approx_bpe_tokens(raw_json)
    c4_csv, bpe_csv = len(raw_csv) // 4, approx_bpe_tokens(raw_csv)
    c4_tsln, bpe_tsln = len(tsln_text) // 4, approx_bpe_tokens(tsln_text)
    c4_pla, bpe_pla = len(pla_md) // 4, approx_bpe_tokens(pla_md)

    return {
        "available": True,
        "dataset": "statsmodels.datasets.co2 (Mauna Loa Weekly CO2, 1958-2001)",
        "n_obs": n_obs,
        "epsilon_ppm": epsilon,
        "raw_json": {"chars": len(raw_json), "chars_div_4": c4_json, "bpe_proxy_tokens": bpe_json},
        "raw_csv": {"chars": len(raw_csv), "chars_div_4": c4_csv, "bpe_proxy_tokens": bpe_csv},
        "tsln_style_delta": {"chars": len(tsln_text), "chars_div_4": c4_tsln, "bpe_proxy_tokens": bpe_tsln},
        "chronosieve_swing_pla_digest": {
            "segments": n_segs,
            "pla_compression_ratio": n_obs / n_segs,
            "max_linf_err_ppm": max_linf_err,
            "markdown_reconstruct_max_linf_ppm": md_linf_err,
            "linf_verified": bool(max_linf_err <= epsilon),
            "chars": len(pla_md),
            "chars_div_4": c4_pla,
            "bpe_proxy_tokens": bpe_pla,
            "reduction_vs_json_bpe_pct": _pct_reduction(bpe_pla, bpe_json),
            "reduction_vs_json_chars4_pct": _pct_reduction(c4_pla, c4_json),
            "reduction_vs_csv_bpe_pct": _pct_reduction(bpe_pla, bpe_csv),
            "reduction_vs_csv_chars4_pct": _pct_reduction(c4_pla, c4_csv),
            "reduction_vs_tsln_bpe_pct": _pct_reduction(bpe_pla, bpe_tsln),
            "reduction_vs_tsln_chars4_pct": _pct_reduction(c4_pla, c4_tsln),
        },
    }


def run_token_benchmark(n_ticks: int = 10_000) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    ts = np.arange(n_ticks, dtype=np.float64)
    # Realistic telemetry: slow sinusoidal drift + small noise + 2 regime shifts + 3 sharp anomalies
    ys = np.sin(ts * 0.004) + rng.normal(0.0, 0.015, size=n_ticks)
    ys[3500:7000] += 1.5
    ys[7000:] -= 0.8
    true_anoms = [1800, 5200, 8900]
    for idx in true_anoms:
        ys[idx] += 4.5

    # 1. Raw JSON (typical agent telemetry payload)
    raw_json = json.dumps(
        [{"t": int(t), "y": round(float(y), 4)} for t, y in zip(ts, ys, strict=True)],
        separators=(",", ":"),
    )

    # 2. Raw CSV
    raw_csv = "t,y\n" + "\n".join(f"{int(t)},{y:.4f}" for t, y in zip(ts, ys, strict=True))

    # 3. TSLN-style compact delta-text baseline (all windows serialized)
    tsln_text = encode_tsln_style(ys, window=250)

    # 4. ChronoSieve: Wake-Up Gate + Pre-Control E-Reset PID + Anomaly-Preserving Swing PLA
    #    Only emits compact Markdown digest packets on certified shifts or anomaly bursts!
    pipe = cs.ChronoSieve(
        channel="telemetry_p99",
        n_agents=1,
        alpha=0.02,
        eta=0.05,
        epsilon=0.05,
        edet_threshold=1500.0,
        anomaly_burst=2,
        wake_cooldown=50,
    )
    # High-recall mode (1-tick anomaly wakeup with conservative alpha=0.002)
    pipe_strict = cs.ChronoSieve(
        channel="telemetry_p99",
        n_agents=1,
        alpha=0.002,
        eta=0.02,
        epsilon=0.05,
        edet_threshold=1500.0,
        anomaly_burst=1,
        wake_cooldown=40,
    )

    wakes_packets: list[str] = []
    for i in range(n_ticks):
        res = pipe.step(float(ts[i]), float(ys[i]))
        if res["should_wake"] and res["digest_markdown"] is not None:
            wakes_packets.append(res["digest_markdown"])
    gated_pinned_ts = {int(round(a["t"])) for a in pipe.digest.anomalies()}

    strict_packets: list[str] = []
    for i in range(n_ticks):
        res = pipe_strict.step(float(ts[i]), float(ys[i]))
        if res["should_wake"] and res["digest_markdown"] is not None:
            strict_packets.append(res["digest_markdown"])
    strict_pinned_ts = {int(round(a["t"])) for a in pipe_strict.digest.anomalies()}

    cs_all_text = "\n\n".join(wakes_packets)
    cs_strict_text = "\n\n".join(strict_packets)

    c4_json, tok_json = len(raw_json) // 4, approx_bpe_tokens(raw_json)
    c4_csv, tok_csv = len(raw_csv) // 4, approx_bpe_tokens(raw_csv)
    c4_tsln, tok_tsln = len(tsln_text) // 4, approx_bpe_tokens(tsln_text)
    c4_cs, tok_cs = len(cs_all_text) // 4, approx_bpe_tokens(cs_all_text)
    c4_cs_strict, tok_cs_strict = len(cs_strict_text) // 4, approx_bpe_tokens(cs_strict_text)

    avg_packet_bpe = (tok_cs / len(wakes_packets)) if wakes_packets else 0.0
    avg_packet_c4 = (c4_cs / len(wakes_packets)) if wakes_packets else 0.0
    avg_strict_bpe = (tok_cs_strict / len(strict_packets)) if strict_packets else 0.0
    avg_strict_c4 = (c4_cs_strict / len(strict_packets)) if strict_packets else 0.0

    gated_pinned_recall = sum(1 for a in true_anoms if a in gated_pinned_ts) / len(true_anoms)
    gated_wake_recall = sum(1 for a in true_anoms if f"t={a}.00" in cs_all_text) / len(true_anoms)
    strict_pinned_recall = sum(1 for a in true_anoms if a in strict_pinned_ts) / len(true_anoms)
    strict_wake_recall = sum(1 for a in true_anoms if f"t={a}.00" in cs_strict_text) / len(true_anoms)

    return {
        "n_ticks": n_ticks,
        "true_anomalies": true_anoms,
        "raw_json": {"chars": len(raw_json), "chars_div_4": c4_json, "bpe_proxy_tokens": tok_json},
        "raw_csv": {"chars": len(raw_csv), "chars_div_4": c4_csv, "bpe_proxy_tokens": tok_csv},
        "tsln_style_delta": {
            "chars": len(tsln_text),
            "chars_div_4": c4_tsln,
            "bpe_proxy_tokens": tok_tsln,
        },
        "chronosieve_gated_burst2": {
            "wake_packets": len(wakes_packets),
            "chars": len(cs_all_text),
            "chars_div_4": c4_cs,
            "bpe_proxy_tokens": tok_cs,
            "avg_bpe_tokens_per_packet": avg_packet_bpe,
            "avg_chars4_tokens_per_packet": avg_packet_c4,
            "reduction_vs_json_bpe_pct": _pct_reduction(tok_cs, tok_json),
            "reduction_vs_json_chars4_pct": _pct_reduction(c4_cs, c4_json),
            "reduction_vs_csv_bpe_pct": _pct_reduction(tok_cs, tok_csv),
            "reduction_vs_csv_chars4_pct": _pct_reduction(c4_cs, c4_csv),
            "reduction_vs_tsln_bpe_pct": _pct_reduction(tok_cs, tok_tsln),
            "reduction_vs_tsln_chars4_pct": _pct_reduction(c4_cs, c4_tsln),
            "injected_anomaly_recall_pinned": gated_pinned_recall,
            "injected_anomaly_recall_wake_packets": gated_wake_recall,
        },
        "chronosieve_high_recall": {
            "wake_packets": len(strict_packets),
            "chars": len(cs_strict_text),
            "chars_div_4": c4_cs_strict,
            "bpe_proxy_tokens": tok_cs_strict,
            "avg_bpe_tokens_per_packet": avg_strict_bpe,
            "avg_chars4_tokens_per_packet": avg_strict_c4,
            "reduction_vs_json_bpe_pct": _pct_reduction(tok_cs_strict, tok_json),
            "reduction_vs_json_chars4_pct": _pct_reduction(c4_cs_strict, c4_json),
            "reduction_vs_csv_bpe_pct": _pct_reduction(tok_cs_strict, tok_csv),
            "reduction_vs_csv_chars4_pct": _pct_reduction(c4_cs_strict, c4_csv),
            "reduction_vs_tsln_bpe_pct": _pct_reduction(tok_cs_strict, tok_tsln),
            "reduction_vs_tsln_chars4_pct": _pct_reduction(c4_cs_strict, c4_tsln),
            "injected_anomaly_recall_pinned": strict_pinned_recall,
            "injected_anomaly_recall_wake_packets": strict_wake_recall,
        },
        "real_world_mauna_loa_co2": run_mauna_loa_co2_benchmark(epsilon=1.0),
    }


def main() -> None:
    print(json.dumps(run_token_benchmark(), indent=2))


if __name__ == "__main__":
    main()
