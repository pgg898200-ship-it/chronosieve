#!/usr/bin/env python3
"""Reproducible LLM token reduction benchmark: Raw JSON vs CSV vs TSLN-style vs ChronoSieve.

Measures real OpenAI `tiktoken` (`o200k_base` and `cl100k_base`) when installed (respecting
`TIKTOKEN_CACHE_DIR`), alongside deterministic zero-dependency `chars // 4` and BPE-proxy counts.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
from typing import Any

import chronosieve as cs
import numpy as np


def approx_bpe_tokens(text: str) -> int:
    """Deterministic BPE-proxy token counter (subword chunks + punctuation, close to cl100k_base)."""
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)
    total = 0
    for p in parts:
        total += max(1, (len(p) + 3) // 4)
    return total


def _load_tiktoken_encoders() -> tuple[dict[str, Any], str | None]:
    """Load real OpenAI tiktoken encoders if installed & cached/reachable; fail gracefully otherwise."""
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ImportError:
        return {}, None

    version: str | None = getattr(tiktoken, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError:
            version = None

    encoders: dict[str, Any] = {}
    for name in ("o200k_base", "cl100k_base"):
        try:
            encoders[name] = tiktoken.get_encoding(name)
        except Exception:
            continue
    return encoders, version


def _token_stats(text: str, encoders: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "chars": len(text),
        "chars_div_4": len(text) // 4,
        "bpe_proxy_tokens": approx_bpe_tokens(text),
    }
    for name, enc in encoders.items():
        out[f"tiktoken_{name}"] = len(enc.encode(text))
    return out


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


def run_mauna_loa_co2_benchmark(
    epsilon: float = 1.0,
    encoders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Real-world Mauna Loa CO2 (N=2,225 weekly obs, eps=1.0 ppm) full-series comparison.

    Gracefully skips if optional `statsmodels` is not installed (runtime stays 0-dependency).
    """
    try:
        sm = importlib.import_module("statsmodels.api")
    except ImportError:
        return {"available": False, "reason": "statsmodels not installed"}

    if encoders is None:
        encoders, _ = _load_tiktoken_encoders()

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

    st_json = _token_stats(raw_json, encoders)
    st_csv = _token_stats(raw_csv, encoders)
    st_tsln = _token_stats(tsln_text, encoders)
    st_pla = _token_stats(pla_md, encoders)

    pla_entry: dict[str, Any] = {
        "segments": n_segs,
        "pla_compression_ratio": n_obs / n_segs,
        "max_linf_err_ppm": max_linf_err,
        "markdown_reconstruct_max_linf_ppm": md_linf_err,
        "linf_verified": bool(max_linf_err <= epsilon),
        **st_pla,
        "reduction_vs_json_bpe_pct": _pct_reduction(st_pla["bpe_proxy_tokens"], st_json["bpe_proxy_tokens"]),
        "reduction_vs_json_chars4_pct": _pct_reduction(st_pla["chars_div_4"], st_json["chars_div_4"]),
        "reduction_vs_csv_bpe_pct": _pct_reduction(st_pla["bpe_proxy_tokens"], st_csv["bpe_proxy_tokens"]),
        "reduction_vs_csv_chars4_pct": _pct_reduction(st_pla["chars_div_4"], st_csv["chars_div_4"]),
        "reduction_vs_tsln_bpe_pct": _pct_reduction(st_pla["bpe_proxy_tokens"], st_tsln["bpe_proxy_tokens"]),
        "reduction_vs_tsln_chars4_pct": _pct_reduction(st_pla["chars_div_4"], st_tsln["chars_div_4"]),
    }
    for enc_name in encoders:
        k = f"tiktoken_{enc_name}"
        pla_entry[f"reduction_vs_json_{enc_name}_pct"] = _pct_reduction(st_pla[k], st_json[k])
        pla_entry[f"reduction_vs_csv_{enc_name}_pct"] = _pct_reduction(st_pla[k], st_csv[k])
        pla_entry[f"reduction_vs_tsln_{enc_name}_pct"] = _pct_reduction(st_pla[k], st_tsln[k])

    return {
        "available": True,
        "dataset": "statsmodels.datasets.co2 (Mauna Loa Weekly CO2, 1958-2001)",
        "n_obs": n_obs,
        "epsilon_ppm": epsilon,
        "raw_json": st_json,
        "raw_csv": st_csv,
        "tsln_style_delta": st_tsln,
        "chronosieve_swing_pla_digest": pla_entry,
    }


def run_token_benchmark(n_ticks: int = 10_000) -> dict[str, Any]:
    encoders, tiktoken_ver = _load_tiktoken_encoders()

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

    st_json = _token_stats(raw_json, encoders)
    st_csv = _token_stats(raw_csv, encoders)
    st_tsln = _token_stats(tsln_text, encoders)
    st_cs = _token_stats(cs_all_text, encoders)
    st_strict = _token_stats(cs_strict_text, encoders)

    def _build_cs_entry(
        st: dict[str, Any], packets: list[str], pinned_ts: set[int], text: str
    ) -> dict[str, Any]:
        n_pk = len(packets)
        entry: dict[str, Any] = {
            "wake_packets": n_pk,
            **st,
            "avg_bpe_tokens_per_packet": (st["bpe_proxy_tokens"] / n_pk) if n_pk else 0.0,
            "avg_chars4_tokens_per_packet": (st["chars_div_4"] / n_pk) if n_pk else 0.0,
            "reduction_vs_json_bpe_pct": _pct_reduction(st["bpe_proxy_tokens"], st_json["bpe_proxy_tokens"]),
            "reduction_vs_json_chars4_pct": _pct_reduction(st["chars_div_4"], st_json["chars_div_4"]),
            "reduction_vs_csv_bpe_pct": _pct_reduction(st["bpe_proxy_tokens"], st_csv["bpe_proxy_tokens"]),
            "reduction_vs_csv_chars4_pct": _pct_reduction(st["chars_div_4"], st_csv["chars_div_4"]),
            "reduction_vs_tsln_bpe_pct": _pct_reduction(st["bpe_proxy_tokens"], st_tsln["bpe_proxy_tokens"]),
            "reduction_vs_tsln_chars4_pct": _pct_reduction(st["chars_div_4"], st_tsln["chars_div_4"]),
            "injected_anomaly_recall_pinned": sum(1 for a in true_anoms if a in pinned_ts) / len(true_anoms),
            "injected_anomaly_recall_wake_packets": (
                sum(1 for a in true_anoms if f"t={a}.00" in text) / len(true_anoms)
            ),
        }
        for enc_name in encoders:
            k = f"tiktoken_{enc_name}"
            entry[f"avg_{enc_name}_tokens_per_packet"] = (st[k] / n_pk) if n_pk else 0.0
            entry[f"reduction_vs_json_{enc_name}_pct"] = _pct_reduction(st[k], st_json[k])
            entry[f"reduction_vs_csv_{enc_name}_pct"] = _pct_reduction(st[k], st_csv[k])
            entry[f"reduction_vs_tsln_{enc_name}_pct"] = _pct_reduction(st[k], st_tsln[k])
        return entry

    return {
        "n_ticks": n_ticks,
        "tiktoken_available": bool(encoders),
        "tiktoken_version": tiktoken_ver,
        "tiktoken_encodings": sorted(encoders.keys()),
        "true_anomalies": true_anoms,
        "raw_json": st_json,
        "raw_csv": st_csv,
        "tsln_style_delta": st_tsln,
        "chronosieve_gated_burst2": _build_cs_entry(st_cs, wakes_packets, gated_pinned_ts, cs_all_text),
        "chronosieve_high_recall": _build_cs_entry(
            st_strict, strict_packets, strict_pinned_ts, cs_strict_text
        ),
        "real_world_mauna_loa_co2": run_mauna_loa_co2_benchmark(epsilon=1.0, encoders=encoders),
    }


def main() -> None:
    print(json.dumps(run_token_benchmark(), indent=2))


if __name__ == "__main__":
    main()
