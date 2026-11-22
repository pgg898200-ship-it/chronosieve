"""Tests for Module 4 (Anomaly-Preserving Swing PLA, Wake-Up Gate, Ring-Buffer zoom)."""

from __future__ import annotations

import array
import math

import chronosieve as cs
import numpy as np
import pytest


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.05, 0.25])
def test_swing_pla_strict_linf_bound_and_verbatim_anomalies(eps: float) -> None:
    """Guarantee max|reconstruct(t) - y| <= eps on ALL points and == 0.0 on pinned anomalies."""
    rng = np.random.default_rng(1234)
    N = 2000
    ts = np.arange(N, dtype=np.float64)
    ys = (
        2.0 * np.sin(ts * 0.02)
        + 0.003 * ts
        + rng.normal(0.0, 0.04, size=N)
        + np.where(ts >= 1000.0, 1.5, 0.0)
    )
    anom_indices = [150, 777, 1234, 1888]
    for idx in anom_indices:
        ys[idx] += 4.25

    flags = np.zeros(N, dtype=np.uint32)
    for idx in anom_indices:
        flags[idx] = cs.FLAG_ANOMALY
    flags[1000] = cs.FLAG_SHIFT

    dig = cs.StreamDigest(epsilon=eps, ring_capacity=4096, max_segments=8192, max_anomalies=256)
    y_hats = ys - 0.1
    qs = np.full(N, 0.5, dtype=np.float64)
    wakes = np.zeros(N, dtype=np.uint8)
    res = dig.digest_batch(ts, ys, y_hats=y_hats, qs=qs, flags=flags, out_wakes=wakes)

    assert res["total_points"] == N
    assert res["total_anomalies"] == len(anom_indices)
    # Only the 4 anomalies + 1 shift should trigger wake-ups (0 wakes on 1995 normal ticks!)
    assert int(np.sum(wakes)) == len(anom_indices) + 1

    rec = np.zeros(N, dtype=np.float64)
    dig.reconstruct(ts, out_y=rec)
    errs = np.abs(rec - ys)

    for idx in anom_indices:
        assert errs[idx] == 0.0, f"Verbatim anomaly at {idx} had nonzero error {errs[idx]}"
    assert float(np.max(errs)) <= eps + 1e-12

    # Compression should be substantial for eps >= 0.05
    if eps >= 0.05:
        assert dig.segment_count < N / 4

    # Anomalies list stores exact verbatim (t, y, residual, q_t)
    anoms = dig.anomalies()
    assert len(anoms) == len(anom_indices)
    for rec_a, idx in zip(anoms, anom_indices, strict=True):
        assert rec_a["t"] == float(idx)
        assert rec_a["y"] == ys[idx]
        assert math.isclose(rec_a["residual"], 0.1, rel_tol=1e-12)
        assert rec_a["q_t"] == 0.5


def test_ring_buffer_lossless_zoom_and_eviction_truncation() -> None:
    dig = cs.StreamDigest(epsilon=0.1, ring_capacity=128)
    ts = array.array("d", [float(i) * 0.5 for i in range(300)])
    ys = array.array("d", [math.sin(i * 0.1) for i in range(300)])
    dig.digest_batch(ts, ys)

    assert dig.ring_size == 128

    # Oldest retained timestamp is index 172 -> t = 86.0
    z_ok = dig.zoom(100.0, 110.0)
    assert z_ok["truncated"] is False
    recs = z_ok["records"]
    assert len(recs) == 21
    assert recs[0]["t"] == 100.0 and recs[-1]["t"] == 110.0
    for r in recs:
        orig_i = int(round(r["t"] * 2.0))
        assert r["y"] == ys[orig_i]

    # Querying evicted window [10.0, 90.0] sets truncated=True and returns retained suffix [86.0..90.0]
    z_tr = dig.zoom(10.0, 90.0)
    assert z_tr["truncated"] is True
    assert z_tr["records"][0]["t"] == 86.0
    assert z_tr["records"][-1]["t"] == 90.0


def test_wakeup_gate_burst_cooldown_and_markdown_size() -> None:
    dig = cs.StreamDigest(epsilon=0.05, wake_cooldown=10, anomaly_burst=2)
    # Stationary ticks produce 0 wakeups
    for i in range(50):
        r = dig.push(float(i), 1.0, y_hat=1.0, q_t=0.2, flags=0)
        assert r["should_wake"] is False

    # Single isolated anomaly does NOT wake when anomaly_burst=2
    r1 = dig.push(50.0, 5.0, y_hat=1.0, q_t=0.2, flags=cs.FLAG_ANOMALY)
    assert r1["should_wake"] is False
    # Second consecutive anomaly wakes!
    r2 = dig.push(51.0, 5.5, y_hat=1.0, q_t=0.2, flags=cs.FLAG_ANOMALY)
    assert r2["should_wake"] is True
    # Third anomaly inside cooldown=10 does NOT wake
    r3 = dig.push(52.0, 5.8, y_hat=1.0, q_t=0.2, flags=cs.FLAG_ANOMALY)
    assert r3["should_wake"] is False
    # Certified SHIFT always bypasses cooldown!
    r4 = dig.push(53.0, 6.0, y_hat=1.0, q_t=0.2, flags=cs.FLAG_SHIFT)
    assert r4["should_wake"] is True

    md = dig.format_markdown(
        channel="cpu_p99",
        emp_coverage=0.905,
        current_q=0.42,
        edet_stat=612.0,
        edet_threshold=500.0,
        rogue_agent=1,
    )
    # Compact ~100-250 token packet (approx 4 chars/token -> ~400-1200 chars)
    approx_tokens = len(md) / 4.0
    assert 40 <= approx_tokens <= 300
    assert 'zoom("cpu_p99"' in md
    assert "rogue=agent_1" in md

    # Out-of-order timestamp rejected
    with pytest.raises(ValueError):
        dig.push(10.0, 1.0)

    # Huge dt=1e160 & dy=1e160 must keep finite slope (no Inf/Inf -> NaN)
    d_big = cs.StreamDigest(epsilon=0.05)
    d_big.push(0.0, 0.0)
    d_big.push(1e160, 1e160)
    segs = d_big.segments()
    assert math.isfinite(segs[0]["slope"]) and math.isfinite(segs[0]["y_end"])

    # Strict L_inf <= eps even at huge offset y ~ 1e11 (ULP rounding guard)
    d_ulp = cs.StreamDigest(epsilon=1e-3, max_segments=8192)
    ts_u = np.arange(500, dtype=np.float64)
    ys_u = 1e11 + np.sin(ts_u * 0.13) * 3e-3 + np.cos(ts_u * 0.07) * 1.5e-3
    d_ulp.digest_batch(ts_u, ys_u)
    rec_u = np.zeros(500, dtype=np.float64)
    d_ulp.reconstruct(ts_u, out_y=rec_u)
    assert float(np.max(np.abs(rec_u - ys_u))) <= 1e-3

    # Negative u32 kwargs & uninitialized __new__ guarded
    with pytest.raises(ValueError):
        cs.StreamDigest(wake_cooldown=-5)
    with pytest.raises(RuntimeError):
        cs.StreamDigest.__new__(cs.StreamDigest).format_markdown()

    # Extreme tiny dt (1e-290) + large eps (1e20): SIMD batch must match scalar push() exactly
    # without letting (dy +/- eps) / dt overflow to +/-inf and bypass finite-slope checks.
    ts_tiny = np.array([0.0, 1e-290, 2e-290, 3e-290], dtype=np.float64)
    ys_tiny = np.zeros(4, dtype=np.float64)
    d_tseq = cs.StreamDigest(epsilon=1e20)
    for tv, yv in zip(ts_tiny, ys_tiny, strict=True):
        d_tseq.push(float(tv), float(yv))
    d_tbat = cs.StreamDigest(epsilon=1e20)
    d_tbat.digest_batch(ts_tiny, ys_tiny)
    assert d_tbat.segment_count == d_tseq.segment_count == 4

    # Mid-batch non-monotonic timestamp raises ValueError & commits exact valid prefix
    d_nm = cs.StreamDigest(epsilon=0.05)
    with pytest.raises(ValueError):
        d_nm.digest_batch(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.5, 5.0], dtype=np.float64),
            np.zeros(7, dtype=np.float64),
        )
    assert d_nm.ring_size == 5
