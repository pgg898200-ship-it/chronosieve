#!/usr/bin/env python3
"""Zero-dependency example: GPU cluster thermal guard with peer-relative rogue quarantine & zoom().

Demonstrates:
  - Unified `ChronoSieve` pipeline (zero-exp E-Detector + Pre-Control E-Reset PID +
    Sleeping Fixed-Share BOA + Anomaly-Preserving Swing PLA + lossless ring-buffer `zoom()`).
  - Peer-relative rogue attribution (`peer_relative_rogue=True`, default in v0.2.0):
    healthy forecasters survive an 86-sigma exogenous thermal spike at t=180 without false
    quarantine at standard `rogue_threshold=200.0`, while hallucinating Agent 2 is isolated.
"""

from __future__ import annotations

import random

import chronosieve as cs


def main() -> None:
    rng = random.Random(2026)
    pipe = cs.ChronoSieve(
        channel="gpu_cluster_temp_c",
        n_agents=3,
        alpha=0.005,
        epsilon=0.5,
        edet_threshold=1000.0,
        rogue_threshold=200.0,
        anomaly_burst=1,
        wake_cooldown=2,
        monitor_level=True,
    )

    wake_ticks: list[int] = []
    shift_ticks: list[int] = []
    last_packet = ""

    for t in range(250):
        base = 62.0 if t < 150 else 74.0
        y = base + rng.gauss(0.0, 0.25)
        if t == 180:
            y = 95.5  # Exogenous 86-sigma thermal spike

        p0 = base + rng.gauss(0.0, 0.20)
        p1 = base + rng.gauss(0.0, 0.25)
        p2 = (base + rng.gauss(0.0, 0.20)) if t < 160 else (base - 12.0)  # Agent 2 hallucinates!

        res = pipe.step(float(t), y, preds=[p0, p1, p2])
        if res["shift_detected"]:
            shift_ticks.append(t)
        if res["should_wake"] and res["digest_markdown"]:
            wake_ticks.append(t)
            last_packet = res["digest_markdown"]

    assert pipe.fuser is not None
    stats, rogue_flags = pipe.fuser.rogue_stats()
    zoom_res = pipe.zoom(178.0, 182.0)

    print(f"ChronoSieve v{cs.__version__} ({cs.simd_backend()} SIMD)")
    print(f"Shift certified at ticks: {shift_ticks}")
    wake_pct = 100.0 * len(wake_ticks) / 250.0
    print(f"LLM wake-up ticks ({len(wake_ticks)}/250 ticks = {wake_pct:.1f}%): {wake_ticks}")
    print(f"Final BOA weights: {[round(w, 4) for w in pipe.fuser.weights()]}")
    print(f"Rogue flags (expected [False, False, True]): {rogue_flags}")
    print(f"Per-agent rogue e-stats: {[round(s, 2) for s in stats]}")
    print(f"Lossless zoom(178..182) truncated={zoom_res['truncated']}:")
    for rec in zoom_res["records"]:
        print(f"  t={rec['t']:.0f} y={rec['y']:.2f} y_hat={rec['y_hat']:.2f} q={rec['q_t']:.2f}")
    print("\nLatest System-2 Wake-Up Digest Packet:\n" + last_packet)

    assert rogue_flags == [False, False, True], f"Unexpected rogue flags: {rogue_flags}"
    assert any(r["t"] == 180.0 and abs(r["y"] - 95.5) < 1e-9 for r in zoom_res["records"])


if __name__ == "__main__":
    main()
