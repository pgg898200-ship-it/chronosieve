"""Tests for Module 3 (NEON Sleeping Fixed-Share BOA + Per-Agent Drift Attribution)."""

from __future__ import annotations

import math

import chronosieve as cs
import numpy as np
import pytest


def test_sleeping_boa_zero_regret_asynchronous_agents() -> None:
    """Slow System-2 agent awake only 1/5 ticks (NaN while asleep) earns dominant weight when best."""
    rng = np.random.default_rng(2026)
    T = 400
    K = 5
    y = np.sin(np.arange(T) * 0.05)

    # Agents 0..3: fast System-1 (always awake, moderate noise sigma=0.4..0.8)
    # Agent 4: slow System-2 LLM (awake every 5th tick, very accurate sigma=0.05!)
    preds = np.zeros((T, K), dtype=np.float64)
    active = np.ones((T, K), dtype=np.uint8)
    sigmas = [0.45, 0.55, 0.70, 0.85, 0.05]
    for k in range(K):
        preds[:, k] = y + rng.normal(0.0, sigmas[k], size=T)
    for t in range(T):
        if t % 5 != 0:
            active[t, 4] = 0
            preds[t, 4] = float("nan")  # Must be safely ignored while sleeping

    fus = cs.SleepingBOA(n_agents=K, share_base=0.01, rogue_threshold=1000.0)
    out_preds = np.zeros(T, dtype=np.float64)
    fus.fuse_batch(preds, y, active_matrix=active, out_preds=out_preds)

    weights = fus.weights()
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-12)
    # Slow accurate agent 4 and best fast agent 0 must dominate noisy agents 2 & 3
    assert weights[4] > weights[2] and weights[4] > weights[3]
    assert weights[0] > weights[3]

    # On ticks where agent 4 is awake, fused MSE beats every always-awake System-1 agent!
    awake_idx = np.arange(50, T, 5)
    mse_fused = float(np.mean((out_preds[awake_idx] - y[awake_idx]) ** 2))
    mse_best_fast = float(np.mean((preds[awake_idx, 0] - y[awake_idx]) ** 2))
    assert mse_fused < mse_best_fast


def test_shift_coupled_fixed_share_beats_static_share() -> None:
    """When best agent switches abruptly, shift-coupled gamma_shift boost adapts much faster."""
    rng = np.random.default_rng(99)
    T = 300
    K = 4
    y = np.zeros(T, dtype=np.float64)
    preds = np.zeros((T, K), dtype=np.float64)

    # Regime 1 (t < 150): Agent 0 accurate, Agents 1..3 noisy
    # Regime 2 (t >= 150): Agent 3 accurate, Agents 0..2 noisy
    for t in range(T):
        if t < 150:
            errs = [0.05, 0.8, 0.9, 1.0]
        else:
            errs = [1.0, 0.9, 0.8, 0.05]
        for k in range(K):
            preds[t, k] = rng.normal(0.0, errs[k])

    shifts = np.zeros(T, dtype=np.uint8)
    shifts[155] = 1  # E-detector certified shift 5 ticks after regime change

    fus_coupled = cs.SleepingBOA(
        n_agents=K, share_base=0.002, share_shift=0.50, shift_var_decay=0.1, quarantine_rogue=False
    )
    fus_static = cs.SleepingBOA(
        n_agents=K, share_base=0.002, share_shift=0.002, shift_var_decay=1.0, quarantine_rogue=False
    )

    p_coupled = np.zeros(T, dtype=np.float64)
    p_static = np.zeros(T, dtype=np.float64)
    fus_coupled.fuse_batch(preds, y, shifts=shifts, out_preds=p_coupled)
    fus_static.fuse_batch(preds, y, shifts=None, out_preds=p_static)

    # Post-shift transition window [155..210]: shift-coupled BOA has far lower MSE
    mse_coupled = float(np.mean(p_coupled[155:210] ** 2))
    mse_static = float(np.mean(p_static[155:210] ** 2))
    assert mse_coupled < mse_static * 0.5


def test_rogue_agent_catoni_attribution_and_quarantine() -> None:
    """Hallucinating/drifting agent is pinpointed by per-agent Catoni and quarantined."""
    rng = np.random.default_rng(7)
    T = 220
    K = 4
    y = np.zeros(T, dtype=np.float64)
    preds = rng.normal(0.0, 0.2, size=(T, K))
    # Agent 2 goes rogue at t=100 (sustained +2.5 hallucination offset)
    preds[100:, 2] += 2.5

    fus = cs.SleepingBOA(
        n_agents=K,
        rogue_threshold=150.0,
        cal_window=48,
        quarantine_rogue=True,
    )
    out_preds = np.zeros(T, dtype=np.float64)
    out_rogue = np.zeros(T, dtype=np.uint8)
    fus.fuse_batch(preds, y, out_preds=out_preds, out_rogue_any=out_rogue)

    stats, flags = fus.rogue_stats()
    assert flags[2] is True
    assert flags[0] is False and flags[1] is False and flags[3] is False
    assert stats[2] >= 150.0

    # Rogue detection happens rapidly (within 30 ticks of t=100)
    first_rogue = int(np.flatnonzero(out_rogue)[0])
    assert 100 <= first_rogue <= 130

    # Quarantined ensemble stays accurate after detection
    post_mae = float(np.mean(np.abs(out_preds[135:] - y[135:])))
    assert post_mae < 0.25

    # clear_rogue resets attribution cleanly
    fus.clear_rogue(2)
    _, flags_after = fus.rogue_stats()
    assert flags_after[2] is False


def test_fuser_edge_cases() -> None:
    fus = cs.SleepingBOA(n_agents=3)
    # All sleeping does not crash or corrupt weights
    r = fus.step([float("nan"), float("nan"), float("nan")], 1.0, active=[0, 0, 0])
    assert r["active_count"] == 0
    assert all(math.isclose(w, 1.0 / 3.0) for w in fus.weights())

    # Active NaN raises ValueError
    with pytest.raises(ValueError):
        fus.step([1.0, float("nan"), 1.0], 1.0, active=[1, 1, 1])

    # Length mismatch raises ValueError
    with pytest.raises(ValueError):
        fus.predict([1.0, 2.0])

    # Quarantined rogue hallucination must NOT freeze healthy agent BOA learning
    boa = cs.SleepingBOA(
        n_agents=3,
        share_base=0.01,
        share_shift=0.35,
        rogue_mu0=0.2,
        rogue_sigma=0.3,
        rogue_threshold=50.0,
        quarantine_rogue=True,
    )
    for _ in range(30):
        boa.step([0.0, 0.15, 0.15], 0.0)
    for _ in range(10):
        boa.step([0.0, 0.15, 1e5], 0.0)
    for t in range(150):
        boa.step([0.15, 0.0, 1e5], 0.0, shift_detected=(t == 0))
    w = boa.weights()
    assert w[1] > 0.80 and w[1] > w[0] * 5.0

    # All-sleeping preserves current rogue_count & any_rogue
    r_sl = boa.step([float("nan")] * 3, 0.0, active=[0, 0, 0])
    assert r_sl["any_rogue"] is True and r_sl["rogue_count"] >= 1

    # Huge finite residual [1e160, 0.0] + shift recovers cleanly (no Inf/NaN freeze)
    boa2 = cs.SleepingBOA(n_agents=2)
    r_big = boa2.step([1e160, 0.0], 0.0)
    assert math.isfinite(r_big["fused_loss"])
    for t in range(50):
        boa2.step([0.0, 1.0], 0.0, shift_detected=(t == 0))
    assert boa2.weights()[0] > 0.80

    # Negative cal_window & uninitialized __new__ guarded
    with pytest.raises(ValueError):
        cs.SleepingBOA(n_agents=2, cal_window=-1)
    with pytest.raises(RuntimeError):
        cs.SleepingBOA.__new__(cs.SleepingBOA).weights()


def test_peer_relative_rogue_spike_immunity_and_fallbacks() -> None:
    """Verify peer-relative excess residual cancels 86-sigma exogenous spikes, protects super-accurate
    agents from all-zero warmup collapse, and falls back to raw residual when active_cnt < 3."""
    rng = np.random.default_rng(2026)
    T = 200
    for K in (3, 5, 8):
        y = np.full(T, 62.0, dtype=np.float64) + rng.normal(0.0, 0.25, size=T)
        y[100] = 95.5  # 86-sigma exogenous common shock
        y[150] = 95.5
        preds = np.full((T, K), 62.0, dtype=np.float64) + rng.normal(0.0, 0.22, size=(T, K))
        fus_peer = cs.SleepingBOA(n_agents=K, rogue_threshold=200.0, peer_relative_rogue=True)
        fus_peer.fuse_batch(preds, y)
        _, flags_peer = fus_peer.rogue_stats()
        assert not any(flags_peer), f"K={K} false-quarantined under exogenous spikes: {flags_peer}"

        # Raw residual mode (peer_relative_rogue=False) triggers false quarantine on the same shocks
        fus_raw = cs.SleepingBOA(n_agents=K, rogue_threshold=200.0, peer_relative_rogue=False)
        fus_raw.fuse_batch(preds, y)
        _, flags_raw = fus_raw.rogue_stats()
        assert any(flags_raw), f"K={K} raw mode expected to false-trigger on 86-sigma spikes"

    # Super-accurate agent (sigma=0.02 vs peers 0.50) has all-zero excess during warmup;
    # peer-median floor prevents sigma collapse so mild normal jitter does NOT false-quarantine it.
    rng2 = np.random.default_rng(77)
    y2 = np.zeros(220, dtype=np.float64)
    p2 = np.column_stack(
        [
            rng2.normal(0.0, 0.02, size=220),
            rng2.normal(0.0, 0.50, size=220),
            rng2.normal(0.0, 0.50, size=220),
            rng2.normal(0.0, 0.50, size=220),
        ]
    )
    p2[120, 0] = 0.25  # Mild single-tick noise on accurate agent
    for cw in (16, 48):
        fus_acc = cs.SleepingBOA(
            n_agents=4, cal_window=cw, rogue_threshold=200.0, peer_relative_rogue=True
        )
        fus_acc.fuse_batch(p2, y2)
        _, fl_acc = fus_acc.rogue_stats()
        assert fl_acc[0] is False, f"super-accurate agent 0 false-quarantined at cal_window={cw}: {fl_acc}"

    # Large ensemble K=80 (> 64 quickselect median path + O(K) symmetric peer calibration)
    y80 = rng2.normal(50.0, 0.2, size=200)
    y80[100] = 150.0  # exogenous spike
    p80 = 50.0 + rng2.normal(0.0, 0.25, size=(200, 80))
    p80[110:, 79] += 10.0  # agent 79 goes rogue
    fus80 = cs.SleepingBOA(n_agents=80, cal_window=16, rogue_threshold=200.0, peer_relative_rogue=True)
    fus80.fuse_batch(p80, y80)
    _, fl80 = fus80.rogue_stats()
    assert fl80 == [False] * 79 + [True]

    # Graceful raw-residual fallback when active_cnt == 2 (< 3)
    fus_2 = cs.SleepingBOA(n_agents=2, rogue_threshold=150.0, peer_relative_rogue=True)
    p_two = rng2.normal(0.0, 0.2, size=(160, 2))
    p_two[80:, 1] += 4.0
    fus_2.fuse_batch(p_two, np.zeros(160, dtype=np.float64))
    _, fl_two = fus_2.rogue_stats()
    assert fl_two == [False, True]
