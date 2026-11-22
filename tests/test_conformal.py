"""Tests for Module 2 (Pre-Control E-Reset Conformal PID, SoA Batch, Online COMA)."""

from __future__ import annotations

import array
import math

import chronosieve as cs
import numpy as np
import pytest


def test_conformal_pid_long_run_coverage_and_ereset_recovery() -> None:
    """Verify Round 2 Pre-Control E-Reset PID invariants:
    1. Long-run coverage converges to 1 - alpha even across abrupt 5x volatility shocks.
    2. Pre-Control E-Reset triggers bounded D-term reset on exogenous score s_t and
       recovers post-shift coverage far faster than pure fixed-eta Conformal PID.
    """
    rng = np.random.default_rng(2026)
    alpha = 0.10
    T = 3000

    # Regime schedule: sigma=1 -> 5x shock at t=600 -> drop to 1 at t=1500 -> 4x shock at t=2200
    sigmas = np.ones(T, dtype=np.float64)
    sigmas[600:1500] = 5.0
    sigmas[2200:] = 4.0
    y_true = rng.normal(0.0, sigmas)
    y_pred = np.zeros(T, dtype=np.float64)

    cp_ereset = cs.ConformalPID(
        alpha=alpha,
        eta=0.05,
        ki=1.0,
        c_sat=1.0,
        score_bound=50.0,
        init_q=1.6,
        edet_threshold=200.0,
        warmup_cal=32,
        score_window=128,
        enable_ereset=True,
    )
    cp_pure = cs.ConformalPID(
        alpha=alpha,
        eta=0.05,
        ki=1.0,
        c_sat=1.0,
        score_bound=50.0,
        init_q=1.6,
        score_window=0,
        enable_ereset=False,
    )

    mc_er = np.zeros(T, dtype=np.uint8)
    rs_er = np.zeros(T, dtype=np.uint8)
    qn_er = np.zeros(T, dtype=np.float64)
    res_er = cp_ereset.ingest_batch(y_true, y_pred, out_q_next=qn_er, out_miscov=mc_er, out_resets=rs_er)

    mc_pu = np.zeros(T, dtype=np.uint8)
    qn_pu = np.zeros(T, dtype=np.float64)
    res_pu = cp_pure.ingest_batch(y_true, y_pred, out_q_next=qn_pu, out_miscov=mc_pu)

    # 1. Pre-Control E-Reset stays tightly on target 1 - alpha = 0.90 across 3 volatility shocks,
    #    outperforming pure fixed-eta Conformal PID in both full-stream and post-shock coverage.
    assert abs(res_er["emp_coverage"] - (1.0 - alpha)) < 0.02
    assert abs(res_er["emp_coverage"] - (1.0 - alpha)) < abs(res_pu["emp_coverage"] - (1.0 - alpha))

    # 2. Pre-Control E-Reset fired certified D-term resets after the shocks
    assert res_er["reset_count"] >= 2
    shock_resets = np.flatnonzero(rs_er[600:700])
    assert shock_resets.size >= 1, "Expected pre-control E-detector to fire within 100 ticks of 5x shock"

    # 3. Post-shock recovery window [600..720] coverage is substantially better with E-Reset
    cov_er_shock = 1.0 - float(np.mean(mc_er[600:720]))
    cov_pu_shock = 1.0 - float(np.mean(mc_pu[600:720]))
    assert cov_er_shock > cov_pu_shock + 0.20, (
        f"Expected faster recovery: ereset={cov_er_shock:.3f} vs pure={cov_pu_shock:.3f}"
    )

    # 4. Stationary regime: both pure PID and E-Reset PID converge tightly to 1 - alpha
    y_stat = rng.normal(0.0, 1.0, size=3000)
    yp_stat = np.zeros(3000, dtype=np.float64)
    cp_stat_er = cs.ConformalPID(alpha=alpha, eta=0.05, enable_ereset=True)
    cp_stat_pu = cs.ConformalPID(alpha=alpha, eta=0.05, score_window=0, enable_ereset=False)
    r_ser = cp_stat_er.ingest_batch(y_stat, yp_stat)
    r_spu = cp_stat_pu.ingest_batch(y_stat, yp_stat)
    assert abs(r_ser["emp_coverage"] - (1.0 - alpha)) < 0.015
    assert abs(r_spu["emp_coverage"] - (1.0 - alpha)) < 0.015


def test_conformal_warmup_enforcement_and_edge_cases() -> None:
    cp = cs.ConformalPID(alpha=0.1, warmup_cal=4, score_bound=2.0, enable_ereset=True)
    # Mandatory W_cal >= 32 must be enforced
    for _i in range(32):
        r = cp.step(10.0, 0.0)
        assert r["calibrating"] is True
        assert 0.0 <= r["q_d"] <= 2.0
    r33 = cp.step(10.0, 0.0)
    assert r33["calibrating"] is False

    # Non-finite input and finite-input score overflow raise ValueError without mutating state
    with pytest.raises(ValueError):
        cp.step(float("nan"), 0.0)
    with pytest.raises(ValueError):
        cp.step(1e308, -1e308)
    r_after = cp.step(10.0, 0.0)
    assert r_after["step_count"] == 34
    with pytest.raises(ValueError):
        cs.ConformalPID(alpha=1.5)
    with pytest.raises(ValueError):
        cs.ConformalPID(warmup_cal=-1)
    with pytest.raises(RuntimeError):
        cs.ConformalPID.__new__(cs.ConformalPID).step(1.0, 0.0)
    with pytest.raises(RuntimeError):
        cs.ConformalBatchPID.__new__(cs.ConformalBatchPID).ingest_batch([1.0], [0.0])


def test_conformal_soa_matches_single_channel_exact() -> None:
    """Multi-channel SoA SIMD kernel must match C single-channel pure PIDs to < 1e-12."""
    rng = np.random.default_rng(77)
    T, C = 400, 19
    yt = np.ascontiguousarray(rng.normal(0.0, 1.5, size=(T, C)), dtype=np.float64)
    yp = np.ascontiguousarray(rng.normal(0.0, 0.5, size=(T, C)), dtype=np.float64)

    soa = cs.ConformalBatchPID(n_channels=C, alpha=0.10, eta=0.05, ki=1.2, c_sat=1.0, init_q=1.0)
    qn_soa = np.zeros((T, C), dtype=np.float64)
    mc_soa = np.zeros((T, C), dtype=np.uint8)
    n_steps = soa.ingest_batch(yt, yp, out_q_next=qn_soa, out_miscov=mc_soa)
    assert n_steps == T

    for c in range(C):
        single = cs.ConformalPID(
            alpha=0.10, eta=0.05, ki=1.2, c_sat=1.0, init_q=1.0, score_window=0, enable_ereset=False
        )
        qn_s = array.array("d", [0.0] * T)
        mc_s = bytearray(T)
        single.ingest_batch(
            array.array("d", yt[:, c]),
            array.array("d", yp[:, c]),
            out_q_next=qn_s,
            out_miscov=mc_s,
        )
        max_diff = float(np.max(np.abs(qn_soa[:, c] - np.frombuffer(qn_s, dtype=np.float64))))
        assert max_diff < 1e-12
        assert np.array_equal(mc_soa[:, c], np.frombuffer(mc_s, dtype=np.uint8))


def test_coma_exact_geometry_and_mc_coverage_theorem() -> None:
    """Verify exact COMA union/hull geometry AND Gasparin & Ramdas (UAI 2024) coverage >= 1 - 2*alpha."""
    # Exact bimodal union + hull + max_agreement
    out = cs.coma_merge([0.0, 0.0, 5.0], [6.0, 1.0, 6.0], weights=[0.4, 0.3, 0.3], tau=0.5)
    assert out["intervals"] == [(0.0, 1.0), (5.0, 6.0)]
    assert out["hull"] == (0.0, 6.0)
    assert math.isclose(out["max_agreement"], 0.7, rel_tol=1e-12)

    # Empty set when no point reaches threshold
    empty = cs.coma_merge([0.0, 10.0, 20.0], [1.0, 11.0, 21.0], tau=0.5)
    assert empty["intervals"] == []
    assert empty["hull"] is None
    assert math.isclose(empty["max_agreement"], 1.0 / 3.0, rel_tol=1e-12)

    # Monte Carlo Gasparin-Ramdas theorem: K=5 correlated agents each at 1-alpha=0.90
    rng = np.random.default_rng(42)
    alpha = 0.10
    z_q = 1.6448536269514722  # 90% normal interval half-width
    K = 5
    N = 2000
    cov_det = 0
    cov_rand = 0
    width_det = 0.0
    width_rand = 0.0

    for i in range(N):
        y = rng.normal(0.0, 1.0)
        # Correlated noisy agent centers around 0
        shared = rng.normal(0.0, 0.3)
        centers = shared + rng.normal(0.0, 0.25, size=K)
        # Calibrated half-width for each agent so marginal coverage ~ 0.90
        half = z_q * math.sqrt(1.0 + 0.3**2 + 0.25**2)
        lowers = centers - half
        uppers = centers + half

        r_det = cs.coma_merge(lowers, uppers, tau=0.5, randomized=False)
        r_rnd = cs.coma_merge(lowers, uppers, randomized=True, seed=10000 + i)
        assert 0.5 <= r_rnd["used_tau"] < 1.0

        if any(lo <= y <= hi for lo, hi in r_det["intervals"]):
            cov_det += 1
        if any(lo <= y <= hi for lo, hi in r_rnd["intervals"]):
            cov_rand += 1
        width_det += sum(hi - lo for lo, hi in r_det["intervals"])
        width_rand += sum(hi - lo for lo, hi in r_rnd["intervals"])

    assert cov_det / N >= 1.0 - 2.0 * alpha
    assert cov_rand / N >= 1.0 - 2.0 * alpha
    # Randomized U in [0.5, 1) strictly tightens average interval width vs deterministic tau=0.5
    assert width_rand < width_det
