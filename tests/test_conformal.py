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


def test_coma_merge_batch_zero_copy_and_edge_cases() -> None:
    """Verify zero-copy Py_buffer coma_merge_batch matches row-wise coma_merge & handles edge cases."""
    rng = np.random.default_rng(202602)
    T, K = 150, 8
    centers = rng.normal(0.0, 1.0, size=(T, K))
    widths = rng.uniform(0.3, 2.5, size=(T, K))
    lowers = np.ascontiguousarray(centers - widths, dtype=np.float64)
    uppers = np.ascontiguousarray(centers + widths, dtype=np.float64)
    shared_w = rng.uniform(0.1, 2.0, size=K).astype(np.float64)
    step_w = rng.uniform(0.1, 2.0, size=(T, K)).astype(np.float64)

    hlo_unif: np.ndarray | None = None
    nivs_unif: np.ndarray | None = None
    for w_in in (None, shared_w, step_w):
        hlo = np.empty(T, dtype=np.float64)
        hhi = np.empty(T, dtype=np.float64)
        agree = np.empty(T, dtype=np.float64)
        nivs = np.empty(T, dtype=np.uint32)
        summary = cs.coma_merge_batch(
            lowers,
            uppers,
            weights_2d=w_in,
            tau=0.5,
            out_hull_lo=hlo,
            out_hull_hi=hhi,
            out_max_agreement=agree,
            out_n_intervals=nivs,
        )
        assert summary["n_steps"] == T and summary["n_agents"] == K
        if w_in is None:
            hlo_unif = hlo.copy()
            nivs_unif = nivs.copy()
        for t in range(T):
            wt = None if w_in is None else (w_in if w_in.ndim == 1 else w_in[t])
            ref = cs.coma_merge(lowers[t], uppers[t], weights=wt, tau=0.5)
            assert int(nivs[t]) == len(ref["intervals"])
            assert math.isclose(float(agree[t]), ref["max_agreement"], rel_tol=1e-12, abs_tol=1e-12)
            if ref["hull"] is None:
                assert math.isnan(float(hlo[t])) and math.isnan(float(hhi[t]))
            else:
                assert math.isclose(float(hlo[t]), ref["hull"][0], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(float(hhi[t]), ref["hull"][1], rel_tol=1e-12, abs_tol=1e-12)

    assert hlo_unif is not None and nivs_unif is not None

    # Strided 2D NumPy slice & stdlib 1D array.array with explicit n_agents
    hlo_s = np.empty(T // 2, dtype=np.float64)
    summary_s = cs.coma_merge_batch(lowers[::2, :], uppers[::2, :], out_hull_lo=hlo_s)
    assert summary_s["n_steps"] == T // 2 and summary_s["n_agents"] == K
    assert np.allclose(hlo_s, hlo_unif[::2], equal_nan=True)

    lo_arr = array.array("d", lowers[:10].ravel().tolist())
    hi_arr = array.array("d", uppers[:10].ravel().tolist())
    out_lo_arr = array.array("d", [0.0] * 10)
    out_ni_arr = array.array("I", [0] * 10)
    cs.coma_merge_batch(lo_arr, hi_arr, n_agents=K, out_hull_lo=out_lo_arr, out_n_intervals=out_ni_arr)
    assert list(out_ni_arr) == [int(x) for x in nivs_unif[:10]]

    # K = 80 heap path (> 64 stack threshold) + randomized repeatability
    lo80 = np.full((12, 80), -1.0, dtype=np.float64)
    hi80 = np.full((12, 80), 2.0, dtype=np.float64)
    r1 = cs.coma_merge_batch(lo80, hi80, randomized=True, seed=999)
    r2 = cs.coma_merge_batch(lo80, hi80, randomized=True, seed=999)
    assert r1 == r2 and 0.5 <= r1["last_used_tau"] < 1.0

    # Empty batch T=0 & invalid error paths
    empty_res = cs.coma_merge_batch(
        np.empty((0, 4), dtype=np.float64), np.empty((0, 4), dtype=np.float64)
    )
    assert empty_res["n_steps"] == 0 and empty_res["last_hull"] is None
    with pytest.raises(ValueError):
        cs.coma_merge_batch(lo_arr, hi_arr)  # 1D without n_agents
    with pytest.raises(ValueError):
        cs.coma_merge_batch(lowers, uppers, weights_2d=np.ones(K + 1))
    ro_out = np.empty(T, dtype=np.float64)
    ro_out.flags.writeable = False
    with pytest.raises((TypeError, ValueError)):
        cs.coma_merge_batch(lowers, uppers, out_hull_lo=ro_out)

    # Overflow/underflow-safe scale-invariant weight normalization (1e308 & 1e-320)
    for extreme_w in ([1e308, 1e308], [1e-320, 1e-320]):
        res_ext = cs.coma_merge([0.0, 0.0], [1.0, 1.0], weights=extreme_w)
        assert res_ext["hull"] == (0.0, 1.0) and math.isclose(res_ext["max_agreement"], 1.0, rel_tol=1e-12)
        res_bext = cs.coma_merge_batch(
            np.zeros((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            weights_2d=np.array(extreme_w, dtype=np.float64),
        )
        assert res_bext["last_hull"] == (0.0, 1.0)
        assert math.isclose(res_bext["last_max_agreement"], 1.0, rel_tol=1e-12)

    # Strict 2D shape & ndim validation (reject transposed weights, 3D arrays, mismatched empty shapes)
    with pytest.raises(ValueError):
        cs.coma_merge_batch(
            np.zeros((2, 3), dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            weights_2d=np.ones((3, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError):
        cs.coma_merge_batch(
            np.zeros((2, 3, 4), dtype=np.float64),
            np.ones((2, 3, 4), dtype=np.float64),
            n_agents=4,
        )
    with pytest.raises(ValueError):
        cs.coma_merge_batch(
            np.empty((0, 4), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
            weights_2d=np.empty((0, 99), dtype=np.float64),
        )

    # T=1 with 2D (1, K) weights and all-empty majority -> hull=None / NaN output buffers
    hlo_1 = np.empty(1, dtype=np.float64)
    hhi_1 = np.empty(1, dtype=np.float64)
    res_t1 = cs.coma_merge_batch(
        np.array([[0.0, 10.0, 20.0]], dtype=np.float64),
        np.array([[1.0, 11.0, 21.0]], dtype=np.float64),
        weights_2d=np.array([[0.34, 0.33, 0.33]], dtype=np.float64),
        tau=0.5,
        out_hull_lo=hlo_1,
        out_hull_hi=hhi_1,
    )
    assert res_t1["last_hull"] is None and res_t1["last_n_intervals"] == 0
    assert math.isnan(float(hlo_1[0])) and math.isnan(float(hhi_1[0]))
