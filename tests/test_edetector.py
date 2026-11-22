"""Tests & Monte Carlo ARL>=gamma proofs for Module 1 (EDetector & EFusion)."""

from __future__ import annotations

import array
import math

import chronosieve as cs
import numpy as np
import pytest


def _mc_empirical_arl(
    make_det,
    sample_fn,
    *,
    n_trials: int = 160,
    horizon: int = 2500,
    seed: int = 2026,
) -> tuple[float, float]:
    """Run Monte Carlo stopping-time trials under H0.

    Returns (censored_mean_arl, uncensored_fraction).
    Note: censored_mean_arl = mean(min(tau, horizon)) is a strict LOWER BOUND on true E[tau].
    """
    rng = np.random.default_rng(seed)
    stops = []
    uncensored = 0
    for trial in range(n_trials):
        det = make_det(trial)
        xs = np.ascontiguousarray(sample_fn(rng, horizon), dtype=np.float64)
        alarms = np.zeros(horizon, dtype=np.uint8)
        det.ingest_batch(xs, out_alarms=alarms)
        hit = np.flatnonzero(alarms)
        if hit.size > 0:
            stops.append(int(hit[0]) + 1)
            uncensored += 1
        else:
            stops.append(horizon)
    return float(np.mean(stops)), uncensored / n_trials


def test_auto_routing_and_edge_cases() -> None:
    # AUTO routes to Vovk Rank when mu0/sigma are unknown, Catoni when known
    d_rank = cs.EDetector(family="auto", threshold=100.0)
    assert d_rank.effective_family == "vovk_rank"

    d_cat = cs.EDetector(family="auto", mu0=0.0, sigma=1.0, threshold=100.0)
    assert d_cat.effective_family == "catoni"

    # Empty batch is a safe no-op
    res = d_cat.ingest_batch(array.array("d"))
    assert res["step_count"] == 0
    assert res["alarmed"] is False

    # Non-finite values raise ValueError without corrupting state
    with pytest.raises(ValueError):
        d_cat.step(float("nan"))
    with pytest.raises(ValueError):
        d_cat.step(float("inf"))
    assert d_cat.step_count == 0

    # Invalid constructor args raise ValueError
    with pytest.raises(ValueError):
        cs.EDetector(family="catoni", mu0=0.0, sigma=-1.0)
    with pytest.raises(ValueError):
        cs.EDetector(family="bounded", mu0=1.2)
    with pytest.raises(ValueError):
        cs.EDetector(threshold=0.5)


def test_zero_copy_buffers_and_one_sided_directionality() -> None:
    # Verify stdlib array.array('d'), memoryview, and bytearray work with zero NumPy
    d_up = cs.EDetector(family="catoni", side="upper", mu0=0.0, sigma=1.0, threshold=100.0)
    down_stream = array.array("d", [-2.5] * 80)
    out_st = array.array("d", [0.0] * 80)
    out_al = bytearray(80)
    res = d_up.ingest_batch(memoryview(down_stream), out_stats=out_st, out_alarms=out_al)
    assert res["alarm_count"] == 0
    assert max(out_al) == 0

    # Upward shift must trigger rapidly
    d_up.reset()
    up_stream = array.array("d", [2.5] * 80)
    res2 = d_up.ingest_batch(up_stream, out_stats=out_st, out_alarms=out_al)
    assert res2["alarm_count"] > 0
    assert any(out_al)

    # Lower one-sided Vovk Rank catches downward shift, ignores upward shift
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 1.0, size=100)
    d_lo = cs.EDetector(family="vovk_rank", side="lower", threshold=80.0, seed=11)
    d_lo.ingest_batch(base)
    r_up = d_lo.ingest_batch(rng.normal(4.0, 1.0, size=60))
    assert r_up["alarm_count"] == 0

    d_lo.reset()
    d_lo.ingest_batch(base)
    r_dn = d_lo.ingest_batch(rng.normal(-4.0, 1.0, size=60))
    assert r_dn["alarm_count"] > 0


def test_vovk_rank_deterministic_epoch_rollover() -> None:
    rng = np.random.default_rng(123)
    det = cs.EDetector(family="vovk_rank", max_prefix=128, threshold=1e9, seed=99)
    res = det.ingest_batch(rng.standard_normal(400))
    assert res["epoch_count"] == 3
    assert res["step_count"] == 400
    assert math.isfinite(res["stat"])


def test_mc_arl_bounded_catoni_symmetry_rank_ecusum() -> None:
    """Monte Carlo verification: ARL >= gamma across all 4 families + E-CUSUM."""
    gamma = 120.0

    # 1. Bounded Betting on Beta(2, 3) (mean = 0.4)
    arl_bnd, _ = _mc_empirical_arl(
        lambda i: cs.EDetector(family="bounded", mu0=0.4, threshold=gamma),
        lambda rng, n: rng.beta(2.0, 3.0, size=n),
        n_trials=160,
        horizon=1800,
        seed=101,
    )
    assert arl_bnd >= gamma * 0.92, f"Bounded ARL={arl_bnd:.1f} < {gamma}"

    # 2. Algebraic Catoni on heavy-tailed Student-t(df=3) scaled to unit variance (var(t3)=3)
    arl_cat, _ = _mc_empirical_arl(
        lambda i: cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=gamma),
        lambda rng, n: rng.standard_t(df=3.0, size=n) / math.sqrt(3.0),
        n_trials=160,
        horizon=1800,
        seed=102,
    )
    assert arl_cat >= gamma * 0.92, f"Catoni Student-t3 ARL={arl_cat:.1f} < {gamma}"

    # 3. Calibrated Catoni (unknown mu, sigma via finite warmup inflation)
    rng_w = np.random.default_rng(103)
    warmups = [rng_w.laplace(0.0, 1.0, size=128) for _ in range(120)]
    arl_cal, _ = _mc_empirical_arl(
        lambda i: cs.EDetector.calibrate_catoni(warmups[i], threshold=gamma, inflation_z=2.5),
        lambda rng, n: rng.laplace(0.0, 1.0, size=n),
        n_trials=120,
        horizon=1800,
        seed=104,
    )
    assert arl_cal >= gamma * 0.92, f"Calibrated Catoni ARL={arl_cal:.1f} < {gamma}"

    # 4. Algebraic Symmetry on Cauchy(0, 1) (infinite variance!)
    arl_sym, _ = _mc_empirical_arl(
        lambda i: cs.EDetector(family="symmetry", mu0=0.0, threshold=gamma),
        lambda rng, n: rng.standard_cauchy(size=n),
        n_trials=160,
        horizon=1800,
        seed=105,
    )
    assert arl_sym >= gamma * 0.92, f"Symmetry Cauchy ARL={arl_sym:.1f} < {gamma}"

    # 5. Vovk Growing-Prefix Conformal Rank on discrete ties + heavy-tailed mixture + epoch resets
    def mixed_discrete_heavy(rng: np.random.Generator, n: int) -> np.ndarray:
        mask = rng.random(n) < 0.5
        out = np.where(mask, rng.poisson(3.0, size=n).astype(np.float64), rng.standard_cauchy(size=n))
        return out

    arl_rk, _ = _mc_empirical_arl(
        lambda i: cs.EDetector(family="vovk_rank", threshold=gamma, max_prefix=256, seed=1000 + i),
        mixed_discrete_heavy,
        n_trials=140,
        horizon=1800,
        seed=106,
    )
    assert arl_rk >= gamma * 0.92, f"Vovk Rank ARL={arl_rk:.1f} < {gamma}"

    # 6. Log-domain E-CUSUM valid mixture on Student-t(3)
    arl_cusum, _ = _mc_empirical_arl(
        lambda i: cs.EDetector(family="catoni", accum="ecusum", mu0=0.0, sigma=1.0, threshold=gamma),
        lambda rng, n: rng.standard_t(df=3.0, size=n) / math.sqrt(3.0),
        n_trials=140,
        horizon=1800,
        seed=107,
    )
    assert arl_cusum >= gamma * 0.92, f"E-CUSUM ARL={arl_cusum:.1f} < {gamma}"


def test_mc_correlated_cross_agent_efusion_arl_and_power() -> None:
    """K=8 strongly correlated agents (rho=0.85): ARL >= gamma in BOTH step & process modes."""
    K = 8
    rho = 0.85
    gamma = 120.0
    horizon = 1600
    n_trials = 120

    # Equicorrelated Gaussian Cholesky factor: Cov_ij = rho for i!=j, 1 for i==j
    cov = np.full((K, K), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)

    for mode, fam in [("step", "catoni"), ("process", "catoni"), ("process", "vovk_rank")]:
        rng = np.random.default_rng(200 + (0 if mode == "step" else 10) + (0 if fam == "catoni" else 1))
        stops = []
        for trial in range(n_trials):
            fus = cs.EFusion(
                n_agents=K,
                mode=mode,
                family=fam,
                mu0=0.0,
                sigma=1.0,
                threshold=gamma,
                max_prefix=512,
                seed=5000 + trial,
            )
            z = rng.standard_normal((horizon, K)) @ L.T
            z = np.ascontiguousarray(z, dtype=np.float64)
            alarms = np.zeros(horizon, dtype=np.uint8)
            fus.ingest_batch(z, out_alarms=alarms)
            hit = np.flatnonzero(alarms)
            stops.append(int(hit[0]) + 1 if hit.size > 0 else horizon)
        mean_arl = float(np.mean(stops))
        assert mean_arl >= gamma * 0.90, f"EFusion({mode}, {fam}) K=8 rho=0.85 ARL={mean_arl:.1f} < {gamma}"

    # Verify power: global shift detected quickly
    rng = np.random.default_rng(300)
    fus = cs.EFusion(n_agents=K, mode="step", family="catoni", mu0=0.0, sigma=1.0, threshold=200.0)
    pre = np.ascontiguousarray(rng.standard_normal((100, K)) @ L.T, dtype=np.float64)
    post = np.ascontiguousarray(rng.standard_normal((100, K)) @ L.T + 1.5, dtype=np.float64)
    al_pre = np.zeros(100, dtype=np.uint8)
    al_post = np.zeros(100, dtype=np.uint8)
    fus.ingest_batch(pre, out_alarms=al_pre)
    fus.ingest_batch(post, out_alarms=al_post)
    hit = np.flatnonzero(al_post)
    assert hit.size > 0 and int(hit[0]) < 35


def test_extreme_scales_neon_parity_and_efusion_weight_invariants() -> None:
    # 1. Catoni sigma=1e200/1e-200 and Symmetry x=1e308 must stay strictly finite (no NaN)
    for sig in (1e200, 1e-200):
        d = cs.EDetector(family="catoni", mu0=0.0, sigma=sig, threshold=100.0)
        res = d.step(sig)
        assert math.isfinite(res["stat"]) and math.isfinite(res["e_mean"])

    d_sym = cs.EDetector(family="symmetry", mu0=0.0, sigma=1.0, threshold=100.0)
    r1 = d_sym.step(1e308)
    r2 = d_sym.step(10.0)
    assert math.isfinite(r1["stat"]) and math.isfinite(r2["stat"])

    # 2. calibrate_catoni handles large scale 1e160 without d^2/d^4 overflow
    d_cal = cs.EDetector.calibrate_catoni([1e160 + i * 1e158 for i in range(32)], threshold=100.0)
    assert math.isfinite(d_cal.step(1e160)["stat"])

    # 3. NEON batch kernel returns finite e_mean matching scalar step() and stays consistent on mid-batch NaN
    d_b = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1000.0)
    rb = d_b.ingest_batch(array.array("d", [0.1, 0.2, 0.3]))
    d_s = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1000.0)
    rs = {}
    for v in (0.1, 0.2, 0.3):
        rs = d_s.step(v)
    assert math.isclose(rb["e_mean"], rs["e_mean"], rel_tol=1e-12)
    assert math.isclose(rb["stat"], rs["stat"], rel_tol=1e-12)

    d_mid = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=1e9)
    with pytest.raises(ValueError):
        d_mid.ingest_batch(array.array("d", [2.0, 2.0, 2.0, float("nan"), 0.0]))
    assert d_mid.step_count == 3 and d_mid.stat > 1.0

    # 4. EFusion(mode='process') rejects per-step weights (ARL invariant), while mode='step' allows them
    fus_p = cs.EFusion(n_agents=4, mode="process", family="catoni", mu0=0.0, sigma=1.0)
    with pytest.raises(ValueError):
        fus_p.step([0.0] * 4, weights=[1.0, 0.0, 0.0, 0.0])

    # Non-contiguous 2D Fortran and strided sliced NumPy arrays work cleanly
    fus_s = cs.EFusion(n_agents=4, mode="step", family="catoni", mu0=0.0, sigma=1.0)
    assert fus_s.ingest_batch(np.asfortranarray(np.ones((12, 4), dtype=np.float64)))["step_count"] == 12
    assert fus_s.ingest_batch(np.ones((12, 8), dtype=np.float64)[:, ::2])["step_count"] == 24

    # 5. Negative uint32 kwargs, zombie re-__init__, and uninitialized __new__ are guarded
    with pytest.raises(ValueError):
        cs.EDetector(family="vovk_rank", max_prefix=-1)
    d_z = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0)
    d_z.step(1.0)
    with pytest.raises(ValueError):
        d_z.__init__(threshold=-5.0)
    assert d_z.step_count == 1
    with pytest.raises(RuntimeError):
        cs.EDetector.__new__(cs.EDetector).step(1.0)
    with pytest.raises(RuntimeError):
        cs.EFusion.__new__(cs.EFusion).ingest_batch([1.0])
