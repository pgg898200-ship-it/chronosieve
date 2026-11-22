#!/usr/bin/env python3
"""Reproducible Monte Carlo mathematical verification suite for ChronoSieve."""

from __future__ import annotations

import importlib
import json
import math
from typing import Any

import chronosieve as cs
import numpy as np


def _run_arl_and_add(
    make_det,
    h0_sampler,
    h1_sampler,
    *,
    n_trials: int = 200,
    horizon: int = 2500,
    pre_len: int = 100,
    seed: int = 2026,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    h0_stops = []
    for i in range(n_trials):
        det = make_det(i)
        xs = np.ascontiguousarray(h0_sampler(rng, horizon), dtype=np.float64)
        al = np.zeros(horizon, dtype=np.uint8)
        det.ingest_batch(xs, out_alarms=al)
        hit = np.flatnonzero(al)
        h0_stops.append(int(hit[0]) + 1 if hit.size > 0 else horizon)

    delays = []
    for i in range(n_trials):
        det = make_det(10000 + i)
        pre = np.ascontiguousarray(h0_sampler(rng, pre_len), dtype=np.float64)
        post = np.ascontiguousarray(h1_sampler(rng, 400), dtype=np.float64)
        al_pre = np.zeros(pre_len, dtype=np.uint8)
        al_post = np.zeros(400, dtype=np.uint8)
        det.ingest_batch(pre, out_alarms=al_pre)
        if np.any(al_pre):
            det.reset()
            det.ingest_batch(pre)
        det.ingest_batch(post, out_alarms=al_post)
        hit = np.flatnonzero(al_post)
        delays.append(int(hit[0]) + 1 if hit.size > 0 else 400)

    return {
        "arl_lower_bound": float(np.mean(h0_stops)),
        "arl_median": float(np.median(h0_stops)),
        "add_mean": float(np.mean(delays)),
        "add_median": float(np.median(delays)),
    }


def run_math_proofs() -> dict[str, Any]:
    gamma = 200.0
    out: dict[str, Any] = {"gamma": gamma, "edetectors": {}, "efusion": {}}

    # 1. All 4 E-Detector families + E-CUSUM
    out["edetectors"]["bounded_beta23"] = _run_arl_and_add(
        lambda i: cs.EDetector(family="bounded", mu0=0.4, threshold=gamma),
        lambda rng, n: rng.beta(2.0, 3.0, size=n),
        lambda rng, n: rng.beta(4.0, 2.0, size=n),
        seed=1,
    )
    out["edetectors"]["catoni_student_t3"] = _run_arl_and_add(
        lambda i: cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=gamma),
        lambda rng, n: rng.standard_t(3.0, size=n) / math.sqrt(3.0),
        lambda rng, n: rng.standard_t(3.0, size=n) / math.sqrt(3.0) + 1.5,
        seed=2,
    )
    out["edetectors"]["symmetry_cauchy"] = _run_arl_and_add(
        lambda i: cs.EDetector(family="symmetry", mu0=0.0, threshold=gamma),
        lambda rng, n: rng.standard_cauchy(size=n),
        lambda rng, n: rng.standard_cauchy(size=n) + 2.0,
        seed=3,
    )
    def _mix_h0(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.where(rng.random(n) < 0.5, rng.poisson(3.0, n).astype(float), rng.standard_cauchy(n))

    out["edetectors"]["vovk_rank_discrete_cauchy"] = _run_arl_and_add(
        lambda i: cs.EDetector(family="vovk_rank", max_prefix=512, threshold=gamma, seed=500 + i),
        _mix_h0,
        lambda rng, n: _mix_h0(rng, n) + 2.0,
        seed=4,
    )
    out["edetectors"]["ecusum_catoni_t3"] = _run_arl_and_add(
        lambda i: cs.EDetector(family="catoni", accum="ecusum", mu0=0.0, sigma=1.0, threshold=gamma),
        lambda rng, n: rng.standard_t(3.0, size=n) / math.sqrt(3.0),
        lambda rng, n: rng.standard_t(3.0, size=n) / math.sqrt(3.0) + 1.5,
        seed=5,
    )

    # 2. Correlated Cross-Agent E-Fusion (K=8, rho=0.85)
    K, rho = 8, 0.85
    cov = np.full((K, K), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)

    for label, mode, fam in [
        ("k8_rho85_step_catoni", "step", "catoni"),
        ("k8_rho85_process_rank", "process", "vovk_rank"),
    ]:
        rng = np.random.default_rng(100 if mode == "step" else 101)
        stops = []
        delays = []
        for trial in range(160):
            fus = cs.EFusion(
                n_agents=K,
                mode=mode,
                family=fam,
                mu0=0.0,
                sigma=1.0,
                threshold=gamma,
                max_prefix=512,
                seed=7000 + trial,
            )
            z = np.ascontiguousarray(rng.standard_normal((2200, K)) @ L.T, dtype=np.float64)
            al = np.zeros(2200, dtype=np.uint8)
            fus.ingest_batch(z, out_alarms=al)
            hit = np.flatnonzero(al)
            stops.append(int(hit[0]) + 1 if hit.size > 0 else 2200)

            fus.reset()
            pre = np.ascontiguousarray(rng.standard_normal((100, K)) @ L.T, dtype=np.float64)
            post = np.ascontiguousarray(rng.standard_normal((300, K)) @ L.T + 1.5, dtype=np.float64)
            fus.ingest_batch(pre)
            al_p = np.zeros(300, dtype=np.uint8)
            fus.ingest_batch(post, out_alarms=al_p)
            hp = np.flatnonzero(al_p)
            delays.append(int(hp[0]) + 1 if hp.size > 0 else 300)

        out["efusion"][label] = {
            "arl_lower_bound": float(np.mean(stops)),
            "add_mean": float(np.mean(delays)),
        }

    # 3. Pre-Control E-Reset Conformal PID vs Pure Conformal PID
    rng = np.random.default_rng(2026)
    T = 4000
    sigmas = np.ones(T)
    sigmas[800:2000] = 5.0
    sigmas[2800:] = 4.0
    y = rng.normal(0.0, sigmas)
    yp = np.zeros(T)

    cp_er = cs.ConformalPID(
        alpha=0.10, eta=0.05, ki=1.0, init_q=1.6, edet_threshold=300.0, score_window=128, enable_ereset=True
    )
    cp_rd = cs.ConformalPID(
        alpha=0.10, eta=0.05, ki=1.0, init_q=1.6, score_window=128, enable_ereset=False
    )
    cp_pu = cs.ConformalPID(alpha=0.10, eta=0.05, ki=1.0, init_q=1.6, score_window=0, enable_ereset=False)
    mc_er = np.zeros(T, dtype=np.uint8)
    mc_rd = np.zeros(T, dtype=np.uint8)
    mc_pu = np.zeros(T, dtype=np.uint8)
    r_er = cp_er.ingest_batch(y, yp, out_miscov=mc_er)
    r_rd = cp_rd.ingest_batch(y, yp, out_miscov=mc_rd)
    r_pu = cp_pu.ingest_batch(y, yp, out_miscov=mc_pu)
    out["conformal_pid"] = {
        "target_coverage": 0.90,
        "ereset_full_coverage": r_er["emp_coverage"],
        "rolling_d_pid_full_coverage": r_rd["emp_coverage"],
        "pure_pid_full_coverage": r_pu["emp_coverage"],
        "ereset_shock_100t_coverage": float(1.0 - np.mean(mc_er[800:900])),
        "rolling_d_pid_shock_100t_coverage": float(1.0 - np.mean(mc_rd[800:900])),
        "pure_pid_shock_100t_coverage": float(1.0 - np.mean(mc_pu[800:900])),
        "ereset_resets": int(r_er["reset_count"]),
    }

    # 4. Online COMA (K=5 correlated agents)
    rng = np.random.default_rng(42)
    z_q = 1.6448536269514722
    N_coma = 3000
    cov_d = cov_r = 0
    w_d = w_r = 0.0
    for i in range(N_coma):
        yt = rng.normal(0.0, 1.0)
        sh = rng.normal(0.0, 0.3)
        c = sh + rng.normal(0.0, 0.25, size=5)
        hw = z_q * math.sqrt(1.0 + 0.09 + 0.0625)
        rd = cs.coma_merge(c - hw, c + hw, tau=0.5, randomized=False)
        rr = cs.coma_merge(c - hw, c + hw, randomized=True, seed=20000 + i)
        if any(a <= yt <= b for a, b in rd["intervals"]):
            cov_d += 1
        if any(a <= yt <= b for a, b in rr["intervals"]):
            cov_r += 1
        w_d += sum(b - a for a, b in rd["intervals"])
        w_r += sum(b - a for a, b in rr["intervals"])
    out["coma_k5"] = {
        "theory_floor": 0.80,
        "det_coverage": cov_d / N_coma,
        "rand_coverage": cov_r / N_coma,
        "det_mean_width": w_d / N_coma,
        "rand_mean_width": w_r / N_coma,
    }

    # 5. Sleeping BOA + Rogue attribution
    rng = np.random.default_rng(7)
    T_f, K_f = 300, 6
    y_f = np.sin(np.arange(T_f) * 0.04)
    p_f = np.zeros((T_f, K_f))
    a_f = np.ones((T_f, K_f), dtype=np.uint8)
    for k in range(K_f):
        p_f[:, k] = y_f + rng.normal(0.0, 0.15 + 0.05 * k, size=T_f)
    for t in range(T_f):
        if t % 4 != 0:
            a_f[t, 5] = 0
            p_f[t, 5] = float("nan")
    p_f[120:, 2] += 3.0  # Agent 2 hallucinates!
    fus = cs.SleepingBOA(n_agents=K_f, rogue_threshold=150.0, cal_window=48, quarantine_rogue=True)
    out_p = np.zeros(T_f)
    out_r = np.zeros(T_f, dtype=np.uint8)
    fus.fuse_batch(p_f, y_f, active_matrix=a_f, out_preds=out_p, out_rogue_any=out_r)
    _, rflags = fus.rogue_stats()
    first_r = int(np.flatnonzero(out_r)[0])
    out["sleeping_boa"] = {
        "rogue_detected_delay": first_r - 120,
        "rogue_isolated_exact": bool(rflags[2] and not any(rflags[j] for j in range(K_f) if j != 2)),
        "post_quarantine_mae": float(np.mean(np.abs(out_p[140:] - y_f[140:]))),
    }

    # 6. Swing PLA L_inf guarantee
    rng = np.random.default_rng(99)
    N_p = 10000
    ts = np.arange(N_p, dtype=np.float64)
    ys = np.sin(ts * 0.01) + rng.normal(0.0, 0.02, size=N_p)
    flags = np.zeros(N_p, dtype=np.uint32)
    anom_idx = [1500, 5000, 8200]
    for idx in anom_idx:
        ys[idx] += 4.0
        flags[idx] = cs.FLAG_ANOMALY
    dig = cs.StreamDigest(epsilon=0.05, ring_capacity=16384, max_segments=16384)
    dig.digest_batch(ts, ys, y_hats=ys - 0.1, qs=np.full(N_p, 0.5), flags=flags)
    rec = np.zeros(N_p)
    dig.reconstruct(ts, out_y=rec)
    out["swing_pla"] = {
        "epsilon": 0.05,
        "max_linf_err": float(np.max(np.abs(rec - ys))),
        "max_anomaly_err": float(max(abs(rec[i] - ys[i]) for i in anom_idx)),
        "compression_ratio": N_p / dig.segment_count,
        "segments": dig.segment_count,
    }

    # 7. Real-world statsmodels verification (Nile 1898 Aswan dam + Sunspots Conformal PID)
    out["real_world_statsmodels"] = _run_real_world_statsmodels()

    return out


def _run_real_world_statsmodels() -> dict[str, Any]:
    """Real-world verification on classic statsmodels datasets (gracefully skipped if absent)."""
    try:
        sm = importlib.import_module("statsmodels.api")
    except ImportError:
        return {"available": False, "reason": "statsmodels not installed"}

    # 7a. Nile annual flow (1871-1970, N=100, famous 1898 Aswan Low Dam changepoint)
    nile_df = sm.datasets.nile.load_pandas().data
    years = nile_df["year"].to_numpy(dtype=np.float64)
    vol = np.ascontiguousarray(nile_df["volume"].to_numpy(dtype=np.float64))
    nile_out: dict[str, Any] = {
        "dataset": "statsmodels.datasets.nile (Nile Annual Flow 1871-1970, 1898 Aswan Dam)",
        "n_obs": int(len(vol)),
        "changepoint_year": 1898,
        "pre_1899_mean": float(np.mean(vol[:28])),
        "post_1898_mean": float(np.mean(vol[28:])),
    }
    for key, gamma in [("vovk_rank_gamma50", 50.0), ("vovk_rank_gamma100", 100.0)]:
        det_rk = cs.EDetector(family="vovk_rank", accum="esr", side="two_sided", threshold=gamma, seed=2026)
        al = np.zeros(len(vol), dtype=np.uint8)
        st = np.zeros(len(vol), dtype=np.float64)
        det_rk.ingest_batch(vol, out_stats=st, out_alarms=al)
        hits = np.flatnonzero(al)
        first_yr = int(years[hits[0]]) if hits.size > 0 else None
        pre_fa = int(np.sum(al[:28]))  # full pre-drop period 1871..1898 (drop begins 1899)
        nile_out[key] = {
            "gamma": gamma,
            "first_alarm_year": first_yr,
            "delay_years": (first_yr - 1898) if first_yr is not None else None,
            "pre_1899_false_alarms": pre_fa,
            "max_stat": float(np.max(st)),
        }

    # Default two-sided Calibrated Catoni (1871-1890 warmup, gamma=100, default z=2.5, 0 direction priors)
    det_cat = cs.EDetector.calibrate_catoni(vol[:20], threshold=100.0)
    al_c = np.zeros(len(vol) - 20, dtype=np.uint8)
    st_c = np.zeros(len(vol) - 20, dtype=np.float64)
    det_cat.ingest_batch(vol[20:], out_stats=st_c, out_alarms=al_c)
    hits_c = np.flatnonzero(al_c)
    first_yr_c = int(years[20 + hits_c[0]]) if hits_c.size > 0 else None
    nile_out["calibrated_catoni_gamma100"] = {
        "gamma": 100.0,
        "side": "two_sided",
        "inflation_z": 2.5,
        "warmup_years": "1871-1890",
        "first_alarm_year": first_yr_c,
        "delay_years": (first_yr_c - 1898) if first_yr_c is not None else None,
        "pre_1899_false_alarms": int(np.sum(al_c[:8])),  # full pre-drop period 1891..1898
        "max_stat": float(np.max(st_c)),
    }

    # 7b. Annual Sunspots (1700-2008, N=309): strictly causal warmup (1700-1731) + out-of-sample
    #     eval (1732-2008, N=277) comparing E-Reset PID vs Rolling-D PID vs Pure P+I PID (zero lookahead!)
    sun_df = sm.datasets.sunspots.load_pandas().data
    sun_y = np.ascontiguousarray(sun_df["SUNACTIVITY"].to_numpy(dtype=np.float64))
    warmup_n = 32
    scale_w = float(np.std(sun_y[:warmup_n]))
    init_q_w = float(np.quantile(np.abs(np.diff(sun_y[:warmup_n])), 0.90))
    y_eval = np.ascontiguousarray(sun_y[warmup_n:], dtype=np.float64)
    yp_eval = np.ascontiguousarray(sun_y[warmup_n - 1 : -1], dtype=np.float64)

    cp_er = cs.ConformalPID(
        alpha=0.10,
        eta=0.05 * scale_w,
        ki=scale_w,
        init_q=init_q_w,
        edet_threshold=100.0,
        warmup_cal=32,
        score_window=64,
        enable_ereset=True,
        seed=2026,
    )
    cp_rd = cs.ConformalPID(
        alpha=0.10,
        eta=0.05 * scale_w,
        ki=scale_w,
        init_q=init_q_w,
        score_window=64,
        enable_ereset=False,
    )
    cp_pu = cs.ConformalPID(
        alpha=0.10,
        eta=0.05 * scale_w,
        ki=scale_w,
        init_q=init_q_w,
        score_window=0,
        enable_ereset=False,
    )
    q_er = np.zeros(len(y_eval), dtype=np.float64)
    q_rd = np.zeros(len(y_eval), dtype=np.float64)
    q_pu = np.zeros(len(y_eval), dtype=np.float64)
    r_er = cp_er.ingest_batch(y_eval, yp_eval, out_q_used=q_er)
    r_rd = cp_rd.ingest_batch(y_eval, yp_eval, out_q_used=q_rd)
    r_pu = cp_pu.ingest_batch(y_eval, yp_eval, out_q_used=q_pu)

    sun_out = {
        "dataset": "statsmodels.datasets.sunspots (Annual Sunspot Activity 1700-2008)",
        "total_obs": int(len(sun_y)),
        "warmup_years": "1700-1731 (N=32, causal scale & init_q only)",
        "eval_years": "1732-2008",
        "eval_obs": int(len(y_eval)),
        "target_coverage": 0.90,
        "ereset_coverage": float(r_er["emp_coverage"]),
        "rolling_d_pid_coverage": float(r_rd["emp_coverage"]),
        "pure_pid_coverage": float(r_pu["emp_coverage"]),
        "ereset_resets": int(r_er["reset_count"]),
        "ereset_mean_q": float(np.mean(q_er)),
        "rolling_d_pid_mean_q": float(np.mean(q_rd)),
        "pure_pid_mean_q": float(np.mean(q_pu)),
    }

    return {
        "available": True,
        "nile_aswan_1898": nile_out,
        "sunspots_conformal_pid": sun_out,
    }


def main() -> None:
    print(json.dumps(run_math_proofs(), indent=2))


if __name__ == "__main__":
    main()
