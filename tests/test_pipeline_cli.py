"""End-to-end tests for unified ChronoSieve pipeline, zero-copy buffers, and native C11 CLI."""

from __future__ import annotations

import array
import importlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from unittest import mock

import chronosieve as cs
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLI_BIN = ROOT / "chronosieve"


def test_unified_pipeline_end_to_end() -> None:
    rng = np.random.default_rng(31415)
    pipe = cs.ChronoSieve(
        channel="reactor_temp",
        n_agents=3,
        alpha=0.10,
        epsilon=0.05,
        edet_threshold=150.0,
        rogue_threshold=100.0,
        anomaly_burst=2,
        wake_cooldown=5,
    )

    wakes_stationary = 0
    wakes_post_shift = 0
    saw_rogue = False
    saw_shift = False
    last_md = None

    for t in range(320):
        # Stationary regime [0..149], shift + rogue agent 2 at t >= 150
        true_y = (0.0 if t < 150 else 3.0) + float(rng.normal(0.0, 0.2))
        p0 = true_y + float(rng.normal(0.0, 0.15))
        p1 = true_y + float(rng.normal(0.0, 0.20))
        p2 = (true_y + float(rng.normal(0.0, 0.18))) if t < 150 else (true_y + 4.5)
        active = [1, 1 if (t % 3 == 0) else 0, 1]
        ivs = [(p0 - 0.5, p0 + 0.5), (p1 - 0.5, p1 + 0.5), (p2 - 0.5, p2 + 0.5)]

        res = pipe.step(float(t), true_y, preds=[p0, p1, p2], active=active, agent_intervals=ivs)
        assert res["coma"] is not None
        if t < 150 and res["should_wake"]:
            wakes_stationary += 1
        if t >= 150:
            if res["should_wake"]:
                wakes_post_shift += 1
                last_md = res["digest_markdown"]
            if res["shift_detected"]:
                saw_shift = True
            if res["any_rogue"] and res["rogue_agent"] == 2:
                saw_rogue = True

    assert saw_shift is True
    assert saw_rogue is True
    assert wakes_post_shift > 0
    assert wakes_stationary < 15  # Very quiet during stationary regime
    assert last_md is not None and "reactor_temp" in last_md

    # Lossless zoom over shift boundary [148..155]
    z = pipe.zoom(148.0, 155.0)
    assert z["truncated"] is False
    assert len(z["records"]) == 8
    assert z["records"][0]["t"] == 148.0

    st = pipe.status()
    assert st["steps"] == 320
    assert st["agent_rogue_flags"][2] is True


def test_stdlib_buffer_zero_dependency_roundtrip() -> None:
    """Confirm zero-copy batch APIs work purely with Python stdlib buffers."""
    n = 512
    xs = array.array("d", [0.1 * (i % 7 - 3) for i in range(n)])
    ys = array.array("d", [0.0] * n)
    out_st = array.array("d", [0.0] * n)
    out_al = bytearray(n)

    det = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=500.0)
    res = det.ingest_batch(memoryview(xs), out_stats=memoryview(out_st), out_alarms=memoryview(out_al))
    assert res["step_count"] == n

    cp = cs.ConformalPID(alpha=0.10)
    out_q = array.array("d", [0.0] * n)
    out_mc = bytearray(n)
    cres = cp.ingest_batch(xs, ys, out_q_next=out_q, out_miscov=out_mc)
    assert cres["step_count"] == n


def test_native_cli_help_describes_stream_formats() -> None:
    for name, expected in (
        ("detect", "zero-based tick"),
        ("conformal", "observed,predicted"),
        ("digest", "time,value"),
        ("bench", "generated stream"),
    ):
        for flag in ("-h", "--help"):
            result = subprocess.run([str(CLI_BIN), name, flag], capture_output=True, text=True, check=True)
            assert f"Usage: chronosieve {name}" in result.stdout
            assert expected in result.stdout
            assert not result.stderr


def test_native_cli_binary_size_and_subcommands() -> None:
    assert CLI_BIN.exists(), f"CLI binary not built at {CLI_BIN}"
    size_bytes = os.path.getsize(CLI_BIN)
    assert size_bytes < 150 * 1024, f"CLI binary size {size_bytes} exceeds 150 KB target"

    # 1. info
    p_info = subprocess.run([str(CLI_BIN), "info"], capture_output=True, text=True, check=True)
    assert "chronosieve 0.1.0" in p_info.stdout

    # 2. detect --json on shifted stream
    stream = "\n".join(["0.0"] * 50 + ["3.0"] * 50) + "\n"
    p_det = subprocess.run(
        [
            str(CLI_BIN),
            "detect",
            "--family",
            "catoni",
            "--mu0",
            "0",
            "--sigma",
            "1",
            "--threshold",
            "100",
            "--json",
        ],
        input=stream,
        capture_output=True,
        text=True,
        check=True,
    )
    det_json = json.loads(p_det.stdout)
    assert det_json["ticks"] == 100
    assert det_json["alarms"] > 0
    assert 50 <= det_json["first_alarm"] < 75

    # 3. conformal --json
    pairs = "\n".join(f"{0.5 if i % 2 == 0 else -0.5},0.0" for i in range(200)) + "\n"
    p_cp = subprocess.run(
        [str(CLI_BIN), "conformal", "--alpha", "0.1", "--json"],
        input=pairs,
        capture_output=True,
        text=True,
        check=True,
    )
    cp_json = json.loads(p_cp.stdout)
    assert cp_json["ticks"] == 200
    assert 0.80 <= cp_json["coverage"] <= 1.0

    # 4. digest markdown + lossless zoom
    series = "\n".join(f"{i},{i * 0.1 if i != 50 else 25.0}" for i in range(100)) + "\n"
    p_dig = subprocess.run(
        [str(CLI_BIN), "digest", "--eps", "0.05", "--channel", "cli_test"],
        input=series,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "## ChronoSieve Digest: `cli_test`" in p_dig.stdout

    p_zoom = subprocess.run(
        [str(CLI_BIN), "digest", "--eps", "0.05", "--zoom", "49:51"],
        input=series,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln for ln in p_zoom.stdout.strip().splitlines() if ln]
    assert lines[0] == "t,y,y_hat,q_t,flags"
    assert len(lines) == 4  # header + 49, 50, 51
    assert "50,25," in lines[2]

    # 5. CLI bench with small --n (regression test for heap-buffer-overflow when N < 64) & bad flag rejection
    p_b = subprocess.run([str(CLI_BIN), "bench", "--n", "4"], capture_output=True, text=True, check=True)
    assert "ChronoSieve Native C11 Benchmark" in p_b.stdout
    p_bad = subprocess.run(
        [str(CLI_BIN), "detect", "--family", "nonexistent"], capture_output=True, text=True
    )
    assert p_bad.returncode == 1


def test_pipeline_no_coma_lookahead_atomic_validation_and_monitor_level() -> None:
    # 1. COMA must use strictly F_{t-1}-predictable pre-update weights (zero lookahead from y_t)
    pipe_c = cs.ChronoSieve(channel="coma_causal", n_agents=2)
    rc = pipe_c.step(
        0.0, 100.0, preds=[0.0, 100.0], agent_intervals=[(0.0, 1.0), (99.0, 101.0)], coma_tau=0.5
    )
    assert rc["coma"]["intervals"] == []

    # 2. Failed non-monotonic step must NOT mutate conformal/step_count state
    pipe_v = cs.ChronoSieve(channel="atomic", n_agents=1)
    pipe_v.step(10.0, 1.0)
    q_before = pipe_v.conformal.q
    try:
        pipe_v.step(5.0, 2.0)
        raise AssertionError("Expected ValueError on non-monotonic timestamp")
    except ValueError:
        pass
    assert pipe_v.status()["steps"] == 1
    assert pipe_v.conformal.q == q_before

    # 3. monitor_level=True preserves warmup B-tree baseline & combined ARL >= gamma
    rng = np.random.default_rng(88)
    gamma = 80.0
    stops = []
    for trial in range(80):
        p = cs.ChronoSieve(
            channel="ml", n_agents=1, edet_threshold=gamma, monitor_level=True, seed=500 + trial
        )
        ys = rng.standard_normal(320)
        stopped = 320
        for t in range(320):
            r = p.step(float(t), float(ys[t]), preds=[0.0])
            # Combined Bonferroni-normalized edet_stat must cross gamma iff shift_detected fired
            if r["shift_detected"]:
                assert r["edet_stat"] >= gamma
                stopped = t + 1
                break
            else:
                assert r["edet_stat"] < gamma
        stops.append(stopped)
    assert float(np.mean(stops)) >= gamma


def _load_bench_module(name: str, filename: str):
    path = ROOT / "benchmarks" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llm_eval_scenario_captures_ground_truth() -> None:
    """Verify bench_llm_eval emits wake-up digests capturing shift, t=180 y=95.5 spike, and agent_2."""
    mod = _load_bench_module("bench_llm_eval_mod", "bench_llm_eval.py")

    # Test default seed plus previously failing regression seeds (2008, 2040, 2049, 2099)
    for seed in (2026, 2008, 2040, 2049, 2099):
        scenario = mod.build_eval_scenario(seed=seed)
        md = scenario["digest_markdown"]
        assert "t=180.00" in md
        assert "95.5" in md
        assert "rogue=agent_2" in md
        assert "180.00..180.00: 95.5->95.5" in md
        assert any(150 <= t <= 170 for t in scenario["shift_ticks"])
        assert scenario["rogue_flags"] == [False, False, True]
        assert 180 in scenario["wake_ticks"]
        edet_pairs = re.findall(r"E=([0-9.]+)/([0-9.]+)", md)
        assert any(float(s) >= float(thr) for s, thr in edet_pairs)


def test_real_world_benchmarks_optional_import_and_markdown_fidelity() -> None:
    """Verify graceful skip when statsmodels is absent and strict Markdown/causal invariants when present."""
    bt = _load_bench_module("bench_tokens_mod", "bench_tokens.py")
    bm = _load_bench_module("bench_math_proofs_mod", "bench_math_proofs.py")

    # 1. Optional statsmodels ImportError must degrade cleanly (0 runtime dependency contract)
    orig_import = importlib.import_module

    def _block_statsmodels(name: str, package: str | None = None):
        if name == "statsmodels.api":
            raise ImportError("mocked missing statsmodels")
        return orig_import(name, package)

    with mock.patch.object(importlib, "import_module", side_effect=_block_statsmodels):
        co2_skip = bt.run_mauna_loa_co2_benchmark(epsilon=1.0)
        rw_skip = bm._run_real_world_statsmodels()
    assert co2_skip["available"] is False
    assert rw_skip["available"] is False

    # 2. When statsmodels is installed, verify exact C L_inf, tight Markdown reconstruction, and causality
    co2 = bt.run_mauna_loa_co2_benchmark(epsilon=1.0)
    if co2["available"]:
        pla = co2["chronosieve_swing_pla_digest"]
        assert pla["linf_verified"] is True
        assert pla["max_linf_err_ppm"] <= 1.0
        # Upgraded %.6g Markdown formatting keeps text-reconstructed error within 0.001 ppm of eps=1.0
        assert pla["markdown_reconstruct_max_linf_ppm"] <= 1.001
        assert pla["reduction_vs_json_bpe_pct"] > 90.0
        assert pla["reduction_vs_tsln_bpe_pct"] > 70.0

    rw = bm._run_real_world_statsmodels()
    if rw["available"]:
        nile = rw["nile_aswan_1898"]
        assert nile["vovk_rank_gamma50"]["pre_1899_false_alarms"] == 0
        assert nile["vovk_rank_gamma100"]["pre_1899_false_alarms"] == 0
        assert nile["calibrated_catoni_gamma100"]["pre_1899_false_alarms"] == 0
        assert nile["calibrated_catoni_gamma100"]["side"] == "two_sided"
        assert nile["vovk_rank_gamma50"]["first_alarm_year"] == 1905
        sun = rw["sunspots_conformal_pid"]
        assert sun["eval_obs"] == 277  # strictly causal out-of-sample 1732-2008
        assert sun["ereset_resets"] >= 1
        assert abs(sun["ereset_coverage"] - 0.90) < abs(sun["pure_pid_coverage"] - 0.90)
        assert abs(sun["ereset_coverage"] - 0.90) < abs(sun["rolling_d_pid_coverage"] - 0.90)

