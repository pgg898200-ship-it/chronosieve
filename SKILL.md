---
name: chronosieve
description: Zero-dependency C11/SIMD streaming change detection (ARL-guaranteed E-Detectors & Cross-Agent E-Fusion), Pre-Control E-Reset Conformal PID + COMA intervals, Sleeping Fixed-Share BOA multi-agent fusion with rogue agent isolation, and 91.6%–96.3% (BPE-proxy) token-reducing Swing PLA Markdown digests + lossless zoom() via Python, CLI, or MCP.
---

# ChronoSieve Agent Skill

Use `chronosieve` whenever you process streaming time series, telemetry, sensor feeds, financial ticks, or multi-agent forecasts and need:
1. **Distribution-free changepoint detection** with certified false-alarm control ($\text{ARL} \ge \gamma$), even on heavy-tailed/Cauchy/discrete/correlated streams (verified on the 1898 Nile Aswan dam changepoint with 0 pre-1899 false alarms).
2. **Non-stationary conformal prediction intervals** that recover immediately after volatility shocks without sacrificing asymptotic $1-\alpha$ coverage.
3. **Asynchronous multi-agent forecast fusion** (fast System-1 + slow System-2 LLMs) that automatically detects and quarantines hallucinating agents.
4. **Massive LLM token reduction** (91.6% BPE-proxy / 88.3% `chars/4` full-series on Mauna Loa CO2 without wake gating; 96.3% BPE-proxy / 95.0% `chars/4` with wake gating on 10k telemetry) by emitting compact ~195–200-token (`chars/4`) Markdown digests and drilling down losslessly via `zoom(channel, t0, t1)`.

## Quick Setup

```bash
python3 -m pip install --no-build-isolation -e .
make cli   # optional 101 KB native C11 CLI
```

MCP server command (zero dependencies):
```bash
python3 -m chronosieve.mcp_server
```

## Decision Rules

1. **Always use zero-copy batch APIs** (`ingest_batch`, `fuse_batch`, `digest_batch`) with `array.array('d')`, `memoryview`, or `numpy.ndarray` on streams $> 10^3$ ticks (`8.47–9.03 ns/tick` Python batch / `8.82 ns/tick` Native CLI vs `347.8 ns/tick` for CPython scalar `step()`).
2. **Pick the right E-detector family**:
   - Known $(\mu_0, \sigma)$ with finite variance $\to$ `EDetector(family="catoni", mu0=..., sigma=...)` (fastest: `8.47 ns/tick` `array.array`, `9.03 ns/tick` NumPy, `8.82 ns/tick` Native CLI).
   - Bounded in $[0, 1]$ with known $\mu_0$ $\to$ `family="bounded"`.
   - Symmetric around $\mu_0$, unknown/infinite variance $\to$ `family="symmetry"`.
   - Unknown $(\mu, \sigma)$, discrete ties, multimodal, or heavy tails $\to$ `family="vovk_rank"` (or `family="auto"`).
   - Correlated multi-agent streams ($K$ agents) $\to$ `EFusion(mode="step")` (Catoni) or `EFusion(mode="process")` (Vovk Rank).
3. **Never feed closed-loop Conformal PID miscoverage into an E-detector**: `ConformalPID(enable_ereset=True)` enforces this automatically by running Vovk Rank strictly on exogenous $s_t = |y_t - \hat{y}_t|$, resetting only bounded $\hat{q}^D \in [0, B]$, and enforcing $W_{\text{cal}} \ge 32$.
4. **For multi-agent intervals**, merge with `coma_merge(lowers, uppers, weights=boa.weights(), randomized=True)` for strictly tighter intervals that preserve Gasparin–Ramdas $\ge 1 - 2\alpha$ coverage under arbitrary dependence.
5. **For System-2 LLM triage**, ingest streams through `ChronoSieve` or the MCP server (`chronosieve_ingest`). Use `anomaly_burst=1` if you want immediate LLM wake-ups on isolated 1-tick spikes (100% wake-packet spike recall), or `anomaly_burst=2` to wake only on certified shifts and $\ge 2$-tick anomaly bursts (1-tick spikes remain pinned verbatim in `StreamDigest` for `chronosieve_zoom(channel, t0, t1)`).
