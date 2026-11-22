---
name: chronosieve
description: Zero-dependency C11/SIMD streaming change detection (492M ticks/s Bounded, 251M ticks/s Catoni, 14.8x faster than Rust river.ADWIN, heavy-tail Cauchy valid), Pre-Control E-Reset Conformal PID (674M ch-tk/s SoA) + zero-copy coma_merge_batch, Sleeping Fixed-Share BOA with peer-relative rogue quarantine (505/505 86-sigma spike immune), and 86.2%–94.8% real tiktoken o200k_base token-reducing Swing PLA Markdown digests + lossless zoom() via Python, 118 KB CLI, or MCP.
---

# ChronoSieve Agent Skill (v0.2.1)

Use `chronosieve` whenever you process streaming time series, telemetry, sensor feeds, financial ticks, or multi-agent forecasts and need:
1. **Distribution-free changepoint detection** with certified false-alarm control ($\text{ARL} \ge \gamma$), even on heavy-tailed Student-$t_3$/Cauchy/discrete/correlated streams (verified on the 1898 Nile Aswan dam changepoint with 0 pre-1899 false alarms; **14.8x faster** than Rust `river.ADWIN` and **2.6–3.0x faster** shift detection on heavy tails).
2. **Non-stationary conformal prediction intervals** that recover immediately after volatility shocks (+20 pp vs Rolling-D PID, +46 pp vs pure P+I PID; 90.6% OOS on 277 years of real Sunspots) without sacrificing asymptotic $1-\alpha$ coverage, plus 64-channel SoA SIMD (`673.9 M ch-tk/s`) and zero-copy `coma_merge_batch` (`7.25 M steps/s`).
3. **Asynchronous multi-agent forecast fusion** (fast System-1 + slow System-2 LLMs) using `SleepingBOA` with **peer-relative excess residual rogue attribution** (`peer_relative_rogue=True` default) that survives 86-sigma common exogenous spikes with zero false quarantines even at default `rogue_threshold=200.0`, while isolating hallucinating agents in 1–10 ticks.
4. **Massive LLM token reduction** (**86.2% real OpenAI `o200k_base`** full-series on Mauna Loa CO2 without wake gating; **91.0%–94.8% real `o200k_base`** with wake gating on 10k telemetry) by emitting compact Markdown digests and drilling down losslessly via `zoom(channel, t0, t1)`.

[Run a latency-spike replay](https://github.com/pgg898200-ship-it/chronosieve#replay-one-latency-spike) · [Python method signatures](python/chronosieve/_core.pyi)

## Install

Build the Python extension with Git, CPython 3.10+ development headers, and a C11 compiler (zero Python runtime dependencies):

```bash
CC=cc python3 -m pip install "git+https://github.com/pgg898200-ship-it/chronosieve.git"
# or from a local checkout:
python3 -m pip install --no-build-isolation -e .
make cli   # optional 118 KB standalone C11 CLI
```

MCP server command (zero dependencies; see `server.json`):
```bash
chronosieve-mcp
# or: python3 -m chronosieve.mcp_server
```

## Decision Rules

1. **Always use zero-copy batch APIs** (`ingest_batch`, `fuse_batch`, `digest_batch`, `coma_merge_batch`) with `array.array('d')`, `memoryview`, or `numpy.ndarray` on streams $> 10^3$ ticks (`2.03 ns/tick` Bounded, `3.98–4.06 ns/tick` Catoni, `4.63 ns/tick` Symmetry, `2.47 ns/tick` Swing PLA vs `332 ns/tick` CPython scalar `step()`).
2. **Pick the right E-detector family**:
   - Bounded in $[0, 1]$ with known $\mu_0$ $\to$ `EDetector(family="bounded", mu0=...)` (**492M ticks/s**).
   - Known $(\mu_0, \sigma)$ with finite variance $\to$ `EDetector(family="catoni", mu0=..., sigma=...)` (**251M ticks/s**).
   - Symmetric around $\mu_0$, unknown/infinite variance (e.g., Cauchy) $\to$ `EDetector(family="symmetry", mu0=...)` (**220M ticks/s**).
   - Unknown $(\mu, \sigma)$, discrete ties, multimodal, or heavy tails $\to$ `EDetector(family="vovk_rank")` or `family="auto"` (**15.8M ticks/s**, 2,530x faster than `river.KSWIN`).
   - Correlated multi-agent streams ($K$ agents) $\to$ `EFusion(mode="step")` (Catoni) or `EFusion(mode="process")` (Vovk Rank).
3. **Never feed closed-loop Conformal PID miscoverage into an E-detector**: `ConformalPID(enable_ereset=True)` enforces this automatically by running Vovk Rank strictly on exogenous $s_t = |y_t - \hat{y}_t|$, resetting only bounded $\hat{q}^D \in [0, B]$, and enforcing $W_{\text{cal}} \ge 32$.
4. **For multi-agent intervals**, use `coma_merge(lowers, uppers, weights=boa.weights(), randomized=True)` for single ticks or `coma_merge_batch(lowers_2d, uppers_2d, weights_2d=..., out_hull_lo=..., out_hull_hi=...)` for zero-copy batch merging (preserves Gasparin–Ramdas $\ge 1 - 2\alpha$ coverage under arbitrary dependence).
5. **For System-2 LLM triage**, ingest streams through `ChronoSieve` (default `peer_relative_rogue=True`) or the MCP server (`chronosieve_ingest`). Use `anomaly_burst=1` to wake the LLM immediately on isolated 1-tick spikes (100% wake-packet spike recall), or `anomaly_burst=2` to wake only on certified shifts and $\ge 2$-tick bursts (1-tick spikes remain pinned verbatim in `StreamDigest` for `chronosieve_zoom(channel, t0, t1)`).
