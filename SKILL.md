---
name: chronosieve
description: Monitor numeric telemetry, detect changes, combine intermittent forecasts and emit a Markdown digest when a wake condition fires. Retrieve retained samples by channel and timestamp.
---

# ChronoSieve

Use this library when a caller receives timestamped observations and needs a local decision before notifying an operator or calling a model. Read original samples from the channel's ring buffer to inspect a flag.

[Run a latency-spike replay](https://github.com/pgg898200-ship-it/chronosieve#replay-one-latency-spike) · [Python method signatures](python/chronosieve/_core.pyi)

## Install

Build the Python extension with Git, CPython 3.10+ development headers and a C11 compiler. On Linux or macOS use the system C compiler.

```bash
CC=cc python3 -m pip install "git+https://github.com/pgg898200-ship-it/chronosieve.git@743092a8c0da995a2e5007d7fea56ae16e669fe2"
```

For local stdio MCP, run `python3 -m chronosieve.mcp_server` in that environment. For the standalone C11 CLI, run `make cli` from a source checkout.

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
