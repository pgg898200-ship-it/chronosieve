# ChronoSieve — Launch Copy Drafts

> All numbers below are verified on Apple M4 (macOS 15.3.2 arm64, Apple Clang 16.0.0, Python 3.14.6, NEON SIMD) via `python3 benchmarks/run_all.py` and stored in `benchmarks/results_m4.json`. Review and publish under your own account.

---

## 1. Hacker News — Show HN

**Title:**
`Show HN: ChronoSieve – 101 KB C11/SIMD E-Detectors, Conformal PID & MCP Digest`

**Body:**
Hi HN,

I built **ChronoSieve**, a zero-dependency C11/SIMD (ARM64 NEON, x86_64 AVX2, scalar) streaming inference and multi-agent digest engine with a zero-copy `Py_buffer` Python C-extension, a 101 KB native CLI, and a zero-dependency stdio MCP server.

The problem I wanted to solve: when running multi-agent or LLM-assisted monitoring over live time-series/telemetry, you either burn tens of thousands of tokens feeding raw JSON/CSV to an LLM, or you stitch together Gaussian drift detectors that break on heavy tails, conformal intervals that lag 150 ticks after volatility shocks, and ad-hoc ensembles that don't handle slow asynchronous LLM agents or hallucinations.

ChronoSieve fuses four 2023–2026 statistical primitives into one zero-dependency C11 core:

1. **Zero-`exp` 16-lane SIMD E-Detectors & Cross-Agent E-Fusion**: Algebraic Catoni (Wang & Ramdas 2023), Bounded Betting, Algebraic Symmetry (valid even on infinite-variance Cauchy), and Vovk Growing-Prefix Conformal Rank Betting via an arena-backed $O(\log n)$ B-tree with deterministic epoch resets. Cross-agent E-Fusion keeps $\text{ARL} \ge \gamma$ even across $K=8$ strongly correlated streams ($\rho=0.85$), and zero-prior Vovk Rank detects the historic **1898 Nile Aswan dam changepoint** in **1905** (+7 yrs, 0 pre-1899 false alarms).
2. **Pre-Control E-Reset Conformal PID + Online COMA**: Exact Angelopoulos et al. (NeurIPS 2023) Conformal PID with a subtle but critical invariant: the E-detector runs strictly on the exogenous pre-control residual $s_t = |y_t - \hat{y}_t|$ (never closed-loop miscoverage $I_t$), and resets *only* the bounded D-term scorecaster $\hat{q}_{t+1}^D \in [0, B]$ with mandatory $W_{\text{cal}} \ge 32$ warmup. That preserves exact asymptotic $1-\alpha$ coverage while jumping post-5x-shock 100-tick coverage from **39.0%** (pure P+I PID) / **65.0%** (Rolling-D PID) to **85.0%** (and tracking 277 strictly causal out-of-sample years of real **Sunspot** cycles at **90.6%** vs 88.8% Rolling-D / 86.6% pure PID). Also includes a 64-channel SoA NEON kernel and $O(K \log K)$ Online COMA interval consensus (Gasparin & Ramdas, UAI 2024).
3. **SIMD Sleeping Fixed-Share BOA + Rogue Agent Quarantine**: Zero-regret sleeping-expert imputation (so slow LLM agents can sleep while fast System-1 models run every tick), fast SIMD Estrin `exp` (`<5e-16` err), shift-coupled Fixed-Share mixing, and Bonferroni-safe per-agent Catoni residual detectors that pinpoint and quarantine hallucinating agents.
4. **Anomaly-Preserving Swing PLA + Wake-Up Gate + Lossless `zoom()`**: Exact $L_\infty \le \epsilon$ trajectory compression with `0.0`-error verbatim pinning of conformal anomalies, zero LLM tokens while stationary, ~195–200-token (`chars/4`) Markdown wake-up packets on shifts/anomalies, and an $O(\log N)$ circular ring buffer for lossless `zoom(channel, t0, t1)` drill-down via Python, CLI, or MCP.

Measured on an Apple M4 (reproducible via `python3 benchmarks/run_all.py`; Native CLI and Python `Py_buffer` batch reported separately):
- **Catoni E-SR (16-lane NEON)**: **8.82 ns/tick** (113.4M/s) Native CLI | **8.47 ns/tick** (118.1M/s) Python `array.array('d')` | **9.03 ns/tick** (110.7M/s) Python NumPy | **347.8 ns/tick** CPython scalar `step()` (honestly reporting CPython dispatch overhead).
- **Vovk Rank B-tree (65k cap)**: **114.4 ns/tick** (8.74M/s) Native CLI | **115.5 ns/tick** (8.66M/s) Python batch.
- **Conformal PID SoA (64ch NEON)**: **2.39 ns/ch-tick** (418.8M ch-tk/s) Native CLI | **2.48 ns/ch-tick** (402.8M ch-tk/s) Python batch.
- **Swing PLA + Ring Buffer**: **6.17 ns/tick** (162.0M/s) Native CLI | **6.69 ns/tick** (149.5M/s) Python batch.
- **Real-world Mauna Loa CO2 ($N=2,225, \epsilon=1.0$ ppm, full-series, NO wake gating)**: Native Swing PLA Digest = 107 segments (20.8x PLA, verified exact C $L_\infty = 1.000000 \le 1.00$ ppm; 1.0004 ppm reconstructed from rendered 6-sig-fig Markdown), **91.6% BPE-proxy / 88.3% `chars/4`** reduction vs Raw JSON and **71.9% BPE / 58.2% `chars/4`** beyond TSLN-style delta text.
- **10k-tick telemetry stream**: **96.3% BPE-proxy / 95.0% `chars/4`** vs Raw JSON (**86.5% BPE / 78.9% `c/4`** beyond TSLN) in gated `burst=2` mode (100% pinned for `zoom()`, 0% 1-tick spike wakes by design); **93.7% BPE / 91.2% `c/4`** vs JSON in high-recall `burst=1` mode (**100%** pinned & wake-packet spike recall).
- **Correctness**: NEON (`1.14e-15`) and x86_64 AVX2 (`7.85e-16`) vs portable scalar parity (`<1e-12`), zero ASan/UBSan issues across NEON/AVX2/Scalar, and Monte Carlo + `statsmodels` proofs in `tests/` & `benchmarks/`.

There are great specialized compiled tools in each silo (`river` in Rust, `conformal-prediction`/`risksieve` in Rust, `stcpR6` in Rcpp, `TerseTS` in Zig, `tsln-python`); ChronoSieve aims to be the tiny, zero-dependency unified System-1 front-end + MCP bridge for streaming agents.

Repo + `llms.txt` + `SKILL.md`: https://github.com/pgg898200-ship-it/chronosieve
Happy to dig into the martingale math, SIMD quirks, or C11/Py_buffer design!

---

## 2. X / Twitter Launch Thread

**Tweet 1 (Hook):**
Stop feeding 50,000 raw JSON telemetry tokens to LLMs—or trusting Gaussian drift detectors that collapse on heavy tails.

Introducing **ChronoSieve**: a 101 KB zero-dependency C11/SIMD engine for streaming change detection, conformal intervals, multi-agent fusion & MCP wake-up digests.

113M+ ticks/s native C11 (118M/s zero-copy Python) on Apple M4. MIT licensed. 🧵👇

**Tweet 2 (The 4 Primitives):**
ChronoSieve fuses four 2023–2026 statistical primitives into one zero-dep C11/NEON/AVX2 library:
1. Zero-`exp` E-Detectors (Catoni, Bounded, Symmetry, Vovk Rank B-tree) + Correlated Cross-Agent E-Fusion (ARL >= gamma on Cauchy & K=8, rho=0.85; catches the 1898 Nile dam shift w/ 0 false alarms)
2. Pre-Control E-Reset Conformal PID + 418.8M ch-tk/s SoA SIMD + Online COMA
3. Sleeping Fixed-Share BOA + automatic rogue/hallucinating agent quarantine
4. Exact L_inf <= eps Swing PLA (91.6% BPE / 88.3% chars/4 full-series reduction on Mauna Loa CO2, no wake gate) + lossless `zoom(ch, t0, t1)`

**Tweet 3 (The Subtle Math Invariant):**
One trap we caught during Monte Carlo stress-testing: never feed closed-loop Conformal PID miscoverage into a changepoint detector or boost eta.

Instead, ChronoSieve runs Vovk Rank strictly on exogenous `s_t = |y_t - y_hat_t|`, resets *only* the bounded D-term `q_D in [0, B]`, and enforces a >=32-tick warmup.

Result across three 4x–5x volatility shocks:
- Full coverage: **90.1%** (target 90.0%)
- Post-5x-shock 100-tick coverage: **85.0%** vs **65.0%** Rolling-D PID (+20 pp) vs **39.0%** pure P+I PID (+46 pp!)
- Strictly causal out-of-sample Sunspots (1732–2008): **90.6%** vs **88.8%** Rolling-D / **86.6%** pure PID.

**Tweet 4 (Honest Apple M4 Numbers):**
Real numbers on Apple M4 (`python3 benchmarks/run_all.py`, CLI vs Python `Py_buffer` kept separate):
- Catoni E-SR (16-lane NEON): **8.82 ns** CLI | **8.47 ns** Py `array.array` | **9.03 ns** Py NumPy | 348 ns CPython scalar `step()`
- Conformal PID SoA (64ch): **2.39 ns/ch-tk** (418.8M/s) CLI | **2.48 ns** (402.8M/s) Py batch
- Swing PLA + RingBuf: **6.17 ns** (162.0M/s) CLI | **6.69 ns** (149.5M/s) Py batch
- Mauna Loa CO2 ($N=2,225$, $L_\infty \le 1.0$ ppm): **91.6% BPE / 88.3% c/4** vs JSON (**71.9% BPE / 58.2% c/4** vs TSLN)
- NEON (`1.14e-15`) & AVX2 (`7.85e-16`) vs scalar parity | ASan+UBSan clean | 0 runtime deps

**Tweet 5 (Agent-Native + CTA):**
Ships with:
- Zero-copy `Py_buffer` Python C-extension (works with stdlib `array.array` or NumPy)
- 101 KB native CLI (`chronosieve`)
- Zero-dep stdio MCP server (`python3 -m chronosieve.mcp_server`)
- Pure-Markdown `llms.txt`, `llms-full.txt`, and drop-in `SKILL.md`

Code & reproducible benchmarks: https://github.com/pgg898200-ship-it/chronosieve

---

## 3. Reddit (`r/MachineLearning` [P] & `r/LocalLLaMA`)

**Title (`r/MachineLearning`):**
`[P] ChronoSieve: Zero-dep C11/SIMD E-Detectors, Pre-Control E-Reset Conformal PID + COMA, Sleeping BOA & MCP Stream Digests (113M+ ticks/s on M4)`

**Body:**
We built **ChronoSieve**, an MIT-licensed zero-dependency C11/SIMD (NEON + AVX2 + scalar) library that brings four recent online statistical inference results into one unified streaming engine (with zero-copy Python `Py_buffer` bindings, a 101 KB CLI, and a stdio MCP server):

- **Zero-`exp` E-Detectors & Cross-Agent E-Fusion** (Shin, Ramdas & Rinaldo 2023; Wang & Ramdas 2023; Vovk 2021): 16-lane SIMD E-SR and log-domain E-CUSUM across Bounded Betting, Algebraic Catoni, Algebraic Symmetry (tested on Cauchy(0,1)), and Vovk Growing-Prefix Conformal Rank Betting (arena-backed $O(\log n)$ B-tree with deterministic epoch resets for bounded memory). Verified $\text{ARL} \ge \gamma$ under Student-$t_3$, Cauchy, discrete ties, and $K=8, \rho=0.85$ correlated streams, plus zero-prior detection of the **1898 Nile Aswan dam changepoint** in **1905** (+7 yrs, 0 pre-1899 false alarms).
- **Pre-Control E-Reset Conformal PID + Online COMA** (Angelopoulos et al. NeurIPS 2023; Gasparin & Ramdas UAI 2024): Resets only the bounded D-term scorecaster $\hat{q}^D \in [0, B]$ on exogenous pre-control residual shifts ($W_{\text{cal}} \ge 32$), preserving Theorem 1 asymptotic coverage (0.901 at $\alpha=0.10$) while raising post-5x-shock 100-tick coverage from 0.390 (pure P+I PID) / 0.650 (Rolling-D PID) to 0.850 (and 0.906 vs 0.888 Rolling-D / 0.866 pure PID on strictly causal out-of-sample **Sunspots** 1732–2008). Includes a 64-channel SoA NEON batch kernel (**418.8 M ch-ticks/s** Native CLI / **402.8 M ch-ticks/s** Python `Py_buffer`) and $O(K \log K)$ COMA interval consensus.
- **Sleeping Fixed-Share BOA + Per-Agent Rogue Quarantine** (Wintenberger 2017; Herbster & Warmuth 1998): Handles asynchronous sleeping experts (fast System-1 + slow LLM System-2) with zero-regret imputation and Bonferroni-safe per-agent Catoni residual attribution.
- **Anomaly-Preserving Swing PLA + Wake-Up Gate + Lossless `zoom()`**: Exact $L_\infty \le \epsilon$ trajectory compression (`0.0` error on conformal anomalies). On real-world **Mauna Loa CO2** ($N=2,225, \epsilon=1.0$ ppm) with NO wake gating, native Swing PLA Markdown achieves **91.6% BPE-proxy / 88.3% `chars/4`** reduction vs Raw JSON (**71.9% BPE / 58.2% `c/4`** beyond TSLN, verified exact C $L_\infty = 1.000000 \le 1.00$ ppm and 1.0004 ppm Markdown text reconstruction). On 10k multi-regime telemetry, wake-up gated `burst=2` achieves **96.3% BPE / 95.0% `c/4`** vs JSON (100% pinned for `zoom()`, 0% 1-tick spike wakes) and high-recall `burst=1` achieves **93.7% BPE / 91.2% `c/4`** with **100%** pinned & wake-packet spike recall.

All benchmarks, ASan/UBSan C11 tests, NEON (`1.14e-15`) & AVX2 (`7.85e-16`) vs scalar `<1e-12` parity tests, and Monte Carlo + `statsmodels` verification scripts are reproducible in https://github.com/pgg898200-ship-it/chronosieve.
