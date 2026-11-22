# ChronoSieve v0.2.0 — Launch Copy Drafts

> All numbers below are verified on Apple M4 (macOS 15.3.2 arm64, Apple Clang 16.0.0, Python 3.14.6, NEON SIMD, median of 5 runs) via `python3 benchmarks/run_all.py` and `python3 benchmarks/bench_external.py`, stored in `benchmarks/results_m4.json` and `benchmarks/results_external_m4.json`. Review and publish under your own account.

---

## 1. Hacker News — Show HN

**Title:**
`Show HN: ChronoSieve v0.2 – 118 KB C11/SIMD E-Detectors (15x vs Rust river), Conformal PID & MCP Digest`

**Body:**
Hi HN,

I built **ChronoSieve v0.2.0**, a zero-dependency C11/SIMD (ARM64 NEON, x86_64 AVX2, portable scalar) streaming inference and multi-agent digest engine with a zero-copy `Py_buffer` Python C-extension, a 118 KB standalone CLI, a zero-dependency stdio MCP server, and a Claude Code / Cursor plugin.

The problem I wanted to solve: when running multi-agent or LLM-assisted monitoring over live time-series/telemetry, you either burn 120,000+ real `o200k_base` tokens feeding raw JSON/CSV to an LLM, or you stitch together Gaussian drift detectors that collapse on heavy tails, conformal trackers that lag 150 ticks after volatility shocks, and ad-hoc ensembles that falsely quarantine healthy models when an exogenous sensor spike hits.

ChronoSieve fuses four 2023–2026 statistical primitives into one zero-dependency C11/SIMD core:

1. **Zero-`exp` Register-Resident SIMD E-Detectors & Cross-Agent E-Fusion**: Dedicated NEON & AVX2 batch kernels for Bounded Betting (**491.6M ticks/s**, `2.03 ns`), Algebraic Catoni (Wang & Ramdas 2023; **251.4M ticks/s** `array.array` / **246.5M ticks/s** NumPy, `3.98–4.06 ns`), Algebraic Symmetry (**220.0M ticks/s**, `4.54 ns`, valid even on infinite-variance Cauchy), and Vovk Growing-Prefix Conformal Rank Betting (cache-line-aligned SIMD arena B-tree at **15.8M ticks/s**, `63.2 ns`). Cross-agent E-Fusion keeps $\text{ARL} \ge \gamma$ even across $K=8$ strongly correlated streams ($\rho=0.85$), and zero-prior Vovk Rank detects the historic **1898 Nile Aswan dam changepoint** in **1905** (+7 yrs, 0 pre-1899 false alarms).
2. **Pre-Control E-Reset Conformal PID + 674M ch-tk/s SoA + Zero-Copy `coma_merge_batch`**: Exact Angelopoulos et al. (NeurIPS 2023) Conformal PID with a subtle but critical invariant: the E-detector runs strictly on the exogenous pre-control residual $s_t = |y_t - \hat{y}_t|$ (never closed-loop miscoverage $I_t$), and resets *only* the bounded D-term scorecaster $\hat{q}_{t+1}^D \in [0, B]$ with mandatory $W_{\text{cal}} \ge 32$ warmup. Post-5x-shock 100-tick coverage jumps from **39.0%** (pure P+I PID) / **65.0%** (Rolling-D PID) to **85.0%** (and tracks 277 out-of-sample years of real **Sunspots** at **90.6%** vs 88.8% / 86.6%). Includes a 4x-unrolled 64-channel SoA SIMD kernel (**673.9M ch-tk/s**, `1.48 ns`) and zero-`malloc` $O(K \log K)$ Online COMA batch consensus (**7.44M steps/s** CLI / **7.25M steps/s** Python for $K=8$, Gasparin & Ramdas UAI 2024).
3. **SIMD Sleeping Fixed-Share BOA + Peer-Relative Rogue Quarantine**: Zero-regret sleeping-expert imputation (mixes fast System-1 models with slow asynchronous LLMs), branchless Batcher sorting-network peer-relative excess residual attribution (`excess_k = max(0, |y - p_k| - median_j |y - p_j|)` with 50%-breakdown peer-median calibration floor and `< 3`-agent fallback), and SIMD Estrin `exp` (`<5e-16` err) at **8.79M steps/s** (`113.8 ns/step` Py batch / `115.3 ns/step` CLI for $K=8$). Immune to 86-sigma common exogenous spikes even at default `rogue_threshold=200.0` (**505/505 seeds** exact rogue isolation).
4. **Anomaly-Preserving Swing PLA + Wake-Up Gate + Lossless `zoom()`**: Exact ULP-guarded $L_\infty \le \epsilon$ trajectory compression at **470.7M ticks/s** (`2.12 ns`) Native CLI / **404.4M ticks/s** (`2.47 ns`) Python batch, `0.0`-error verbatim anomaly pinning, and an $O(\log N)$ ring buffer for lossless `zoom(channel, t0, t1)` drill-down via Python, CLI, or MCP.

Verified head-to-head on Apple M4 (median of 5 runs, CLI vs Python `Py_buffer` vs scalar `step()` kept strictly separate):
- **vs. Rust-backed `river 0.27.0` (throughput)**: Catoni batch (`4.07 ns`) is **14.8x faster** than `river.ADWIN` (`60.3 ns`) and **54.6x faster** than `river.PageHinkley` (`222.1 ns`); nonparametric Vovk Rank B-tree (`62.5 ns`) is **2,530x faster** than `river.KSWIN` (`158.1 us`). (For fairness: per-tick CPython `det.step(x)` is `332 ns` due to 8-key dict allocation; batch `ingest_batch()` eliminates that overhead.)
- **vs. Rust-backed `river 0.27.0` (heavy-tail validity, 60 trials, $H=1000$)**: On Student-$t_3$ $+2.0$ shifts, ChronoSieve detects **2.6–3.0x faster** than `river.ADWIN` (`12.4–14.2` vs `37.3` ticks mean ADD). On Cauchy(0,1), `river.PageHinkley` collapses (**100% FA, ARL 50.1**) while Symmetry & Vovk Rank strictly honor $\text{ARL} \ge 1000$.
- **Real OpenAI `tiktoken 0.14.0` (`o200k_base`)**:
  - **Mauna Loa CO2 ($N=2,225, \epsilon=1.0$ ppm, full history, NO wake gating)**: `23,477` JSON -> **`3,242`** ChronoSieve Swing PLA Markdown (**-86.2%** vs JSON, **-77.8%** vs CSV, **-62.3%** vs TSLN).
  - **10,000-tick telemetry**: `121,906` JSON -> **`6,310`** burst-2 gated (**-94.8%** vs JSON) / **`10,920`** high-recall with 100% spike wake recall (**-91.0%** vs JSON).
- **Correctness**: ARM64 NEON (`1.141e-15`) & x86_64 AVX2 via Rosetta 2 (`1.887e-15`) vs scalar parity across 90 outputs (`<1e-12`), zero ASan/UBSan issues, zero `-Wall -Wextra -Wpedantic -Werror` warnings, and 0 runtime deps.

Repo + SVG charts + `llms.txt` + `SKILL.md` + `server.json`: https://github.com/pgg898200-ship-it/chronosieve
Happy to dig into the martingale math, SIMD sorting networks, or C11/`Py_buffer` internals!

---

## 2. X / Twitter Launch Thread

**Tweet 1 (Hook):**
Stop burning 120,000 real `o200k_base` tokens feeding raw JSON telemetry to LLMs—or trusting drift detectors that collapse on heavy tails.

**ChronoSieve v0.2.0**: a 118 KB zero-dep C11/SIMD engine for streaming e-detectors, conformal PID, multi-agent fusion & MCP digests.

**492M ticks/s** Bounded, **251M/s** Catoni (**14.8x faster than Rust `river.ADWIN`**), **674M ch-tk/s** Conformal SoA on Apple M4. MIT licensed. 🧵👇

**Tweet 2 (Head-to-Head vs Rust `river 0.27.0`):**
Head-to-head on Apple M4 (median of 5):
- ChronoSieve Bounded E-SR: **2.03 ns/tick** (492M/s)
- ChronoSieve Catoni E-SR: **3.98–4.06 ns/tick** (246–251M/s — **14.8x vs Rust `river.ADWIN`**, **54.6x vs `PageHinkley`**)
- ChronoSieve Vovk Rank B-Tree: **63.2 ns/tick** (**2,530x vs `river.KSWIN`**)
- On Cauchy(0,1): `PageHinkley` collapses (**100% FA, ARL 50**); Symmetry & Vovk Rank honor exact ARL >= 1000 & detect Student-t3 shifts **2.8–3.0x faster** than ADWIN!

**Tweet 3 (Peer-Relative Rogue Quarantine + Shock Recovery):**
Two subtle failure modes solved cleanly in C11:
1. **Exogenous 86-sigma sensor spikes**: Peer-relative excess residual attribution (`max(0, |y - p_k| - median_j |y - p_j|)`) + 50%-breakdown peer-median floor gives **505/505 seeds** zero false quarantines at default `rogue_threshold=200`.
2. **5x volatility shocks**: Pre-Control E-Reset Conformal PID jumps post-shock 100-tick coverage from **39.0%** (pure PID) / **65.0%** (Rolling-D) to **85.0%** (and **90.6%** on 277 yrs of Sunspots).

**Tweet 4 (Real OpenAI `tiktoken o200k_base` Compression):**
Measured with real OpenAI `tiktoken 0.14.0` (`o200k_base`):
- **Mauna Loa CO2** ($N=2,225$, full 43-yr series, $L_\infty \le 1.0$ ppm, NO wake gating): `23,477` JSON -> **`3,242`** Swing PLA Markdown (**-86.2%** vs JSON, **-62.3%** vs TSLN)
- **10k Telemetry**: `121,906` JSON -> **`6,310`** burst-2 (**-94.8%**) / **`10,920`** high-recall (**-91.0%**, 100% spike recall)
- Plus lossless $O(\log N)$ ring-buffer `zoom(ch, t0, t1)`!

**Tweet 5 (Agent-Native Distribution + CTA):**
Ships with zero runtime dependencies:
- Prebuilt wheels (macOS arm64, Linux manylinux, Windows amd64) + 118 KB standalone CLI
- 1-command Claude Code / Cursor plugin (`.claude-plugin/marketplace.json`)
- Official MCP Registry `server.json` + zero-dep `chronosieve-mcp`
- `llms.txt`, `llms-full.txt`, `SKILL.md`, and 3 zero-dep `examples/`

Code & charts: https://github.com/pgg898200-ship-it/chronosieve

---

## 3. Reddit (`r/MachineLearning` [P] & `r/LocalLLaMA`)

**Title (`r/MachineLearning`):**
`[P] ChronoSieve v0.2: Zero-dep C11/SIMD E-Detectors (246M–492M ticks/s, 14.8x vs Rust river.ADWIN), Pre-Control E-Reset Conformal PID (674M ch-tk/s), Peer-Relative BOA & Real tiktoken MCP Digests`

**Body:**
We just released **ChronoSieve v0.2.0**, an MIT-licensed zero-dependency C11/SIMD (ARM64 NEON, x86_64 AVX2, portable scalar) streaming inference and multi-agent digest engine with zero-copy Python `Py_buffer` bindings, a 118 KB standalone CLI, and a zero-dependency stdio MCP server:

- **Zero-`exp` Register-Resident SIMD E-Detectors & Cross-Agent E-Fusion**: Bounded Betting (`2.03 ns`, **492M/s**), Algebraic Catoni (`3.98–4.06 ns`, **246–251M/s**), Algebraic Symmetry (`4.54–4.63 ns`, **216–220M/s**), and SIMD arena B-tree Vovk Rank (`63.2 ns`, **15.8M/s**). Head-to-head against real Rust-backed `river 0.27.0` on Apple M4: **14.8x faster** than `ADWIN` (`60.3 ns`), **54.6x faster** than `PageHinkley` (`222.1 ns`), **2,530x faster** than `KSWIN` (`158.1 us`), with **2.6–3.0x faster** shift detection on Student-$t_3$ (`12.4–14.2` vs `37.3` ticks) and exact Cauchy(0,1) $\text{ARL} \ge \gamma$ where `PageHinkley` collapses (`100%` FA, ARL `50.1`).
- **Pre-Control E-Reset Conformal PID + SoA SIMD + Zero-Copy `coma_merge_batch`**: Resets only the bounded D-term scorecaster $\hat{q}^D \in [0, B]$ on exogenous pre-control residual shifts ($W_{\text{cal}} \ge 32$), raising post-5x-shock 100-tick coverage from `39.0%` (pure P+I PID) / `65.0%` (Rolling-D) to `85.0%` (and `90.6%` OOS on 277 years of real Sunspots), plus **673.9M ch-tk/s** 64-channel SoA SIMD and zero-`malloc` **7.44M steps/s** CLI / **7.25M steps/s** Python batch COMA consensus (Gasparin & Ramdas UAI 2024).
- **Sleeping Fixed-Share BOA + Peer-Relative Rogue Quarantine**: Branchless Batcher sorting-network median excess attribution with 50%-breakdown calibration floor (`113.8 ns/step` for $K=8$); survives 86-sigma exogenous common shocks across **505/505 seeds** with 0% false quarantine at default `rogue_threshold=200.0`.
- **Anomaly-Preserving Swing PLA + Real `tiktoken o200k_base` Reduction**: **470.7M ticks/s** Native CLI (**404.4M/s** Python batch), exact $L_\infty \le \epsilon$ + verbatim `0.0`-error anomalies; **-86.2%** real `o200k_base` tokens on Mauna Loa CO2 ($N=2,225$, full series, NO wake gating) and **-94.8%** (`burst=2`) / **-91.0%** (`burst=1`, 100% spike wake recall) on 10k telemetry, plus lossless $O(\log N)$ ring-buffer `zoom()`.

Reproducible benchmarks, SVG charts, ASan/UBSan + Rosetta 2 AVX2 `<1e-12` parity tests, and 0-dep examples: https://github.com/pgg898200-ship-it/chronosieve
