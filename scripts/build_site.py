#!/usr/bin/env python3
"""Zero-dependency static docs & AI-discovery site builder for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://pgg898200-ship-it.github.io/chronosieve"
REPO_URL = "https://github.com/pgg898200-ship-it/chronosieve"


def _build_json_ld(base_url: str, version: str) -> str:
    payload: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "ChronoSieve",
        "alternateName": "chronosieve",
        "version": version,
        "description": (
            "Zero-dependency C11/SIMD streaming inference engine combining zero-exp E-Detectors, "
            "Pre-Control E-Reset Conformal PID, Sleeping Fixed-Share BOA multi-agent fusion with "
            "peer-relative rogue quarantine, and lossless-zoom Swing PLA LLM digests."
        ),
        "url": f"{base_url}/",
        "codeRepository": REPO_URL,
        "license": "https://opensource.org/licenses/MIT",
        "programmingLanguage": ["C", "Python"],
        "operatingSystem": ["macOS", "Linux", "Windows"],
        "runtimePlatform": ["ARM64 NEON", "x86_64 AVX2", "Portable C11 Scalar"],
        "keywords": [
            "streaming change detection",
            "e-detector",
            "conformal prediction",
            "conformal PID",
            "sleeping BOA",
            "time-series compression",
            "MCP server",
            "SIMD",
        ],
    }
    return json.dumps(payload, indent=2)


def _render_index_html(m4: dict[str, Any], ext: dict[str, Any], base_url: str) -> str:
    version = str(m4["throughput"].get("version", "0.2.0"))
    tp = m4["throughput"]["kernels"]
    ext_tp = ext["throughput"]
    tok = m4["tokens"]
    co2 = tok["real_world_mauna_loa_co2"]["chronosieve_swing_pla_digest"]
    cp = m4["math_proofs"]["conformal_pid"]
    cli_kb = m4["throughput"].get("native_cli_bytes", 120352) / 1024.0

    bounded_m = tp["bounded_esr_batch_numpy"]["m_ticks_per_sec"]
    catoni_m = tp["catoni_esr_batch_numpy"]["m_ticks_per_sec"]
    soa_m = tp["conformal_pid_soa_64ch_batch"]["m_ch_ticks_per_sec"]
    soa_ns = tp["conformal_pid_soa_64ch_batch"]["ns_per_ch_tick"]
    pla_m = tp["swing_pla_ringbuf_batch"]["m_ticks_per_sec"]
    cli_stdout = str(m4["throughput"].get("native_cli_bench_stdout", ""))
    cli_pla_match = re.search(r"Swing PLA \+ RingBuf:.*?\(\s*([0-9.]+)\s*M ticks/s\)", cli_stdout)
    pla_cli_m = float(cli_pla_match.group(1)) if cli_pla_match else pla_m
    sp_adwin = ext_tp["river_adwin"]["speedup_chronosieve_catoni_batch"]
    sp_ph = ext_tp["river_page_hinkley"]["speedup_chronosieve_catoni_batch"]
    sp_ks = ext_tp["river_kswin"]["speedup_chronosieve_vovk_rank_batch"]
    o200k_10k = tok["chronosieve_gated_burst2"]["reduction_vs_json_o200k_base_pct"]
    o200k_co2 = co2["reduction_vs_json_o200k_base_pct"]
    shock_er = cp["ereset_shock_100t_coverage"] * 100.0
    shock_pu = cp["pure_pid_shock_100t_coverage"] * 100.0

    json_ld = _build_json_ld(base_url, version)
    title = f"ChronoSieve v{version} — Zero-Dependency C11/SIMD Streaming Inference & MCP Digest Engine"
    desc = (
        f"{cli_kb:.0f} KB zero-dep C11/SIMD engine: {bounded_m:.0f}M/s Bounded E-SR, "
        f"{catoni_m:.0f}M/s Catoni ({sp_adwin:.1f}x faster than Rust river.ADWIN), "
        f"{soa_m:.0f}M ch-tk/s Conformal PID SoA, "
        f"and -{o200k_10k:.1f}% real OpenAI o200k_base token reduction."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <meta name="description" content="{escape(desc)}"/>
  <link rel="canonical" href="{escape(base_url)}/"/>
  <meta property="og:type" content="website"/>
  <meta property="og:title" content="{escape(title)}"/>
  <meta property="og:description" content="{escape(desc)}"/>
  <meta property="og:url" content="{escape(base_url)}/"/>
  <meta name="twitter:card" content="summary_large_image"/>
  <script type="application/ld+json">
{json_ld}
  </script>
  <style>
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #f0f6fc;
      --muted: #8b949e; --cyan: #38bdf8; --emerald: #34d399; --amber: #fbbf24;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.6;
    }}
    .wrap {{ max-width: 1020px; margin: 0 auto; padding: 36px 22px 64px; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .badge {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px; padding: 4px 10px; border-radius: 999px;
      border: 1px solid var(--border); background: var(--surface); color: var(--cyan);
    }}
    h1 {{ font-size: clamp(28px, 4.5vw, 44px); margin: 0 0 10px; letter-spacing: -0.02em; }}
    .lead {{ font-size: 17px; color: var(--muted); max-width: 780px; margin: 0 0 20px; }}
    .cta-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .btn {{
      display: inline-block; padding: 10px 18px; border-radius: 8px; text-decoration: none;
      font-weight: 600; font-size: 14px; border: 1px solid var(--border);
      color: var(--text); background: var(--surface);
    }}
    .btn-primary {{ background: var(--cyan); color: #071018; border-color: var(--cyan); }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px; margin: 28px 0 36px;
    }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px 18px;
    }}
    .metric {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 26px; font-weight: 700; color: var(--cyan); line-height: 1.2;
    }}
    .metric.em {{ color: var(--emerald); }}
    .metric-lbl {{ font-size: 13px; color: var(--muted); margin-top: 6px; }}
    h2 {{
      font-size: 23px; margin: 40px 0 14px;
      border-bottom: 1px solid var(--border); padding-bottom: 8px;
    }}
    .chart {{ margin: 18px 0 28px; }}
    .chart img {{ width: 100%; height: auto; display: block; border-radius: 12px; }}
    pre {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 16px; overflow-x: auto; font-size: 13px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .links {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; }}
    .links a {{
      color: var(--cyan); text-decoration: none;
      font-family: ui-monospace, monospace; font-size: 13.5px;
    }}
    .links a:hover {{ text-decoration: underline; }}
    footer {{
      margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border);
      color: var(--muted); font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="badge-row">
        <span class="badge">v{escape(version)}</span>
        <span class="badge">0 Runtime Dependencies</span>
        <span class="badge">{cli_kb:.0f} KB Native CLI</span>
        <span class="badge">ARM64 NEON · x86_64 AVX2 · Scalar</span>
        <span class="badge">MIT License</span>
      </div>
      <h1>ChronoSieve</h1>
      <p class="lead">
        Zero-dependency C11/SIMD streaming inference &amp; multi-agent digest engine.
        Fuses zero-<code>exp</code> E-Detectors, Pre-Control E-Reset Conformal PID + COMA,
        Sleeping Fixed-Share BOA with peer-relative rogue quarantine, and lossless-zoom Swing PLA
        into a single Python C-extension, standalone CLI, and stdio MCP server.
      </p>
      <div class="cta-row">
        <a class="btn btn-primary" href="{REPO_URL}">View on GitHub</a>
        <a class="btn" href="{REPO_URL}/releases">Releases &amp; Binaries</a>
        <a class="btn" href="llms.txt">llms.txt</a>
        <a class="btn" href="llms-full.txt">llms-full.txt</a>
        <a class="btn" href="SKILL.md">SKILL.md</a>
      </div>
    </header>

    <section class="grid" aria-label="Key Apple M4 Benchmarks">
      <div class="card">
        <div class="metric">{soa_m:.1f}M/s</div>
        <div class="metric-lbl">Conformal PID SoA (64-ch NEON, {soa_ns:.2f} ns/ch-tk)</div>
      </div>
      <div class="card">
        <div class="metric">{bounded_m:.0f}M / {catoni_m:.0f}M/s</div>
        <div class="metric-lbl">
          Bounded / Catoni E-SR ({sp_adwin:.1f}x vs Rust river.ADWIN, {sp_ph:.0f}x vs PageHinkley)
        </div>
      </div>
      <div class="card">
        <div class="metric em">-{o200k_10k:.1f}% / -{o200k_co2:.1f}%</div>
        <div class="metric-lbl">Real OpenAI tiktoken o200k_base tokens (10k telemetry / Mauna Loa CO2)</div>
      </div>
      <div class="card">
        <div class="metric em">{shock_er:.0f}% vs {shock_pu:.0f}%</div>
        <div class="metric-lbl">
          Post-5x-shock 100-tick coverage (E-Reset PID vs Pure P+I PID; +{shock_er - shock_pu:.0f} pp)
        </div>
      </div>
      <div class="card">
        <div class="metric">{pla_m:.0f}M/s</div>
        <div class="metric-lbl">
          Swing PLA + RingBuf batch ({pla_cli_m:.1f}M/s Native CLI, exact L_inf &lt;= eps)
        </div>
      </div>
      <div class="card">
        <div class="metric em">500 / 500</div>
        <div class="metric-lbl">
          Seeds with 0% false quarantine on 86-sigma exogenous spikes (peer-relative BOA)
        </div>
      </div>
    </section>

    <h2>1. Head-to-Head Throughput vs. Rust <code>river</code></h2>
    <p class="lead">
      Median-of-5 Apple M4 arm64 measurements separating zero-copy <code>Py_buffer</code> batch kernels,
      compiled Rust <code>river 0.27.0</code>, and pure Python. Nonparametric Vovk Rank B-Tree is
      <strong>{sp_ks:,.0f}x faster</strong> than <code>river.KSWIN</code> while preserving exact
      Cauchy(0,1) ARL &gt;= gamma validity.
    </p>
    <div class="chart">
      <img src="assets/benchmark_throughput.svg"
           alt="ChronoSieve v{escape(version)} Throughput Benchmark Chart"/>
    </div>

    <h2>2. Conformal PID Shock Recovery &amp; Real-World Sunspots</h2>
    <div class="chart">
      <img src="assets/conformal_shock_recovery.svg" alt="Conformal PID Shock Recovery Chart"/>
    </div>

    <h2>3. Real OpenAI <code>tiktoken</code> (<code>o200k_base</code>) Token Reduction</h2>
    <div class="chart">
      <img src="assets/token_compression.svg" alt="Real OpenAI tiktoken o200k_base Token Compression Chart"/>
    </div>

    <h2>4. Quick Install (Python, CLI, Claude Code Plugin &amp; MCP)</h2>
    <pre><code># Python C11/SIMD extension (0 runtime dependencies)
python3 -m pip install --no-build-isolation -e .

# Standalone {cli_kb:.0f} KB stripped CLI
make cli &amp;&amp; ./chronosieve bench --n 2000000

# Claude Code / Cursor 1-command plugin
/plugin marketplace add pgg898200-ship-it/chronosieve
/plugin install chronosieve@chronosieve

# Zero-dependency stdio MCP server
chronosieve-mcp</code></pre>

    <h2>5. AI &amp; Agent Discovery Artifacts</h2>
    <div class="links">
      <a href="llms.txt">llms.txt</a>
      <a href="llms-full.txt">llms-full.txt</a>
      <a href="SKILL.md">SKILL.md</a>
      <a href="server.json">server.json (MCP Registry)</a>
      <a href="sitemap.xml">sitemap.xml</a>
      <a href="robots.txt">robots.txt</a>
    </div>

    <footer>
      ChronoSieve v{escape(version)} · MIT License ·
      Reproducible benchmarks: <code>benchmarks/results_m4.json</code> &amp;
      <code>benchmarks/results_external_m4.json</code>.
    </footer>
  </div>
</body>
</html>
"""


def build_site(out_dir: Path, base_url: str) -> None:
    base_url = base_url.rstrip("/")
    m4 = json.loads((ROOT / "benchmarks" / "results_m4.json").read_text(encoding="utf-8"))
    ext = json.loads((ROOT / "benchmarks" / "results_external_m4.json").read_text(encoding="utf-8"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets_out = out_dir / "assets"
    assets_out.mkdir(parents=True, exist_ok=True)

    # 1. Copy SVG charts
    src_assets = ROOT / "docs" / "assets"
    for svg in sorted(src_assets.glob("*.svg")):
        shutil.copy2(svg, assets_out / svg.name)

    # 2. Copy AI-discovery & MCP manifests to site root
    for fname in ("llms.txt", "llms-full.txt", "SKILL.md", "server.json"):
        shutil.copy2(ROOT / fname, out_dir / fname)

    # 3. Render index.html with Schema.org SoftwareSourceCode JSON-LD
    (out_dir / "index.html").write_text(_render_index_html(m4, ext, base_url), encoding="utf-8")

    # 4. Generate sitemap.xml & robots.txt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls = [
        f"{base_url}/",
        f"{base_url}/llms.txt",
        f"{base_url}/llms-full.txt",
        f"{base_url}/SKILL.md",
        f"{base_url}/server.json",
    ]
    sitemap_entries = "\n".join(
        f"  <url>\n    <loc>{escape(u)}</loc>\n    <lastmod>{today}</lastmod>\n  </url>" for u in urls
    )
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{sitemap_entries}\n"
        "</urlset>\n"
    )
    (out_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    robots = f"User-agent: *\nAllow: /\n\nSitemap: {base_url}/sitemap.xml\n"
    (out_dir / "robots.txt").write_text(robots, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChronoSieve static docs & AI-discovery site")
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="Output directory")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Canonical site base URL")
    args = parser.parse_args()

    build_site(args.out, args.base_url)
    files = sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file())
    print(f"Built static site in {args.out} ({len(files)} files): {', '.join(files)}")


if __name__ == "__main__":
    main()
