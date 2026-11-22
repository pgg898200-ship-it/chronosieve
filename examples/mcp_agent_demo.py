#!/usr/bin/env python3
"""Zero-dependency example: End-to-end MCP stdio client talking to `chronosieve-mcp`.

Demonstrates spawning `python -m chronosieve.mcp_server` over NDJSON JSON-RPC 2.0,
creating a streaming channel, ingesting telemetry with a regime shift + spike,
inspecting the compact Markdown digest, performing lossless `chronosieve_zoom`,
and merging multi-agent intervals via `chronosieve_coma`.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from typing import Any


def _rpc(proc: subprocess.Popen[bytes], payload: dict[str, Any]) -> dict[str, Any]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout unexpectedly")
    return json.loads(line.decode("utf-8"))


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "chronosieve.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # 1. Initialize handshake
        init = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        info = init["result"]["serverInfo"]
        print(f"Connected to MCP server: {info['name']} v{info['version']}")

        # 2. Discover tools
        tools = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in tools["result"]["tools"]]
        print(f"Available MCP tools ({len(names)}): {', '.join(names)}")

        # 3. Create channel
        _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_create_channel",
                    "arguments": {
                        "channel": "latency_p99_ms",
                        "alpha": 0.01,
                        "epsilon": 0.5,
                        "edet_threshold": 500.0,
                    },
                },
            },
        )

        # 4. Ingest 160 ticks with regime shift at t=90 and sharp spike at t=130
        rng = random.Random(2026)
        ts = [float(t) for t in range(160)]
        ys = []
        for t in range(160):
            base = 20.0 if t < 90 else 35.0
            y = base + rng.gauss(0.0, 0.3)
            if t == 130:
                y = 88.0
            ys.append(y)

        ing = _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_ingest",
                    "arguments": {"channel": "latency_p99_ms", "ts": ts, "ys": ys, "force_markdown": True},
                },
            },
        )
        ing_data = ing["result"]["structuredContent"]
        print(
            f"Ingested {ing_data['ingested']} ticks -> wakes={ing_data['wakes']}, "
            f"shifts={ing_data['shifts']}, miscoverages={ing_data['miscoverages']}"
        )
        print("\nCompact MCP Markdown Digest:\n" + ing_data["digest_markdown"])

        # 5. Lossless drill-down around t=130 spike
        zm = _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_zoom",
                    "arguments": {"channel": "latency_p99_ms", "t0": 129.0, "t1": 131.0},
                },
            },
        )
        recs = zm["result"]["structuredContent"]["records"]
        print("Lossless zoom(129..131):", [(r["t"], round(r["y"], 2)) for r in recs])
        assert any(r["t"] == 130.0 and abs(r["y"] - 88.0) < 1e-9 for r in recs)

        # 6. Multi-agent COMA majority-vote consensus
        cm = _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_coma",
                    "arguments": {
                        "lowers": [33.5, 34.0, 33.8, 12.0],
                        "uppers": [36.5, 36.2, 36.0, 15.0],
                        "weights": [0.3, 0.3, 0.3, 0.1],
                        "tau": 0.5,
                    },
                },
            },
        )
        cm_data = cm["result"]["structuredContent"]
        print(
            f"COMA 4-agent consensus (rogue rejected): hull={cm_data['hull']}, "
            f"agree={cm_data['max_agreement']:.2f}"
        )
        assert cm_data["hull"] == [33.8, 36.2]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
