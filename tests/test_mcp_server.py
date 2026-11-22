"""End-to-end JSON-RPC 2.0 stdio tests for chronosieve.mcp_server."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def _send_ndjson(proc: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    assert proc.stdin is not None
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    proc.stdin.write(raw)
    proc.stdin.flush()


def _recv_ndjson(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    assert line, "MCP server closed stdout unexpectedly"
    return json.loads(line.decode("utf-8"))


def _send_content_length(proc: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    assert proc.stdin is not None
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    proc.stdin.flush()


def _recv_content_length(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    assert proc.stdout is not None
    hdr = proc.stdout.readline()
    assert hdr.lower().startswith(b"content-length:"), f"Unexpected header: {hdr!r}"
    n = int(hdr.split(b":", 1)[1].strip())
    blank = proc.stdout.readline()
    assert blank in (b"\r\n", b"\n")
    body = proc.stdout.read(n)
    return json.loads(body.decode("utf-8"))


def test_mcp_stdio_ndjson_full_workflow() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "chronosieve.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # 1. initialize
        _send_ndjson(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        r1 = _recv_ndjson(proc)
        assert r1["id"] == 1
        assert r1["result"]["serverInfo"]["name"] == "chronosieve"

        # 2. notification (produces no reply) + ping
        _send_ndjson(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send_ndjson(proc, {"jsonrpc": "2.0", "id": 2, "method": "ping"})
        r2 = _recv_ndjson(proc)
        assert r2["id"] == 2 and r2["result"] == {}

        # 3. tools/list
        _send_ndjson(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        r3 = _recv_ndjson(proc)
        tool_names = {t["name"] for t in r3["result"]["tools"]}
        assert {
            "chronosieve_create_channel",
            "chronosieve_ingest",
            "chronosieve_zoom",
            "chronosieve_digest",
            "chronosieve_coma",
            "chronosieve_status",
        }.issubset(tool_names)

        # 4. create_channel
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_create_channel",
                    "arguments": {
                        "channel": "latency_p99",
                        "n_agents": 1,
                        "epsilon": 0.05,
                        "edet_threshold": 80.0,
                        "anomaly_burst": 1,
                    },
                },
            },
        )
        r4 = _recv_ndjson(proc)
        assert r4["result"]["isError"] is False
        assert r4["result"]["structuredContent"]["created"] == "latency_p99"

        # 5. ingest stationary stream -> should stay quiet (0 or minimal wakes)
        ts_stat = [float(i) for i in range(60)]
        ys_stat = [10.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(60)]
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_ingest",
                    "arguments": {"channel": "latency_p99", "ts": ts_stat, "ys": ys_stat},
                },
            },
        )
        r5 = _recv_ndjson(proc)
        sc5 = r5["result"]["structuredContent"]
        assert sc5["ingested"] == 60
        assert sc5["shifts"] == 0

        # 6. ingest sharp regime shift + spike -> wakes System-2 with Markdown digest!
        ts_shift = [float(i) for i in range(60, 100)]
        ys_shift = [25.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(40)]
        ys_shift[10] = 80.0  # Huge spike at t=70
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_ingest",
                    "arguments": {"channel": "latency_p99", "ts": ts_shift, "ys": ys_shift},
                },
            },
        )
        r6 = _recv_ndjson(proc)
        sc6 = r6["result"]["structuredContent"]
        assert sc6["should_wake"] is True
        assert sc6["shifts"] >= 1
        assert "## ChronoSieve Digest: `latency_p99`" in sc6["digest_markdown"]

        # 7. lossless zoom around spike t=70
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_zoom",
                    "arguments": {"channel": "latency_p99", "t0": 69.0, "t1": 71.0},
                },
            },
        )
        r7 = _recv_ndjson(proc)
        sc7 = r7["result"]["structuredContent"]
        assert sc7["truncated"] is False
        assert len(sc7["records"]) == 3
        assert sc7["records"][1]["t"] == 70.0
        assert sc7["records"][1]["y"] == 80.0

        # 8. chronosieve_coma tool
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_coma",
                    "arguments": {
                        "lowers": [0.0, 1.0, 2.0],
                        "uppers": [4.0, 5.0, 6.0],
                        "tau": 0.5,
                    },
                },
            },
        )
        r8 = _recv_ndjson(proc)
        sc8 = r8["result"]["structuredContent"]
        assert sc8["intervals"] == [[1.0, 5.0]]
        assert sc8["hull"] == [1.0, 5.0]

        # 9. unknown method & unknown tool error handling
        _send_ndjson(proc, {"jsonrpc": "2.0", "id": 9, "method": "nonexistent/method"})
        r9 = _recv_ndjson(proc)
        assert r9["error"]["code"] == -32601

        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "chronosieve_zoom", "arguments": {"channel": "missing", "t0": 0, "t1": 1}},
            },
        )
        r10 = _recv_ndjson(proc)
        assert r10["result"]["isError"] is True
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0


def test_mcp_stdio_content_length_framing() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "chronosieve.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _send_content_length(proc, {"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}})
        resp = _recv_content_length(proc)
        assert resp["id"] == 42
        assert resp["result"]["serverInfo"]["name"] == "chronosieve"

        _send_content_length(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 43,
                "method": "tools/call",
                "params": {"name": "chronosieve_status", "arguments": {}},
            },
        )
        resp2 = _recv_content_length(proc)
        assert resp2["id"] == 43
        assert resp2["result"]["structuredContent"]["version"] == "0.1.0"
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0


def test_mcp_stdio_adversarial_framing_and_atomic_ingest() -> None:
    """Server must never crash on malformed UTF-8, bad Content-Length, non-dict JSON, or bad batches."""
    cases = [
        (b"123\n", -32600),
        (b"Content-Length: abc\r\n\r\n", -32700),
        (b"Content-Length: -5\r\n\r\n", -32700),
        (b"Content-Length: 9999999999\r\n\r\n", -32700),
        (b"\xff\xfe\n", -32700),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":[1,2]}\n', -32602),
    ]
    for payload, expected_code in cases:
        p = subprocess.run(
            [sys.executable, "-m", "chronosieve.mcp_server"],
            input=payload,
            capture_output=True,
            timeout=5,
        )
        assert p.returncode == 0, (payload, p.stderr)
        out_s = p.stdout.decode("utf-8")
        body = out_s.split("\r\n\r\n", 1)[1] if out_s.startswith("Content-Length:") else out_s.strip()
        resp = json.loads(body)
        assert resp["error"]["code"] == expected_code

    # Atomic ingest: mid-batch non-monotonic timestamp must NOT create or mutate the channel
    proc = subprocess.Popen(
        [sys.executable, "-m", "chronosieve.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "chronosieve_ingest",
                    "arguments": {"channel": "bad_ch", "ts": [0.0, 1.0, 0.5], "ys": [1.0, 2.0, 3.0]},
                },
            },
        )
        r1 = _recv_ndjson(proc)
        assert r1["result"]["isError"] is True

        _send_ndjson(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "chronosieve_status", "arguments": {}},
            },
        )
        r2 = _recv_ndjson(proc)
        assert "bad_ch" not in r2["result"]["structuredContent"]["channels"]
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0
