"""Zero-dependency stdio JSON-RPC 2.0 MCP server for ChronoSieve."""

from __future__ import annotations

import json
import math
import sys
from typing import Any, BinaryIO

from ._core import coma_merge, simd_backend, version
from .pipeline import ChronoSieve

PROTOCOL_VERSION = "2024-11-05"
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MiB framing guard


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for strict RFC 8259 JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    return obj


def _strict_float(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    out = float(val)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _strict_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer")
    return val

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "chronosieve_create_channel",
        "description": (
            "Create or reset a ChronoSieve streaming channel with zero-exp E-detector, "
            "Pre-Control E-Reset Conformal PID, optional multi-agent Sleeping BOA fusion, "
            "and anomaly-preserving Swing PLA + lossless ring buffer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Unique channel name."},
                "n_agents": {"type": "integer", "minimum": 1, "default": 1},
                "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "default": 0.10},
                "eta": {"type": "number", "exclusiveMinimum": 0, "default": 0.05},
                "epsilon": {"type": "number", "minimum": 0, "default": 0.05},
                "edet_threshold": {"type": "number", "exclusiveMinimum": 1, "default": 500.0},
                "rogue_threshold": {"type": "number", "exclusiveMinimum": 1, "default": 200.0},
                "peer_relative_rogue": {"type": "boolean", "default": True},
                "ring_capacity": {"type": "integer", "minimum": 64, "default": 65536},
                "anomaly_burst": {"type": "integer", "minimum": 1, "default": 1},
                "wake_cooldown": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["channel"],
        },
    },
    {
        "name": "chronosieve_ingest",
        "description": (
            "Ingest one or more timestamped observations into a ChronoSieve channel. "
            "Returns System-1->System-2 wake-up decision and compact ~100-250 token Markdown "
            "digest packet when a certified shift, rogue agent, or conformal anomaly triggers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "ts": {"type": "array", "items": {"type": "number"}, "description": "Monotonic timestamps."},
                "ys": {"type": "array", "items": {"type": "number"}, "description": "Observed values."},
                "preds": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Optional per-tick list of K agent predictions.",
                },
                "active": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Optional per-tick K-length active bitmask (1=awake, 0=sleeping).",
                },
                "force_markdown": {"type": "boolean", "default": False},
            },
            "required": ["channel", "ts", "ys"],
        },
    },
    {
        "name": "chronosieve_zoom",
        "description": (
            "Lossless O(log N) drill-down over [t0, t1] from a channel's fixed-memory "
            "circular ring buffer. Recovers exact verbatim raw ticks around an anomaly or shift."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "t0": {"type": "number"},
                "t1": {"type": "number"},
            },
            "required": ["channel", "t0", "t1"],
        },
    },
    {
        "name": "chronosieve_digest",
        "description": "Return the latest compact Markdown digest packet for a ChronoSieve channel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "max_segments": {"type": "integer", "minimum": 1, "default": 6},
                "max_anomalies": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["channel"],
        },
    },
    {
        "name": "chronosieve_coma",
        "description": (
            "Compute O(K log K) COMA weighted majority-vote interval consensus across K agents "
            "(Gasparin & Ramdas UAI 2024). Returns exact interval union, convex hull, and max agreement."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lowers": {"type": "array", "items": {"type": "number"}},
                "uppers": {"type": "array", "items": {"type": "number"}},
                "weights": {"type": "array", "items": {"type": "number"}},
                "tau": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "randomized": {"type": "boolean", "default": False},
                "seed": {"type": "integer", "default": 42},
            },
            "required": ["lowers", "uppers"],
        },
    },
    {
        "name": "chronosieve_status",
        "description": "Inspect channel health, coverage, agent weights, rogue flags, and SIMD backend.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Optional channel name (omit to list all)."},
            },
        },
    },
]


class ChronoSieveMCPServer:
    """Stateful MCP JSON-RPC 2.0 handler with zero external dependencies."""

    def __init__(self) -> None:
        self.channels: dict[str, ChronoSieve] = {}

    def _get_channel(self, name: str, auto_create: bool = False) -> ChronoSieve:
        if name not in self.channels:
            if auto_create:
                self.channels[name] = ChronoSieve(channel=name)
            else:
                raise KeyError(f"unknown channel '{name}'")
        return self.channels[name]

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "chronosieve_create_channel":
            if "channel" not in args or not isinstance(args["channel"], str) or not args["channel"]:
                raise ValueError("channel must be a non-empty string")
            ch = args["channel"]
            prr_raw = args.get("peer_relative_rogue", True)
            if not isinstance(prr_raw, bool):
                raise ValueError("peer_relative_rogue must be a boolean")
            pipe = ChronoSieve(
                channel=ch,
                n_agents=_strict_int(args.get("n_agents", 1), "n_agents"),
                alpha=_strict_float(args.get("alpha", 0.10), "alpha"),
                eta=_strict_float(args.get("eta", 0.05), "eta"),
                epsilon=_strict_float(args.get("epsilon", 0.05), "epsilon"),
                edet_threshold=_strict_float(args.get("edet_threshold", 500.0), "edet_threshold"),
                rogue_threshold=_strict_float(args.get("rogue_threshold", 200.0), "rogue_threshold"),
                peer_relative_rogue=prr_raw,
                ring_capacity=_strict_int(args.get("ring_capacity", 65536), "ring_capacity"),
                anomaly_burst=_strict_int(args.get("anomaly_burst", 1), "anomaly_burst"),
                wake_cooldown=_strict_int(args.get("wake_cooldown", 0), "wake_cooldown"),
            )
            self.channels[ch] = pipe
            return {"created": ch, "status": pipe.status(), "simd_backend": simd_backend()}

        if name == "chronosieve_ingest":
            if "channel" not in args or not isinstance(args["channel"], str) or not args["channel"]:
                raise ValueError("channel must be a non-empty string")
            ch = args["channel"]
            ts_raw = args.get("ts")
            ys_raw = args.get("ys")
            if not isinstance(ts_raw, list) or not isinstance(ys_raw, list):
                raise ValueError("ts and ys must be arrays of numbers")
            if len(ts_raw) != len(ys_raw):
                raise ValueError("ts and ys must have identical length")
            preds_raw = args.get("preds")
            active_raw = args.get("active")
            force_md = bool(args.get("force_markdown", False))

            # Pre-validate entire batch atomically before creating or mutating channel state
            ts_vals = [_strict_float(v, f"ts[{i}]") for i, v in enumerate(ts_raw)]
            ys_vals = [_strict_float(v, f"ys[{i}]") for i, v in enumerate(ys_raw)]
            existing = self.channels.get(ch)
            prev_t = existing._last_t if existing is not None else None
            for i, t_v in enumerate(ts_vals):
                if prev_t is not None and t_v < prev_t:
                    raise ValueError(f"ts[{i}]={t_v} is not monotonic (< {prev_t})")
                prev_t = t_v

            expected_k: int | None = existing.n_agents if existing is not None else None
            parsed_active: list[list[int]] | None = None
            if active_raw is not None:
                if not isinstance(active_raw, list) or len(active_raw) != len(ts_vals):
                    raise ValueError("active must be an array matching ts length")
                parsed_active = []
                for i, row in enumerate(active_raw):
                    if not isinstance(row, list) or len(row) == 0:
                        raise ValueError(f"active[{i}] must be a non-empty array")
                    if expected_k is None:
                        expected_k = len(row)
                    elif len(row) != expected_k:
                        raise ValueError(f"active[{i}] length ({len(row)}) != n_agents ({expected_k})")
                    parsed_active.append([1 if bool(v) else 0 for v in row])

            parsed_preds: list[list[float]] | None = None
            if preds_raw is not None:
                if not isinstance(preds_raw, list) or len(preds_raw) != len(ts_vals):
                    raise ValueError("preds must be an array matching ts length")
                parsed_preds = []
                for i, row in enumerate(preds_raw):
                    if not isinstance(row, list) or len(row) == 0:
                        raise ValueError(f"preds[{i}] must be a non-empty array")
                    if expected_k is None:
                        expected_k = len(row)
                    elif len(row) != expected_k:
                        raise ValueError(f"preds[{i}] length ({len(row)}) != n_agents ({expected_k})")
                    a_row = parsed_active[i] if parsed_active is not None else None
                    p_floats: list[float] = []
                    for k, val in enumerate(row):
                        if a_row is not None and a_row[k] == 0:
                            p_floats.append(0.0 if val is None else float(val))
                        else:
                            p_floats.append(_strict_float(val, f"preds[{i}][{k}]"))
                    parsed_preds.append(p_floats)
            elif existing is not None and existing.n_agents > 1 and len(ts_vals) > 0:
                raise ValueError("preds must be provided for multi-agent channels")

            if existing is None:
                inferred_k = expected_k if (expected_k is not None and expected_k >= 1) else 1
                self.channels[ch] = ChronoSieve(channel=ch, n_agents=inferred_k)
            pipe = self.channels[ch]

            wakes = 0
            shifts = 0
            anoms = 0
            last_res: dict[str, Any] | None = None
            last_md: str | None = None

            for i, (t_val, y_val) in enumerate(zip(ts_vals, ys_vals, strict=True)):
                p_row = parsed_preds[i] if parsed_preds is not None else None
                a_row = parsed_active[i] if parsed_active is not None else None
                res = pipe.step(t_val, y_val, preds=p_row, active=a_row)
                last_res = res
                if res["should_wake"]:
                    wakes += 1
                    last_md = res["digest_markdown"]
                if res["shift_detected"]:
                    shifts += 1
                if res["miscovered"]:
                    anoms += 1

            if force_md and last_md is None:
                last_md = pipe.format_markdown()

            return {
                "channel": ch,
                "ingested": len(ts_vals),
                "wakes": wakes,
                "shifts": shifts,
                "miscoverages": anoms,
                "should_wake": wakes > 0,
                "last_state": (
                    {
                        "t": last_res["t"],
                        "y_hat": last_res["y_hat"],
                        "q_next": last_res["q_next"],
                        "emp_coverage": last_res["emp_coverage"],
                        "edet_stat": last_res["edet_stat"],
                        "rogue_agent": last_res["rogue_agent"],
                    }
                    if last_res is not None
                    else None
                ),
                "digest_markdown": last_md,
            }

        if name == "chronosieve_zoom":
            pipe = self._get_channel(str(args["channel"]))
            t0 = _strict_float(args["t0"], "t0")
            t1 = _strict_float(args["t1"], "t1")
            z = pipe.zoom(t0, t1)
            return {"channel": pipe.channel, "t0": t0, "t1": t1, **z}

        if name == "chronosieve_digest":
            pipe = self._get_channel(str(args["channel"]))
            md = pipe.format_markdown(
                max_segments=_strict_int(args.get("max_segments", 6), "max_segments"),
                max_anomalies=_strict_int(args.get("max_anomalies", 5), "max_anomalies"),
            )
            return {"channel": pipe.channel, "digest_markdown": md}

        if name == "chronosieve_coma":
            lowers_raw = args.get("lowers")
            uppers_raw = args.get("uppers")
            if not isinstance(lowers_raw, list) or not isinstance(uppers_raw, list):
                raise ValueError("lowers and uppers must be arrays of numbers")
            lowers = [_strict_float(x, f"lowers[{i}]") for i, x in enumerate(lowers_raw)]
            uppers = [_strict_float(x, f"uppers[{i}]") for i, x in enumerate(uppers_raw)]
            weights_raw = args.get("weights")
            weights: list[float] | None = None
            if weights_raw is not None:
                if not isinstance(weights_raw, list):
                    raise ValueError("weights must be an array of numbers")
                weights = [_strict_float(x, f"weights[{i}]") for i, x in enumerate(weights_raw)]
            return coma_merge(
                lowers,
                uppers,
                weights=weights,
                tau=_strict_float(args.get("tau", 0.5), "tau"),
                randomized=bool(args.get("randomized", False)),
                seed=_strict_int(args.get("seed", 42), "seed"),
            )

        if name == "chronosieve_status":
            ch_opt = args.get("channel")
            if ch_opt is not None:
                pipe = self._get_channel(str(ch_opt))
                return {"version": version(), "simd_backend": simd_backend(), "channel": pipe.status()}
            return {
                "version": version(),
                "simd_backend": simd_backend(),
                "channels": {k: v.status() for k, v in self.channels.items()},
            }

        raise ValueError(f"unknown tool '{name}'")

    def handle_request(self, req: Any) -> dict[str, Any] | None:
        if not isinstance(req, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "invalid request: expected JSON object"},
            }

        method = req.get("method")
        has_id = "id" in req
        req_id = req.get("id")

        if not isinstance(method, str):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "invalid request: missing string method"},
            }

        # JSON-RPC 2.0 notifications omit "id" and require no response
        if not has_id:
            return None

        raw_params = req.get("params")
        if raw_params is None:
            params: dict[str, Any] = {}
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "invalid params: expected JSON object"},
            }

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "chronosieve",
                        "version": version(),
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments")
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "invalid tools/call params"},
                }
            try:
                payload = _sanitize_json(self.call_tool(tool_name, tool_args))
                text = (
                    payload["digest_markdown"]
                    if (tool_name == "chronosieve_digest" and "digest_markdown" in payload)
                    else json.dumps(payload, separators=(",", ":"), allow_nan=False)
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": payload,
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    """Read exactly `length` bytes from stream unless EOF occurs early."""
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _FramingError(ValueError):
    """Protocol framing / UTF-8 / Content-Length error carrying framing mode."""

    def __init__(self, message: str, framed: bool) -> None:
        super().__init__(message)
        self.framed = framed


def _read_message(stream: BinaryIO) -> tuple[Any, bool, bool]:
    """Read one JSON-RPC message supporting both NDJSON and Content-Length framing.

    Returns (parsed_payload, use_content_length, is_eof).
    """
    while True:
        line = stream.readline()
        if not line:
            return None, False, True
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith(b"content-length:"):
            # Consume remaining headers until blank line before validating length
            raw_len = stripped.split(b":", 1)[1].strip()
            while True:
                hdr = stream.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
            try:
                length = int(raw_len.decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise _FramingError(f"invalid Content-Length header: {exc}", True) from exc
            if length < 0 or length > MAX_CONTENT_LENGTH:
                raise _FramingError(
                    f"Content-Length out of range (0..{MAX_CONTENT_LENGTH}): {length}", True
                )
            body = _read_exact(stream, length)
            if len(body) < length:
                if len(body) == 0:
                    return None, True, True
                raise _FramingError("truncated Content-Length message body", True)
            try:
                return json.loads(body.decode("utf-8")), True, False
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _FramingError(f"parse error: {exc}", True) from exc

        # Standard MCP stdio newline-delimited JSON
        try:
            return json.loads(stripped.decode("utf-8")), False, False
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _FramingError(f"parse error: {exc}", False) from exc


def _write_message(stream: BinaryIO, msg: dict[str, Any], use_content_length: bool) -> None:
    clean = _sanitize_json(msg)
    raw = json.dumps(clean, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        stream.write(header)
        stream.write(raw)
    else:
        stream.write(raw)
        stream.write(b"\n")
    stream.flush()


def serve_stdio(
    in_stream: BinaryIO | None = None,
    out_stream: BinaryIO | None = None,
) -> None:
    """Run the stdio JSON-RPC 2.0 MCP server loop until EOF."""
    inp = in_stream if in_stream is not None else sys.stdin.buffer
    outp = out_stream if out_stream is not None else sys.stdout.buffer
    server = ChronoSieveMCPServer()

    while True:
        try:
            msg, framed, is_eof = _read_message(inp)
        except _FramingError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
            _write_message(outp, err, exc.framed)
            continue
        if is_eof:
            break
        resp = server.handle_request(msg)
        if resp is not None:
            _write_message(outp, resp, framed)


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
