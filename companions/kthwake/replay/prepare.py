#!/usr/bin/env python3
"""Prepare trace ticks, synthetic subscriptions and independent expected outputs."""
import argparse
import bisect
import collections
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import urllib.request

TRACES = {
    "mooncake": (
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/e94a0b86ba067455d8b0524eb2cbb5fbac2db024/FAST25-release/traces/toolagent_trace.jsonl",
        "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71",
    ),
    "azure-code": (
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/data/AzureLLMInferenceTrace_code.csv",
        "54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6",
    ),
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def azure_arrival_ms(content):
    # Preserve all seven fractional digits until flooring elapsed time to ms.
    stamps = []
    epoch = datetime(1970, 1, 1)
    for row in csv.DictReader(content.decode("ascii").splitlines()):
        seconds, fraction = row["TIMESTAMP"].split(".")
        whole = datetime.fromisoformat(seconds) - epoch
        ticks = (whole.days * 86400 + whole.seconds) * 10_000_000
        stamps.append(ticks + int(fraction.ljust(7, "0")))
    stamps.sort()
    start = stamps[0]
    return [(t - start) // 10_000 for t in stamps]


def read_trace(content, trace):
    if digest(content) != TRACES[trace][1]:
        raise ValueError("Source SHA-256 mismatch; use the pinned revision")
    if trace == "mooncake":
        stamps = [json.loads(line)["timestamp"] for line in content.splitlines()]
        if stamps != sorted(stamps) or any(type(t) is not int or t < 0 for t in stamps):
            raise ValueError("Expected ordered, nonnegative integer timestamps")
    else:
        stamps = azure_arrival_ms(content)
    return [(t, sum(1 for _ in group)) for t, group in itertools.groupby(stamps)]


def write_pairs(path, pairs):
    raw = "".join(f"{a} {b}\n" for a, b in pairs).encode("ascii")
    path.write_bytes(raw)
    return digest(raw)


def guards(name):
    rates = [2, 4, 5, 6, 7, 8, 10, 12]
    shared_windows = [1000, 3000, 5000, 10000, 15000, 30000, 45000, 60000]
    for idx in range(4096):
        if name == "burst":
            # Per-subscriber tolerance, fleet-wide cardinality budgets.
            window = 1000 + (idx * 7919) % 59001
            yield (4 << (idx % 8)), window
            continue
        if name == "shared":
            window = shared_windows[(idx // len(rates)) % len(shared_windows)]
            rate = rates[idx % len(rates)]
        else:
            # Coprime stride gives different windows to all 4,096 guards.
            window = 1000 + (idx * 7919) % 59001
            rate = rates[idx % len(rates)]
        yield (window * rate + 999) // 1000, window


def oracle(ticks, rules):
    """Recount each window via cumulative arrivals; hash per-tick sorted flips."""
    by_window = collections.defaultdict(list)
    for guard_id, (k, window) in enumerate(rules):
        by_window[window].append((k, guard_id))
    times = [t for t, _ in ticks]
    prefix = [0, *itertools.accumulate(n for _, n in ticks)]
    states = bytearray(len(rules))
    stream_hash = hashlib.sha256()
    rises = falls = 0
    for i, (now, _) in enumerate(ticks):
        flips = []
        for window, items in by_window.items():
            begin = bisect.bisect_right(times, now - window, 0, i + 1)
            count = prefix[i + 1] - prefix[begin]
            for k, guard_id in items:
                state = count >= k
                if state != states[guard_id]:
                    states[guard_id] = state
                    flips.append((guard_id, state))
                    if state:
                        rises += 1
                    else:
                        falls += 1
        for guard_id, state in sorted(flips):
            stream_hash.update(f"{now} {guard_id} {int(state)}\n".encode("ascii"))
    return {"guards": len(rules), "distinct_rules": len(set(rules)),
            "distinct_windows": len(by_window), "distinct_ks": len({k for k, _ in rules}),
            "rising": rises, "falling": falls, "final_active": sum(states),
            "edge_sha256": stream_hash.hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", choices=TRACES, default="mooncake")
    origin = p.add_mutually_exclusive_group()
    origin.add_argument("--source", type=Path, help="local copy of the pinned upstream file")
    origin.add_argument("--fetch", action="store_true", help="regenerate ticks from the pinned upstream URL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--heartbeat-ms", type=int, default=0,
                   help="add zero-arrival ticks at this interval (default 0)")
    args = p.parse_args()
    if args.heartbeat_ms < 0:
        p.error("heartbeat interval must be nonnegative")
    source, checksum = TRACES[args.trace]
    if args.source:
        ticks = read_trace(args.source.read_bytes(), args.trace)
    elif args.fetch:
        with urllib.request.urlopen(source, timeout=60) as response:
            ticks = read_trace(response.read(), args.trace)
    else:
        bundled = Path(__file__).with_name(f"{args.trace}-ticks.txt").read_bytes()
        manifest_name = "oracle.json" if args.trace == "mooncake" else "azure-code-oracle.json"
        expected = json.loads(Path(__file__).with_name(manifest_name).read_text())["ticks_sha256"]
        if digest(bundled) != expected:
            raise ValueError("Bundled trace checksum mismatch")
        # The last tick is a synthetic zero-arrival flush, regenerated below.
        ticks = [tuple(map(int, line.split())) for line in bundled.splitlines()[:-1]]
    observed_times = len(ticks)
    last_arrival = ticks[-1][0]
    events = sum(n for _, n in ticks)
    ticks.append((last_arrival + 60000, 0))  # final expiry of the longest window
    if args.heartbeat_ms:
        counts = dict(ticks)
        for t in range(0, ticks[-1][0], args.heartbeat_ms):
            counts.setdefault(t, 0)
        ticks = sorted(counts.items())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats = {"trace": args.trace, "source": source, "source_sha256": checksum, "source_requests": events,
             "source_distinct_times": observed_times, "last_arrival_ms": last_arrival,
             "ticks": len(ticks), "heartbeat_ms": args.heartbeat_ms,
             "ticks_sha256": write_pairs(out / "ticks.txt", ticks), "scenarios": {}}
    for shape in ["shared", "varied", "burst"]:
        rules = list(guards(shape))
        rules_sha = write_pairs(out / f"{shape}.guards", rules)
        stats["scenarios"][shape] = dict(oracle(ticks, rules), guards_sha256=rules_sha)
    manifest = json.dumps(stats, sort_keys=True, indent=2) + "\n"
    (out / "oracle.json").write_text(manifest, encoding="ascii")
    print(manifest, end="")


if __name__ == "__main__":
    main()
