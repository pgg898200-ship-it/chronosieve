#!/usr/bin/env python3
"""Build and run all exact snapshot-routing scenarios on a single logical CPU."""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
FLAGS = ["-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-Wall", "-Wextra", "-pedantic"]


def cpu_name():
    if Path("/proc/cpuinfo").exists():
        m = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
        if m:
            return m.group(1)
    return platform.processor() or platform.machine()


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["name"], r["guards"], r["method"])].append(r)
    for (name, size, method), run in groups.items():
        def median(key):
            return statistics.median(v[key] for v in run)
        us = median("elapsed_ns") / run[0]["ticks"] / 1000
        print(f"{name:18} {size:>8} {method:>16}  {us:>9.3f} us/step"
              f"  p99 {median('p99_ns') / 1000:9.3f} us"
              f"  build {median('build_ns') / 1e6:8.2f} ms", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="JSONL results path")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    p.add_argument("--shapes", nargs="+", default=["many-windows", "few-windows", "bursty", "duplicates"])
    p.add_argument("--cpu", type=int)
    args = p.parse_args()
    if args.reps < 1 or args.steps < 1 or any(n < 1 for n in args.sizes):
        p.error("sizes, reps, and steps must all be positive")
    if hasattr(os, "sched_getaffinity"):
        cores = os.sched_getaffinity(0)
        cpu = min(cores) if args.cpu is None else args.cpu
        if cpu not in cores:
            p.error(f"CPU {cpu} is outside the current affinity mask")
        os.sched_setaffinity(0, {cpu})
    else:
        cpu = None
        if args.cpu is not None:
            p.error("pinning a CPU is unsupported on this platform")
    cxx = os.environ.get("CXX", "c++")
    meta = {"kind": "metadata", "cpu": cpu_name(), "pinned_cpu": cpu,
            "compiler": subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
            "flags": FLAGS, "platform": platform.platform(),
            "kernel_sha256": hashlib.sha256((ROOT / "include/kthwake.hpp").read_bytes()).hexdigest(),
            "bench_sha256": hashlib.sha256((BENCH / "bench.cpp").read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "bench"
        subprocess.run([cxx, *FLAGS, str(BENCH / "bench.cpp"), "-o", str(exe)], check=True)
        with args.out.open("w") as out:
            out.write(json.dumps(meta) + "\n")
            for size in args.sizes:
                for shape in args.shapes:
                    completed = subprocess.run([str(exe), "--shape", shape, "--size", str(size),
                                                "--reps", str(args.reps), "--steps", str(args.steps)],
                                               check=True, text=True, stdout=subprocess.PIPE)
                    rows = [json.loads(line) for line in completed.stdout.splitlines()]
                    out.write(completed.stdout)
                    out.flush()
                    summarize(rows)


if __name__ == "__main__":
    main()
