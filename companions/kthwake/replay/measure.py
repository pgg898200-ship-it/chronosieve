#!/usr/bin/env python3
"""Build and run the native counter baselines on all trace subscription profiles."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model():
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True, help="directory written by prepare.py")
    p.add_argument("--out", type=Path, required=True, help="output JSON containing timings and provenance")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--cpu", type=int, help="pin to a logical CPU (Linux)")
    p.add_argument("--cxx", default="c++")
    p.add_argument("--native", action="store_true", help="compile with -march=native")
    args = p.parse_args()
    if args.reps < 1:
        p.error("repetitions must be positive")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            p.error("CPU pinning requires sched_setaffinity support")
        os.sched_setaffinity(0, {args.cpu})
    inputs = args.inputs
    expected = json.loads((inputs / "oracle.json").read_text())
    header = ROOT / "include/kthwake.hpp"
    source = ROOT / "bench/bench.cpp"
    flags = ["-std=c++17", "-O3", "-DNDEBUG"]
    if args.native:
        flags.append("-march=native")
    result = {"machine": platform.machine(), "os": f"{platform.system()} {platform.release()}",
              "cpu": cpu_model(), "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
              "compiler_flags": flags, "source_sha256": sha(source), "header_sha256": sha(header),
              "input_manifest_sha256": sha(inputs / "oracle.json"), "repetitions": args.reps,
              "summary": {}, "runs": []}
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp) / "bench"
        subprocess.run([args.cxx, *flags, str(source), "-o", str(bench)], check=True)
        for shape, oracle in expected["scenarios"].items():
            command = [str(bench), "--guards", str(inputs / f"{shape}.guards"), "--ticks",
                       str(inputs / "ticks.txt"), "--name", f"{expected['trace']}-{shape}",
                       "--reps", str(args.reps), "--lazy", "1"]
            lines = subprocess.check_output(command, text=True)
            groups = collections.defaultdict(list)
            for line in lines.splitlines():
                row = json.loads(line)
                if row["edges"] != oracle["rising"] + oracle["falling"]:
                    raise ValueError(f"Edge count mismatch in {shape}: {row}")
                groups[row["method"]].append(row)
                result["runs"].append(row)
            result["summary"][shape] = {
                method: {key: statistics.median(r[key] for r in rows)
                         for key in ["elapsed_ns", "build_ns", "p50_ns", "p99_ns"]}
                for method, rows in groups.items()}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
