#!/usr/bin/env python3
"""Compare every native CLI transition to the independent counting oracle's digest."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import subprocess


def check(binary, directory, scenario, expected):
    command = [str(binary), str(directory / f"{scenario}.guards"), str(directory / "ticks.txt")]
    stream_hash = hashlib.sha256()
    counts = collections.Counter()
    previous = -1
    with subprocess.Popen(command, text=True, stdout=subprocess.PIPE) as process:
        with process.stdout:
            edges = (json.loads(line) for line in process.stdout)
            for now, batch in itertools.groupby(edges, key=lambda edge: edge["t"]):
                if now <= previous:
                    raise ValueError("Unordered tick stream")
                previous = now
                for edge in sorted(batch, key=lambda edge: edge["id"]):
                    guard_id, active = edge["id"], edge["active"]
                    if type(active) is not bool:
                        raise ValueError("Expected boolean edge state")
                    counts[active] += 1
                    stream_hash.update(f"{now} {guard_id} {int(active)}\n".encode("ascii"))
    if process.returncode:
        raise SystemExit(f"CLI exited {process.returncode}")
    actual = (counts[True], counts[False], stream_hash.hexdigest())
    wanted = (expected["rising"], expected["falling"], expected["edge_sha256"])
    if actual != wanted:
        raise SystemExit(f"{scenario} mismatch: got {actual}, wanted {wanted}")
    print(f"{scenario}: {sum(counts.values()):,} edges match {actual[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "oracle.json").read_text())
    for name, expected in manifest["scenarios"].items():
        check(args.binary.resolve(), args.directory, name, expected)
