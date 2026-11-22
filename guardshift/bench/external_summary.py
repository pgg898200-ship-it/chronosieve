"""Print the measured native SuperIntervals comparison as a Markdown table."""
import collections
import json
import statistics
import sys


def summarize(path):
    by_workload = collections.defaultdict(lambda: collections.defaultdict(dict))
    checksums = collections.defaultdict(set)
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            workload = row["scenario"], row["guards"]
            sample = row["seed"], row["rep"]
            rows = by_workload[workload][row["method"]]
            if sample in rows:
                raise ValueError(f"duplicate sample {workload} {sample}")
            rows[sample] = row
            checksums[(workload, row["seed"])].add((row["changes"], row["checksum"]))
    if any(len(outputs) != 1 for outputs in checksums.values()):
        raise ValueError("output checksum disagreement")
    print("| Workload | Guards | Guardshift µs | Fastest external µs | External adapter | External / Guardshift |")
    print("| --- | ---: | ---: | ---: | --- | ---: |")
    expected = {(seed, rep) for seed in (17, 42, 217) for rep in range(3)}
    for (scenario, guards), methods in by_workload.items():
        if len(methods) != 5 or any(set(rows) != expected for rows in methods.values()):
            raise ValueError(f"incomplete measurements for {scenario}, {guards}")
        medians = {
            name: statistics.median(row["ns_per_patch"] for row in rows.values()) / 1000
            for name, rows in methods.items()
        }
        native = medians.pop("guardshift")
        name = min(medians, key=medians.get)
        fastest = medians[name]
        print(f"| {scenario} | {guards:,} | {native:.3f} | {fastest:.3f} | "
              f"{name} | {fastest / native:.2f}× |")


if __name__ == "__main__":
    summarize(sys.argv[1])
