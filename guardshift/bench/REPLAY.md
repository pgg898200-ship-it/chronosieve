# Benchmark your own guard workload

Run Guardshift and the native reference implementations on the exact same event sequence. The command reads a local file or standard input and writes one JSON object per method per round. It makes no network requests. Output contains aggregate counts and timings. Signal values, rule definitions, guard IDs, file names and trace fingerprints stay out of the report.

Run these commands from the Guardshift project directory with a C++17 GCC or Clang compiler. In the GitHub release checkout, enter it with `cd guardshift`.

```sh
c++ -std=c++17 -O2 -DNDEBUG -Iinclude bench/replay.cpp -o /tmp/guardshift-replay
/tmp/guardshift-replay examples/fleet.trace
/tmp/guardshift-replay - < examples/fleet.trace > comparison.jsonl
```

The six patches in the example produce six transitions for each method. Replace the sample with a longer trace from your application for timing comparisons. For each method, take the median of its three `ns_per_patch` values, then compare methods at equal trace and build options. `p50_ns` and `p99_ns` are per-round nearest-rank percentiles over individual patches. The per-patch timer includes the API call and returned transitions. Compilation, parsing, oracle verification, result destruction and timing bookkeeping are outside it. Index construction is reported separately in `build_ns`.

## Compare your trace with SuperIntervals

To run the four [SuperIntervals adapters](external.md) on your own input, compile the GitHub release version of the replay runner with the optional flag below. The default build needs only the Guardshift sources. The clone downloads the external comparator once. Running the compiled comparison stays local.

```sh
git clone https://github.com/kcleal/superintervals.git /tmp/guardshift-superintervals
git -C /tmp/guardshift-superintervals checkout --detach e37e4031ca0ce84f2e9b7f8e906e2ac0e7496645
c++ -std=c++17 -O2 -DNDEBUG -DGUARDSHIFT_WITH_SUPERINTERVALS \
    -Iinclude -I/tmp/guardshift-superintervals/src bench/replay.cpp -o /tmp/guardshift-replay-si
/tmp/guardshift-replay-si - < examples/fleet.trace > comparison.jsonl
```

Replace `examples/fleet.trace` with your local trace. The example produces 30 verified rows across three rounds, with six transitions in each row. The optional methods are `superintervals` and `superintervals_eytz` for point queries, and `superintervals_edges` and `superintervals_edges_eytz` for crossed endpoints. Each consumes the same coalesced patches and must match the same ordered transitions and final guard states. Compare the median of three round means for Guardshift with the lowest such median among the four external methods. Include the winning adapter name and build time when reporting results. The six-patch example checks the installation. Use a representative longer trace to assess performance.

## Trace format

Use whitespace-separated decimal integers, one record per line. Blank lines and comments beginning with `#` are ignored. Name fields in local comments if helpful.

| Record | Meaning |
| --- | --- |
| `init v0 v1 ...` | Initial signed 64-bit signal values. The first signal has ID 0. Exactly one initialization line must precede the rules. |
| `guard field low high ...` | Each triple is a closed interval. All terms must hold. A guard can mention a field once. Guard IDs follow declaration order. An empty guard always holds. |
| `patch field value ...` | Apply all pairs atomically. A repeated field keeps its last value. The reader coalesces repeat writes before timing so all engines receive identical input. An empty patch has no effect. |

Define at least one guard followed by at least one patch. Invalid numbers, signal IDs, repeated terms, reversed bounds and malformed records are rejected. Identical definitions on separate guard lines represent distinct subscribers.

## What a verified row means

The runner compares every emitted guard ID and direction, in order, against an independent native full scan and checks the final state of every guard. A mismatch exits nonzero. Use the output only if the process exits successfully. The runner rotates method order across three rounds and always includes full scan, incidence counting, endpoint counting, interval stabbing, and Guardshift. It adds the independent slab bitmap baseline for traces with at most four fields whose bitmap cells fit 8 MiB. For unary single-write workloads with coordinates within ±2⁶¹, it also includes direct endpoint and bounded-width interval indexes. See the [method descriptions](README.md).

This local runner keeps the trace and reference transitions in memory. It supports up to 4,096 signals, one million guards, one million patches, four million input terms and writes combined, and four million emitted reference transitions. Input is limited to 64 MiB total and 1 MiB per line. The full-scan oracle visits every guard for every patch. A shorter contiguous replay can make verification practical on very large fleets. Preserve the correct starting state when extracting that segment.
