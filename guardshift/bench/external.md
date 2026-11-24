# Native SuperIntervals comparison

The comparator uses [SuperIntervals 1.0.3](https://github.com/kcleal/superintervals/tree/e37e4031ca0ce84f2e9b7f8e906e2ac0e7496645) by Kez Cleal, distributed under MIT. Guardshift itself has no dependency on it. Both libraries process identical 64-bit integer rules, initial states and updates in the same binary. All returned transitions and the final active state of every guard must agree with a full recomputation.

The point-query adapter builds a static interval index per field and caches its previous matching interval indices. It queries the new value once per changed field, merges the two descending index lists, and updates per-guard mismatch counts for the symmetric difference. This favors jumps between sparse active sets.

The endpoint-query adapter stores guard IDs and transition directions at singleton interval endpoints. It sorts these points before calling the library’s build operation and releases each field’s temporary input to bound build memory. It issues one range query per changed field and updates counters for each crossed threshold. This favors sparse movements across a dense population of active ranges. Both adapters use the library's native C++ API with the default and Eytzinger layouts. The benchmark sorts and emits final changed guard IDs in all cases.

For a local workload, [run these four adapters on your own trace](REPLAY.md#compare-your-trace-with-superintervals).

## Reproduce

Run from the project root with GCC, Git and Linux taskset. Set the CPU to one in your process affinity mask. Current raw results are in [external.jsonl](../results/external.jsonl).

```sh
git clone https://github.com/kcleal/superintervals.git /tmp/guardshift-superintervals
git -C /tmp/guardshift-superintervals checkout --detach e37e4031ca0ce84f2e9b7f8e906e2ac0e7496645
g++ -std=c++17 -O3 -DNDEBUG --param ggc-min-expand=20 --param ggc-min-heapsize=32768 \
    -Iinclude -I/tmp/guardshift-superintervals/src bench/external.cpp -o /tmp/guardshift-external
MALLOC_TRIM_THRESHOLD_=0 taskset -c 8 /tmp/guardshift-external replay/azure-code.csv > /tmp/external.jsonl
python3 bench/external_summary.py /tmp/external.jsonl
```

The measured host reported AMD EPYC 9V74 and GCC 14.2.0. The harness rotates implementation order across three repetitions for each of three seeds. Timings cover the call to apply a patch and materialize its sorted transition list. Index construction, oracle checks, output destruction, ingestion, network delivery and agent execution are outside the timed region. Each record includes build time, mean patch time, p50, p99 and an ordered-transition checksum. The rolling Azure signals are trace-derived, while the guard definitions are generated. The other workloads and guards are synthetic. The host was shared, and the microbenchmarks cover warmed in-process engines with fixed guard sets.

## Reproduce a win and a loss

After compiling the executable above, use `--crossover` to run just the 8,192-guard atomic and 32,768-guard jump workloads. This skips the million-guard builds and does not require the Azure fixture. It keeps the same three seeds, three repetitions, four external adapters and full-recomputation checks.

```sh
MALLOC_TRIM_THRESHOLD_=0 taskset -c 8 /tmp/guardshift-external --crossover > /tmp/crossover.jsonl
python3 bench/external_summary.py /tmp/crossover.jsonl
```

On November 23, 2026, an AMD EPYC 9V74 core with GCC 14.2.0 produced the following medians of nine run means. The fastest external adapter on both workloads used the Eytzinger endpoint layout. [All 90 measurements](../results/crossover.jsonl)

| Workload | Guards | Guardshift µs/patch | SuperIntervals µs/patch |
| --- | ---: | ---: | ---: |
| Atomic four-field patch | 8,192 | 0.136 | 0.295 |
| Jump | 32,768 | 23.107 | 15.916 |

## Measurements

Each entry is the median of nine run means. The fastest external column selects the lowest median among all four SuperIntervals configurations for that workload. Ratios above one favor Guardshift.

| Workload | Guards | Guardshift µs | Fastest external µs | External adapter | External / Guardshift |
| --- | ---: | ---: | ---: | --- | ---: |
| atomic | 8,192 | 0.142 | 0.316 | superintervals_edges_eytz | 2.23× |
| atomic | 32,768 | 0.376 | 0.633 | superintervals_edges_eytz | 1.68× |
| azure | 8,192 | 1.119 | 8.010 | superintervals_edges_eytz | 7.16× |
| mixed_jump | 100,000 | 32.779 | 18.565 | superintervals_eytz | 0.57× |
| teleport | 100,000 | 1.034 | 0.719 | superintervals_eytz | 0.70× |
| jump | 32,768 | 25.167 | 17.400 | superintervals_edges | 0.69× |
| sparse | 1,000,000 | 4.375 | 5.207 | superintervals_edges_eytz | 1.19× |
| mixed_jump | 1,000,000 | 836.391 | 404.764 | superintervals_eytz | 0.48× |

See [native controls](README.md) for direct endpoint counting, bitsets and other specialized alternatives.
