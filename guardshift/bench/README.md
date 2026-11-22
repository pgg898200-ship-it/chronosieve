# Reproduce the comparative benchmark

From the checkout root, use a C++17 compiler and an otherwise idle CPU.

```sh
c++ -std=c++17 -O3 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/guardshift-bench
/tmp/guardshift-bench --quick > quick.jsonl
/tmp/guardshift-bench --full --trace replay/azure-code.csv > full.jsonl
```

Each JSON line records one engine, configuration, seed and repetition. Quick mode runs two rounds with seed 17 at 1,024 and 8,192 guards. Full mode runs five rounds at 1,024, 8,192 and 32,768 guards for seeds 17, 42 and 217. Single-column teleport uses 100,000 guards. The optional replay uses 8,192. Engine order rotates each round.

| Method | Work on a patch |
| --- | --- |
| `full_scan` | Evaluate every guard, short-circuiting on its first failed term. Uses the caller's immutable guard definitions. |
| `incidence_count` | Visit only terms attached to a changed signal, range-test old and new values, and update the number of failed terms for affected guards. |
| `endpoint_count` | Binary-search sorted lower and upper endpoints for each changed signal, visit crossed endpoints, update the same counts, and emit final changes. |
| `direct_1d` | Unary specialization with lower/upper endpoint searches, direct evaluation at the two positions and no counters. |
| `bounded_width_1d` | Unary specialization with sorted interval starts. Two point queries use the maximum interval width to restrict the candidate range. |
| `slab_bits` | Low-cardinality fields compile into sorted ranges, each with a guard bitmap. Intersect the selected ranges, XOR with the prior truth bits, and enumerate changed IDs by scanning set bits. Applied to the Azure and quantized cases. |
| `guardshift` | The public `Index` API, including validation, patch coalescing and ordered result construction. |

Every engine returns changes sorted by guard ID. The timer includes the call and materialization of those changes. Output order and uniqueness checks, checksum calculation, result deallocation, final-state verification and input generation run outside the timer. All timed patches are compared exactly with the full-scan truth-transition list, even for engines that lose. A disagreement exits nonzero. Durations are individual steady-clock measurements without subtracting timer cost. `elapsed_ns` sums patch durations, `ns_per_patch` divides by patch count, and `p50_ns` and `p99_ns` are nearest-rank order statistics. `build_ns` measures construction separately. Compare medians across repetitions and seeds at equal size and scenario. On the shortest paths the timer itself is a meaningful fraction of latency.

| Scenario | Workload |
| --- | --- |
| `sparse` | 32 signals, four distinct fields per guard, single-field random walk, steps up to 64 over a domain of ±32,768. Most updates cross few boundaries. |
| `atomic` | Same guards and drift, four different fields updated in one atomic patch. |
| `jump` | Same broad domain, one signal jumps to a new random position on each patch. |
| `churn` | Four-term rules over a small integer domain with frequent tied endpoints and jumps. |
| `unary` | One field, broad overlapping intervals, small local updates. Both specialist indexes run here. |
| `teleport` | One field jumps uniformly over [0, 1,000,000]. Interval starts are uniform over [0, 999,900] and widths uniform over [1, 100]. Many boundaries lie between old and new positions while few intervals contain either position. Both specialists run here. |
| `quantized` | Two ordinal fields each have 64 possible integer values. Every rule has two independently generated inclusive ordinal ranges. Both fields jump each patch. |
| `azure` | Two rolling workload signals derived from a public Azure inference trace, passed together once per request. The first 128 pairs select synthetic interval bounds from historical per-field quantiles and initialize the state. Subsequent requests are evaluated in file order. These guards represent a generated subscription workload. |

The product and incidence indexes support arbitrary guard structures. Both 1D specialists assume one interval per guard on a single signal and one write per patch. Each specializes a different workload. This benchmark measures a single-writer, in-process data structure with static guards. Network transit, guard recompilation, distributed coordination, ingestion, contention, and end-to-end inference time require separate measurements.

## Recorded run

The raw measurements in `results/native.jsonl` used GCC 14.2.0 (Debian 14.2.0-19), C++17, `-O3 -DNDEBUG`, Linux x86-64 6.18.57, and a virtualized AMD EPYC 9V74. One benchmark process was pinned to logical CPU 2 on a shared host. Wall-clock observations include any scheduling delays. The full run took 194.95 seconds. Its 1,380 rows cover 1,594,080 timed method calls and 742,733,275 returned changes, counting repeated measurements and alternative engines. The release header SHA-256 was `a56be79698363b9245cbcb86be06f9f4a0234f4c417c27d5938901599747dd44`.

```sh
taskset -c 2 /tmp/guardshift-bench --full --trace replay/azure-code.csv > results/native.jsonl
taskset -c 2 /tmp/guardshift-bench --million > results/million.jsonl
taskset -c 2 /tmp/guardshift-bench --mixed > results/mixed.jsonl
```

The million-guard scale run took 26.38 seconds with 192 patches per workload, two rounds and seed 17. The 100,000-guard mixed-width test took 11.86 seconds with 512 patches per workload, five rounds and three seeds. Its two-term guards span eight signals. Each combines a route interval and a broad eligibility interval. Route intervals have a 90% chance of width 1–100 and a 10% chance of width 1,000–90,999 over a domain of one million. A route field jumps each patch and an eligibility field also jumps every eighth patch. An additional `stabbing_count` comparator queries a sorted-low interval tree with subtree maximum-high summaries at the old and new positions. It maintains guard truth counts and discards intervals active at both positions.

Median of each round's mean ns/patch, with the fastest comparator measured in the corresponding native suite. Changes/patch is the median across runs. Close timings can reverse on another host. Raw records contain p99, build times and every comparator.

| Guards | Scenario | Changes/patch | GuardShift ns | Fastest comparator ns | Comparator |
| ---: | --- | ---: | ---: | ---: | --- |
| 8,192 | `sparse` | 0.26 | 72 | 97 | `endpoint_count` |
| 8,192 | `atomic` | 0.91 | 153 | 286 | `endpoint_count` |
| 8,192 | `azure` | 117.29 | 1,292 | 1,241 | `slab_bits` |
| 8,192 | `quantized` | 1490.50 | 10,776 | 10,494 | `slab_bits` |
| 8,192 | `jump` | 62.83 | 10,527 | 4,091 | `endpoint_count` |
| 8,192 | `churn` | 62.81 | 12,281 | 3,908 | `endpoint_count` |
| 8,192 | `unary` | 7.81 | 240 | 282 | `direct_1d` |
| 32,768 | `sparse` | 1.08 | 197 | 219 | `endpoint_count` |
| 32,768 | `atomic` | 4.01 | 442 | 654 | `endpoint_count` |
| 100,000 | `teleport` | 10.12 | 1,207 | 537 | `bounded_width_1d` |
| 100,000 | `mixed_jump` | 161.90 | 40,968 | 42,712 | `stabbing_count` |
| 1,000,000 | `sparse` | 30.10 | 10,380 | 5,006 | `endpoint_count` |
| 1,000,000 | `teleport` | 101.29 | 10,908 | 4,164 | `bounded_width_1d` |

[Four adapters for upstream SuperIntervals](external.md) · [Compare a local trace](REPLAY.md)
