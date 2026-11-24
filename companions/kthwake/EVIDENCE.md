# KthWake 0.1.0 performance evidence

**Measured fit:** an in-process agent scheduler with many distinct count thresholds, heterogeneous windows, and a requirement to deliver every transition during silence. The comparison below charges every scheduler for emitting the same chronological edges.

The Azure code-serving trace was selected, and its policy registries frozen, before scoring it. Source timestamps are observed. All 4,096-subscriber registries are modeled. Measurements came from the release team on a shared AMD EPYC 9V74 host under Linux with GCC 14.2. A single process replayed the logical stream as fast as possible. Elapsed time is CPU-side replay wall time. Network, model inference, remote clock synchronization and agent callback I/O are outside these numbers.

## Every idle transition

KthWake and the three alternative implementations received identical timestamped input and drained idle deadlines. Comparators share timestamp storage. They maintain either one timer per distinct rule or one per rank; the strongest rank variant replaces the heap root in a single downward sift. The harness checks complete chronological, per-tick transition lists before timing. Each timing includes an identical callback digest. The winner per row is the alternative with the lowest median elapsed time. Twelve runs per method with rotated order.

| Input | Policy registry | Transitions | KthWake (ms) | Fastest alternative (ms) | Alternative | Alternative time / KthWake |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Mooncake | Individual rates | 1,357,284 | 42.623 | 46.414 | rank-heap-sift | 1.09× |
| Mooncake | Shared rates | 2,514,048 | 4.997 | 6.224 | rank-heap-sift | 1.25× |
| Mooncake | Burst windows | 375,744 | 9.542 | 4.421 | rank-heap-sift | 0.46× |
| Azure holdout | Individual rates | 330,890 | 37.112 | 75.008 | rank-heap-sift | 2.02× |
| Azure holdout | Shared rates | 880,384 | 7.669 | 6.055 | rank-heap-sift | 0.79× |
| Azure holdout | Burst windows | 241,740 | 5.388 | 3.311 | rank-heap-sift | 0.61× |

Values above 1 favor KthWake. The smaller Mooncake individual-rates advantage has overlapping timing distributions. The burst workload favors a dedicated rank heap on both providers. Full measurements and build commands are in [timerbench](timerbench/README.md) and [the Azure replay](replay/README.md).

On November 24, 2026, the release team measured index construction separately on a held-out prefix of the Azure conversational trace. With construction included, KthWake had the lowest median time for individual rates. The single-sift rank heap led on shared rates and burst windows. [Workload, timing boundaries and per-run results](timerbench/README.md#short-lived-scheduler-holdout-november-24-2026)

## Arrival-time snapshots only

These are cheaper semantics: report the net change between observations. The independent Python counting oracle verified the timestamp, subscriber ID and direction of all 1,712,462 observed transitions across these six replays. The comparator set includes rank-grouped and window-grouped native indices with cached deadlines and specialized variants that skip deadline maintenance. Ten rotated repetitions for each of five methods. Construction is timed separately. Both the per-step latency and the total timings include instrumentation.

| Input | Policy registry | KthWake (ms) | Fastest alternative (ms) | Alternative | Alternative time / KthWake |
| --- | --- | ---: | ---: | --- | ---: |
| Mooncake | Individual rates | 3.788 | 3.015 | rank-snapshot | 0.80× |
| Mooncake | Shared rates | 0.324 | 0.344 | window-snapshot | 1.06× |
| Mooncake | Burst windows | 0.155 | 0.149 | rank-snapshot | 0.96× |
| Azure holdout | Individual rates | 17.357 | 12.365 | rank-snapshot | 0.71× |
| Azure holdout | Shared rates | 1.416 | 1.192 | window-snapshot | 0.84× |
| Azure holdout | Burst windows | 1.198 | 1.473 | rank-snapshot | 1.23× |

[Mooncake raw runs](replay/mooncake-results.json) · [Azure raw runs](replay/azure-code-results.json) · [Trace provenance, licenses and reproduction](replay/README.md)

## Synthetic scale and build cost

On the million-subscriber registry with eight ranks and 125,000 windows, the native index averaged **0.276 µs per supplied snapshot** compared with **0.682 µs** for the fastest of four alternatives. This includes callback digest work and amortizes over 2,048 generated observations. It excludes building the registry. KthWake had the lowest average in ten of twelve synthetic cells; duplicate-heavy registries at 10,000 and 100,000 subscribers favored an alternative.

The synthetic suite ranges from 10,000 to 1,000,000 subscribers and includes eight-rank/many-window, eight-window/many-rank, burst/quiet and duplicate-rule registries. Each method receives exactly the same output work. The results include construction, overall elapsed time, p50 and p99. A rare fan-out burst can dominate total time and escape p99. Building a registry can cost more than running a short trace.

[Full crossover table](bench/README.md) · [Machine-readable synthetic measurements](bench/synthetic.jsonl)

## Correctness, scope and replication

The native suite passed 54,604 oracle-checked observations, including a billion-scale rank, uint64 clock limits, duplicate guards, coincident expirations and arrivals, bulk counts, silent periods, invalid inputs, copied indices and a throwing callback. The final release was also checked with address and undefined-behavior sanitizers. The public examples exercise the CLI, native Python binding, asyncio and weighted token reports.

General-purpose stream processors support capabilities beyond this kernel. This evaluation tests implementations of matched native algorithms written for the benchmark; it establishes a workload-specific engineering result on one host. Multiple machines, independently supplied policy registries, and an externally maintained deployment are still needed to establish market leadership or defensibility.

Frozen header SHA-256: `cbe5699dc92016fe49bd0d003836ef6d7916d11bd07cbc746874d6ad5109287b`. Snapshot harness SHA-256: `8452371c48c97f9e74bd7264c35e8b11bd556c0053dd9929cd89fda81c1619c6`.

[Independent reproduction instructions](REPRODUCE.md) · [Research and adoption criteria](PROGRAM.md)
