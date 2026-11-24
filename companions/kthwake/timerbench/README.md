# Exact timer replay

This harness delivers every state change, including a predicate becoming false between recorded arrivals. It advances straight to the earliest outstanding deadline or the next arrival. An arrival at an exact deadline produces one post-batch observation. The driver drains the outstanding timers after the final arrival.

All four methods consume the same Mooncake timestamps and guard profiles. Before timing, the harness compares the complete sorted sequences of timestamp, subscriber ID, and direction for equality. The timed callbacks also compute a shared timestamp-sensitive checksum. Each replay starts with a newly constructed index. The replay timer includes scheduler work and output hashing. The optional final flag `--with-build` appends `build_ns` to each CSV row, measuring index construction from the in-memory rules. Both intervals exclude fixture parsing, correctness checks, destruction and CSV output.

| Method | Timer management |
| --- | --- |
| KthWake | Public C++ header and its exact `next_expiry()` interface |
| Pair heap | One timer per distinct `(k, window)`, with a shared timestamp ring and deduplicated rule definitions |
| Rank heap | One timer per rank, with a shared timestamp ring and sorted windows |
| Rank heap, single sift | Same rank heap with a fused root replacement, independently applying the same heap optimization as KthWake |

All three heap comparators construct the timer heap in linear time at arrivals. Only groups that reach their deadline are visited during idle periods. Equal rules each dispatch their subscriber callbacks.

## Results

Each number is the median elapsed milliseconds to replay one full trace, from 12 consecutive runs with rotating method order. The host reported an AMD EPYC 9V74 80-Core Processor. GCC 14.2.0 compiled the C++17 benchmark with `-O3 -DNDEBUG`. The process was pinned to logical CPU 8 on a shared Linux host. These results price in-process time. Actual callback I/O, network transport, waking a thread, and operating-system timer jitter require separate measurements.

| Mooncake arrival replay with modeled guards | Idle wakes | Transitions | KthWake | Pair heap | Rank heap | Rank heap, single sift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared policies | 3,784 | 2,514,048 | 5.00 | 7.07 | 6.43 | 6.22 |
| Individual rate policies | 610,513 | 1,357,284 | 42.62 | 123.68 | 65.80 | 46.41 |
| Individual burst windows | 185,433 | 375,744 | 9.54 | 86.49 | 6.79 | 4.42 |

The single-sift rank heap is 2.16× faster on individual burst windows. The individual rate profile's per-run timings overlap. Raw repetitions are in [shared.csv](shared.csv), [varied.csv](varied.csv), and [burst.csv](burst.csv).

## Reproduce

Run from the project root. Linux users can prefix the timing command with `taskset -c 8` or a CPU available on their own host.

```sh
python3 replay/prepare.py --out /tmp/kthwake-mooncake
c++ -O3 -DNDEBUG -std=c++17 timerbench/timers.cpp -o /tmp/kthwake-timers
for profile in shared varied burst; do
  /tmp/kthwake-timers "/tmp/kthwake-mooncake/$profile.guards" /tmp/kthwake-mooncake/ticks.txt 12
done
```

The checks and timed runs use the actual request-arrival timestamps with synthetic guard policies described in the [trace documentation](../replay/README.md). A full timer replay generates more transitions than a consumer observing only arrival snapshots. The harness accepts other strictly increasing timestamp/multiplicity fixtures in the same two-column format.

For custom fixtures, put one whitespace-separated pair of unsigned decimal integers on each nonblank line. Guard rows contain a positive rank and window. Tick rows contain a strictly increasing timestamp and a nonnegative arrival count. The harness checks both files before reporting timings. A malformed row reports its file and line.

On November 23, 2026, the release team verified the fixture reader against the six published Mooncake and Azure profiles. For each of the four implementations, the edge count, idle wake-up count and timestamped digest matched the archived CSV for every profile. The six timelines contained 5,700,090 transitions in total. This check concerns correctness. The release timing results above and in the Azure replay retain their original workload and compiler settings.

## Short-lived scheduler holdout, November 24, 2026

The [reproduction thread](https://github.com/pgg898200-ship-it/chronosieve/issues/2) asks runners to account for setup separately. We tested the new column on a distinct file from the Azure code-service release benchmark. Before collecting timings, we selected the first 256 distinct arrival milliseconds from Microsoft's conversational inference trace, all three existing 4,096-subscriber policy sets and 12 rotating repetitions. This prefix contains 258 requests over 73.501 seconds of event time. Every method drained idle deadlines after the final arrival and passed the full edge-list equality check.

| Policy | Edges | KthWake build (ms) | KthWake replay (ms) | KthWake build + replay (ms) | Best heap build + replay (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared rates | 40,576 | 0.291 | 0.202 | 0.502 | 0.325 |
| Individual rates | 11,942 | 0.228 | 0.493 | 0.726 | 0.819 |
| Burst windows | 10,228 | 0.183 | 0.159 | 0.346 | 0.284 |

Each total is the median of the per-run sums. The single-sift rank heap had the lowest median total among the three comparators on every profile. It beat KthWake on shared rates and burst windows. KthWake had the lower median on individual rates. These are release-team measurements on a shared AMD EPYC 9V45 host, Linux 6.18.57 x86-64, GCC 14.2.0, pinned to CPU 4. The process reused its allocator across repetitions after correctness checks. These numbers describe constructing an index inside a running process. Process startup and external callback delivery were outside the measurement. [All 144 measurements](azure-conversation-prefix.csv)

Run from `companions/kthwake`. Choose an available CPU for your machine.

```sh
python3 replay/prepare.py --out /tmp/kthwake-rules
c++ -O3 -DNDEBUG -std=c++17 timerbench/timers.cpp -o /tmp/kthwake-timers
for profile in shared varied burst; do
  taskset -c 4 /tmp/kthwake-timers "/tmp/kthwake-rules/$profile.guards" timerbench/azure-conversation-prefix.ticks 12 --with-build
done
```

The bundled [tick file](azure-conversation-prefix.ticks) derives from the [pinned Microsoft CSV](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/data/AzureLLMInferenceTrace_conv.csv), distributed under [CC BY 4.0](../replay/LICENSE-Azure). Source attribution is to Microsoft Azure and Patel et al., *Splitwise: Efficient generative LLM inference using phase splitting*, ISCA 2024. Only relative arrival times and multiplicities are retained. To verify the conversion, download the pinned CSV to the working directory and run:

```python
from itertools import groupby, islice
from pathlib import Path
from replay.prepare import azure_arrival_ms, digest

source = Path("AzureLLMInferenceTrace_conv.csv").read_bytes()
assert digest(source) == "2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8"
groups = islice(groupby(azure_arrival_ms(source)), 256)
ticks = "".join(f"{t} {sum(1 for _ in rows)}\n" for t, rows in groups)
assert ticks.encode("ascii") == Path("timerbench/azure-conversation-prefix.ticks").read_bytes()
```
