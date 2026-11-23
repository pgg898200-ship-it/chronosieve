# Exact timer replay

This harness delivers every state change, including a predicate becoming false between recorded arrivals. It advances straight to the earliest outstanding deadline or the next arrival. An arrival at an exact deadline produces one post-batch observation. The driver drains the outstanding timers after the final arrival.

All four methods consume the same Mooncake timestamps and guard profiles. Before timing, the harness compares the complete sorted sequences of timestamp, subscriber ID, and direction for equality. The timed callbacks also compute a shared timestamp-sensitive checksum. Each process builds its index and performs correctness checks outside the measured interval. The timed interval includes scheduler work and output hashing.

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
