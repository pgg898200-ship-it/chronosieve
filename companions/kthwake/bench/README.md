# Snapshot benchmark

All five C++17 methods evaluate the same operation. For each supplied `now, arrivals` tick, apply all arrivals atomically and emit an edge `(guard ID, active)` for every guard whose truth value changed since the preceding tick. An event at time `s` counts while `now − s < window`. A zero-arrival heartbeat can therefore emit falling edges. A rise and fall between observations are invisible. For exact on-deadline delivery, the caller has to wake at a deadline; see the separate timer benchmark.

## Reproduce

From the repository root, use Python 3 and GCC or Clang.

```sh
python3 bench/run.py --out /tmp/kthwake-results.jsonl --steps 2048 --reps 5
```

Linux runs pin all implementations to the first available logical CPU. Pass `--cpu N` to choose another allowed CPU. Native tuning targets the host CPU. The first JSONL record includes CPU, compiler, flags and SHA-256 of the benchmark and kernel sources. Subsequent records give elapsed time, build time, p50 and p99 step latency in nanoseconds, input counts, output count and digest for every repetition. Compute throughput as `ticks * 1e9 / elapsed_ns`. Each tick includes two steady-clock calls. The elapsed timer also includes sample recording and loop overhead. For very short calls timer resolution and overhead are material. Construction timing includes index construction and excludes generation, process startup, destruction and checking.

To supply your own trace and guard inventory, use whitespace-separated nonnegative decimal integers. Each guard row contains `k window`; each trace row contains `now arrivals`. Custom runs require positive guards, strictly increasing tick times, and a cumulative number of arrivals that fits in uint64. The rank comparator allocates one timestamp per occurrence up to the largest k. Choose sizes that fit available memory.

```sh
c++ -std=c++17 -O3 -DNDEBUG -march=native bench/bench.cpp -o /tmp/kthwake-bench
/tmp/kthwake-bench --guards guards.txt --ticks ticks.txt --name my-trace --reps 5
```

## Comparators

- **group-by-k** keeps one bounded timestamp ring shared by all ranks. Each eligible rank gets its supporting timestamp with one indexed lookup. Guards sharing that rank are sorted by window. The comparator checks the previous boundary and uses binary search when the boundary moves. The first active window gives an exact next expiry per rank, permitting clock-only invocations to return early.
- **group-by-window** shares a single timestamp and cumulative-count log across window groups. A monotone left cursor per window removes expired arrivals. Each group uses the same boundary check and binary search among sorted ranks. The earliest live event expiry in an active window is a conservative wake time. This fixed-trace comparator retains its shared log until the run ends.
- **rank-snapshot** and **window-snapshot** use those same indexes with all deadline maintenance disabled at compile time. Each processes the supplied observations directly. This tests the snapshot contract independently of any speed gained or lost by maintaining a deadline.
- All four comparators store 16 bytes per guard entry, group equal keys together, and emit every distinct guard ID including guards with identical predicates. They perform one boundary query per visited group. Callers invoke all implementations on the same clock ticks. Work to request a dynamic wakeup from an OS, kernel timer, event loop or broker lies outside this benchmark.

Before timing, an independent instance of each engine emits a complete edge list per tick. The harness sorts and requires exact equality of all five lists. During timing, every callback feeds its ID and state into an identical order-insensitive digest and increments the edge count. This measures the in-process enumeration of transitions. Serialization, network delivery and client callbacks with real work would add cost. Trial order cycles through all five starting positions. The check warms input data, and every timed run constructs a new, initially empty engine. No engine receives historical arrivals for free.

## Synthetic guard and arrival distributions

The published raw measurements use seed 4242. The engine sees the same shuffled guard IDs and arrival trace on every repetition. The random stream is seeded separately for guard ordering and ticks. Each scenario ran with 10k, 100k and 1M guards. These fixed observation times are part of the workload.

| Workload | Guard distribution | Arrivals over 2,048 ticks |
|---|---|---|
| many-windows | Eight ranks from 1 to 128 (powers of two), crossed with consecutive integer windows from 1 through N/8 | At each tick, 25% chance of one to four arrivals |
| few-windows | Eight windows from 8 to 1,024 (powers of two), crossed with consecutive integer ranks from 1 through N/8 | Same arrivals as many-windows |
| bursty | Same guard inventory as many-windows | A batch of 128–383 every 128 ticks, zero arrivals otherwise |
| duplicates | Repeated copies of 32 distinct guards, four ranks crossed with eight windows | Same arrivals as many-windows |

In the widest inventories, many windows exceed the measurement horizon and many ranks exceed the total observed arrivals. These guards are realistic only for deployments with such an inventory. The report includes the opposite orientation and duplicate-heavy cases so the crossover remains visible. See [the recorded-request replay](../replay/README.md) for separate workload assumptions.

## Results

[Raw measurements](synthetic.jsonl) contain the full distribution-level summaries for each run. The table below reports the median per-run mean microseconds per observed tick, including every emitted guard ID. The result depends on both the arrival sequence and the inventory definition.

| Guards | Workload | KthWake (µs) | By k, cached (µs) | By k, snapshot (µs) | By window, cached (µs) | By window, snapshot (µs) |
|---:|---|---:|---:|---:|---:|---:|
| 10,000 | many-windows | 0.138 | 0.366 | 0.268 | 4.121 | 3.866 |
| 10,000 | few-windows | 0.177 | 1.738 | 1.402 | 0.218 | 0.209 |
| 10,000 | bursty | 0.096 | 0.156 | 0.143 | 3.635 | 3.387 |
| 10,000 | duplicates | 0.064 | 0.058 | 0.060 | 0.103 | 0.084 |
| 100,000 | many-windows | 0.144 | 0.347 | 0.304 | 44.325 | 30.350 |
| 100,000 | few-windows | 0.186 | 1.875 | 1.443 | 0.234 | 0.231 |
| 100,000 | bursty | 0.136 | 0.181 | 0.227 | 43.276 | 39.904 |
| 100,000 | duplicates | 0.075 | 0.073 | 0.074 | 0.109 | 0.100 |
| 1,000,000 | many-windows | 0.276 | 0.869 | 0.682 | 1198.919 | 825.966 |
| 1,000,000 | few-windows | 0.188 | 1.874 | 1.383 | 0.253 | 0.247 |
| 1,000,000 | bursty | 0.244 | 0.586 | 0.505 | 1115.185 | 792.548 |
| 1,000,000 | duplicates | 0.213 | 0.420 | 0.381 | 0.510 | 0.409 |
