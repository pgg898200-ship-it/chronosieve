# Guardshift

Wake a worker when its conditions change. Feed Guardshift one batch of live signal updates and get the sorted IDs of guards that entered or exited. A queue crossing its dispatch threshold while the last worker goes offline produces zero spurious wakeups.

**153 ns per four-signal patch across 8,192 guards** on one AMD EPYC core in the sparse synthetic benchmark. The independent sorted-endpoint implementation took 286 ns. On an Azure-derived replay, Guardshift took 1.29 µs to return about 117 transitions per patch. A dedicated bitmap implementation took 1.24 µs. In a separate head-to-head using four adapters for the published SuperIntervals C++ library, Guardshift was 2.23× faster for sparse four-field patches at 8,192 guards and 1.68× faster at 32,768. SuperIntervals won the heterogeneous million-guard jump test. [External benchmark](bench/external.md) Both results are native in-process times.

[Try the guard explorer](https://guardshift.tancurl3.chatgpt.site/) · [Download the standalone source](https://huggingface.co/spaces/pgg-engineering/guardshift/resolve/main/packages/guardshift-0.1.0.zip) · [Benchmark and raw results](bench/README.md) · [Replay your own trace](bench/REPLAY.md)

## Try it

**Linux x86-64** with glibc 2.24+ and a compatible C++ runtime can install the 26 KB stable-ABI wheel with no compiler. The same binary was checked against a Python oracle on CPython 3.12 and 3.13. The wheel targets standard CPython 3.9+.

```sh
python -m pip install https://huggingface.co/spaces/pgg-engineering/guardshift/resolve/main/packages/guardshift-0.1.0-cp39-abi3-manylinux_2_24_x86_64.whl
```

**From source**, use a C++17 compiler and CPython development headers. From outside the checkout, use `python -m pip install https://huggingface.co/spaces/pgg-engineering/guardshift/resolve/main/packages/guardshift-0.1.0.zip`. The Python wrapper uses the same native implementation as the C++ API and has zero runtime package dependencies.

```sh
python -m pip install .
python -m guardshift < examples/fleet.jsonl
```

The four output lines are `[]`, `[[0,true]]`, `[]`, and `[[0,false]]`. The first input line defines the initial signals and guards. Each later line is one ordered JSON array of signal ID and value pairs. The process flushes one output line per patch. Invalid input names the line on stderr and exits with status 2.

```python
from guardshift import Index

# Queue depth in [100,1000] AND GPU utilization in [0,70].
idx = Index([95, 60], [[(0, 100, 1000), (1, 0, 70)]])
assert idx.apply([(0, 105), (1, 85)]) == []
assert idx.apply([(1, 65)]) == [(0, True)]
assert idx.active(0)
```

C++ users can copy `include/guardshift.hpp` and its MIT license into a project. [Edit the dispatch rule in Compiler Explorer](https://godbolt.org/z/d5jYnjzKW) and see how queue depth and worker utilization change it across three batches.

```sh
c++ -O3 -std=c++17 -Iinclude examples/dispatch.cpp -o /tmp/dispatch
/tmp/dispatch
```

[Async worker dispatch](examples/dispatch.py) · [Dispatch on a ChronoSieve alarm and available capacity](examples/chronosieve_routing.py)

## When to use it

Use Guardshift inside the process that owns an ordered stream of integer snapshots. It fits scheduling gates, capacity alerts, replica placement conditions and telemetry-driven agent wakeups. It is especially useful when several signals move together and few gates change per patch. An index is single-writer. Partition independent guard sets across owners or synchronize access to shared instances.

Each guard is a conjunction of closed intervals over signed 64-bit integers. Give a signal at most one interval per guard. Signal and guard IDs are zero-based. An empty guard starts true. Construction records initial truth without emitting events. Duplicate writes within a patch take their last value. Invalid patches are rejected before updating the index. Changes describe the net transition after the entire patch. A leap from below an interval to above it emits nothing if both endpoints are outside. Keep explicit intermediate samples if your application must react to a path between observations.

Rule definitions are static for an index lifetime. Rule updates require rebuilding. Transport ordering, deduplication of nonconsecutive messages, durable delivery, worker cancellation and statistical change detection belong to the caller. In Python an allocation failure while converting a successful native result into Python objects can occur after commit. Read back `active()` to reconcile the caller's state.

## How it works

For each field the compiler sorts interval endpoints. During small moves a cursor walks the crossed boundaries, adjusts missing-term counts and remembers guards whose truth could change. The final check emits sorted deltas. For large jumps a tree over blocks excludes intervals skipped completely. When at most four fields share a small dictionary of cutpoints, the compiler can materialize compatible-guard bitsets between thresholds. It intersects the selected rows and enumerates changed bits. The precomputed rows have an 8 MiB cap. Total index memory also includes state, thresholds and scratch space.

Dense jumps can favor a simpler endpoint counter. Uniformly narrow unary intervals favor a bounded-width point index. A million-guard deployment should include index construction and a representative replay in its evaluation. The benchmark ships those cases alongside the favorable ones. Bitmaps and range indexing have extensive published prior art. [BOP](https://ksiresearch.org/seke/seke23paper/paper142.pdf) · [PUBSUB](https://www.cise.ufl.edu/~sahni/papers/pubsubISCC.pdf)

## Verify

```sh
make test
make quick
```

The benchmark checks the IDs, order, resulting truth and final state against full recomputation. Raw JSONL includes the seed, build cost, mean, p50, p99 and output checksum. The Azure signals come from a public production trace of inference requests. The guard definitions are synthetic. A replay times native dispatch only. [Benchmark methods](bench/README.md) · [Azure provenance](replay/README.md)

To report a crossover, include CPU, compiler, guard count, fields per patch, repeat count and the anonymized aggregate results. Private trace contents can stay on your machine. [Report a result or integration](https://github.com/pgg898200-ship-it/chronosieve/issues/3)

## License

Original software and documentation use the [MIT license](LICENSE). The derived Azure fixture uses [CC BY 4.0](replay/README.md). The optional external comparator retains its upstream license.
