# KthWake · Sliding-window threshold scheduler

Wake a waiting task when three tool results arrive within 20 ms. If results stop, clear the signal at the exact expiry time. KthWake turns these rolling-count rules into callbacks for an event loop.

Place one index between each ordered stream of arrivals and your dispatcher. Pass in a timestamp and a count. Read out the IDs of rules that changed, their new states, and the next idle deadline. A single host timer can serve all rules on that stream.

[Try the interactive timeline](https://pgg-kthwake.tancurl3.chatgpt.site) · [Single C++17 header](include/kthwake.hpp) · [Python binding](#use-from-python) · [Benchmark results](EVIDENCE.md)

| Example time | Input to the index | Output for the three-in-20-ms rule |
| --- | --- | --- |
| 100 ms | Two results finish | Waiting |
| 105 ms | One more result finishes | Rule 0 turns on. The idle deadline is 120 ms. |
| 120 ms | Timer fires with zero arrivals | Rule 0 turns off. |

On a held-out Azure code-serving trace with 4,096 modeled individual-rate subscriptions, KthWake emitted 330,890 timestamped transitions in **37.1 ms**, against **75.0 ms** for the fastest of three native heap alternatives. A dedicated rank heap won the shared-rate and burst registries on the same trace.

## Run in a minute

With Make and a C++17 compiler, clone the release branch and run the bundled trace.

```sh
git clone --depth 1 --branch releases/kthwake-0.1.0 https://github.com/pgg898200-ship-it/chronosieve.git
cd chronosieve/companions/kthwake
make example
```

The example applies two arrivals at 100 ms and one at 105 ms. The three-in-20-ms rule turns on at 105 and turns off at 120. A separate five-ms activity rule stays on when the 105-ms batch replaces events expiring at that instant.

```sh
# GUARDS contains k,window. TICKS contains time,arrivals.
./build/kthwake --idle examples/guards.csv examples/ticks.csv
# {"t":100,"id":1,"active":true}
# {"t":105,"id":0,"active":true}
# {"t":110,"id":1,"active":false}
# {"t":120,"id":0,"active":false}
```

Omit `--idle` to compare snapshots only at supplied ticks. Pass `-` for the tick file to stream from standard input. The CLI writes JSONL, accepts integer CSV or whitespace-delimited inputs and reports malformed rows on stderr.

The Python [rolling budget example](examples/rolling_budget.py) shows how an ordered stream of weighted token charges crosses a rolling threshold and when it clears.

## Embed the header

Copy `include/kthwake.hpp`. There are no native dependencies beyond the C++17 standard library. A subscriber's ID is its zero-based position in the input.

```cpp
#include "kthwake.hpp"

kthwake::Index ix({{3, 20}, {1, 5}});
auto changed = [](std::uint32_t id, bool active) {
    // Queue the indicated agent's state change here.
    (void)id;
    (void)active;
};
ix.step(100, 2, changed);
ix.step(105, 1, changed);
auto next = ix.next_expiry();          // 110
if (next) ix.step(*next, 0, changed); // the five-ms rule turns off
```

## Use from Python

Install the source directory using CPython, setuptools, wheel, and a C++17 compiler. The package name below is a local build.

```sh
python3 -m pip install --no-build-isolation .
python3 examples/asyncio_scheduler.py
```

```python
from kthwake import Index
ix = Index([(3, 20), (1, 5)])
assert ix.step(100, 2) == [(1, True)]
assert ix.step(105, 1) == [(0, True)]
assert ix.next_expiry() == 110
```

The asyncio example arms one timer per stream, reuses it while its deadline holds, and handles event-loop delays by emitting intervening transitions at their logical times. The Python binding materializes tuples for the changed subscribers. The native benchmark measures C++ callbacks.

The [prebuilt Linux x86-64 CLI](https://pgg-kthwake.tancurl3.chatgpt.site/kthwake-linux-x86_64) has [published SHA-256 checksums](https://pgg-kthwake.tancurl3.chatgpt.site/SHA256SUMS). It requires glibc ≥2.34 and libstdc++ with GLIBCXX_3.4.32. The source build uses the host toolchain.

## Time and delivery contract

- A rule `(k, W)` is true when at least `k` arrivals lie in `(t-W, t]`. Use any consistent integer time unit.
- `step(t, n, cb)` atomically adds `n` arrivals and evaluates all rules at that timestamp. A jump in time yields the net change since the prior evaluation. Combine simultaneous events into one batch when a single snapshot is required. Equal-time calls create distinct snapshots.
- For a complete logical timeline, apply each idle deadline strictly *before* the next arrival, then apply the arrival. An expiry equal to an arrival belongs to the same snapshot. Drain remaining deadlines if the stream ends.
- `next_expiry()` is the first representable tick at which a currently true guard falls if arrivals stop. It returns empty when no such tick exists. The maximum unsigned 64-bit tick is valid.
- Guard registration is static. Each index is single-threaded. Callbacks are synchronous and must not reenter the index. Group/order your output explicitly if consumers require a global ordering. Different guards can have identical rules.
- Positive `uint32_t` thresholds and positive `uint64_t` windows are supported, as are `uint64_t` arrival multiplicities. Timestamps must be nondecreasing. An upstream watermark is needed to order late network events.
- The index computes local state changes. Applications supply delivery, persistence, synchronization between shards and the clock.

## Reproduce

```sh
make test
python3 replay/prepare.py --out /tmp/kthwake-replay
python3 replay/verify.py ./build/kthwake /tmp/kthwake-replay
python3 replay/measure.py --inputs /tmp/kthwake-replay --out /tmp/kthwake-results.json
python3 bench/run.py --out /tmp/kthwake-synthetic.jsonl
```

Bundled inputs support offline replay. The benchmark verifies emitted transitions before timing; raw per-run measurements and unfavorable cases ship with the release. See [EVIDENCE.md](EVIDENCE.md) for the measured workload and boundaries and [PROGRAM.md](PROGRAM.md) for the research and adoption criteria.

## License

Apache-2.0. The bundled trace files carry their upstream attribution and licenses in `replay/`.
