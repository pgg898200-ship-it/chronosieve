---
title: KthWake · Sliding-window threshold scheduler
emoji: ⏱️
colorFrom: yellow
colorTo: green
license: apache-2.0
tags:
  - cpp
  - stream-processing
  - time-series
  - multi-agent
sdk: static
app_file: index.html
short_description: Turn rolling event counts into callbacks and idle deadlines
---

# KthWake · Sliding-window threshold scheduler

Wake a waiting task when three results arrive in 20 ms. KthWake sits between an ordered stream of events and your scheduler. It receives timestamped arrival counts and emits the IDs and new states of rolling rules that changed. One host timer handles the next expiry even after traffic stops.

The browser timeline opens with two completions at 0 ms and one at 5 ms. Rule 0 turns on at 5 ms. Choose **Next callback** to see it turn off at 20 ms. Edit the times and rule to replay your own sample. The interactive timeline runs in JavaScript. The production index is a C++17 header with a CPython binding and JSONL CLI.

## Install the Python binding

With a C++17 compiler, CPython, wheel and setuptools:

```sh
python -m pip install --no-build-isolation 'git+https://github.com/pgg898200-ship-it/chronosieve.git@7cca93890a5bfa1e120f6aaabda8d1b8d7957b1f#subdirectory=companions/kthwake'
```

```python
from kthwake import Index
ix = Index([(3, 20)])
assert ix.step(100, 2) == []
assert ix.step(105, 1) == [(0, True)]
assert ix.next_expiry() == 120
assert ix.step(120) == [(0, False)]
```

The [release downloads](https://github.com/pgg898200-ship-it/chronosieve/tree/f6df7cdb54362fea22003a814ac83fcc7b179a8f/companions/kthwake/downloads/v0.1.0) include the source archive, header, CPython 3.12 Linux x86-64 wheel, native Linux x86-64 CLI and benchmark results. The file named `SHA256SUMS` lists the digest for each download.

## Benchmarks and contributions

The runnable release includes recorded arrivals from Mooncake and an Azure code-serving holdout, with three modeled policy registries per trace. It also includes million-subscriber synthetic cases, a timestamp-aware correctness oracle and competing native implementations. Evaluations distinguish observation-time snapshots from every timer-driven crossing. On the Azure individual-rate workload, KthWake delivered 330,890 timestamped transitions in 37.1 ms. The fastest of three native heap alternatives took 75.0 ms. The heap won the other two Azure timer workloads.

[Performance evidence](https://github.com/pgg898200-ship-it/chronosieve/blob/7cca93890a5bfa1e120f6aaabda8d1b8d7957b1f/companions/kthwake/EVIDENCE.md) · [Source files](https://github.com/pgg898200-ship-it/chronosieve/tree/7cca93890a5bfa1e120f6aaabda8d1b8d7957b1f/companions/kthwake) · [Independent benchmark instructions](https://github.com/pgg898200-ship-it/chronosieve/blob/7cca93890a5bfa1e120f6aaabda8d1b8d7957b1f/companions/kthwake/REPRODUCE.md) · [Community](https://huggingface.co/spaces/pgg-engineering/kthwake/discussions)
