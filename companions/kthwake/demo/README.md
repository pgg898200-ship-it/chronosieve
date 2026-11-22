---
title: KthWake
emoji: ⏱️
colorFrom: green
colorTo: gray
license: apache-2.0
tags:
  - cpp
  - stream-processing
  - time-series
  - multi-agent
sdk: static
app_file: index.html
short_description: Explore exact k-of-window transitions and idle deadlines
---

# KthWake

Give an agent scheduler one ordered stream of timestamped arrivals. KthWake identifies the rules that just became true or false and returns an exact idle deadline. The native library is one C++17 header. The interactive Space illustrates its semantics in JavaScript.

A rule can mean “three tool results in 20 milliseconds” or “at least 1,000 tokens charged in the last second.” Subscribers retain individual IDs even when their rules match. One host timer can deliver all idle expirations on the stream. The source includes a JSONL CLI, a CPython binding, an asyncio example, and the production-trace benchmarks.

## Install the Python binding

With a C++17 compiler, CPython, wheel and setuptools:

```sh
python -m pip install --no-build-isolation https://huggingface.co/spaces/pgg-engineering/kthwake/resolve/main/kthwake-0.1.0.tar.gz
```

```python
from kthwake import Index
ix = Index([(3, 20)])
assert ix.step(100, 2) == []
assert ix.step(105, 1) == [(0, True)]
assert ix.next_expiry() == 120
assert ix.step(120) == [(0, False)]
```

## Benchmarks and contributions

The runnable release includes recorded arrivals from Mooncake and an Azure code-serving holdout, with three modeled policy registries per trace. It also includes million-subscriber synthetic cases, a timestamp-aware correctness oracle and competing native implementations. Evaluations distinguish observation-time snapshots from every timer-driven crossing. On the Azure individual-rate workload, KthWake delivered 330,890 timestamped transitions in 37.1 ms. The fastest of three native heap alternatives took 75.0 ms. The heap won the other two Azure timer workloads.

[Performance evidence](https://huggingface.co/spaces/pgg-engineering/kthwake/blob/main/EVIDENCE.md) · [Source files](https://huggingface.co/spaces/pgg-engineering/kthwake/tree/main) · [Independent benchmark instructions](https://huggingface.co/spaces/pgg-engineering/kthwake/blob/main/REPRODUCE.md) · [Community](https://huggingface.co/spaces/pgg-engineering/kthwake/discussions)
