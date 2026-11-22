# Replay production LLM request arrivals

The first replay includes all **23,608 inference request arrivals** from Mooncake's FAST'25 tool-and-agent serving trace. The 1,180 distinct relative timestamps and their multiplicities come from the published trace. A request here means an inference request in Mooncake's tool-and-agent workload. The trace does not identify individual tool calls, coding tasks, subscriptions, or alerts.

KthWake observes one stream at each recorded arrival timestamp, then receives one closing tick 60 seconds after the final arrival. Arrivals that share a timestamp enter atomically. The benchmark measures in-process snapshot transitions for modeled subscriptions. Network transport, callback I/O, and delays between observations are outside its timings.

## Reproduce from the repository root

```sh
make
python3 replay/prepare.py --out /tmp/kthwake-mooncake
cmp replay/oracle.json /tmp/kthwake-mooncake/oracle.json
python3 replay/verify.py ./build/kthwake /tmp/kthwake-mooncake
python3 replay/measure.py --inputs /tmp/kthwake-mooncake --out /tmp/kthwake-timings.json --native
```

The bundled 12.6 kB tick file supports offline reproduction. Pass `--fetch` to `prepare.py` to download, hash, and convert the 4.2 MiB source from its pinned commit. Pass `--source PATH` to use a local raw file. All three routes produce byte-identical ticks, guards and oracle manifests with the default settings. For an explicit polling cadence, `--heartbeat-ms 100` inserts a zero-arrival tick at each multiple of 100 ms without an arrival. This changes the number and timing of observable edges and should be reported as a separate workload.

## Modeled subscriptions

Each profile registers 4,096 subscribers before replay. A guard is active when at least `k` events lie in `(now − window, now]`. Each subscriber receives its own rising and falling transitions. Shared rule definitions keep distinct subscriber identities.

| Profile | Rules | Distinct k | Distinct windows | Edge transitions |
| --- | ---: | ---: | ---: | ---: |
| Shared rate policies | 64 | 42 | 8 | 404,864 |
| Individual rate policies | 4,096 | 649 | 4,096 | 179,976 |
| Individual burst windows | 4,096 | 8 | 4,096 | 46,662 |

Shared policies cross eight round windows from 1 to 60 seconds with target rates of 2, 4, 5, 6, 7, 8, 10 and 12 requests/s and round thresholds up. Individual rate policies use the same rate choices and different millisecond windows. Individual burst windows use fixed cardinalities 4, 8, …, 512 and the same individual windows. The assignments are deterministic and independent of measured kernel performance. These are designed workloads. The operator who collected the trace supplied no alert definitions.

The timing harness rotates method order across repetitions. Each native tick executes a callback that updates a digest. Construction time is recorded separately, and elapsed replay time includes the per-tick timer reads. Pass `--cpu N` to `measure.py` to pin a Linux run to one allowed logical CPU. The output stores every repetition alongside the CPU, compiler flags and source hashes.

The Python oracle counts arrivals independently using prefix sums and binary search on observation times. It hashes UTF-8 rows `<time> <id> <0 or 1>\n`, ordered by time then subscriber ID. The verified digest covers every transition, including its direction. Expected hashes and source metadata are in [oracle.json](oracle.json). The native benchmark separately compares all four optimized C++ baselines' full per-tick edge lists before timing them.

## Attribution

The Mooncake team published these anonymized timestamps with **Mooncake: Trading More Storage for Less Computation — A (Mostly) KVCache-Centric Architecture for LLM Serving**, USENIX FAST 2025. KthWake grouped equal millisecond timestamps, counted rows, removed token lengths and remapped block hashes, and added a zero-count closing tick. The upstream repository supplies the Apache License 2.0, included in [LICENSE-Mooncake](LICENSE-Mooncake).

[Source data at the pinned commit](https://github.com/kvcache-ai/Mooncake/blob/e94a0b86ba067455d8b0524eb2cbb5fbac2db024/FAST25-release/traces/toolagent_trace.jsonl) · [Trace description](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md) · [Upstream license](https://github.com/kvcache-ai/Mooncake/blob/e94a0b86ba067455d8b0524eb2cbb5fbac2db024/LICENSE-APACHE)

## Independent Azure code-service holdout

The second replay uses the **complete 8,819-row Azure 2023 code-service inference trace**, collected independently by Microsoft. We chose the full file and fixed all three guard sets before scoring this holdout. The file records code inference requests and token lengths. It supplies neither per-agent tool invocations nor alert policies.

The converter preserves all seven fractional timestamp digits before subtracting the earliest observation, floors elapsed times to whole milliseconds, groups ties, discards token counts, and adds the same 60-second closing observation. This preserves chronology at the guard clock's resolution. Subscriber configurations are byte-identical across providers.

```sh
python3 replay/prepare.py --trace azure-code --out /tmp/kthwake-azure
cmp replay/azure-code-oracle.json /tmp/kthwake-azure/oracle.json
python3 replay/verify.py ./build/kthwake /tmp/kthwake-azure
python3 replay/measure.py --inputs /tmp/kthwake-azure --out /tmp/kthwake-azure-timings.json --native
```

Add `--fetch` to refetch and verify the source SHA-256. The Azure tick file is adapted from Microsoft Azure's data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The license text is included in [LICENSE-Azure](LICENSE-Azure). Cite Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini, *Splitwise: Efficient generative LLM inference using phase splitting*, ISCA 2024. [Pinned data](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/data/AzureLLMInferenceTrace_code.csv) · [Microsoft's trace description and attribution](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/AzureLLMInferenceDataset2023.md)

## Recorded snapshot results

Single-threaded replay on one AMD EPYC 9V74 logical CPU, GCC 14.2, `-O3 -DNDEBUG -march=native`. Each cell is the median elapsed milliseconds for the complete trace over ten rotated-order repetitions. CLI validation against the independent count-window oracle covered all **1,712,462** timestamped transitions across the six cases. The two “snapshot” controls omit deadline bookkeeping and only observe at supplied ticks. The other two optimized references also maintain a next-expiry hint.

| Trace | Guard profile | KthWake (ms) | Group by k (ms) | Group by window (ms) | Rank snapshot (ms) | Window snapshot (ms) | KthWake fastest, paired runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mooncake | Shared | 0.324 | 0.478 | 0.359 | 0.492 | 0.344 | 6 / 10 |
| Mooncake | Varied | 3.788 | 3.416 | 19.466 | 3.015 | 16.306 | 1 / 10 |
| Mooncake | Burst | 0.155 | 0.156 | 15.400 | 0.149 | 13.018 | 2 / 10 |
| Azure code holdout | Shared | 1.416 | 1.723 | 1.297 | 1.442 | 1.192 | 0 / 10 |
| Azure code holdout | Varied | 17.357 | 13.698 | 130.254 | 12.365 | 131.061 | 0 / 10 |
| Azure code holdout | Burst | 1.198 | 1.506 | 132.678 | 1.473 | 131.068 | 9 / 10 |

On the independent Azure trace, KthWake used 19% less elapsed time than the fastest reference for snapshots of burst subscriptions. The snapshot-only references led on individual rate and shared subscriptions. These results apply to the supplied rule sets and four native controls on one host. Source hashes, per-repetition timings and latency percentiles appear in the [Mooncake measurements](mooncake-results.json) and [Azure measurements](azure-code-results.json).

### Exact idle deadlines on Azure

The timer harness emits every edge at its exact event-time deadline between arrivals, combines simultaneous expiries and arrivals, and drains remaining deadlines after the trace. This changes the output contract and the number of transitions. Before timing, all four engines compare complete ordered lists of timestamp, subscriber ID, and direction. The table shows median total milliseconds from 12 rotating repetitions of the same compiler settings on CPU 4, including deadline queries and digest callbacks, excluding index construction.

| Azure policy | Timestamped transitions | Idle wake-ups | KthWake (ms) | Pair heap (ms) | Rank heap (ms) | Single-sift rank heap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared | 880,384 | 6,568 | 7.669 | 8.438 | 6.200 | 6.055 |
| Varied | 330,890 | 156,230 | 37.112 | 446.982 | 83.096 | 75.008 |
| Burst | 241,740 | 118,822 | 5.388 | 457.810 | 4.729 | 3.311 |

For individual rate policies, KthWake completed the full timeline in 37.1 ms versus 75.0 ms for the fastest heap. For burst and shared policies, the single-sift rank heap completed the replay faster. Raw output is provided for [shared](azure-timers-shared.csv), [varied](azure-timers-varied.csv) and [burst](azure-timers-burst.csv) policies.

```sh
c++ -O3 -DNDEBUG -march=native -std=c++17 -Iinclude timerbench/timers.cpp -o /tmp/kthwake-timers
for shape in shared varied burst; do
  /tmp/kthwake-timers "/tmp/kthwake-azure/$shape.guards" /tmp/kthwake-azure/ticks.txt 12
done
```

The timer benchmark source SHA-256 is `7950ead85613d44471e857ff4fa58244c9ca7aa52e0a87b9644da447769503f1`. The release header SHA-256 is `cbe5699dc92016fe49bd0d003836ef6d7916d11bd07cbc746874d6ad5109287b`.
