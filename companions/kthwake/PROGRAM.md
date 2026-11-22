# Research program

## Why spend the first release on window triggers?

| Candidate | Product hypothesis | Experiment or reason to defer |
| --- | --- | --- |
| Temporal count triggers on a shared event stream | Agents with different windows can receive exact edge-triggered callbacks and share a single idle timer. | Chosen. The result has a simple oracle, fits in one header, and supports an in-process event loop. Compare against indexed native algorithms, including the full cost of idle expiry. |
| Changed-input fan-out in a graph executor | Intern channel names, union posting lists and return affected actors in deterministic order. | LangGraph already indexes triggers. An adequate comparison needs the cost of converting Python names, materializing ordered IDs and projecting them back into real tasks. A fast bitmap microbenchmark would not settle this. |
| Early finish for ranked multi-agent retrieval | Cancel outstanding retrieval pages once a bound proves the top-k fused set cannot change. | Classical threshold algorithms already give bounds. First demonstrate that an actual retriever supports backpressure and avoids upstream work. Saving rank-fusion CPU alone has a weaker value proposition. |

[LangGraph scheduler](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_algo.py) · [Fagin, Lotem and Naor on top-k aggregation](https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware)

Multiple-window burst detection and shared stream aggregation have a long history. KthWake's engineering question is narrower: how cheaply can an embedded scheduler deliver exact per-subscriber state changes *and* expose the exact next idle deadline? The matched baselines implement that work directly. We have not benchmarked a general-purpose streaming cluster under equivalent service guarantees.

[Bulut and Singh on burst monitoring](https://cs.ucsb.edu/sites/default/files/documents/2003-24.pdf) · [IBM sliding-window reference implementations](https://github.com/IBM/sliding-window-aggregators)

## What could be defensible

The header is published under Apache-2.0. Competitors can reproduce the technique and reuse the code. A defensible advantage would require information that improves engineering decisions across deployments: anonymized policy shapes with permission to redistribute, exact-timer workload recordings, independent hardware reruns, and maintained integrations with agency schedulers. The release starts that corpus with public traces and correctness oracles. The rules on those traces are modeled. Actual operator policy configurations and a first independent integration remain the highest-value missing evidence.

## What counts as traction

Record the first external repository that imports the header or binding and executes it. Record a third-party benchmark with the workload and machine attached. Track stars, forks, downloads and views independently where exposed. Exclude the publisher, its workers, validation downloads, starter forks and mirrors from evidence of integration. Keep links, observation dates and deduplication by repository. Horizon reported 158 downstream repositories across its portfolio in seven weeks; this is an ambitious external reference for a single product.

Public evaluation material is the same for human and automated readers. It includes a one-command run, supported/unsupported workloads, executable counterexamples, license, and raw results. An agent citation or crawler request counts separately from a deployment. Revisit the product choice if external traffic fails to convert or a competing implementation erases the measured edge.

[Horizon Labs report](https://horizon-labs.ai/) · [Ramp's agent-discovery experiment](https://engineering.ramp.com/post/marketing-to-ai-agents)
