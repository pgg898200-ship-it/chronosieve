# Azure code-inference replay

This fixture measures how quickly a rule engine can process updates to two token-volume signals. A scheduler might use their conjunction to switch policies as the workload changes. The source is Microsoft's public trace of Azure LLM code-inference requests. The workload contains real recorded token counts. Guard definitions and the choice of metrics are part of the benchmark.

## What the CSV contains

Take the first 4,096 requests in source order. For each request, sum context tokens and generated tokens separately over the last 64 requests, including the current request. Use the available prefix for the first 63 requests. Each request occupies two rows in the following order.

| Field | Value | Observed range |
| --- | --- | --- |
| 0 | Rolling context-token total | 4,808 to 180,021 |
| 1 | Rolling generated-token total | 10 to 3,994 |

The supplied benchmark groups the two rows for each request into one atomic update, so each guard sees both new totals together. Its first 128 request pairs calibrate the synthetic guard distribution. The trace includes the first-to-last timestamps 18:17:03.9799600 and 18:39:59.9342470 on November 16, 2023. The replay preserves order and drops timestamps. Its throughput measures an in-memory update loop running at full speed. Generated-token counts describe completed outputs, so these features are suitable for retrospective load monitoring. The source has no guard definitions, agent identities, or observation time for completed output counts.

## Reproduce

From the repository root with Python 3, run the command below. The script streams the pinned source prefix. To work offline, append `--source /path/to/AzureLLMInferenceTrace_code.csv` to the Python invocation.

```sh
python3 replay/prepare_azure.py > /tmp/guardshift-azure-code.csv
cmp /tmp/guardshift-azure-code.csv replay/azure-code.csv
```

The output contains 8,192 updates plus a header, with SHA-256 `53c7f46c83f2b6613bd9872df7ddb5000f2cd236ab60560534b7bd0dce1c43a0`. The importer verifies the first 4,097 input lines, including the header, after normalizing line endings to LF. Their SHA-256 is `00eac94f2e9cfadd042700d3fff36964013c686059bc171e82afa537ebaf5dda`.

## Attribution and terms

Microsoft Azure. *Azure LLM inference trace 2023*, code service, from revision `5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e` of AzurePublicDataset. The source data and the derived fixture `azure-code.csv` are provided under Creative Commons Attribution 4.0 International. Guardshift selected the prefix, computed rolling sums, removed timestamps and reformatted the values. [Source data](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/data/AzureLLMInferenceTrace_code.csv) · [Dataset description](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/AzureLLMInferenceDataset2023.md) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki and Ricardo Bianchini. “Splitwise: Efficient generative LLM inference using phase splitting.” ISCA 2024. [Paper](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/)
