# Reproduce KthWake on another machine

The most valuable rerun uses an independently selected event stream, unchanged rules and a competing implementation. A failure or slower result is as informative as a win. To reproduce the included numbers, run the exact commands and flags in EVIDENCE.md first.

1. Record CPU, OS, compiler/version, flags and the release SHA-256. Pin a core if available. For exact-timer runs, append `--with-build` to record construction and replay separately for every method and repetition. See the [short-run example and raw CSV](timerbench/README.md#short-lived-scheduler-holdout-november-24-2026).
2. Describe the stream or share a publishable sequence of integer timestamps and arrival counts. Describe the number of subscribers, distinct thresholds and distinct windows. Keep operator telemetry private unless you have permission to redistribute it.
3. State whether you measured snapshots at arrivals or every idle deadline. Verify the ordered-by-time edge set including IDs and state before collecting timings.
4. Include each repetition, median elapsed time and p99 when available for each implementation. Report rebuild/setup cost, intermittent stalls and losses.
5. Post a public benchmark link or a reproducible issue in the [GitHub reproduction thread](https://github.com/pgg898200-ship-it/chronosieve/issues/2). An integration is easiest to verify when it links to executable source and records which work KthWake replaced.

First-party runs and the provided public input traces are development evidence. The project's organic adoption count starts with a link to the first independent integration.

## Publish the browser timeline

Open `demo/index.html` locally and select **Next callback**. The clock advances from 5 ms to 20 ms and the three-result rule clears. The release downloads point to the versioned files covered by the SHA-256 manifest.

From `companions/kthwake` in a release checkout, publish the timeline and its Space card together with an authenticated Hugging Face CLI.

```sh
hf upload pgg-engineering/kthwake demo . --repo-type space
```
