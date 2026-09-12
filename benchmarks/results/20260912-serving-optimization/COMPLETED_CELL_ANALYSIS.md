# Completed-cell descriptive analysis

`benchmarks/scripts/analyze_serving_token_optimization_v3.py` adds a separate
descriptive view for fully completed cells of a finalized campaign. A cell is one
predeclared setting and comparison, including **all five prescribed pairs**.
V1 and V2 remain unchanged. V3 calls frozen V2 first and preserves its complete
output under `campaign_analysis_v2`, including its campaign status and comparison
eligibility.

A cell is emitted only when:

- The plan declares an initial measurement, exactly five pairs and at least
  1,000 retained requests per process. Every process must contain exactly its
  declared count; Round14 declares 1,000.
- Every pair passes V2's current-source/reference/client proof checks, exact raw
  token/text/usage replay, warmup and retained accounting, timing/order checks,
  fresh process identity, saved summaries and rederived ratios.
- All five pair indices and alternating AB/BA order match the plan. Finalized
  pairs and launched lanes form the exact predeclared execution prefix.
- The campaign has finalization and verified restoration of the original three
  sessions, on their original ports and with the prepared private runtime.
  Every launched process, including a later failed lane, has a verified exit and
  no remaining owned process. Cleanup logs and restoration references are hashed.

Missing or unsuccessful repeats exclude the entire cell. Missing finalization,
restoration or any owned-process cleanup excludes every cell. An inconsistent
campaign completion claim also blocks this view. No later successful subset is
selected after an incomplete pair.

`completed_cells[].paired_ratios` describes the distribution of the five
per-pair right/left ratios. It does not pool request percentiles. Ratios whose
left latency is zero remain undefined under the frozen aggregator. Grouped SSE
IDs keep V2's actual shared delivery timestamps and zero-ITL semantics.

For a partial campaign, `whole_campaign_completed` remains false. V3 always
sets `overall_acceptance`, `performance_acceptance` and `performance_claim` to
false and `winner` to null. A completed cell does not establish numerical
equivalence beyond existing gates, broad concurrency scaling or P99 stability.

## Offline use

```sh
python3 benchmarks/scripts/analyze_serving_token_optimization_v3.py \
  --campaign /path/to/copied/token-serving-round14 \
  --path-map /tmp/riley-opt-260912=/path/to/copied/campaign-root \
  --output /path/to/new/completed-cells.json
```

Path maps locate copied evidence while retaining original referenced paths and
hashes. Output creation is exclusive. The analyzer performs no network, process,
GPU or model operation; restoration evidence describes the recorded restoration,
not a new live-process observation. Exit status2 means the additional cleanup or
qualification gate is unavailable.

## Frozen local validation

- V3 source SHA256: `aed03c52fd1fa9f4cbdcf360cd3cb0604a5e753a2f2bd3ab94c82a95fd81c3ba`
- Tests SHA256: `e860b8e805770560daa3a9cc9b84653cc30f07365f1e1a8b38971579806e37ac`
- Unchanged V2 SHA256: `776c1dcb45c2ba676ea9e795df32d46aa043cef26c25069a55945f9f9b5d7a61`

Seven CPU tests passed, including 10,000 saved synthetic retained responses
replayed through V2, copied-path validation, incomplete-cell exclusion, missing
finalization/restoration/exit, failed-lane cleanup, and order/scope mutations.
These are analyzer tests, not new GPU correctness or serving measurements.
