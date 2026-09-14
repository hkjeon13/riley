# Wide verification with actual prefix reuse

Wide verification now accepts immutable, position-identical shared prefix pages.
The host validates the complete ownership ledger, including cache leases and
readers outside the current batch. Appends into shared writable pages remain
rejected by KV reservation and wire validation until COW completes. The serving
cache exports/imports only full immutable pages and leaves a private prompt tail,
so this integration does not need to copy a shared tail in its supported path.
The narrow verification session remains unsupported with shared prefixes.

Both GPU lanes warm the cache before admitting the same 32-request correctness
fixture (8 natural requests and 24 repetition controls), with the same model,
4090, prefix capacity, and generation limit. Each lane records **52 cache hits
and 6,640 reused prompt tokens**, including warmup reuse.

| Fixture | Serial calls | Wide calls | Accepted drafts | Output differences |
|---|---:|---:|---:|---:|
| C32 with warmed prefix cache | 39 | 33 | 629 | 0 / 1,024 |

Normal and full-model memcheck runs reproduce these outputs; memcheck reports
zero errors and both sessions close with zero CUDA allocations. This is a
correctness fixture, not a serving benchmark. Calls exclude cache warmup.
Artificial repetition controls do not establish representative performance.

Scheduler library: 71 tests pass, including shared-prefix rejection and deferred
cancellation settlement. Runtime wire suite and native ASan/UBSan packet probe
cover retained capabilities, missing owners, off-batch writers, position aliases,
and duplicate pages. A native compile failure from an incorrectly scoped check
is retained as build-v1; corrected build-v2 passes. No new selector kernel was
introduced; no new racecheck or Hopper/Blackwell runtime result is claimed.
All three Blender RPC processes were restored after GPU validation.

Serving integration remains pending: opt-in selection, multi-token detokenization,
stop-string/token truncation, deferred cancellation, commit-before-publication,
and acceptance/work counters. Existing text/audit publication requires staged
entries for every committed token. Do not bypass these contracts or equate this
GPU fixture with HTTP throughput. Then compare matched serving with current
Riley and vLLM, retaining the earlier 0.27.1 baseline separately from a newer lane.

```sh
python3 benchmarks/analysis/verify_speculative_shared_wide.py
```
