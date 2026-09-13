# FlashInfer serving integration diagnostic — 2026-09-13

**Experimental, not promoted.** The native FlashInfer v2 path now runs through the Rust HTTP server via `--graph-numerics flashinfer-smol-experimental-v2`. It requires graph policy `require`, accepts only loopback IP socket addresses, and logs `quality=unqualified`. Existing V7/default behavior is unchanged. Rust → C/C++ ABI → CUDA; the Python files here are offline benchmark/export tools only.

## Conditions

SmolLM2-135M BF16; one RTX 4090/SM89, driver 580.173.02, 450 W limit, GUI retained, Blender left stopped. Same weight/tokenizer hashes checked against the archived launch plan. Natural prompts have 16/128/398 tokens and generate 32/64/128 tokens respectively. C32 closed-loop streaming, active capacity32, token budget512, context1024, greedy sampling, vLLM prefix caching disabled. Riley chunk512/KV2048 blocks; vLLM memory utilization0.3. Allocated memory budgets are engine-specific and are not a matched-memory efficiency claim. vLLM 0.27.1 uses FlashAttention2 and CUDA graphs; torch2.13.0, FlashInfer headers0.6.16.post3. Every argv is retained in launch JSON.

Previous/new Riley use **the same release binary**, SHA256 `189d05728de55b3e54d660ccff89730add89adcc45f38ddc4f74070e1bd3d5cb`, and differ only in explicit graph numerical mode. [Source hashes](flashinfer-serving-source-hashes.json) cover439 source/build files; [remote verification](flashinfer-serving-source-verification.json) has zero mismatches. Base commit is `fae5322f`; the integration changes accompany this evidence. Build log and dynamic dependencies are retained. No Python runtime dependency was added.

The initial screen used96 warmup and384 retained requests per process. Its vLLM throughput varied10,101–11,577 tokens/s between orders, so an extended screen increased warmup to384 and retained requests to1,536 **for all three engines**. Both screens are retained; no unfavorable trial was discarded. Each screen uses baseline → FlashInfer → vLLM, then reversed order. Initialization/build/export are outside measured runs. No process restart occurs between warmup and retained phases.

## Extended comparison

Throughput is summed output/request count divided by summed active measurement wall time, excluding process startup and gaps between runs. Latencies below are P50/P95/P99 from **3,072 pooled retained requests per engine**, using median and nearest-rank tails; these are empirical statistics, not long-duration stability qualification. The summarizer independently reconciles timing values against recorded token arrivals, full counts, and completion records.

| Metric | Previous Riley V7 | Experimental FlashInfer v2 | vLLM |
| --- | ---: | ---: | ---: |
| Output tokens/s | 10,281.82 | 11,026.11 | 11,649.97 |
| Requests/s | 137.70 | 147.67 | 156.03 |
| TTFT P50 / P95 / P99 (ms) | 10.009 / 15.511 / 45.768 | 9.365 / 16.102 / 50.856 | 22.845 / 41.865 / 61.203 |
| TPOT P50 / P95 / P99 (ms) | 3.000 / 3.119 / 3.171 | 2.790 / 2.886 / 2.926 | 2.368 / 2.856 / 3.283 |
| E2E P50 / P95 / P99 (ms) | 198.242 / 402.333 / 408.446 | 184.200 / 373.483 / 378.386 | 173.841 / 359.866 / 394.112 |
| Failed retained requests | 0 / 3,072 | 0 / 3,072 | 0 / 3,072 |
| Exact match to frozen Riley reference | 3,072 / 3,072 | 3,072 / 3,072 | 2,254 / 3,072 |
| Peak GPU memory | Not measured | Not measured | Not measured |
| General numerical quality | Existing contract | **FAIL; unaccepted** | Not established by Riley-reference agreement |

| Candidate change | vs Previous Riley | vs vLLM |
| --- | ---: | ---: |
| Output throughput | +7.24% | -5.36% |
| TTFT P50 | -6.43% | -59.01% |
| TPOT P50 | -7.00% | +17.82% |
| TTFT P99 | +11.12% | -16.91% |
| E2E P99 | -7.36% | -3.99% |

Per-order throughput ranges: V7 10,231.97–10,332.17, FlashInfer 10,932.22–11,121.62, vLLM 11,527.88–11,774.67 tokens/s. Both extended orders show FlashInfer ahead of V7 and behind vLLM. Two independent process repetitions do not support a strong confidence interval; no winner/stability claim is made from pooled request counts.

## Correctness and decision

The two screens total14,400 requests including warmups, with no request transport failures. Output lengths are equal across engines. Agreement on these three repeated natural prompts does **not** resolve the [free-generation mismatch](../20260913-flashinfer-free-generation/README.md) or [failed predeclared natural KL screen](../20260913-flashinfer-natural-screen/README.md). vLLM disagreement with the frozen Riley reference is an arithmetic observation, not a vLLM quality failure.

FlashInfer improves measured V7 serving throughput, but remains below vLLM and has longer median TPOT. Candidate TTFT P99 also regresses relative to V7. Keep the experimental mode unaccepted; do not enable it by default or relax numerical criteria. The goal is not achieved. Attention replacement alone, in this shape/backend, is insufficient. Next work should pursue a larger execution-area batch from the research roadmap (mixed prefill/decode policy or memory-aware FFN/layer execution), with unchanged exact V7 as the comparison anchor; FlashInfer precision/backend alternatives remain a separate unaccepted track. Re-profile an actual candidate only to choose between those concrete areas, not to restart isolated micro-tuning.

## Verification and artifacts

- Release CUDA/server build passed; all30 server CLI tests and3 numerical-profile configuration tests passed. Existing V7 full serving reference output checks passed in both screens.
- All server processes were stopped by the controller. vLLM logs include shutdown process-manager warnings and a resource-tracker semaphore warning; zero failed HTTP requests must not be read as proof of clean host resource lifetime. The final GPU snapshot reports no compute processes. Blender was not restored.
- [Extended machine-readable comparison](extended/comparison.json), [initial comparison](comparison.json), per-run launch/summary/accounting/exit JSON and logs are retained.
- `compact/` and `extended/compact/` retain every request's token IDs, token timestamps, numerical comparisons, transport state and metrics. Repeated frames/text/request bodies are replaced by canonical-JSON hashes. Full originals remain under `/tmp/riley-opt-260912/flashinfer-serving-screen{,-extended}-v1` on `ai-assistant`; raw-manifest JSON retains every original rows file SHA256 and size.
- Reproduction: `benchmarks/analysis/flashinfer_serving_screen.py ROOT FROZEN_BINARY NEW_OUTPUT_DIR 384 1536`, with the documented archived Round62 launch plan/fixtures/clients. Export only after terminal completion, then run `benchmarks/analysis/summarize_flashinfer_serving_screen.py EXPORTED_DIRECTORY`.

Rollback is selecting `--graph-numerics variable-smol-v7`; no accepted profile was replaced. Hopper/Blackwell/multi-GPU serving were not measured by this4090 diagnostic.
