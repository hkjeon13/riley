# Q16 compensated prefill: serving decision

The smaller query tile recovers part of the Q128 compensated-prefill regression, but still loses to existing V7 and vLLM. **Do not promote Q16.** Strict free-running equivalence also fails. This is a diagnostic implementation and a completed optimization-batch measurement, not completion of PR03 or the serving goal.

## Changes tested together

- Keep the previous FP32 softmax denominator and two BF16 probability-product accumulations. Q/K/V/O storage remains BF16.
- Compare CTA query tiles Q128, Q64, Q16 using 64, 128, 256 work slots respectively. Packed GQA query work uses `ceil(3*q/tile)` per request. Q16 raises the retained metadata allocation from 33,996 to 36,492 bytes in both native planning and the Rust owner. At 32 requests/1,024 total query rows the 256 slots cover per-request rounding.
- Q16 uses four KV warps. Add a CTA barrier after their reduction reads and before shared storage is reused for output; select the exact hash-verified overlay in a separate build directory.
- Give the isolated model copy a distinct graph identity and explicit diagnostic CLI. V7 and Q16 use one frozen binary; Q128 uses the previous frozen binary. Serving execution remains Rust → C/C++ → CUDA. Python is offline build/reference/measurement tooling only.

No production backend/default was changed. Residual-fragment lifetime redesign was not implemented in this batch. The generator reproduces all eight modified diagnostic source files byte-for-byte (`isolated-source-verification.json`). Some inherited diagnostic-copy comments still mention the original 33,996-byte allocation; actual allocation and native extent checks use 36,492.

## Correctness and failed attempts

Q32 compiled but upstream runtime configuration validation rejected its head64 geometry: `NUM_MMA_Q=2`, `NUM_WARPS_Q=1`, `NUM_WARPS_KV=4`. Upstream prunes this CTA32 configuration for head dimensions below512. This is a configuration failure, not a 4090 capability skip. Q64 replaced it in the three-scale screen.

Initial Q16 ordinary numerical tests and memcheck passed, but mixed racecheck found **24 errors**: output writes reused shared storage while KV-warp reduction reads remained in flight. The corrected CTA barrier gives **0 memcheck errors and 0 racecheck errors/warnings** on the original ten-replay mixed harness, including invalid metadata, one-token prefill, all-decode and recovery. The failing logs are retained. Pre-barrier model observations are not acceptance evidence.

Final Q128/Q64/Q16 synthetic runs use equal BF16 input values at Q/K scales0.125/1/4 and the unchanged maximum-error limit0.01. Each variant checks868,032 output values per scale; all pass with nearly identical FP64-reference RMSE. These are numerical tests, not performance measurements. Q64 was not integrated or measured in serving.

| Full-model measure | Existing V7 | Q16 residual |
|---|---:|---:|
| Natural NLL | 2.95928943 | 2.95714459 |
| KL(FP32 reference || engine) | 0.0007591519 | 0.0007462197 |
| FP32 argmax matches /256 | 244 | 250 |

The fixed eight-passage/256-target screen passes its existing relative conditions; natural logits equal the previous Q128 candidate bitwise (SHA256 `354490dbc3408cfde8e5e1a2676f4935f99998bcedaa7ba38dedff03c6288dc1`). This limited agreement does not cover other shapes or free generation. Independent generation differs from V7 in **16/32 sequences and232/1,024 token positions**, versus148 token differences for previous Q128. Repeated-prompt invariance passes. The strict assertion fails with exit101; general quality remains unaccepted.

Final whole-model native memcheck reports0 errors; both logit dumps match ordinary execution bitwise and the test reports zero remaining allocations. No full-model racecheck was run. Compiler resource logs are archived; there is no new active-kernel trace or serving kernel-time decomposition, so compiled variant resource figures are not asserted to explain performance.

## Matched serving comparison

RTX4090, same BF16 SmolLM2-135M checkpoint, vLLM0.27.1, C32 natural corpus with16/128/398 prompt tokens, budget/chunk512, context1024, GPU greedy, synchronous metadata and required graphs. The FFN pipeline is not selected. Exact launch arguments, input/reference hashes, telemetry and binary identities are archived. GPU clocks were not locked; GUI stays running and Blender stays stopped.

Orders: V7 → Q128 → Q16 → vLLM, then the reverse. Each process has192 warmup and768 retained requests: **7,680 total requests**,1,536 retained per engine and114,688 output tokens per engine. All eight transport runs complete with zero failures. Throughput uses summed tokens divided by summed retained wall time; P95/P99 use nearest rank over pooled retained requests, P50 uses the median.

| Metric | Existing V7 | Q128 residual | Q16 residual | vLLM |
|---|---:|---:|---:|---:|
| Output tokens/s ↑ | 10,464.6 | 9,592.5 | 10,113.7 | 11,850.6 |
| Requests/s ↑ | 140.151 | 128.471 | 135.452 | 158.713 |
| TTFT P50 (ms) ↓ | 10.385 | 11.857 | 10.995 | 18.473 |
| TTFT P95 (ms) ↓ | 17.229 | 19.241 | 17.045 | 43.330 |
| TTFT P99 (ms) ↓ | 66.941 | 71.649 | 64.779 | 68.491 |
| TPOT P50 (ms) ↓ | 2.917 | 3.186 | 3.018 | 2.343 |
| TPOT P95 (ms) ↓ | 3.020 | 3.297 | 3.137 | 2.803 |
| TPOT P99 (ms) ↓ | 3.074 | 3.356 | 3.179 | 3.250 |
| E2E P50 (ms) ↓ | 193.746 | 211.293 | 200.337 | 167.364 |
| E2E P95 (ms) ↓ | 391.228 | 426.240 | 406.133 | 354.112 |
| E2E P99 (ms) ↓ | 395.640 | 433.027 | 413.208 | 385.871 |

Q16 throughput changes: **+5.43% versus Q128, -3.35% versus V7, -14.66% versus vLLM**. Both run orders preserve this ranking. Q16 TTFT is better than vLLM, while median TPOT and E2E are worse. The objective is unmet.

| Retained evidence | V7 | Q128 | Q16 | vLLM |
|---|---:|---:|---:|---:|
| Exact frozen-reference matches /1,536 | 1,536 | 1,532 | 1,033 | 1,122 |
| Transport failures | 0 | 0 | 0 | 0 |
| Run0 tokens/s | 10,473.3 | 9,642.4 | 10,110.6 | 11,787.1 |
| Run1 tokens/s | 10,456.0 | 9,543.1 | 10,116.9 | 11,914.8 |

The Q16 serving reference-match loss is material and remains visible independently of transport success. The fixed teacher-forced corpus did not predict this free-running outcome. Matching output counts does not mean matching outputs. This is a short single-concurrency workload screen, not long-running stability, broad quality or peak-memory qualification. Peak GPU memory was not measured. No Hopper/Blackwell/multi-GPU runtime result is claimed.

## Decision and next area

Keep the existing backend. Smaller tiles recover some performance, but this compensation family has not demonstrated an acceptable quality/performance combination. Do not keep making isolated tile-size changes hoping to reach the serving target. Pause this candidate family and return to the architectural roadmap: PR07/PR19 persistent layer execution with dependency/scratch ownership and full-model invocation, reusing accepted numerical operations. Its feasibility and serving benefit remain to be measured; the roadmap does not imply an expected speedup. Broader numerical validation across prompt/chunk shapes is required before any attention backend promotion.

At the next meaningful integrated milestone, repeat the current Riley/new Riley/vLLM table with correctness and tail results. Intermediate build or primitive milestones alone do not trigger a new serving comparison report.

## Reproduction and artifacts

`prefill_tile_screen.py` generates three exact pinned-header variants. `prepare_prefill_residual_model.py --tile 16` creates the isolated model source. `prefill_q16_serving_screen.py ROOT Q16_BINARY OUT 192 768 Q128_BINARY` runs the four lanes. Full-model commands are in `run.sh` and `build-server.sh`; final sanitizer binary hash is in `memcheck-verification.json`.

- Q16/V7 frozen binary: `/tmp/riley-opt-260912/prefill-q16-serving-binary-v1/riley`, SHA256 `a2ac366e721d1092377df41804a3682b6de4740a340b420de6f3e2396088e949`.
- Q128 frozen binary: `/tmp/riley-opt-260912/prefill-residual-serving-binary-v1/riley`, SHA256 `230a372994253cce30a1005b6cf0e79688cb1dd3fdbcde5e526b45a8bc8ed86f`.
- Raw HTTP frames remain in `/tmp/riley-opt-260912/prefill-q16-serving-v1`; `serving/compact` retains timestamps, token IDs, accounting and raw hashes. Both comparisons reconcile per-request data independently.
- `final-verification.json` confirms unchanged binaries and no remaining GPU compute process. Blender was not restored.
- Failed race/configuration logs and corrected final logs are separate. An initial Q32 diagnostic lacked the driver library environment; it was corrected before the actual configuration failure was diagnosed. The first export used a nonexistent utility filename; export was rerun using the verified repository utility. Neither preparation error changes the measurements.
