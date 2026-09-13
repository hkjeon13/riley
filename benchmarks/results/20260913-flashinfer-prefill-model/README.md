# FlashInfer prefill-only full-model integration

The experimental Rust session now captures the 30-layer SmolLM2 model with FlashInfer **prefill only**. Existing V7 decode arithmetic remains selected in both pure and mixed stages. Build and execution succeed, but strict greedy equivalence and the predeclared natural-language numerical screen **fail**. This profile is not accepted or selected by default; no serving performance claim is made.

## Implementation batch

1. A separate native recorder validates and retains an exactly 33,996-byte device workspace, including context, retained parent, mutable/weight alias and extent checks. The workspace survives both graphs and is released with the owned session. Optional-build absence fails rather than selecting a fallback.
2. The mixed model uses the previously verified prefill-only metadata planner. The old mapped attention kernel skips prefill rows and continues computing decode rows. FlashInfer fills the complementary prefill outputs in the same packed buffer. Pure decode uses the original grouped V7 path. No Q/K/V copies or Python runtime calls are introduced.
3. `into_owned_variable_flashinfer_prefill_session` creates the independent `riley.experimental.flashinfer-0.6.16.post3.prefill-only.v1` identity, binding the prefill source and verified overlay helper. Existing FlashInfer decode and FFN experiments remain distinct.
4. New GPU probes exercise independent free-running generation and fixed natural-language full logits through the actual scheduler and owned model session. The server/CLI does not expose this new profile yet.

## Matched numerical results

Same RTX 4090, BF16 SmolLM2-135M, model path `/data/riley-benchmark/20260827T051948Z-d7ad713a/model`. Optional dependency is FlashInfer 0.6.16.post3 with the already verified build overlay. Python runs only for the offline FP32 reference evaluator.

| Measure | Existing V7 | Prefill-only candidate | Result |
|---|---:|---:|---|
| Natural NLL, 256 targets | 2.95928943 | 2.95632435 | Lower |
| KL(FP32 reference || engine) | 0.0007591519 | 0.0008375119 | 10.32% higher; gate fails |
| FP32 argmax matches / 256 | 244 | 242 | Lower |
| Independent greedy token differences vs V7 / 1,024 | 0 | 56 | Strict equivalence fails |
| Independent sequences differing vs V7 / 32 | 0 | 8 | First divergence in prompt-length 15/16 cases |

The natural screen reuses the previous eight passages and fixed 32 prompt/32 target tokens per passage. Both NLL and KL were required to be no greater than baseline; no threshold was changed. The small screen does not establish general quality ranking. The free-running inputs are eight independent synthetic boundary prompts repeated four times; repeated-prompt invariance passes within each engine. These failures are recorded as failures, not hardware skips or successful tests.

## Build, runtime and evidence scope

- The no-CUDA `cargo check -p riley-runtime` passes. All twelve changed source/test files match the remote hashes in `source-verification.json`.
- Final Cargo release builds both new GPU tests. `final-build.log` and `final-verification.json` identify the final binaries; `final-check.py` preserves exact final commands and expected outcomes.
- Final free-running test exits 101 with 56 token differences. Both runs close the session/context with zero remaining allocations. This is a numerical failure, not an execution crash.
- Natural full-logit dump passes: 8 × 32 × 49,152 = 12,582,912 BF16 values per engine, 25,165,824 bytes each. Teacher-forced targets are identical across engines.
- Native full-model memcheck passes with **0 errors**. Outputs under memcheck are bitwise equal to the original natural dumps. This is not full-model racecheck evidence.
- Existing V7 natural logits are bitwise identical to the prior FFN screen's V7 baseline (`492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`), supporting that the default path did not change for this corpus.
- Current verification runs on SM89 only. Hopper/Blackwell runtime and multi-GPU are unavailable; no runtime pass is claimed for them.
- The first evaluator invocation referenced a nonexistent remote script path, before loading the model. The repository evaluator was copied explicitly and rerun successfully. `natural-evaluation.log` is that completed run.

Raw logits remain in `/tmp/riley-opt-260912/prefill-natural-v1` and `/tmp/riley-opt-260912/prefill-natural-memcheck-v1`. Committed `metrics.json` includes per-passage metrics and output hashes. No long-concurrency stability or latency/throughput/vLLM test was performed for this unaccepted profile.

## Decision and remaining work

Keep the new factory explicit and experimental. Do not promote it or combine it with the prior failed FlashInfer decode profile to conceal the numerical failure. Before serving acceptance, use identical intermediate Q/K/V inputs and independent reference arithmetic to localize prefill error accumulation and evaluate a precision/backend alternative against the unchanged gates. A later serving milestone must compare previous Riley, the qualified candidate and vLLM under matched conditions. The overall serving objective remains unmet.
