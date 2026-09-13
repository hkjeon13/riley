# FlashInfer model integration: numerical gates not met

2026-09-13, base `ea8fd037a8472ca5c62957a0e385cb491a1127fc`.
**The experimental full model executes, but greedy equivalence and batch invariance fail. It is not promoted and serving performance has not been measured.**

## Implementation

The Rust factory `into_owned_variable_flashinfer_experimental_session` explicitly selects the new arithmetic. A separate 33,456-byte device workspace is included in the reservation parents and retained until graph completion/close. Native recording checks context, active parent lease, exact extent, and aliases against mutable buffers and weights before capture. Compact and full-logit graphs both use the new pure-decode model entry. Existing exact factories never pass this workspace. A build without FlashInfer rejects the experimental recorder instead of silently falling back.

The catalog digest binds the experimental profile identifier, adapter source, shared ABI declarations and dependency lock verifier. Prefill and mixed prefill/decode graphs still use existing arithmetic. No server flag selects this experimental profile yet.

## Actual model observations

RTX 4090; fixed SmolLM2-135M checkpoint and the existing V11 reference fixture. Inputs are teacher-forced from the unchanged reference continuation, isolating arithmetic differences without letting an early mismatch alter later inputs. These are numerical/lifecycle observations, not a serving workload or quality acceptance test.

| Case | Outputs | Reference argmax matches | BF16 logits inspected | Max abs error | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Partial, up to 3 decode rows | 224 | 224 | 11,010,048 | 0.78125 | 0.11260756 |
| 32 requests, alternating compact/full | 4,096 | 4,080 | 67,043,328 | 0.703125 | 0.10845302 |

Full-logit values match bitwise for 1,898,066 and 11,319,617 values respectively. Compact outputs have no downloaded full logits and are not included in those denominators. All checked logits are finite. The fixture completed 129/141 iterations, scheduler completion, pending-close abort and zero retained CUDA allocation checks. The observation harness returns success when measurements and lifecycle checks complete; **that is not numerical gate success**. [outcomes.json](outcomes.json) records the failed gates explicitly.

The 32-request case repeats the same longest prompt for every request and teacher-forces the same reference tokens. Diagnostics reproducibly find 16 mismatches, all on pure decode at generated indices 11 or 27 (zero-based). Examples: at index 11, requests 13/14 choose token 1977 but 16/17 choose 1079 (reference 4771); at index 27, requests 19/20 choose 1584 but 21/24 choose 476 (reference 9411). These pairs all use compact output, so the cross-request disagreement is not explained merely by comparing compact versus full output. It is a counterexample to invariance across the observed scheduling histories.

Only pure decode uses FlashInfer; mixed stages retain the original backend. That difference is a leading explanation to investigate, **not an established root cause**. Layerwise comparisons and consistent arithmetic across decode-containing stages are needed before setting any new quality threshold or promoting this backend. Do not relax tolerance to turn this result green.

## Memory and build evidence

Native Rust model harness under Compute Sanitizer memcheck: **0 errors**, partial case. It uses no Python FlashInfer imports and is separate from the previous Python harness's import-related sanitizer failures. Full 32-request memcheck is not claimed.

The first Cargo build correctly rejected a cuBLASLt symlink resolving outside the selected toolkit root. The failed log is preserved. The task-local toolkit was then assembled with copies of those same existing libraries, local aliases, and content hashes in `flashinfer-toolkit-materialization.json`; the library content and host installation were unchanged. The guard was not weakened. CUDA compiler remains 13.0.88, dependency headers remain pinned. This is an assembled task-local toolkit, not a claim that a new official toolkit package was installed.

With the same FlashInfer-enabled native build, the existing exact `loaded_v7_partial` regression passed: 224 reference logit outputs over 129 iterations, with allocation/close conditions checked. This confirms the existing factory still selects its original arithmetic on that fixture.

Local no-CUDA checks for runtime/scheduler passed. CUDA Cargo compilation and GPU observations passed their execution/lifecycle checks. Hopper/Blackwell runtime and multi-GPU behavior are untested because those devices are unavailable.

## Evidence identity and reproduction

[sources.json](sources.json) verifies final local/remote source equality. `initial-harness.rs` and the initial binary hash identify the partial/compact32/memcheck runs before per-mismatch logging was added. The final diagnostic changed logging only and reproduced identical totals. Logs are retained separately.

From the integration checkout, use the task-local CUDA 13.0.88 toolkit and pinned `RILEY_FLASHINFER_DATA`, then run:

```sh
cargo test -p riley-scheduler --features cuda --test flashinfer_model_observation_gpu flashinfer_model_observation_partial -- --ignored --nocapture
cargo test -p riley-scheduler --features cuda --test flashinfer_model_observation_gpu flashinfer_model_observation_compact32 -- --ignored --nocapture
```

Set `RILEY_REAL_CHECKPOINT` and `RILEY_V3_MODEL_FIXTURE` to the same checkpoint/reference as earlier V7 tests. Historical values are `/data/riley-benchmark/20260827T051948Z-d7ad713a/model` and `/tmp/riley-opt-260912/loaded-rope-fixture-v11`. For memcheck, run the built test binary with `compute-sanitizer --tool memcheck --error-exitcode 86` and the partial test filter. Required driver library isolation follows the preceding build evidence.

The objective remains real serving superiority with correctness. These failed numerical gates change the next action: investigate/resolve arithmetic consistency, then validate independent prompts and only then measure matched baseline/vLLM serving. Existing exact serving remains the fallback.
