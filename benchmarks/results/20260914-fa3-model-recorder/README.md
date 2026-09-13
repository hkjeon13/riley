# FA3 model recorder integration — 2026-09-14

FA3 now has an explicit experimental SmolLM2 model/session/server path. It remains disabled by default and unqualified. Runtime is Rust → C ABI → native CUDA; Python is an offline build/benchmark tool.

## Implementation batch

1. Connect mixed prefill/decode and pure decode to the existing 30-layer recorder. Existing QKV, RoPE, KV writes, projections, FFN and output selection surround FA3 attention. Prepare scheduling once per batch and retain the external 71,424-byte workspace with graph parents. Cold kernel preparation precedes capture.
2. Correct pure-decode metadata translation: its wire total and row offsets are zero, while FA3 needs one query per active request and cumulative offsets. Mixed packets retain their packed offsets. Add valid and malformed pure-decode replay cases.
3. Add Rust ownership/session selection and a distinct source-hashed numerical profile. `--graph-numerics fa3-smol-experimental-v1` is loopback-only, requires graphs, and excludes paired decode, FFN and wall-time-policy experiments. Exact profiles reset the FA3 flag.
4. Add real-model unsupported-device rollback coverage and compile strict Hopper full-model fixture tests. No numerical thresholds were relaxed.

Pinned FA3 `98eb7998a0eba4047c7a30375522569d4b8efb20` and CUTLASS `7127592069c2fe01b041e174ba4345ef9b279671` remain unchanged. Main metadata/model code targets SM89 in this build; FA3 objects target SM90a.

## Validation

| Scope | Result |
|---|---|
| Local runtime numerical-profile tests | 8 passed |
| Local server tests | 35 passed |
| SM89 metadata graph comparisons | 40 passed |
| Metadata memcheck / racecheck | 0 errors / 0 hazards |
| Optional native model recorder, Rust model and server release build | Passed |
| Actual SmolLM2 load and FA3 recorder entry on RTX 4090 | Rejected before capture: `FA3 cold preparation: operation not supported` |
| Rejected model session resource rollback | All tracked allocations zero; context closed |
| Hopper full-model fixture test compilation | Passed; runtime not run |
| Existing single / paired serving, C32 | Each: 96 warmup + 192 retained responses, all reference matches; 96 stop + 32 cancel checks; clean exit |
| Linked server runtime | Native CUDA/C++/system libraries; no Python/Torch dependency in `ldd` |

The first recorder build failed because FA3 declarations were inside a FlashInfer conditional; the second reached Rust test compilation and failed on a missing `?`. Both were corrected. Final v3, Hopper test build and server build passed. Failed logs remain under the remote recovery root's `logs/fa3-model-recorder-v1.log` and `v2.log`.

Serving validation used `/data/riley-serving-260913-recovery/workload.json`, the frozen SmolLM2 checkpoint, GUI retained, Blender down and no competing compute process. Raw responses, launch argv and logs remain in `/data/riley-serving-260913-recovery/fa3-model-existing-smoke-v1`; `validation-receipt.json` records their hashes and checks. The source is the worktree change accompanying this report, based on `9bd3e8b6`; the remote source is an archived recovery tree plus mirrored changes, not a clean git checkout. Server SHA-256: `3abb1af2281811909ca58ed306b163f676bd7fa0f57b9f4bfb32ec7b29ce72ff`.

This single-order smoke is a correctness/regression check, not a paired performance experiment. `completion.json` retains incidental timings without claiming improvement. FA3 did not execute attention, so no new FA3/vLLM performance comparison is possible on this GPU.

## Outstanding gates and rollback

Hopper is required for attention numerical correctness, empty/failed batch behavior, scheduling reuse across 30 layers, graph replay, sanitizer and serving measurements. The binary full-logit fixtures used by the new scheduler tests also need restoration on the eventual test host (`RILEY_V3_MODEL_FIXTURE`: requests and per-length decode logits); compilation alone does not validate those fixtures or model outputs.

Run `fa3_model_observation_partial` and `fa3_model_observation_compact32` from `flashinfer_model_observation_gpu` with the optional build, pinned checkpoint and fixtures. Preserve strict greedy/batch-invariance checks. Only after those gates pass, compare actual serving against the current exact baseline and vLLM in both orders with matching workload and hardware. Wider models, multi-GPU and Blackwell remain separate work.

Rollback selects the existing exact profile or omits `RILEY_FA3_SOURCE` at build time. There is no automatic quality promotion or unsupported-device fallback hidden behind the FA3 profile. The overall vLLM performance goal remains unmet.
