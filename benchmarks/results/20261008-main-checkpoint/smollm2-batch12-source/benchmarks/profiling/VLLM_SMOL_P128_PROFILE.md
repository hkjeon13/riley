# SmolLM2 P128 profile matched to vLLM

`vllm-smol-p128-v1` is an explicit numerical contract for SmolLM2-135M BF16 on SM89, CUDA runtime 13.0 and cuBLASLt 13.1.1. It requires one 128-token prompt and at most 32 output tokens. The recorded complete-model qualification uses `Hello` repeated 128 times against vLLM 0.27.1 with compiled model segments, FlashAttention2 and cache disabled. Other models, lengths and numerical environments are not qualified by this receipt.

The default graph and HF eager contracts are unchanged. This profile uses reverse-tile tensor-core attention, FP32 fused residual lifetime, the compiled RMSNorm reduction/FMA order, and BF16 split-K partial sums matching packed prefill GEMMs. It reuses the scheduler's physical KV mapping and the existing retained parent allocations. Request admission rejects unsupported lengths; eager fallback is forbidden.

Server selection:

```
--execution-graph-policy require --graph-numerics vllm-smol-p128-v1
```

Use the qualified single-row executor configuration: one active sequence, batch token budget and prefill chunk size 1, separate residual-norm selection, iteration completion, packed metadata and at least ten KV blocks. CPU and GPU greedy sampling are both qualified for the recorded request. Graph-specific numerical operators replace the configured eager arithmetic only under this explicit profile.

Engine-only runner identity:

```
--runtime-flag-name graph_numerics --runtime-flag-value vllm-smol-p128-v1
--semantic-class VLLM_REFERENCE --correctness-gate-id g04-vllm-smol-p128-v1
```

The runner still requires all source, environment, workload and correctness-report bindings. `--prepare-only true` loads the model, prepares the graph and trial, and closes resources without executing a performance trial. Its receipt is `riley.native-profile-preparation.v1`. Actual measured runs use the separate closed `riley.vllm-profile-run.v1` schema and `check_vllm_profile_run.py --binding FILE`. A binding contains the preparation receipt's `source`, `environment`, `workload`, plus exact `input_token_ids` and `generated_token_ids` from the reference.

This is not an E0 bitwise optimization of the previous Riley eager path. The existing E0 schema and pair checker remain unchanged and must reject these results. GPU graph event timing remains unmeasured; host execution time is not GPU time. SmolLM2 remains a diagnostic cell and does not establish broader M4/M5 performance claims.

The first implementation still records canonical GEMMs followed by the bounded prefill override; the override returns immediately during decode. This is a known performance cost, not a correctness gap. No speedup is implied by correctness qualification. Future changes to launch structure, arithmetic, model, environment or workload require fresh qualification.
