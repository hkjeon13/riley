# Attention, RoPE and fused residual normalization diagnosis

Status: **57 raw-tensor comparisons verified; full generated-token equivalence unresolved; performance trials 0.**

## Results

All comparisons use fixed layer-0 Riley tensors for 1, 128 and 136 tokens. The 136-token input includes the first eight common generated tokens. Counts below are unequal BF16 elements, not token errors.

| Comparison against Riley | 1 token | 128 tokens | 136 tokens |
| --- | ---: | ---: | ---: |
| Selected uncompiled RoPE Q / K | 0 / 0 | 0 / 0 | 0 / 0 |
| Isolated compiled RoPE Q / K | 0 / 0 | 15414 / 4965 | 16678 / 5318 |
| Selected RoPE + explicit attention probabilities | 0 | 1 | 2 |
| Selected RoPE + explicit attention context | 0 | 0 | 0 |
| Compiled RoPE + explicit attention probabilities | 0 | 23859 | 27267 |
| Compiled RoPE + explicit attention context | 0 | 0 | 1035 |
| Selected RoPE + PyTorch SDPA context | 0 | 0 | 1083 |
| Compiled RoPE + PyTorch SDPA context | 0 | 0 | 1305 |
| O projection + residual add, fixed Riley context | 0 | 0 | 0 |
| Selected fused RMSNorm, fixed projection + embedding | 136 | 14976 | 18315 |
| Compiled fused RMSNorm, same inputs | 205 | 26880 | 27815 |
| Residual output of either fused RMSNorm call | 0 | 0 | 0 |

RoPE outputs were directly compared with Riley's CUDA `rope` primitive. The added test uses the same reviewed SmolLM2 FP32 angle construction and CUDA `rope_table` path as `build_rope_angles`, then the exported actual Q/K projections. This is a primitive replay, not an additional tensor tap inside the forward executor.

Compiled fused RMSNorm is byte-identical to normalizing the FP32 residual sum and applying the weight before one BF16 conversion in all three cases. Its BF16 residual output still matches Riley exactly. Rounding the sum to BF16 first and then applying FP32 norm gives a different result (149, 19584, 20223 elements unequal to Riley). Thus agreement of the stored residual does not establish agreement of the fused normalization. This also explains why separate norm/activation probes alone are insufficient to choose a complete numerical policy.

The selected-RoPE explicit attention reproduces Riley context exactly despite 1–2 unequal probability elements. For the mixed 136-token input, changing the RoPE compilation or switching to SDPA introduces context differences. The repeated-token 128 case masks those context differences and must not be used alone as an attention equivalence test.

## Evidence boundaries

- The worker uses the actual loaded default vLLM model's RoPE, O projection and fused RMSNorm modules. `selected` denotes a call without an outer `torch.compile`; it is not a full eager serving configuration or a custom native-op override.
- A scoped `enable_torch_wrap(False)` context allows isolated expression compilation. None of these outputs is a tap of the full default vLLM compiled serving graph. Fusion and specialization may differ there.
- Explicit causal attention uses BF16 QK scores and probabilities with FP32 softmax. SDPA is PyTorch's selected backend; **neither is claimed to be vLLM's paged-attention backend**. No actual paged-backend equivalence is established by this artifact.
- The Riley forward trace deliberately uses reference attention. It does not validate the complete decode graph's intermediate tensors.
- Before/after default vLLM generation is the same 32 token IDs. No model weights/methods, canonical kernels, qualification gates or candidate binaries were modified.

## Files and checks

`comparison.json` contains 45 worker comparisons. `direct-rope-comparison.json` adds 12 direct Riley RoPE comparisons. `verify.py` independently decodes BF16 bytes, checks every count/error/hash, verifies the compiled fused-norm formula identity and checks generation invariance.

```sh
python3 benchmarks/results/20260911-g04-attention-trace/verify.py
```

`attention_trace_gpu.rs` is an additive ignored GPU test. Its actual CUDA run passed for three inputs and asserted zero remaining device/pinned allocations. Rust formatting, Python syntax and `git diff --check` passed. `riley-capture.log`, `vllm-capture.log`, the raw tensors and `provenance.txt` retain evidence.

Remote source `/tmp/riley-g04-prefix-source-260911`, target `/tmp/riley-g04-prefix-target`, evidence `/tmp/riley-g04-attention-trace-260911`. The canonical source remains clean at `9b53ffa14cea7c066fda8eff65b977c860b12afb`; both release hashes remain unchanged. CUDA 13 / RTX 4090, same SmolLM2 checkpoint and vLLM environment as preceding artifacts.

Reproduce the ignored test with `RILEY_REAL_CHECKPOINT`, `RILEY_TRACE_CASES`, `RILEY_TRACE_OUTPUT` set to the same checkpoint and this artifact's remote paths, using `cargo test -p riley-runtime --features cuda --test attention_trace_gpu --release -- --ignored --nocapture` and the existing CUDA 13/source-specific target environment. Then run `PYTHONPATH=/tmp/riley-g04-attention-trace-260911 /data/riley-vllm-interim.CfrT9T/venv/bin/python capture.py` from the remote evidence directory.

## Remaining integration gate

Numerical differences have now been localized across the first-layer prefix, MLP and residual-normalization operations. The next requirement is to observe these boundaries in the actual serving/paged execution on a common token prefix, then select a coherent profile and verify all generated tokens and integrated graph/HTTP behavior. Isolated expressions are not enough to adopt a production patch. The prior mismatch at generated index 8 remains unresolved.

No timing campaign ran. The GPU still has unrelated Blender processes and 743 MiB usage; exclusive-GPU preflight has not passed. Those processes were not interrupted.
