# Native full-graph numerical investigation

**Experimental source only. Not a qualified Riley release or a performance receipt.**

The isolated source is `/tmp/riley-g04-native-profile-source-260911`, based on `9b53ffa14cea7c066fda8eff65b977c860b12afb`. The established candidate and its binaries are preserved. The experiment redirects the HF profile to a private numerical profile; this redirect, trace hooks and test changes must not be promoted as a production patch.

## Confirmed numerical fixes

The new attention implementation matches all 960 actual vLLM attention invocations (see sibling attention-policy receipts). Model integration also requires:

- retaining the post-attention residual in FP32 until the next input norm;
- one BF16 store after norm weighting and fused SwiGLU;
- the observed RMSNorm reduction order: four contiguous values per thread, explicit FMA partial sums, warp and eight-warp reduction;
- an explicit FMA for `sum * (1 / 576) + epsilon` before reciprocal square root.

`tail-comparison.json` localizes the first residual error to one BF16 output of the next input norm, after exact O projection, post norm, gate/up, SwiGLU and down projection. `tail22-comparison.json` found a second one-element error. A direct experiment showed the original vLLM cubin matched the expected norm while JIT recompilation of its PTX did not. `norm-vllm.sass` confirms that the original assembler fused the mean and epsilon addition into FFMA. Explicitly preserving that operation removed the difference.

## Controlled native decode result

`graph-normfma-test.log` and `normfma-comparison.json`: with the observed prefill KV supplied to the original scheduler-owned paged pool, all 31 decode positions × 30 layers × Q/K/V/context are exact against the actual serving trace. All 32 generated tokens match. This is an oracle-input control and does not qualify native prefill.

`compute_prefill.py` separately computes prefill from checkpoint weights, Torch-compiled model segments and our CUDA attention. It does not consume observed context or KV files. All 60 resulting KV tensors match (`computed-prefill-validation.json`). `graph-computed-test.log` connects these independently calculated KV tensors to the native graph and also matches all 32 tokens. This file-based diagnostic bridge is not a production prefill implementation.

Both graph checks retain three requests, a 23-token canceled prefix, non-contiguous physical block mappings, 341 replays, one capture and zero CUDA allocation counter deltas. Those counters do not cover Python allocations or host vectors used by the diagnostic file bridge. The former Riley-eager logits equality assertion is intentionally disabled in the isolated numerical experiment because the numerical contract differs; passing its lifecycle test must not be presented as full logits parity with the old profile.

The current independent native-prefill rerun removes all KV file injection. Its result is recorded separately as `graph-own-final-test.log`. Production integration, end-to-end qualification of the final source/binary and exclusive-GPU preflight remain required. Performance trials: zero.

## Final resolution and promotion

The later `graph-all-final-test.log` and `native-all-final-validation.json` supersede the incomplete own-prefill result above: BF16 split-K partial sums (QKV 192, O 128, down 320; gate/up unsplit) resolve the packed-prefill GEMM difference. All 19,080 full native prefill/decode tensor comparisons and all 32 tokens match, without external KV or context input. The validated implementation was cleaned of diagnostics and promoted as an explicit profile; see `../20260911-g04-vllm-profile/README.md`. The original failed and controlled experiments remain as diagnostic evidence, not release claims.
