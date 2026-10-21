# Wide-HF prefill RMSNorm batching, v10

Two-file layer over the immutable qualified v9 source snapshot, **not a patch
against clean repository HEAD**. Earlier dirty/untracked optimization layers
are prerequisites and remain intact. See `source-layer-manifest.json` for
the exact before/after hashes and qualified binary binding.

The change merges adjacent wide-HF prefill request groups only when each
original group has at least 16 rows (the original 32-thread reduction).
A CUDA kernel packs four independent rows into four warps of one CTA.
Each row retains its four FP32 accumulation chains, descending shuffle tree,
epsilon/rsqrt operations, and both BF16 rounding boundaries. Singleton and
short-request reduction geometries remain unchanged. Smol routing, native
buffer validation/use accounting, FFI, allocation and completion contracts
are retained.

The shared wide-HF primitive serves widths 2048/2560/4096. Standalone numerical
coverage includes those widths; whole-model and serving qualification here
is Qwen3-4B on A100 SM80. This does not qualify other model/GPU combinations.

No new CLI flag is introduced. Normal qualification uses the existing
owned-HF BF16 path with:

```text
--owned-decode-batch native-prefill-sm80-v5
--execution-completion iteration-batch
--sampling-backend gpu-greedy
--batch-token-budget 576
--step-uploads staged-batch-v1
--mlp-activation fused-silu-multiply-v1
```

Riley Graph is off in normal serving comparisons. Stock vLLM keeps its
default compile/Graph configuration. The all-R Graph run is a separate
correctness qualification.

Status: CPU (522 passed), standalone component, original native/HF
whole-vocabulary eager/Graph, five fresh C16/C32 serving repetitions,
C1/C8 supplements, three longer C1 confirmations and raw-SSE/resource
audits passed. Throughput increased 2.01%/3.75% at C16/C32; TTFT p95
decreased 11.50%/9.85%. This does not achieve the plan's +10% throughput
target or qualify other models/GPUs. The first C1 supplement's bad repeat
is retained alongside the longer confirmation results. See the
[execution report](../../../docs/qwen-prefill-norm-and-gap-v10-20261020.md)
for current results, rejected experiments, limits, and final adoption status.

Rollback uses the immutable v9 binary, or reverses this layer on a source
tree whose hashes match the manifest. Do not reverse unrelated earlier
layers or overwrite the original native/HF oracle.
