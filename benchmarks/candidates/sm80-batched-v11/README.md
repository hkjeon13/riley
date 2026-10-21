# Qwen batched projection, attention and decode body Graph, v11

Seven-file layer over the immutable qualified **v10 source snapshot**, not
clean repository HEAD. Earlier dirty optimization layers are prerequisites.
`source-layer-manifest.json` binds the exact input root, seven output hashes,
patch hash, 935-file manifest and candidate binary.

The final E candidate implementation:

- Saves the original mode10 selector before native mode normalization, so the
  new fast paths and their workspace retention guards are actually reached.
- Extends the pinned native O/Down cuBLASLt recipes to R2..32, using checked
  per-row layouts and a separate cold workspace capped at 4 MiB per plan.
- Shares K/V projection tiles without changing native FFMA/reduction order.
- Fuses Gate/Up and BF16 SiLU/multiply for independent singleton decode groups,
  including mixed prefill/decode iterations; prefill keeps its native recipes.
- Shares KV between GQA query heads while retaining token/depth CTA parallelism
  in three QK/softmax/PV kernels. Positions/family selection come from the GPU.
- Processes LM-head R17..32 in one native-order WMMA traversal.
- Captures the qualified Qwen decoder body, final norm and LM head by actual row
  count. Greedy/logits use disjoint keys. Startup prepares serving keys before
  accepting requests. Embedding, metadata, sampling and output transfer remain
  outside the Graph. R1 and unsupported context lengths use eager execution.

This qualifies Qwen3-4B BF16 on A100 SM80 in the tested geometry. It does not
qualify other models, GPUs or arbitrary inputs. No tolerance, quantization,
forced EOS or new CLI flag is introduced. Existing owned-HF safety guards,
buffer validation/use accounting and failure poison contracts are retained.

Serving selection:

```text
--owned-decode-batch native-prefill-sm80-v5
--execution-completion iteration-batch
--sampling-backend gpu-greedy
--batch-token-budget 576
--step-uploads staged-batch-v1
--mlp-activation fused-silu-multiply-v1
--owned-decode-graph segments-v1
```

See the [execution report](../../../docs/qwen-batched-attention-whole-graph-v11-20261020.md)
for measured performance, exact accuracy scope and remaining limitations.
Rollback uses the immutable v10 binary, or reverses this patch only on a tree
whose qualified output hashes match. Preserve unrelated prior layers and the
original native/HF oracle. GPU6/7 holders must remain running during rollback.

Qualification: CPU 523 passed (four approved private `/tmp` C02 fixtures removed),
2464 original native/HF BF16 files byte-equal, actual integrated GPU fast paths
verified, and five fresh C16/C32 comparisons passed independent SSE/resource
audits. Throughput improved 17.93%/17.59%; vLLM throughput ratios are
60.99%/52.72%. Both remain below the +20% target. Original GPU6/7 holder
contexts remain alive with idle20GiB allocations.
