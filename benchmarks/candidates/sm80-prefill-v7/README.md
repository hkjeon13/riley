# Native-order SM80 prefill MLP v7

Layer on the qualified v6 snapshot, not on clean HEAD. Default modes remain unchanged. The opt-in `native-prefill-sm80-v5` retains decode v6 and adds cold-prepared cuBLASLt prefill recipes.

- Original adjacent request groups: M64..130 only; merged M161..576. Attention/KV/output metadata remains request-specific.
- Down N2560/K9728: algo6, tile24, stages15, splitK5, BF16 partial reduction4, swizzle1, custom0. Only merged groups use it. Standalone long prefill keeps native GEMM.
- Gate/Up N9728/K2560: algo21, tile20, stages11, splitK0, reduction0, swizzle0, custom0. Only merged M257..512 uses it; other qualified merges keep native GEMM.
- Private native selector11 is not a new public decode mode. Unsupported hardware/software/workspace configuration fails closed. CUDA12.4/cuBLASLt120405, A100 SM80/108SM qualification remains required.
- Descriptors and bounded workspace are prepared cold. Dispatch registers operand and private workspace uses. Cleanup destroys descriptors, handle and allocation; existing graph and buffer ownership rules remain enforced.

Finite components: actual L0 weights/seed, M64..576 x6 states x3 projections:9234 comparisons,21,684,879,360 BF16 values, zero differences against native M128. Original M64..130 native paths additionally checked in six states. These are finite diagnostics, followed by full-model and HTTP checks; they do not prove arbitrary-model equivalence.

Remote artifacts: `/mnt/efs/nlp/psyche/riley-prefill-v7-20261020` on `aws-a100`. GPU6/7 holders keep their original PID/context and restore20GiB between campaigns; no holder shutdown or foreign job termination.

Final validation:521 CPU tests after the four authorized fixture-only reruns, C1/allR full-vocabulary exactness and optional MLP Graph compatibility;31 audited HTTP/profile arms,5195 complete SSE values. C16/C32 p95 medians improve9.90%/9.84%; throughput improves2.44%/3.02%. C16 p50 is almost unchanged (+0.075ms); instrumented C16 profile span increases1.49%. No universal performance promotion.

[Full report](../../../docs/qwen-c16-prefill-mlp-v7-20261020.md) · [Evidence](../../evidence/20261020-prefill-v7/independent-audit.json). Binary SHA256: `e3fc21feb8bed3d5e63440e75bbc1c8a159147b8ad8109eef9a9cbf00c04176d`.
