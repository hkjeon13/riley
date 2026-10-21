# v16E mixed Gate/Up component evidence

Not adopted. Two finite executions each cover240 generated-weight cases. The second compares both the qualified legacy route and the optimized v16C prefill route. Both BF16 references match; timing shows no broad improvement over v16C. This is not whole-model or HTTP qualification.

- [Report](../../../docs/qwen-mlp-small-v16e-20261021.md), [full audit and times](component-audit.json).
- [Archive](../20261021-mlp-small-v16e.zip), [receipt](evidence-archive-receipt.json), [all CRC/SHA/size checks](local-archive-verification.json).
- Raw CUDA/host source, two build logs, actual direct-child PID/UID/environment receipts and memory samples are in the ZIP; ELF SHA/size points to EFS `/mnt/efs/nlp/psyche/riley-mlp-small-v16e-20261021`.
