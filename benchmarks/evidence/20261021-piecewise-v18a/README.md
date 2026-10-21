# v18A evidence

Rejected multi-prefill/variable-R MLP Graph candidate. Main runtime remains v15C.

- [Report](../../../docs/qwen-piecewise-v18a-20261021.md)
- [All five-repeat statistics and paired tails](final-statistics.json)
- [Numerical/contract checks](extended-accuracy-audit.json)
- [Actual profiled dispatch](actual-candidate-dispatch.json)
- [Initial request cohort](initial-cohort-audit.json)
- [Resource audit](resource-audit-final.json)
- [Source lineage](source-lineage.json) and [patch](../../candidates/piecewise-v18a/candidate.patch)
- [Archive](../20261021-piecewise-v18a.zip), [SHA/size receipt](evidence-archive-receipt.json), [local reconstruction verification](local-archive-verification.json)

ZIP includes all normal/profile raw SSE values, helpers, attempt/correction logs and indexed heavy artifacts. BF16 tensors, binaries and SQLite traces remain under `/mnt/efs/nlp/psyche/riley-piecewise-v18a-20261021` with SHA256 in heavy-evidence-index.json inside the archive. No completed normal arm was excluded. See post-export-holder-audit.json for the separate fresh hold check.
