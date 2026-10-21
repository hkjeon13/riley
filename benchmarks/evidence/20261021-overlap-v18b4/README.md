# v18B4 evidence

Actual CPU–GPU overlap implemented and measured; rejected for default adoption. Matched five-run median throughput changes: C16−0.29%, C32−0.26%. Main runtime remains v15C.

- [Implementation, limitations and all repeats](../../../docs/qwen-overlap-v18b4-20261021.md)
- [Normal statistics and paired tails](final-statistics.json), [initial cohort](initial-cohort-audit.json)
- [Numerical/contract qualification](extended-accuracy-audit.json), [resource/SSE audit](resource-audit-final.json)
- [Actual CPU/GPU interval overlap](actual-candidate-dispatch.json), [process-wide handoff counters](overlap-handoff-proof.json)
- [Source lineage](source-lineage.json), [reconstructable patch](../../candidates/overlap-v18b4/candidate.patch)
- [Core archive](../20261021-overlap-v18b4.zip), [SHA receipt](evidence-archive-receipt.json), [local CRC/SHA/source verification](local-archive-verification.json)
- [Supplemental inspected archive](../20261021-overlap-v18b4-inspected.zip), [receipt](inspection-archive-receipt.json), [verification](inspection-archive-verification.json), [four-arm audit](inspection-independent-audit.json), [stage summary](profile-summary-inspected.json)
- [Original IPC startup/continuation correction](ipc-startup-correction.json), [inspection correction](inspection-correction.json), [trusted helper correction](trusted-helper-correction.json)
- [Supplemental attempt logs and remote hashes](inspection-attempts/remote-sha256.json) (stored outside the ZIP)
- [Fresh final GPU6/7 holds](post-supplement-holder-audit.json)

Core ZIP retains all 30 normal and 4 original native-NVTX profile arms, raw SSE, helpers, prototype/correction logs and heavy-file SHA index. Original external GEMM injection libraries were absent; use the separate four inspected profiles for stage classification, not for normal throughput. Core CPU NVTX interval proof remains valid. No completed normal arm was repeated/excluded. Separate224-response B4 functional pilot and B3 preflight are retained and never mixed into normal metrics.

Heavy BF16/binary/SQLite evidence remains under `/mnt/efs/nlp/psyche/riley-overlap-v18b4-20261021`; supplemental traces/libraries under `/mnt/efs/nlp/psyche/riley-overlap-v18b4-inspected-20261021`, with exact SHA256 indexes inside each archive. Original failed process status is preserved; C16 completion is explicitly an aggregate of two original and 13 continuation arms using the unchanged runner.
