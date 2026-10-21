# v15C adopted prefill/host evidence

See [the implementation and full report](../../../docs/qwen-prefill-host-v15c-20261021.md).

- [Archive](../20261021-prefill-host-v15c.zip): 5,507,329 bytes / 580 entries,
  SHA256 `d6a1acd9922efd8811a52aa45017a86065dc92eefc2ce7e53d4f2ef74c60418b`.
- [Five-repeat statistics](final-statistics.json), [decision](adoption-decision.json),
  [resource audit](resource-audit-final.json), [stock startup proof](stock-startup-proof.json).
- [Actual final-binary dispatch](actual-prefill-dispatch-proof.json),
  [source937](source-manifest.json), [source/binary lineage](source-lineage.json),
  [normal input stability](normal-input-stability.json).
- [Original GPU6/7 persistent contexts and idle20GiB holds](final-holder-audit.json).
- Archive contains 30 normal arms plus one separate final-binary C32 profile,
  6,154 audited full SSE values, literal native/HF byte checks of 4,480 good BF16
  files, nine observed teacher/GPU-test jobs and 120 alignment/view cases.
- The initial observer-binding failure and pre-arm queue failure remain in the
  archive. Distinct `*-r2` reruns are the passing resource/accuracy evidence.
  No completed primary normal arm was removed to produce this comparison.
- Heavy index links 4,696 original BF16/ELF/trace files on EFS, including 208 BF16
  files from the initial failed-resource attempt. Those do not add unique
  correctness coverage. All 4,688 raw BF16 files remain on EFS.
- [Local verification](local-archive-verification.json) checks every archive
  entry's SHA256, size and CRC. Current source937 matches the qualified manifest.
- [CPU ledger lifecycle recheck](check-ledger-final.json) is supplemental evidence
  produced after the archive, using the committed helper and qualified headers;
  it does not change the frozen source, binary or normal benchmark inputs.

EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15c-20261021`.
