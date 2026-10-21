# v15B experiment evidence

**Not adopted at this checkpoint.** See [the report](../../../docs/qwen-prefill-host-v15b-20261021.md).

- [Archive](../20261021-prefill-host-v15b.zip): 6,620,083 bytes /700 entries,
  SHA256 `7022abcfc37e8dfec2a29d409d9ca8d249661f7bfca35e6c6cacb64bc5d20613`.
- [Five-repeat normal statistics](final-statistics.json),
  [resource audit](resource-audit-final.json), [decision](adoption-decision.json),
  [source/binary lineage](source-lineage.json), [source936](source-manifest.json),
  [normal input stability](normal-input-stability.json), [stock startup proof](stock-startup-proof.json).
- Archive includes35 full-SSE arm audits(30 normal+5 separate profiles), raw requests,
  graph-node attribution, actual pointer-batched calls, full native/HF byte audits,
  fresh whole-device resource reruns and156 in-place attention component cases.
  Old incomplete resource audits and failed setup/test-build attempts remain.
- Heavy index links8,977 BF16/ELF/trace files on EFS. Original failed and corrected
  attempts are distinct; duplicate reruns are not unique logical coverage.
- The early script snapshot in source-lineage predates later diagnostic helpers.
  Actual normal plan runner/binary hashes and post-normal source936 stability were
  independently verified; final helper contents are in the archive.
- All archive entries were locally verified for size, SHA256 and CRC; see
  [verification](local-archive-verification.json).

EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15b-20261021`.
