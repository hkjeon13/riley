# v15A prefill-only experiment evidence

This is a **screening checkpoint**, not the final repeated vLLM comparison or
an adopted runtime change. See [the report](../../../docs/qwen-prefill-host-v15a-20261021.md).

- [Verified archive](../20261021-prefill-host-v15a.zip): 1,001,607 bytes,
  211 entries, SHA256 `ecca0de32d600b3c564be32b5f3ea3ed62c1cf21fa640b127a264fcd62fad26b`.
- [Source/binary lineage](source-lineage.json), [source935](source-manifest.json),
  [screening metrics](screen-summary.json), [SSE/resource audit](audit-partial.json).
- Archive includes full teacher byte audit (2,464 BF16 files), 156 attention
  component cases, CPU/C02 receipts, five stored-SSE arm audits, host diagnostics,
  graph node attribution and raw serving records. Heavy index links 2,469 BF16/ELF
  files on EFS. Original binary/full tensors remain on EFS.
- All archive entries were checked locally for size, SHA256 and CRC; see
  [verification](local-archive-verification.json).
- C16 is one 64-warm/64-keep pair; C32 is one 64-warm/192-keep pair with host
  tracing enabled for both engines. Neither is the later five-repeat normal campaign.
- `source-lineage.json` records the earlier script freeze; subsequent helper and
  diagnostic script contents are preserved exactly in the archive. Source935 and
  the measured binary were rechecked when exporting. Setup failures are retained.

EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15a-20261021`.
