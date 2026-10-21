# v16G evidence

**Unadopted.** The coalesced Gate/Up + QK/PV candidate passes6,528 original-reference BF16 file checks, but the original paired tail gate fails. The qualified runtime remains v15C.

- [Report](../../../docs/qwen-mlp-batch-v16g-20261021.md), [all five repeats](final-statistics.json), [decision](adoption-decision.json).
- [Archive](../20261021-mlp-batch-v16g.zip):5,996,613bytes/647entries, SHA256 `90ce769950a5ea8361ce83e16b152ab08e376883b950febbe37f850fd6aa082e`.
- [Local verification](local-archive-verification.json): CRC, every entry SHA/size, unchanged sources and reconstruction of all six changed source hashes from baseline57502e7 using the [exact patch](../../candidates/native-coalesced-v16g/candidate.patch).
- [Source lineage](source-lineage.json), [937-source manifest](source-manifest.json), [extended accuracy](extended-accuracy-audit.json), [GPU resources](resource-audit-final.json), [stock startup](stock-startup-proof.json), [original holders](final-holder-audit.json).
- [SSE audit](independent-audit.json):34arms and6490stored SSE values verified; exactlyfive fresh processes per engine/C,30normal and4separate profile arms. No completed normal sample excluded.
- [Actual coalesced dispatch](actual-coalesced-dispatch-proof.json), [PV geometry](actual-pv-dispatch-proof.json), [profile summary](profile-summary.json). The summary is derived from archived trace analyses; Nsight brackets only64keep after correctness and64warm. Normal measurements have192keep and no profiler. Aggregate timing differences with different batch counts do not prove fixed-work kernel speedups.

All10finite accuracy/contract receipts succeed, CPU524pass/40ignored and sixGPU contract functions include120alignment/view cases and66new raw merged cases. The old4,480native/HF file gate and additional2,048capacity576files use unchanged references/tolerance. Only the four preauthorized C02 unit fixture trees use private/tmp, then are removed. Binaries, scripts, models, tests, other scratch and heavy BF16/ELF/SQLite/Nsight artifacts stay EFS `/mnt/efs/nlp/psyche/riley-mlp-batch-v16g-20261021`, indexed by SHA/size. GPU6foreignwork remains untouched; all finite work runs sequentially on7. Three final samples verify the original6/7holders idle20GiB with unchanged PID/start ticks and no release/restart.
