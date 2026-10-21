# v16C evidence

**Not adopted.** The combined prefill merged Gate/Up and decode QK/PV candidate preserved the original accuracy gate, but normal throughput improved only +2.90%/+1.69% and paired tail flags remain.

- [Report](../../../docs/qwen-mlp-prefill-v16c-20261021.md), [all repeats](final-statistics.json), [decision](adoption-decision.json).
- [Archive](../20261021-mlp-prefill-v16c.zip): 5,993,745 bytes, 721 entries; SHA256 `738c000926e27d2e3c24d0fe1557ee84a9f22ee12debc2b68f0729cb19e6953d`.
- [Local verification](local-archive-verification.json): CRCs, every entry SHA/size, baseline source937 and exact reconstruction of all five changed source hashes from archived candidate patch.
- [Full SSE audit](independent-audit.json): 34 arms, 6490 stored SSE values verified; exactly five before/after/stock repetitions per concurrency, plus four separate profile arms.
- [Source/binary](source-lineage.json), [source manifest](source-manifest.json), [stable inputs](normal-input-stability.json), [GPU resources](resource-audit-final.json), [actual merged dispatch](actual-merged-dispatch-proof.json), [stock startup](stock-startup-proof.json), [persistent holders](final-holder-audit.json).

The first C16 stock startup failed because its EFS IPC directory was missing. The two completed Riley arms were retained. [Continuation](normal-continuation-proof.json) resumed stock r0 and later runs; [r4 supplement](normal-continuation-r6-proof.json) completed the exclusive-end repeat range after C32. No completed normal arm was dropped. The first triple and last C16 repetition therefore have protocol interruptions, recorded in the report. The original runner, source, binary and original plans stayed unchanged; continuation plans were separately hashed and bound to executed argv.

The archive includes 4,480-file native/HF byte audits, CPU/C02 logs, six actual GPU contract tests (original120 alignment cases plus36 new raw merged cases), raw commands/SSE, profile analysis, preflight failures and corrected analysis scripts. Large artifacts are named with sizes/SHA256 in the archived heavy-evidence index on EFS `/mnt/efs/nlp/psyche/riley-mlp-prefill-v16c-20261021`. GPU6 foreign jobs remained untouched; all finite GPU runs used GPU7. Original GPU6/7 holder contexts remained alive.
