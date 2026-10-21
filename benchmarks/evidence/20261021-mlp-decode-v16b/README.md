# v16B evidence

The candidate is **not adopted**. Its literal native/HF accuracy passed, but the five-repeat normal comparison produced only +1.25%/+1.35% throughput and retained paired tail flags. The normal matrix contains all 30 completed arms; separate diagnostics add four arms. No completed normal sample was dropped.

- [Report](../../../docs/qwen-mlp-decode-v16b-20261021.md), [all repeat statistics](final-statistics.json), [decision](adoption-decision.json).
- [Archive](../20261021-mlp-decode-v16b.zip): 5,928,687 bytes, 654 entries, SHA256 `db3007f4c5b3e1fae87eedfbdb2c3ab47719f64c10b4252c64605c56744eb0ee`.
- [Local verification](local-archive-verification.json): all CRCs, entry sizes/SHA256; 937-source baseline lineage and exact reconstruction of changed sources from archived patch.
- [Server/SSE audit](independent-audit.json): 34 normal/profile arms; 6490 stored full SSE values verified. Stock outputs are not Riley's exact numerical reference.
- [Source and binary](source-lineage.json), [source manifest](source-manifest.json), [normal input stability](normal-input-stability.json), [finite GPU resources](resource-audit-final.json), [actual QK/PV dispatch](actual-decode-dispatch-proof.json), [stock startup](stock-startup-proof.json).

The archive includes teacher/mixed byte audits, all raw commands, SSE, profiles' analysis JSON, failed preflight attempts and scripts. BF16 files, ELF binaries and Nsight SQLite/reports remain indexed by exact SHA/size on EFS `/mnt/efs/nlp/psyche/riley-mlp-decode-v16b-20261021`. They are not embedded in Git. GPU6 foreign work was left untouched and all finite GPU jobs used GPU7 under the original persistent holders.
