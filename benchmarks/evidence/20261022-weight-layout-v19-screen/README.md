# v19 independent operator evidence

The sibling ZIP contains standalone CUDA probes, cold packing kernels, exact current-reference cuBLASLt metadata, raw per-case/per-pair JSONL, build/run argv, errors, original finite observers and resource samples. It deliberately excludes model bytes and ELF files. SHA256 and every ZIP CRC were checked locally.

- ZIP: `20261022-weight-layout-v19-screen.zip`,181534bytes,106entries.
- SHA256: `74d726a40c82c0bdc734da73ce98fa64cc498c7a785fababd1821ad49b985d02`.
- EFS root: `/mnt/efs/nlp/psyche/riley-projection-layout-v19-20261022`.
- Original incomplete screen and GPU7 admission failures are retained. Neither counts as numerical qualification or timing arms.
- `component-summary.json` is derived from original raw JSONL, retaining layer identity and each timed R/variant. No HTTP throughput is inferred.
- Model weights remain on EFS, with original manifest/shard and extracted-tensor identities in `weight-extraction.json`.
- Server integration/accuracy/C16/C32 comparison remain pending at this checkpoint. Accepted runtime stays v15C.
