# Riley decode fast-path evidence — 2026-08-27

This directory is append-only evidence for the `c1/p128/o32/greedy` native Riley
decode campaign run on `ai-assistant` (RTX 4090). It intentionally keeps failed
or non-promoted profiling slices beside the final candidate so the promotion
decision can be audited.

## Final promoted candidate

- Source commit: `436dad64526f228292c24cec73a652a72e0b1e38`
- Source archive SHA-256: `385107119fef5ebe65a80632bc3b4c765a77fea9bac5a07feb51331fc3ac03fa`
- Release ELF SHA-256: `a6870817235c80342878258a689973d979ebdcd0e04bc2638235a82a135b1ffc`
- Correctness receipt: [correctness-report-cooperative-shared-kv-heads.json](./correctness-report-cooperative-shared-kv-heads.json)
- AB/BA replay script, 10 raw runs, strict report, and NCU output:
  [cooperative-kv-profile](./cooperative-kv-profile/)

The strict five-pair report records TPOT p50 `7.165961 -> 4.108705 ms`
(-42.66%), E2E p50 `227.594989 -> 131.403612 ms` (-42.26%), and throughput
`140.534702 -> 243.536158 tok/s` (+73.29%), with zero failures and dropped
trace records.

## Experiment history

- [native-profile](./native-profile/) contains the Phase 1–3 bucket/packed/GPU-greedy
  candidate evidence.
- [grouped-head-profile](./grouped-head-profile/) records the generic grouped-head
  attention experiment. It passed correctness but regressed against the prior
  bucket candidate and was not promoted.
- [shared-kv-profile](./shared-kv-profile/) records the serial shared-KV prototype.
  It improved over the fixed baseline but was superseded by cooperative preload.

Every profile JSON binds source, release ELF, model, prompt, GPU, image, runtime
flag, correctness report hash, and generated-token hash. NCU CSVs are attribution
artifacts only; promotion is based on the paired profile reports.
