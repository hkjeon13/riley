# Attention split compatibility — legacy profile rejects independent merge

The existing decode path already partitions QK scores over context tiles and value products over eight output-dimension blocks. Normalize-once/value-remapping experiments also already failed native performance screening. See [earlier experiments](../20260913-attention-task-costs/README.md). These are not missing implementations to repeat.

[LeanAttention v2](https://arxiv.org/html/2405.10480v2) decomposes context work using online-softmax rescaling as a reduction. That mathematical property does not establish compatibility with Riley’s existing BF16 probability rounding after each128-token running-maximum update. This probe tests a minimal independent two-tile merge, not the LeanAttention library, complete algorithm, or paper correctness.

## GPU counterexample

Q has one nonzero component equal to1. K is zero in the first128 tokens and51/64 in the second128, producing an exactly representable QK score51/512 after scaling. V is1 for the first tile and0 for the second. All inputs are BF16 representable; scores are produced by the actual existing QK kernel. Existing `independent_values` is the comparison oracle.

| Context | Second-tile K component | Differing BF16 outputs | Existing output | Independent split output |
|---|---:|---:|---:|---:|
|128|0|0|1|1|
|128|0.796875|0|1|1|
|256|0|0|0.5|0.5|
|256|0.796875|576|0.4765625|0.474609375|

The last case fails exact compatibility: independently rounded local probabilities and their later rescaling differ from rounding relative to the running maximum. Two graph replays are executed per case, and all inactive outputs retain sentinels. Probe exit0 means the expected rejection was reproduced; it is not a candidate correctness pass. No full-model or serving performance is measured here.

SM89 execution and bounded memcheck/racecheck completed with0 errors/0 warnings. SM90a and SM100a compile-only builds exited0; runtime is skipped because hardware is unavailable. Blender restoration passed. The10-member archive includes source and binary identities, compiler version, build logs and execution/restoration receipts. `verification.json` binds the archive and checks all four source hashes against this checkout.

## Decision

Do not attach a standard independent-partial merge to the legacy exact profile. PR06 is still incomplete, and this result does not reject all context parallelism or all numerical profiles. Before a different profile is tested, declare its numerical behavior, independent reference and full-model/serving acceptance criteria; preserve all legacy gates and never widen tolerances after observing a failure. Long-context benefits from the paper are not predictions for the current context1024 workload.

No serving default or runtime library changed. This bounded compatibility result prevents an invalid integration; it is not progress evidence for throughput itself.
