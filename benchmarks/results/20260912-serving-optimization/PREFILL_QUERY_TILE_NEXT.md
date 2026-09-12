# Next attention experiment after Round51

Round51 remains the acceptance gate for V43. Do not change or rebuild the frozen candidate during that run. Prior turn implemented and verified V43; this turn verified controller PID3588885 and local session9614 live. Blender must remain stopped.

## Evidence and hypothesis

V42 node-level traces (`raw/native-trace-v42/analysis.json`) put prefill attention at mean37.947/38.695us per layer at C16/C32, versus total prefill kernel medians2780.278/2791.030us. These are profiler observations, not production timing estimates. Current `kernels/src/prefill_shape_attention.cuh` runs one query per warp, duplicates that query across the16 MMA rows, and publishes only group0. It reloads the same K/V data for adjacent queries. This is a source-confirmed source of redundant arithmetic; a speedup is not yet established.

## Proposed bounded batch

1. Prototype one warp per query head with8 or16 adjacent query rows mapped to MMA M rows. Reuse each loaded key fragment across those queries. Keep four K16 score MMAs in their existing arithmetic order.
2. Keep per-query causal masks and reverse128-token tile traversal. Maintain independent maxima, rescaling factors and denominator partials for each represented query. Preserve original per-t denominator accumulation order and final XOR2/XOR1 reduction. Queries whose causal end precedes a tile must retain state without evaluating an invalid negative-infinity subtraction.
3. Reuse BF16 probability and V fragments in value MMAs across queries. Preserve ordered16-token value accumulation, BF16 rounding, final normalization, and KV page mappings. Bound shared memory and verify inactive/tail outputs.

A16-row tile cuts parallel blocks; at128 input rows and9 query heads it yields72 CTAs, fewer than4090 SMs. Compare8-row tiles as well instead of presuming maximal reuse wins. Compile and run only after Round51 terminates. Prototype against the saved independent V41 attention oracle with exact BF16 comparison over disjoint/permuted pages, causal poison, partial query tiles, starts0/13/128/1024 and contexts through4096. Memcheck and racecheck must pass before model integration. A changed numerical result requires analysis, not relaxed tolerance.

## Acceptance

Measure isolated attention only to screen variants, then owned full-logit tests and mixed streaming/nonstreaming HTTP. Freeze the selected integrated build and compare it against the accepted preceding build and vLLM on the complete serving matrix. Include TTFT/TPOT and sampled P95/P99, and retain the unmet long-run stability qualification. A microbenchmark win alone cannot establish the user goal.

## Prototype preparation

`prefill_query_tile_v44.cuh` now contains an experimental8/16-row query tile. `prepare_query_probe_v44.py` generates the independent-oracle comparison with additional7/8/9/15/16/128 row boundaries. `check_query_tile_v44.py` refuses to run while Round51's controller is live, then builds both variants and runs memcheck/racecheck. Files are staged remotely under `/tmp/riley-opt-260912/query-tile-v44`, outside the authoritative source. Nothing has been compiled or run yet; no numerical or performance claim applies.

Remaining correctness concern before any serving integration: sharing V across queries also presents a later query's causal suffix to earlier queries as zero-probability MMA terms. Finite-data exactness must be tested, and nonfinite V within a mixed-query tile needs explicit treatment or a verified precondition; zero probability times NaN is not harmless. Existing single-query suffix poisoning alone does not prove this multi-query case. Do not relax the oracle or silently accept contamination.

## Nonfinite causal fallback prepared

The prototype now detects nonfinite BF16 V fragments and uniformly falls back to the existing per-query attention body for the entire query tile before publishing output. This preserves per-query V masking when a later visible value is NaN/Inf; it adds an exceptional path and a warp vote whose cost still needs measurement. `augment_query_probe_v44.py` additionally poisons suffixes while keeping all query rows active. It requires bit-exact finite prefixes and agreement with the independent oracle for the full tile (NaN payload differences alone are classified as NaN). The original all-finite oracle comparisons remain bit-exact. Compilation and execution are still pending Round51 completion.
