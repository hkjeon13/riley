# V48 packed prefill qualification

Status: V48 integrated and correctness-qualified; Round55 serving measurement complete. General serving baseline remains V46. V48 candidate source is `4195aa5dd08e5ebbc604c89a09ae8831fc339e0d`. Blender remains stopped with no restoration scheduled, as requested.

## Motivation and batch boundary

Round54 widened decode improves natural C32 throughput but regresses fixed P128 TTFT. The next batch combines dense token packing, owner-isolated RoPE/causal attention/KV writes, multi-owner publication, and scheduler admission of up to four available prefill owners. It must operate on available requests within the actual iteration token budget, without waiting to fill a batch or switching on benchmark identity.

## Prototype and evidence scope

The prototype shares projection, norm and MLP work over packed tokens while retaining the established BF16 arithmetic. Attention and KV writes use separate owner page tables and committed prefix positions. Partial and inactive owners publish zero hidden rows. The full-model probe compares selected hidden rows and the entire KV allocations against sequential execution, including guards and mixed prefix lengths.

The first 24-case run used identical checkpoint weights but an older RoPE fixture. Its 20-file archive is preserved and SHA256 verified under `raw/packed-prefill-v48`; its timings are preliminary and must not be described as current-serving measurements. A separate 98-case attention guard suite passes memcheck/racecheck, including nonfinite values and invalid owner counts; that primitive suite does not depend on RoPE.

The corrected run uses `loaded-rope-fixture-v11`: weights SHA256 `b18c04c7b10287fa00671830ae48c3453bdc91a2d1ceaac3f803cc7bb441aa47`, RoPE SHA256 `b37f657fa03a71cbc0879b6352710990223ff98d4f9633a8434c83f9abc8852e`. It expands to 32 cases and up to 1,024 aggregate tokens, comparing 3,021,078,528 bytes per run. Correctness, memcheck and racecheck all pass. Timing completed with 96 records over three reversed-order pairs, five warmups and twenty measured graph replays per sample. All 15 corrected evidence files are SHA256 verified under `raw/packed-prefill-loaded-v48`.

Timing includes graph H2D, 30 layers and selected hidden output, but excludes the LM head, result D2H, CPU scheduling and HTTP. It cannot establish serving throughput, TTFT, TPOT or parity with vLLM.

## Required integration contracts

- Introduce an explicit V6 wire/session capability; V5 reserved metadata cannot silently acquire packed meanings. Validate owner count, aggregate token count, per-owner packed offset, page ownership and dense published slots before GPU access.
- Allocate GPU token capacity from the iteration token budget, including 1,024-token packing when each owner's maximum chunk is 512. Preserve existing V3/V4/V5 paths.
- Produce selected hidden rows for the shared 32-row head and versioned full/compact results. The existing four-byte publication allocation cannot hold the prototype's 32 flags; use an optional flag pointer or a correctly sized allocation.
- Select up to four ready prefill requests within the real budget, including partial prefills. Give unpublished partial owners distinct internal slots without exposing them as token outputs. Audit class selection and dispatch accounting for multiple prefill rows.
- Require model/ownership/corruption/abort tests, concurrent HTTP and stochastic full-result fallback before matched V46/V48/vLLM serving measurements. The fixed short-output TTFT regression and high-concurrency tails are acceptance gates.

No V48 performance adoption or overall goal completion is claimed.

## Corrected component timings

| Pattern | Owners | Total tokens | Sequential us | Packed us | Change |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 128 | 2522.92 | 2509.06 | -0.55% |
| 0 | 2 | 256 | 5045.81 | 3124.38 | -38.08% |
| 0 | 3 | 384 | 7569.87 | 3919.16 | -48.23% |
| 0 | 4 | 512 | 10113.32 | 4818.64 | -52.35% |
| 1 | 1 | 16 | 2096.33 | 2103.75 | +0.35% |
| 1 | 2 | 144 | 4739.53 | 3004.41 | -36.61% |
| 1 | 3 | 273 | 7704.52 | 3774.05 | -51.02% |
| 1 | 4 | 512 | 11485.34 | 5564.98 | -51.55% |
| 5 | 1 | 1 | 1875.15 | 1884.11 | +0.48% |
| 5 | 2 | 2 | 3774.31 | 1917.03 | -49.21% |
| 5 | 3 | 3 | 5845.69 | 2095.16 | -64.16% |
| 5 | 4 | 4 | 8085.30 | 2270.71 | -71.92% |
| 6 | 1 | 256 | 3484.88 | 3496.70 | +0.34% |
| 6 | 2 | 512 | 6967.81 | 5166.08 | -25.86% |
| 6 | 3 | 768 | 10453.25 | 7167.23 | -31.44% |
| 6 | 4 | 1024 | 13943.29 | 9205.40 | -33.98% |

Patterns: 0 = P128 per owner; 1 = mixed 16/128/129/239 with prefixes; 5 = one token per owner with prefixes; 6 = P256 per owner. Medians are across three samples per variant. These remain component timings, with no LM head or serving loop.

## Integrated candidate

V48 frozen binary SHA256 `83cc7abb76d78360a9ae1e4df70b292d56458385b0aca0d57b0a9ab8d369dc01`; build log SHA256 `3c82ae72edd289c8ee09b89200403c6da8dc4f0a0cbb42035884532543f78bd0`. Source changes are isolated remotely and exported as `raw/packed-integration-v48/packed-v48.patch`; local application changes are preserved.

The explicit `variable-smol-v6` profile packs up to four available prefill owners, uses an iteration-budget-sized retained buffer, and dispatches up to32 decode rows. V6 wire magic/version, aggregate count, per-owner offsets and full/compact result identities are distinct from V5. The session binds the packed capability to its native reservation. Published slots precede internal partial slots; partial owners never create token outputs. Packed selection uses a null optional publication pointer instead of overwriting the legacy four-byte allocation. Existing V3/V4/V5 paths remain selectable.

Validation completed:

- 16 Rust wire tests, including every-byte authority corruption and pre-write capacity rejection.
- Native ASan/UBSan parser:19 accepted fixtures,587 rejected cases and1,091,968 single-byte memory-safety mutations.
- 36 scheduler tests, including packed selection, token budget, partial publication, NotDispatched retry and cancellation.
- 13 real-model owned-session tests; three new V6 tests check8,416 outputs with full logits or compact greedy reference comparisons. Full and alternating modes reach32 decode owners; mixed prefill publishes only completed owners. Pending-close abort and zero remaining allocations pass.
- V6 model memcheck:3 tests pass,0 errors.
- V6 CLI profile test; HTTP37 reference responses including32 concurrent requests, invalid bound and disconnect/recovery;22 ordered stochastic/fallback responses per CPU/GPU backend exactly match.

All59 integration evidence files are SHA256 verified in `raw/packed-integration-v48-manifest.json`. Archive SHA256 `8a800eedb7950f1fc5c5bb7832f34621c32ba225023eb1654c111fc984ea99d6`.

Round55 compares frozen V46 GPU16, V48 packed GPU32, and vLLM at C16/C32 on fixed and natural workloads, with two reversed-order pairs,96 warmups and384 retained requests per lane. All engines retain the same512-token budget; per-request chunks remain128 fixed/512 natural. The1,024-token capacity was correctness-tested but is not given solely to V48 in this comparison. Round55 is complete: see `V48_SERVING_RESULTS.md`. The overall goal remains unachieved.

Round55 analysis: V48 C32 throughput improves36–40% overV46, but remains14–20% belowvLLM with slowerTPOT. C16 fixed throughput beatsvLLM12.10% whileTPOT remains16.23% slower. TTFT tradeoffs versusV46 prevent a universal latency-win claim. Fresh profiling completed; see `V48_SERVING_RESULTS.md` and `MIXED_PREFILL_DECODE_V49.md`.
