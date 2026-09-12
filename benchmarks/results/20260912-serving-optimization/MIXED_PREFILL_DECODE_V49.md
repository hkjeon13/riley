# V49 mixed execution batch

Status: V49 V7 mixed serving integrated and correctness-qualified; Round56 serving comparison is running. No V49 serving performance claim yet. V48 is the next direct optimization comparison candidate; retain frozen V46 for latency context and compare to vLLM under matched serving conditions.

## Evidence and intended effect

V48 C32 fixed throughput is14.28% below vLLM and TPOT70.77% slower. In the middle80% of its96-request diagnostic trace, prefill consumes57.64% of GPU graph time (131.203ms), decode42.36% (96.431ms). Separate graph medians are3.400ms and1.522ms. Natural C16/C32 decode consumes73.13%/67.54% of graph time; scores/values attention kernels are major costs. These are node-profile observations, including instrumentation effects, not production timing.

Packing reduced fixed96-request prefill submissions to41 and decode submissions to84. The remaining scheduler policy still alternates exclusive prefill and decode iterations. A mixed token batch can potentially let existing decode owners progress during prefill work and share projections. This is a hypothesis to test, not a measured speedup. Natural workloads may still need separate decode attention improvements afterward.

## One optimization batch

1. Extend the prototype to up to32 total owners, with existing decode owners contributing one token and up to four prefill owners contributing bounded chunks. Preserve per-owner committed positions, causal attention, page ownership and partial publication. Compare the combined full model to established execution before integrating it.
2. Avoid multiplying maximum per-owner query capacity by32 in attention launch geometry. Evaluate a compact mapping of actual per-owner query tiles, including nonaligned packed offsets and one-token decode owners. Keep exact BF16 recurrence, softmax denominator ordering and nonfinite behavior. Select this change only after correctness and measured kernel results.
3. Add an explicit V7 wire/session capability for per-row stage and aggregate geometry. Full/compact result identities must bind each row's real stage. Existing V6 cannot silently change meaning. Validate all ownership, offset, capacity and publication metadata before GPU access; retain legacy paths.
4. Add scheduler mixed admission with ready decode tokens budgeted first, then available prefill chunks within the same total token/owner budget. Do not wait for a full batch. Preserve retries, cancellation, class/progress accounting, output slots and KV settlement. Capture pure decode and mixed execution with full-result sampling fallback.

## Required gates

- Full-model equality across pure decode, pure prefill and mixed boundaries; one-token and long-prefix owners; total512/1024 token boundaries; partial owners; physical-page permutations; inactive guards; memcheck/racecheck.
- Packet corruption and retained capability mismatch rejection; scheduler NotDispatched retry, cancellation, zero-output partial batches and post-dispatch abort cleanup.
- Concurrent HTTP, stream disconnect/recovery and stochastic fallback parity.
- Matched V48/V49/vLLM C16/C32 fixed/natural serving comparison with repeated orders, followed by broader concurrency and stability qualification when competitive performance is established.

Do not replace the serving success criterion with a microbenchmark win. If mixed execution regresses sustained decode or tails, preserve the evidence and revise the batch. Blender remains stopped; no restoration helpers are authorized or needed.

## Prototype evidence and revision

The first compact attention dispatch uses per-owner tile offsets and a persistent CTA loop. It passes242 configurations with finite/NaN/Inf values, causal prefixes, inactive guards and owner counts1/4/8/16/32 under memcheck and racecheck. An additional correctness/memcheck run resets the output guard before each64/128/256-CTA variant, preventing earlier output from hiding missing writes. Original logs are retained.

Thirty full-model geometry cases pass selected-hidden and entire-KV equality plus memcheck using the current serving loader weights/RoPE;11,325,726,720 bytes are compared per run. This uses the established sequential prefill arithmetic as reference for mixed chunk sizes, including one-token rows with populated prefixes. It does not yet qualify the actual decode-route protocol, LM head, V7 authority, scheduler or HTTP.

The persistent approach regresses with many small prefill chunks: at32 owners×31 tokens,128 CTAs take462.49us versus294.14us for a hypothetical rectangular32 launch in the follow-up measurement. It is not selected as a general replacement.

A direct tile→owner/local-query mapping eliminates the persistent loop and repeated owner search. In the same attention-only measurement it takes290.45us for32×31 tokens, and29.85us versus41.66us for one128-token owner plus31 one-token owners. Across the six32-owner patterns it is1.25–28.34% shorter than the hypothetical rectangular32 launch. That comparator is not existing V48 serving, which supports at most four prefill owners. At four owners, the mapped variant has both gains and regressions (approximately−2.93% to+5.93%), so do not claim an across-the-board improvement or replace the small-owner path without serving evidence.

Timing uses synthetic Q/K/V, four reversed-order pairs,10 warmups and100 graph replays. Capacity is512 unless total tokens exceed512, then1024. Original timing with a fixed1024 capacity/rectangular32 geometry is retained as preliminary; `guards-timing.log` and the mapped `timing.log` are the analyzed comparisons.

The selected mapped model prototype stores its map after the token slab (offset57472, total extent61568), avoiding overlap with input tokens. Its30 full-model correctness cases have also passed remotely; mapped full-model memcheck and mapped attention racecheck are still running in controller session21123. These ongoing checks are excluded from the first export.

All40 completed prototype evidence files are SHA256 verified under `raw/prototypes-v49-manifest.json`; archive SHA256 `fa51f5a07343a5e3ef74187d4736c65b98ffc8cbef64a872c38432607d69c18f`. Remote application source remains clean V48 `4195aa5dd08e5ebbc604c89a09ae8831fc339e0d`; prototype files are outside that checkout. Next complete mapped qualification, then implement the V7 mixed reservation/session and scheduler batch and compare real serving to V48/vLLM.

## Integrated V49 candidate

V49 source `c63a395cdbaea050b8281aa58bfb0f9ac681bb14`; frozen binary SHA256 `3e822d5171511b95a8be77bbd104b3dc7b4ef52395a5e0f39a449fe4ea3a60e9`; build log SHA256 `71692762cf196fb1112b6cfcee82be8a9ce06e7f3dd965a89b872cfa291af4d2`.

The explicit `variable-smol-v7` profile reserves ready decode tokens first, then up to four prefill chunks in the remaining iteration budget. It permits up to32 total owners. V7 retains the token slab and appends a canonical tile map, with request extent61568, per-row stage, per-row token/tile offsets and global stage0/1/2 for prefill/decode/mixed. CPU live authority and the native parser reject mismatched stages, ownership, slots, maps and extents. Session capability is bound to the recorded graph. Full and compact results carry V7 magic and each row's actual stage. Existing V6 remains separately selectable.

Mapped attention uses direct tile lookup; its full-model30-case memcheck and242-case attention racecheck now pass. The retained production graph allows the optional publication pointer to be null and selects32 hidden rows safely. The model's map begins after the token array, avoiding the attention-only prototype's different map placement.

Integrated validation:18 Rust wire tests,37 scheduler tests,5 native cross-language fixtures with473 rejection checks and307,840 byte mutations under ASan/UBSan;16 real-model owned-session tests including three new V7 tests with8,416 checked outputs, actual mixed iterations, full/compact switching, partial publication and close/abort cleanup; three V7 model tests under memcheck with0 errors. V7 CLI passes. HTTP37 reference responses include32 simultaneous requests and stream disconnect/recovery. CPU/GPU sampling fallback matches22 ordered responses per backend including stochastic generation.

All64 integration evidence files are SHA256 verified in `raw/integration-v49-manifest.json`; archive SHA256 `6bd9bbe1ecca52f494d4d351d83019f208c9ac58eeb06b1750a7aaa930708524`. Local application changes remain untouched; isolated remote source is committed and clean.

Round56 compares frozenV48/V49/vLLM at C16/C32 on fixed/natural workloads with the same512-token budget,96 warmups and384 retained requests per lane, two reversed orders. V49 is an experimental candidate; no default promotion, long-run stability qualification or overall goal completion is claimed.


## Round56 및 V49 trace 완료

24 lane,9,216 요청 실패0, Riley6,144 기준 일치. V48 대비 처리량 +2.43~10.05%이나 vLLM 대비 TPOT 목표 미달이다. 별도288 요청 Nsight trace도 기준 일치로 완료했다. [최신 결과와 다음 후보](V49_SERVING_RESULTS.md)를 따른다. Blender는 종료 상태를 유지한다.
