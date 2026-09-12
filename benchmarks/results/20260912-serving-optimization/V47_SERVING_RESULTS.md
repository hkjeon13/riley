# V47 serving results and remaining prefill bottleneck

Decision: do not replace V46 globally. V47 is a validated experimental32-row path and improves C32 natural workloads, but the C32 short-output workload has a major TTFT regression. Keep V46 as the general measured baseline; use the wider-path evidence to guide the next combined prefill/scheduling batch. The overall vLLM objective remains incomplete.

Round54: same pinned SmolLM2-135M BF16 checkpoint/tokenizer and RTX4090, external workloads, admission/KV/token budgets. V46 uses16-row compact GPU greedy. V47 preserves16-row execution atC16 and uses32 atC32. Two reversed orders,96 warmups and384 retained requests per lane. All9,216 retained requests complete with zero failures; all6,144 Riley references match. vLLM references are not all identical; no cross-engine numerical/quality-equivalence claim. The fixed workload is P128 with mixed requested output lengths, including8/16/32, not exclusively O32.

| Condition | V46 tok/s | V47 tok/s | vLLM tok/s | V47 vs V46 |
|---|---:|---:|---:|---:|
| c16-fixed | 4418.4 | 4407.9 | 5023.9 | -0.24% |
| c16-natural | 6997.7 | 6965.3 | 7866.6 | -0.46% |
| c32-fixed | 4418.5 | 4432.8 | 7433.7 | +0.32% |
| c32-natural | 6739.2 | 8515.4 | 11753.2 | +26.36% |

| Condition | TTFT V46→V47 ms | TPOT V46→V47 ms | P99 V46→V47 ms |
|---|---:|---:|---:|
| c16-fixed | 6.354→6.454 | 3.390→3.381 | 132.26→129.42 |
| c16-natural | 6.989→7.018 | 2.170→2.175 | 311.95→309.40 |
| c32-fixed | 7.997→59.466 | 7.165→4.275 | 253.57→232.47 |
| c32-natural | 5.827→6.091 | 4.647→3.588 | 708.84→566.54 |

C32 natural throughput improves26.36%, TPOT22.79%, P95 21.91% and P99 20.07%; TTFT increases4.52% while remaining below vLLM. V47 still trails vLLM throughput27.55%, with worse TPOT and tails. C32 fixed throughput changes only+0.32%, while TTFT grows7.997→59.466ms (+643.62%). Its lower TPOT cannot justify global acceptance. C16 remains on the16-row path, with throughput−0.24/−0.46%; small latency changes are mixed, not proof of zero regression.

Fresh profile evidence: natural workloads96 requests atC16 andC32, plus fixed workloads96 requests each onV46/V47 atC32. All384 profiler reference checks pass. These traces are diagnostic, not serving timing. Natural C32 V47 has96 prefill submissions and317 decode submissions, versus499 decode submissions in the prior V46 natural trace; median GPU prefill/decode spans2881.188/1703.541us. Different trace populations and profiler overhead prevent summing kernel means or treating API durations as production CPU cost.

Fixed C32 traces each publish1,920 tokens. V46 uses138 decode submissions (13.22 mean published decode rows), V47 uses126 (14.48 rows); startup/drain are included. Prefill graph medians2853.089→2776.512us, decode1283.055→1404.433us. The wider path is often underfilled here: fewer dispatches are largely offset by more expensive individual decode steps.

Independent serving-client evidence: in both reversed orders, the median number of other requests awaiting their first token when a request starts is1 onV46 and13 onV47 for fixed P128. Natural workloads have median0 for both versions. This measures client-observed waits, not the backend queue directly. Scheduler source chooses one prefill request and alternates prefill/decode classes. The combined evidence supports investigating prefill supply and class scheduling together; it does not prove a CUDA correctness failure or justify changing RNG semantics.

Next batch: increase useful prefill work per dispatch and coordinate it with wider decode. A packed multi-owner prefill path must preserve each request’s token offsets, absolute positions, page ownership and causal attention boundaries, and publish only completed prefill rows. Couple that path with scheduler selection using actual token/queue budgets and latency protection, then repeat both short and long output workloads. Avoid a corpus-specific switch or claiming a win from the natural workload alone. Existing V3/V4 compatibility and full sampling fallback remain required.

Implementation and correctness details: `DECODE_WIDTH32_V47.md`. Frozen candidate commit `ead5c09f9cb7bd3d62ad39902e65801582ab6479`, binary SHA256 `0da2d8f74f3fb1fa7d5692fdecd6fa271519a328c6574227e94ef01b7f991090`. Source remains on the experimental candidate; frozen V46 is unchanged and remains the general baseline.

292 evidence files under `raw/serving-round54`, SHA-verified against `raw/serving-round54-manifest.json`; archive SHA256 `b034775d1c7f053a28756020240eb81957b7f52ee497c047ea0bab35f167ecde`. All benchmark/profile controllers are terminal. Blender PIDs and ports were verified stopped again; no restore helper was invoked.
