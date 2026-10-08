# Native sixteen-row decode prototype

The four `decode_shared16*.cuh` headers are independently validated native building blocks. Production V3 still dispatches eight rows; these headers are not connected to its graph recorder, Rust wire, or server. Do not claim serving speed from these tests.

`shared16_model_probe.cu` compares graph-replayed hidden states, the entire K/V allocation, all 49152 logits, greedy output and inactive completion rows against the existing single-request model path. It runs active 1,2,4,8,9,15,16 for eight steps each with mixed real-prefill contexts and disjoint permuted pages, both original and packed weights. The 16-row head uses a zero-workspace, no-split reduction plan; the validated environment selected algorithm21 for M1 and M16.

`shared16_primitive_probe.cu` checks original/packed projection recurrences and attention across active1..16, invalid0/17, inactive guards and divergent contexts through4096.

Build with nvcc C++17, `-arch=sm_89 -O3 -Ikernels/src`; the model probe also links `-lcublasLt`. Pass the materialized loaded-RoPE fixture directory (`weights.bin`, `rope.bin`, `requests.bin`) to the model probe; add the second argument `packed` to exercise packed projections. Run compute-sanitizer memcheck and racecheck. These probes require the same pinned CUDA/private-driver environment as the serving campaign.

## Serving integration completed in V40

Commit `f09ef11281fbbd9a4f341c3d57c9848ecfac4e1b` connects these building blocks through V4 wire validation, graph resource ownership, M16 head, retained Rust session, sixteen-row scheduler selection and explicit `--graph-numerics variable-smol-v4`. V3 remains available and default selection is unchanged. The standalone probe keeps its historical single-reference metadata setup; V4 serving uses canonical token offset26752 and completion magic0x34524d52.

Whole Rust-owned32-request testing matched all2336 full-vocabulary outputs with maximum16 decode rows. V3 regression2336+224 outputs also matched. Final owned and native recorder memchecks reported zero errors. Native recorder covers partial prefill and active1/2/4/8/9/15/16, cold alias/head rejection, stale-output hiding and cleanup. HTTP32 mixed streaming/nonstreaming, invalid bounds and disconnect recovery passed; all allocations and KV reservations returned to zero.

Round48 measured frozen V40 against V37 and vLLM0.27.1 at client/admission4/8/16/32, fixed and natural mixed lengths, two reversed orders, each96 warmup+384 retained requests. All18432 retained requests completed; all12288 Riley references matched. V40 improves high-concurrency throughput but regresses C4/C8; it is an explicit experimental profile, not a general replacement or vLLM performance win. See the campaign status and Round48 analysis for full latency/throughput results. Further result-transfer/validation and prefill profiling is required; long-run stability remains unqualified.
