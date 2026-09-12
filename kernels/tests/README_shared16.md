# Native sixteen-row decode prototype

The four `decode_shared16*.cuh` headers are independently validated native building blocks. Production V3 still dispatches eight rows; these headers are not connected to its graph recorder, Rust wire, or server. Do not claim serving speed from these tests.

`shared16_model_probe.cu` compares graph-replayed hidden states, the entire K/V allocation, all 49152 logits, greedy output and inactive completion rows against the existing single-request model path. It runs active 1,2,4,8,9,15,16 for eight steps each with mixed real-prefill contexts and disjoint permuted pages, both original and packed weights. The 16-row head uses a zero-workspace, no-split reduction plan; the validated environment selected algorithm21 for M1 and M16.

`shared16_primitive_probe.cu` checks original/packed projection recurrences and attention across active1..16, invalid0/17, inactive guards and divergent contexts through4096.

Build with nvcc C++17, `-arch=sm_89 -O3 -Ikernels/src`; the model probe also links `-lcublasLt`. Pass the materialized loaded-RoPE fixture directory (`weights.bin`, `rope.bin`, `requests.bin`) to the model probe; add the second argument `packed` to exercise packed projections. Run compute-sanitizer memcheck and racecheck. These probes require the same pinned CUDA/private-driver environment as the serving campaign.

## Remaining serving integration

- The Rust V4 wire and native structural parser now exist (see `README_variable_wire16.md`); the graph recorder is still pending. Retain the eight-row baseline. Canonical 16-row input needs128+16*1664+1024*4 = 30848bytes, with token offset26752. The current standalone probe deliberately uses existing prefill token offset13440 in single-row reference setup; that overlapping layout is NOT a serving wire proposal.
- Request encoding, identity/digest/max-row checks, zero unused-row validation and result capacities have V4 coverage. Runtime buffers and CPU-validated argmax slot arrays still need integration. A 16-row result is1574912bytes; pinned input/output staging and offsets must be derived, not copied from the eight-row constants.
- Bind16-row hidden/logits/head outputs, attention scratch16*9*4096*4, 16-row M16 head leases, mutable alias/extent checks, prefill token reading, compiled native wrappers and resource cleanup. Keep prefill single-request semantics until separately extended.
- Connect the scheduler's maximum decode selection to actual prepared execution width. Active16/32 admission and HTTP workers already exist, but current GPU descriptors remain8.
- Before performance adoption, run whole Rust-owned GPU32-request correctness, malformed/stale/unused-row fixtures, HTTP cancellation and mixed-context tests, then matched C4/C8/C16/C32 serving against frozen V33/V37 and tuned vLLM. Native parity alone is not sufficient.
