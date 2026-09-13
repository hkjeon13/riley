# Persistent attention task extraction — native gate

The attention portion of the next layer-DAG optimization batch is implemented and passes native equality and sanitizer checks. **The batch is still in progress:** full-model ownership/graph integration and the matched serving comparison are not complete. No speedup is claimed and the previous serving table remains authoritative.

## Implementation

`attention_tasks.cuh` extracts V7 score/value arithmetic into device-callable tasks with explicit logical row/head/tile indices and caller-owned per-warp probability/exponential scratch. The source SHA256 is recorded in the header and verification receipt. The original `decode_gqa_attention_v50.cuh` is unchanged and serves as the independently launched reference in the probe. Four-K16 score MMA, reverse128-token softmax traversal, lane-local FP32 denominator accumulation, BF16 rounding and value MMA order remain intact.

`persistent_attention.cuh` runs a finite cooperative score→value DAG. Each CTA constructs a request-prefix table for actual `3*ceil(context/8)` score tasks, eliminating work for inactive requests and invalid contexts. Warp workers process those tasks with a bounded grid-stride loop. One grid barrier publishes scores before value tasks. Value tasks interleave requests across resident workers instead of binding whole CTAs to one request. Each warp has its own128-entry probability and exponential buffers; a warp barrier protects their reuse.

A host plan checks cooperative-launch capability and occupancy, chooses one CTA per SM, and validates device and maximum resident block count before launching. There is no spinning device queue, Python inference path or production dispatch selection. The raw internal helper requires the native owner to validate page bounds, disjoint spans, metadata immutability and stream lifetimes. It does not independently make arbitrary page indices safe. Score storage remains the existing32×9×4096 FP32 layout.

## Native evidence

The graph contains both original V7 and candidate attention, with different score/output buffers. It is replayed with changed queries and metadata across15 cases: rows0/1/16/32/33, ragged contexts spanning1/7/8/9/15/16/17/127/128/129/511/512/513/4095/4096, an all4096 context case, and invalid4097 context. Per-request page tables use noncontiguous permutations of256 physical pages. Pages are shared read-only in this standalone test; it does not test KV writes or ownership transfer.

All1,179,648 FP32 score-storage entries and18,432 BF16 output entries per replay compare bitwise, including untouched padding. Invalid/inactive output sentinels remain intact. Oversized cooperative plans are rejected. Memcheck reports0 errors and racecheck reports0 errors/0 warnings. These checks cover the primitive only; no full-model memory/race or quality result is claimed for this new candidate.

RTX4090 SM89 runtime:128 CTAs,72 registers/thread,6,276 bytes static shared memory,0 local memory/thread. SM90a and SM100a object compilation succeeds; runtime is skipped because those GPUs are unavailable. This does not implement multi-GPU scheduling. No native performance timing was taken, so resource counts are not interpreted as a speedup.

## Next integration boundary

Connect the attention task body to the post-attention DAG with one shared owner. All value consumers must finish before the score scratch is reused for projection/down partials. Attention output must be complete before projection consumes it; all cooperative participants must reach that barrier even when their own task range is empty. Preserve the per-request metadata layout and inactive-row rules, bind the new source and task layout into the graph identity, then run full-model greedy/logit parity and native memcheck before a matched V7/prior/candidate/vLLM screen. The difference between128-CTA attention residency and the earlier48-CTA post-attention plan is a resource tradeoff to measure, not a reason to assume combining them helps.

PR07/PR19 still lack a full attention+MLP layer DAG, QKV/RoPE persistence, mixed/prefill persistence, async-ticket integration and high-concurrency/long-soak validation. Do not promote this primitive or declare the overall serving goal achieved.

## Reproduction

Compile `benchmarks/analysis/persistent_attention_probe.cu` with the pinned nvcc13.0.88 toolchain using `-std=c++17 -O3 -lineinfo -arch=sm_89`. Run the resulting probe directly and with Compute Sanitizer12.8.1 `--tool memcheck`/`--tool racecheck --error-exitcode 99`. Cross-architecture compile uses `-c -arch=sm_90a` or `-c -arch=sm_100a`. Logs and exact source/binary/object hashes are adjacent; original binaries remain at `/tmp/riley-opt-260912/persistent-attention-v1`.
