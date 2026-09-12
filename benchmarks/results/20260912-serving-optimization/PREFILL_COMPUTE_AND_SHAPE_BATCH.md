# Prefill compute and shape batch

Status: profiling and source-contract audit complete; implementation pending. Previous goal turn was progress: round27 serving evidence and restoration were verified. Full serving objective remains active.

## Evidence and choice

Fresh analysis of the frozen V10 Nsight trace finds 48 prefill submissions with181.742ms summed kernel time. Custom inline-MMA projection kernels account for141.549ms (77.884%). Down projection accounts for44.393ms; gate/up39.638ms; K/V26.551ms; Q15.776ms; O15.191ms. Attention25.377ms is secondary. The prior name-based dense_gemm_ns metric includes custom kernels when applied to prefill; it must not be interpreted as cuBLAS time. This is nonexclusive diagnostic timing, not new serving throughput evidence.

Source audit: kernels/src/graph_numerics_precise.cu implements M16 tiles with128 rows, position==127, K16 recurrence and explicit intermediate BF16 chunk rounding. Q/K/V use interval192, O128, down320, gate/up0. Replacing these with unrestricted cuBLAS changes the numerical contract. Prior packed-load Q/O regression is documented in the launcher; do not repeat that change blindly.

Fixed geometry spans more than a launch dimension: graph_resources.cu validates rows1/128, position127 or decode>=128, context<160, metadata token extension and 128-row scratch sizes; graph_decode_multi_session.rs dispatches Stage::Prefill128; server admission caps context160/output32. A CLI-only relaxation would create an invalid execution contract.

## Coupled implementation scope

1. Introduce a prefill projection execution path with explicit row count/stride and cached plan or tile identity. Target down and gate/up first because their combined84.031ms is46.24% of prefill kernel time. Preserve exact chunk recurrence for the existing profile; separately qualify any arithmetic-changing option using the frozen natural teacher-forced policy and FP32 reference.
2. Extend prefill staging, scratch sizing, capture selection and validation together to bounded prompt-length buckets with an actual live-token count. Correctly mask padded rows, select the final live row for logits and prohibit padded KV publication. Preserve transactional cancellation and resource ownership.
3. Extend scheduler/server shape admission and context/KV sizing with the same validated geometry, including mixed prefill/decode load. Do not merely pad every request to128 or drop unsupported samples from reported throughput.

## Verification and benchmark

Before implementation choose bucket bounds from a persisted natural request corpus and report its full length distribution. Test non-bucket-boundary lengths, page boundaries, mixed rows, cancellation/page reuse, full/greedy alternation and poisoned-owner behavior. For arithmetic changes retain exact mismatch counts and evaluate the predeclared numerical policy; a passed finite corpus is not general quality equivalence.

Benchmark the completed batch against frozen V10 and vLLM on identical supported fixed-P128 requests to isolate regression, then against vLLM on the full new variable-length request corpus. V10 unsupported variable shapes are reported as unsupported, never as zero-latency requests or silently omitted. Include C1/4/8/16 and sustained runs with output lengths beyond32 when implemented; publish failed/admitted/completed counts, achieved concurrency, throughput, TTFT/TPOT and P95/P99. Use authorized exclusive GPU lifecycle with verified exact Blender restoration. Do not claim success until broader correctness and serving gates hold.

Frozen V10 remains unchanged. No new GPU execution, driver changes or Blender pause were needed for this audit.
