# Native HF model inference: A100 evidence

Work continues in `codex/hf-owned-graphs`, isolated from concurrent improvements
in the primary checkout. New GPU work uses physical GPU 2 only. The original
checkout, foreign GPU processes and shared Python/CUDA installations are preserved.
These measurements do not establish the complete architecture or performance goal.

## Explicit native execution profile

The opt-in `--inference-profile native-bf16-v1` selects the measured native
route without requiring every implementation flag at the command line:

```sh
riley serve --model /models/Qwen3-1.7B \
  --inference-profile native-bf16-v1 --max-output-tokens 128
```

The profile supplies a 1024-token context, 256-token iteration/prefill budgets,
the measured row buckets, merged QKV/gate-up projections, fused operations,
native tensor-core paged attention, recorded decode and GPU greedy sampling.
Explicit options override profile defaults and remain subject to the usual
compatibility validation. Resource bounds remain in effect; for example,
Mistral-7B requires an explicit weight bound above the normal native 8 GiB
limit, such as `--max-weight-bytes 17179869184`. The profile leaves ordinary
launches unchanged and does not enable unsupported model semantics. It is
separate from `--numerics-profile owned-hf-eager-bf16-v1` and cannot combine
with that profile's incompatible arithmetic options.

CPU server library/CLI tests pass 72/36 and CUDA-feature tests pass 72/37,
with zero failures (one library test ignored in each build). Four models,
fifteen responses each, match their explicit-fast-flag control byte for byte
in token IDs and generated text. The binary is `riley-native-profile-v26`,
SHA256 `33e3d4ef0002f3fe7582e02d5e9a9a05aaa1fee0eee2c90fc90a76f43ff9f0b0`;
the complete compiled source archive SHA256 is
`9b6f3cf6c22554d5f9ba1939f8180645ae37abcdd117bc92dcdfcfff8825675e`.

The v25 explicit-fast controls measure HF generation agreement separately:

| Model | C1 token/text matches | C8 token/text matches | C8 native tokens equal C1 |
|---|---:|---:|---:|
| Qwen3-1.7B | 14/15 | 42/45 | 45/45 |
| Mistral-7B-Instruct-v0.3, full attention | 13/15 | 39/45 | 45/45 |
| Qwen2.5-3B-Instruct | 9/15 | 27/45 | 45/45 |

The same fifteen prompts run three times at C8; these are not forty-five
unique prompts. At most 128 tokens are generated per response. No quality or
performance acceptance is inferred from these agreement counts. The owned HF
route and its exact-generation evidence remain available separately.

## Optimized BF16 route versus the HF-compatible route

The owned HF route preserves independent M1 projections and qualified CUDA
arithmetic. Its managed MLP graphs have generation agreement for SmolLM2,
Qwen3 and full-attention Mistral; see [graph qualification](owned-hf-graphs.md).
It remains slower than vLLM in the measured Smol cells.

The existing optimized route uses merged QKV/gate-up projections, batched GEMM,
native paged attention, fused residual/RMSNorm and recorded decode steps.
Numerical agreement is measured separately. On fifteen C1 Smol prompts with
up to 128 generated tokens, native and vLLM each match thirteen HF responses.
They diverge on different prompts and match each other on thirteen responses.
This is diagnostic generation agreement, not equivalent-quality acceptance.
Ten teacher-forced full-vocabulary rows have eight matching HF top-1 choices;
maximum HF-to-native KL is 0.004561. No acceptance tolerance is fitted to these
observations.

Two fresh-process AB/BA repeats, fixed prompts, GPU2, CPU8-15, native CUDA12.4:

| Concurrency | Native tokens/s | Modified vLLM tokens/s | Native TPOT p50 ms | Modified vLLM TPOT p50 ms |
|---|---:|---:|---:|---:|
| 1 | 574.77 | 434.08 | 1.652 | 1.479 |
| 8 | 3669.62 | 1433.91 | 1.886 | 4.114 |

These are medians across two repeats. All errors and foreign GPU process maps
are empty. C1 token latency still regresses. C8 improves throughput and every
reported TTFT/TPOT/E2E quantile in this screen. The vLLM column uses the retained
process-local event-spin diagnostic described in the evidence; it is a modified
baseline, and must not be presented as an unmodified official vLLM result.
Default vLLM controls from prior experiments remain separate. Natural-EOS
workloads, further concurrency levels, quality and other model families remain
to be accepted.

The v24 narrow-norm candidate also completed a separate C16/C32 fixed-workload
screen: two AB/BA repeats, 32 warm-up requests and 192 retained requests per
process, same GPU/CPU affinity and modified vLLM control. All errors and foreign
process maps are empty. Median metrics across repeats are:

| Concurrency | Native tokens/s | Modified vLLM tokens/s | Native TPOT p50 ms | Modified vLLM TPOT p50 ms |
|---|---:|---:|---:|---:|
| 16 | 6301.38 | 2265.85 | 2.101 | 5.186 |
| 32 | 8991.09 | 4845.40 | 2.609 | 2.911 |

Every reported TTFT/TPOT/E2E median quantile also improves in these cells.
This remains a fixed-workload screen with two repeats and separately reported
arithmetic differences, rather than full model/quality/performance acceptance.

Qwen3-1.7B was also measured through the actual native preset, using two
fresh-process AB/BA repeats, sixteen warm-up and ninety-six retained fixed
requests per process. The binary and model are SHA-bound in the raw plans;
all eight runs have zero errors and no foreign GPU process observations.

| Concurrency | Native tokens/s | Modified vLLM tokens/s | Native TPOT p50 ms | Modified vLLM TPOT p50 ms |
|---|---:|---:|---:|---:|
| 1 | 242.87 | 215.15 | 3.922 | 3.940 |
| 8 | 1391.91 | 1115.56 | 4.914 | 5.677 |

Throughput rises 12.88/24.77 percent, but C1 TPOT p95/p99 regress 0.52/7.43
percent. C1 TPOT p99 is 4.265 ms versus 3.970 ms. C8 improves all reported
median quantiles in this screen. The C1 classification prompt also differs
between engines, and output hashes do not all match at C8. These results are
not full performance or quality acceptance. A post-screen Nsight capture at
C8 records 389.02 ms total GPU kernel time; the largest GEMM kernel accounts
for 96.50 ms, attention production 33.20 ms and fused residual/RMSNorm 21.01 ms.
The diagnostic capture is retained separately from the unprofiled measurements.

## Canonical 576-column normalization

Nsight on the optimized Smol C1 route records 25.68 ms in residual/RMSNorm out
of 164.47 ms total GPU kernel time. The previous canonical kernel repeatedly
synchronizes a 256-thread reduction and reloads the stored residual.

The new BF16 kernel retains three values per thread in registers and reconstructs
the same reduction tree in warp zero. The residual, normalization and weight
multiply keep their original BF16 rounding boundaries and FP32 operation order.
Selection is limited to 576 columns and 1..256 rows; other widths, larger row
counts and FP32 use their previous implementations. HF-specific norm entry
points retain their existing selection.

Validation on GPU2:

- Seventy-two frozen pre-change canonical GPU output buffers match byte for
  byte, including residual outputs and padded-parent tails. Coverage includes
  rows 1/2/8/32/128/256, fallback row257, widths512/640 and two epsilons.
- Ten complete model vocabulary rows match the previous optimized route byte
  for byte on fixed HF teacher prefixes, including chunked 398-token prefill.
- All fifteen long-generation responses match the previous optimized route.
- Five alternating fresh-process native pairs per C1/C8 retain identical output
  hashes, zero errors and no foreign GPU processes. Median paired throughput
  gains are 2.73/2.33 percent; TPOT p50 changes -2.63/-2.59 percent. Median paired
  changes of reported TTFT/TPOT p95/p99 also improve. This is a bounded native
  comparison; it does not close the C1 vLLM gap or prove broader parity.

Immutable binary: `riley-narrow-norm-v24`, SHA256
`22b0ff4efd7a6d7549bbd89ed5302bfe3dffa12e30f3a3e9c08eb9751dfd19f1`.
The complete compiled source archive SHA256 is
`6c429eec248e27b107ffe0fe7c341d68c36a11164c8c54460020b7cdc5bc9dff`.
The change is retained in the isolated worktree with a three-file review patch,
source manifest, raw outputs and benchmark logs under `artifacts/owned-graphs-v17`.
It has not been merged over the concurrent primary-checkout changes.
