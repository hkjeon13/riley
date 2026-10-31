# Tensor parallelism design (2026-10-30)

Riley serves one model on one GPU. A BF16 checkpoint above ~16 B parameters
does not fit an A100-40GB (Qwen2.5-32B-Instruct is ~65.5 GB of weights), so
serving it needs the weights split across GPUs. This document fixes the design
before code lands; phases are listed at the end.

## Goals and constraints

- Serve dense Llama / Qwen2 / Qwen3 checkpoints across 2, 4 or 8 GPUs of one
  host connected by NVLink, in one process, through the same HTTP API.
- Keep Riley's numerical contract: a request's tokens do not depend on what
  else is in the batch (batch invariance), and are identical across repeats.
  Tokens may differ between TP sizes (the reduction order changes), so TP=N is
  its own numerics configuration, verified like a new GPU.
- Keep the native profile's speed features: merged QKV and gate|up, fused
  RoPE/KV write, recorded decode-step graphs, GPU greedy sampling.
- No Python, and no new mandatory dependency for single-GPU builds.

## Partitioning (Megatron style)

Per layer, with TP size `T` and rank `r`:

| Weight | Split | Rank holds |
|---|---|---|
| Q / K / V (merged `[Q;K;V]`) | output (column) | heads `r*H/T ..`, KV heads `r*KV/T ..` → `[Q_r;K_r;V_r]` |
| Q/K norm (Qwen3), QKV bias | follows its heads | rank's slice |
| o_proj | input (row) | columns of its heads; output is a partial sum |
| gate / up (merged) | output (column) | `[gate_r;up_r]`, intermediate `I/T` |
| down | input (row) | rows of its intermediate slice; partial sum |
| embedding, norms, LM head | replicated | full copy |

Two all-reduces per layer: after o_proj (before its bias, which is added once)
and after down (before the fused residual + next-layer norm). Attention needs
no communication because each rank owns whole KV heads. Requirements: query
heads, KV heads and the intermediate size divisible by `T`; the GQA ratio is
unchanged, so the existing attention kernels apply with per-rank head counts.

The replicated LM head costs `vocab × hidden × 2` bytes per GPU (1.5 GB for a
5120-hidden Qwen) and keeps sampling local; vocabulary-parallel LM head is a
later optimisation.

## All-reduce: deterministic one-shot over peer access

NCCL chooses ring/tree algorithms and protocols by message size, so the
summation order of an element can change with the number of batch rows. That
would break batch invariance. Riley instead uses its own all-reduce:

- At start-up every rank allocates a registered input buffer and a signal
  array, and enables peer access to every other rank (NVLink, one process).
- A step's reduce: each rank copies its partial sums (the GEMM output) into
  its own registered staging buffer with a device-to-device copy; a kernel sets this rank's flag in every peer's
  signal array and waits for theirs (epoch counter, no host round trip); then
  every rank computes `out[i] = p_0[i] + p_1[i] + ... + p_{T-1}[i]` in FP32,
  always in rank order, rounds to BF16, and a second barrier keeps the
  buffers alive until all peers have read them.
- The order of terms depends only on `T`, never on rows or batch, so
  batch invariance holds, and every rank gets bit-identical results.
- The kernels are ordinary launches on the rank's stream, so they can be
  recorded in the decode-step graphs. The reduced extent is the padded
  bucket (`dense_rows × hidden`), which is fixed for a recorded shape.
- Decode messages are small (`rows × hidden × 2` bytes: 320 KB for 32 rows
  of a 5120-hidden model), where one-shot peer reads over NVLink beat ring
  algorithms. Prefill chunks are at most 256 rows (2.6 MB); a two-shot
  (reduce-scatter + all-gather) variant is the planned optimisation if the
  one-shot path shows up in prefill profiles.

## Execution model

- One host thread per rank, each with its own CUDA context, stream, cuBLASLt
  plans and recorded graphs (the native handles are `!Sync`, and the step
  capture state is thread-local).
- Rank 0 owns the scheduler, KV block tables and HTTP side. Each iteration it
  broadcasts the same iteration command (rows, positions, block tables, the
  capture key) to the other ranks over channels; all ranks execute the same
  `execute_fixed_graph` with their own weights and KV cache, so they record
  and replay the same graph keys in lockstep.
- Only rank 0 downloads sampled tokens. KV caches are per rank with
  `KV/T` heads each, so the same `--kv-blocks` budget frees memory on every
  rank, and the block tables are identical.
- Weight upload slices each tensor on the host before the pinned staging
  copy; the existing merged QKV / gate|up builders run on the sliced pieces.
  `LlamaDimensions` gets per-rank head and intermediate counts.

## CLI

`riley serve --tensor-parallel-devices 4,5 ...` lists 2–8 distinct CUDA
ordinals in rank order (rank 0 first) and excludes `--device`; without it the
server runs on one GPU exactly as before. Start-up fails if peer access
between any pair is unavailable or a dimension is not divisible.

## Phases

1. **All-reduce primitive**: native kernels + FFI + Rust wrapper, GPU tests
   on 2 and 4 GPUs: bit-identical results on all ranks, rank-order sums equal
   a host reference, results independent of row count, works inside a
   recorded graph.
2. **Sharded weights and per-rank dimensions**: slicing at upload, per-rank
   `LlamaDimensions`, validation errors for non-divisible shapes.
3. **Multi-rank executor**: rank threads, command broadcast, the two
   all-reduce sites in `execute_fixed_graph`, recorded graphs per rank.
4. **Server**: `--tensor-parallel` / `--devices`, start-up checks.
5. **Verification and measurement**: token agreement with HF for TP=2
   against TP=1 on a 14B model, batch invariance at C1/C8/C32, then
   Qwen2.5-32B-Instruct on 2 GPUs against vLLM `--tensor-parallel-size 2`
   (8 cells, same procedure as the single-GPU tables).

## Status (2026-10-31)

- Phases 1–4 are implemented: `kernels/src/peer_all_reduce.cu` (group) and
  `peer_all_reduce_bf16_kernel` in `kernels/src/primitives.cu`,
  `riley_cuda::CudaPeerAllReduceGroup`, `TensorParallelShard` slicing in
  `riley-runtime/src/cuda_weights.rs`, `ModelSpec::tensor_parallel_rank_spec`,
  the two reduce sites in `executor/dispatch.rs`, follower ranks in
  `llama/tensor_parallel.rs`, and `--tensor-parallel-devices`.
- Verified on A100 (GPUs 4 and 6): the all-reduce GPU test (rank-ordered sums
  bit-identical on both ranks, independent of row count, 20 repeats);
  Qwen2.5-0.5B at TP=2 gives the same tokens as one GPU on a smoke prompt;
  Qwen3-8B at TP=2 matches HF BF16 on 14 of 15 fixture prompts (one GPU: 13)
  with identical outputs at concurrency 1, 8 and 32, recorded decode steps
  included.
- Qwen2.5-32B-Instruct at TP=2 (GPUs 6 and 7, NVLink) against vLLM 0.27.1
  `--tensor-parallel-size 2 --gpu-memory-utilization 0.9 --generation-config
  vllm`, same 8-cell procedure as the single-GPU tables
  (`evidence/tp2-q25-32b-gemm4-attempt01`, 2 repeats; Riley is eager, vLLM
  uses its default CUDA graphs):

  | cell | tok/s Riley | tok/s vLLM | TPOT p50 Riley | TPOT p50 vLLM | TTFT p50 Riley | TTFT p50 vLLM |
  |---|---:|---:|---:|---:|---:|---:|
  | c1-fixed | 33.6 | 34.6 | 29.19 | 28.00 | 39.7 | 45.5 |
  | c1-natural | 33.5 | 35.2 | 29.23 | 28.00 | 40.5 | 45.5 |
  | c8-fixed | 214.2 | 216.5 | 32.66 | 31.17 | 99.1 | 127.1 |
  | c8-natural | 237.2 | 246.1 | 31.60 | 30.24 | 102.5 | 132.1 |
  | c16-fixed | 366.7 | 353.2 | 37.41 | 37.39 | 110.3 | 144.3 |
  | c16-natural | 426.5 | 429.0 | 33.96 | 33.58 | 109.4 | 149.6 |
  | c32-fixed | 549.7 | 512.4 | 46.39 | 49.96 | 141.0 | 202.6 |
  | c32-natural | 702.2 | 685.6 | 38.58 | 39.24 | 138.7 | 244.6 |

  Times are ms. TTFT is lower in every cell and C32 is ahead; C1 and C8 are
  1–5 % behind in throughput and 4–5 % behind in TPOT, so the mandatory
  criterion is not met. Outputs
  are identical across repeats and concurrency in every cell.
- GEMM equivalence-table entries for the per-rank shapes 5120→3584 (merged
  QKV) and 2560→5120 (o_proj) were added from `gemm-equiv-a100-attempt04`.
  The per-rank gate|up and down shapes kept the anchor (gain under 2 %).
  Against the previous binary the 32 verification prompts give identical
  tokens at C1 and C8 and C1 is 1.1 % faster (33.9 → 34.3 tok/s).
- Rejected: fusing the staging copy into the reduce kernel. At M=1 the copy
  then runs on only two blocks and needs a system-scope fence, and C1 got
  slower (33.75 tok/s).
- Next candidate: drop the end barrier with double-buffered staging. The
  buffer parity must follow a global call counter and the waits must accept
  `>= epoch`, so a fast rank can enter the next reduce without overwriting a
  buffer a peer is still reading.
- `--execution-graph-policy` defaults to disabled, so these runs (like the
  single-GPU tables) are eager on the Riley side.
- Not yet done: 4-rank runs, a two-shot variant for prefill, and a
  vocabulary-parallel LM head.
