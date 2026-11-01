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
  algorithms.
- Low-latency path (2026-11-01) for reduces of at most 2^19 elements whose
  count is a multiple of 4: no staging copy and no barrier. Each rank writes
  16-byte packets `{bf16×2, epoch, bf16×2, epoch}` straight into every peer's
  receive buffer with one vector store, and the receiver spins until both
  epoch words of a packet match its own epoch. The buffers alternate by epoch
  parity. The sum order and arithmetic are the staged kernel's, so results are
  bit-identical. Safety rests on two facts. A packet index always maps to the
  same block (`index / 512 mod 64`), and that block's epoch only grows, so a
  stale packet never carries the current epoch. A rank rewrites a parity only
  after it has received the peer's packets of an intermediate call, which the
  peer sends only after its earlier kernel (and its last read of that parity)
  has finished. The path depends only on the element count, so all ranks take
  the same one. Prefill chunks are at most 256 rows (2.6 MB); a two-shot
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

## Status (2026-11-01)

- Phases 1–4 are implemented: `kernels/src/peer_all_reduce.cu` (group),
  `peer_all_reduce_bf16_kernel` and `peer_all_reduce_bf16_low_latency_kernel`
  in `kernels/src/primitives.cu`, `riley_cuda::CudaPeerAllReduceGroup`,
  `TensorParallelShard` slicing in `riley-runtime/src/cuda_weights.rs`,
  `ModelSpec::tensor_parallel_rank_spec`, the two reduce sites in
  `executor/dispatch.rs`, follower ranks in `llama/tensor_parallel.rs`, and
  `--tensor-parallel-devices`.
- Verified on A100: the all-reduce GPU tests (rank-ordered sums bit-identical
  on both ranks, independent of row count, 20 repeats, rows 1–256, both
  paths); Qwen2.5-0.5B at TP=2 gives the same tokens as one GPU on a smoke
  prompt; Qwen3-8B at TP=2 matches HF BF16 on 14 of 15 fixture prompts (one
  GPU: 13) with identical outputs at concurrency 1, 8 and 32, recorded decode
  steps included.
- The C1/C8 gap of 2026-10-31 was the all-reduce: 14.2 µs per call at one
  row, 128 calls per decode step (64 layers × 2), about 1.8 ms of a 29 ms
  step. A recorded-graph microbenchmark (`recorded_reduce_latency` in
  `crates/riley-cuda/tests/peer_all_reduce_gpu.rs`, 128 reduces replayed 50
  times, hidden 5120) gives, in µs per call:

  | rows | 1 | 8 | 32 | 64 | 100 | 256 |
  |---|---:|---:|---:|---:|---:|---:|
  | staged (before) | 14.22 | 14.87 | 16.11 | – | – | 29.05 |
  | low-latency | 3.11 | 3.98 | 7.11 | 11.13 | 15.60 | 27.01 (staged) |

  256 rows exceed 2^19 elements and stay on the staged path. Blocks of 256 and
  128 threads were slower than 512 at every size.
- Qwen2.5-32B-Instruct at TP=2 (GPUs 6 and 7, NVLink) against vLLM 0.27.1
  `--tensor-parallel-size 2 --gpu-memory-utilization 0.9 --generation-config
  vllm`, same 8-cell procedure as the single-GPU tables
  (`evidence/tp2-q25-32b-ll-attempt01`, binary `bin/riley-tp-ll`, 2 repeats;
  Riley is eager, vLLM uses its default CUDA graphs):

  | cell | tok/s Riley | tok/s vLLM | TPOT p50 Riley | TPOT p50 vLLM | TTFT p50 Riley | TTFT p50 vLLM | verdict |
  |---|---:|---:|---:|---:|---:|---:|---|
  | c1-fixed | 35.2 | 34.7 | 27.77 | 27.99 | 39.5 | 42.3 | win |
  | c1-natural | 35.2 | 35.3 | 27.79 | 28.02 | 39.5 | 41.9 | tie |
  | c8-fixed | 221.9 | 216.5 | 31.44 | 31.42 | 95.9 | 114.8 | miss (TPOT +0.1 %) |
  | c8-natural | 247.3 | 246.2 | 30.31 | 30.23 | 98.4 | 131.8 | miss (TPOT +0.3 %) |
  | c16-fixed | 375.6 | 352.9 | 36.51 | 37.44 | 107.6 | 144.6 | win |
  | c16-natural | 443.2 | 429.4 | 32.63 | 33.54 | 117.6 | 149.8 | win |
  | c32-fixed | 557.9 | 511.5 | 46.42 | 50.26 | 139.6 | 192.6 | win |
  | c32-natural | 724.4 | 687.1 | 37.61 | 39.27 | 129.4 | 237.9 | win |

  Times are ms. Against the staged binary (`tp2-q25-32b-gemm4-attempt01`) C1
  went from 33.6 to 35.2 tok/s and C1 TPOT from 29.19 to 27.77 ms. Throughput
  is now at or above vLLM in every cell and TTFT is lower in every cell. The
  mandatory criterion is still not met, by the C8 TPOT p50 alone: 0.02 and
  0.08 ms. Riley's C8 TPOT was the same in both repeats (31.44 / 31.44 and
  30.31 / 30.30); vLLM's spread was 31.38–31.46 and 30.22–30.23. Outputs are
  identical across repeats and concurrency in every cell. On the 32 GSM8K
  verification prompts the low-latency binary gives the same tokens as the
  staged one at C1 and C8 (32/32 each), C1 36.0 vs 34.3 tok/s.
- Not pursued: fusing the reduce into the following residual add + RMSNorm
  (one block per row). It would save one kernel boundary per reduce (at most
  about 0.15 ms per step), but it changes the packet-to-block mapping that the
  low-latency path's stale-packet argument relies on, and the norm's
  reduction order would have to be reproduced exactly.
- Rejected earlier: fusing the staging copy into the staged kernel (C1 got
  slower, 33.75 tok/s).
- GEMM equivalence-table entries for the per-rank shapes 5120→3584 (merged
  QKV) and 2560→5120 (o_proj) come from `gemm-equiv-a100-attempt04`.
- `--execution-graph-policy` defaults to disabled, so these runs (like the
  single-GPU tables) are eager on the Riley side.
- Not yet done: 4-rank runs, a two-shot variant for prefill, and a
  vocabulary-parallel LM head.
