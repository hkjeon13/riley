# Prefix cache

`--prefix-cache-blocks N` lets requests that start with the same tokens reuse
the prompt KV blocks an earlier request already computed. It is off by default.

A server with a long shared system prompt skips most of the prefill work. On
Qwen2.5-7B with 448 of 477 prompt tokens shared, 32 clients went from 765 to
2156 tok/s and TTFT p50 from 498 to 42 ms, with every output unchanged. When no
prefix repeats, the cache costs nothing measurable.

## Using it

```bash
./target/release/riley serve \
  --model ./Qwen2.5-7B-Instruct \
  --inference-profile native-bf16-varlen-v1 \
  --prefix-cache-blocks 4096
```

- Each block holds 16 tokens, so N × 16 tokens stay available.
- Without `--kv-blocks`, N is added to the default pool, so running requests
  keep their full space. With an explicit `--kv-blocks`, the cache lives inside
  that pool and takes only what requests leave free.
- It needs `--decode-attention-producer tensor-core-v1`, which both
  `--inference-profile` values set. Other routes are rejected at startup because
  their output has not been shown to be independent of how a prompt is split.
- Responses report the reused tokens in `usage.prompt_tokens_details.cached_tokens`
  ([API](api.md)).

Size N to hold every prefix you want kept, not one. Eight prefixes of 448 tokens
need about 232 blocks: in the eviction test below, 128 blocks served only about
half the shared tokens.

Memory per block is `layers × 2 × KV heads × head size × 16 × 2 bytes` for
BF16. For Qwen2.5-7B that is about 0.9 MB (28 × 2 × 4 × 128 × 16 × 2), so 4096
blocks take about 3.6 GB of GPU memory. This figure is computed, not measured.

## What is guaranteed

**Output does not change.** Reused blocks hold the bytes the request would have
computed. This rests on prefill being independent of its split, which was
checked first: 23 of 23 prompts were identical with prefill chunks of 256, 16
and 48 tokens on Qwen3-4B and Qwen2.5-7B. Every measured cell then compared the
outputs with the cache on and off, and all 192 of 192 matched in all 20 cells.

**Prefixes never match by accident.** The index is a trie keyed by the parent
entry and the exact 16 token IDs. There is no hash to collide.

**Admission never fails because of the cache.** Blocks held by requests stay
within their promises, and promises stay within the pool, so evicting every
unheld block always makes enough room.

**A device failure cannot leak a stale prefix.** If the device stops with
unknown memory contents, the index is cleared. Requests that still hold blocks
keep them, and none can link new entries to the old ones.

## How it works

KV memory is already [paged](architecture.md) into 16-token blocks behind a
block table, so sharing is a table entry and a reference count. Nothing is
copied.

- **Only full prompt blocks are shared.** The tail of a prompt stays private.
- **Decode-written blocks are not cached.** Decode uses different attention
  kernels, so its bits could differ from a prefill of the same tokens.
- **The last prompt token always runs**, because the model has to produce the
  first output logits. At most floor((length − 1) / 16) blocks are reused.
- **Adoption** happens at plan time, before a request has started prefill, and
  is retried each plan while the request is waiting.
- **Publication** happens after each committed prefill step. It stops at the
  first key another request has already listed.
- **Eviction** is LRU over entries with no children and no reader. Readers hold
  their whole ancestor path, so an evictable entry exists whenever an unheld one
  does. It runs only when a plan needs blocks.

Shared blocks are read-only. The block alias checks in the plan, the batch
builder and the CUDA layer now forbid a written block from aliasing anything,
while read-only blocks may repeat across rows.

The `allocated_kv_blocks` gauge excludes cached blocks, so an idle server still
reads zero. Cached blocks have their own gauge, `prefix_cache_blocks`.

## Limits

- **Clients can observe each other.** There is no per-tenant salt. A client can
  tell from timing or from `cached_tokens` that someone recently sent the same
  prefix. Enable it only among clients that may share that knowledge.
- **Identical requests that arrive together do not share.** Each block has to be
  committed before others can adopt it, so a burst of cold requests all run
  prefill. Later requests hit.
- **Only the prompt is cached.** A chat turn does not reuse the previous turn's
  generated tokens, because those blocks were written by decode.
- **Tested for correctness, not endurance.** The cache has not been through the
  30-minute soak, and device failure is covered by the scheduler simulation
  rather than on real hardware.

## Measurements

A100, GPUs 6 and 7, 2026-11-02. 32 warm-up plus 192 measured requests, 32 output
tokens. Prompts average 477 tokens, of which 448 are shared in `one` and 436 in
`eight`, so these are close to the best case. All cells ran with no errors.
Evidence is under `evidence/pfx-*` on the host.

### Effect of the cache, Qwen2.5-7B, one shared prefix

| Clients | tok/s off → on | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---:|---:|---:|---:|---:|
| 1 | 78.3 → 83.6 | 42.1 → 16.1 | 42.3 → 16.3 | 11.82 → 11.81 |
| 8 | 470 → 615 | 51.1 → 39.4 | 195 → 57.8 | 15.46 → 12.06 |
| 32 | 765 → 2156 | 498 → 41.7 | 1104 → 115 | 25.04 → 13.53 |

### Against vLLM, both caches on, one shared prefix

| Clients | tok/s Riley / vLLM | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---:|---:|---:|---:|---:|
| 1 | **83.6** / 81.5 | **16.1** / 20.3 | **16.3** / 22.3 | **11.81** / 12.00 |
| 8 | **615** / 609 | **39.4** / 41.9 | 57.8 / **49.7** | **12.06** / 12.16 |
| 32 | **2156** / 2026 | **41.7** / 59.5 | 115 / **102** | **13.53** / 13.87 |

Riley is ahead except on TTFT p95 at 8 and 32 clients.

Remeasured on 2026-11-04 with `benchmarks/live/compare.py` (`shared-c8`,
`shared-c32`: a 330-word system document shared by every request, 20-word
user prompts, 32 output tokens; `--prefix-cache 4096` turns on both caches; vLLM
at `--gpu-memory-utilization 0.8`). Qwen2.5-7B, two runs each on GPU 7:

| Cell | tok/s Riley / vLLM | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| shared-c8 | **630** / 596 | **30.1** / 46.3 | **30.8** / 47.2 | **12.0** / 12.2 |
| shared-c32 | 2096 / **2127** | 94.3 / **62.0** | 105.4 / **67.9** | **12.8** / 13.4 |

At 32 clients the harness runs after `shared-c8`, so every request hits the
cache from the start. All 32 then finish their fixed 32 tokens in the same step
and send their next prompts together: about 1,120 uncached tokens at once. The
first token waits for that whole wave. Riley needs about 90 ms for it (steps of
256, 512 or 1,024 rows all give 95–99 ms), vLLM 62 ms in one step of up to
2,048 tokens. With a cold cache the first misses break the lockstep, and the
same server gave 37.3 / 55.0 ms. Without the cache Riley's shared-c32 TTFT p50
is 378 ms.

Most of that wave was tokenization, not the prompt steps. An engine with no
active request tokenizes a submitted prompt itself, so the 32 prompts of the
wave (about 460 tokens each) were tokenized one after another at about 4.5 µs
a token, some 65 ms. With tokens merged by ID (2026-11-05, build `lp13`, same
token IDs) the same cells on GPU 6 (`shared13/`):

| Cell | tok/s Riley / vLLM | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| shared-c8 | **627** / 596 | **30.5** / 45.7 | **31.0** / 47.0 | **12.1** / 12.2 |
| shared-c32 | **2377** / 2119 | **26.2** / 57.2 | **33.2** / 72.8 | **12.9** / 13.5 |

vLLM 0.27.1 changes some
of its own outputs when its cache is on: 170, 158 and 161 of 192 matched its
cache-off run for 1, 8 and 32 clients. Why was not investigated.

### Other workloads, Qwen2.5-7B

| Workload | Clients | tok/s off → on | TTFT p50 ms | TTFT p95 ms | Cached tokens |
|---|---:|---:|---:|---:|---:|
| 8 prefixes, 4096 blocks | 1 | 78.3 → 83.7 | 41.7 → 16.0 | 42.3 → 16.4 | 436 |
| 8 prefixes, 4096 blocks | 8 | 460 → 615 | 64.0 → 39.9 | 185 → 41.9 | 436 |
| 8 prefixes, 4096 blocks | 32 | 762 → 2103 | 445 → 39.2 | 1176 → 138 | 436 |
| 8 prefixes, 128 blocks | 8 | 462 → 534 | 63.8 → 43.9 | 183 → 78.8 | 230 |
| 8 prefixes, 128 blocks | 32 | 770 → 1425 | 444 → 39.8 | 1125 → 484 | 256 |
| No shared prefix | 1 | 78.2 → 78.1 | 41.8 → 41.8 | 42.4 → 42.5 | 0 |
| No shared prefix | 8 | 461 → 461 | 64.1 → 64.1 | 184 → 185 | 0 |
| No shared prefix | 32 | 775 → 775 | 440 → 440 | 1120 → 1123 | 0 |
| Two GPUs (6,7), one prefix | 8 | 650 → 889 | 37.9 → 35.2 | 151 → 42.4 | 448 |
| Two GPUs (6,7), one prefix | 32 | 1021 → 3024 | 406 → 29.9 | 848 → 122 | 448 |

### Other models, 8 clients, one shared prefix

| Model | tok/s off → on | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 2749 → 2863 | 12.5 → 23.1 | 34.6 → 30.0 | 2.50 → 2.08 |
| Llama-3.2-3B-Instruct | 833 → 1144 | 29.1 → 16.4 | 66.4 → 17.3 | 8.76 → 6.65 |
| Qwen3-4B | 670 → 867 | 36.0 → 34.0 | 138 → 52.5 | 10.84 → 8.31 |
| Qwen2.5-14B-Instruct | 236 → 320 | 102 → 59.3 | 393 → 67.2 | 30.82 → 23.79 |

TTFT p50 gets worse on Qwen2.5-0.5B, the one regression measured. With the cache
on, the eight closed-loop clients finish and resubmit together: 88 % of request
starts fell within 2 ms of the previous one, against 5 % with it off. So a
request waits behind the others in its group, where off it usually arrived
alone. That the lockstep alone explains the rise was not tested separately. It
should not apply to independently arriving traffic.
