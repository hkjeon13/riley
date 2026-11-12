# Changelog

All notable changes to Riley are documented in this file.

## 0.1.0 — 2026-11-12

The first release. It covers the serving work since `0.1.0-rc2`, measured in
[docs/performance.md](docs/performance.md) and
[docs/serving-validation.md](docs/serving-validation.md). The release owner
closed the remaining long-prompt gaps (first-token time on six 4–14B models,
tail first-token time under long-prompt load) as limits of the current
structure; see [docs/goal.md](docs/goal.md).

- Qwen2.5-3B-Instruct support through a paged, two-stage attention backend,
  with tensor-core arithmetic for attention pages.
- CUDA graph capture and replay of generation steps.
- Fused kernels for residual and norm, activation, RoPE and KV-cache writes;
  one GEMM for the Q, K and V projections; GPU-side greedy token selection.
- Per-row-count GEMM algorithm selection restricted to algorithms that
  reproduce the reference algorithm's bytes, keeping output independent of
  batch composition.
- A scheduler policy that keeps steps under GEMM cost plateaus
  (`--prefill-plateau-rows`).
- A reference-exact route that reproduces Hugging Face Transformers output
  byte for byte, used as a correctness anchor.
- `riley-checkpoint`, a tool that prepares and verifies checkpoint directories.
- An opt-in [prefix cache](docs/prefix-cache.md) (`--prefix-cache-blocks`):
  requests with the same leading tokens share read-only prompt KV blocks, and
  `usage.prompt_tokens_details.cached_tokens` reports the reuse. Outputs are
  unchanged.
- `logprobs` summarized on the GPU: one kernel per step reduces each output
  row to its maximum, normalizer partial sums and top tokens, and the CPU
  finishes the same bytes from them. Qwen2.5-7B with `top_logprobs: 5` went
  from 358 to 626 tok/s at 8 clients and 588 to 2,186 at 32.
- Prompt steps of up to 1,024 rows: wide GEMM shapes keep the anchor's bytes
  (new A100 equivalence-table rows, workspace cap scaled above 256 rows), and
  `fixed-tile-unsplit-v1` accepts a budget of 1,024. With the burst options a
  long-context server reaches the first token of a long prompt 11–26 % sooner.
  Steps above 256 rows are not recorded as graphs.
- The Qwen3 head norm with RoPE and the KV write runs one warp per head with
  the same bytes, 2–3 % off a long prompt's first token on Qwen3 models.
- `fixed-tile-unsplit-v1` sizes the attention workspace to its planner instead
  of the two-stage layout (7 GB less on Qwen2.5-7B with 1,024-row steps), and a
  step it cannot plan fails instead of falling back to kernels with other
  bytes. Log-probability buffers are reserved at start-up.
- `--attention-kv-chunk-tokens` defaults to 512 (was 1,024): decoding rows at
  long context get more parallel work. This changes outputs for rows longer
  than 512 tokens; batch invariance is unchanged. On models with two KV heads
  or fewer (Qwen2.5 0.5B–3B, Qwen2.5-Coder 1.5B and 3B,
  DeepSeek-R1-Distill-Qwen-1.5B) generated tokens split at 256 while prompt
  tokens keep 512: the planner takes each request's split from the row kinds
  the prompt-row producer already receives. This changes those models' outputs
  for generated tokens past position 256.
- Prompt tokenization merges BPE pairs by token ID through a hashed pair
  table instead of string keys in ordered maps, with the same merge order and
  the same IDs. Qwen prompts tokenize about 8× faster: on Qwen3-0.6B a
  2,600-token prompt's first token arrives in 33.3 ms instead of 43.5 ms.
- With `--prefill-burst-min-prompt-tokens`, a long prompt that has started
  keeps the whole budget for its last chunk instead of finishing in steady
  steps (about 1 % off a long prompt's first token).
- Benchmarks: `gsm8k.py`, `logprob_parity.py`, shared-prefix and closed c1/c16
  cells and `--tp` in `compare.py`, the `wide` mode of `gemm_anchor_search`.
- Documentation rewritten for users; development records were removed from the
  tree and summarised in [docs/HISTORY.md](docs/HISTORY.md).

## 0.1.0-rc2 — 2026-08-27

This candidate supersedes `0.1.0-rc1` and fixes a mixed-stage output-routing
correctness defect. When completing-prefill and decode work shared an
iteration, output slots could be sampled in stage order and the resulting
tokens associated with the wrong requests. Riley now canonicalizes dense
output slots before sampling and covers the mixed-stage ordering with a
regression test.

The release owner accepted a previously completed soak and authorized this
prerelease without rerunning the 7-hour-15-minute soak on this candidate. The
release gate (correctness, Python-free packaging, performance regression and
soak on one candidate, called Gate E below) therefore remains unpassed for
`0.1.0-rc2`; this is an explicit owner-approved prerelease, not a relaxation
of the gate.

## 0.1.0-rc1 — 2026-08-27

> **Superseded by `0.1.0-rc2`.** RC1 contains a known mixed-stage
> output-routing correctness defect, observed in the `C=5` and `C=8`
> P128/O32 lanes under the default 512-token budget, and must not be used.

Riley's first source release candidate establishes a Rust-native, CUDA-first
LLM inference engine with:

- bounded Rust workspace ownership and a narrow CUDA C ABI;
- strict Hugging Face artifact loading and canonical Llama/Qwen model IR;
- prefill, decode, paged KV cache, sampling, continuous batching, and an
  OpenAI-compatible completions streaming boundary;
- reference and reviewed `E0` execution paths with exact fallback controls;
- Python-free production packaging and fail-closed correctness, performance,
  reliability, and extension-admission contracts; and
- the project, package, binary, environment, schema, and ABI rename from the
  development name `rustinfer` to Riley (`riley`).

### Release-owner qualification exception

The release owner accepted a previously completed soak run and explicitly
authorized this prerelease without rerunning the full candidate-bound soak.
That prior run is not cryptographically bound to the final Riley source,
binary, and v2 raw-evidence contract. Consequently, `0.1.0-rc1` is an
owner-approved prerelease and must not be represented as a passed Gate E final
candidate. The fail-closed gate remains unchanged for later qualification.
