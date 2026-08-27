# Riley 0.1.0-rc1

**Riley — a Rust-native LLM inference engine.**

Riley (Rust Inference LLM Engine) is a Rust-native, CUDA-first LLM inference
engine focused on low-latency execution, predictable memory management, and
reusable model execution patterns.

This first release candidate is a source prerelease for Linux x86_64, CUDA
12.8, one NVIDIA `sm_89` GPU, and the pinned SmolLM2 release lane. It also
contains the previously validated dense Qwen compatibility path. Crates remain
`publish = false`; this release does not publish packages to crates.io.

## Highlights

- Rust-owned server, scheduler, model runtime, tensor metadata, and KV cache.
- Native CUDA C ABI with cuBLASLt and reviewed custom kernels.
- Strict safetensors, tokenizer, checkpoint provenance, and canonical IR.
- Llama and Qwen prefill/decode/generation with exact fallback controls.
- Continuous batching and an OpenAI-compatible completions streaming API.
- Python-free production build boundary and closed extension admission gate.

## Qualification disclosure

The release owner reported that a prior soak run had completed and explicitly
approved this prerelease without repeating the 7-hour-15-minute run. The prior
run is not a promotable `riley-0.1.0-rc1` artifact: it is not bound to the final
Riley revision, source archive, binary, image, report, and raw-evidence hashes.
The final candidate checker therefore has not reported Gate E as passed.

This is an explicit release-owner exception, not a relaxed checker or a claim
of candidate-bound soak qualification. A later fully qualified release must
rerun the complete evidence set on one exact revision.

## Default semantic paths

- Canonical reduction profile.
- Iteration-batched command completion, with per-operation rollback.
- Separate residual RMSNorm.
- `E1`, `A1`, and `M1` paths are not stable defaults.

See [the changelog](../../CHANGELOG.md) and the
[release contract](../../deploy/16-reliability-and-release.md) for details.
