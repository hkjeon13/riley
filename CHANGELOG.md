# Changelog

All notable changes to Riley are documented in this file.

## 0.1.0-rc1 — 2026-08-27

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
