# Changelog

All notable changes to Riley are documented in this file.

## 0.1.0-rc2 — 2026-08-27

This candidate supersedes `0.1.0-rc1` and fixes a mixed-stage output-routing
correctness defect. When completing-prefill and decode work shared an
iteration, output slots could be sampled in stage order and the resulting
tokens associated with the wrong requests. Riley now canonicalizes dense
output slots before sampling and covers the mixed-stage ordering with a
regression test.

The release owner accepted a previously completed soak and authorized this
prerelease without a candidate-bound 7-hour-15-minute rerun. Gate E therefore
remains unpassed for `0.1.0-rc2`; this is an explicit owner-approved
prerelease, not a relaxation of the fail-closed gate.

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
