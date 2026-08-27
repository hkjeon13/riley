# Riley 0.1.0-rc2

**Riley — a Rust-native LLM inference engine.**

Riley (Rust Inference LLM Engine) is a Rust-native, CUDA-first LLM inference
engine focused on low-latency execution, predictable memory management, and
reusable model execution patterns.

This source prerelease supersedes `0.1.0-rc1`. It targets Linux x86_64, CUDA
12.8, one NVIDIA `sm_89` GPU, and the pinned SmolLM2 release lane. It also
contains the previously validated dense Qwen compatibility path. Crates remain
`publish = false`; this release does not publish packages to crates.io.

## Correctness fix

RC1 could associate generated tokens with the wrong requests when
completing-prefill and decode work shared an iteration. The scheduler assigned
dense output slots in candidate order, but the iteration plan exposed them in
stage order. Sampling used that stage order while result commit interpreted
samples positionally as dense slots.

RC2 canonicalizes output slots into ascending dense-slot order before sampling
and adds a mixed-stage regression test. Remote RTX 4090 validation compared
the `C=5` and `C=8` lanes against the vLLM oracle for 30 measured iterations;
the `C=8` lane was checked with both iteration-batched completion and the
per-operation fallback. Prompt and generated-token hashes matched exactly.

## Riley versus vLLM benchmark

The comparison uses the release-code commit `e8a88e5`, vLLM 0.27.1, and
`HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` in BF16 on the same RTX 4090.
Each cell runs in three fresh processes with five warmups and 30 measured
iterations. Sampling is greedy with fixed output length and pretokenized
prompts. Percentiles use R7 within each run and the reported value is the
median across runs. Output throughput is also the median across runs.
Riley uses CUDA runtime 12.8.1; the pinned vLLM wheel uses CUDA runtime 13.0.
This is a warm, three-run, single-host comparison of each engine's pinned
stack, not formal repeatability evidence or a Gate E result.
The lanes ran sequentially rather than in randomized order and do not carry a
thermal or GPU-contention preflight receipt. The result covers one 135M model
on one GPU and must not be generalized to larger models. vLLM's C1/P128 TTFT
p95 also varied materially by run (10.45, 16.83, and 7.61 ms), so the
three-run median should be read with that limitation.

Lower latency is better. Values are `p50 / p95` in milliseconds.

| Workload `(C/P/O)` | Token exact | Riley TTFT | vLLM TTFT | Riley TPOT | vLLM TPOT | Riley E2E | vLLM E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1/128/32` | yes (3/3) | 5.45 / 5.49 | 6.93 / 10.45 | 7.17 / 7.19 | 1.12 / 1.22 | 227.75 / 228.41 | 41.57 / 45.96 |
| `1/4096/128` | yes (3/3) | 926.61 / 929.03 | 16.33 / 17.31 | 122.04 / 122.33 | 1.19 / 1.23 | 16,425.71 / 16,464.98 | 167.39 / 173.37 |
| `8/128/32` | yes (3/3) | 10.98 / 22.79 | 11.62 / 20.13 | 8.18 / 8.22 | 1.48 / 1.50 | 265.86 / 276.19 | 57.76 / 66.23 |

Higher output-token throughput is better.

| Workload `(C/P/O)` | Riley tok/s | vLLM tok/s | Riley / vLLM | Throughput lead |
|---|---:|---:|---:|---:|
| `1/128/32` | 140.46 | 750.39 | 18.72% | vLLM 5.34x |
| `1/4096/128` | 7.79 | 758.31 | 1.03% | vLLM 97.34x |
| `8/128/32` | 926.75 | 4,306.82 | 21.52% | vLLM 4.65x |

For every measured batch, the comparison requires multiset equality of the
per-request prompt hash, generated-token hash, and token counts. Prompts are
unique within each batch, so this also verifies the prompt-to-generation
association that RC1 violated. A non-exact cell is invalid for performance
comparison.

The raw archive is published as
`riley-vllm-rc2-benchmark-raw-e8a88e5.tar` (2.4 MiB), SHA-256
`158cb1e8f262cabc71640f9165baaf2af647bf82c5131724cd294873ffe505cd`.
It contains both raw result trees, the exact runner scripts, and the
fail-closed aggregation script.

Measurements were collected at `e8a88e5`, the inference-code parent of the
RC2 release-documentation commit. The intervening changes are limited to
release documentation and CI release metadata/tests; no Cargo manifest, Rust
crate, or CUDA source changes. Because the raw revision is not the tag
revision, this comparison is informative parent-commit evidence and the
candidate checker must not accept it as candidate-bound performance evidence.

## Qualification disclosure

The release owner accepted a previously completed soak and explicitly approved
this prerelease without repeating the 7-hour-15-minute run. That accepted soak
is not bound to the RC2 revision or artifacts. Gate E has not passed for RC2;
this remains an owner-approved prerelease, not a relaxed checker or a claim of
candidate-bound soak qualification.

## Default semantic paths

- Canonical reduction profile.
- Iteration-batched command completion, with per-operation rollback.
- Separate residual RMSNorm.
- `E1`, `A1`, and `M1` paths are not stable defaults.

See [the changelog](../../CHANGELOG.md), the
[release contract](../../deploy/16-reliability-and-release.md), and the
[benchmark contract](../../benchmarks/README.md) for details.
