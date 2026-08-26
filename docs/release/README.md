# Release bundle and runtime contract

The release package is a deterministic `tar.gz` containing exactly one
versioned directory:

```text
rustinfer-VERSION-linux-x86_64-cuda12.8/
├── LICENSE
├── NOTICE                         # optional, only when the repository has one
├── SHA256SUMS
├── bin/rustinfer
└── manifest/
    ├── native-dependencies.txt
    └── release.json
```

`release.json` is the machine-readable license, feature, default, support,
unsupported-scope, validation-lane, known-limitation, configuration, and
rollback contract. Its artifact license is exactly the `MIT` SPDX expression.
`native-dependencies.txt` is generated from the CLI ELF's direct `DT_NEEDED`
entries and is limited to the reviewed Linux/CUDA runtime allowlist.
`SHA256SUMS` covers every payload file except itself.

## Reviewed project license

The project uses the standard MIT license with
`Copyright (c) 2026 rustinfer contributors`. Release preflight and bundle
production require the root `LICENSE` bytes to match that reviewed text,
`workspace.package.license` to equal `MIT`, and every workspace package to use
`license.workspace = true`. Bundle verification independently requires the
same exact `LICENSE` bytes and canonical embedded SPDX field. A root `NOTICE`
remains optional and is packaged only when present.

## Deterministic bundle

After a locked Linux x86_64 CUDA 12.8.1 release build, create and verify the
bundle with immutable provenance inputs:

```sh
python3 ci/release/check_release_preflight.py \
  --source-revision FULL_40_CHARACTER_GIT_SHA \
  --source-date-epoch UNIX_TIMESTAMP

python3 ci/release/build_release_bundle.py \
  --binary target/release/rustinfer \
  --output dist/rustinfer.tar.gz \
  --source-revision FULL_40_CHARACTER_GIT_SHA \
  --source-date-epoch UNIX_TIMESTAMP

python3 ci/release/verify_release_bundle.py dist/rustinfer.tar.gz
```

The producer fixes archive order, gzip/tar timestamp, uid/gid, owner names,
file modes, JSON serialization, dependency ordering, and checksum ordering.
It verifies its own output before returning success. Identical binary,
license, notice, version, revision, and epoch inputs produce identical archive
bytes. This package-level guarantee does not by itself claim that two native
CUDA compilations produce identical ELF bytes; the release workflow must add a
separate clean double-build comparison before making that claim.

The verifier does not use `tar.extract`. It rejects absolute or parent paths,
backslash paths, duplicates, links, devices, FIFOs, PAX metadata, unexpected
files, Python-family artifact names, non-canonical metadata, unreviewed native
libraries, ELF/manifest disagreement, non-canonical release configuration, and
missing or mismatched checksums.

## Minimal CUDA runtime image

`ci/release/Dockerfile` has separate toolchain, builder, and final runtime
stages:

```sh
docker build \
  --file ci/release/Dockerfile \
  --build-arg RUSTINFER_CUDA_ARCHITECTURES=89 \
  --build-arg RUSTINFER_SOURCE_REVISION=FULL_40_CHARACTER_GIT_SHA \
  --build-arg SOURCE_DATE_EPOCH=UNIX_TIMESTAMP \
  --tag rustinfer:VERSION-cuda12.8 \
  .
```

The builder selects the already installed exact
`1.85.0-x86_64-unknown-linux-gnu` Rust toolchain, preventing checkout-local
rustup reconciliation or downloads. The final stage is a digest-pinned NVIDIA
CUDA 12.8.1 runtime image. It copies only the already verified bundle payload
to `/opt/rustinfer`, does not inherit the rustup environment or toolchain, runs
as numeric user `65532:65532`, and contains no repository source, Rust/CUDA
compiler, build system, Python/Pip executable, or Python-family package
artifact. Model, tokenizer, and configuration files remain operator-mounted
inputs; they are not embedded into the image.

`python3 ci/release/verify_runtime_dockerfile.py` is a CPU-only static guard
for that stage boundary. A GPU release lane must additionally start the final
image with NVIDIA Container Runtime, validate its injected `libcuda.so.1`, and
run the Python-free real-model API/generation suite.

Ubuntu 22.04 supplies Python 3.10 in the build stage. The builder installs its
distribution-provided `tomli` package and runs every TOML-reading release
helper through `run_release_python.py`, which exposes that parser under the
Python 3.11 `tomllib` name. Python and the compatibility package remain
confined to the discarded build stage and are absent from the final runtime
image.

## Supported model and serving scope

The source config parser and canonical dense-decoder runtime recognize exactly
these Hugging Face identities:

- `model_type=llama` with `LlamaForCausalLM`;
- `model_type=qwen2` with `Qwen2ForCausalLM`.

Qwen2.5 compatibility applies only where it retains that `qwen2` /
`Qwen2ForCausalLM` identity and matches the pinned NFC + Split + ByteLevel BPE
and no-tools `tokenizer_config.json` profile. The Llama artifact path uses the
SmolLM2-compatible ByteLevel BPE profile. This is source-family support, not a
claim that every checkpoint bearing one of those names is compatible.

The strict config and safetensors parsers accept BF16 and FP16 metadata and
tensor payloads. The production CUDA execution plan is BF16-only; an FP16
checkpoint therefore fails before execution rather than being converted or
silently reinterpreted. Checkpoints must be local, covered by an exact
`rustinfer-checkpoint.json`, and use either `model.safetensors` or a declared
`model.safetensors.index.json` shard set. No checkpoint transform is accepted.

Configs are closed and duplicate-free. They require dense SiLU gated MLPs
without MLP bias at execution, finite positive RMSNorm epsilon and RoPE theta,
an even head dimension at the parser boundary,
`num_attention_heads * head_dim == hidden_size`, and `num_key_value_heads`
dividing `num_attention_heads`. Production CUDA serving is narrower: its
continuous-batch executor requires `head_dim == 64`; other even head dimensions
fail during backend preparation. Only standard, non-interleaved, full RoPE is
supported: no scaling, a missing or `1.0` `partial_rotary_factor`, no Llama
sliding window, and Qwen `use_sliding_window` missing or false. `architectures`
may be missing or empty; when present it must contain exactly the matching
`LlamaForCausalLM` or `Qwen2ForCausalLM` identity.

The explicit unsupported inventory closes over quantized, PyTorch pickle/bin,
and GGUF weights; MoE, multimodal/VL, Qwen3, encoder, and encoder-decoder
architectures; remote code and network downloads; CPU, FP16 CUDA, multi-GPU,
non-64 head dimensions, tensor/pipeline-parallel, and distributed execution;
and Python, PyTorch, Transformers, or Triton fallback. The server is a strict
close-delimited HTTP/1.1 completions surface. It does not provide
chat-completions, embeddings, responses, HTTP/2, TLS termination, or built-in
authentication.

The PR16 release-qualification lane is pinned to
`HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16, one GPU, and `sm_89`.
The dense Qwen compatibility statement is backed by the earlier PR12
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` evidence on the same one-GPU
architecture class; it is not a second PR16 release-qualification lane. The
release build/qualification matrix is Linux x86_64, CUDA 12.8.1, one GPU, and
`sm_89` only.

## Semantic path and approximation policy

The bundle's closed `features.semantic_paths` table records every selectable
optimized serving path that ships in the first candidate:

| Path | Class | Stable default | Exact fallback | Approval evidence |
|---|---|---:|---|---|
| iteration command batch | `E0` | enabled | `--execution-completion per-operation` | `pr15-iteration-command-batch-exact-v1` |
| fused residual RMSNorm | `E0` candidate | disabled, unsupported | `--residual-rmsnorm separate` | prior PR15 gate only; no candidate-bound approval |
| fixed-contiguous-37 balanced reductions | `E0` | disabled | `--reduction-profile canonical-v1` | `smollm2-fp32-bf16-native-e0-v2` **and** `pr16-fixed37-production-batch-e0-v1` |

The supported release surface includes only `reference` and candidate-approved
`E0` semantics. `E1`, `A1`, and `M1` paths are absent, approximation is
disabled, and both error and quality budgets are `null`; the release cannot
imply a configured approximation budget for code it does not ship. The fused
selector remains in the binary for development compatibility, but it is not
first-release-qualified: its prior
PR15 E0 result predates the candidate and no current-revision fused report is
bound to the final gate. Operators must keep the documented default
`--residual-rmsnorm separate`.

Fixed37 release qualification is conjunctive. The native gate provides the
FP32/BF16 numeric-oracle contract, while the candidate-revision production
batch gate runs all 31 immutable golden cases (481 generated steps, exact
window 16) through `PreparedLlamaBatchExecutor` in separate residual-norm and
iteration-batch mode. It requires exact golden top-1 tokens, zero same-path
prefill byte mismatches, zero allocation deltas/leaks, and the immutable
cached-decode versus growing-prefix numeric bounds. Neither gate alone
qualifies this selectable serving path.

The three stable selector defaults are also bound to the exact reviewed bytes
of `crates/rustinfer-server/src/main.rs`. Release preflight rejects any source
digest change until this contract is deliberately reviewed and updated. The
Python-free E2E invocation omits all three selectors and checks the embedded
release defaults/source binding, so it exercises the Rust resolver rather than
merely restating the expected values in Docker arguments.

## Default configuration and rollback

The production defaults recorded in every bundle are the `canonical-v1`
reduction profile, iteration-batched completion, and separate residual
RMSNorm. Fused residual RMSNorm remains incompatible with iteration-batched
completion. The opt-in `fixed-contiguous-37-balanced-v1` profile requires an
effective `--max-sequence-tokens` no greater than 8192.

For optimization isolation, drain or cancel active work, stop the process, and
restart the same current checksummed bundle with all conservative E0 flags:

```text
--reduction-profile canonical-v1 --execution-completion per-operation --residual-rmsnorm separate
```

`--reduction-profile canonical-v1` is the numerical-kernel rollback. The
separate `--execution-completion per-operation` flag isolates command-batch
completion, so operators can roll back either dimension independently.
The pinned PR16 soak manifest executes this exact combination in its
`rollback-per-operation` arm, so the evidence validates this current-bundle
conservative restart path without relying on any of the three defaults. It
does not manufacture a preceding stable binary.

For binary release rollback, restart a preceding checksummed stable rustinfer
bundle with the same model/configuration only when such a release actually
exists, verify `/v1/models`, then restore traffic. The first stable release
candidate has no preceding stable rustinfer bundle, so only the current-bundle
conservative E0 restart is available at that point. Rollback never reuses an
unverified executable or edits a published bundle in place.

## Graceful shutdown and final metrics

The production CLI blocks `SIGINT` and `SIGTERM` before starting backend or
HTTP threads and consumes either signal synchronously. A received signal stops
admission, interrupts incomplete HTTP framing, drains bounded active work,
closes scheduler/CUDA resources, and exits with status zero only when the
global shutdown deadline and native close contract succeed.

Release/soak automation may set `RUSTINFER_SHUTDOWN_METRICS_PATH` to a new
absolute path. After successful native close, the CLI atomically requests a
backend-captured allocation snapshot and creates that file with the same
fixed, prompt-free JSON shape as `GET /metrics`. Existing paths are never
replaced. Missing post-close evidence, a non-absolute path, serialization or
sync failure, a shutdown deadline, or a CUDA/context close failure makes the
process exit unsuccessfully instead of emitting synthetic zero gauges.
