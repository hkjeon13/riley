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

`release.json` is the machine-readable feature, default, support,
configuration, and rollback contract. `native-dependencies.txt` is generated
from the CLI ELF's direct `DT_NEEDED` entries and is limited to the reviewed
Linux/CUDA runtime allowlist. `SHA256SUMS` covers every payload file except
itself.

## Known release blocker: project license

The repository owner has not selected a root `LICENSE`. Release preflight
therefore fails closed. Packaging must not infer or generate a license. Before
the first release candidate, the owner must add an approved, non-placeholder
root `LICENSE`, decide whether a root `NOTICE` is required, and align Cargo
package license metadata with that decision.

This blocker is intentional and does not affect ordinary development builds.

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
stages. Build it only after the license blocker is resolved:

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

## Default configuration and rollback

The production defaults recorded in every bundle are iteration-batched
completion and separate residual RMSNorm. Fused residual RMSNorm remains
incompatible with iteration-batched completion.

For optimization isolation, drain or cancel active work, stop the process, and
restart with both conservative flags:

```text
--execution-completion per-operation --residual-rmsnorm separate
```

For release rollback, restart the preceding checksummed bundle with the same
model/configuration, verify `/v1/models`, then restore traffic. Rollback never
reuses an unverified executable or edits a published bundle in place.

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
