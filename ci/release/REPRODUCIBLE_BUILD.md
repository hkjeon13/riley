# PR16 release build reproducibility gate

This gate proves that two independent clean-container builds of one immutable
source produce the same release bytes. It is a release gate, not an ordinary
local development command.

Run the build only on the designated remote host (`server-4096`). The gate
does not pass `--gpus` and never initializes a device, but it compiles the CUDA
release target and therefore belongs with the remote CUDA release workflow.
Local execution is limited to the standard-library checker unit tests.

## Immutable inputs

The gate fixes all of the following before build A starts:

- a full 40-character Git revision;
- the byte-exact uncompressed `git archive --format=tar` for that revision;
- `SOURCE_DATE_EPOCH`, taken from that commit's author-independent commit
  timestamp;
- one Linux/amd64 OCI builder image addressed by its resolved `sha256:` image
  ID;
- Rust 1.85.0, CUDA toolkit 12.8.1, nvcc 12.8.93, and CUDA architecture 89;
- `cargo build --locked --offline --release --features cuda,server`; and
- Docker `network=none`, the explicit `runc` runtime, no device request, no
  external Cargo cache mount, a read-only source mount, and a new anonymous
  workspace volume for each build.

`ReproducibleBuild.Dockerfile` prepares the dependency-complete build image.
Network access is allowed only during this preparation step. Its base images
are digest-pinned, it performs `cargo fetch --locked`, and it does not compile
the release. The resulting local image ID is the immutable builder input to
both offline builds. Changing that ID invalidates the evidence.

## Remote execution

On `server-4096`, check out the candidate revision and prepare the builder
image. The release preflight deliberately fails until the repository owner has
selected the root `LICENSE` and matching Cargo license metadata.

```sh
docker build \
  --file ci/release/ReproducibleBuild.Dockerfile \
  --target build-environment \
  --progress plain \
  --tag rustinfer-repro-builder:pr16 \
  .

BUILDER_IMAGE_ID=$(docker image inspect \
  --format '{{.Id}}' rustinfer-repro-builder:pr16)

ci/run_release_reproducibility.sh \
  --builder-image "${BUILDER_IMAGE_ID}" \
  --source-revision HEAD \
  --output-dir /home/psyche/rustinfer-artifacts/pr16/reproducible-build
```

The host runner refuses an existing output directory. It creates build A and B
as separate Docker containers, captures the Docker daemon's raw pre-start
`inspect` JSON for each, starts them serially, and removes each container plus
its anonymous workspace volume only after retaining the evidence. It selects
the byte-identical A artifacts as `final/` and then runs the independent
checker over A, B, and final.

## Evidence contract

Each `repro-build-{a,b}.tar` is an uncompressed USTAR archive with one closed
root and exactly these payloads:

```text
SHA256SUMS
bin/rustinfer
bundle/rustinfer.tar.gz
logs/bundle-build.log
logs/bundle-verify.log
logs/cargo-build.log
logs/container-inspect.json
logs/container-invocation.txt
logs/preflight.log
logs/toolchain.txt
manifest/build.json
manifest/native-dependencies.txt
source.tar
```

The checker never extracts supplied member paths. It rejects absolute or
traversing paths, duplicates, links, devices, PAX metadata, wrong ownership,
wrong modes, inconsistent timestamps, unexpected members, size overflow,
non-canonical or duplicate-key JSON, and incomplete internal checksums.

Producer status is insufficient for approval. The checker additionally:

- validates the embedded source tar's Git PAX revision, member timestamps,
  safe types, and byte equality with the canonical source archive;
- parses the raw Docker inspect receipts and requires different container and
  workspace-volume IDs, the exact builder image, `network=none`, `runc`, no GPU
  or other device request, and the reviewed mount/isolation contract;
- validates raw rustc/Cargo/nvcc output and the exact locked/offline command;
- runs the hostile-input release bundle verifier independently for A, B, and
  final;
- derives native dependencies from each ELF instead of accepting a producer
  declaration; and
- byte-compares the standalone binary, deterministic bundle, and native
  dependency manifest across A, B, and final.

Success produces `reproducibility-report-v1.json` plus a top-level
`SHA256SUMS`. The final release-candidate manifest should bind the exact report,
both raw evidence tars, the canonical source tar, builder image ID, and selected
final artifacts.

## CPU-only contract tests

These tests build only synthetic ELF fixtures and tar files. They do not
compile CUDA, load a model, or initialize a GPU:

```sh
python3 -m unittest ci/release/test_reproducible_build.py -v
python3 -m py_compile \
  ci/release/check_reproducible_build.py \
  ci/release/package_reproducible_build_evidence.py
bash -n \
  ci/run_release_reproducibility.sh \
  ci/release/run_reproducible_build_once.sh
```
