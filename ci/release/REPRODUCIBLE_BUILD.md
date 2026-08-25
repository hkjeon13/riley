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
- its expected SHA-256 supplied from the reviewed release-candidate contract,
  independently of the evidence tar's own manifest;
- `SOURCE_DATE_EPOCH`, taken from that commit's author-independent commit
  timestamp;
- one Linux/amd64 OCI builder image addressed by its resolved `sha256:` image
  ID;
- Rust 1.85.0 selected by the exact rustup toolchain override
  `1.85.0-x86_64-unknown-linux-gnu`, CUDA toolkit 12.8.1, nvcc 12.8.93,
  and CUDA architecture 89;
- nvcc intermediates placed beside their stable object outputs with
  `--objdir-as-tempdir`, preventing process-derived `tmpxft` file symbols from
  changing otherwise identical unstripped ELFs;
- `cargo build --locked --offline --release --features cuda,server` for the
  production server, followed by `cargo build --locked --offline --release
  --features bench,cuda --bin rustinfer-profile` for the profiling subject; and
- Docker `network=none`, the explicit `runc` runtime, no device request, no
  external Cargo cache mount, an explicit shell entrypoint/root UID and GID,
  disabled inherited health checks, closed image-baseline environment plus
  empty proxy variables, read-only inputs, and a new anonymous `NoCopy`
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
  --expected-source-archive-sha256 \
    SHA256_FROM_REVIEWED_RELEASE_CANDIDATE_CONTRACT \
  --source-revision HEAD \
  --output-dir /home/psyche/rustinfer-artifacts/pr16/reproducible-build
```

The host runner requires `HEAD` to equal the selected revision and the entire
checkout (including untracked files) to be clean. It refuses an existing or
in-checkout output directory and requires the trusted source-archive digest as
a caller-supplied input. It regenerates the canonical archive and fails before
either build if its digest differs. It checks the exact clean checkout again
before running the host checker, binding that checker to the archived source
revision. It then creates build A and B as separate Docker containers, captures
the Docker daemon's raw builder-image inspect plus raw pre-start container
`inspect` JSON for each, starts them serially, captures a second daemon inspect
receipt proving that the same container exited once with status zero, and then
removes each container plus its anonymous workspace volume. The last successful
in-container action writes a closed completion receipt binding the source,
build/image/container identities, exact Cargo command, and hashes of every raw
artifact and build log. The host packager cross-checks that receipt against both
daemon inspect receipts and the bytes it packages. It selects the byte-identical
A artifacts as `final/` and then runs the independent checker over A, B, and
final.

## Evidence contract

Each `repro-build-{a,b}.tar` is an uncompressed USTAR archive with one closed
root and exactly these payloads:

```text
SHA256SUMS
bin/rustinfer
bin/rustinfer-profile
bundle/rustinfer.tar.gz
logs/bundle-build.log
logs/bundle-verify.log
logs/build-completion.json
logs/builder-image-inspect.json
logs/cargo-build.log
logs/container-inspect.json
logs/container-inspect-post.json
logs/container-invocation.txt
logs/preflight.log
logs/profile-build.log
logs/toolchain.txt
manifest/build.json
manifest/native-dependencies.txt
source.tar
```

The checker never extracts supplied member paths. It rejects absolute or
traversing paths, duplicates, links, devices, PAX metadata, non-canonical USTAR
headers or trailing records, wrong ownership, wrong modes, inconsistent
timestamps, unexpected members, size overflow, non-canonical or duplicate-key
JSON, and incomplete internal checksums.

Producer status is insufficient for approval. The checker additionally:

- validates the embedded source tar's Git PAX revision, member timestamps,
  safe types, and byte equality with the canonical source archive;
- parses the raw Docker inspect receipts and requires different container and
  workspace-volume IDs, an exact pre-start state followed by an exited/zero
  state for the same container, nonzero ordered start/finish timestamps, zero
  restarts, the exact builder image configuration, exact command/environment,
  `network=none`, `runc`, no GPU or other device request, and an anonymous
  `NoCopy` mount/isolation contract;
- rehashes every artifact and build log named by the closed in-container
  completion receipt and requires its source, image, build, and container IDs
  to match the independent host receipts;
- validates raw rustc/Cargo/nvcc output and the exact locked/offline command;
- runs the hostile-input release bundle verifier independently for A, B, and
  final;
- derives native dependencies from each ELF instead of accepting a producer
  declaration; and
- byte-compares the standalone server binary, `rustinfer-profile` binary,
  deterministic bundle, and native dependency manifest across A, B, and final.

Success produces `reproducibility-report-v1.json` plus a top-level
`SHA256SUMS`. The final release-candidate manifest should bind the exact report,
both raw evidence tars, the canonical source tar, builder image ID, and selected
final artifacts. The checker's `--expected-source-archive-sha256` value is a
trusted release-candidate input; deriving it from an evidence manifest would
collapse the intended source trust boundary.

## CPU-only contract tests

These tests build only synthetic ELF fixtures and tar files. They do not
compile CUDA, load a model, or initialize a GPU:

```sh
python3 -m unittest ci/release/test_reproducible_build.py -v
python3 -m py_compile \
  ci/release/check_reproducible_build.py \
  ci/release/package_reproducible_build_evidence.py \
  ci/release/run_release_python.py \
  ci/release/write_reproducible_build_completion.py
bash -n \
  ci/run_release_reproducibility.sh \
  ci/release/run_reproducible_build_once.sh
```

Synthetic inspect/receipt fixtures exercise parser rejection and cross-binding
only; they are not evidence that Docker executed a build. Release evidence is
accepted operationally only when produced by the reviewed remote host runner,
whose start/attach, post-run daemon inspect, and last-action in-container
completion sequence is also covered by the static contract test.
