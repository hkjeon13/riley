# Remote optimizer execution evidence

CUDA compilation, GPU tests, and the SmolLM2 parity test in this workflow run
only on the designated `server-4096` host (`psyche-MS-7D91`, GPU UUID
`GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`).  The host runner refuses any
other hostname or GPU.  Local use is limited to reviewing the scripts and
running the CPU-only writer tests.

## Prepare the dependency image on server-4096

Image preparation is the only networked phase: it installs the Clippy
component for the exact `1.85.0-x86_64-unknown-linux-gnu` toolchain and fetches
the locked Cargo dependencies.  `--no-cache` is required because the BuildKit
seed is a read-only bind mount.  No candidate source is baked into the
resulting image.

```sh
docker build \
  --no-cache \
  --file ci/release/OptimizationEvidence.Dockerfile \
  --tag riley-optimization-evidence:pr16 \
  .

BUILDER_IMAGE_ID=$(docker image inspect \
  --format '{{.Id}}' riley-optimization-evidence:pr16)
```

The actual evidence container resolves that immutable image ID, uses
`--network none`, mounts the canonical source archive and pinned model
read-only, grants only the designated GPU, and builds in a fresh anonymous
workspace volume.  The immutable image pins
`RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu`, so a checkout-local rustup
directory override cannot select or download another toolchain.

## Run

The profile executable must be the profile artifact already selected by the
reproducibility/native gates.  All four expected hashes are trusted promotion
inputs; do not copy them from a newly produced optimizer report.

```sh
ci/run_remote_optimization_evidence.sh \
  --builder-image "${BUILDER_IMAGE_ID}" \
  --expected-source-archive-sha256 SOURCE_ARCHIVE_SHA256 \
  --expected-model-tree-sha256 MODEL_SHA256SUMS_SHA256 \
  --expected-profile-binary-sha256 PROFILE_BINARY_SHA256 \
  --model-dir /tmp/riley-pr16-model-93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --profile-binary /path/to/reproducible/final/riley-profile \
  --output-dir /home/psyche/riley-artifacts/pr16/optimizer-evidence \
  --source-revision HEAD
```

The runner rechecks the clean checkout before and after GPU execution.  It
regenerates the canonical Git archive and model `SHA256SUMS`, validates pinned
weight/tokenizer bytes, compares the newly built profile ELF byte-for-byte
with the trusted reproducible profile ELF, writes the report/receipt, and
finally calls `check_optimization_evidence.py` to package and replay the raw
archive.

## Execution receipt v3

Receipt schema `riley.optimizer-execution-receipt.v3` records these exact
commands in order:

1. Python-free locked CUDA compile smoke.
2. locked/offline workspace all-features/all-targets tests.
3. locked/offline Cargo `--no-run` for `host_runtime_gpu`.
4. direct execution of copied `/evidence/host-runtime-gpu-test`.
5. locked/offline Cargo `--no-run` for `primitives_gpu`.
6. direct execution of copied `/evidence/primitives-gpu-test`.
7. locked/offline Cargo `--no-run` for `llama_batch_gpu`.
8. direct execution of copied `/evidence/llama-batch-gpu-test` with the pinned
   checkpoint mounted at `/model`.
9. a second locked/offline Cargo `--no-run` for `llama_batch_gpu` in a separate
   empty fixed37 target directory.
10. direct execution of the separately copied
    `/evidence/fixed37-production-batch-gpu-test` with the pinned checkpoint.

Every recorded command carries the exact
`RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu` environment entry.  Both the
writer and the independent replay checker reject a missing, substituted, or
additional command environment entry.

Each compile command uses its own empty absolute target directory and emits
Cargo JSON.  The runner accepts exactly one fresh `compiler-artifact` test ELF,
copies it to `/evidence`, rehashes both paths, then directly executes the copied
path.  Each subject receipt binds:

- Cargo test target and compiler-artifact path;
- original and copied SHA-256 (which must be equal) and size;
- stable copied execution path;
- producing compile command ID and consuming execution command ID.

The sixth semantic log is the candidate-revision
`pr16-fixed37-production-batch-e0-v1` gate. It forbids prompt filtering and
runs the immutable `benchmarks/reference/smollm2-135m-bf16.json` corpus:
31 cases, 481 generated steps, and exact window 16. Both fixed37 cached decode
and fixed37 growing-prefix prefill must match every golden token; canonical
cached batch is a golden-token control. The structurally identical first
prefill is raw-byte exact. Later cached-decode versus growing-prefix logits use
the immutable full-BF16 cosine/max-absolute/mean-absolute E0 bounds because
those are different attention paths. The marker records the exact fixture and
little-endian-U32 golden-token SHA-256 values, profiles, completion and
residual modes, observed worst metrics, zero threshold/token/prefill mismatch
counts, and zero live-allocation delta/owner-close count.

In addition to the six semantic logs, the raw input inventory contains:

```text
command-batch-lifecycle-build.log
command-batch-resource-ledger-build.log
smollm2-multi-step-greedy-exact-build.log
fixed37-production-batch-e0-build.log
```

`write_optimization_execution_evidence.py` rejects command substitution,
reordering, nonzero exits, compiler-artifact substitution, subject-byte
substitution, GPU substitution, and semantic/parity marker mismatch before the
independent package/replay checker runs. Host-side packaging uses a clean
Python environment through `run_release_python.py`, so the same checker also
runs on the designated Ubuntu 22.04 host's Python 3.10 without importing user
packages or weakening the standard-library-only contract.

The replay checker must recognize receipt v3, the four build logs, the
ten-command order above, and the expanded subject fields. The separate fixed37
Cargo receipt proves that its copied ELF is a fresh candidate-revision
compiler artifact rather than a second unrecorded execution of the parity ELF.
Older receipt versions cannot express this closed command/subject inventory
and are intentionally not emitted by this runner.
