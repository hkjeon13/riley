# Remote optimizer execution evidence

CUDA compilation, GPU tests, and the SmolLM2 parity test in this workflow run
only on the designated `server-4096` host (`psyche-MS-7D91`, GPU UUID
`GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`).  The host runner refuses any
other hostname or GPU.  Local use is limited to reviewing the scripts and
running the CPU-only writer tests.

## Prepare the dependency image on server-4096

The dependency fetch is the only networked phase.  `--no-cache` is required
because the BuildKit seed is a read-only bind mount.  No candidate source is
baked into the resulting image.

```sh
docker build \
  --no-cache \
  --file ci/release/OptimizationEvidence.Dockerfile \
  --tag rustinfer-optimization-evidence:pr16 \
  .

BUILDER_IMAGE_ID=$(docker image inspect \
  --format '{{.Id}}' rustinfer-optimization-evidence:pr16)
```

The actual evidence container resolves that immutable image ID, uses
`--network none`, mounts the canonical source archive and pinned model
read-only, grants only the designated GPU, and builds in a fresh anonymous
workspace volume.

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
  --model-dir /tmp/rustinfer-pr16-model-93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --profile-binary /path/to/reproducible/final/rustinfer-profile \
  --output-dir /home/psyche/rustinfer-artifacts/pr16/optimizer-evidence \
  --source-revision HEAD
```

The runner rechecks the clean checkout before and after GPU execution.  It
regenerates the canonical Git archive and model `SHA256SUMS`, validates pinned
weight/tokenizer bytes, compares the newly built profile ELF byte-for-byte
with the trusted reproducible profile ELF, writes the report/receipt, and
finally calls `check_optimization_evidence.py` to package and replay the raw
archive.

## Execution receipt v2

Receipt schema `rustinfer.optimizer-execution-receipt.v2` records these exact
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

Each compile command uses its own empty absolute target directory and emits
Cargo JSON.  The runner accepts exactly one fresh `compiler-artifact` test ELF,
copies it to `/evidence`, rehashes both paths, then directly executes the copied
path.  Each subject receipt binds:

- Cargo test target and compiler-artifact path;
- original and copied SHA-256 (which must be equal) and size;
- stable copied execution path;
- producing compile command ID and consuming execution command ID.

In addition to the five semantic logs, the raw input inventory contains:

```text
command-batch-lifecycle-build.log
command-batch-resource-ledger-build.log
smollm2-multi-step-greedy-exact-build.log
```

`write_optimization_execution_evidence.py` rejects command substitution,
reordering, nonzero exits, compiler-artifact substitution, subject-byte
substitution, GPU substitution, and semantic/parity marker mismatch before the
independent package/replay checker runs.

The replay checker must recognize receipt v2, the three additional build logs,
the eight-command order above, and the expanded subject fields.  Keeping the
old v1 five-command contract would force the receipt to claim `cargo test`
while actually executing a copied ELF, so v1 is intentionally not emitted by
this runner.
