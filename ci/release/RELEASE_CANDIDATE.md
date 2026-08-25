# Final release-candidate evidence gate

`check_release_candidate.py` is the last CPU-only gate before an owner creates
a release tag. It does not build an image, start the server, load a model, use
CUDA, create a tag, or push anything. It consumes immutable artifacts produced
by the earlier gates and emits a single source-bound decision.

Run it from a Python 3.11+ POSIX/Linux host with no-follow `openat` support
after copying the complete evidence set into one read-only directory:

```sh
python3 ci/release/check_release_candidate.py \
  --manifest /evidence/release-candidate.json \
  --evidence-root /evidence \
  --expected-revision FULL_LOWERCASE_40_CHARACTER_COMMIT \
  --expected-source-archive-sha256 LOWERCASE_SHA256 \
  --expected-release-image-id sha256:LOWERCASE_SHA256 \
  --expected-reproducible-build-image-id sha256:LOWERCASE_SHA256 \
  --expected-cuda-build-image-id sha256:LOWERCASE_SHA256 \
  --expected-optimization-build-image-id sha256:LOWERCASE_SHA256 \
  --expected-correctness-golden-sha256 LOWERCASE_SHA256 \
  --report /evidence/release-candidate-report.json
```

The seven `--expected-*` values are trusted promotion inputs. Obtain them from
the reviewed commit decision, canonical archive publication, immutable OCI
image inspections for each distinct role, and the separately reviewed
correctness golden; never populate them by copying values out of the candidate
manifest being checked. The reproducibility image may contain the release
packaging Python tool, while the CUDA-fault and optimizer images are
Python-free; their IDs are deliberately independent external anchors.

The report path is create-only. Only `status=passed` and `passed=true` exits
zero. The report schema is `rustinfer.release-candidate-report.v2`. A passed
report binds the SHA-256 of the exact input manifest, source archive, release
binary, release bundle, separate native-calibration and profile executables,
seven gate decisions, all raw/replay artifacts, and the immutable release plus
role-specific build image IDs.

## Closed candidate manifest

The manifest schema version is
`rustinfer.release-candidate-manifest.v2`. Every `path` is a normalized POSIX
path relative to `--evidence-root`; absolute paths, `..`, links, devices,
duplicate paths, and paths resolving outside that root are rejected. Every
file descriptor is exactly `{"path": ..., "sha256": ...}`. The complete
shape is:

```json
{
  "schema_version": "rustinfer.release-candidate-manifest.v2",
  "candidate_id": "rustinfer-X.Y.Z-rcN",
  "source": {
    "git_revision": "FULL_LOWERCASE_40_CHARACTER_COMMIT",
    "git_dirty": false,
    "archive": {"path": "source.tar", "sha256": "LOWERCASE_SHA256"}
  },
  "release": {
    "binary": {"path": "rustinfer", "sha256": "LOWERCASE_SHA256"},
    "bundle": {"path": "rustinfer-X.Y.Z-linux-x86_64-cuda12.8.tar.gz", "sha256": "LOWERCASE_SHA256"},
    "image_digest": "sha256:LOWERCASE_SHA256"
  },
  "evidence": {
    "python_free_e2e": {
      "report": {"path": "python-free-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "python-free-evidence.tar", "sha256": "LOWERCASE_SHA256"},
      "correctness_golden": {"path": "correctness-golden.json", "sha256": "LOWERCASE_SHA256"}
    },
    "cuda_fault": {
      "build_image_id": "sha256:LOWERCASE_SHA256",
      "report": {"path": "cuda-fault-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "cuda-fault-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "native_correctness": {
      "report": {"path": "native-correctness-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_replay": {"path": "native-correctness-replay.tar", "sha256": "LOWERCASE_SHA256"},
      "candidate_executable": {"path": "rustinfer-native", "sha256": "LOWERCASE_SHA256"}
    },
    "reproducible_build": {
      "build_image_id": "sha256:LOWERCASE_SHA256",
      "source_date_epoch": 1700000000,
      "build_a": {"path": "reproducible-build-a.tar", "sha256": "LOWERCASE_SHA256"},
      "build_b": {"path": "reproducible-build-b.tar", "sha256": "LOWERCASE_SHA256"},
      "profile_binary": {"path": "rustinfer-profile", "sha256": "LOWERCASE_SHA256"},
      "native_manifest": {"path": "native-dependencies.txt", "sha256": "LOWERCASE_SHA256"}
    },
    "optimization_correctness": {
      "build_image_id": "sha256:LOWERCASE_SHA256",
      "report": {"path": "optimization-correctness-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "optimization-correctness-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "performance": {
      "report": {"path": "release-performance-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "release-performance-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "reliability_soak": {
      "report": {"path": "reliability-soak-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "reliability-soak-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    }
  }
}
```

The uppercase values above document positions only; they are deliberately not
accepted as evidence. All-zero digests/revisions and common placeholder text
are rejected.

Every artifact is opened with no-follow semantics, hashed from that open file
descriptor, and copied into a private snapshot. All parsers consume the
snapshot, so replacing or editing a path after its digest check cannot alter
the bytes seen by a later gate.

The source archive must be the exact output of `git archive --format=tar` for
the clean revision named by `git_revision`. Besides hashing the bytes, the
checker opens the tar safely and requires the pax global `comment` and every
member's inherited comment to equal that revision. A tar containing the right
files but lacking the Git commit marker is not evidence. The deterministic
release bundle is then passed through `verify_release_bundle.py`; its embedded
source revision must match and its embedded `bin/rustinfer` must be
byte-identical to the separately supplied executable. Each image is identified
only by an immutable `sha256:` image ID. Runtime reports repeat the release
image exactly; build evidence repeats the external anchor for its own
reproducibility, CUDA-fault, or optimizer role.

The reproducibility entry is replayed before GPU evidence is accepted. Build A
and B must be independent clean, networkless containers using the externally
trusted immutable build image. Their production binary, `rustinfer-profile`,
bundle, native dependency manifest, and source archive must be byte-identical
to one another and to the selected final artifacts. The selected profile is
the separate `reproducible_build.profile_binary` artifact. It must equal the
profile executable replayed by optimizer and performance evidence.

`native_correctness.candidate_executable` has a different role: it is the
source-bound `rustinfer-native calibrate` development executable embedded in
the native correctness raw evidence. It must equal that raw executable, but it
must not be substituted for `rustinfer-profile` or shipped as the production
server. Source/archive and report hashes cross-bind the calibration result to
the release without pretending the two executables have identical bytes.

## Raw-replayed gate attestations

The Python-free clean-runtime and CUDA fault runners preserve deterministic
raw evidence tar files and create this closed standard-library-readable
attestation:

```json
{
  "schema_version": "rustinfer.release-gate-attestation.v1",
  "gate": "python-free-clean-runtime-e2e",
  "status": "passed",
  "source": {
    "git_revision": "FULL_LOWERCASE_40_CHARACTER_COMMIT",
    "git_dirty": false,
    "source_archive_sha256": "LOWERCASE_SHA256",
    "release_binary_sha256": "LOWERCASE_SHA256",
    "release_bundle_sha256": "LOWERCASE_SHA256",
    "release_image_sha256": "LOWERCASE_SHA256"
  },
  "raw_evidence_sha256": "LOWERCASE_SHA256",
  "checks": [{"id": "release_bundle_verified", "passed": true}]
}
```

`checks` is a closed, duplicate-free set. The Python-free attestation requires:

- `release_bundle_verified`
- `no_python_executable`
- `no_python_child`
- `no_forbidden_runtime_artifact`
- `native_dependencies_verified`
- `model_load`
- `prefill`
- `decode`
- `greedy_golden`
- `sampling`
- `streaming`
- `cancellation`
- `graceful_shutdown`

The CUDA attestation uses `gate=cuda-fault-injection` and requires:

- `test_inventory_exact`
- `create_rollback_ambiguity`
- `explicit_close_ambiguity`
- `confirmed_completion_deferred_error`
- `unconfirmed_completion_retained`
- `subprocess_isolation`
- `production_fault_symbols_absent`

An attestation is an index into the preserved raw evidence, not a replacement
for it. Its `raw_evidence_sha256` must equal the separately hashed artifact in
the candidate manifest. For Python-free E2E, the canonical USTAR excludes the
attestation to avoid a circular hash and preserves the reviewed golden, every
model file, canonical model checksum manifest, extracted image ELF and native
dependency observations, immutable OCI inspect output, both containers'
pre/runtime/post state and process observations, exact request bytes, raw
JSON/SSE/cancellation responses, shutdown metrics, `raw-evidence.json`, and an
internal `SHA256SUMS`. The final candidate opens that tar, validates its closed
inventory and canonical metadata, replays HTTP/SSE/cancellation semantics and
container lifecycle transitions, validates the extracted executable ELF,
recomputes the full model tree/tokenizer/shutdown bindings, and requires the
replayed attestation object to equal the submitted object exactly. The golden
artifact SHA must also equal the seventh external promotion input. Model ID,
revision, canonical full-tree manifest, weights, and `tokenizer.json` are
cross-bound to optimizer evidence.
For CUDA fault injection, `build_image_id` names the immutable CUDA toolchain
image used by the raw runner. The final candidate opens the canonical closed
tar, rechecks its internal `SHA256SUMS`, exact two-test/four-subprocess marker
inventory, sanitizer result when enabled, production ELF logs, source archive,
standalone binary, release bundle, build image, and release image. It then
requires the recomputed attestation to equal the submitted object exactly.

## Native and optimization correctness are separate gates

The two correctness roles must never be collapsed into one hash:

- native correctness report `1.0.0`, gate
  `smollm2-fp32-bf16-native-e0-v2`: must pass all 31 cases and both E0
  variants from `benchmarks/schemas/correctness-report.schema.json`, bind the
  exact candidate revision, and record the clean-tree digest. Merely parsing
  that report is not approval. `raw_replay` is deterministic closed USTAR with
  both source revisions, FP32/BF16 manifests and sidecars, the replayed oracle
  report, candidate manifest/sidecar/executable, submitted correctness report,
  and internal checksums. The final gate reads F32/BF16 safetensors with its
  standard-library reader, reruns `replay_validate_correctness_report`, and
  requires the embedded source/report/executable bytes to equal the separately
  supplied candidate artifacts exactly;
- optimizer equivalence report schema `1`, gate
  `pr15-iteration-command-batch-exact-v1`: must be passed E0 evidence for the
  same source/archive, pinned SmolLM2 BF16 artifacts, network-none locked/offline
  CUDA sm89 build, and exact `per-operation` versus `iteration-batch` flags with
  `residual_rmsnorm=separate`. Its exact five-test inventory and every expected
  zero-mismatch/CUDA-live-allocation result are checked. Its canonical USTAR
  contains the submitted report, ordered v2 execution receipt, three executable
  Linux x86-64 Rust test ELFs, five execution logs, three Cargo JSON build logs,
  and internal `SHA256SUMS`. The receipt records the three locked/offline
  `--no-run` builds separately from direct execution of the copied
  `/evidence/*-gpu-test` ELFs. Replay verifies nonzero ELF entry points and
  executable `PT_LOAD` segments, unique fresh Cargo compiler-artifact
  provenance, original/copied subject equality, exact eight-command
  environment/exit contracts, source/build/model bindings, semantic log
  records, report bytes, and every subject/log digest. Its immutable optimizer
  image ID is independently trusted and need not equal the reproducibility or
  CUDA-fault image ID. The replayed profile binary must be the same bytes
  selected by reproducibility and performance evidence; native correctness
  retains its separate calibration executable.

Create that bundle only after both source revisions and all raw calibration
artifacts are available:

```sh
python3 ci/release/check_native_correctness_evidence.py \
  --candidate-source-archive source.tar \
  --oracle-source-archive oracle-source.tar \
  --fp32-manifest fp32-manifest.json \
  --bf16-manifest bf16-manifest.json \
  --oracle-report oracle-calibration-report.json \
  --candidate-manifest candidate-manifest.json \
  --correctness-report native-correctness-report.json \
  --output native-correctness-replay.tar
```

The remaining cross-bindings are:

- `rustinfer.release-performance-report.v1`: must be `passed`, have no errors,
  have the exact four reviewed passing checks, select semantic class E0, bind the exact source
  archive/release binary/runtime image, and bind the exact optimizer
  equivalence report bytes and gate ID. Its profile image must equal the
  optimizer build image, and its profile executable SHA must equal the
  reproducible/optimizer profile artifact. It must not bind
  the native 31-case report in the optimizer-report field. Its raw evidence tar must contain only
  `candidate-1.json` through `candidate-5.json`; the final gate revalidates the
  closed native profile schema and all source/model/environment/workload/raw
  hashes, then recomputes the R7 metrics, baseline ratios, and thresholds;
- `rustinfer.reliability-soak-report.v1`: must be `passed`, have no errors and
  only passing checks, and bind the same clean revision, archive, release
  binary, runtime image, model ID/revision, and canonical model-tree digest as
  the Python-free E2E raw evidence. The report must also bind the canonical
  reviewed `pr16-release-soak-v1` template, retain the exact 10-scenario/150-check
  inventory, show every scenario ran for its reviewed duration with samples
  spanning that interval, and retain the reviewed cancellation/disconnect/
  overload and resource-slope bounds. A shortened or threshold-relaxed soak
  report cannot be promoted. Its deterministic uncompressed USTAR must contain
  exactly canonical `manifest.json`, `run.json`, `events.jsonl`, and internal
  `SHA256SUMS` members. The final gate safely materializes those known members,
  rejects any noncanonical tar bytes or checksum/inventory drift, reruns the
  soak checker from that run directory, and requires the recomputed report to
  equal the submitted report exactly. Thus raw event sequencing, final
  allocation/KV quiescence, restart, and rollback golden parity are evidence,
  not self-asserted report fields.

Any missing or extra top-level contract field, duplicate JSON key, non-finite
JSON value, failed check, hash mismatch, source drift, artifact substitution,
or placeholder makes the final decision fail closed. The final report is
evidence that a tag may be considered; tag creation and push remain an
explicit owner action outside this tool.

## CPU-only verification

```sh
python3 -m unittest ci/release/test_release_candidate.py -v
python3 -m unittest ci/release/test_optimization_evidence.py \
  ci/release/test_write_optimization_execution_evidence.py -v
python3 -m unittest \
  benchmarks.scripts.tests.test_check_python_free_release_e2e -v
python3 -m unittest benchmarks.scripts.tests.test_check_reliability_soak -v
python3 -m unittest discover -s ci/release -p 'test_*.py' -v
python3 -m py_compile ci/release/check_release_candidate.py \
  benchmarks/scripts/check_reliability_soak.py \
  benchmarks/scripts/package_reliability_soak_evidence.py
```
