# Final release-candidate evidence gate

`check_release_candidate.py` is the last CPU-only gate before an owner creates
a release tag. It does not build an image, start the server, load a model, use
CUDA, create a tag, or push anything. It consumes immutable artifacts produced
by the earlier gates and emits a single source-bound decision.

Run it from any Python 3.11+ host after copying the complete evidence set into
one read-only directory:

```sh
python3 ci/release/check_release_candidate.py \
  --manifest /evidence/release-candidate.json \
  --evidence-root /evidence \
  --report /evidence/release-candidate-report.json
```

The report path is create-only. Only `status=passed` and `passed=true` exits
zero. A passed report binds the SHA-256 of the exact input manifest, source
archive, release binary, release bundle, six gate decisions, six raw/replay
archives, and the immutable release image digest.

## Closed candidate manifest

The manifest schema version is
`rustinfer.release-candidate-manifest.v1`. Every `path` is a normalized POSIX
path relative to `--evidence-root`; absolute paths, `..`, links, devices,
duplicate paths, and paths resolving outside that root are rejected. Every
file descriptor is exactly `{"path": ..., "sha256": ...}`. The complete
shape is:

```json
{
  "schema_version": "rustinfer.release-candidate-manifest.v1",
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
      "raw_evidence": {"path": "python-free-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "cuda_fault": {
      "build_image_id": "sha256:LOWERCASE_SHA256",
      "report": {"path": "cuda-fault-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "cuda-fault-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "native_correctness": {
      "report": {"path": "native-correctness-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_replay": {"path": "native-correctness-replay.tar", "sha256": "LOWERCASE_SHA256"},
      "replay_validation": {"path": "native-replay-validation.json", "sha256": "LOWERCASE_SHA256"}
    },
    "optimization_correctness": {
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

The source archive must be the exact output of `git archive --format=tar` for
the clean revision named by `git_revision`. Besides hashing the bytes, the
checker opens the tar safely and requires the pax global `comment` and every
member's inherited comment to equal that revision. A tar containing the right
files but lacking the Git commit marker is not evidence. The deterministic
release bundle is then passed through `verify_release_bundle.py`; its embedded
source revision must match and its embedded `bin/rustinfer` must be
byte-identical to the separately supplied executable. The image is identified
only by an immutable `sha256:` image ID, which every runtime report must repeat
exactly.

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
the candidate manifest. For Python-free E2E, the tar excludes the attestation
to avoid a circular hash and contains exactly `raw-evidence.json`, the reviewed
golden, canonical model checksum manifest, two shutdown-metrics documents, and
an internal `SHA256SUMS`. The final candidate opens that tar, validates its
closed inventory and canonical metadata, replays all runtime observations,
recomputes model/tokenizer/shutdown bindings, and requires the replayed
attestation object to equal the submitted object exactly. It also cross-binds
the native five-file tokenizer aggregate and optimizer `tokenizer.json` hash.
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
  that report is not approval. `raw_replay` preserves the replay inputs/output,
  and `replay_validation` must bind both exact byte hashes and attest that the
  closed schema, raw hashes, all cases, and summary were independently replayed;
- optimizer equivalence report schema `1`, gate
  `pr15-iteration-command-batch-exact-v1`: must be passed E0 evidence for the
  same source/archive, pinned SmolLM2 BF16 artifacts, network-none locked/offline
  CUDA sm89 build, and exact `per-operation` versus `iteration-batch` flags with
  `residual_rmsnorm=separate`. Its exact five-test inventory and every expected
  zero-mismatch/allocation result are checked. The raw evidence tar must contain
  exactly these regular log files, whose bytes must match the report hashes:
  `cuda-compile-only.log`, `workspace-all-features-all-targets.log`,
  `command-batch-lifecycle-gpu.log`, `command-batch-primitives-gpu.log`, and
  `iteration-command-batch-model-parity-gpu.log`.

The closed native replay validation shape is:

```json
{
  "schema_version": "rustinfer.native-correctness-replay-validation.v1",
  "status": "passed",
  "source": {
    "git_revision": "FULL_LOWERCASE_40_CHARACTER_COMMIT",
    "git_dirty": false,
    "source_archive_sha256": "LOWERCASE_SHA256"
  },
  "correctness_report_sha256": "LOWERCASE_SHA256",
  "raw_replay_sha256": "LOWERCASE_SHA256",
  "case_count": 31,
  "failure_count": 0,
  "checks": [
    {"id": "schema-closed-validation", "passed": true},
    {"id": "raw-input-hashes-replayed", "passed": true},
    {"id": "all-cases-replayed", "passed": true},
    {"id": "summary-recomputed", "passed": true}
  ]
}
```

The remaining cross-bindings are:

- `rustinfer.release-performance-report.v1`: must be `passed`, have no errors,
  have the exact four reviewed passing checks, select semantic class E0, bind the exact source
  archive/release binary/runtime image, and bind the exact optimizer
  equivalence report bytes and gate ID. Its profile image must equal the
  optimizer report's build image. It must not bind the native 31-case report
  in that field. Its raw evidence tar must contain only
  `candidate-1.json` through `candidate-5.json`; the final gate revalidates the
  closed native profile schema and all source/model/environment/workload/raw
  hashes, then recomputes the R7 metrics, baseline ratios, and thresholds;
- `rustinfer.reliability-soak-report.v1`: must be `passed`, have no errors and
  only passing checks, and bind the same clean revision, archive, release
  binary, and runtime image. The report must also bind the canonical reviewed
  `pr16-release-soak-v1` template, retain the exact 10-scenario/150-check
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
python3 -m unittest benchmarks.scripts.tests.test_check_reliability_soak -v
python3 -m unittest discover -s ci/release -p 'test_*.py' -v
python3 -m py_compile ci/release/check_release_candidate.py \
  benchmarks/scripts/check_reliability_soak.py \
  benchmarks/scripts/package_reliability_soak_evidence.py
```
