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
archive, release binary, release bundle, all five gate reports, both raw log
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
      "report": {"path": "cuda-fault-report.json", "sha256": "LOWERCASE_SHA256"},
      "raw_evidence": {"path": "cuda-fault-evidence.tar", "sha256": "LOWERCASE_SHA256"}
    },
    "correctness": {"report": {"path": "correctness-report.json", "sha256": "LOWERCASE_SHA256"}},
    "performance": {"report": {"path": "release-performance-report.json", "sha256": "LOWERCASE_SHA256"}},
    "reliability_soak": {"report": {"path": "reliability-soak-report.json", "sha256": "LOWERCASE_SHA256"}}
  }
}
```

The uppercase values above document positions only; they are deliberately not
accepted as evidence. All-zero digests/revisions and common placeholder text
are rejected.

The source archive must be the exact clean revision named by `git_revision`.
The checker verifies its bytes against the declared digest. The deterministic
release bundle is then passed through `verify_release_bundle.py`; its embedded
source revision must match and its embedded `bin/rustinfer` must be byte-identical
to the separately supplied executable. The image is identified only by an
immutable `sha256:` image ID, which every runtime report must repeat exactly.

## Log-backed gate attestation

The Python-free clean-runtime and CUDA fault runners currently produce logs
and `SHA256SUMS`, not a common JSON result. Preserve each output directory as a
deterministic tar file (do not discard its internal checksum manifest), hash
that tar, and create this closed standard-library-readable attestation:

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
the candidate manifest.

## Existing report contracts and cross-bindings

The other three inputs remain in their existing formats:

- native correctness report `1.0.0`, gate
  `smollm2-fp32-bf16-native-e0-v2`: must pass all 31 cases and both E0
  variants from `benchmarks/schemas/correctness-report.schema.json`, bind the
  exact candidate revision, and record the clean-tree digest;
- `rustinfer.release-performance-report.v1`: must be `passed`, have no errors,
  have only passing checks, select semantic class E0, bind the exact source
  archive/release binary/runtime image, and bind the exact correctness report
  bytes;
- `rustinfer.reliability-soak-report.v1`: must be `passed`, have no errors and
  only passing checks, and bind the same clean revision, archive, release
  binary, and runtime image. Its own checker remains responsible for scenario
  presence, final allocation/KV quiescence, resource slopes, sample gaps,
  restart, and rollback golden parity.

Any missing or extra top-level contract field, duplicate JSON key, non-finite
JSON value, failed check, hash mismatch, source drift, artifact substitution,
or placeholder makes the final decision fail closed. The final report is
evidence that a tag may be considered; tag creation and push remain an
explicit owner action outside this tool.

## CPU-only verification

```sh
python3 -m unittest ci/release/test_release_candidate.py -v
python3 -m unittest discover -s ci/release -p 'test_*.py' -v
python3 -m py_compile ci/release/check_release_candidate.py
```
