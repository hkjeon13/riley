# C02-P1 v2 raw-provenance integration worklist

Status: read-only source audit plus `/tmp` design material.  This is **not** a
qualification report and must not be copied into a candidate result directory
as evidence.

## Boundary to preserve

The v2 raw binder returns only `status: "bound"` and
`qualification_status: "not-run"`.  It proves exact raw leaves and process
ownership tuples; it validates metric field types but does not impose a
failure/KV/quiescence threshold.  A later semantic checker is the only layer
that may replay the reviewed workload/Gate E and issue a pass/fail result.
This prevents a Python wrapper or a self-authored trace from becoming the
audit source of truth.

Every v2 descriptor is exactly:

```json
{"path":"...","sha256":"<lowercase sha256>","byte_length":123}
```

The binder must hold one no-follow FD for an external, euid-owned, exact-0700
evidence root for its entire parse.  It rejects symlinks, hard links, aliases,
inode swaps, noncanonical manifest/session JSON, an unremoved
`capture-incomplete.json`, and every v1 input before semantic replay.

## Required raw evidence inventory

| Scope | Create-only raw leaves to bind |
| --- | --- |
| Every observation sample | metrics response bytes; `/proc/<pid>/stat` before and after; `/proc/net/tcp` before and after; PID FD-to-socket snapshots before and after; `/proc/<pid>/status`; GPU **index+UUID** selection-query output; GPU compute-apps output; canonical sample/session descriptors. |
| Every bound target | Derive, do not assert: `{server_pid, server_start_ticks, listener_inode, gpu_index, gpu_uuid}`.  Match both stat snapshots, both TCP snapshots, both FD socket snapshots, and the GPU PID row. |
| Soak scenario | raw HTTP request/response ledger, native runtime event log, generation-audit index, and exact-backend-fallback event log when that scenario runs.  Those fields stay generic descriptors so the Rust sampling audit remains the source of event payload semantics. |
| Candidate shutdown | v2 shutdown artifact with PID **and start ticks**, final raw C02 metrics, plus a create-only matching completion marker whose hash covers the exact artifact bytes.  Use a nonhidden v2 marker name such as `shutdown.json.complete`; v1's hidden marker is historical-only. |
| Rollback drill | candidate and rollback observation sequences, both HTTP/audit leaves, raw binary/bundle/image-inspect evidence for each phase, and raw `pre_active_stat`, `post_active_stat`, `candidate_staged_stat`, `rollback_staged_stat`, and successful rename transcript. |

The eventual atomic-switch semantic checker must reconstruct same-device and
inode transitions from those raw stat/transcript bytes; a JSON
`"strategy":"atomic-rename"` string is not proof.  The exact-backend
fallback checker must derive it from a native event, never from a static
runtime configuration field or a trace counter.

## File-by-file source work

1. `ci/release/provenance_v2_common.py` (new)
   - Use strict `O_NOFOLLOW`, `O_DIRECTORY`, `O_CLOEXEC`, and `O_NONBLOCK`;
     missing flags fail closed.
   - Add absolute no-follow chain traversal and
     `open_private_evidence_directory()` (euid owner, exact 0700 terminal
     root, safe ancestor rule, held FD).
   - Provide canonical JSON, bounded regular/single-link read with inode
     stability, unique descriptors, and create-only 0600+fsync leaf/marker
     helpers.  Do not use `getattr(..., 0)` fallbacks.

2. `ci/release/capture_c02_observations.py` and
   `ci/release/run_remote_c02_observations.sh`
   - Preserve v1 only for history; add a self-contained v2 producer because
     the runner invokes Python with `-I -S`.
   - Emit v2 session/sample/socket schema names and descriptors with lengths;
     retain the raw sample inventory above and the incomplete marker closure.
     Change the v1 UUID-only GPU query to a raw `index,uuid` selection query,
     so the declared GPU index is also independently proven.
   - Require a new external 0700 evidence root, GPU UUID/memory preflight,
     exclusive lock, loopback-only target, and `env -i`.  No source-tree
     output and no public `/metrics` fallback.

3. `crates/riley-server/src/service.rs` (and the startup plumbing that owns
   the server PID)
   - Introduce v2 shutdown artifact/marker schemas; persist `server_start_ticks`
     with the PID and bind the exact artifact hash in a create-only marker.
   - Replace `C02GenerationAuditWriter`'s `Path` plus pending-file/hard-link
     flow with a private 0700 root held directory FD and direct-child
     `O_CREAT|O_EXCL|O_NOFOLLOW` create-only writes.  A sampling record should
     be committed as `<request-id>.json` then matching nonhidden
     `<request-id>.json.complete`, both durable.
   - Add native per-request audit events needed by the reviewed backend
     fallback scenario.  Static `cross_profile_fallback: forbidden` config is
     not an event.

4. `ci/release/run_remote_c02_soak_v2.sh` (new) and
   `ci/release/bind_raw_c02_soak_v2.py` (new)
   - The runner captures raw scenario material only; the binder emits
     `riley.soak-v2-raw-provenance.v2`, never a semantic receipt.
   - Bind each declared scenario target to its raw observation session and
     request/runtime/generation/fallback descriptor leaves.  Require fallback
     leaf presence for `exact-backend-fallback`.

5. `ci/release/run_remote_rc3_rollback_capture.sh` (new) and
   `ci/release/bind_raw_rc3_rollback_capture.py` (new)
   - Capture candidate/rollback target tuples and reject equal
     `(pid,start_ticks)` identities.
   - Bind candidate shutdown artifact+marker, both phase artifact maps, and
     all raw atomic-switch leaves.  It still returns `not-run`; the later
     semantic checker owns filesystem and health/generation interpretation.

6. `ci/release/check_soak_v2_receipt.py`,
   `ci/release/check_rc3_rollback_receipt.py`, and
   `ci/release/check_rc3_qualification.py`
   - Keep v1 checkers/schemas historical but do not upconvert or accept them.
   - Add v2 raw-binder invocation first, then v2 semantic replay.  The outer
     qualification checker accepts only exact v2 report versions and produces
     an explicit historical-v1 rejection reason before generic gate failure.
   - Replace all existing no-follow fallback flag patterns.

7. `benchmarks/release/candidates/soak-v2-receipt-v2.schema.json`,
   `benchmarks/release/candidates/rollback-receipt-v2.schema.json`,
   `benchmarks/release/candidates/README.md`, and
   `deploy/vllm-competitive-roadmap/02-rc3-candidate-qualification.md`
   - Publish v2 raw-manifest and raw-binding report schemas separately from
     v2 semantic receipts.
   - State that v1 artifacts are historical/rejected and that raw binders do
     not qualify a candidate.  Any reconstructed baseline must be labelled
     reconstructed: remote history has RC tags/C02 dev images but no verified
     historical release bundle/image to call a shipped rollback source.

## Implementation order

1. Land strict common primitives plus focused hostile tests.
2. Land Rust private-FD v2 audit/shutdown producer and unit tests.
3. Land v2 observation/remote capture runners and raw binders.
4. Land schema + binder unit tests, including known-good fixtures only in
   temporary/nonqualification paths.
5. Land semantic soak/rollback replay and outer qualification v2-only policy.
6. Freeze only the clean source revision after all mechanism tests pass; then
   capture candidate evidence on the remote GPU host.

## Minimum adversarial tests

- missing `O_NOFOLLOW`/`O_DIRECTORY`, unsafe root mode/owner/ancestor, final
  and intermediate symlink, hard link, descriptor collision, inode swap,
  oversized/noncanonical/duplicate-key JSON;
- lingering incomplete marker, marker filename/hash mismatch, candidate PID
  reuse, start-tick mismatch, listener inode mismatch, GPU UUID/PID mismatch;
- absent exact-fallback event leaf, v1 manifest/session/shutdown/marker input,
  ambiguous raw artifact path, and fake atomic rename whose inode/stat proof
  does not reconstruct the claimed replacement.

## Existing v1 audit findings

- `capture_c02_observations.py` is raw-only v1 and has `getattr(..., 0)` open
  flag construction despite other private-root checks.
- `check_soak_v2_receipt.py` and `check_rc3_rollback_receipt.py` consume
  self-authored v1 trace/timeline fields; no remote soak or rollback raw
  producer exists.
- `check_rc3_qualification.py` imports those v1 report versions and also has
  fallback open flags.
- `C02ShutdownArtifactWriter` v1 is close to safe but records PID without
  start ticks and uses a hidden completion marker.  `C02GenerationAuditWriter`
  opens paths and stages via hard links rather than retaining a root FD.
