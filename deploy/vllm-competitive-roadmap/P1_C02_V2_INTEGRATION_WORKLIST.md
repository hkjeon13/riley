# C02-P1 v2 raw-provenance integration worklist

Status: initial v4 lifecycle-supervisor/receipt, the source-owned native
sampling fallback event/marker, raw source-pair capture v2, and the separate
terminal fallback binder v5 are implemented with CPU/static hostile-path
coverage; this worklist remains design and integration material, not a
qualification report. No actual GPU capture, candidate freeze, lifecycle-v5
receipt, or semantic qualification has been performed, and this file must not
be copied into a candidate result directory as evidence.

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
and noncanonical manifest/session JSON; each controlled leaf read detects an
inode/path swap during that read.  These are point-in-time per-leaf checks, not
an atomic cross-file snapshot, so the exact-0700 trusted-writer boundary remains
part of the P1 threat model.  It also rejects an unremoved
`capture-incomplete.json` and every v1 input before semantic replay.

### Raw soak manifest binder v3 and serial-scenario binder v4

`bind_raw_c02_soak_v2.py` and its canonical
`riley.soak-v2-bind-request.v3` input remain a closed configuration-process
bridge.  They may bind the existing v3 observation-session shape, but they
**must not** consume `riley.c02-raw-scenario-capture.v1`.  In particular, v3's
caller-declared ledger/runtime/index leaves are opaque, so accepting the new
producer there would make a terminal manifest without replaying the producer's
closure.  Both the v3 binder and verifier therefore reject the v1 serial
contract, request-ledger, and generation-audit-index schema versions before
publication or terminal verification; the generic historical runtime-event
leaf itself remains unchanged.  v2 and v3 must not be widened in place.

The serial non-stream path therefore uses a separate v4 contract:
`riley.soak-v2-bind-request.v4`,
`riley.soak-v2-raw-provenance.v4`, and
`riley.soak-v2-raw-provenance-complete.v4`.  The v4 request has one top-level
`scenario_capture_session_path`, rather than repeating that descriptor in each
scenario; it has no caller-supplied scenario-contract, ledger, runtime-event,
or generation-audit path.  Each v4 scenario contains only its canonical
`scenario_id` and its C02 `observation_session_path`.  The binder derives every
other descriptor from the one session through one held root FD and publishes a
create-only root `NAME.json` plus `NAME.json.complete` marker.

Every v4 `*_session_path` is exactly one direct
`<capture>/session.json` child below the private root.  The source audit
record/marker pair must live in one *different* direct-child audit directory
for the entire serial capture; it cannot borrow the capture directory or mix
audit parents across scenarios.  While a nonblocking exclusive root lock is
held, both terminal output names must be absent before a manifest is created.

Before publication v4 must require the session's parent capture directory to
have no `capture-incomplete.json`, validate the exact v1 session and its
contract inventory, derive each ledger/request/response ID, source audit
record, and hash-bound audit marker, and prove every session PID/start-tick/
loopback-listener tuple agrees with the corresponding observation tuple and
the v3 configuration bridge.  Candidate ID, profile, configuration SHA-256,
completion port, GPU index/UUID, and scenario order/IDs must agree as well.
The full config-bridge/serial/observation tuple replay completes before any
output is created.  v4 publishes its completion marker through a separately
durable, nonterminal intent leaf and a create-only linked final marker: a
pre-publication file-sync failure therefore leaves no final marker at all.
If the final marker has become visible but its parent-directory sync reports
an error, the binder returns the distinct nonzero
`ambiguous-terminal-publication` result and no lifecycle success receipt may
be emitted.  The raw verifier treats that paired intent/final marker shape as
an evidence format only; a later qualification/finalizer must additionally
require the runner's successful-supervisor receipt, rather than interpreting
a visible marker after that ambiguous result as lifecycle authority.
The initial v4 subset rejects `exact-backend-fallback` entirely. Its
source-owned distinct native fallback leaf/marker is now published by Rust.
v1 capture, v4 binder, and the lifecycle runner/receipt still may not
synthesize or consume it. The separate raw capture-v2 branch records the two
source marker pairs as descriptors, and the separate v5 terminal binder now
replays them together with a bound effective `gpu-greedy` `/v1/config` arm.
Both remain `qualification_status: not-run`; lifecycle-v5 receipt, semantic
workload replay, Gate E, and candidate promotion remain later work.

### Landed initial lifecycle supervisor and receipt

`run_remote_c02_soak_v2.sh` is the deliberately narrow host-binary supervisor.
Its clean Python parent creates and authenticates one no-follow, nonblocking
host GPU lock before it launches the Bash child; the child cannot forge its
control sentinel or retain the lock FD. The runner uses `env -i`, creates a new
no-follow private evidence root plus source-audit child, and revalidates the
permitted host binary and model-tree inputs before launch and after process
exit. It owns the C02 `serve` flags and SIGTERM shutdown; it does not accept a
caller-supplied server command, configuration hash, PID/start-tick target, or
GPU UUID.

One invocation freezes exactly one canonical serial non-stream scenario, takes
one immediate C02 observation, and produces at most one v4 raw manifest. Its
terminal writer, `write_c02_lifecycle_supervisor_receipt_v1.py`, is the only
same-process finalizer: it first binds v4, then replays the completed v4
manifest and the source-owned shutdown artifact plus matching marker through
the held private root FD. A v4 `ambiguous-terminal-publication` error therefore
cannot be turned into a later successful lifecycle receipt. The published
`riley.c02-lifecycle-supervisor-receipt.v1` is strictly `status: completed` and
`qualification_status: not-run`.

This is CPU/static hostile-path-tested mechanism code only. Receipt presence
does not establish a GPU capture, candidate freeze, Gate E replay, native
fallback event, rollback result, semantic qualification, or C02 decision.
Native fallback/rollback flows, semantic checker/finalizer work, clean freeze,
and remote GPU qualification remain subsequent gates.

### Landed native sampling fallback source leaf

For a `max-performance-exact` source audit only, Rust now writes
`cmpl-<request-id>.fallback.json` and the matching nonhidden `.complete`
marker only after the matching generation-audit-v2 JSON and marker succeed.
The event binds the exact generation-audit filename/SHA, candidate, runtime
identity, PID/start tick, request ID, and the full ordered projection of
committed request-induced `gpu-greedy` to `cpu-normative` selections. Every
selection must have originated from a one-output-slot scheduler plan; a
multi-output CPU decision may have been caused by a peer request, so it leaves
only the ordinary audit and no fallback pair. The only allowed reasons are
`nonzero-temperature`, `repetition-penalty`, and `finish-token-mask`; cold
configuration incompatibility, a CPU profile, GPU-selected work, cancellation,
abort, and an uncommitted selection produce no terminal fallback event. A
fallback marker collision fails the request and leaves any created event
nonterminal.

This is a source boundary only. It does not make the event a v4 scenario
leaf, does not prove the endpoint actually exposed GPU-greedy, and does not
allow the lifecycle runner/receipt to claim fallback. Capture v2 and binder
v5 must replay both source marker pairs and the endpoint's effective sampling
backend before an `exact-backend-fallback` scenario can be admitted.

### Landed native sampling fallback raw capture v2

The self-contained `capture_c02_raw_soak_scenarios_v1.py` now dispatches by
canonical contract version without widening v1. The new
`c02-raw-soak-runner-contract-v2.schema.json` accepts exactly one
`max-performance-exact`, non-stream `exact-backend-fallback` completion with
`max_tokens: 1`, `top_p: 1`, and exact `temperature: 1`. Pinning temperature
to one prevents a positive Python JSON value from underflowing to zero in the
Rust `f32` request decoder; public completion bytes cannot establish the other
source reasons, so this initial raw arm is closed to `nonzero-temperature`.

For that v2 arm, the response ID derives exactly four sibling leaves under the
same held `source-audit` FD: audit JSON/ordinary marker and native fallback
JSON/ordinary marker. The event must bind the exact audit filename/SHA and
identity tuple, and its ordered selections must exactly equal the audit's
committed `gpu-greedy` → `cpu-normative`, `nonzero-temperature` projection.
It accepts at most the source schema's 65,536 selections. The v2 index preserves
descriptors for all four leaves and the v2 session's `fallback_event_log` is
the original source event descriptor. A missing, unsafe, mismatched, or
nonterminal source leaf leaves `capture-incomplete.json` in place. This is raw
source-pair capture only: it does not use GPU/SSH/Docker, does not change
v1/v4/lifecycle acceptance, and does not bind `/v1/config`.

### Landed native sampling fallback terminal binder v5

`bind_raw_c02_soak_v5.py` is a separate path-only terminal binder, not a
widening of v4. It accepts only one direct `capture/session.json` whose source
session is `riley.c02-raw-scenario-capture.v2`, with exactly one
`max-performance-exact` `exact-backend-fallback` request. Before creating a
manifest it replays the canonical request/response ledger, the four
response-ID-derived ordinary source leaves (audit/event and both markers),
their candidate/runtime/PID/start-tick/request-ID/audit-SHA joins, and every
ordered `gpu-greedy` → `cpu-normative`, `nonzero-temperature`, committed
selection.

The same held private root FD also replays the endpoint/startup/config-bridge
and C02 observation tuple. The v5 branch explicitly requires the validated
endpoint's `effective_config.sampling_backend == "gpu-greedy"`; a profile
label or `fallback_policy` string cannot stand in for that proof. It then
publishes only `riley.soak-v2-raw-provenance.v5` plus its durable v5
`.intent`/`.complete` hardlink pair, and returns `bound`/`not-run`. Source
markers remain ordinary one-link leaves; only this terminal output uses the
paired-hardlink protocol. v1/v4 and the lifecycle-v4 receipt remain fallback
rejecting, and v5 neither launches a service nor uses GPU/SSH/Docker nor makes
a qualification decision.

### Rollback raw compatibility boundary

No authenticated rollback runner or semantic checker is landed yet. The
existing `riley.rc3-rollback-raw-provenance.v2` verifier is only a raw descriptor
replayer: it binds stable-default, distinct candidate/rollback process tuples,
the candidate shutdown-v2 pair, artifact maps, and five opaque switch leaves.
It does not interpret health/generation/audit bytes, baseline identity, or
atomic-rename success, and it has no terminal completion-marker protocol.

More importantly, v2 requires a C02 observation-session-v2 for both phases.
The reconstructed `riley-0.1.0-rc2` target (`6093006…`) has no
`/v1/c02/metrics`, `--c02-audit-dir`, generation-audit-v2, or shutdown-v2
surface. It is therefore incompatible with a direct v2 rollback capture.
Synthesizing those leaves around the legacy binary, accepting a v1 receipt, or
up-converting an old trace would violate the provenance boundary.

The distinct v3 raw schema/checker is now landed for the reconstructed-tag
case rather than widening v2. `check_rc3_rollback_provenance_v3.py` replays
the full reconstructed baseline A/B manifest through one held FD, pins the
reviewed RC2 tag target, and checks each declared phase PID/start tick,
listener port/inode, and GPU tuple against its held-FD `/proc`, TCP/FD-socket,
and GPU raw leaves. It records declared candidate audit availability plus its
opaque audit-index descriptor separately from the baseline's explicit
`not-supported` audit state; source-audit content remains a later replay
responsibility. It also binds candidate shutdown-v2, active baseline
bundle/image, and raw atomic-switch material. It returns only `bound`/`not-run`;
it does not evaluate HTTP or rename meaning and has no terminal marker.

The path-only v3 binder is landed: it derives every descriptor and phase tuple
from one held private root FD, replays the full reconstructed baseline, and
publishes only a self-verified nonterminal `captured/not-run` manifest. The
RC2-compatible `capture_rc3_rollback_phase_v1.py` companion appends a new raw
phase directory to that same prepopulated root and never creates a root or
claims source audit for RC2. Its candidate no-generation mode deliberately
leaves source-owned generation/audit capture to the existing scenario producer.
The fixed-layout `prepare_rc3_rollback_artifacts_v1.py` companion is also
landed.  It accepts six absolute host artifact inputs, opens every source
parent through no-follow FDs, and streams each trusted nonempty single-link
root/euid-owned non-group/world-writable input into a new immutable mode-0600
leaf below `rollback-v3-artifacts/`.  It then creates distinct single-link
mode-0700 `rollback-v3-switch/{active,rollback-staged}` copies only from the
two new binary snapshots, not by reopening the host sources.  Its fixed
create-only children and terminal session verifier record the runtime-to-
snapshot hash/length/inode mapping; runtime paths are deliberately not v3
artifact descriptors. Terminal evidence requires both an absent incomplete
marker and a mode-0600 two-link `capture-complete.intent`/
`capture-complete.json` receipt pair bound to `session.json` SHA-256 and byte
length—absence alone is not completion. It does not replay the reconstructed
baseline or make a bind request, so an authenticated runner must still require
the full prepopulated baseline closure and replay this session plus its
completion receipt.
Its verifier takes a short shared switch lock only while replaying one explicit
`pre-switch` or `post-switch` layout; `post-switch` proves byte/inode mapping,
not that an exchange occurred. Both helpers now also expose held-switch-FD
cores. `capture_rc3_rollback_atomic_transaction_v1.py` holds one exclusive
evidence-root and switch FD across pre-switch preparation replay, the isolated
exchange, terminal atomic replay, and post-switch preparation replay. It
create-only records `rollback-v3-atomic-transaction/session.json`, binds the
two terminal child sessions, and joins the candidate/rollback pre/post runtime
SHA-256·identity layouts to the atomic pre/post stat leaves. The atomic helper
hashes private runtime bytes on both sides of the exchange, so a same-inode,
same-size in-place mutation cannot become terminal evidence. Its replayer
rechecks the session-bound completion receipt, incomplete state, and child-FD
identity; all records remain `captured/not-run`.
If the completion hardlink's post-link parent-directory sync fails, the helper
raises `ambiguous-terminal-publication` and returns no producer success. A
visible pair after that error is structurally replayable raw evidence only;
the future authenticated binder/runner must consume only the normal-return
branch of the same invocation under its held locks, never resume from a fresh
verification of that ambiguous on-disk pair.
The companion `capture_rc3_rollback_atomic_switch_v1.py` applies one
same-directory Linux `renameat2(RENAME_EXCHANGE)` only inside a runner-owned
isolated evidence-root child, never an external deployment path, and captures
the five opaque raw switch leaves. This closes only the artifact-exchange
subtransaction: it is not an authenticated host supervisor and does not bind
phase/source/config evidence, launch or stop a service, or touch a deployment
path. A future runner must still replay terminal phase/source/config sessions,
their absent `capture-incomplete.json` markers, and their session-bound paired
completion receipts before constructing a bind request; the raw v3 binder does
not infer those boundaries from arbitrary leaf paths. The v3 manifest also does
not retain their closure, so a later
semantic/terminal version must close it.

Before an actual runner exists,
`verify_rollback_provenance_fd()` keeps v2 replay on one held private root FD,
and the v3 verifier does the same for its own full baseline replay. Its path
wrapper also rejects a root under this source checkout; an FD-only caller must
perform that preflight before passing the held FD. Neither API is compatibility
approval or qualification evidence.

## Required raw evidence inventory

| Scope | Create-only raw leaves to bind |
| --- | --- |
| Soak configuration arm | canonical raw `/v1/config` endpoint and startup artifact plus config bridge session; bind candidate/profile/runtime identity, exact embedded endpoint payload/hash, and pre/post PID/listener/GPU tuple equal to every scenario. |
| Every observation sample | metrics response bytes; `/proc/<pid>/stat` before and after; `/proc/net/tcp` before and after; PID FD-to-socket snapshots before and after; `/proc/<pid>/status`; GPU **index+UUID** selection-query output; GPU compute-apps output; canonical sample/session descriptors. |
| Every bound target | Derive, do not assert: `{server_pid, server_start_ticks, listener_inode, gpu_index, gpu_uuid}`.  Match both stat snapshots, both TCP snapshots, both FD socket snapshots, and the GPU PID row. |
| Soak scenario | raw HTTP request/response ledger, native runtime event log, generation-audit index, and exact-backend-fallback event log when that scenario runs.  Those fields stay generic descriptors so the Rust sampling audit remains the source of event payload semantics. |
| Candidate shutdown | v2 shutdown artifact with PID **and start ticks**, final raw C02 metrics, plus a create-only matching completion marker whose hash covers the exact artifact bytes.  Use a nonhidden v2 marker name such as `shutdown.json.complete`; v1's hidden marker is historical-only. |
| Rollback drill | candidate and rollback observation sequences, both HTTP/audit leaves, raw binary/bundle/image-inspect evidence for each phase, and raw `pre_active_stat`, `post_active_stat`, `candidate_staged_stat`, `rollback_staged_stat`, and successful rename transcript. |

### Landed serial raw-scenario producer

`c02-raw-soak-runner-contract-v1.schema.json`, the published
`c02-raw-scenario-capture-v1.schema.json`, and the self-contained
`capture_c02_raw_soak_scenarios_v1.py` close the first narrow producer gap for
an **already-running** single host process.  The canonical contract binds one
serial, non-stream `/v1/completions` request per scenario.  The producer keeps
the exact emitted request bytes, response head/body bytes, a canonical ledger,
and a canonical index over the source-written `cmpl-*.json` audit record plus
its hash-bound completion marker.  It also keeps pre/post/final raw
`/proc/<pid>/stat`, `/proc/net/tcp`, and PID FD-to-socket snapshots so the
literal completion port is tied to the requested process while the audit
marker becomes visible.  Its `runtime_event_log` is that original source audit
record, never a wrapper-generated summary.  GPU evidence remains solely with
the existing observation producer under the lifecycle runner's held lock; this
local helper does not query or operate a GPU.

The versioned v2 branch reuses that transport and held-FD boundary rather than
copying a second producer. Its new contract/capture/index schemas record only
the closed native fallback source-pair arm described above. v1 output remains
fallback-free and byte-shape compatible with the existing v4 replay; v2 is not
input to v4 or the initial lifecycle runner.

This helper cannot start/stop a service, acquire a GPU lock, use Docker/SSH,
or issue a qualification result. Its v1 branch deliberately rejects streaming,
restart/rollback/multi-PID semantics, and `exact-backend-fallback`; v1/v4 do
not replay the newly published source fallback pair. The landed initial lifecycle
runner performs the GPU/port/process preflight, config bridge, one immediate
C02 metrics observation, and graceful shutdown around exactly one producer
scenario before it reaches the same-process raw finalizer.

The landed first lifecycle runner is deliberately narrower than the producer:
one runner invocation owns one canonical scenario, one immediately following
C02 observation, one v4 manifest, and one raw-only lifecycle receipt. A
multi-scenario producer capture followed by delayed observations cannot prove
per-scenario timing, so aggregate or interleaved soak remains a later
v5/semantic contract rather than an implied property of v4.

`check_c02_config_bridge_v1.py` publishes the pure held-FD replay boundary used
by the landed lifecycle runner for the existing config bridge. It
accepts only a private evidence root, endpoint/startup/session paths, expected
candidate ID, and expected profile; the session is a direct
`<capture>/session.json` child and the helper derives (rather than accepts)
the configuration SHA-256 and observed PID/start-tick/listener/GPU tuple.
Its canonical stdout report is diagnostic `bound` / `not-run` data, never an
evidence leaf or a qualification verdict, and it must not invoke GPU, network,
or subprocess tooling.  Its exact diagnostic report shape is published as
`benchmarks/release/candidates/c02-config-bridge-replay-v1.schema.json`; the
runner consumes the derived configuration SHA and target only from that report,
never from caller input.

The v4 binder is the first terminal consumer for this producer: it takes one
explicit `scenario_capture_session_path`, rejects a retained
`capture-incomplete.json`, replays its source-audit marker/hash and contract
inventory, and compares its PID/listener proof with the C02 observation tuple.
The initial runner reaches it only through the same-process receipt writer; the
producer is not a v3 candidate-capture path.

### Shutdown v2 leaf contract

The source-owned producer and raw binder share these published schemas:

- `benchmarks/release/candidates/c02-capture-metrics-v2.schema.json`
  (`riley.c02-capture-metrics.v2`);
- `benchmarks/release/candidates/c02-shutdown-quiescence-v2.schema.json`
  (`riley.c02-shutdown-quiescence.v2`); and
- `benchmarks/release/candidates/c02-shutdown-quiescence-completion-v2.schema.json`
  (`riley.c02-shutdown-quiescence-complete.v2`).

The shutdown artifact has exactly seven top-level fields:
`schema_version`, `capture_status`, `qualification_status`, `server_pid`,
`server_start_ticks`, `worker_ready`, and `final_metrics`.  Its only allowed
statuses are `captured` and `not-run`; `worker_ready` is exactly `false`.
`final_metrics` is exactly the v2 metrics object with `request_states`
(`active`, `pending_requests`, `completed`, `failed`, `cancelled`,
`capacity_rejections`), `kv_blocks` (`free`, `reserved`, `active`),
`allocation` (`device_live_count`, `device_live_bytes`,
`pinned_live_count`, `pinned_live_bytes`), and `quiescence`
(`completion_outbox`, `outstanding_iterations`,
`riley_owned_live_allocations`, `worker_accepting`, `scheduler_accepting`).
The raw schema checks exact field names and types only; semantic thresholds
remain a later checker concern.

The matching marker has exactly `schema_version`, `artifact_filename`, and
`artifact_sha256`.  `artifact_filename` is a nonhidden direct-child JSON leaf
and the physical marker name is exactly `<artifact_filename>.complete` in the
same held private root.  It is written only after the artifact file is
`fsync`ed, contains the SHA-256 of those exact artifact bytes, and is itself
`fsync`ed before the root directory is synced.  A v1 artifact or hidden
`.<basename>.complete` marker is historical-only and rejected by v2 input.

The artifact leaf is exactly
`^[A-Za-z0-9][A-Za-z0-9._-]{0,240}\\.json$`: ASCII, nonhidden, and at most
246 bytes so its sibling `.complete` stays within the POSIX 255-byte filename
limit.  The term “direct-child” is relative to the source writer's held
private audit-root FD, not necessarily the global evidence root; thus a
descriptor such as `candidate-phase/shutdown.json` remains valid.  The writer
opens that root at startup through a no-follow component chain, keeps its FD,
and never reopens the pathname during shutdown.  A pathname rename/swap must
not redirect publication to a replacement directory or make that replacement
an equivalent evidence root.

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

2. `ci/release/capture_c02_observations_v2.py` and
   `ci/release/run_remote_c02_observations_v2.sh`
   - Keep their v1 predecessors historical-only.  The v2 producer must remain
     self-contained because the runner invokes Python with `-I -S`.
   - Emit v2 session/sample/socket schema names and descriptors with lengths;
     retain the raw sample inventory above and the incomplete marker closure.
     Change the v1 UUID-only GPU query to a raw `index,uuid` selection query,
     so the declared GPU index is also independently proven.
   - Require a new external 0700 evidence root, GPU UUID/memory preflight,
     exclusive lock, loopback-only target, and `env -i`.  No source-tree
     output and no public `/metrics` fallback.

3. `crates/riley-server/src/main.rs`, `crates/riley-server/src/service.rs`,
   and `crates/riley-server/src/engine.rs`
   - Implement the published v2 shutdown artifact/marker schemas; persist
     `server_start_ticks` with the PID and bind the exact artifact hash in a
     create-only nonhidden `<artifact>.complete` marker.  Do not add candidate,
     freeze, Gate E, semantic, or pass/fail fields to the exact seven-field
     artifact.
   - Replace `C02GenerationAuditWriter`'s `Path` plus pending-file/hard-link
     flow with a private 0700 root held directory FD and direct-child
     `O_CREAT|O_EXCL|O_NOFOLLOW` create-only writes.  A sampling record should
     be committed as `<request-id>.json` then matching nonhidden
     `<request-id>.json.complete`, both durable.
   - Add native per-request audit events needed by the reviewed backend
     fallback scenario.  Static `cross_profile_fallback: forbidden` config is
     not an event.

4. `ci/release/bind_raw_c02_soak_v4.py`,
   `capture_c02_raw_soak_scenarios_v1.py`,
   `ci/release/run_remote_c02_soak_v2.sh`, and
   `ci/release/write_c02_lifecycle_supervisor_receipt_v1.py`
   - First publish the v4 request/manifest/marker schemas, strict binder, and
     hostile fixture tests.  The completed serial binder emits
     `riley.soak-v2-raw-provenance.v4`, never a semantic receipt.  Its local
     wrapper does not start/stop a service or invoke GPU/SSH/container tools.
   - The landed serial scenario producer is local-to-an-already-running
     service and emits no bind request or terminal manifest.  It is the only
     layer that may pair a public response ID to the source-owned audit record
     for this initial non-stream subset.  v3 must not consume it.  v4 derives
     contract/ledger/runtime/index leaves from one explicit capture session
   and admits only its serial non-stream scenarios; it rejects
   `exact-backend-fallback`. The separate capture-v2 branch preserves the
   native source pair, and the separate terminal binder v5 replays it with the
   config bridge's effective `gpu-greedy` setting without widening v4.
   - Bind each declared scenario target to its raw observation session and the
     v4-derived request/runtime/generation leaves.  The raw soak binder accepts
     only `stable-default` or `max-performance-exact` and remains
     `qualification_status: not-run`.
   - The landed first lifecycle invocation is single-scenario: config bridge,
     producer, one immediate observation, shutdown, one v4 manifest, then the
     same-process receipt binding v4 and the shutdown artifact/marker. It is
     `completed`/`not-run` raw mechanism code only; do not claim per-scenario
     timing, GPU capture, freeze, or qualification from it.
   - Bind canonical endpoint/startup configuration bytes and derive the arm
     identity from the endpoint runtime identity. The isolated
     `capture_c02_config_endpoint_observation_v1.py` captures the required
     same-process bridge; the initial runner invokes it before binding scenario
     material.
   - Expose that bridge through `check_c02_config_bridge_v1.py`: strict
     held-FD replay of direct endpoint/startup/session paths derives the
     configuration SHA and observed target for the lifecycle runner.  It takes
     no caller-supplied SHA or target tuple and performs no operational action.
     Publish its closed canonical stdout report schema alongside the helper.

5. `ci/release/run_remote_rc3_rollback_capture.sh` (new) and
   `ci/release/bind_raw_rc3_rollback_capture.py` (new)
   - Capture candidate/rollback target tuples and reject equal
     `(pid,start_ticks)` identities.
   - Publish `rollback-bind-request-v3.schema.json` as a closed path-only
     local binder input. The binder derives every manifest descriptor and both
     target tuples through one private held root FD; it does not accept a
     caller target, descriptor, hash, audit availability, profile, verdict,
     or operational action.
   - Bind rollback evidence only to `stable-default`; max-performance-exact
     is opt-in soak evidence, never a rollback arm.
   - Bind candidate shutdown artifact+marker, both phase artifact maps, and
     all raw atomic-switch leaves.  It still returns `not-run`; the later
     semantic checker owns filesystem and health/generation interpretation.
   - The raw v3 manifest preserves the three binding inputs only as SHA-256
     scalars because that retained schema predates this binder. The binder
     derives those scalars from raw leaves at bind time, but v3 does not retain
     their descriptors for independently replaying those input files. It
     publishes no completion marker and reserves matching `.complete` and
     `.intent` names against stale terminal-looking siblings.

6. `ci/release/check_soak_v2_receipt.py`,
   `ci/release/check_rc3_rollback_receipt.py`, and
   `ci/release/check_rc3_qualification.py`
   - Keep v1 checkers/schemas historical but do not upconvert or accept them.
   - Add v2 raw-binder invocation first, then v2 semantic replay.  The outer
     qualification checker accepts only exact v2 report versions and produces
     an explicit historical-v1 rejection reason before generic gate failure.
   - Replace all existing no-follow fallback flag patterns.

7. `benchmarks/release/candidates/c02-raw-scenario-capture-v1.schema.json`,
   `benchmarks/release/candidates/soak-v2-bind-request-v4.schema.json`,
   `benchmarks/release/candidates/soak-v2-receipt-v4.schema.json`,
   `benchmarks/release/candidates/soak-v2-bind-request-v5.schema.json`, and
   `benchmarks/release/candidates/soak-v2-receipt-v5.schema.json`,
   `benchmarks/release/candidates/c02-lifecycle-supervisor-receipt-v1.schema.json`,
   `benchmarks/release/candidates/c02-config-endpoint-observation-v1.schema.json`,
   `benchmarks/release/candidates/rollback-receipt-v2.schema.json`,
   `benchmarks/release/candidates/README.md`, and
   `deploy/vllm-competitive-roadmap/02-rc3-candidate-qualification.md`
   - Publish v4 serial raw-manifest/raw-binding schemas and the separate
     raw-only lifecycle receipt schema independently from later semantic
     receipts. Retained v2/v3 schemas are historical for the serial capture
     path and must not be accepted or upconverted.
   - State that v1/v2 historical soak artifacts are rejected and that raw binders do
     not qualify a candidate.  Any reconstructed baseline must be labelled
     reconstructed: remote history has RC tags/C02 dev images but no verified
     historical release bundle/image to call a shipped rollback source.

## Implementation order

1. Land strict common primitives plus focused hostile tests.
2. Land Rust private-FD v2 audit/shutdown producer and unit tests.
3. Land the v4 serial-session binder/schema and hostile fixture tests before
   the initial lifecycle runner.
4. Land the one-scenario v2 observation/remote lifecycle runner and its
   same-process v4/shutdown receipt closure; retain CPU/static-only scope.
5. Land raw capture v2 and its separate terminal binder v5 for the
   already-published native fallback source leaf (complete), then add rollback
   raw capture/binding and a separately versioned lifecycle/semantic closure.
6. Land semantic soak/rollback replay and outer qualification v2-only policy.
7. Freeze only the clean source revision after all mechanism tests pass; then
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
