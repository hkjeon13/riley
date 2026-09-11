# RC3 candidate-bound qualification envelope

This directory defines the **static contract** for C02.  It does not contain
an RC3 candidate, a release decision, raw GPU evidence, model files, or an
assertion that Riley passes any gate.

The [C02-P0 effective runtime-configuration receipt amendment](c02-p0-runtime-config-receipt.md)
is the required implementation prerequisite for `startup_configuration`; its
raw-endpoint versus post-capture semantic-binding split is normative here.

`rc3-candidate.json` is deliberately an unfilled template.  It has
`template_only: true`, `status: "template"`, and no digest, revision, candidate
ID, or evidence path.  It is not input to
`ci/release/check_rc3_qualification.py` and must never be relabelled as a
frozen candidate.  The template is validated by
`rc3-candidate-template.schema.json`; the checker's frozen input and output
are described by `rc3-qualification.schema.json`.

## Freeze protocol

Before any C02 gate starts, an authorized external orchestrator must make one
new, create-only evidence root and publish a fresh frozen manifest there.  It
must replace every `null` in the template with reviewed, immutable values and
use the frozen-candidate schema version
`riley.rc3-qualification-candidate.v1` with `status: "frozen"`.

The frozen manifest has closed top-level `source`, `release`, `images`,
`toolchain`, `models`, `arms`, `rollback`, `outputs`, and `required_gates`
objects.  Together these bind the candidate source/archive/binary/bundle/Cargo
lock/CUDA C ABI, extension registry, canonical correctness golden, release/build images, Rust/CUDA
toolchain, SmolLM2 and Qwen identities, both configuration arms, and the
rollback artifact.  The operating protocol—not a repository-local template—
requires publication to an approved external create-only root.

Create the external immutable freeze from a reviewed filled object, then use
its emitted SHA-256 as a trusted checker input:

```sh
python3 ci/release/write_rc3_candidate_freeze.py \
  --input /decision-input/reviewed-rc3.json \
  --output /decision/riley-0.1.0-rc3.freeze.json \
  --repository-root /clean/rustinfer

bash ci/release/run_rc3_qualification.sh \
  --freeze /decision/riley-0.1.0-rc3.freeze.json \
  --expected-candidate-sha256 <writer-output-sha256> \
  --evidence-root /evidence/riley-0.1.0-rc3 \
  --decision-dir /decision/riley-0.1.0-rc3-result
```

The finalizer only verifies completed evidence; it never hides SSH, CUDA, or
partial reruns. Run GPU producers explicitly on the approved remote host, then
invoke the finalizer once from the clean, frozen candidate checkout.

`arms.stable_default` and `arms.max_performance_exact` each contain the exact
argument vector, an explicit string environment map, and
`configuration_sha256`: the SHA-256 of canonical JSON
`{"argv": ..., "environment": ...}`.  Receipt bindings use the profile name
`stable-default` for the `stable_default` arm; only that arm can qualify a
stable release.

The final release-candidate manifest, the replayed Gate E report, and six C02
gate receipts are **future outputs** at freeze time.
`outputs.final_release_candidate_manifest`, `outputs.final_release_candidate`,
and `outputs.receipts` therefore declare only immutable relative paths; they
cannot honestly contain hashes that do not yet exist.  The checker replays
Gate E from the declared manifest and requires the declared report to be
identical to that replay before it resolves receipt descriptors into its
create-only qualification report.  A passed qualification report also carries
the closed Gate E evidence-hash map (canonical correctness, Python-free E2E,
CUDA fault, performance, soak, reproducible-build artifacts, and dependency
manifest) so an RC3 decision does not hide those prerequisite receipts behind
one opaque report hash.

Each future gate must use its own closed semantic receipt schema with one of
these gates:

- `startup_configuration`
- `qwen_multistep`
- `routing`
- `fault_extension`
- `soak_v2`
- `rollback`

Every semantic receipt must bind the frozen candidate ID, frozen manifest
SHA-256, replayed Gate E report SHA-256, `stable-default` configuration
profile, and that profile's configuration SHA-256 where applicable.  A raw
file hash or a generic `status: "passed"` envelope is not a gate: the RC3
finalizer rejects it until a gate-specific replay/validator exists.  A receipt
that self-binds a different candidate or configuration is incomparable, not
partial evidence.

All six listed gates are registered in `check_rc3_qualification.py`.  For a
freeze-declared semantic-report path, the finalizer parses the report, checks
its direct bindings, replays the distinct raw receipt it names, and requires
the complete replayed report to be exactly equal to the submitted one.

`startup_configuration` is the first registered semantic receipt.  Its file
is one create-only `riley.effective-runtime-config-check.v1` report defined by
`effective-runtime-config-receipt-v1.schema.json`.  A passed report has a
closed `arms` inventory with exactly `stable-default` and
`max-performance-exact`.  Each arm records its own frozen configuration hash,
canonical `/v1/config` endpoint payload descriptor, create-only startup
artifact descriptor, and resolved effective-config hash.  The four raw paths
and their bytes must be distinct from one another and cannot reuse Gate E or
any frozen receipt output.  The resolved effective configuration hashes must
also differ, so a different command-line arm cannot be represented as the
same runtime configuration under two names.

Only `stable-default` is the promotion profile, but a stable promotion is not
valid without the independently captured and replayed
`max-performance-exact` evidence.  The outer finalizer does not trust the
report's `passed` field: it validates the closed descriptors, reruns the
config checker from the same external evidence root for all four raw inputs,
and requires the submitted report to match the replay exactly.  The config
checker in turn replays Gate E and binds each arm to its matching frozen arm
and exact canonical endpoint/startup-artifact bytes.  That **semantic check
report** is where the freeze SHA-256 and replayed Gate E report SHA-256 live;
they are not raw startup facts.

### Runtime configuration implementation prerequisite

The checker is intentionally not a substitute for a server implementation.
At this checkout, Riley has no `GET /v1/config` route, so no actual C02
candidate can qualify yet.  Before a frozen candidate is created, a separate
corrective production change must expose one immutable canonical body *after*
cold prepare, and return `503` rather than inventing a default when that body
is unavailable.  The body must contain only the reviewed C02 inventory:
`schema_version`, `candidate_id`, `runtime_identity`, `effective_config`, and
`effective_config_sha256`; both nested and top-level JSON must use the exact
canonical byte encoding checked here.  Its ten effective-config dimensions
must come from prepared runtime facts, including actual attention/fallback
selection, prepared shape buckets, KV geometry, and a defined aggregate GEMM
policy—not merely echoed CLI arguments.  `runtime_identity` contains exactly
the launch-time `configuration_profile` and `configuration_sha256`, while the
top-level candidate ID is also known at candidate-freeze time.  It must not
contain the frozen-manifest SHA-256 or Gate E report SHA-256: Gate E is a
future output when the frozen candidate is launched.  The operational order is
`freeze → launch/cold prepare → canonical endpoint + create-only startup
artifact → Gate E replay → C02 semantic report`; the final semantic report
post-capture binds the raw descriptors to the freeze and Gate E hashes.  The
same canonical payload is the input to the create-only startup artifact.

`fault_extension` has a CPU-only semantic checker.  Its input is the closed
`riley.fault-extension-receipt.v2` replay descriptor, not a generic `passed`
envelope.  It binds the frozen candidate, freeze SHA, exact replayed Gate E
report, `stable-default` configuration SHA, source archive, release binary,
release bundle, CUDA report/raw archive, and an additional canonical raw trace.
Every path is snapshotted no-follow and re-hashed before replay.  The earlier
v1 descriptor is deliberately not accepted because it did not bind the C02 §8
engine-fault trace.

The checker first replays the known Gate E CUDA archive against those exact
candidate artifacts and requires its actual `compute-sanitizer` logs; this
is reported only as `real-gpu-sanitizer`.  It then parses the separate
`injectable-synthetic` raw trace directly, rather than trusting a case list
or a `passed` flag.  That trace has to bind the same freeze/report/artifacts
and prove one isolated subprocess plus the exact ordered transitions and
safe terminal state for each of these C02 §8 cases:

- `post-kv-write-runtime-error`
- `output-status-corruption-test-double`
- `scheduler-commit-failure`
- `worker-channel-close-race`

The output keeps those execution classes distinct: Gate E's four CUDA memory
fault cases are real-GPU/sanitizer evidence, while the four engine-state cases
remain explicitly labelled injectable synthetic evidence.  A missing sanitizer,
hand-authored CUDA attestation, self-authored `passed` raw trace, incomplete
case inventory, different artifact hash, non-isolated child, reordered event,
or unsafe terminal state cannot qualify the gate.

```sh
python3 ci/release/check_fault_extension_receipt.py \
  --freeze /decision/riley-0.1.0-rc3.freeze.json \
  --expected-freeze-sha256 <freeze-sha256> \
  --evidence-root /evidence/riley-0.1.0-rc3 \
  --receipt receipts/fault-extension-input.json \
  --report /decision/riley-0.1.0-rc3-fault-extension-check.json
```

The freeze-declared `fault_extension` path is the create-only
`riley.fault-extension-check.v2` semantic report, never this raw receipt.
`--receipt` must point at a distinct path outside every freeze-declared output;
the checker rejects receipt/output aliases before parsing evidence.  The
outer qualification finalizer reads the frozen semantic report, obtains its raw
descriptor from the report's `receipt` field, reruns this checker with that
path, and requires exact equality rather than accepting either file's
`passed` field.

`qwen_multistep` is the matching closed generation/streaming gate.  Its
source-controlled `qwen-multistep-golden-v2.json` fixes the supported Qwen2.5
model identity plus the PR12 code, English, and Korean prompt-token and
eight-step greedy-token cases.  Crucially, every expected committed output
piece binds both its token ID and its exact text, including the Korean final
empty-text committed token; this detects bad detokenization rather than merely
matching a final concatenated string.  The companion source-controlled
`qwen-multistep-wire-v2.json` maps those three cases to the reviewed semantic
messages and exact UTF-8 rendered Qwen prompts from the compatibility
reference.

For each case and delivery mode, the v2 raw evidence contains separately
SHA-bound request-body, response-header, response-body, and generation-audit
record descriptors.  The checker parses the native public bytes, verifies the
literal request/prompt and HTTP/SSE shape, and binds the public response ID to
the matching audit record.  Non-stream output must equal the concatenation of
all committed audit texts.  SSE emits exactly one public delta for each
nonempty committed text, never exposes a token ID, then emits one empty-text
terminal frame with the finish reason followed by `[DONE]`.  The stream and
non-stream server request IDs must be distinct, while each is required to bind
its own public and audit artifacts.  A manifest that merely supplies different
expected tokens, hides a bad public byte sequence behind a descriptor, or
claims a shared request ID cannot qualify.

The freeze-declared `outputs.receipts.qwen_multistep.path` is the create-only
`riley.qwen-multistep-check.v2` semantic check report, not the raw receipt.
That report contains a descriptor for the separate
`riley.qwen-multistep-receipt.v2` raw evidence.  The outer finalizer
validates the report, reruns the semantic checker from that descriptor, and
requires exact equality with the replay.  Raw evidence may not reuse any
freeze-declared final report or semantic-receipt path; descriptor aliases are
rejected.  The v1 source files remain historical evidence only and are never
silently reinterpreted as v2.

`soak_v2` extends—rather than replaces—Gate E's reviewed PR16 soak.  Its
source-controlled `soak-v2-scenarios-v1.json` fixes an ordered, 52,200-second
full-duration inventory: all ten inherited Gate E scenarios followed exactly
once by cancellation rate 0/10/50%, KV utilisation 70% and 90%, a 100%
capacity-boundary/rejection case, an ordered exact-backend fallback case, and
at least eight complete model load/ready/unload cycles.  A raw trace carries
measured request, cancellation, terminal-event, capacity-rejection, KV,
RSS/pinned/VRAM, and backend/lifecycle evidence at every five-minute-or-less
interval for every scenario.  The verifier requires exact terminal accounting
and the exact cancellation count/rate—not an aggregate claim or a generic
`passed` flag.

The freeze-declared `outputs.receipts.soak_v2.path` is the create-only
`riley.soak-v2-check.v1` semantic check report.  Its `receipt` descriptor
names a distinct `riley.soak-v2-receipt.v1` raw input below the external
evidence root.  The semantic verifier snapshots every descriptor without
following links, rejects textual/hard-link/reserved-output aliases, validates
the current source-contract bytes/hash, replays the Gate E raw soak archive
against its correctness/native inputs, and requires the replayed Gate E report
to equal the captured report before it accepts the C02 trace.  The outer
finalizer parses the semantic report, reruns this verifier from its
raw-receipt descriptor, and requires exact equality with the entire result; it must never
accept the raw receipt or an opaque hash as the `soak_v2` gate.

`routing` is a separate C02 fixed release-binary gate, not C03 fuzzing.  Its
source-controlled `rc3-routing-corpus-v1.json` fixes five traces: C=1, a dense
permuted C=5 mixed prefill/decode iteration, C=8 precommit cancellation,
malformed-plan rejection before dispatch, and post-execute commit-failure
containment.  Together they cover C1/C5/C8, dense slot permutation, mixed
work kinds, cancellation, KV ownership, and shutdown quiescence.  The corpus
bytes are SHA-pinned in `check_rc3_routing_receipt.py`; a receipt cannot
provide its own workload, expected token routing, or generic
`passed`/hash-only substitute.

The raw `riley.rc3-routing-receipt.v1` binds the frozen candidate and freeze
SHA, replayed Gate E report, `stable-default` arm, frozen SmolLM2 model,
source revision/archive, release binary/bundle/image, CUDA test image, and an
ELF descriptor whose digest is the frozen release binary digest.  Its trace
manifest and every raw trace repeat those bindings.  The CPU-only verifier
follows no symlinks, rejects duplicate/aliased/reserved descriptors, validates
the closed slot → request → downloaded-token → publication/terminal chain, and
requires each trace body to equal the reviewed fixed corpus case.

```sh
python3 ci/release/check_rc3_routing_receipt.py \
  --freeze /decision/riley-0.1.0-rc3.freeze.json \
  --expected-freeze-sha256 <freeze-sha256> \
  --evidence-root /evidence/riley-0.1.0-rc3 \
  --receipt routing/raw-receipt.json \
  --report /decision/riley-0.1.0-rc3-routing-check.json
```

The freeze-declared `outputs.receipts.routing.path` remains the create-only
`riley.rc3-routing-check.v1` report, never the raw receipt.  The outer
qualification finalizer calls `validate_check_report`, reruns `evaluate` using the
returned raw receipt descriptor, and requires exact equality of the two reports.  The
verifier itself performs no SSH/GPU work; an approved remote producer creates
the fixed release-binary captures before this local replay step.

`rollback` is the final C02 release-artifact drill.  Its frozen semantic
`riley.rc3-rollback-check.v1` report names a distinct raw
`riley.rc3-rollback-receipt.v1`, which in turn binds the full candidate and
prior-artifact identities plus a raw drill trace.  The trace must drain the
candidate worker to zero resources, atomically switch to the frozen prior
binary/bundle/image, produce the exact expected token result under a distinct
worker/model identity, and leave no candidate allocation or process behind.
The outer finalizer independently validates both artifact sets, rejects any
raw-path collision with a freeze output, reruns the checker, and requires
exact equality with the returned semantic report.

## Files and retention

| File | Role |
| --- | --- |
| `rc3-candidate.json` | Safe, non-runnable/unfrozen template only |
| `rc3-candidate-template.schema.json` | Strict schema for that template |
| `c02-p0-runtime-config-receipt.md` | Required corrective production prerequisite before any C02 freeze |
| `rc3-qualification.schema.json` | Strict Draft 2020-12 schemas for frozen candidate input and qualification report output; each C02 receipt has a separate semantic schema |
| `effective-runtime-config-receipt-v1.schema.json` | Closed `startup_configuration` endpoint, startup-artifact, and semantic check-report schema |
| `fault-extension-receipt-v2.schema.json` | Closed Gate E + real-GPU/sanitizer + injectable C02 §8 raw-trace descriptor and semantic check-report schema |
| `qwen-multistep-receipt-v2.schema.json` | Closed v2 `qwen_multistep` raw-public/audit evidence, case manifest, and semantic check-report schema; v1 is retained only as historical source material |
| `qwen-multistep-golden-v2.json` | Reviewed Qwen2.5 source contract for exact committed token IDs and text pieces in the three C02 multi-step cases |
| `qwen-multistep-wire-v2.json` | Canonical reviewed mapping from each C02 semantic message list to its exact rendered Qwen prompt and prompt SHA-256 |
| `soak-v2-receipt-v1.schema.json` | Closed C02 `soak_v2` raw receipt, full-duration trace, Gate E archive descriptors, and semantic check-report schema |
| `soak-v2-scenarios-v1.json` | Reviewed ordered 52,200-second Gate E v1 + C02 soak scenario contract |
| `rc3-routing-receipt-v1.schema.json` | Closed `routing` raw receipt, fixed-corpus trace manifest/trace, and semantic check-report schema |
| `rc3-routing-corpus-v1.json` | Reviewed finite C02 release-binary routing cases; distinct from C03 generative fuzz input |

Keep raw traces, model weights, credentials, and large evidence archives out
of this repository.  Store them beneath the approved external create-only
evidence root.  Paths recorded in the frozen manifest and report are relative
to that root, must not contain `..`, and must not be reused for a replacement
candidate.  A failed candidate remains immutable evidence; create a new
candidate ID and a new root after a corrective change.
