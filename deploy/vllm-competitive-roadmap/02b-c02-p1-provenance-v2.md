# C02-P1 — Provenance v2와 Reconstructed Rollback Baseline

**상태:** In progress — v1 provenance 결함을 확인했고, candidate freeze 전에 v2 contract와 source-owned raw producer를 구현한다. initial C02 lifecycle supervisor/raw receipt, native sampling fallback source leaf/marker, source pair raw capture v2, fresh lifecycle-evidence preparer v5, fixed lifecycle bind-request writer v5, private held-lock terminal binder core와 fixed-name raw compositor v5, authenticated native-fallback raw-v5 runner, reconstructed RC2 전용 rollback raw-provenance v3 verifier/schema와 path-only binder, RC2-compatible raw phase collector, immutable artifact snapshot→separate runtime-copy preparation, isolated atomic-switch producer, held-FD artifact-exchange transaction, fixed candidate/source rollback bind-request writer와 same-stack v3/v4 finalizer 및 same-stack finalizer normal-return receipt v1, same-invocation rollback terminal-provenance v4, reviewed RC2 source/OCI input closures, cross-root baseline content bridge v1, A/B reproducibility build-input closure, static reconstructed-runtime assembly recipe contract, arm별 raw runtime assembly/capture receipt verifier, source-free arm별 raw runtime-assembly host runner/USTAR composer, original freeze-input closure를 full reconstructed-baseline-v2 replay와 함께 bind하는 create-only frozen-candidate input-identity manifest/FD-safe replayer, fresh opaque fourteen-leaf Gate E input snapshot preparer와 four-gate closed input-inventory replayer, no-action authenticated Gate E supervisor handoff probe, native·optimizer canonical-E0, Python-free, performance, soak semantic component adapter와 aggregate Gate E semantic replayer는 구현됐다. CPU/static hostile-path 검증과 Arm A의 source-free raw Docker capture/structural closure는 완료됐지만, 그것은 GPU/service 실행이나 qualification을 뜻하지 않는다. authenticated remote rollback runner, authenticated actual Gate E producer, actual GPU capture·actual candidate freeze·lifecycle-v5 receipt·full Gate E semantic qualification은 아직 수행하지 않았다.
**Gate E v3 private-core template:** `rc3_gate_e_private_raw_core_v1.py`는 root-installed
external anchor가 나중에 hash-bind할 private child의 audit/source template이다. public
checkout path는 direct invocation을 FD/socket/lock/child action 전에 fail-closed한다.
sealed core/config memfd와 private credential-authenticated `SOCK_SEQPACKET`에서
nonce/config-digest-bound `INIT → READY → RUN_NO_ACTION → COMPLETE`만 CPU-only로 검증하며, lock/GPU/Docker/raw
producer/evidence/semantic replay/receipt/qualification capability는 추가하지 않는다.
**Gate E v3 root-bound no-action bootstrap template:**
`run_remote_rc3_gate_e_session_v3.py`는 이 child를 future installed bundle에서만
parent-side로 넘기기 위한 source/audit template이다. fixed root invocation 외 checkout/CLI
override는 FD/anchor/lock/socket/fork 전에 fail-closed하며, installed copy는 raw empty
environment, unblocked termination signals/default `SIGCHLD`, PID 1 namespace, approved
local filesystem 위 full initial UID/GID map/ACL-free held anchor, compiled core pin 및 pre-existing lock을
독립적으로 검사한다. parent FD 7는 child에 전달되지 않고 sealed FDs 8/9와
credential-authenticated FD 10만 전달된다. CPU fixture는 no-action handoff와 tamper/lock
rejection만 검증하며, dynamic-loader injection/parent-SIGKILL lifetime gap은 future native
secure-exec launcher가 Python load 전 bootstrap leaf를 authenticate하고 guardian/lease contract를
함께 갖추지 않으면 해결되지 않는다. 따라서 이
`COMPLETE`도 producer/receipt/qualification authority가 아니다.
**Gate E guardian/lease v1 model:** `ci/release/RC3_GATE_E_GUARDIAN_LEASE.md`와
`rc3_gate_e_guardian_lease_contract_v1.py`는 위 native/PID1 boundary를 CPU-only로 고정한
`guardian-lease-contract-only`/`not-authoritative`/`not-installed` contract다. 이는 guardian,
warden, PID1 admission controller, durable ledger 또는 future bootstrap을 설치·실행하지 않는다.
future release는 file lock availability가 아니라 exact held non-delegated cgroup의 fresh
`populated=false`와 registered terminal worker pidfd만 release 조건으로 쓰며, restart의 active
durable ledger는 `DRAINING`으로 rehydrate한다. future sealed FD 31/32 successor is not a v3
FD 7/8/9/10 compatibility path; GPU/Docker/raw capture/evidence/receipt/qualification은 계속 없다.
**의미 등급:** `reference` + C02 release-gate corrective prerequisite
**한 가지 목적:** soak와 rollback의 self-authored summary를 실제 process/socket/GPU/HTTP/audit evidence로 교체하고, historical stable artifact가 없는 RC2를 정직하게 reconstructed-tag baseline으로 한정한다.

[이전: C02-P0](02a-effective-runtime-config-receipt.md) | [목차](README.md) | [다음: C02](02-rc3-candidate-qualification.md)

## 1. 왜 C02-P1이 필요한가

현재 `soak-v2-receipt.v1`은 interval 수치와 backend event 문자열이 들어 있는 trace JSON의 hash만 검증한다. `rollback-receipt.v1`도 worker/model label, `strategy: atomic-rename`, zero resource summary를 검증할 뿐 실제 PID/start tick, listener socket, raw HTTP/audit, shutdown completion marker, inode 전환을 재생하지 않는다. 둘 다 정규 candidate qualification 근거가 될 수 없다.

또한 `/v1/config`의 `fallback_policy.runtime_selection`은 cold-prepared policy ID다. 실제 fallback event가 아니다. 지금의 CUDA worker는 attention/GEMM/executor backend를 실행 중에 재시도하지 않으므로, 존재하지 않는 `exact backend fallback`을 evidence로 주장하면 안 된다.

따라서 C02-P1은 아래 네 가지가 clean committed source에서 검증되기 전까지 candidate freeze를 금지한다.

1. v1 soak/rollback receipt를 historical-only로 두고, finalizer가 v1 qualification input을 명시적으로 거절한다.
2. raw leaf를 strict FD/no-follow replay 가능한 v2 descriptor로 결속한다.
3. source-owned C02 audit에 실제 sampling selection을 기록한다.
4. prior stable이라고 부를 수 없는 RC2 artifact를 reconstructed-tag baseline으로 한정한다.

## 2. Trust boundary와 strict I/O

모든 P1 producer와 semantic checker는 source tree 밖의 새 private `0700` evidence root만 사용한다. 해당 root와 parent chain을 held file descriptor로 열고 다음을 필수로 강제한다.

- `O_NOFOLLOW`와 `O_DIRECTORY`가 플랫폼에 없으면 fallback 값 `0`을 쓰지 않고 즉시 fail closed한다.
- input은 regular non-link file, path uniqueness, bounded byte size, FD inode stability, canonical JSON, SHA-256을 함께 확인한다.
- output은 basename-only `O_CREAT|O_EXCL|O_NOFOLLOW`, file/directory `fsync`, completion marker hash binding으로 한 번만 기록한다.
- raw descriptor 하나가 다른 leaf/path에 재사용되거나, symlink/hardlink alias, controlled leaf read 중 관측된 parent/inode swap, incomplete marker, PID start-tick/socket/GPU UUID mismatch가 있으면 semantic replay는 실패한다. held root FD 검증은 cross-file atomic snapshot이 아니라 point-in-time leaf 검증이므로, P1은 exact-0700 trusted-writer boundary를 전제로 한다.

raw producer가 `qualification_status: "not-run"`을 쓰는 것은 mechanism 검증에만 허용된다. actual C02 candidate의 semantic report는 freeze와 Gate E를 별도 input으로 replay해야 하며 raw file의 `passed` 문자열을 신뢰하지 않는다.

### Shutdown raw leaf v2 contract

shutdown source producer는 다음 세 published schema를 함께 사용한다.

- `benchmarks/release/candidates/c02-capture-metrics-v2.schema.json`
  — `riley.c02-capture-metrics.v2`
- `benchmarks/release/candidates/c02-shutdown-quiescence-v2.schema.json`
  — `riley.c02-shutdown-quiescence.v2`
- `benchmarks/release/candidates/c02-shutdown-quiescence-completion-v2.schema.json`
  — `riley.c02-shutdown-quiescence-complete.v2`

shutdown artifact는 정확히 `schema_version`, `capture_status`,
`qualification_status`, `server_pid`, `server_start_ticks`, `worker_ready`,
`final_metrics`의 일곱 field만 가진다. `capture_status: "captured"`,
`qualification_status: "not-run"`, `worker_ready: false`는 고정이며,
PID와 Linux `/proc/self/stat` start tick은 해당 server process에서 source가 직접
capture한다. candidate ID, freeze SHA, Gate E, semantic receipt, `passed`,
rollback verdict 같은 self-authored conclusion은 이 raw leaf에 넣지 않는다.

`final_metrics`는 exact v2 object이다. `request_states`에는 `active`,
`pending_requests`, `completed`, `failed`, `cancelled`,
`capacity_rejections`; `kv_blocks`에는 `free`, `reserved`, `active`;
`allocation`에는 `device_live_count`, `device_live_bytes`,
`pinned_live_count`, `pinned_live_bytes`; `quiescence`에는
`completion_outbox`, `outstanding_iterations`,
`riley_owned_live_allocations`, `worker_accepting`, `scheduler_accepting`만
있다. 이 schema/binder 단계는 raw fact의 exact shape와 type만 확인하며
failure/KV/quiescence threshold나 candidate pass/fail을 해석하지 않는다.

artifact file을 `fsync`한 다음, 같은 held private root FD 아래 nonhidden
`<artifact_filename>.complete` marker를 `O_CREAT|O_EXCL|O_NOFOLLOW`로 한 번만
생성한다. marker는 정확히 `schema_version`, `artifact_filename`,
`artifact_sha256` 세 field이고 raw artifact bytes의 SHA-256을 bind한다. marker와
root directory도 `fsync`한다. v1 `riley.c02-shutdown-quiescence.v1` 또는
hidden `.<basename>.complete` convention은 historical-only이며 P1 v2 input으로
수용하지 않는다.

`artifact_filename`은 정확히 `^[A-Za-z0-9][A-Za-z0-9._-]{0,240}\\.json$`인
ASCII nonhidden leaf여야 한다. 최대 246 byte로 제한해 sibling `.complete`까지
POSIX 255-byte filename bound를 넘지 않게 한다. global evidence root에서
`candidate-phase/shutdown.json`처럼 보일 수 있지만, `direct-child`는 Rust
writer가 startup부터 보유하는 private audit-root FD에 대한 표현이다. writer는
path를 shutdown 시점에 다시 열지 않고 그 held FD로만 artifact와 marker를
create-only publish한다. 따라서 path rename/swap은 새 경로로 redirect하지 않고,
writer는 원래 보유한 audit root 외의 replacement를 절대 다시 열어 동등한
evidence root로 취급하지 않는다.

## 3. 실제 selection trace

C02-P1이 증명할 수 있는 실제 selection 전환은 executor backend fallback이 아니라 다음으로 한정한다.

```text
max-performance-exact arm
  configured sampling backend = gpu-greedy
  non-zero temperature/stop-token/repetition condition makes GPU greedy ineligible
  → CPU normative sampling is selected and committed
```

Rust worker는 scheduler commit 성공 뒤에만 bounded typed event를 기록한다. event는 `iteration_id`, configured/selected backend, typed ineligibility reason, committed flag만 가진다. terminal `length|stop`, completion token count, source-issued server request ID와 C02 runtime-config identity가 같은 create-only `riley.c02-generation-audit.v2` record에 결합된다. 실패, abort, cancel, duplicate/missing/overflow trace는 successful fallback evidence가 될 수 없다.

현재 source는 이 audit/marker가 먼저 durable하게 생성된 뒤에만 별도
`riley.c02-native-fallback-event.v1` leaf와 matching nonhidden completion marker를
create-only로 발행한다. leaf는 audit record filename/SHA, candidate/runtime/PID
start-tick, request ID와 ordered committed `gpu-greedy → cpu-normative` selection
projection을 결합하며, request-induced
`nonzero-temperature|repetition-penalty|finish-token-mask`만 허용한다. 이 leaf는
모든 audited selection이 one-output-slot scheduler plan에서 나온 경우에만
발행된다. multi-output plan의 CPU 결정은 다른 request가 강제했을 수 있어
request-local attribution이 불완전하므로 일반 audit만 남기고 fallback leaf는
fail closed한다. 이는 source-level raw event일 뿐 아직 capture session이나 terminal
manifest가 아니다.

stable-default arm이 CPU sampling이면 이 evidence를 재사용할 수 없다. max-performance-exact endpoint/startup artifact raw bytes가 GPU-greedy configuration임을 먼저 bind해야 한다. attention/GEMM/executor fallback이 필요해지면 별도 runtime feature를 먼저 설계하고 이 P1을 그 증거로 바꾸지 않는다.

## 4. Reconstructed-tag baseline

`riley-0.1.0-rc2`는 owner-approved prerelease이고 prior stable shipped binary/bundle/OCI image가 아니다. 원격 cache와 published RC2 release에도 serving artifact가 없으므로, C02-P1은 그것을 `previous_stable` 또는 `historical_shipped`로 표기하지 않는다.

P1은 pinned annotated tag object (`a3f5203c3a72122e9da818c1e441c2a789f7aa8c`)와 target commit에서 독립 clean A/B network-none build를 수행해 `riley.reconstructed-prior-baseline.v2` manifest를 만든다. v2는 각 arm의 실제 server binary를 별도 raw descriptor로 보존하고, build receipt와 recipe inspect에 같은 descriptor를 exact-bind한 뒤 A/B SHA-256·byte length equality와 distinct-path replay를 요구한다. 이 object pin은 같은 tag name/target으로 annotation만 교체하는 경우를 막는 reviewed value이며, 현재 unsigned tag의 signature validation 주장은 아니다.

그 A/B reconstruction의 앞단계로 `ci/release/prepare_reconstructed_rc2_inputs_v1.py`는 local Git object store에서 위 reviewed annotated tag object/target을 확인하고, direct target의 bounded uncompressed `git archive` tar grammar를 검증한다. caller가 **independently reviewed** source-archive SHA-256을 제공해 생성된 tar와 일치할 때만 새 normalized external mode-0700 root에 `source/git-tag-object.json`, `source/git-tag-target.json`, `source/riley-0.1.0-rc2.tar`, `reconstructed-rc2-source-inputs.json`을 create-only로 발행한다. receipt/schema (`riley.reconstructed-rc2-source-inputs.v1`, `benchmarks/release/candidates/reconstructed-rc2-source-inputs-v1.schema.json`)는 `prepared/not-run` source-input closure일 뿐 baseline manifest, build, OCI image, service/GPU observation, rollback 또는 qualification을 만들거나 주장하지 않는다. observed archive digest를 source default/pin으로 넣지 않으며, producer와 모든 후속 replay caller는 같은 reviewer-provided SHA-256을 다시 제공해야 한다. source-date epoch은 self-authored receipt field로 보존하지 않고 후속 A/B builder가 held archive bytes에서 직접 derive/replay한다.

Runtime image content는 별도 per-arm closure로 준비한다. `ci/release/prepare_reconstructed_runtime_oci_inputs_v1.py`는 이미 캡처된 raw runtime image inspect와 **uncompressed OCI image-layout tar**만 받고 `--reconstruction-id {a,b}`별 새 external private root에 raw inspect/tar 및 tar 내부의 정확한 `oci-layout`, `index.json`, selected manifest/config bytes를 create-only로 저장한다. 같은 held archive FD에서 `inspect[0].Id == sha256(config raw bytes)`, 단일 index manifest, raw inspect/config의 필수 `linux/amd64` platform, 선언된 경우 exact `linux/amd64`여야 하는 index platform, manifest/config/layer descriptor의 hash·size, 모든 referenced layer blob, 그리고 정확한 blob closure를 replay한다. tarfile parser보다 앞선 raw header preflight는 bounded regular file/zero-payload directory만 허용해 PAX/GNU longname/sparse/link/special payload를 메모리화하기 전에 거절한다. OCI spec에서 optional인 index/manifest top-level media type은 존재할 경우만 exact OCI value로 검증한다. Docker-save는 OCI layout이 아니므로 이 contract에서 명시적으로 거절한다. receipt/schema (`riley.reconstructed-runtime-oci-inputs.v1`, `benchmarks/release/candidates/reconstructed-runtime-oci-inputs-v1.schema.json`)는 `prepared/not-run` content-binding만 뜻하며 source/bundle/build invocation/A-B independence/rollback/qualification은 모두 `not-established`로 유지한다. 이 preparer는 Docker·build·GPU·service를 실행하지 않고, existing v2 baseline의 `oci_archive_content_binding: not-validated`도 변경하지 않는다. 후속 A/B materializer만 source inputs, independently built artifacts, 그리고 두 OCI closures를 명시적으로 함께 소비해 그 경계를 승격할 수 있다.

원격 Docker는 `docker image save`에 clean OCI output만을 보장하지 않으므로, raw runtime image export를 OCI라고 라벨링하거나 OCI v1 parser를 넓히지 않는다. `ci/release/prepare_reconstructed_runtime_image_export_oci_normalization_v1.py`는 이미 존재하는 raw inspect와 runtime image export tar를 arm별 새 private root의 `image-export/`에 보존하고, legacy one-image Docker-save, clean OCI layout(선택적 root `manifest.json`/`repositories` opaque sidecar 포함), 또는 별도 이름의 Docker-28/Moby hybrid profile을 엄격히 판별한다. 이 hybrid는 임의 unreferenced blob 허용이 아니다. root `manifest.json` 한 행이 selected config·ordered layer·`LayerSources`를 exact-bind하고 `repositories`가 없으며, selected layer마다 정확히 하나의 작은 SHA-addressed strict-JSON legacy config record가 단일 acyclic/non-branching parent chain과 selected linux/amd64 config head를 만들어야 한다. 그 조건의 raw compatibility bytes는 raw archive descriptor에만 보존하고 canonical output에는 selected config/layer bytes만 쓴다. 일반 opaque root sidecar는 extra blob이 없을 때만 bounded bytes로 보존·hash하며 의미를 해석하지 않는다. inspect Id와 selected raw config SHA-256, `linux/amd64`, selected config/layer descriptor·size·digest를 held FD에서 replay한 뒤, 선택 config raw bytes와 layer raw bytes만 사용한 canonical uncompressed OCI USTAR 및 derived layout/index/manifest/config snapshots를 create-only로 발행한다. replay는 OCI JSON/content closure뿐 아니라 USTAR member header·순서·metadata·zero padding·record trailer의 exact canonical form까지 byte-for-byte 재구성해 확인한다. legacy uncompressed layer는 config `rootfs.diff_ids`와 exact order/hash가 일치해야 하며, PAX/GNU/sparse/link/special/traversal/duplicate/unknown closure와 oversized zero trailer는 tarfile 전 차단한다. output은 existing OCI-inputs v1 consumer로 다시 parse 가능한 clean OCI layout이지만, raw Docker export/build 실행, same-invocation assembly capture, source/bundle→image, A/B independence, runtime/GPU/service, rollback/freeze/qualification은 여전히 `not-established`/`not-run`이다. 이 `riley.reconstructed-runtime-image-export-oci-normalization.v1` receipt/schema (`benchmarks/release/candidates/reconstructed-runtime-image-export-oci-normalization-v1.schema.json`)의 authority는 `runtime-image-export-to-canonical-oci-content-normalization-only`이며, 후속 cross-root bridge가 raw export와 assembly capture를 실제로 함께 bind하기 전에는 둘의 인과를 주장하지 않는다.

그 후속 per-arm bridge인 `ci/release/prepare_reconstructed_runtime_image_export_assembly_content_bridge_v1.py`는 source-input/repro-build/image-export-normalization/runtime-OCI-input/assembly-capture의 다섯 private root와 reviewer source SHA-256·builder image ID를 모두 held FD로 재검증한다. 결과 `riley.reconstructed-runtime-image-export-assembly-content-bridge.v1` receipt/schema (`benchmarks/release/candidates/reconstructed-runtime-image-export-assembly-content-bridge-v1.schema.json`)는 normalization의 raw inspect·canonical OCI·derived JSON이 OCI-input closure와 같은 `(sha256, byte_length)`인지, 그리고 capture 내부 `image-inspect.json`/`oci-image-layout.tar`가 그 OCI closure와 같은지만 bind한다. raw image-export tar descriptor는 보존하지만 OCI archive와 같은 bytes라고 요구하지 않는다. 이 bridge는 Docker export/build 실행, export와 capture의 same invocation, capture provenance/independence, source/bundle→image, A/B image equality, rollback/freeze/qualification/service/GPU를 모두 `not-established`/`not-run`으로 유지하며 receipt 하나만 create-only로 쓴다.

그 materializer보다 앞선 좁은 검증 경계로 `ci/release/prepare_reconstructed_prior_baseline_content_bridge_v1.py`가 있다. 이는 새 빈 mode-0700 bridge root, v2 baseline root와 canonical relative manifest, RC2 source-input root와 **매 replay마다 다시 제공하는** independently reviewed source SHA-256, arm A/B OCI-input roots를 받는다. 다섯 root를 no-follow로 열고 bridge 전체 동안 held FD만 사용한다. source v1, v2 baseline, OCI v1 A/B verifier를 모두 다시 실행한 뒤, baseline source의 tag object/tag target/archive와 source-input closure의 세 leaf를 `(sha256, byte_length)`로 bind하고, 각 arm의 raw Docker image inspect/OCI archive/layout/manifest/image ID를 matching OCI closure에 bind한다. OCI `index.json`/`config.json`은 v2 peer가 없는 OCI-internal replay leaf로만 receipt에 보존한다. normalized path가 겹치거나 root inode가 alias인 role, arm swap, mutable/extra bridge leaf를 거절하며 입력 root의 절대 경로는 receipt에 직렬화하지 않는다.

결과 receipt/schema (`riley.reconstructed-prior-baseline-content-bridge.v1`, `benchmarks/release/candidates/reconstructed-prior-baseline-content-bridge-v1.schema.json`)는 `bound/not-run`, `authority: cross-root-content-bridge-only` 한 장뿐이다. 그 안에서만 `oci_archive_content_binding: validated-via-runtime-oci-inputs-v1`를 말할 수 있고, 기존 v2 report의 source/OCI binding은 둘 다 계속 `not-validated`로 고정 기록한다. 이는 source→runtime image, bundle→runtime image, runtime build invocation, runtime-capture independence, rollback, freeze, qualification을 **전혀** 증명하지 않으며 모두 `not-established`/`not-run`으로 보존한다. bridge는 freeze admission·rollback·qualification input이 아니고 Docker·build·GPU·service·network를 실행하지 않는다. 실제 materializer에는 arm별 A/B binary/bundle을 runtime image로 조립·capture한 same-invocation raw receipt가 먼저 필요하며, v2 closed schema를 self-authoring하거나 OCI label만으로 그 인과관계를 승격하면 안 된다.

그 same-invocation receipt의 바로 앞에는 `ci/release/prepare_reconstructed_repro_build_inputs_v1.py`가 만든 A/B 입력 closure가 온다. caller는 RC2 source-input root와 **매 replay마다 다시 제공하는** reviewer SHA-256, reviewer-pinned immutable builder image ID, 이미 캡처된 `repro-build-a.tar`/`repro-build-b.tar`를 제공한다. tool은 외부 tar를 새 private root에 create-only snapshot한 뒤, held descriptor에서 만든 별도 private checker copy에만 기존 PR16 parser를 적용한다. raw A/B tar의 closed inventory, source tar, builder image, pre/post container receipts, offline/network-none command, completion receipt, binary/profile/bundle/native equality와 independent container/workspace를 다시 검증하고 각 arm의 raw tar·`build.json`·`riley`·`riley.tar.gz`만 저장한다. receipt/schema (`riley.reconstructed-repro-build-inputs.v1`, `benchmarks/release/candidates/reconstructed-repro-build-inputs-v1.schema.json`)는 `prepared/not-run` 입력 closure이며 A/B binary/bundle/source-archive equality와 PR16 execution independence까지만 `validated`라고 한다. runtime-image assembly/capture, source/bundle→runtime-image, OCI content, rollback, freeze, qualification은 모두 `not-established`/`not-run`이다. 이 tool도 Docker·build·GPU·service·network를 실행하지 않는다.

이 historical replay에는 active release contract를 재사용하지 않는 닫힌 RC2
manifest profile(verify_reconstructed_rc2_pr16_bundle_v1.py)이 필요하다. 이 profile은 target
6093006ec2b01b784b01ba278296b676f2dfd03a, epoch 1787811743, version 0.1.0,
archive root riley-0.1.0-linux-x86_64-cuda12.8, canonical 10,909-byte release
manifest SHA-256
3da42b3d0bbf1a56ce8768a5cc7bfb175cc969d57c3727ee9b9b0cfd1df6028e, 그리고
crates/riley-server/src/main.rs의 당시 source-contract SHA-256
1f50fec5b886703fe110c9f0c62560a51193baaaf1d498713c9ba8c17f00d9be만 함께
허용한다. 이 profile은 이 A/B closure와 reconstructed runtime-assembly capture의
private selected-bundle replay에만 연결된다. active verifier/CLI의 현재 479902
candidate contract는 그대로이며, generic legacy admission, archive에서 읽어
실행하는 verifier, 또는 historical source code import는 금지된다. tar, checksum,
ELF, native-dependency의 공통 grammar 검증은 shared verifier 내부에서 계속
수행한다.

그 closure를 실제 image build에 넘기기 전에는 `ci/release/ReconstructedRuntimeAssembly.Dockerfile`과 `verify_reconstructed_runtime_assembly_dockerfile.py`가 source-free assembly recipe를 고정한다. recipe의 future canonical context는 exact `Dockerfile`, `input/riley`, `input/riley.tar.gz` 세 leaf뿐이다. 두 stage는 같은 reviewed CUDA runtime index digest와 explicit `linux/amd64` platform을 사용한다. 첫 stage는 caller-supplied archive를 non-root로 unpack하고 A/B reconstruction ID와 closed provenance build arguments의 문법, raw binary/bundle SHA-256, bundle `SHA256SUMS`, no-link/no-special extracted tree, 그리고 raw binary와 bundle의 `bin/riley` byte equality를 확인한다. final stage는 fresh base에서 그 verified `/opt/riley/`만 numeric non-root ownership으로 복사하며 controlled system PATH에서 input ELF를 실행하지 않고 set-ID/hard-link/special file과 bundle-local Python/toolchain executable을 다시 거절한다. static verifier는 normalized instruction SHA-256와 exact stage/inventory를 고정하여 `COPY .`, source checkout, Cargo/Rust/CUDA build stage, package install, `ADD`, build mount/secret/SSH frontend drift를 수용하지 않는다. 이것은 reviewed assembly **tool**의 CPU-only source contract일 뿐 Docker build, bundle→image/source→image binding, OCI content, runtime capture A/B independence, rollback, freeze 또는 qualification을 주장하지 않는다.

그 다음의 source-only post-capture boundary로 `ci/release/prepare_reconstructed_runtime_assembly_capture_v1.py`와 `benchmarks/release/candidates/reconstructed-runtime-assembly-capture-v1.schema.json`가 한 arm의 이미 존재하는 raw USTAR capture를 새 private root에 create-only snapshot한다. 매 replay는 reviewer SHA와 builder image ID를 다시 받아 RC2 source v1 → PR16 A/B reproducibility v1 → matching-arm OCI v1을 held FD로 재실행하고 root inode alias/경로 overlap을 거절한다. capture는 fixed `SHA256SUMS`, canonical three-leaf context, exact source-free `docker build` logical argv와 seven provenance args, Docker raw format의 trailing newline 없는 iidfile, raw image inspect/OCI export binding, `docker create` 직후의 `created`/not-started/no-mount/network-none inspect, 그리고 selected bundle의 strip-root `/opt/riley` file tree를 가진다. runtime tree는 final numeric non-root `65532:65532` ownership도 요구한다. image/container runtime config는 recipe가 소유한 exact user·entrypoint·command·세 `ENV` instruction과, 같은 digest-pinned CUDA runtime base가 상속한 값을 합친 **exact 21-entry final environment map**만 허용한다. 그 canonical map SHA-256은 `b4192ae0e6a063fd9eb049f9204c75928ffaaa6c854ce4a2a2901752afae96ac`이고, 이름 family allowlist나 OCI에서 동적 추출하지 않는다. unknown/missing/duplicate environment name과 상속 값·recipe override의 모든 value drift는 계속 fail-closed이며 working directory, volume, healthcheck, deferred OnBuild surface도 거절한다. container는 network-none/unprivileged뿐 아니라 host/container namespace mode도 거절하고 private/daemon-default namespace만 수용한다. bind/tmpfs/device/capability/security option도 모두 거절한다. USTAR raw preflight는 PAX/GNU/sparse/link/device/FIFO, traversal, duplicate, noncanonical metadata와 nonzero trailer를 tar parser 이전에 차단한다. outer snapshot은 fixed member ceilings, headers, end marker, one 20-block zero trailer만 합산한 약 13.06 GiB max를 사용하여 sparse/zero-tail input이 evidence volume을 소모하지 못하게 한다. POSIX USTAR의 single-member size field 한계 때문에 이 v1 capture는 embedded `oci-image-layout.tar`가 8 GiB−1 이하인 OCI v1 closure만 수용한다; 더 큰 OCI closure는 PAX로 우회하지 않고 future directory-snapshot contract로 분리한다. OCI archive와 raw image inspect는 기존 OCI v1 closure bytes와 exact `(sha256, byte_length)` equality를 요구하고, bundle은 held repro descriptor에서 만든 private checker copy에서만 replay한다. receipt의 `bound/not-run`은 raw record의 구조적 cross-check만 의미한다. raw record는 Docker build/container-copy가 실제로 수행됐다는 독립 증거가 아니므로 runtime build execution, container filesystem capture provenance, source/bundle→runtime image, A/B capture independence, image equality, runtime/service/GPU execution, rollback, freeze, historical distribution, candidate qualification은 모두 계속 `not-established`/`not-run`이다. 이 preparer는 Docker, compiler, GPU, service, network를 호출하지 않는다.

그 두 arm을 함께 소비하는 좁은 후속 경계로 `ci/release/prepare_reconstructed_runtime_a_b_materialization_v1.py`와 `benchmarks/release/candidates/reconstructed-runtime-a-b-materialization-v1.schema.json`가 추가됐다. 이것은 shared source-input/repro-build root와 서로 다른 A/B runtime-OCI/capture root 여섯 개를 한 번씩 held FD로 열고, normalized path overlap 또는 어떤 `(st_dev, st_ino)` alias도 거절한다. 각 arm capture를 다시 replay한 뒤 source/repro anchor가 같음, selected `riley`와 `riley.tar.gz`의 SHA-256·byte length equality, 그리고 이미 bundle과 대조된 `/opt/riley` runtime-tree summary의 SHA-256·entry count·byte length equality만 create-only `riley.reconstructed-runtime-a-b-materialization.v1` `bound/not-run` receipt로 기록한다. recipe가 arm `a`/`b` reconstruction ID를 OCI label에 의도적으로 넣으므로 OCI image ID/config/manifest/archive는 arm별 관측값으로 보존할 뿐 A/B equality를 요구하거나 주장하지 않는다. 이 receipt는 reconstructed baseline manifest, freeze/rollback/qualification input, Docker/image-export 실행, same-invocation, capture provenance/independence, source/bundle→image 인과, runtime/service/GPU 실행을 만들지 않는다. 실제 승격에는 authenticated same-stack producer의 별도 lineage contract가 여전히 필요하다.

이 full replay는 PR16 checker의 `tomllib` 때문에 Python 3.11+가 필요하다. 현재 remote host의 Python 3.10에서는 `unsupported-python-runtime`으로 fail-closed하며, 이번 CPU-only wrapper test는 held-FD/receipt boundary를 검증할 뿐 실제 captured evidence replay를 대체하지 않는다. pinned 3.11+ runtime provisioning은 actual materialization 전의 명시적 host prerequisite다.

그 prerequisite를 installation 없이 점검하는 별도
`ci/release/check_reconstructed_runtime_python_prerequisite_v1.py`는 existing
Linux x86_64 CPython `3.13.15` executable pin을 explicit external `--python`
path에서 no-follow held FD로 hash한 뒤, hash 직후 held-descriptor
`/proc/self/fd/...` path로 clean Python configuration `-I -S -E -B` fixed stdlib probe를
요청한다. probe는 CPython/Linux/x86_64/version과 `tomllib`, `tarfile`, `hashlib`,
`lzma`, `bz2`, `sqlite3` availability를 확인하고 canonical transient `checked/not-run`
stdout만 낸다. held FD는 pathname replacement를 줄일 뿐 hash 뒤 exact executable bytes,
same-inode mutation, stdlib/dynamic-loader를 포함한 full runtime-tree integrity, 또는
same-UID writer exclusion을 증명하지 않는다. Python flags도 sandbox가 아니며, controller
자체가 download/uv/package install, output receipt/evidence, Docker/GPU/materializer를
요청하지 않는다는 사실은 supplied runtime 실행의 network/Docker/GPU 격리를 보장하지
않는다. 따라서 checker를 실행하기 **전** operator가 full runtime tree와 ancestor를
외부에서 신뢰·검증하고 same-UID writer를 배제해야 한다. 이 transient output은 later
materializer와 same-FD handoff, capture/materialization 또는 qualification도 주장하지
않는다. `/tmp` 계열, source checkout, symlink/hardlink, unsafe executable과 128 MiB 초과
executable은 거절하지만, 그 byte-volume 상한은 host resource/hashing-time isolation이 아니다.
detailed operation boundary는
`ci/release/RECONSTRUCTED_RUNTIME_PYTHON_PREREQUISITE.md`에 고정한다.

그 후속 raw producer는 `ci/release/run_remote_reconstructed_runtime_assembly_capture_v1.sh`, filesystem-only `initialize_reconstructed_runtime_assembly_evidence_v1.py`, stdlib-only `compose_reconstructed_runtime_assembly_capture_v1.py`로 구현됐다. runner는 caller 환경을 `env -i`로 제거하고 authenticated no-follow Docker lock의 ready-handshaked process group 아래에서 동작한다. `HOME`은 의도적으로 사용 불가로 두므로 Docker CLI가 설정 경로를 필요로 할 때는 runner가 verified scratch root 아래의 fresh owner-only `docker-config/` child만 만들고 export한다. 이 child는 caller input이나 evidence가 아니며 같은 cleanup에서 제거된다. initializer는 no-follow ancestor FD로 fresh external `0700` root와 fixed `raw` child를 create-only로 만들고 source checkout의 lexical·mount alias 아래 출력을 거절한다. runner는 reviewed digest-pinned CUDA base가 daemon에 **이미 존재**하는지 pull 없는 inspect로 먼저 확인한다. 따라서 `--network none`은 Docker build-step network setting일 뿐 daemon/control-plane의 host egress 격리를 주장하지 않는다. Docker 명령 표면은 exact seven-argument `docker build` (`--network none`, `--pull=false`, `--no-cache`, no tag) → Docker raw-format(후행 newline 없음) iidfile의 held-FD exact parser → bounded raw `docker image inspect`/`docker image save` → existing normalizer의 canonical OCI → `docker create --network none --restart no` → pre-copy inspect/`docker cp`/post-copy inspect byte-equality 순서로 닫혀 있다. build log, JSON, raw image export, cp tar와 create ID stream에는 replayer와 같은 hard byte bound가 있으며, output cap/producer 오류는 success가 아니라 retained incomplete evidence다. container는 절대 start/run/exec하지 않으며 GPU, device, mount, privileged, secret, SSH, host namespace option을 받지 않는다. Docker cp tar는 extract하지 않고 canonical USTAR runtime tree로 재작성하고, outer capture도 fixed 11-member USTAR/completion/SHA256SUMS로 생성한다. remote Python 3.10에서는 PR16 bundle replay가 Python 3.11+를 요구하므로 runner가 assembly-capture/OCI/bridge full verifier를 호출하지 않는다. 이 raw producer는 same-invocation receipt, Docker execution attestation, source/bundle-to-image, capture provenance/independence, A/B equality, runtime/service/GPU execution, rollback, freeze, 또는 qualification claim을 만들지 않으며, current evidence는 CPU/static hostile-path 검증만 완료했다.

```text
baseline_kind = reconstructed-tag-baseline
provenance_class = reconstructed-from-source
historical_distribution = not-attested
was_previously_shipped = false
historical_stable_artifact_status = unavailable
```

manifest는 tag object/target, source archive, exact build recipe and image inspect, A/B server-binary/profile/bundle equality, runtime OCI inspect/archive, final artifact descriptors를 create-only로 bind한다. legacy `riley.reconstructed-prior-baseline.v1`은 binary equality가 없는 historical checker input으로만 남고 RC3 freeze admission, static rollback evidence preparation, raw rollback v3/v4 chain에는 수용하지 않는다. 공개 RC2 release API raw response는 mutable information이므로 보존할 수 있지만 trusted expected digest는 reviewer-provided input으로 따로 비교한다.

이 baseline으로 가능한 판정은 `reconstructed_operational_rollback`뿐이다. `historical_stable_binary_rollback`은 `not-established`로 유지한다. 실제 historical stable bundle이 나중에 제공되면 새 immutable manifest/version과 별도 gate로 추가한다.

## 5. v3 raw config bridge, v4 serial closure와 semantic replay

### Soak v2

`run_remote_c02_soak_v2.sh` initial lifecycle supervisor는 GPU UUID/used-memory
preflight, authenticated no-follow host lock, frozen arm의 `env -i`, 새 private
evidence root를 강제하도록 구현됐다. outer caller가 만든 sentinel이나 inherited
lock FD를 신뢰하지 않고, clean Python supervisor가 canonical host lock을
`O_NOFOLLOW`로 열어 inode/mode/owner와 nonblocking flock을 확인한 뒤 Bash child를
인증한다. runner는 새 root와 source-audit child를 create-only로 만들고, host
binary와 model tree의 safe ownership/permission/hash를 launch 전과 process exit 후에
다시 확인한다.

이 initial runner는 host binary만 실행하며 frozen canonical contract의 정확히 한
serial non-stream scenario, 그 completion 직후 한 번의 C02 metrics observation,
하나의 v4 raw manifest만 허용한다. config bridge, scenario producer, immediate
observer, SIGTERM graceful shutdown, shutdown-v2 artifact/marker replay까지의 raw
순서를 한 held host lock 아래에서 연결한다. 이는 CPU/static hostile-path로만
검증된 구현이다. 아직 원격 GPU capture, candidate freeze, Gate E semantic replay, semantic
qualification을 실행하거나 주장하지 않는다.

그 lifecycle runner에 앞서 landed한
`c02-raw-soak-runner-contract-v1.schema.json`,
`c02-raw-scenario-capture-v1.schema.json`과
`capture_c02_raw_soak_scenarios_v1.py`는 이미 실행 중인 **단일** host process에서
serial non-stream completion의 exact request/response bytes와 source-written
generation-audit-v2 record/marker를 보존한다. 이는 runner나 GPU operation이
아니며 `qualification_status: "not-run"`만 낸다. source audit record가
`runtime_event_log` 원본이고, wrapper는 fallback event나 sampling summary를
합성하지 않는다. producer는 completion 전후와 audit marker 확인 뒤의 raw
PID/start-tick, loopback listener inode, `/proc/net/tcp`, PID FD-socket snapshot도
보존한다. GPU query는 여기서 하지 않으며 held lock 아래 existing C02 observer가
GPU tuple을 보존한다.

v1 serial contract는 streaming, concurrency/cancel/disconnect,
restart/rollback/multi-PID, `exact-backend-fallback`을 계속 fail closed한다.
source에는 이제 generation-audit record와 별개인 native fallback-event leaf/marker가
있지만, v1 capture와 v4 binder는 이를 아직 소비하지 않는다. audit record를 복사하거나
config 문자열로 대체해서 fallback을 주장해서는 안 된다. `c02-raw-soak-runner-contract-v2`,
`c02-raw-scenario-capture-v2`, `c02-generation-audit-index-v2`와 같은 self-contained
producer의 explicit v2 branch는 `max-performance-exact`의 단일 non-stream
`exact-backend-fallback` 요청만 받으며, Rust `f32` decoder에서도 0으로 반올림되지 않는
exact `temperature: 1`의 `nonzero-temperature` 이유만 수용한다. response ID에서 파생한 같은 held
`source-audit` child의 audit JSON/marker와 fallback JSON/marker 네 leaf를 모두
replay하고, event가 exact audit filename/SHA·candidate/runtime/process/request identity와
ordered selection을 재현할 때만 v2 raw session/index descriptor를 쓴다. 이 capture는
wrapper fallback을 합성하지 않고 `qualification_status: "not-run"`만 남긴다.

capture v2도 endpoint가 실제 GPU-greedy arm이었다는 사실이나 terminal fallback을
증명하지 않는다. `prepare_c02_lifecycle_evidence_v5.py`는 fallback-v2의 one
`exact-backend-fallback` contract와 exact `max_tokens: 1`, `temperature: 1`,
`top_p: 1`, `stream: false`만 검증한 뒤 외부 create-only 0700 root, fixed
`source-audit` child, frozen contract copy를 만든다. GPU/service/terminal marker나
lifecycle receipt를 만들지 않으며 v1 preparer를 fallback path로 넓히지도 않는다.
`write_c02_lifecycle_bind_request_v5.py`는 external canonical bridge
stdout와 held-FD bridge/capture-v2/effective `gpu-greedy` endpoint/fallback observation을
먼저 다시 join하여 고정 `fallback-capture/session.json` 및
`fallback-observation/session.json`만 담은 canonical v5 path-only request를 create-only로
쓴다. 이 helper는 binder, terminal marker, receipt, lifecycle success를 만들지 않는다.
별도 `bind_raw_c02_soak_v5.py`는 이제 canonical path-only request에서
capture-v2의 단일 scenario와 두 source pair를 replay하고, held-FD `/v1/config` endpoint의
validated `effective_config.sampling_backend == "gpu-greedy"`, config bridge, C02
observation PID/start-tick/listener/GPU tuple까지 같을 때만 v5 raw terminal manifest를
발행한다. v5의 source audit/fallback marker는 ordinary one-link evidence이고 terminal
manifest의 `.intent`/`.complete`만 paired hardlink이다. public v5 binder는 자체 output
lock을 잡지만 private held-lock core는 outer root FD를 열거나 lock/unlock하지 않는다.
`compose_c02_lifecycle_v5_raw.py`는 public CLI/path reopen/callback 없이, caller-held
root FD와 EX lock에서만 fixed `c02-lifecycle-v5-bind-request.json` → fixed
`c02-lifecycle-v5-raw-manifest.json` normal-return chain을 만든다. request 전에 root
path를 no-follow ancestor policy로 다시 열어 held FD의 dev/inode와 비교하고, request와
terminal pair의 모든 fixed name을 reserve한다. post-link `fsync` ambiguity는 raw report나
후속 receipt를 만들지 않으며, visible pair의 structural replay로 재개할 수 없다.
`bind_raw_c02_soak_v5.py`와 `compose_c02_lifecycle_v5_raw.py` 자체는 service/GPU/SSH/Docker를
조작하지 않는 raw binding helper다. 별도 `run_remote_c02_soak_v5.sh`만 outer authenticated
host-binary raw producer로서 canonical GPU lock, clean `env -i`, 새 v5 private root와 frozen
one-scenario fallback-v2 contract를 사용한다. runner-owned `--sampling-backend gpu-greedy`를
강제하고 args-file의 두 `--sampling-backend` 표현을 거부한 뒤 config bridge → source-pair
capture → 즉시 observation → source shutdown check → private finalizer의 same-invocation
normal-return chain을 연결한다. finalizer는 fresh private root FD와 nonblocking EX lock을
잡은 상태에서만 compositor를 호출하며 public resume/retry, lifecycle-v5 receipt, candidate
freeze, semantic/Gate E qualification을 만들지 않는다. 결과는 최대 하나의 v5 raw manifest와
`qualification_status: "not-run"`뿐이고, 현 검증은 CPU/static hostile-path 범위라 실제 GPU
capture는 아직 실행하지 않았다. v1/v4/lifecycle receipt는 계속 fallback을 거부한다. initial lifecycle runner는 config bridge → scenario producer → C02 observer
→ source shutdown marker → same-process v4/receipt finalizer 순서를 하나의 held host
lock 아래에서 연결한다. 첫 version은 **contract 1 scenario / observation 1회 / v4
manifest 1개**만 허용한다. multi-scenario capture 뒤의 관측은 어느 completion 직후인지
증명하지 못하므로 aggregate/interleaved soak은 timing을 별도 증명하는 후속 v5/semantic
contract로 미룬다.

현재 v3 bind-request/manifest에는 scenario producer의 `session.json` field가 없어
opaque ledger/index leaf만으로 이 producer를 terminal bind하면 안 된다. 따라서
v3를 넓히지 않고 serial non-stream subset 전용 **v4**를 추가한다.
v3 binder와 verifier는 v1 serial contract, request ledger, generation-audit index의
versioned JSON을 명시적으로 거부해, rename/copy나 retained
`capture-incomplete.json`이 있어도 opaque v3 leaf로 producer closure를 우회하지
못하게 한다. 기존 generic historical runtime-event leaf의 grammar는 바꾸지 않는다.
`riley.soak-v2-bind-request.v4`에는 top-level
`scenario_capture_session_path` 하나만 두며, scenario별 caller input은
`scenario_id`와 C02 `observation_session_path`뿐이다. scenario contract,
ledger, runtime event, generation-audit index, target tuple은 caller가 다시
선언하지 않고 held private-root FD로 읽은 capture session에서 derive한다.

v4에서 쓰는 모든 `*_session_path`는 evidence root 바로 아래의 정확한
`<capture>/session.json` direct child여야 한다. source audit record/marker는 capture
directory와 다른 하나의 direct-child audit directory에만 있어야 하며, scenario마다
audit parent를 섞을 수 없다. private root의 nonblocking exclusive lock을 잡은 뒤
`NAME.json`, `NAME.json.complete`, `NAME.json.intent`가 모두 없는 경우에만 terminal output을 생성한다.

v4 binder는 session descriptor, parent `capture-incomplete.json` 부재,
contract inventory, request/response ID-to-source-audit marker, 그리고
scenario PID/start-tick/listener proof를 explicit replay한 뒤에만 lifecycle
runner output을 수용한다. 그 tuple은 config bridge와 observation session의
PID/start-tick/listener/GPU tuple에도 일치해야 한다. 새 terminal manifest와 marker는
각각 `riley.soak-v2-raw-provenance.v4` 및
`riley.soak-v2-raw-provenance-complete.v4`이고, marker는 정확히
`schema_version`, `artifact_filename`, `artifact_sha256`만 가진다. v4는
`exact-backend-fallback`을 계속 fail closed한다; source-owned native leaf가
추가됐더라도 그것을 replay하는 capture/binder의 별도 version bump 뒤에만 다시 다룬다.

이 full config-bridge/serial/observation join은 manifest 생성 **전에** 완료한다.
따라서 정상적인 target/GPU/bridge drift는 create-only nonterminal manifest를 남기지
않는다. v4 completion marker는 먼저 별도 durable nonterminal intent leaf를 만든 뒤
create-only linked final marker로 공개한다. 따라서 final 공개 전 file-sync 실패는 final
marker를 만들지 않는다. final marker가 보인 뒤 parent-directory sync가 실패하면 binder는
`ambiguous-terminal-publication` nonzero로 끝나며 lifecycle success receipt를 만들면 안
된다. raw verifier가 paired intent/final marker를 읽을 수 있어도, 이후 qualification/
finalizer는 이 ambiguous 결과 뒤의 visible marker만으로 lifecycle authority를 인정하지
않고 runner supervisor의 성공 receipt를 추가로 요구한다.

`bind_raw_c02_soak_v2.py` 계열과 별도 `bind_raw_c02_soak_v5.py`는 runner를 대체하거나
service/GPU/SSH/container를 조작하지 않는다. v3 schema는 config bridge의
historical closed shape로 남기고,
`benchmarks/release/candidates/soak-v2-bind-request-v4.schema.json`와
`soak-v2-bind-request-v5.schema.json`가 각각 fallback-free v4와 closed native-fallback
v5 path-only shape를 publish한다. 어느 버전도 caller-supplied descriptor/hash를 받지 않는다.

raw soak provenance v4는 `stable-default`와 `max-performance-exact`만 수용하며,
initial serial subset에서는 `exact-backend-fallback`을 수용하지 않는다. 이는 raw
binding 조건일 뿐 semantic selection replay나 candidate qualification을 판정하지
않는다. rollback raw provenance는 `stable-default` arm으로만 제한한다.

### Initial lifecycle supervisor receipt v1

`write_c02_lifecycle_supervisor_receipt_v1.py`와
`c02-lifecycle-supervisor-receipt-v1.schema.json`은 initial runner의 raw-only
terminal edge다. 이 writer는 독립된 나중 단계가 아니라 **같은 process**에서 먼저
closed v4 binder를 호출하고, 성공한 v4 manifest를 held private-root FD로 replay한
다음 source-owned shutdown-v2 artifact와 matching nonhidden completion marker를
replay하여 receipt를 publish한다. 따라서 v4 final marker의
`ambiguous-terminal-publication` 결과 뒤에 보이는 leaf를 다른 process가 성공
lifecycle로 재해석할 수 없고, 그 경우 receipt는 생성되지 않는다.

receipt는 v4 manifest, derived target, config endpoint/startup/bridge,
한 scenario capture, 한 observation, shutdown artifact/marker의 exact descriptors만
bind한다. schema 상태는 정확히 `status: "completed"`와
`qualification_status: "not-run"`이다. 이는 좁은 raw lifecycle이 완료됐다는 뜻일
뿐 fallback event의 존재, rollback 성공, candidate freeze, Gate E, semantic
workload 판정 또는 C02 pass를 뜻하지 않는다. native fallback **capture/binder**,
rollback raw runner와 private raw operational replay는 landed했지만, semantic
receipt/finalizer, clean candidate freeze와 실제 GPU capture는 이후 versioned work로
남는다.

lifecycle runner보다 먼저 `check_c02_config_bridge_v1.py`를 추가한다. 이 helper는
private evidence root와 endpoint/startup/direct `<capture>/session.json` path, expected
candidate/profile만 받아 held-FD로 config bridge를 pure replay하고 configuration SHA와
PID/start-tick/listener/GPU tuple을 derive한다. caller-supplied SHA/target, GPU/network/
subprocess 실행, evidence leaf 생성, qualification verdict는 금지하며 canonical stdout의
`bound` / `not-run` diagnostic report만 낸다. 그 closed report는
`c02-config-bridge-replay-v1.schema.json`으로 publish하며, lifecycle runner는 이
report가 derive한 SHA/target 외의 caller input을 같은 사실로 받아들이지 않는다.

- raw `/v1/config` 및 startup artifact bytes
- raw public request/response or SSE bytes와 matching generation audit v2
- same PID/start-tick pre/post C02 metrics, `/proc` status/stat/fd/socket and `/proc/net/tcp`
- selected GPU UUID/compute-app raw output
- scenario lifecycle/process exit and create-only completion markers

Binder는 raw `/v1/config` endpoint 및 startup artifact canonical bytes도
descriptor로 bind한다. endpoint `runtime_identity`에서 candidate, profile,
`configuration_sha256`를 derive하고 startup artifact의 embedded endpoint
payload/hash와 같음을 확인한다. v3는 별도 create-only
`riley.c02-config-endpoint-observation.v1` leaf를 추가한다. 이 leaf는 exact
loopback GET request/HTTP response head/body hash+length와 pre/post stat, TCP,
FD socket, status, GPU selection/compute-app raw leaves를 보존한다. binder는
그 raw facts에서 `(PID,start-tick,listener port/inode,GPU index/UUID)`를
derive하고 모든 scenario observation tuple과 일치시킨다. 따라서 config-to-
scenario **process identity**는 raw layer에서 닫히지만, workload/Gate E와
sampling semantic verdict는 여전히 later semantic replay만 담당한다.

현재 `check_soak_v2_receipt.py`는 이 later checker의 이름을 선점하거나 semantic
receipt를 내지 않는다. source checkout 밖의 private `0700` evidence root를 하나의
held FD와 nonblocking shared lock으로 열어 direct nonhidden completed raw v4/v5
manifest만 exact raw verifier로 replay하고, canonical
`riley.soak-v2-semantic-replay-precheck.v1`의 `bound`/`not-run`,
`authority: raw-structural-only` diagnostic만 stdout으로 낸다. v1/v2/v3, raw report,
marker, bind request, nested/alias input은 거부한다. marker pair는 post-link `fsync`
ambiguity 뒤에도 visible할 수 있으므로 이것은 producer/lifecycle success나 fallback,
rollback, freeze, Gate E, threshold, campaign, interval/monotonicity semantic 결과를
뜻하지 않는다. outer RC3 finalizer는 이 precheck를 semantic receipt input으로
수용해서는 안 된다.

`check_soak_v2_receipt_v2.py`와 raw layer의 held-FD semantic-input closure는 위
원본 manifest/capture/audit/observation leaf를 같은 private root FD에서 두 번
replay한다. 이 checker는 saved precheck report, summary counter, free-form backend
event를 입력으로 받지 않고, source generation-audit의 positional output/selection
관계, request ID와 process/config identity, 각 observation session 내부의 strict
`elapsed_monotonic_millis` order, 그리고 `completed`/`failed`/`cancelled`/
`capacity_rejections` 누적 counter의 비감소를 다시 계산한다. v5는 별도 source
fallback-event가 audit selection의 exact projection이며 모든 selection이 typed
`gpu-greedy -> cpu-normative`, `nonzero-temperature`, committed transition임도
재확인한다. 결과 `riley.soak-v2-semantic-replay.v2`의 `passed`는 이 좁은
held-FD replay만 뜻하고 producer normal return, actual GPU capture, candidate freeze,
Gate E, semantic receipt, qualification, deployment, rollback, campaign threshold,
scenario 간/전역 interval order는 계속 `not-established`다. 따라서 outer finalizer는
이 public diagnostic을 durable semantic receipt input으로 승격해서는 안 된다.

### Rollback v2

`run_remote_rc3_rollback_capture.sh`와 `bind_raw_rc3_rollback_capture.py`는 candidate와 reconstructed baseline 각각의 PID/start tick, listener inode, health/generation/audit raw bytes, candidate shutdown artifact+marker, atomic rename 전후 device/inode/stat evidence를 보존한다. label 문자열이나 `atomic-rename` declaration은 증거가 아니다.

현재 raw closure 위의 다음 단계는 public path receipt checker가 아니라 private
`replay_rc3_rollback_operational_semantics_v1.py`다. 이 landed helper는 authenticated
runner가 같은 root/switch EX와 normal-return stack에 nested하여 original
candidate/source/rollback/atomic raw leaf를 다시 해석해 candidate drain/zero
ownership, replacement process/socket, generation response, shutdown marker와
isolated filesystem switch를 재구성한다. v4 raw manifest는 v3/session closure
cross-bind에만 쓰고 v4 completion pair, finalizer receipt, structural precheck를
semantic input으로 읽지 않는다. 반환값은 in-memory
`passed/not-run`, `raw-operational-semantics-only` diagnostic이며 semantic receipt,
freeze, Gate E, deployment rollback 또는 qualification authority가 아니다. caller-held
FD만으로 prior finalizer normal-return lineage는 증명되지 않으므로, 이 diagnostic은
same-stack capability나 receipt를 대체할 수 없다.

Landed `write_rc3_rollback_finalizer_receipt_v1.py`는 same root/switch EX stack에서
v3/v4 typed closure를 만든 직후 이 private replay를 **failure-only pre-publication
veto**로 호출한다. 결과는 disk나 v1 receipt JSON에 저장하지 않고 typed
candidate/bindings/v3/v4/transaction closure와만 cross-bind한다. 따라서 v1 receipt의
authority는 계속 `raw-finalizer-normal-return-only`이며, visible v1 pair나 ephemeral
diagnostic 어느 것도 semantic receipt/qualification input으로 승격되지 않는다.

`write_rc3_frozen_candidate_v1.py`와
`replay_rc3_frozen_candidate_v1.py`는 이제 separate fresh `0700` root의 exact one-leaf
`riley.rc3-frozen-candidate.v1` manifest를 만들고 held-FD로 다시 검증한다. writer는
freeze-input admission output이나 `freeze.raw`를 input으로 승격하지 않고, original canonical
request와 all declared raw leaf를 직접 replay한다. reconstructed baseline은 structural
vocabulary만 보지 않고 v2 raw graph 전체를 replay하며 candidate/baseline 간 raw descriptor
path reuse를 거부한다. 이 manifest의 authority는
`frozen-candidate-input-identity-only`, status는 `frozen/not-run`뿐이다. model weight를
복사하지 않는 input pin이므로 모든 future consumer는 original input root도 다시 hash해야
하며, visible manifest는 writer normal-return, source→archive/ELF/image/model provenance,
Gate E, rollback/deployment, semantic receipt 또는 qualification을 증명하지 않는다.

full frozen-candidate replay는 bounded control-plane JSON을 먼저 읽은 뒤 original request와
정확히 23개 nested baseline physical leaf의 combined closure를 최대 8,192 descriptor/1 TiB로
제한하고서만 raw recipe·artifact streaming을 시작한다. candidate/baseline path reuse는 그
budget preflight에서 fail-closed한다. source/input/frozen root는 held FD ancestry와 Linux mount
backing coordinate까지 disjoint해야 하고, writer는 create와 self-replay 뒤 visible output path를
held root FD에 다시 bind한다. private Python code는 caller FD를 직접 mutate하지 않지만 pinned
FD를 이용하는 trusted read-only Git source oracle에 의존하므로 configured Git executable/PATH와
그 behavior는 explicit trusted boundary다. schema의 not-established object는 writer
normal-return과 input-root immutability도 machine-readable하게 보존한다.

`replay_rc3_gate_e_input_inventory_v1.py`는 이제 four-gate RC3 evidence root의
canonical `gate-e-inputs.json`과 fixed 14 direct leaves를 held-FD로 replay한다. release bundle,
native/optimizer canonical-E0, Python-free, performance, soak leaf는 inventory descriptor로
stream-hash되고, frozen candidate/original source-input closure는 artifact replay 전후 다시
replay된다. root topology, exact entry set, path uniqueness, 1 TiB budget과 frozen
manifest/candidate/source cross-binding은 확인하지만 output은
`bound/frozen/not-run`, `gate-e-input-inventory-replay-only`일 뿐이다.

`write_rc3_gate_e_input_snapshot_v1.py`는 이 exact root를 새로 만드는 static preparer다.
14 fixed role의 already-produced opaque host file을 no-follow/single-link source policy 아래
각각 direct private `0600` leaf로 copy하고, fixed total budget에서 inventory의 최대 control
plane byte를 먼저 reserve한다. source checkout FD는 held 상태로 유지하고 input/frozen root에만
shared lock을 잡아 frozen identity를 먼저 replay하며, fresh output root는 exclusive lock 아래 all snapshot private
rehash → canonical inventory final leaf → held-FD structural self-replay 순서로만 진행한다.
source path/source descriptor와 status/authority/producer field는 inventory에 넣지 않으며,
partial root에는 marker/receipt/temporary child를 만들지 않고 보존만 한다.

`run_remote_rc3_gate_e_session_v1.sh --supervisor-smoke-test`는 subsequent
authenticated producer를 위해 scoped shared lock과 parent/child handoff만 verify하는
no-action probe다. Bash는 `BASH_ENV` 평가 뒤에는 trust anchor가 될 수 없으므로,
legacy five-run public Bash entry의 privileged body는 제거됐고 historical internal sentinel도
no-action으로 끝난다.
`run_remote_rc3_gate_e_session_v2.py --performance-source-contract-probe`는
reviewed isolated Python에서 fixed private performance body를 no-follow/nonblocking으로
열고 bounded bytes를 sealed `memfd`로 snapshot하는 별도 no-action foundation이다.
v2도 GPU lock/Bash/Docker/evidence/receipt/qualification을 만들지 않으며, retired no-action
stub bytes를 snapshot할 뿐이다. sealed source FD를 통해 new private raw core를 호출하는
versioned producer/envelope은 후속 단계다.
그 후속 producer의 execution authority는 mutable checkout이 아니라 fixed root-owned
external anchor여야 한다. `verify_rc3_gate_e_execution_anchor_v1.py`는
`/opt/riley/rc3-gate-e-v1`의 bootstrap/core/canonical manifest와
`/var/lib/riley/rc3-gate-e/lock` mode-`0700` directory를 no-follow held FD로
검증하는 no-action prerequisite다. bootstrap/core를 실행하거나 lock/GPU/Docker/evidence/
receipt/qualification을 만들지 않으며, 현재 root-installed bundle 부재는 fail-closed다.
그 `checked` output도 mutable checkout source에서 나온 installation preflight일 뿐
execution authority나 producer/semantic receipt/qualification input이 아니며 host mount
namespace·ACL·verifier source integrity를 establish하지 않는다.
artifact, evidence-root, candidate, GPU-selection grammar가 없고 Gate E-root entry, raw
evidence, marker, receipt, serialized handoff도 만들지 않는다.
GPU query/selection, Docker, subproducer, aggregate replay, semantic receipt, qualification은
모두 absent이며 zero exit는 authenticated-lock probe만 뜻한다. relative script path는
parent handoff 전에 fail closed한다.

write_rc3_gate_e_aggregate_replay_receipt_v1.py는 actual producer receipt가 아닌
private aggregate-replay-only terminal record다. caller-held input/frozen/Gate-E/source FD와
fresh·empty separate 0700 root를 exact topology로 다시 확인하고, aggregate private core
두 invocation의 canonical bytes가 같을 때만 candidate/source, inventory/frozen descriptor,
policy/anchor 및 aggregate digest/length projection을 fixed JSON/intent/completion pair로
publish한다. Gate E root의 fixed fourteen-leaf closure에는 receipt leaf를 쓰지 않으며
gate_e_status, raw capture, producer normal-return, durable semantic receipt, qualification은
계속 not-established다. final hardlink 뒤 directory sync가 ambiguous하면 visible pair가
남아도 resume하거나 actual producer authority로 해석할 수 없다. future actual producer는
이 v1 record를 승격하지 않고 자체 same-stack v2 producer/semantic receipt를 작성해야 한다.

따라서 이 writer는 immutable local copy와 structural input inventory 외에는 아무 것도
증명하지 않는다. original raw producer, source-to-artifact provenance, 14 leaf의 atomic
coherence, writer/producer normal-return, actual capture/GPU, evidence-root immutability,
semantic Gate E pass, durable receipt, qualification은 모두 not-established다. legacy public
runner output을 사후 snapshot한 결과를 same-stack producer closure로 승격하는 것은 금지한다.

`replay_rc3_gate_e_python_free_v1.py`는 full legacy E2E evaluator를 호출하지 않고 fixed
inventory와 frozen input closure로 닫을 수 있는 raw semantic subset만 replay한다. raw tar와 release
bundle은 held FD에서 private scratch로 copy해 path-based legacy loader에만 주고, report/golden/native
report는 held descriptor에서 직접 rehash/strict parse한다. frozen source archive, release ELF/image
digest와 selected model descriptor tuple은 raw replay result에 cross-bind하며, bundle 내부 binary는
frozen ELF와 같아야 한다. raw loader의 retained payload는 768 MiB, bundle verifier의 uncompressed
retained payload는 640 MiB로 component에서만 별도 제한하며 이는 전체 process RSS claim이 아니다.
caller가 주는 image/golden SHA는 동일성 anchor일 뿐이며, 그 승인 검토, 외부
model-mount/source/producer provenance, native/optimizer semantic pass, aggregate Gate E 및 qualification은
명시적으로 아직 확립하지 않는다.

따라서 check_rc3_rollback_receipt_v2.py와 outer qualification은 aggregate Gate E replay나
이 replay-only terminal record 자체가 아니라, **후속** authenticated actual Gate E producer의
same-stack normal-return closure와 versioned durable semantic receipt가 생긴 뒤에만 만들 수
있다. inventory와
aggregate replayer는 report의 `passed`/threshold를 해석해 actual Gate E pass를 만들지 않는다. legacy path-based
`check_release_candidate.py`는 RC3 Gate E replayer가 아니며 same-stack finalizer input으로
수용할 수 없다. `freeze.raw`, freeze-input admission, v4 completion pair, structural precheck와
raw finalizer receipt도 semantic input을 대체할 수 없다.

#### Current qualification-input denial boundary

그 durable receipt가 아직 없는 현재 경계에서는
`ci/release/rc3_qualification_input_policy_v2.py`가 **승인자가 아니라 pure
denial policy**로 동작한다. CLI, path-open, output receipt, replayer import를 두지 않고
caller가 already-held descriptor에서 읽은 canonical JSON bytes만 받는다. `ADMITTED_INPUTS`는
의도적으로 비어 있으므로 이 버전에는 success path가 없다. historical soak/rollback,
raw/structural precheck, narrow soak/rollback semantic diagnostic, freeze-input/frozen-candidate
identity, reconstructed content bridge/A-B materialization, Gate E component/aggregate replay 및
aggregate replay-record, legacy release-candidate 보고서는 각각 exact schema/authority별
reason으로 거절한다. 알려지지 않은 schema, `.v2` suffix, `status: passed`, 혹은
`qualification_status: not-run`도 allowlist를 만들지 않으며 모두 fail-closed한다.

이 denial 자체도 receipt나 qualification input이 아니다. future authenticated actual producer가
same-stack normal-return와 versioned durable semantic receipt를 함께 제공하는 PR에서만, 그
exact schema/authority를 별도 리뷰한 allowlist로 추가할 수 있다. 그 전에는 이 policy를
완화하거나 replayer를 호출해 결과를 up-convert해서는 안 된다.

#### Reconstructed RC2 compatibility boundary

현재 reconstructed `riley-0.1.0-rc2` annotated tag object (`a3f5203…`)와 tag target (`6093006…`)에는
`/v1/c02/metrics`, `--c02-audit-dir`, generation-audit-v2, shutdown-v2
artifact surface가 없다. 반면 published rollback raw-provenance **v2**는 candidate와
rollback 양쪽에 그 C02 observation-session-v2 grammar를 요구한다. 따라서
reconstructed RC2를 v2 rollback runner에 직접 넣을 수 없으며, wrapper가 C02
metrics/audit를 합성하거나 v1 receipt를 v2로 up-convert하는 것도 금지한다.

v2의 closed schema를 이 legacy case에 맞춰 넓히지 않는다. 새
`rollback-receipt-v3.schema.json`과 `check_rc3_rollback_provenance_v3.py`는
reconstructed RC2 annotated tag object `a3f5203c3a72122e9da818c1e441c2a789f7aa8c`와 target `6093006ec2b01b784b01ba278296b676f2dfd03a`에 pin된
별도 raw-only surface를 정의한다. v3는 same held private root FD로 existing
reconstructed-baseline A/B checker를 replay한다. 선언된 phase PID/start tick,
listener port/inode, GPU tuple은 `/proc` stat/status, TCP/FD-socket, GPU selection/
compute-app raw leaves와 교차검증한다. candidate의 `source-owned` audit은
availability와 audit-index **opaque descriptor inventory**만 이 단계에서 bind하며,
index content/source replay는 후속 layer의 책임이다. baseline은 명시적으로
`not-supported` audit 상태만 가진다. candidate shutdown-v2 pair, active baseline
bundle/image ID, raw atomic-switch material도 bind한다. 이는 `bound`/`not-run` raw
report만 내며 HTTP/rename의 의미, rollback success, historical stable status는
semantic checker가 후속 단계에서 재구성한다.

`bind_raw_rc3_rollback_capture.py`는 이제 closed path-only request에서 full
reconstructed baseline replay와 raw target derivation을 마친 뒤 create-only
nonterminal v3 manifest 하나를 self-verify해 publish한다. service/GPU/SSH,
rename, qualification을 조작하지 않으며 `.complete`/`.intent`는 만들지 않는다.
`capture_rc3_rollback_phase_v1.py`는 legacy-compatible `/readyz`, optional
non-stream completion, PID/TCP/FD-socket/GPU raw leaves를 같은 prepopulated
private baseline root의 새 capture child에만 append한다. candidate source-audit
generation은 여전히 별도의 source-owned scenario producer가 담당하고, RC2에는
audit을 합성하지 않는다.
fixed-name writer의 phase 선행 조건은
`capture_rc3_rollback_phase_v1.py`의 strict held-FD replay API
`replay_rc3_rollback_phase_v1_fd()`가 phase session을 먼저 재생한다. 이
replayer는 같은 private root FD 아래 exact capture/raw child를 no-follow로 열고
canonical `riley.rc3-rollback-raw-phase-capture.v1` session, fixed raw inventory,
descriptor의 path/SHA-256/byte length, exact private `0700`/`0600` euid ownership,
`capture-incomplete.json` 부재를 재검증한다. `/proc`·TCP·FD-socket·GPU raw
leaf에서 target tuple을 재derive하여 session 값과 교차검증하고, health 및 optional
non-stream completion HTTP exchange는 bounded fixed grammar로만 replay한다. 이는
raw structural input을 derive할 뿐 service/GPU/network/SSH/Docker/rename을 실행하거나
terminal·rollback authority를 만들지 않는다.
그 다음 fixed-name writer의 선행 조건인 held-FD **candidate-source join**은 이제
`replay_rc3_rollback_candidate_source_v1.py`로 landed 되었다. 유일한 callable entry
`_replay_candidate_source_join_on_held_root_fd()`는 이미 보유한 private root FD와
fixed candidate-phase·serial-capture·source-audit/shutdown·config-bridge 이름만 쓰며,
CLI나 caller-selected path surface를 노출하지 않는다. phase replay만으로는 여전히
충분하지 않다. v1 serial-scenario replayer는 이미 검증한 request·response-head·
response-body descriptor만 typed `ReplayedScenario` field로 노출하여 후속 writer가
추측한 ledger path를 다시 열지 않게 한다. join은 정확히 하나의 stable-default
source scenario를 replay하고 그 PID/start-tick/listener tuple이 candidate phase와
같은지 확인하며, candidate phase의 local generation exchange를 거부하고 candidate
generation/audit-index를 source replay에서만 derive한다. source-owned shutdown-v2
artifact/marker도 같은 derived target에 대해 별도 replay하고, config bridge의
PID/start-tick/listener/GPU tuple도 candidate phase와 일치해야 한다. 전체 held-FD
replay를 한 번 더 수행하여 drift를 거부한 뒤 typed raw input만 반환하며, bind request나
terminal evidence는 만들지 않는다. static snapshot digest는 runtime `/v1/config`의
launch-identity SHA-256과 다른 hash domain이다. landed
`rc3-static-effective-config-v1.schema.json`와 held-FD-only
`replay_static_effective_config_v1_fd()`는 join 안에서 이 관계를 좁게 cross-bind한다.
fixed static snapshot은 canonical
`riley.rc3-static-effective-config.v1` intent로 candidate, stable-default profile,
모든 effective-config dimension과 canonical effective-config digest를 담아야 한다.
helper는 terminal static preparation에서만 candidate/profile을 derive하고 independent
config bridge를 replay하여 effective-config value/digest의 일치를 확인한다. 이때
bridge-derived launch-identity digest는 별도 값으로 retain하며 snapshot digest와 같다고
비교하지 않는다. 기존 opaque static snapshot은 up-convert하지 않고 이 새 path에서
fail closed한다.
이제 private fixed writer
`write_rc3_rollback_candidate_source_bind_request_v1.py`가 landed 되었다. 유일한
callable entry
`_write_fixed_candidate_source_bind_request_on_held_root_switch_fds()`는 caller가
계속 보유한 private root FD와 fixed switch FD만 받으며, caller는 전체 lexical call
동안 root EX와 switch EX를 유지해야 한다. CLI, path-open wrapper, lock/relock,
output-name parameter, caller-supplied candidate/profile/target/descriptor surface는 없다.
writer는 candidate/source join, non-stream generation이 반드시 있는 fixed
`rollback-phase`, terminal artifact-exchange transaction을 replay하고 legacy v3
path-only request를 typed 결과에서만 derive한다. candidate process/health는 candidate
phase에서, candidate generation/audit/shutdown은 source join에서만 오며 static
baseline/snapshot, transaction artifact, atomic-switch path도 추측하거나 받지 않는다.
그 결과 fixed nonterminal root leaf
`rollback-v3-candidate-source-bind-request.json` 하나만 create-only로 쓰고
`.intent`/`.complete` sibling을 미리 reserve한다.

writer는 create-only request publication 직전에 모든 typed input을 독립적으로 다시
rebuild한다. static checkpoint
`_recheck_static_preparation_bindings_on_held_root_fd()`는 terminal
static-preparation session에서 시작하여 initial typed binding의 baseline descriptor와 세
snapshot descriptor뿐 아니라 candidate ID/stable-default profile도 다시 비교한다. 단순
path rehash만으로는 static replay 뒤 같은 EUID가 바꾼 bytes 또는 preparation identity가
완료 receipt와 무관하게 bind되는 것을 막지 못한다. candidate-source join의 immutable
complete consumed-path inventory를 rollback phase/artifact/atomic-switch collision set의
seed로 써서 config bridge·serial capture의 보조 leaf도 후속 역할과 alias하지 못하게
한다. 따라서 이 static/cross-role TOCTOU edge는 request publication에서 닫히지만,
writer는 v3/v4 manifest, terminal marker, rollback/lifecycle/qualification claim을 만들지
않는다.

이제 private same-stack finalizer
`finalize_rc3_rollback_candidate_source_v4.py`도 landed 되었다. compatibility report helper
`_finalize_rollback_candidate_source_v4_on_held_root_switch_fds()`와 receipt 전용 typed closure
helper는 caller가 같은 root EX/switch EX를 유지할 때만 fixed request → nonterminal v3 → v4를
normal-return stack에서 연결한다. v3 직전과 v4 paired terminal marker 직전에 candidate/source·rollback phase·atomic
transaction의 **전체** typed replay equality와 held root FD의 fixed request descriptor/document를
모두 다시 비교한다. static binding만 recheck하면 path-only v3 binder가 다시 읽는 raw leaf의
TOCTOU를 닫지 못한다. v4 manifest는 preflight/create-only/self-replay 뒤에만 durable intent와
hard-linked completion pair를 시도하고, post-link sync ambiguity는 resume이나 producer success가
아닌 `ambiguous-terminal-publication`으로 fail closed한다.

그 정상 반환 edge를 보존하는 다음 private continuation
`write_rc3_rollback_finalizer_receipt_v1.py`도 landed 되었다. 유일한 entry
`_finalize_and_write_rollback_receipt_on_held_root_switch_fds()`는 caller-held root EX/switch
EX에서 먼저 finalizer를 직접 호출하고, 반환된 typed v3/v4 closure와 fixed request, static
preparation, candidate/source complete consumed-path inventory, candidate/rollback phase 및
atomic transaction을 **같은 FD stack에서** 다시 비교한 뒤 fixed
`rollback-finalizer-receipt-v1.json`과 paired `.intent`/`.complete`를 create-only로
publish한다. 그 첫 receipt leaf 전에는 private operational replay가 same held FD에서
failure-only veto로 실행되어 typed v3/v4/transaction closure와 다시 cross-bind된다.
그 diagnostic은 disk나 v1 receipt JSON에 남지 않는다. schema는 `riley.rc3-rollback-finalizer-receipt.v1`,
`completed/not-run`, `raw-finalizer-normal-return-only` authority로 한정된다. receipt output
충돌은 finalizer 전에 막고, 모든 closure/receipt replay도 receipt leaf/marker 전에 끝낸다.
terminal hardlink helper가 성공하면 그 뒤 즉시 반환한다. v4 또는 receipt marker의 post-link
sync ambiguity와, 정상 완료된 v4 뒤 operational veto의 failure 모두 receipt 성공을 만들지
않는다. veto failure는 completed v4 pair만 남기고 receipt pair를 전혀 만들지 않을 수 있으며,
fixed output collision 때문에 그 root를 path/FD reopen으로 재개할 수 없다. runner는 해당 root를
조사용으로 보존·retire하고 새 private root에서만 새 시도를 시작해야 한다. receipt pair
자체도 filesystem-only success 증명이 아니므로
path reopen/CLI/resume 또는 future semantic checker가 독립 input으로 수용할 수 없으며, 같은
authenticated runner normal-return stack만 이 함수의 성공 반환을 소비할 수 있다.

후속 authenticated runner가 고정 rollback source bridge를 만들 때 필요한 좁은
`materialize_rc3_rollback_candidate_config_v1.py`도 landed 되었다. private
`_initialize_candidate_config_directory()`는 preexisting private evidence root 아래에만 새
mode-0700 `config/` child를 만들고, candidate launch 뒤 observer가
`config-bridge/raw/config-endpoint.json`을 capture하고 server가 `config/startup.json`을
create-only로 남긴 뒤에만 `_materialize_candidate_config_bridge()`가 실행된다. 이 helper는
`stable-default` profile만 받아 먼저 raw observer endpoint로 held-FD config bridge를 replay하고,
exact private inventory `{startup.json}`을 확인한 뒤 raw body를 새
`config/endpoint.json`으로 create-only projection한다. 그 다음
`{startup.json, endpoint.json}` inventory와 fixed endpoint bridge를 다시 replay해 candidate,
profile, endpoint/startup/session/effective-config/target descriptor drift를 거절한다. 이는
endpoint observer·startup artifact·process·GPU·HTTP lifecycle을 만들거나 성공을 판정하지 않는다.
따라서 visible fixed config leaf는 standalone runner/rollback/semantic authority가 아니며, 미래
authenticated runner의 같은 held-FD normal-return sequence 안에서만 raw capture와 함께 소비되어야 한다.

그 fixed topology를 실제 finalizer receipt까지 끊기지 않게 잇는 private
`compose_rc3_rollback_finalizer_receipt_v1.py`와 runner-only wrapper
`finalize_rc3_rollback_finalizer_receipt_v1.py`도 landed 되었다. wrapper는 preexisting private root를
no-follow로 열어 EX를 한 번만 잡고, core는 **artifact preparation → atomic transaction → fixed
candidate/source v3/v4 finalizer → ephemeral operational veto → finalizer receipt**를 한 lexical
callback chain으로만 호출한다.
fixed request/v3/v4/receipt 및 모든 reserved sibling은 artifact preparation 전에 preflight하며,
terminal receipt branch는 별도 caller-supplied candidate/config/phase target, fixed evidence
path/name, descriptor, continuation을 받지 않는다.
receipt hardlink가 성공한 뒤에는 transaction의 ordinary post-callback FD recheck를 쓰지 않고
terminal-only continuation과 quiet cleanup만 남겨 visible receipt를 later cleanup failure로 failed
return으로 바꾸지 않는다. 이 compositor/wrapper는 dynamic config/phase/source/HTTP evidence를
capture하거나 process·GPU·network·deployment을 실행하지 않는다. 따라서 아직 authenticated
operational rollback runner가 아니며 preexisting fixed raw topology를 같은 invocation에서 소비하는
CPU-only closure일 뿐이다.

#### Authenticated RC3 rollback raw-capture runner

`run_remote_rc3_rollback_capture.sh`는 위 fixed topology를 실제 dynamic raw
producer와 연결하는 유일한 authenticated host runner다. outer clean Python
supervisor가 canonical no-follow GPU lock을 보유하고 child는 parent PID/executable,
inherited lock FD, kernel flock 및 random token을 모두 검증한 뒤 lock FD를 닫는다.
runner는 caller server command, profile/configuration SHA, PID/start tick,
listener/GPU UUID, capture/audit/manifest/receipt path를 받지 않는다. legacy
`--id=...`를 포함한 unknown option도 lock/GPU/filesystem 접근 전에 거절한다.

한 successful invocation은 preexisting private reconstructed RC2 root의 static
preparation과 fixed `config/` child를 먼저 create-only로 만든다. candidate는
runner-owned `stable-default` C02 identity/startup/source-audit/shutdown paths로만
`env -i` launch된다. config observer 뒤 private projector가
`config/endpoint.json`을 materialize하고, candidate health-only phase와 정확히 하나의
source-owned serial scenario를 capture한 뒤 guarded shutdown을 수행한다. reconstructed
RC2 arm은 C02 option 없이 별도 loopback port에서 launch되어 canonical non-stream
generation request를 가진 rollback phase만 capture한다. candidate/rollback host binary와
model tree는 launch 전후 다시 hash-verified되며, PID reuse 방지는 every capture edge와
guarded TERM/KILL path에서 유지된다.

serial contract는 GPU preflight 전에 canonical stable-default standard grammar와 **exactly
one** scenario를 재검증하고, rollback request도 같은 시점에 canonical non-stream grammar로
revalidate한다. 두 port는 서로 달라야 한다. dynamic evidence가 모두 normal-return하고 두
server identity가 제거된 뒤에만 shell은 private
`_finalize_authenticated_rollback_raw_once(PreparationRequest(...))`를 `exec`한다.
그 held-FD compositor는 artifact preparation 전에 candidate/config/source와 rollback phase
전체를 read-only로 두 번 replay하여 malformed·drifting·cross-role-reused evidence가 새
artifact/atomic surface를 남기지 못하게 한다.
그 final process는 fixed artifact preparation → atomic transaction → v3/v4 → ephemeral
operational veto → finalizer-receipt chain 외의 binder/writer/checker를 호출하지 않으며 successful receipt 뒤
post-return I/O를 하지 않는다. scratch logs는 evidence root 밖에 두고 failure 때만 보존한다.

이 runner의 landed scope는 **CPU/static contract validation**이다. 실제 GPU capture,
candidate freeze, deployment-path mutation, semantic rollback decision, Gate E 또는
qualification result는 아직 생성하거나 주장하지 않는다.

dynamic phase/source-audit path가 정해지기 전에는
`prepare_rc3_rollback_evidence_v1.py`가 이미 complete한 private reconstructed
RC2 root와 root-relative manifest를 held-FD로 full replay해 reviewed
`riley-0.1.0-rc2` annotated tag object/target만 admit한다. 이 helper는 same-semver immediate RC
candidate ID와 서로 다른 세 absolute opaque input(freeze, base release candidate
report, stable-default configuration)만 받고, external host path를 session에
기록하지 않은 채 fixed `rollback-v3-evidence-inputs/`의 immutable 0600 leaves와
`rollback-v3-evidence-preparation/`의 closed `captured/not-run`,
`raw-static-preparation-only` session/paired completion receipt를 create-only로
남긴다. baseline을 import·copy·생성하지 않고 opaque input을 actual config/freeze
evidence로 해석하지도 않으며 service/GPU/network/rename/bind request/qualification을
조작하지 않는다. 따라서 완료 pair는 raw static preparation일 뿐 authenticated
runner나 rollback authority가 아니다.
`capture_rc3_rollback_atomic_switch_v1.py`는 같은 root 안 runner-owned
isolated switch child의 이미 staged된 private runtime files에만 Linux
`renameat2(RENAME_EXCHANGE)`를 적용하고, active 전후·rollback/candidate staged
stat과 transcript 다섯 raw leaf를 create-only capture child에 남긴다. 실제
deployment path에는 rename하지 않으며 `mv`/ordinary rename fallback도 없다. 이
raw helper만으로 runtime copy와 artifact-map binary/bundle/image bytes의 content
linkage는 주장하지 않는다. 그 선행 mapping은 이제 별도
`prepare_rc3_rollback_artifacts_v1.py`가 담당한다. 이 helper는 fixed
create-only `rollback-v3-artifact-snapshot/`, `rollback-v3-artifacts/`,
`rollback-v3-switch/`만 append하고, 여섯 absolute host input을 no-follow
streaming으로 immutable 0600 snapshot에 복제한 뒤 candidate/rollback binary
snapshot에서만 distinct 0700 `active`/`rollback-staged` runtime copy를 만든다.
terminal session verifier는 snapshot/runtime hash·length·inode mapping을
재생하지만 runtime path를 artifact descriptor로 승격하지 않는다. 세 raw producer의
terminal state는 `capture-incomplete.json` 부재만으로 결정되지 않는다. 각 session의
SHA-256/byte length에 bind된 mode-0600 two-link
`capture-complete.intent`/`capture-complete.json` receipt pair를 held-FD reader가
함께 검증해야 하며, unlink/fsync/복구 중단으로 pair가 없거나 반쪽만 남으면 fail closed한다. source
checkout/deployment/GPU/network/Docker/rename는 건드리지 않는다.
verifier의 `pre-switch`/`post-switch` layout은 각각 그 순간의 bytes/inode mapping만
재생한다. `post-switch`는 단독으로 exchange 발생을 주장하지 않는다. 새
`capture_rc3_rollback_atomic_transaction_v1.py`는 one exclusive root/switch FD 아래
pre-switch replay → isolated `renameat2` capture → terminal atomic replay → post-switch
replay를 하나의 raw subtransaction으로 묶는다. create-only transaction session은
preparation/atomic terminal session descriptor를 bind하고, 두 runtime의 pre/post
SHA-256·inode·mode·link·size·time layout과 atomic pre/post stat 방향을 직접 join한다.
atomic helper는 exchange 직전과 직후 private staged bytes를 hash해 same-inode/same-size
in-place mutation도 reject한다. held child FD와 0600/euid session JSON, incomplete-marker
부재 및 completion receipt pair를 terminal replayer가 다시 확인하며 결과는 계속
`captured/not-run`이다.

completion hardlink의 post-link parent-directory `fsync`가 오류를 반환하면 helper는
`ambiguous-terminal-publication`으로 실패하고 captured return을 내지 않는다. pair가 disk에
남아 fresh raw verifier로 structural replay될 수 있어도, 그것은 failed invocation의
producer-success/terminal authority가 아니다. future authenticated binder/runner는 같은
invocation·held lock에서 normal return한 branch만 소비해야 하며, ambiguous error 뒤에
filesystem을 다시 읽어 bind나 operational action을 재개하면 안 된다.

### Same-invocation rollback terminal provenance v4

`check_rc3_rollback_provenance_v4.py`는 held root/switch FD에서 v3 manifest와 fixed
transaction closure를 구조적으로 replay하는 diagnostic helper다. v3 candidate/rollback
artifact descriptor maps는 preparation session의 immutable snapshot maps와, v3의 다섯
atomic-switch descriptors는 atomic child session과 각각 **path·SHA-256·byte length까지**
일치해야 한다. transaction의 session descriptor와 pre/post runtime inode/hash join도
같이 replay한다. 결과는 여전히 `bound`/`not-run` raw report이고 source/config/HTTP의
semantic meaning, host lifecycle, GPU, deployment 또는 rollback success는 주장하지 않는다.

`check_rc3_rollback_structural_precheck.py`는 completed v4 manifest와 paired
`.intent`/`.complete`를 같은 held private root→switch shared-lock FD stack에서 다시
읽는 좁은 admission diagnostic이다. direct v4 root leaf만 allowlist로 수용하고 v1/v2,
의도적으로 nonterminal인 v3, raw report/marker/bind request 및 alias를 거부한다. 출력은
canonical `riley.rc3-rollback-raw-structural-precheck.v1`의 `bound`/`not-run`,
`authority: raw-structural-only`뿐이다. visible pair가 `ambiguous-terminal-publication`
뒤에 남아도 producer success, host rollback, lifecycle, freeze, Gate E, semantic receipt나
qualification 결과가 아니며 future `check_rc3_rollback_receipt_v2.py`와 outer RC3
finalizer는 이를 semantic input으로 수용해서는 안 된다.

`bind_raw_rc3_rollback_terminal_v4.py`에는 path-based CLI, 기존 preparation/transaction을
reopen하는 wrapper, 또는 held-FD publisher API가 없다. 유일한 public raw producer는 새
`PreparationRequest`로 fixed preparation을 만든 뒤, **같은 held exclusive root/switch FD
stack의 nested normal-return closure**에서만 transaction → nonterminal v3 → v4 publication을
연결한다. 따라서 fresh v4 checker나 visible preparation/transaction completion pair는 다음
단계를 시작할 권한을 만들지 못한다. preparation, transaction, v4 중 어느 단계든 post-link
directory `fsync` ambiguity가 나면 `ambiguous-terminal-publication`으로 실패하고 뒤 callback은
실행되지 않는다. v4 own manifest는 full held-FD replay, create-only JSON, self replay,
durable `.intent`, hard-linked `.complete` 순서로 publish하며, 보이는 pair도 structural
evidence일 뿐 이후 process가 successful rollback/terminal authority로 재해석할 수 없다.

이것은 artifact-exchange subtransaction과 그 narrow raw v4 join까지만 닫는다. complete reconstructed-baseline,
candidate/rollback phase, source audit/shutdown/config bridge를 연결하는 authenticated
host runner는 아직 없고 v3 binder도 이 transaction session을 소비하지 않는다. future
runner는 `replay_rc3_rollback_phase_v1_fd()`로 candidate/rollback phase 각각의
exact session/raw inventory/derived target와 `capture-incomplete.json` 부재를 먼저
replay하고, static preparation·artifact preparation·atomic transaction에 대해서만
session-bound completion receipt pair를 별도로 replay한 뒤에만 raw path를 bind
request에 넣어야 한다. phase collector 자체는 paired completion receipt를 publish하지
않으므로 replayer가 이를 합성하거나 추론해서는 안 된다.
따라서 actual GPU rollback drill, candidate
freeze, deployment rollback success verdict를 실행하거나 주장하지 않는다.

새 binder가 input replay → create-only manifest publication → self-verification을
중간 root 재open 없이 수행할 수 있도록, existing v2 raw verifier also exposes a
held-FD replay entry point. 그것은 raw descriptor 안전성만 보강하며 legacy RC2를
v2-compatible하다고 바꾸거나 terminal/semantic authority를 만들지 않는다. v3
path wrapper는 source checkout 내부 evidence root를 거부하며, 향후 FD-only binder도
root를 열기 전 같은 preflight를 수행해야 한다.

## 6. 변경 순서

1. v2 schemas와 strict shared evidence primitive를 추가하고 v1 rejection policy를 문서화한다.
2. reviewed RC2 source-input preparer와 per-arm OCI image-layout input preparer, 그리고 opaque v2 leaves를 각 closure에 held-FD cross-bind하는 bridge를 추가한 뒤 reconstructed baseline builder/checker와 adversarial tests를 추가한다. 이 three-way preparer/bridge 자체는 build/capture/qualification을 실행하지 않는다.
3. typed sampling selection, private-FD generation audit, 그리고 source-owned
   shutdown v2 artifact/marker producer를 Rust source에 추가한다.
4. v4 serial capture-session binder/schema와 hostile fixture tests를 추가한다.
5. config endpoint process bridge, initial one-scenario lifecycle runner, same-process
   v4/shutdown receipt closure와 hostile/static tests를 추가한다. 이 구현만으로는
   GPU capture나 qualification을 실행하지 않는다.
6. landed native fallback source leaf를 replay하는 capture/binder와 reconstructed
   baseline rollback v3 raw verifier/schema, strict held-FD phase replay와 typed
   source HTTP descriptor replay, versioned static-to-effective-config replay, fixed
   held-FD candidate-source/config/shutdown join, complete consumed-path inventory 및
   publication-bound static identity/descriptor TOCTOU closure를 가진 fixed-name raw
   bind-request writer와 same-stack private normal-return v3/v4 finalizer까지 추가한다.
7. landed raw-structural precheck를 semantic checker로 승격하지 않고, 별도 soak/rollback
   v2 semantic checker를 추가한 뒤 outer RC3 finalizer를 v2-only로 바꾼다.
8. 이 P1 source가 clean commit으로 고정된 뒤에만 new candidate를 freeze하고 GPU qualification capture를 시작한다.

## 7. 완료 조건

- v1 receipt, self-authored fallback string, missing strict-open flags, self-authored worker/model IDs는 final C02 input에서 fail closed한다.
- v4 serial soak provenance는 raw descriptor hash뿐 아니라 capture session의
  incomplete-marker closure, contract/request/audit linkage와 config endpoint 및
  every scenario의 same candidate/config/PID/start-tick/listener inode/GPU tuple을
  replay한다. v3 input은 serial-capture path에서 허용하지 않는다.
- max-performance-exact GPU-greedy ineligible case는 Rust-written audit v2,
  native fallback event/marker pair와 public generation bytes가 후속 versioned
  capture에서 one-to-one으로 bind된다.
- shutdown v2 artifact와 nonhidden completion marker는 same PID/start-tick tuple과
  exact final-metrics bytes를 bind하며, v1 artifact/hidden marker는 fail closed한다.
- initial lifecycle receipt는 same-process successful-v4 edge에서만 shutdown
  artifact/marker를 다시 bind하고 `completed`/`not-run` raw status만 낸다. CPU/static
  tests 또는 receipt 존재만으로 GPU capture, freeze, fallback/rollback semantic result를
  주장할 수 없다.
- reconstructed baseline manifest는 previous stable artifact라고 주장하지 않으며, A/B reconstruction equality와 artifact provenance를 검증한다.
- C02 finalizer가 soak-v2/rollback-v2만 수용하고 resulting final report에서 operational rollback과 historical-stable rollback status를 분리한다.
- 이 단계는 candidate freeze, Gate E pass, C02 pass, vLLM win을 주장하지 않는다.
