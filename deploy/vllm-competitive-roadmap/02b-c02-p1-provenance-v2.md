# C02-P1 — Provenance v2와 Reconstructed Rollback Baseline

**상태:** In progress — v1 provenance 결함을 확인했고, candidate freeze 전에 v2 contract와 source-owned raw producer를 구현한다. initial C02 lifecycle supervisor와 raw receipt는 구현됐지만 현재 검증 범위는 CPU/static hostile-path뿐이며, GPU capture·candidate freeze·semantic qualification은 아직 수행하지 않았다.
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

stable-default arm이 CPU sampling이면 이 evidence를 재사용할 수 없다. max-performance-exact endpoint/startup artifact raw bytes가 GPU-greedy configuration임을 먼저 bind해야 한다. attention/GEMM/executor fallback이 필요해지면 별도 runtime feature를 먼저 설계하고 이 P1을 그 증거로 바꾸지 않는다.

## 4. Reconstructed-tag baseline

`riley-0.1.0-rc2`는 owner-approved prerelease이고 prior stable shipped binary/bundle/OCI image가 아니다. 원격 cache와 published RC2 release에도 serving artifact가 없으므로, C02-P1은 그것을 `previous_stable` 또는 `historical_shipped`로 표기하지 않는다.

P1은 pinned annotated tag와 target commit에서 독립 clean A/B network-none build를 수행해 `riley.reconstructed-prior-baseline.v1` manifest를 만든다.

```text
baseline_kind = reconstructed-tag-baseline
provenance_class = reconstructed-from-source
historical_distribution = not-attested
was_previously_shipped = false
historical_stable_artifact_status = unavailable
```

manifest는 tag object/target, source archive, exact build recipe and image inspect, A/B binary/profile/bundle equality, runtime OCI inspect/archive, final artifact descriptors를 create-only로 bind한다. 공개 RC2 release API raw response는 mutable information이므로 보존할 수 있지만 trusted expected digest는 reviewer-provided input으로 따로 비교한다.

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
검증된 구현이다. 아직 원격 GPU capture, candidate freeze, Gate E replay, semantic
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
restart/rollback/multi-PID, `exact-backend-fallback`을 fail closed한다.
특히 현재 source에는 generation-audit record와 별개인 native fallback-event leaf가
없으므로 audit record를 복사하거나 config 문자열로 대체해서 fallback을 주장하면
안 된다. initial lifecycle runner는 config bridge → scenario producer → C02 observer
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
`NAME.json`과 `NAME.json.complete`가 모두 없는 경우에만 terminal output을 생성한다.

v4 binder는 session descriptor, parent `capture-incomplete.json` 부재,
contract inventory, request/response ID-to-source-audit marker, 그리고
scenario PID/start-tick/listener proof를 explicit replay한 뒤에만 lifecycle
runner output을 수용한다. 그 tuple은 config bridge와 observation session의
PID/start-tick/listener/GPU tuple에도 일치해야 한다. 새 terminal manifest와 marker는
각각 `riley.soak-v2-raw-provenance.v4` 및
`riley.soak-v2-raw-provenance-complete.v4`이고, marker는 정확히
`schema_version`, `artifact_filename`, `artifact_sha256`만 가진다. v4는
`exact-backend-fallback`을 fail closed한다; 해당 source-owned native leaf가
추가된 후 별도 version bump에서만 다시 다룬다.

이 full config-bridge/serial/observation join은 manifest 생성 **전에** 완료한다.
따라서 정상적인 target/GPU/bridge drift는 create-only nonterminal manifest를 남기지
않는다. v4 completion marker는 먼저 별도 durable nonterminal intent leaf를 만든 뒤
create-only linked final marker로 공개한다. 따라서 final 공개 전 file-sync 실패는 final
marker를 만들지 않는다. final marker가 보인 뒤 parent-directory sync가 실패하면 binder는
`ambiguous-terminal-publication` nonzero로 끝나며 lifecycle success receipt를 만들면 안
된다. raw verifier가 paired intent/final marker를 읽을 수 있어도, 이후 qualification/
finalizer는 이 ambiguous 결과 뒤의 visible marker만으로 lifecycle authority를 인정하지
않고 runner supervisor의 성공 receipt를 추가로 요구한다.

`bind_raw_c02_soak_v2.py` 계열은 runner를 대체하거나
service/GPU/SSH/container를 조작하지 않는다. v3 schema는 config bridge의
historical closed shape로 남기고,
`benchmarks/release/candidates/soak-v2-bind-request-v4.schema.json`가 v4의
closed path-only shape를 publish한다. 어느 버전도 caller-supplied
descriptor/hash를 받지 않는다.

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
workload 판정 또는 C02 pass를 뜻하지 않는다. native fallback leaf, rollback raw
runner, semantic checker/finalizer, clean candidate freeze와 실제 GPU capture는 이후
versioned work로 남는다.

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

`check_soak_v2_receipt_v2.py`는 summary counter나 free-form backend event를 믿지 않고 위 leaf에서 identity, interval order, request/audit binding, metrics monotonicity, actual typed sampling selection을 재구성한다.

### Rollback v2

`run_remote_rc3_rollback_capture.sh`와 `bind_raw_rc3_rollback_capture.py`는 candidate와 reconstructed baseline 각각의 PID/start tick, listener inode, health/generation/audit raw bytes, candidate shutdown artifact+marker, atomic rename 전후 device/inode/stat evidence를 보존한다. label 문자열이나 `atomic-rename` declaration은 증거가 아니다.

`check_rc3_rollback_receipt_v2.py`는 frozen candidate가 pin한 baseline manifest를 replay하고, candidate drain/zero ownership, replacement process/socket, generation response, shutdown marker, filesystem switch를 raw leaf에서 재구성한다.

## 6. 변경 순서

1. v2 schemas와 strict shared evidence primitive를 추가하고 v1 rejection policy를 문서화한다.
2. reconstructed baseline builder/checker와 adversarial tests를 추가한다.
3. typed sampling selection, private-FD generation audit, 그리고 source-owned
   shutdown v2 artifact/marker producer를 Rust source에 추가한다.
4. v4 serial capture-session binder/schema와 hostile fixture tests를 추가한다.
5. config endpoint process bridge, initial one-scenario lifecycle runner, same-process
   v4/shutdown receipt closure와 hostile/static tests를 추가한다. 이 구현만으로는
   GPU capture나 qualification을 실행하지 않는다.
6. native fallback leaf와 rollback raw runner/binder를 추가한다.
7. soak/rollback v2 semantic checker를 추가하고 outer RC3 finalizer를 v2-only로 바꾼다.
8. 이 P1 source가 clean commit으로 고정된 뒤에만 new candidate를 freeze하고 GPU qualification capture를 시작한다.

## 7. 완료 조건

- v1 receipt, self-authored fallback string, missing strict-open flags, self-authored worker/model IDs는 final C02 input에서 fail closed한다.
- v4 serial soak provenance는 raw descriptor hash뿐 아니라 capture session의
  incomplete-marker closure, contract/request/audit linkage와 config endpoint 및
  every scenario의 same candidate/config/PID/start-tick/listener inode/GPU tuple을
  replay한다. v3 input은 serial-capture path에서 허용하지 않는다.
- max-performance-exact GPU-greedy ineligible case는 Rust-written audit v2와 public generation bytes가 one-to-one으로 bind된다.
- shutdown v2 artifact와 nonhidden completion marker는 same PID/start-tick tuple과
  exact final-metrics bytes를 bind하며, v1 artifact/hidden marker는 fail closed한다.
- initial lifecycle receipt는 same-process successful-v4 edge에서만 shutdown
  artifact/marker를 다시 bind하고 `completed`/`not-run` raw status만 낸다. CPU/static
  tests 또는 receipt 존재만으로 GPU capture, freeze, fallback/rollback semantic result를
  주장할 수 없다.
- reconstructed baseline manifest는 previous stable artifact라고 주장하지 않으며, A/B reconstruction equality와 artifact provenance를 검증한다.
- C02 finalizer가 soak-v2/rollback-v2만 수용하고 resulting final report에서 operational rollback과 historical-stable rollback status를 분리한다.
- 이 단계는 candidate freeze, Gate E pass, C02 pass, vLLM win을 주장하지 않는다.
