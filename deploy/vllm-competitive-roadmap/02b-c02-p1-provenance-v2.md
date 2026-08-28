# C02-P1 — Provenance v2와 Reconstructed Rollback Baseline

**상태:** In progress — v1 provenance 결함을 확인했고, candidate freeze 전에 v2 contract와 source-owned raw producer를 구현한다.
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
- raw descriptor 하나가 다른 leaf/path에 재사용되거나, symlink/hardlink alias, parent swap, incomplete marker, PID start-tick/socket/GPU UUID mismatch가 있으면 semantic replay는 실패한다.

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

## 5. v2 raw producer와 semantic replay

### Soak v2

`run_remote_c02_soak_v2.sh`와 `bind_raw_c02_soak_v2.py`는 GPU UUID/used-memory preflight, exclusive lock, frozen arm의 `env -i`, 새 evidence root를 강제한다. 각 scenario는 다음 raw leaf를 descriptor로 남긴다.

- raw `/v1/config` 및 startup artifact bytes
- raw public request/response or SSE bytes와 matching generation audit v2
- same PID/start-tick pre/post C02 metrics, `/proc` status/stat/fd/socket and `/proc/net/tcp`
- selected GPU UUID/compute-app raw output
- scenario lifecycle/process exit and create-only completion markers

`check_soak_v2_receipt_v2.py`는 summary counter나 free-form backend event를 믿지 않고 위 leaf에서 identity, interval order, request/audit binding, metrics monotonicity, actual typed sampling selection을 재구성한다.

### Rollback v2

`run_remote_rc3_rollback_capture.sh`와 `bind_raw_rc3_rollback_capture.py`는 candidate와 reconstructed baseline 각각의 PID/start tick, listener inode, health/generation/audit raw bytes, candidate shutdown artifact+marker, atomic rename 전후 device/inode/stat evidence를 보존한다. label 문자열이나 `atomic-rename` declaration은 증거가 아니다.

`check_rc3_rollback_receipt_v2.py`는 frozen candidate가 pin한 baseline manifest를 replay하고, candidate drain/zero ownership, replacement process/socket, generation response, shutdown marker, filesystem switch를 raw leaf에서 재구성한다.

## 6. 변경 순서

1. v2 schemas와 strict shared evidence primitive를 추가하고 v1 rejection policy를 문서화한다.
2. reconstructed baseline builder/checker와 adversarial tests를 추가한다.
3. typed sampling selection, private-FD generation audit, 그리고 source-owned
   shutdown v2 artifact/marker producer를 Rust source에 추가한다.
4. soak/rollback raw runners·binders와 hostile-evidence tests를 추가한다.
5. soak/rollback v2 semantic checker를 추가하고 outer RC3 finalizer를 v2-only로 바꾼다.
6. 이 P1 source가 clean commit으로 고정된 뒤에만 new candidate를 freeze하고 GPU qualification capture를 시작한다.

## 7. 완료 조건

- v1 receipt, self-authored fallback string, missing strict-open flags, self-authored worker/model IDs는 final C02 input에서 fail closed한다.
- v2 report는 raw descriptor hash뿐 아니라 same candidate/config/PID/start-tick/socket/GPU tuple을 replay한다.
- max-performance-exact GPU-greedy ineligible case는 Rust-written audit v2와 public generation bytes가 one-to-one으로 bind된다.
- shutdown v2 artifact와 nonhidden completion marker는 same PID/start-tick tuple과
  exact final-metrics bytes를 bind하며, v1 artifact/hidden marker는 fail closed한다.
- reconstructed baseline manifest는 previous stable artifact라고 주장하지 않으며, A/B reconstruction equality와 artifact provenance를 검증한다.
- C02 finalizer가 soak-v2/rollback-v2만 수용하고 resulting final report에서 operational rollback과 historical-stable rollback status를 분리한다.
- 이 단계는 candidate freeze, Gate E pass, C02 pass, vLLM win을 주장하지 않는다.
