# C02-P1 — Provenance v2와 Reconstructed Rollback Baseline

**상태:** In progress — v1 provenance 결함을 확인했고, candidate freeze 전에 v2 contract와 source-owned raw producer를 구현한다. initial C02 lifecycle supervisor/raw receipt, native sampling fallback source leaf/marker, source pair raw capture v2, fresh lifecycle-evidence preparer v5, fixed lifecycle bind-request writer v5, private held-lock terminal binder core와 fixed-name raw compositor v5, authenticated native-fallback raw-v5 runner, reconstructed RC2 전용 rollback raw-provenance v3 verifier/schema와 path-only binder, RC2-compatible raw phase collector, immutable artifact snapshot→separate runtime-copy preparation, isolated atomic-switch producer, held-FD artifact-exchange transaction, same-invocation rollback terminal-provenance v4, reviewed RC2 source/OCI input closures, cross-root baseline content bridge v1, A/B reproducibility build-input closure, static reconstructed-runtime assembly recipe contract, 그리고 arm별 raw runtime assembly/capture receipt verifier는 구현됐다. 현재 검증 범위는 CPU/static hostile-path뿐이며, authenticated remote rollback runner, actual Docker/GPU capture·candidate freeze·lifecycle-v5 receipt·semantic qualification은 아직 수행하지 않았다.
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

그 materializer보다 앞선 좁은 검증 경계로 `ci/release/prepare_reconstructed_prior_baseline_content_bridge_v1.py`가 있다. 이는 새 빈 mode-0700 bridge root, v2 baseline root와 canonical relative manifest, RC2 source-input root와 **매 replay마다 다시 제공하는** independently reviewed source SHA-256, arm A/B OCI-input roots를 받는다. 다섯 root를 no-follow로 열고 bridge 전체 동안 held FD만 사용한다. source v1, v2 baseline, OCI v1 A/B verifier를 모두 다시 실행한 뒤, baseline source의 tag object/tag target/archive와 source-input closure의 세 leaf를 `(sha256, byte_length)`로 bind하고, 각 arm의 raw Docker image inspect/OCI archive/layout/manifest/image ID를 matching OCI closure에 bind한다. OCI `index.json`/`config.json`은 v2 peer가 없는 OCI-internal replay leaf로만 receipt에 보존한다. normalized path가 겹치거나 root inode가 alias인 role, arm swap, mutable/extra bridge leaf를 거절하며 입력 root의 절대 경로는 receipt에 직렬화하지 않는다.

결과 receipt/schema (`riley.reconstructed-prior-baseline-content-bridge.v1`, `benchmarks/release/candidates/reconstructed-prior-baseline-content-bridge-v1.schema.json`)는 `bound/not-run`, `authority: cross-root-content-bridge-only` 한 장뿐이다. 그 안에서만 `oci_archive_content_binding: validated-via-runtime-oci-inputs-v1`를 말할 수 있고, 기존 v2 report의 source/OCI binding은 둘 다 계속 `not-validated`로 고정 기록한다. 이는 source→runtime image, bundle→runtime image, runtime build invocation, runtime-capture independence, rollback, freeze, qualification을 **전혀** 증명하지 않으며 모두 `not-established`/`not-run`으로 보존한다. bridge는 freeze admission·rollback·qualification input이 아니고 Docker·build·GPU·service·network를 실행하지 않는다. 실제 materializer에는 arm별 A/B binary/bundle을 runtime image로 조립·capture한 same-invocation raw receipt가 먼저 필요하며, v2 closed schema를 self-authoring하거나 OCI label만으로 그 인과관계를 승격하면 안 된다.

그 same-invocation receipt의 바로 앞에는 `ci/release/prepare_reconstructed_repro_build_inputs_v1.py`가 만든 A/B 입력 closure가 온다. caller는 RC2 source-input root와 **매 replay마다 다시 제공하는** reviewer SHA-256, reviewer-pinned immutable builder image ID, 이미 캡처된 `repro-build-a.tar`/`repro-build-b.tar`를 제공한다. tool은 외부 tar를 새 private root에 create-only snapshot한 뒤, held descriptor에서 만든 별도 private checker copy에만 기존 PR16 parser를 적용한다. raw A/B tar의 closed inventory, source tar, builder image, pre/post container receipts, offline/network-none command, completion receipt, binary/profile/bundle/native equality와 independent container/workspace를 다시 검증하고 각 arm의 raw tar·`build.json`·`riley`·`riley.tar.gz`만 저장한다. receipt/schema (`riley.reconstructed-repro-build-inputs.v1`, `benchmarks/release/candidates/reconstructed-repro-build-inputs-v1.schema.json`)는 `prepared/not-run` 입력 closure이며 A/B binary/bundle/source-archive equality와 PR16 execution independence까지만 `validated`라고 한다. runtime-image assembly/capture, source/bundle→runtime-image, OCI content, rollback, freeze, qualification은 모두 `not-established`/`not-run`이다. 이 tool도 Docker·build·GPU·service·network를 실행하지 않는다.

그 closure를 실제 image build에 넘기기 전에는 `ci/release/ReconstructedRuntimeAssembly.Dockerfile`과 `verify_reconstructed_runtime_assembly_dockerfile.py`가 source-free assembly recipe를 고정한다. recipe의 future canonical context는 exact `Dockerfile`, `input/riley`, `input/riley.tar.gz` 세 leaf뿐이다. 두 stage는 같은 reviewed CUDA runtime index digest와 explicit `linux/amd64` platform을 사용한다. 첫 stage는 caller-supplied archive를 non-root로 unpack하고 A/B reconstruction ID와 closed provenance build arguments의 문법, raw binary/bundle SHA-256, bundle `SHA256SUMS`, no-link/no-special extracted tree, 그리고 raw binary와 bundle의 `bin/riley` byte equality를 확인한다. final stage는 fresh base에서 그 verified `/opt/riley/`만 numeric non-root ownership으로 복사하며 controlled system PATH에서 input ELF를 실행하지 않고 set-ID/hard-link/special file과 bundle-local Python/toolchain executable을 다시 거절한다. static verifier는 normalized instruction SHA-256와 exact stage/inventory를 고정하여 `COPY .`, source checkout, Cargo/Rust/CUDA build stage, package install, `ADD`, build mount/secret/SSH frontend drift를 수용하지 않는다. 이것은 reviewed assembly **tool**의 CPU-only source contract일 뿐 Docker build, bundle→image/source→image binding, OCI content, runtime capture A/B independence, rollback, freeze 또는 qualification을 주장하지 않는다.

그 다음의 source-only post-capture boundary로 `ci/release/prepare_reconstructed_runtime_assembly_capture_v1.py`와 `benchmarks/release/candidates/reconstructed-runtime-assembly-capture-v1.schema.json`가 한 arm의 이미 존재하는 raw USTAR capture를 새 private root에 create-only snapshot한다. 매 replay는 reviewer SHA와 builder image ID를 다시 받아 RC2 source v1 → PR16 A/B reproducibility v1 → matching-arm OCI v1을 held FD로 재실행하고 root inode alias/경로 overlap을 거절한다. capture는 fixed `SHA256SUMS`, canonical three-leaf context, exact source-free `docker build` logical argv와 seven provenance args, iidfile, raw image inspect/OCI export binding, `docker create` 직후의 `created`/not-started/no-mount/network-none inspect, 그리고 selected bundle의 strip-root `/opt/riley` file tree를 가진다. runtime tree는 final numeric non-root `65532:65532` ownership도 요구한다. loader/interpreter injection environment, image/container volume, bind/tmpfs/device/capability/security option도 거절한다. USTAR raw preflight는 PAX/GNU/sparse/link/device/FIFO, traversal, duplicate, noncanonical metadata와 nonzero trailer를 tar parser 이전에 차단한다. outer snapshot은 fixed member ceilings, headers, end marker, one 20-block zero trailer만 합산한 약 13.06 GiB max를 사용하여 sparse/zero-tail input이 evidence volume을 소모하지 못하게 한다. POSIX USTAR의 single-member size field 한계 때문에 이 v1 capture는 embedded `oci-image-layout.tar`가 8 GiB−1 이하인 OCI v1 closure만 수용한다; 더 큰 OCI closure는 PAX로 우회하지 않고 future directory-snapshot contract로 분리한다. OCI archive와 raw image inspect는 기존 OCI v1 closure bytes와 exact `(sha256, byte_length)` equality를 요구하고, bundle은 held repro descriptor에서 만든 private checker copy에서만 replay한다. receipt의 `bound/not-run`은 raw record의 구조적 cross-check만 의미한다. raw record는 Docker build/container-copy가 실제로 수행됐다는 독립 증거가 아니므로 runtime build execution, container filesystem capture provenance, source/bundle→runtime image, A/B capture independence, image equality, runtime/service/GPU execution, rollback, freeze, historical distribution, candidate qualification은 모두 계속 `not-established`/`not-run`이다. 이 preparer는 Docker, compiler, GPU, service, network를 호출하지 않으며 same-invocation raw runner는 후속 별도 단계다.

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
rollback raw runner, semantic checker/finalizer, clean candidate freeze와 실제 GPU
capture는 이후 versioned work로 남는다.

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

`check_soak_v2_receipt_v2.py`는 summary counter나 free-form backend event를 믿지 않고 위 leaf에서 identity, interval order, request/audit binding, metrics monotonicity, actual typed sampling selection을 재구성한다.

### Rollback v2

`run_remote_rc3_rollback_capture.sh`와 `bind_raw_rc3_rollback_capture.py`는 candidate와 reconstructed baseline 각각의 PID/start tick, listener inode, health/generation/audit raw bytes, candidate shutdown artifact+marker, atomic rename 전후 device/inode/stat evidence를 보존한다. label 문자열이나 `atomic-rename` declaration은 증거가 아니다.

`check_rc3_rollback_receipt_v2.py`는 frozen candidate가 pin한 baseline manifest를 replay하고, candidate drain/zero ownership, replacement process/socket, generation response, shutdown marker, filesystem switch를 raw leaf에서 재구성한다.

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
runner는 해당 terminal session, 각각의 `capture-incomplete.json` 부재, 그리고
session-bound completion receipt pair를 replay한 뒤에만 raw path를 bind request에 넣어야 한다.
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
   baseline rollback v3 raw verifier/schema를 추가하고, 별도 raw runner/binder를 추가한다.
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
