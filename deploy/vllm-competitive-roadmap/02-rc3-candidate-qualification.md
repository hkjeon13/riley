# C02 — RC3 Candidate-bound Qualification

**상태:** In progress — C02-P0 two-profile와 Qwen v2 `riley-0.1.0-rc99` raw smoke, fixed-routing 및 CPU-only fault raw producer의 source/release-ELF 검증은 완료했다. C02-P1 initial one-scenario lifecycle supervisor/receipt, create-only frozen-candidate **input-identity** manifest/FD-safe replay, four-gate closed input-inventory replay도 CPU/static hostile-path 범위로 구현됐지만, actual candidate freeze, Gate E semantic replay, actual GPU raw capture/semantic replay, qualification decision은 미완료다.<br>
**의미 등급:** `reference` + 기존 승인 `E0` 검증  
**한 가지 목적:** 최신 단일 Riley revision과 exact release binary를 대상으로 Gate E, Python-free, correctness, performance regression, soak를 모두 다시 실행해 정식 candidate를 판정한다.

[이전: C02-P1](02b-c02-p1-provenance-v2.md) | [목차](README.md) | [다음: C03](03-scheduler-output-routing-fuzz.md)

## 1. 배경

`0.1.0-rc2`는 mixed prefill/decode output routing 결함을 수정했지만, 최종 revision과 binary에 결합된 전체 soak를 다시 수행하지 않은 owner-approved prerelease다. 이후 decode fast path도 추가됐으므로 이전 evidence만으로 최신 main의 안정성을 주장할 수 없다.

C01, C02-P0, C02-P1이 merge된 뒤의 C02 pre-freeze mechanism 단계에서는 raw producer와
provenance verifier를 구현·검토할 수 있다. 그러나 candidate가 freeze된 뒤의 actual
qualification execution 동안에는 production 동작을 바꾸지 않는다. gate 실패가 코드 결함을
발견하면 qualification을 중단하고 별도 corrective PR을 만든 뒤 새 candidate SHA로 처음부터
재실행한다.

그 production prerequisite가 C02-P0다. source-level endpoint/artifact 구현과 비-device
검증만으로는 충분하지 않다. RTX 4090에서 GPU used-memory `<=256MiB` preflight 뒤
`riley-0.1.0-rc99`로 실행한 two-profile container smoke는 실제 cold-prepare와 raw
endpoint/artifact binding을 확인했지만, nonqualifying mechanism evidence일 뿐이다. 새
clean committed candidate를 freeze한 뒤 같은 candidate/binary/model/Gate E evidence에
bind된 two-profile raw capture와 semantic replay가 끝나기 전에는 actual C02
qualification을 시작할 수 없다.

### Pre-freeze mechanism capture 상태

Qwen v2 raw producer는 source-controlled code/English/Korean literal prompt 각각을
non-stream과 stream으로 capture하고, raw public request/header/body/SSE와 matching
create-only `riley.c02-generation-audit.v1` record를 별도 SHA descriptor로 남긴다. 한국어의
마지막 empty decoded committed token은 audit에 보존하되 public SSE delta로 만들지 않는다.
RTX 4090에서 `riley-0.1.0-rc99` development image로 이 raw mechanism을 실행해 세 case를
검증했지만, raw receipt는 명시적으로 `qualification_status: not-run`이며 v2 semantic receipt
또는 C02 verdict가 아니다.

Routing producer는 `riley c02-routing-capture --format canonical-json` feature-gated fixed
five-case scheduler fixture와 GPU 없는 read-only/network-none container runner다. development
release ELF의 `case_id`+`trace` projection이 reviewed corpus와 일치함은 확인했지만, raw runner는
frozen candidate manifest와 replayed Gate E report를 필수로 요구한다. 따라서 현재 source-level
producer 검증은 candidate-bound routing evidence 또는 C03 시작 조건을 충족하지 않는다.

Fault raw producer는 `riley c02-fault-capture --format canonical-json`의 feature-gated 네 case
fixture를 read-only, network-none, GPU-free container에서 실행하고 parent capture, child marker
log, test ELF를 create-only external evidence root에 보존한다. 현재 development capture의 raw
receipt는 `riley.c02-fault-raw-capture.v1` 및 `qualification_status: not-run`이다. 따라서 이는
injectable-synthetic producer mechanism 확인일 뿐, frozen candidate binding, Gate E semantic replay,
`riley.fault-extension-receipt.v3` semantic input/report, C02 verdict 또는 C03 시작 조건이
아니다. future v3 replay는 test ELF도 frozen source archive/revision, frozen reproducible
build/execution image, exact feature build command에 결박한다. v2 fault-extension receipt/schema는
historical only다.

### Historical v1 observation/shutdown proposal — superseded

이전 문서의 `riley.c02-capture-metrics.v1`,
`riley.c02-shutdown-quiescence.v1`, 그리고 hidden
`.<basename>.complete` shutdown marker 제안은 historical-only다. 그것은 candidate
qualification input이나 P1 v2 producer contract가 아니며, v2 binder/finalizer는 이를
up-convert하거나 수용하지 않는다.

대신 C02-P1은 complete C02 runtime configuration과 loopback
`--c02-audit-dir`가 함께 켜진 server에 prompt-free
`GET /v1/c02/metrics` **v2** surface를 구현해야 한다. 이 endpoint는
compatibility-closed public `/metrics`를 바꾸지 않는다. raw response는
`riley.c02-capture-metrics.v2`이고 정확히 `request_states`, `kv_blocks`,
`allocation`, `quiescence`의 source-owned facts를 담는다. scheduler/native
allocation snapshot을 얻지 못하거나 degraded이면 `503`으로 fail-closed한다. raw
schema/binder는 field type만 확인하고 threshold나 qualification result를 만들지 않는다.

`ci/release/check_rc3_prefreeze.py`는 freeze를 쓰기 전 clean source snapshot만
fail-closed로 확인하는 별도 도구로 구현됐다. `HEAD` 별칭이 아닌 full SHA와 proposed RC ID를 받고,
현재 HEAD 일치, tracked/untracked를 모두 포함한 clean checkout 및 ordinary-visible Git
index(assume-unchanged·skip-worktree 거부), release metadata/default, workspace version,
`Cargo.lock`·extension registry의 no-follow hash를 확인한다. 성공 JSON은
`scope: source-pre-freeze-only`, `candidate_status: not-frozen`,
`qualification_status: not-run`만 기록한다. archive/image/ELF/Gate E/raw receipt/freeze SHA를
생성하거나 검증했다고 주장하지 않으며, actual frozen candidate 또는 C02 qualification의 대체물이
아니다. 이 checker는 fixture와 hostile-path 범위만 통과했으며 actual candidate freeze는 아직
수행하지 않았다. reviewed server-defaults pin은 `21f445f4870a140346509144c36c7294f2f677f3`의
`crates/riley-server/src/main.rs` SHA-256
`47990249835eed190ee73521ede239841eae0eb73f20e71577258790f1734e4b`로 갱신됐다.
이전 `1195cf20` pin과 대조해 기존 serve 기본값(루프백 bind, device 0, 8/64 scheduler
capacity, 512 token budget/prefill, iteration-batch, fixed-max, synchronous, CPU,
canonical-v1)은 변하지 않았고, 추가된 C02 runtime-config/audit/shutdown/native-fallback
경로는 모두 명시 opt-in·fail-closed임을 검토했다. 이후 이 source가 바뀌면 별도 reviewed
release-contract PR이 새 exact pin을 승인해야 하며, pre-freeze checker는 hash를 자동 발견,
허용 또는 갱신하지 않는다. 이 갱신도 candidate freeze나 Gate E/semantic qualification을
수행하거나 승인하지 않는다. 같은 reviewed literal은 active
`ci/release/release_common.py` release preflight/bundle/manifest contract에도 별도로
적용된다. 두 checker를 import로 결합하지 않으며, Python 3.10 held-FD pre-freeze boundary와
build-oriented release helper는 각각의 고정 pin이 drift에서 fail-closed해야 한다.

check_rc3_freeze_input_admission.py는 이 source-only pre-freeze report를 actual freeze로
바꾸지 않는 별도 structural admission이다. 이것은 python3 -B 또는
PYTHONDONTWRITEBYTECODE=1을 요구하고, source checkout 밖의 exact-0700 evidence root를
held-FD shared lock으로 열어 canonical riley.rc3-freeze-input-request.v1 하나와 external
archive, release ELF, container inspect, toolchain probe, model/tokenizer/config/weights,
two-profile args/env, correctness descriptors, reconstructed baseline manifest를 재해시한다.
external Cargo.lock과 extension registry는 source pre-freeze report가 방금 읽은 bytes와
SHA-256 및 byte length 모두 같아야 한다. workspace manifests와 reviewed server-defaults
source는 request가 대체할 수 없다.

이 admission의 reconstructed baseline 처리는 canonical manifest descriptor와
binary-bound reconstructed-v2 vocabulary의 structural check까지만 수행한다. legacy
binary-unbound reconstructed-v1 manifest는 explicit rejection이다. nested baseline leaves,
Gate E, semantic receipt 또는 qualification을 replay하지 않는다. runner-owned C02 launch
identity와 freeze/Gate E/configuration self-reference는 args/env에서 거부한다. output은
riley.rc3-freeze-input-admission.v1의 bound/not-frozen/not-run,
freeze-input-structural-only뿐이며 freeze writer, marker, candidate result, Gate E report,
semantic result를 만들지 않는다. 따라서 이 output은 actual candidate freeze 또는 outer
finalizer의 semantic input이 아니며 후속 단계는 original request와 raw leaves를 다시
replay해야 한다.

baseline manifest가 어휘상 valid하더라도 arbitrary historical 또는 future tag를 candidate
baseline으로 바꿔치기할 수는 없다. 이 admission은 candidate와 같은 semver의 바로 앞 RC tag
및 그 reconstructed baseline ID를 요구하고 output에 그 structural relationship을 기록한다.
이는 canonical manifest의 declared tag-name/ID 비교일 뿐 Git tag object/target, source
archive, nested baseline leaves의 history proof를 replay하거나 주장하지 않는다.

hostile request가 rehash I/O를 무제한 증폭하지 않도록 이 checker는 최대 8,192개 external
descriptor와 총 1 TiB declared byte budget을 구조적 admission 한계로 둔다. 이 request-only
cap에는 nested baseline raw graph의 full replay가 포함되지 않으며, candidate freeze, evidence
completeness 또는 semantic success 판정도 아니다.

`ci/release/capture_c02_observations_v2.py`는 이미 loopback C02 audit server로 실행 중인
process에 attach하는 raw-only sampler다. `GET /v1/c02/metrics` 원본 bytes와 같은 PID의
`/proc` RSS/start-tick, 지정 GPU UUID/compute-apps 원본을 새 external evidence directory에
create-only로 보존한다. endpoint가 `200`/정확한 C02 schema가 아니거나 PID start tick, GPU UUID,
누적 terminal counter가 역행하면 fail-closed하며 public `/metrics`로 fallback하지 않는다. session은 명시적으로
`qualification_status: not-run`이고 freeze, candidate ID, semantic receipt, soak-v2 trace 또는
C02 verdict를 만들지 않는다.

종료 직후 raw ownership을 별도 보존할 경우 source는 complete C02 identity,
loopback `--c02-audit-dir`, audit root의 direct child인
`--c02-shutdown-artifact PATH`를 함께 요구해야 한다. 이 flag는 일반
`RILEY_SHUTDOWN_METRICS_PATH`와 mutual exclusion이고, server가 listener/HTTP worker를
join하고 backend close가 성공한 뒤에만 writer에 authoritative final snapshot을 넘긴다.

성공 artifact는 v2 schema의 정확히 일곱 field만 가진다:
`schema_version: "riley.c02-shutdown-quiescence.v2"`,
`capture_status: "captured"`, `qualification_status: "not-run"`,
`server_pid`, `server_start_ticks`, `worker_ready: false`, `final_metrics`.
`final_metrics`는 `riley.c02-capture-metrics.v2` exact nested object이며
candidate ID, freeze SHA, Gate E, semantic receipt, `passed`, rollback verdict를
포함할 수 없다. source writer는 worker accepting/scheduler ownership/KV
reservation or allocation/native allocation 등 source-owned shutdown facts가
남으면 fail-closed하고 artifact/marker를 만들지 않는다. raw binder가 이 field들을
threshold로 판정하지 않는 것은 이후 semantic replay가 담당하기 때문이다.

audit root는 euid 소유·정확히 `0700`이어야 하며, `/`부터 root까지의 각 ancestor는 root
또는 euid 소유이고 peer-writable이면 sticky bit이어야 한다. writer는 startup에 열린
trusted root FD를 유지해 direct child를 `O_CREAT|O_EXCL|O_NOFOLLOW`로 create-only write,
file `fsync`, 같은 root의 nonhidden `<artifact_filename>.complete` marker write,
marker/root-directory `fsync` 순서로 commit한다. marker는 정확히
`schema_version: "riley.c02-shutdown-quiescence-complete.v2"`,
`artifact_filename`, `artifact_sha256` 세 field이며 exact artifact bytes hash를 bind한다.
final file만 있거나 matching v2 marker/hash가 다르면 incomplete다. existing file,
symlink, hard link, sibling/subdirectory path, root parent swap은 replacement/cleanup
없이 reject한다.

### C02-P1 provenance v2 closure (candidate freeze 전 필수)

[`C02-P1 provenance v2와 reconstructed rollback baseline`](02b-c02-p1-provenance-v2.md)이 이 선행 gate의 규범적 계약이다. v1 soak/rollback receipt의 self-authored interval, worker/model label, `atomic-rename` 선언, fallback 문자열은 historical-only이며 C02 qualification input으로 수용하지 않는다. C02는 P1의 v2 raw descriptor replay와 source-owned selection audit, reconstructed-tag baseline manifest를 사용한다.

v2 observation sampler와 shutdown artifact producer는 원시 관측 surface일 뿐이며,
`riley.soak-v2-receipt.v1` 또는 `riley.rc3-rollback-receipt.v1`을 실제로 생성하는
producer는 아직 없다. initial v4 raw lifecycle/receipt는 이 historical receipt를
up-convert하지 않으며, 여전히 semantic receipt 또는 C02 verdict가 아니다. 따라서 C02
candidate를 freeze하기 전, C01, C02-P0, C02-P1이 각각 clean merge된 source에서 다음
mechanism과 adversarial test를 닫아야 한다.

- initial `run_remote_c02_soak_v2.sh`는 authenticated no-follow host lock과 `env -i`,
  새 private external evidence root, host binary/model의 launch 전·후 safe hash/input
  revalidation을 강제하는 host-binary raw supervisor로 구현됐다. 한 invocation은 frozen
  contract의 한 serial scenario, 그 직후 한 C02 observation, 하나의 v4 manifest만
  허용한다. `capture_c02_config_endpoint_observation_v1.py`는 loopback server의
  canonical `/v1/config` response를 pre/post PID/start-tick, listener inode, GPU
  index/UUID raw leaves에 결박하고, v4 binder는 derived tuple이 scenario observation과
  같지 않으면 fail closed한다. same-process receipt finalizer만 successful v4 bind 뒤
  source-owned shutdown-v2 artifact/marker를 replay하여 `completed`/
  `qualification_status: not-run` receipt를 publish한다. 현재 이 경로는 CPU/static
  hostile-path 검증만 마쳤으며, actual GPU capture, candidate freeze, Gate E semantic replay,
  semantic receipt 또는 C02 verdict가 아니다.
- v3 path-only rollback binder와 RC2-compatible raw phase collector는 landed했다.
  binder는 same held private root FD에서 descriptor/target과 reviewed RC2 annotated tag object를 포함한 reconstructed baseline을
  replay하고 nonterminal `captured/not-run` manifest만 publish한다. rollback binary는
  reconstructed baseline v2의 independently captured A/B server-binary descriptor와
  SHA-256·byte length가 같아야 하며, bundle/Docker image-ID binding도 함께 유지한다.
  binary-unbound v1 baseline은 phase evidence 전에 거부된다. phase collector는
  existing baseline root의 create-only child에 loopback health/optional generation 및
  process/socket/GPU raw leaves를 남긴다. 아직 authenticated remote rollback runner는
  없으므로 candidate source audit/shutdown join, atomic rename, terminal-session replay,
  실제 deployment rollback을 실행하거나 주장하지 않는다.
- fixed artifact preparation raw producer, isolated atomic-switch raw producer, held-FD artifact-exchange transaction, 그리고
  same-invocation rollback terminal-provenance v4는 landed했다. pre-existing complete
  reconstructed RC2 root를 new root에 import하지 않고 held-FD replay로 admit한 뒤,
  future v3 binding의 freeze/base-report/stable-default configuration 세 opaque external
  input을 fixed private snapshot으로 고정하는 `prepare_rc3_rollback_evidence_v1.py`도
  landed했다. 이것은 `captured/not-run`, `raw-static-preparation-only` raw evidence일
  뿐 actual freeze/config observation/rollback authority가 아니다. artifact preparation은 여섯 absolute host artifact를 no-follow streaming으로
  immutable 0600 `rollback-v3-artifacts/` leaf에 snapshot하고, candidate/rollback
  binary snapshot에서만 distinct 0700 `rollback-v3-switch/active`와
  `rollback-staged`를 materialize한다. fixed create-only session은 두 runtime inode를
  snapshot hash/length에 bind한다. terminal은 단순 marker 부재가 아니라 incomplete
  marker 부재와 `session.json`의 hash/length에 bind된 mode-0600 two-link
  `capture-complete.intent`/`capture-complete.json` receipt pair를 verifier가 함께
  replay할 때만 성립한다; runtime leaf 자체는 artifact-map descriptor가 아니다. atomic producer는 evidence root 내부에
  이미 staged된 private runtime files만 same-directory
  `renameat2(RENAME_EXCHANGE)`로 교환하고 v3의 다섯 opaque switch leaf를 남긴다.
  실제 deployment path, `mv`/ordinary rename fallback, rollback success verdict는
  범위 밖이다. `capture_rc3_rollback_atomic_transaction_v1.py`는 one exclusive
  root/switch FD 아래 pre-switch preparation replay → exchange → terminal atomic replay
  → post-switch preparation replay를 수행하고, transaction session에서 preparation/atomic
  session descriptor, pre/post runtime SHA-256·identity layout, 그리고 candidate/rollback
  runtime-to-inode 방향을 cross-bind한다. atomic capture도 exchange 직전과 직후의
  private runtime bytes hash를 stat leaf에 기록해 same-inode/same-size in-place mutation을
  terminal evidence로 승격하지 않는다. 이
  raw subtransaction도 `captured/not-run`이며 phase/source/config/baseline closure를
  연결하거나 deployment를 조작하지 않는다. full reconstructed baseline replay와
  preparation/phase/switch를 모두 요구하는 authenticated host runner는 여전히 없다.
  completion hardlink의 post-link parent-directory `fsync`가 오류를 반환하면 helper는
  `ambiguous-terminal-publication`으로 실패하고 성공 값을 반환하지 않는다. 그때 남을 수
  있는 pair는 structural raw replay 대상일 뿐 failed invocation의 성공/terminal authority가
  아니다. 후속 authenticated runner는 같은 invocation과 held lock에서 정상 반환한 success
  branch만 소비해야 하며, 새 verifier가 ambiguous pair를 읽었다는 이유만으로 operation을
  재개하면 안 된다.
  v3 manifest는 session/completion-receipt closure를 bind하지 않으므로, v4는 v3 artifact/atomic descriptor maps를 transaction child sessions와 exact path/hash/length로 join한다. v4의 유일한 public producer는 새 preparation을 root EX 아래에서 시작하고, preparation normal-return closure → switch EX → transaction normal-return closure → v3/v4 publication을 같은 stack에 중첩한다. serialized continuation object나 기존 pair를 reopen하는 publisher는 없고, preparation·transaction·v4 어느 completion pair의 ambiguous 오류도 다음 closure를 호출하지 않는다. fresh replayer가 ambiguous pair를 읽어도 raw structural `bound/not-run` diagnostic일 뿐 operational terminal authority가 아니다. full phase/source/config/lifecycle semantic closure는 계속 future runner의 책임이다.
  preparation verifier의 `post-switch` mode는 content/inode mapping 재생일 뿐
  `renameat2` success evidence가 아니며 atomic switch session을 별도로 replay해야 한다.
  transaction helper는 그 artifact-exchange 범위에서 동일 held-FD exclusive lock을
  유지한다. 그러나 phase/source/config session과 actual process lifecycle는 이 범위에
  포함되지 않으며 후속 authenticated runner가 닫아야 한다.
- fixed `write_rc3_rollback_finalizer_receipt_v1.py`도 landed했다. 이 private helper는
  caller-held root EX/switch EX 안에서 v3/v4 finalizer를 직접 호출한 **정상 반환**만
  in-memory typed closure로 유지하고, fixed request·static preparation·candidate/source
  complete inventory·candidate/rollback phase·transaction·v3/v4 descriptor를 다시
  held-FD replay해 fixed receipt/paired marker를 낸다. status는 `completed/not-run`,
  authority는 `raw-finalizer-normal-return-only`뿐이다. v4나 receipt marker의 post-link
  fsync ambiguity는 successful return/receipt를 만들지 않고, visible receipt pair는
  standalone rollback/lifecycle authority 또는 semantic checker input이 될 수 없다.
  authenticated runner의 same-stack normal-return branch만 이 helper의 반환을 소비할 수
  있으며, 이 코드 역시 service/GPU/deployment를 실행하지 않는다.
- native fallback leaf, soak/rollback semantic receipt contract와 qualification checker는
  후속 work다. authenticated rollback raw runner와 private held-FD
  `replay_rc3_rollback_operational_semantics_v1.py`는 landed했으며, 후자는 original
  raw leaves와 v4 raw-manifest cross-binding만 다시 읽어 in-memory
  `passed/not-run`, `raw-operational-semantics-only` diagnostic을 낸다. v4 completion
  pair/finalizer receipt/precheck를 semantic input으로 승격하지 않으며 freeze, Gate E,
  deployment 또는 qualification authority가 없다. caller-held FD 자체는 prior finalizer
  normal-return lineage를 증명하지 않으므로 same-stack capability나 receipt를 대체할 수
  없다. 이후 semantic receipt checker도 위
  descriptor를 no-follow hash replay로 필수 검증하고, 기존 v1 public schema를
  불명확하게 약화하지 않은 호환 불가 새 version으로 추가해야 한다.

이 단계는 actual GPU raw receipt나 semantic report, candidate freeze, qualification decision을
미리 만들거나 backfill하지 않는다. 구현과 테스트가 끝난 clean C02 source revision만
후속 freeze의 입력이 될 수 있다.

## 2. Candidate freeze

qualification 시작 전에 다음을 create-only로 고정한다.

```text
full Git commit SHA
clean source archive SHA-256
release ELF SHA-256
container image ID/digest
Cargo.lock SHA-256
CUDA C ABI version
Rust, nvcc, driver, CUDA runtime/toolkit, cuBLAS versions
model/tokenizer/config/weights hashes
exact CLI and environment configuration
extension registry bytes/hash
correctness contract/report hashes
reconstructed-tag rollback baseline manifest hash
```

`main`이 이후 이동해도 candidate는 바뀌지 않는다. 하나의 gate라도 다른 revision/binary를 사용하면 final report는 `incomparable`이다.

create-only freeze writer를 호출하기 전에는 위 admission으로 proposed external input envelope를
structurally bind할 수 있다. request와 report의 closed schema는 각각
benchmarks/release/candidates/rc3-freeze-input-request-v1.schema.json 및
rc3-freeze-input-admission-v1.schema.json이다. 이 절차는 입력을 읽고 bound diagnostic만
stdout으로 반환하며 evidence root나 source checkout에 output을 쓰지 않는다. 특히
not-frozen/not-run 결과는 freeze가 가능하다는 판정, GPU capture의 성공, Gate E semantic replay,
correctness/soak/rollback semantic 성공, release promotion을 뜻하지 않는다.

### Frozen-candidate input-identity boundary

`write_rc3_frozen_candidate_v1.py`는 `--frozen-candidate-root`로 지정되는 source/input tree
밖의 새 private `0700` root에 정확히 하나의 `frozen-candidate.json`을 create-only로 쓴다.
`replay_rc3_frozen_candidate_v1.py`는 해당 root와 original freeze-input root, source checkout을
caller-held FD로 유지한 채 manifest와 original request/raw leaves를 두 번 재생한다. 이 경계는
admission JSON 또는 `freeze.raw`를 input으로 쓰지 않으며, original request의 all declared
descriptors, source pre-freeze/Cargo.lock/registry binding, two profile self-reference rejection,
그리고 reconstructed baseline v2 raw graph를 다시 확인한다. candidate request와 baseline raw
leaf가 같은 path를 재사용하면 fail-closed한다.

결과 manifest schema는 `riley.rc3-frozen-candidate.v1`
(`benchmarks/release/candidates/rc3-frozen-candidate-v1.schema.json`)이며 authority는
`frozen-candidate-input-identity-only`, candidate status는 `frozen`, qualification status는
`not-run`이다. 대용량 model/weight를 복사하지 않는 recheckable identity pin이므로 source
archive→revision, ELF/container/toolchain/model/correctness의 의미적 provenance, Gate E,
semantic receipt, deployment/rollback result, promotion/qualification을 전혀 주장하지 않는다.
manifest가 남아 있어도 writer normal-return lineage는 증명되지 않는다. future same-stack
semantic producer는 exact manifest **및** original input root를 다시 replay해야 한다.

full frozen-candidate replay는 bounded control-plane JSON을 먼저 검증한 뒤 original request와
정확히 23개 baseline physical leaf의 union에 같은 8,192 descriptor/1 TiB 한계를 적용하고,
그 뒤에만 raw recipe와 artifact를 stream-hash한다. candidate/baseline path alias도 이 지점에서
거부된다. source/input/frozen root는 normalized spelling만이 아니라 held-FD ancestry와 Linux
mount backing coordinate까지 disjoint해야 하며, writer/replayer public wrapper는 visible frozen
root가 held FD를 계속 가리키는지 create와 replay 전후에 확인한다. private Python core는 caller
FD를 직접 mutate하지 않지만 pinned FD를 통한 trusted read-only Git source oracle에 의존하므로
configured Git executable/PATH와 그 behavior는 trusted boundary다. manifest의 not-established
object는 writer normal-return과 input-root immutability도 명시한다.

`riley.rc3-gate-e-input-inventory.v1`와
`replay_rc3_gate_e_input_inventory_v1.py`는 **이미 landed된** 후속 Gate E semantic adapter용 첫
closed inventory boundary다. 별도 private `0700` root는 canonical `gate-e-inputs.json`과
정확히 14 direct regular leaf만 포함한다: release bundle, canonical E0의 native
31-case report/raw·candidate executable 및 optimizer report/raw·profile binary,
Python-free report/raw/golden, performance report/raw, soak report/raw. wrapper는
frozen candidate와 original source/input closure를 held FD로 Gate E artifact 전후에 다시
replay하고, frozen manifest descriptor/candidate/source identity, root topology, exact entry
set, unique direct paths, 1 TiB byte budget을 확인한다.

이 output은 `bound/frozen/not-run`,
`gate-e-input-inventory-replay-only` structural authority뿐이다. report의 `passed`,
threshold, GPU 결과를 해석하지 않으므로 Gate E semantic replay, pass, promotion 또는
qualification을 만들지 않는다. CUDA fault/reproducible-build legacy superset, Qwen, two
profile runtime config, rollback은 four-gate v1 inventory에 넣지 않으며 별도 versioned
semantic boundary에서 다뤄야 한다. legacy path-based `check_release_candidate.py`는 별도
final release checker일 뿐 이 boundary 또는 same-stack Gate E input을 대체하지 않는다.

그 다음의 `replay_rc3_gate_e_native_e0_v1.py`는 이 고정 inventory의 native canonical-E0
한 component만 held FD에서 scratch copy로 옮겨 기존 31-case tensor replayer에 넘긴다. native
raw의 16 GiB 상한은 전체 Gate E artifact streaming 전에 control-plane descriptor로 먼저
거부하고, inventory와 frozen source/input closure는 semantic replay 전후에 다시 replay한다.
scratch bytes는 SHA-256/length/mode까지 재검증하며 native raw result의 source archive, native
report, calibration executable은 모두 SHA-256 **및** length로 frozen/inventory descriptor에
교차 결속된다. failed diagnostic report는 semantic consumer에서 거부된다. 여기의
`native_candidate_executable`은 production/profile binary가 아니라 source-bound calibration
executable이며 둘을 동일시하지 않는다.

이 component output은 `bound/frozen/not-run`, `native_e0_status: passed`,
`gate-e-native-e0-semantic-replay-only`만 뜻한다. optimizer E0, Python-free, performance,
soak, aggregate Gate E pass, semantic receipt, qualification, deployment은 모두 명시적으로
`not-established`다. 현재 원격 host의 기본 Python 3.10에서는 `tomllib`가 없으므로 실제 native
raw replayer는 Python 3.11+ 또는 reviewed `run_release_python.py` compatibility wrapper로만
실행할 수 있다. 이 source-level component은 actual GPU capture나 actual candidate의 Gate E
pass를 기록하지 않는다.

`replay_rc3_gate_e_optimizer_e0_v1.py`는 optimizer report/raw와 release profile binary의 세
direct leaf를 모두 held FD에서 private `0700` scratch로 복사한 뒤 기존 optimizer raw
replayer에 전달한다. raw archive(USTAR padding 포함) 2,151,677,952 bytes, report 8 MiB,
profile binary 512 MiB 및 합산 scratch 한계는 full Gate E artifact stream 전에 거부된다.
scratch root와 세 leaf의 mode/inode/SHA-256/length는 legacy replayer 전후에 다시 검증된다.
scratch는 caller `TMPDIR`가 아니라 고정 external `/var/tmp` parent 아래에 만들며, public
wrapper는 그 parent까지 source/input/frozen/Gate E roots와 held-FD ancestry 및 Linux mount
coordinate 수준에서 disjoint한지 먼저 확인한다.
frozen source revision/archive SHA, 세 inventory descriptor hash, 그리고 caller가 별도로
제공한 `--expected-optimizer-build-image-id`가 결과에 모두 교차 결속된다. report의
top-level/model/GPU/implementation/six-test/fixed37 contract는 legacy final checker와 같은
stdlib-only validator로 다시 검사한다.

이 external image ID는 optimizer report나 raw receipt에서 발견해선 안 되는 caller-supplied
trust input이다. Gate E v1 inventory의 14 leaf와 frozen manifest에는 이 builder image ID가
없으므로, 자기기재 report 값을 anchor처럼 재사용하면 안 된다. 이 adapter는 supplied value와
report/receipt의 equality만 검증할 뿐 그 value가 독립 review에서 승인됐다는 provenance는
증명하지 않는다. 그러한 approval을 Gate E input으로 만들려면 future versioned held-FD policy
artifact가 필요하다. `native_candidate_executable`과 마찬가지로 optimizer의 `profile_binary`는
release bundle/production binary와는 별도 역할이다.

optimizer component output도 `bound/frozen/not-run`, `optimizer_e0_status: passed`,
`gate-e-optimizer-e0-semantic-replay-only`만 뜻한다. native E0, Python-free, performance,
soak, aggregate Gate E pass, semantic receipt, qualification, deployment 및 optimizer build-image
review approval은 명시적으로 `not-established`다. 현재 원격 host에서는 actual raw replay에 Python 3.11+ 또는 reviewed
`run_release_python.py` compatibility wrapper가 필요하며, source-level component은 actual GPU
capture나 actual candidate의 Gate E pass를 기록하지 않는다.

`replay_rc3_gate_e_python_free_v1.py`는 Python-free report/raw/golden, canonical native report와
release bundle을 fixed inventory에서 읽고, frozen candidate의 source archive·release ELF·image
digest·model descriptors는 original input-root replay에서 다시 얻는다. legacy full `evaluate()`는
model directory/weights/producer sidecar/source path를 다시 열므로 호출하지 않는다. raw tar와
bundle만 fixed `/var/tmp` parent의 private `0700` scratch에 held-FD copy하고, legacy raw
loader와 `validate_bound_raw_archive()`에는 그 두 private pathname만 제공한다. scratch root와
leaf의 mode/inode/SHA-256/length는 legacy call 전후 재검증한다. component 전용으로 raw loader의
retained payload는 768 MiB, bundle verifier의 uncompressed retained payload는 640 MiB를 넘지
않아야 한다. 이는 process RSS 전체가 아닌 replay가 보관하는 payload의 상한이며, large model body는
stream-hash해 raw tar 전체 admission 한계와 구별한다.

caller는 nonzero lowercase `--expected-release-image-id`와
`--expected-correctness-golden-sha256`를 제공한다. 전자는 frozen immutable image digest와,
후자는 Gate E golden descriptor와 각각 exact-match해야 하며 raw receipt가 스스로 image/golden을
선택하는 것을 막는다. 이것은 target equality binding일 뿐 independent review approval을 증명하지
않는다. bundle verifier는 embedded `bin/riley`가 frozen release ELF와 같고 release revision이
frozen source revision과 같은지 확인한다. raw model은 frozen model ID/revision/tree/config/tokenizer와
exactly one frozen weight SHA/length에 교차 결속된다.

이 component output은 `bound/frozen/not-run`, `python_free_status: passed`,
`gate-e-python-free-semantic-replay-only`만 뜻한다. full legacy E2E가 요구하는 external model-mount
provenance, producer-sidecar equality와 source archive content, canonical native/optimizer E0 pass,
performance, soak, aggregate Gate E pass, semantic receipt, qualification, deployment 및 두 external
anchor의 review approval은 명시적으로 `not-established`다. source-level component은 actual GPU
capture나 actual candidate의 Gate E pass를 기록하지 않는다.

`replay_rc3_gate_e_performance_v1.py`는 performance report/raw, canonical optimizer report,
profile binary와 native candidate executable만 Gate E held FD에서 읽는다. frozen candidate replay에서
다시 얻은 source archive/release ELF/model tree와 caller-supplied release·optimizer image ID를 각각
exact-match해 self-authored report/raw가 anchor를 선택하지 못하게 한다. native candidate executable은
frozen release ELF와 descriptor pathname이 아니라 SHA-256와 byte length가 같아야 한다.

performance raw archive는 private mode-`0600` scratch copy 한 개로 옮긴 뒤 held FD만 사용한다.
새 streaming replayer는 archive 전체·모든 receipt·canonical re-encoding을 함께 보관하는 legacy
path를 호출하지 않는다. 첫 pass에서 canonical USTAR header/padding/record footer와 `SHA256SUMS`를
stream 검증하고, 두 번째 semantic pass에서는 최대 64 MiB candidate JSON 하나 또는 16 MiB image
inspect 하나만 차례로 파싱해 five independent run의 R7 metrics와 runner cross-binding을 재유도한다.
raw scratch archive 상한은 543,686,656 bytes이며 이는 retained raw bytes의 경계일 뿐 JSON object
expansion을 포함한 process RSS 전체 보장은 아니다.

output은 `bound/frozen/not-run`, `performance_status: passed`,
`gate-e-performance-semantic-replay-only`만 뜻한다. native/optimizer/Python-free component pass,
soak, image-review approval, aggregate Gate E pass, semantic receipt, qualification, deployment은
명시적으로 `not-established`이며 source-level adapter는 actual GPU capture나 actual candidate의
Gate E pass를 기록하지 않는다.

`replay_rc3_gate_e_soak_v1.py`는 Gate E held FD에서 soak report/raw와 shared correctness golden,
canonical native E0 report만 읽고, frozen replay로 다시 얻은 source archive/release ELF/image/model
identity에 모두 exact-match한다. raw archive는 private mode-`0600` scratch copy의 held FD만으로
canonical USTAR header/padding/footer와 `SHA256SUMS`를 먼저 확인한다. 최대 512 MiB `events.jsonl`은
행당 4 MiB 제한으로 streaming하며, 전체 event list를 보관하지 않고 첫 semantic pass와 두 tail-slope
pass로 scenario ordering, transport, counters, golden parity, final quiescence를 재유도한다. raw
scratch archive 상한은 613,556,224 bytes이며 JSON object expansion을 포함한 process RSS 전체 보장은
아니다.

이 component의 `soak_status: passed`는 현재 held raw evidence가 reviewed soak contract와 일치한다는
뜻뿐이다. raw receipt의 designated host/GPU/test-layer 값은 서로의 consistency를 검증하지만 독립된
review approval 또는 actual capture authority가 아니다. aggregate Gate E, semantic receipt,
qualification, deployment 및 actual GPU capture는 명시적으로 `not-established`다.

`replay_rc3_gate_e_aggregate_v1.py`는 detached component JSON이나 legacy
`check_release_candidate.py` 결과를 입력으로 합치지 않는다. 하나의 held-FD/공유-lock stack에서
native E0, optimizer E0, Python-free, performance, soak의 private semantic core를 이 순서로 다시
실행하고, 시작·종료의 닫힌 14-leaf Gate E inventory도 다시 대조한다. 모든 component의 candidate,
source revision, inventory descriptor, frozen manifest는 exact-match해야 하며, canonical native report,
optimizer report/profile binary, correctness golden, source archive, release image·optimizer image·golden
external anchor도 교차 결합한다. native calibration executable과 frozen release ELF는 pathname이
달라도 SHA-256와 byte length가 같아야 한다.

aggregate의 `gate_e_status: passed`는 이 한 invocation에서 다섯 semantic component가 같은 frozen
input closure에 대해 일치했다는 제한된 뜻이다. 결과는 Gate E evidence root에 쓰이지 않으며, actual
GPU capture, frozen-candidate writer의 normal-return lineage, evidence-root immutability, 독립 image/
golden review, release/model/source content provenance, semantic receipt, qualification, deployment,
rollback success는 계속 `not-established`다. 이는 별도의 same-stack semantic receipt 이전 단계이며
CPU/static replay 통과만으로 candidate qualification 또는 원격 GPU 실행을 주장하지 않는다.

## 3. 범위

### 포함

- canonical E0 correctness 31-case
- SmolLM2와 Qwen multi-step generation
- cache on/off 또는 연속/paged KV parity
- fixed/active-row, synchronous/packed, CPU/GPU greedy 등 candidate가 제공하는 exact backend 조합
- mixed prefill/decode와 output slot ordering
- Python 없는 clean release environment
- API streaming, cancellation, client disconnect, overload
- long-running soak와 memory accounting
- reproducible build와 dependency manifest
- release rollback drill

### 비범위

- 새로운 CUDA Graph 구현
- 새로운 kernel fusion
- 신규 model family
- threshold 완화
- failing test의 ignored 처리
- C02-P0의 production `/v1/config` endpoint 또는 startup-artifact 구현

## 4. Candidate configuration 명시

현재 문서들 사이의 historical default 표현이 다를 수 있으므로 qualification은 추론으로 default를 결정하지 않는다. C02-P0가 cold prepare 뒤 release binary의 canonical `GET /v1/config` 응답과 같은 payload를 담은 create-only startup artifact를 제공해야 한다. `/metrics`는 이 계약의 대체물이 아니다.

```text
execution completion mode
batch shape policy and buckets
metadata transport
sampling backend
attention backend
GEMM reduction policy
experimental flags
fallback policy
batch token budget
KV block/page geometry
```

qualification arm은 `stable-default`와 `max-performance-exact` 두 개로 분리한다. stable release 승격은 stable-default만 판정하며, max-performance-exact는 opt-in evidence로 별도 보고한다.

### Raw capture contract

각 arm은 원격 append-only evidence root에 canonical endpoint bytes와 create-only startup-artifact bytes를 별도 raw file로 남긴다. raw startup facts에는 freeze manifest SHA-256, Gate E report SHA-256, 또는 self-authored `passed` 판정을 넣지 않으며, prepared runtime 사실만 기록한다.

### Semantic qualification contract

`startup_configuration`은 raw 파일 자체가 아니라 `check_effective_runtime_config_receipt.py`가 두 arm의 endpoint/artifact bytes, frozen candidate, replayed Gate E report를 재검증해 생성한 semantic check report다. RC3 finalizer는 이 report의 `passed` field를 신뢰하지 않고 같은 외부 evidence root에서 replay한 결과와 byte-identical한지 확인한다.

## 5. Correctness matrix

### Request shape

- concurrency `1,2,5,8`
- prefill-only, decode-only, completing-prefill+decode mixed iteration
- prompt `empty, short, 128, 1024, 4096, near-context-limit`
- output `1,2,32,128`

### KV boundary

- token positions `15→16→17`
- multiple physical page permutations
- reserve/commit/abort/cancel paths
- capacity near-OOM admission

### Sampling

- greedy exact token sequence
- deterministic seed sampling
- request order permutation
- cancelled/rejected branch RNG consumption
- non-finite logits and invalid status handling

### Output routing

- dense slot permutation
- prefill output at last slot, decode outputs at preceding slots
- mid-iteration cancellation
- commit failure after GPU result
- duplicate/unknown request rejection

## 6. Python-free release gate

Python이 설치되지 않고 `PATH`에도 없는 clean image에서 다음을 수행한다.

1. release artifact 설치 또는 copy
2. model/config/tokenizer/safetensors load
3. prefill/decode
4. greedy 및 승인된 sampling
5. streaming HTTP
6. cancellation/client disconnect
7. model shutdown
8. golden token 검증

추가 검사는 다음과 같다.

- `ldd` 또는 loader manifest
- process tree에 Python child 0
- image filesystem에 production dependency로 PyTorch/Transformers/Triton JIT 없음
- network disabled 상태에서 startup/generation 성공
- runtime fallback이 Python subprocess가 아님

## 7. Soak 시나리오

기존 release contract의 전체 duration을 그대로 사용한다. 최소 시나리오는 다음을 순환한다.

- steady short-chat load
- burst 후 idle
- short/long prompt 혼합
- cancellation 0%, 10%, 50%
- client disconnect
- malformed request 반복
- KV utilization 70%, 90%, capacity boundary
- `max-performance-exact` arm에서 GPU-greedy sampling이 ineligible일 때 CPU normative sampling이 선택된 실제 source-owned trace
- graceful shutdown/restart
- model load/unload 반복

매 interval에서 RSS, pinned bytes, VRAM, `GET /v1/c02/metrics`의 free/reserved/active KV
capacity ledger, active/pending request state totals, terminal event counts를 기록한다. 이 C02-only surface가
없는 normal `/metrics` 응답이나 hand-authored zero는 soak/rollback raw evidence를 대체할 수 없다.

## 8. Fault injection

- host allocation failure
- device/pinned allocation rollback ambiguity
- H2D/D2H deferred error
- synchronize/query failure
- post-KV-write runtime error
- output status corruption test double
- scheduler commit failure
- worker/channel close race

실제 device-loss를 안전하게 재현할 수 없는 경우에는 feature-gated injectable backend와
subprocess isolation을 사용하되, synthetic 결과와 실제 GPU sanitizer 결과를 결코 합산하거나
대체하지 않는다. `fault_extension`의 C02 통과는 v3 semantic replay가 frozen candidate와
stable-default binding 위에서 Gate E의 실제 `compute-sanitizer` CUDA evidence와 네 injectable
case capture를 모두 재검증할 때만 가능하다.

## 9. Reproducible build

동일 source archive를 독립 clean environment 두 곳에서 build한다.

- native library와 release binary hash 비교
- build metadata와 embedded source revision 비교
- 허용된 비결정 요소가 있으면 binary section별 원인을 문서화
- 최소한 executable behavior, dependency manifest, ABI hash는 exact해야 함

binary hash exact를 목표로 하며 불가능한 toolchain metadata가 있으면 결과를 본 뒤가 아니라 PR 시작 전에 normalization 정책을 선언한다.

## 10. 예상 파일 및 evidence 경계

현재 checkout에는 다음 static C02 contract와 별도의 C02-P0 소스 구현이 있다. RTX
4090의 `riley-0.1.0-rc99` two-profile container raw smoke capture는 endpoint/artifact
mechanism을 확인한 retained nonqualifying evidence일 뿐이며, 이들 어느 것도 실제 candidate
qualification evidence나 C02 closed decision을 뜻하지 않는다.

```text
benchmarks/release/candidates/rc3-candidate.json
benchmarks/release/candidates/rc3-candidate-template.schema.json
benchmarks/release/candidates/rc3-qualification.schema.json
benchmarks/release/candidates/effective-runtime-config-evidence-v1.schema.json
ci/release/run_rc3_qualification.sh
ci/release/check_rc3_qualification.py
ci/release/check_rc3_prefreeze.py
ci/release/capture_c02_observations.py
ci/release/run_remote_c02_observations.sh
ci/release/check_effective_runtime_config_receipt.py
ci/release/write_effective_runtime_config_startup_artifact.py
ci/release/test_rc3_qualification.py
ci/release/test_rc3_prefreeze.py
ci/release/test_capture_c02_observations.py
ci/release/test_run_remote_c02_observations.py
ci/release/test_effective_runtime_config_receipt.py
benchmarks/release/candidates/qwen-multistep-golden-v2.json
benchmarks/release/candidates/qwen-multistep-wire-v2.json
benchmarks/release/candidates/qwen-multistep-receipt-v2.schema.json
benchmarks/release/candidates/rc3-routing-corpus-v1.json
benchmarks/release/candidates/fault-extension-receipt-v3.schema.json
ci/release/run_remote_qwen_multistep_capture.sh
ci/release/validate_raw_qwen_multistep_capture.py
ci/release/run_remote_rc3_routing_capture.sh
ci/release/bind_raw_rc3_routing_capture.py
ci/release/check_fault_extension_receipt.py
ci/release/run_remote_c02_fault_capture.sh
ci/release/bind_raw_c02_fault_capture.py
benchmarks/release/candidates/reconstructed-prior-baseline-v1.schema.json
benchmarks/release/candidates/reconstructed-prior-baseline-v2.schema.json
benchmarks/release/candidates/soak-v2-receipt-v3.schema.json
benchmarks/release/candidates/soak-v2-bind-request-v3.schema.json
benchmarks/release/candidates/soak-v2-bind-request-v4.schema.json
benchmarks/release/candidates/c02-lifecycle-supervisor-receipt-v1.schema.json
benchmarks/release/candidates/c02-config-endpoint-observation-v1.schema.json
benchmarks/release/candidates/rollback-receipt-v2.schema.json
benchmarks/release/candidates/rollback-receipt-v3.schema.json
benchmarks/release/candidates/rollback-bind-request-v3.schema.json
benchmarks/release/candidates/rollback-receipt-v4.schema.json
ci/release/check_reconstructed_prior_baseline.py
ci/release/check_reconstructed_prior_baseline_v2.py
ci/release/run_remote_c02_soak_v2.sh
ci/release/bind_raw_c02_soak_v2.py
ci/release/write_c02_lifecycle_supervisor_receipt_v1.py
ci/release/prepare_c02_lifecycle_evidence_v1.py
ci/release/verify_c02_lifecycle_launch_inputs_v1.py
ci/release/verify_c02_lifecycle_shutdown_v1.py
ci/release/capture_c02_config_endpoint_observation_v1.py
ci/release/run_remote_c02_config_endpoint_observation_v1.sh
ci/release/check_soak_v2_receipt.py
ci/release/check_soak_v2_receipt_v2.py
ci/release/run_remote_rc3_rollback_capture.sh
ci/release/bind_raw_rc3_rollback_capture.py
ci/release/check_rc3_rollback_provenance_v3.py
ci/release/capture_rc3_rollback_phase_v1.py
ci/release/run_capture_rc3_rollback_phase_v1.sh
ci/release/capture_rc3_rollback_atomic_switch_v1.py
ci/release/run_capture_rc3_rollback_atomic_switch_v1.sh
ci/release/capture_rc3_rollback_atomic_transaction_v1.py
ci/release/run_capture_rc3_rollback_atomic_transaction_v1.sh
ci/release/check_rc3_rollback_provenance_v4.py
ci/release/bind_raw_rc3_rollback_terminal_v4.py
ci/release/check_rc3_rollback_structural_precheck.py
ci/release/replay_rc3_rollback_operational_semantics_v1.py
```

C02 raw-only observation은 frozen candidate나 semantic receipt를 만들지 않는다. 이미 실행 중인
loopback C02 audit server에 대해 endpoint listener inode가 supplied PID의 `/proc/<pid>/fd` socket
set에 request 전후 모두 존재하는지 확인하고, `/proc/net/tcp`·PID stat/status·UUID-selected
`nvidia-smi` rows를 create-only raw input으로 보존한다. 증거 leaf는 source tree 밖의 private
`0700` evidence parent 아래 새로 만들며, `capture-incomplete.json` marker가 남아 있으면
`session.json`이 존재해도 incomplete/nonqualifying이다. 이 관측은 freeze, Gate E, soak,
rollback, qualification을 대체하지 않는다.

현재 landed `check_soak_v2_receipt.py`는 completed raw v4/v5 marker pair를 strict FD
replay로 읽는 precheck일 뿐이며 `bound`/`not-run`과 `raw-structural-only` authority만
출력한다. 이는 listed future `check_soak_v2_receipt_v2.py` semantic receipt가 아니고,
outer RC3 finalizer 또는 C02 qualification checker는 이를 semantic receipt input으로
수용해서는 안 된다.

현재 landed `check_rc3_rollback_structural_precheck.py`도 completed terminal rollback
raw v4 pair의 strict root→switch held-FD replay만 수행하고 `bound`/`not-run`,
`raw-structural-only` authority만 출력한다. v3는 nonterminal이라 명시적으로 거부하며,
visible v4 pair는 prior final-directory-sync ambiguity 뒤에도 남을 수 있다. 따라서 이
precheck는 rollback success·lifecycle·freeze·Gate E·semantic receipt·qualification이 아니고,
future rollback receipt checker와 outer RC3 finalizer는 이를 semantic input으로
수용해서는 안 된다.

다음 semantic 단계는 public `check_rc3_rollback_receipt_v2.py`가 아니라
authenticated runner의 same-stack held-FD private core
`replay_rc3_rollback_operational_semantics_v1.py`다. 이 core는 original raw
candidate/source/rollback/atomic leaves를 재해석해 operational consistency만
`passed/not-run`으로 반환하며 evidence를 쓰지 않는다. frozen-candidate input-identity
manifest/FD-safe replayer와 closed Gate E **input-inventory** replayer는 별도로 landed했지만,
Gate E **semantic** replayer가 아직 없으므로 이 단계는 semantic receipt, actual candidate
freeze, Gate E pass 또는 qualification을 만들거나
주장하지 않는다.

재구성 RC2 baseline을 위한 rollback raw v3 binder도 같은 원칙의 local-only path-only
단계다. request에는 evidence relative path만 넣으며, binder가 private held root FD에서
descriptor와 candidate/rollback target을 derive하고 `stable-default`를 강제한다. raw v3
manifest는 binding input의 descriptor가 아니라 SHA-256 scalar만 유지하므로 binder 시점의
binding은 exact raw leaf에서 derive되지만 이후 v3 replay만으로 그 세 input file을 다시
독립 검증할 수는 없다. 이 단계의 output은 `captured/not-run` nonterminal manifest 하나뿐이며
`.complete`/`.intent` sibling은 publish하지 않고 collision으로만 예약한다. HTTP,
generation, atomic rename, rollback success, historical shipment, 또는 qualification을
주장하지 않으며, 그 해석은 이후 semantic checker의 책임이다.

C02 실행은 새 frozen candidate manifest, 두 configuration arm의 endpoint/startup raw captures, reconstructed-tag baseline manifest, 그리고 `startup_configuration`, Qwen, routing, fault-extension, soak-v2, rollback-v2의 semantic receipts를 append-only result directory와 승인된 외부 evidence root에 남긴다. source-tree의 schema/checker fixture나 template candidate는 실제 raw evidence를 대신할 수 없다.

## 11. Final report 조건

final report는 다음 모든 receipt hash를 포함한다.

- candidate manifest
- startup_configuration semantic receipt와 두 arm의 raw endpoint/startup-artifact descriptor
- canonical correctness
- Qwen regression
- Python-free E2E
- performance regression
- soak-v2 (initial v4 raw provenance와 same-process lifecycle receipt; later semantic receipt remains a separate gate)
- fault injection
- reproducible build
- dependency manifest
- rollback drill (v2 provenance receipt)

하나라도 없거나 다른 candidate에 묶이면 `passed`가 될 수 없다.

## 12. 승인 기준

- canonical failure/token mismatch 0
- cancellation/overload에서 request 또는 KV leak 0
- duplicate terminal/output routing mismatch 0
- soak 종료 후 RSS/VRAM/KV 추세가 contract bound 내
- Python-free gate 통과
- stable-default performance가 승인 baseline threshold를 통과
- rollback binary로 전환 후 health/generation 정상
- 모든 evidence가 같은 candidate SHA/source/binary에 결합

## 13. 실패 처리

qualification PR 안에서 production 코드를 고치지 않는다.

1. failing candidate를 immutable evidence로 보존한다.
2. failure class와 최소 재현을 issue/별도 corrective PR로 분리한다.
3. 수정 PR이 병합되면 새 candidate ID를 만든다.
4. 모든 gate를 처음부터 재실행한다.

부분 통과 결과를 이전 candidate와 조합하지 않는다.

## 14. 롤백

정식 tag 생성 전에는 기존 prerelease/tag를 유지한다. RC3 배포 후 이상이 발견되면 문서에 고정한 이전 stable/prerelease artifact로 atomic rollback하고, candidate worker/model을 drain한 뒤 재사용하지 않는다.

## 15. 완료 정의

최신 단일 Riley revision에 대해 release checker가 예외나 waiver 없이 `Gate E passed`, `Python-free passed`, `candidate qualified`를 반환할 때 완료다.

현재 static schema/checker 또는 initial lifecycle supervisor/receipt의 존재와 CPU/static fixture 통과, 혹은 `riley-0.1.0-rc99` nonqualifying smoke는 이 완료 정의를 충족하지 않으며, actual GPU capture, frozen candidate 또는 qualification 완료를 뜻하지 않는다.
