# Native model decode window integration

Scheduler의 shared reservation/authority를 실제 두 CUDA graph 제출과 연결했다. Rust → C ABI → CUDA 경로이며 Python interpreter를 serving에 넣지 않는다. HTTP serving 선택 경로는 아직 기존 단일 iteration이다.

## Optimization batch

1. **GPU future input**: buffered V7 compact decode graph를 cold preparation에서 복제하고 `status reset → reference H2D → resolver → embedding` dependency를 추가한다. predecessor의 compact device result를 직접 읽는다. 4480-byte reference는 pinned staging의 사용하지 않는 input 영역에 복사하고, embedding 이후 재사용되는 attention scratch를 잠시 사용하므로 새 device allocation은 없다. 기존 model kernel·수치 profile은 유지한다.
2. **Native ownership**: 각 slot에 future graph/exec를 보유하고 함께 close한다. 같은 owner의 가장 최근 미소비 predecessor ticket, compact output, 후행 decode profile, packet/sidecar extent를 검사한다. 같은 stream에서 두 graph를 순서대로 실행한다. 첫 host result는 별도 staging에 보존되며, 실패 시 owner가 close/drain하기 전 KV를 해제하지 않는다. Resolver source를 graph digest와 build dependency에 포함한다.
3. **Runtime/scheduler connection**: 두 cookie 집합과 expectation을 유지하고 두 graph를 먼저 제출한다. 두 번째 completion을 기다린 후 양쪽 결과를 검증하며, 후행 기대 입력은 검증된 첫 token으로 확인한다. 실제 accepted replay는 scheduler가 두 결과를 settlement한 후에만 전진한다. `execute_llama_decode_window`는 실행·검증 동안 pair authority를 borrow한다. 일반 single-step commit API는 window를 거절한다.

## Actual model validation

RTX 4090 SM89, SmolLM2-135M BF16, GUI 유지·Blender 중지. 기존 고정 checkpoint를 사용했다. [GPU 검사](model-v3.log)는 다음 세 실행을 각각 새 owner로 수행한다.

- 직렬 V7: 길이가 다른 32개 prompt, 요청당 32–34개 출력, 총 1055개 token.
- Paired V7: 같은 요청과 출력 예산, 12개 window, 모든 token이 직렬 결과와 정확히 일치.
- Paired V7 terminal: 첫 window에서 선행 stop flag와 deferred cancel을 주입한다. 후행 출력은 폐기하고, 나머지 출력은 직렬 reference prefix와 일치한다. 12개 window를 실행한다. 이는 실제 EOS token을 모델이 생성했다는 주장이 아니라 stop/cancel settlement 검증이다.

세 실행 모두 완료 후 context allocation이 0이다. 잘못된 단일 commit, 잘못된 pair commit ID, 미정산 상태에서 재발급을 거절한다. 로컬 scheduler host 검사 44개와 Rustdoc 2개도 통과했다. 전체 모델 [memcheck console](memcheck-console.txt)은 exit 0·오류 0건이다. 이 검사는 debug binary의 correctness 검사이며 실행 시간은 serving 성능 수치가 아니다.

초기 CUDA 13 graph edge API 인자 오류는 version guard로 수정했다. 확대 검사 편집 중 normal branch에 중복 삽입된 window assertion은 제거했다. 실패 로그와 수정 후 로그를 구분해 보존한다.

## Remaining serving milestone

HTTP engine에서 이 API를 명시적 opt-in 경로로 선택하고, 기존 greedy/EOS·sampling/audit·cancel 계약을 유지해야 한다. 현재 두 결과가 모두 drain된 뒤 token을 공개하므로 token 전달 간격과 P95/P99에 불리할 수 있다. 실제 serving에서 직전 Riley·window Riley·vLLM을 같은 workload와 반복 순서로 비교하기 전에는 기본 경로로 승격하지 않는다.

Throughput·TTFT·TPOT·P95/P99의 새 serving 측정은 **미측정**이다. Hopper/Blackwell/multi-GPU 실행은 이 검사에서 수행하지 않았다. PR02와 최종 성능 목표는 미완료다.

## Pending remote observation

추가 full-model racecheck를 실행한 후 SSH banner exchange까지 응답하지 않는 상태가 관찰됐다. racecheck의 성공·실패·종료는 아직 확인하지 못했으며 원인은 확정하지 않는다. 실행 handle은 `42095`, 원격 예상 로그는 `/tmp/riley-opt-260912/native-window-v1/racecheck.log`이다. 원격 상태 확인 전에는 GPU 작업을 다시 시작하지 않는다. 원본 memcheck 로그를 가져오는 scp handle `86058`은 이후 exit 0으로 완료됐다. [원본 로그](memcheck.log)와 [exit](memcheck.exit)를 보존한다. 위 memcheck console은 그 전에 exit 0으로 완료된 실행의 tool 출력이다.

[소스 snapshot](source-hashes.json)은 업로드에 사용한 로컬 소스를 기록한다. 추가 원격 hash readback은 연결 복구 후 수행한다. 이번 통합은 serving 성능 승격을 의미하지 않는다.
