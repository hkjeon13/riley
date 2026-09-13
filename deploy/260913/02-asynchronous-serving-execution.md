# PR 02 — 비동기 iteration 실행과 응답 처리 중첩

상태: **구현 진행 중**. native submit/query/wait, variable session, scheduler authority를 보유하는 제출 ticket을 연결했다. 두 staging slot의 native 제출·회수 및 모델 session 선택 경로를 연결했다. GPU token 전달과 ahead scheduling은 남아 있다. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

CPU의 batch 준비·stop 처리 중 GPU가 쉬는 구간을 줄여 TTFT/TPOT와 throughput을 함께 개선한다.

## 의존성과 변경 위치

선행: 01.

예상 수정 위치: crates/riley-server/src/engine.rs, riley-runtime의 variable session, kernels/src/graph_resources.cu 및 riley-cuda FFI; variable session은 원격 V56 revision 기준. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. submitted/device_completed/scheduler_committed를 분리한 최대 두 iteration ticket을 도입한다.
2. GPU token 전달과 descriptor·completion buffer 이중화를 연결한다.
3. event 기반 completion 조회와 ticket별 KV page 참조·generation 수명을 구현한다.
4. 다음 batch 준비·제출과 이전 결과 검증·EOS/cancel·HTTP 응답 처리를 중첩한다.

## 범위 경계

다중 stream 동시 model 실행, attention 수치 변경, 무제한 ahead scheduling은 제외한다.

## Correctness·수명 계약

last_accepted_replay는 commit 증거로 유지한다. EOS 뒤 추가 제출된 token은 외부에 내보내지 않는다. cancel은 GPU 완료 증거가 아니며 마지막 접근 완료 전 page/slot을 재사용하지 않는다. launch/event 오류로 완료 불명 시 owner를 격리한다.

## 검증과 하드웨어 skip

page 경계에서 추가 decode 예약, cancel/EOS와 admission 교차, 늦은 completion의 generation 불일치, 중복 commit, launch/event 실패를 검증한다. 4090 exact full-model·serving 회귀 및 CPU/GPU overlap trace. 불필요한 전체 activation 이중 할당이 없는지도 확인한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

실제 overlap trace와 README의 serving 비교가 모두 필요하다. host bubble이 작아 성능 효과가 없으면 기본값으로 승격하지 않고 결과와 원인을 기록한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

동기 실행 설정으로 복귀한다. 진행 중 ticket은 drain 이후 해제하며 실행 도중 owner를 강제 반환하지 않는다.

## 연구 근거

[SGLang overlap](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [TRT-LLM overlap](https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html). 논문 성능 배수는 Riley의 예상 개선율이 아니다.

## 첫 실행 경계 검증

기존 synchronous replay와 별개로 submit/query/wait를 추가했다. input은 제출 함수가 반환하기 전에 retained staging으로 복사한다. 정상 async 제출은 event를 기록한 후 반환하며, 완료 전 read/re-submit은 거부한다. close/Drop은 pending event를 기다린 뒤 graph와 parents를 해제한다. 완료 여부가 오류로 불명확하면 owner를 보유한다. event는 재사용하며 동기 API에는 event 생성을 추가하지 않았다.

4090 GPU transfer lifecycle 검사 1개가 통과했고 같은 검사에 대한 Compute Sanitizer memcheck는 0 errors다. 반복 replay와 명시적 close/Drop 경로가 포함된다. [GPU 로그](../../benchmarks/results/20260913-async-execution/transfer-gpu.log), [memcheck](../../benchmarks/results/20260913-async-execution/transfer-memcheck.log).

이 검사는 전송 graph의 lifecycle 증거다. actual model·dynamic admission·EOS/cancel·GPU future-token 전달·metadata 이중화·serving 성능은 아직 검증하지 않았다. 이 API만으로 한 iteration 앞서는 scheduler가 완성되지는 않는다. fault-injection 설정은 기존 drain-first 동기 오류 경로를 사용하며 실제 async event 실패의 포괄적 주입 검사는 남아 있다.

## Variable session 통합 검증

submit_rows/query_completion/wait_rows를 추가하고 기존 execute_rows를 event 완료 경로로 실행하는 선택 옵션을 제공했다. 기본값은 기존 동기 경로다. 제출 이후 결과 검증 및 scheduler commit까지 기존 retained owner를 유지하고 다음 issue를 거부한다. 아직 두 iteration을 동시에 허용하지 않는다.

SmolLM2 실제 모델의 loaded_v7_compact32/full32/partial 3개 GPU 테스트가 통과했다. 각각 4096/4096/224 출력 위치의 기존 reference 검사와 scheduler settlement, pending close/abort, allocation_zero 검증이 포함되어 있다. compact 경로는 greedy token, full 경로는 BF16 logits를 대조한다. partial 경로에 대한 Compute Sanitizer memcheck는 0 errors다. [모델 GPU 결과](../../benchmarks/results/20260913-async-execution/model-gpu.log), [모델 memcheck](../../benchmarks/results/20260913-async-execution/model-memcheck.log).

이는 비동기 제출 경계의 correctness 증거다. 테스트의 compatibility 경로는 제출 후 대기하므로 CPU/GPU overlap이나 serving 성능 향상을 입증하지 않는다. 다음 batch는 GPU token 전달, completion/metadata 이중화, 두 iteration ticket과 scheduler 반영 순서를 함께 구현하고 실제 serving으로 비교한다.


## Scheduler 제출 ticket 경계

`submit_llama_iteration_variable_graph`는 제출 후 즉시 ticket을 반환한다. ticket이 `AuthorizedExecution`과 mutable session을 빌리므로 wait 이전에 scheduler 예약이나 session을 다시 사용할 수 없다. `query_completion`은 결과를 공개하지 않으며 `wait`만 wire 검증 후 기존 `DownloadedLlamaIteration`을 만든다. Greedy workspace와 full-logit 저장소는 제출 전에 확보하고, 기존 workspace 반환 계약을 유지한다. 제출 전 workspace 부족은 GPU mutation 없이 거절한다.

정상 pending ticket의 Drop은 완료 대기·검증을 수행하되 scheduler commit을 수행하지 않는다. 오류로 완료를 확정하지 못하면 session의 poison/retained owner를 유지하며, KV를 반환하기 전에 owner close가 필요하다. 검증을 마친 ticket도 scheduler settlement와 `confirm_scheduler_commit`이 따로 필요하다. 이 단계는 한 iteration의 host work 분리를 제공하며 두 iteration 실행은 아직 지원하지 않는다.

다음 구현에서 연결해야 하는 실제 의존성:

- `scheduler.rs`의 단일 `inflight: Option<InflightPlan>`과 authority의 immutable scheduler borrow를 유지한 채 단순히 두 번째 launch를 허용할 수 없다. 선행 결과에 의존하는 다음 plan의 tentative reservation과 순서대로 반영하는 상태 전이가 필요하다.
- shared32 embedding은 descriptor의 `shape[row*416]` CPU token을 사용한다. GPU future token은 선행 replay·request/cookie·generation·output slot·status에 묶고 다음 descriptor 검증 및 embedding 전에 전달해야 한다. 숫자 token만 복사해 row 재배치에 사용하는 방식은 금지한다.
- metadata·host staging·completion·event를 두 slot으로 분리한다. model activation·KV 실행은 동일 stream에서 직렬화해 전체 activation 이중 할당을 피한다. input staging 재사용과 output read는 각각 해당 slot의 완료 증거를 요구한다.
- 선행 EOS/cancel로 불필요해진 다음 token은 공개하지 않는다. 추가 KV append 및 page-boundary 예약은 후행 GPU 접근 완료 후에만 정리하고, 실패 시 두 ticket의 소유 관계를 유지한다.

이는 PR 02의 남은 optimization batch이며 별도 성능 개선 완료로 계산하지 않는다. 실제 serving 연결·overlap trace·V56 및 vLLM 비교 전에는 기본값으로 승격하지 않는다.


### Ticket 검증 결과

4090 actual-model compact32/full32/partial 3개 검사가 통과했다(출력 위치 4096/4096/224). compact/full 결과 대조, 부족한 workspace의 제출 전 거절 후 정상 재제출, workspace allocation 반환, pending ticket Drop 후 재사용 거절 및 close/abort의 allocation_zero를 확인했다. partial model Compute Sanitizer는 0 errors였다. CPU 실행 adapter 12개와 Rustdoc 2개(진행 중 ticket의 scheduler 변경을 막는 compile-fail 포함)가 통과했다.

[검증 manifest](../../benchmarks/results/20260913-async-execution/ticket-manifest.json), [모델 GPU](../../benchmarks/results/20260913-async-execution/ticket-model-gpu.log), [memcheck](../../benchmarks/results/20260913-async-execution/ticket-model-memcheck.log), [Rustdoc](../../benchmarks/results/20260913-async-execution/ticket-doc.log), [CPU](../../benchmarks/results/20260913-async-execution/ticket-cpu.log). 원격 테스트 소스는 V56 대비 변경된 crates/kernels 파일의 SHA256을 로컬과 대조했다. 로컬 macOS CUDA check는 nvcc 부재로 실패했고 원격 CUDA build·실행으로 검증했다. Hopper/Blackwell/multi-GPU 실행 증거와 serving overlap 성능 증거는 이번 결과에 포함하지 않는다.


## 두 staging slot의 실행 경로

native graph owner에 두 slot의 ticket·event·staging 수명을 구현했다. 서로 다른 두 입력을 대기 없이 같은 stream에 연속 제출할 수 있으며, 각 slot의 결과를 읽기 전에는 해당 slot을 덮어쓸 수 없다. ticket은 owner generation과 순번을 포함하므로 owner 교체 후 이전 값도 거절한다. query/wait는 device completion만 확인하고 read가 해당 staging slot을 소비한다. read 순서는 허용하지만 이는 scheduler commit 순서를 변경하는 권한이 아니다.

모델 경로는 기존 combined input/output staging과 DAG를 첫 slot으로 재사용한다. 두 번째 staging에 맞춰 H2D/D2H node의 host 주소를 바꾼 DAG를 준비한다. 공유 model activation·device metadata·KV는 같은 stream에서 순서대로 실행한다. 32-row full-logit 호환성을 위해 추가 pinned staging은 `2 × 32 × (128 + 49152 × 2) = 6,299,648 bytes`다. 이는 소스에서 계산한 버퍼 크기이며 CUDA graph 내부 driver 메모리까지 측정한 수치가 아니다. compact-only staging 축소는 아직 하지 않았다.

close/Drop은 두 event를 기다린 후 graph·event·retained parents를 해제한다. CUDA 완료가 불명확하면 다른 slot의 성공으로 오류를 지우지 않고 전체 owner를 보유한다. 기존 synchronous/단일-event API와 buffered API의 혼용은 거절한다. child graph, 외부 host pointer, bound-attention metadata 보정 등 지원하지 않는 기록 형태를 자동 추정해서 변환하지 않는다.

runtime에는 명시적으로 선택하는 `into_owned_buffered_variable_mixed_session`을 추가했다. 준비한 buffer mode를 catalog digest에 포함하고 compact/full/prefill/decode stage 선택을 유지한다. **runtime expectation과 scheduler는 여전히 단일 in-flight다.** 실제 모델 테스트는 두 staging slot을 번갈아 쓰며 순서대로 settlement하는 경로를 검증한다. native의 두 pending 제출 검사와 실제 모델의 두 iteration ahead 실행 증거를 혼동하지 않는다.

### 남은 PR 02 범위

GPU future-token 참조를 선행 replay·request/cookie·generation·output slot에 연결하고, scheduler의 두 tentative plan·KV 예약 및 순차 commit을 구현해야 한다. 그 뒤 EOS/cancel 뒤의 후행 결과 폐기, page-boundary 예약 정리, 실제 serving overlap trace와 V56/vLLM 비교를 한 batch로 수행한다. 현재 serving 기본값은 변경하지 않았으며 성능 승격은 하지 않는다.


### 두 slot 검증 결과

최종 소스로 전송 GPU 검사 2개, 실제 모델 compact32/full32/partial 3개가 통과했다. 전송 검사는 두 번의 연속 제출, 역순 결과 회수, 한 slot을 재사용하는 동안 다른 slot의 unread 결과 보존, owner 교체 이후 ticket 거절, pending close/Drop을 포함한다. 모델 검사는 출력 위치 4096/4096/224의 reference 검사 및 allocation_zero를 유지했다. 모델 partial 및 전송 각각의 Compute Sanitizer memcheck는 0 errors다. CPU riley-cuda 92개와 CUDA server check도 통과했다.

[manifest 및 소스 SHA256](../../benchmarks/results/20260913-buffered-execution/manifest.json), [전송 GPU](../../benchmarks/results/20260913-buffered-execution/transfer-gpu.log), [모델 GPU](../../benchmarks/results/20260913-buffered-execution/model-gpu.log), [전송 memcheck](../../benchmarks/results/20260913-buffered-execution/transfer-memcheck.log), [모델 memcheck](../../benchmarks/results/20260913-buffered-execution/model-memcheck.log), [CUDA server check](../../benchmarks/results/20260913-buffered-execution/server-check.log). serving overlap·vLLM 대비 성능과 미보유 하드웨어 실행은 이 검사 범위에 포함하지 않는다.


## KV 예약의 부분 commit

`SequenceState::commit_prefix`는 하나의 최종 target까지 확보한 예약에서 완료된 앞부분만 logical length로 commit한다. 후행 block·tentative table·reservation 잠금은 유지한다. 기존 `commit`은 나머지를 확정하고, 실행하지 않은 후행의 `rollback` 또는 완료 후 실패한 후행의 `poison`은 이미 commit된 앞부분에 필요한 블록을 반환하지 않는다. 새로운 host block table을 추가 할당하지 않는다.

prefix commit마다 detached reservation과 내부 pending reservation의 nonce를 함께 갱신한다. page 수가 같아도 이전 token을 다시 사용할 수 없다. 전체 남은 block 소유권과 nonce 여유를 먼저 확인하고, 앞부분의 변경된 block sidecar를 무효화한 후 logical length를 반영한다. 일반 `block_table`, 새 `reserve_to`, sidecar 부착은 후행 settlement까지 계속 거절한다.

이 함수는 **GPU 완료를 증명하지 않는다**. 호출자는 앞부분의 device 쓰기 완료와 후행이 앞부분을 덮어쓰지 않는 append 범위를 증명해야 한다. 후행 page를 rollback·poison·close하기 전에는 후행 GPU 접근을 끝내야 한다. 기존 scheduler에는 아직 연결하지 않았으므로 두 in-flight plan이나 EOS/cancel 중의 page 해제를 자동으로 안전하게 만들지 않는다. 다음 scheduler 변경은 이 예약을 두 plan에 연결하고 첫 결과의 조기 종료 시 후행 접근 종료까지 completion/page 반환을 보류해야 한다.

CPU KV 검사 14개와 기존 scheduler 검사 37개가 통과했다. 252개 경계 조합에서 prefix commit 뒤 suffix commit/rollback, block ID·valid-token 합계·pool accounting을 대조했다. 같은 page의 stale nonce, foreign 예약, 역순/중복 prefix, nonce 소진의 무변경 거절, sidecar 무효화, 후행 실패·예약 유실 후 prefix 유지도 검사했다. CUDA feature build에서도 동일 host KV 검사를 실행하며, 이를 GPU 동시 실행 검증으로 계산하지 않는다.

[부분 commit 검증 manifest](../../benchmarks/results/20260913-prefix-kv/manifest.json), [CPU KV](../../benchmarks/results/20260913-prefix-kv/kv-cpu.log), [scheduler](../../benchmarks/results/20260913-prefix-kv/scheduler-cpu.log), [CUDA feature의 host 검사](../../benchmarks/results/20260913-prefix-kv/cuda-feature-host-tests.log). 최종 소스 SHA256을 원격과 대조했고, CUDA feature에서도 14개 host 검사가 통과했다.


## 시간 예산에 따른 다음 구현 순서

[기존 serving trace의 graph 사이 시간 재계산](../../benchmarks/results/20260913-overlap-headroom/README.md)에서 natural C16/C32의 간격은 각 trace window의 12.65%/17.47%였다. 모든 간격을 제거한 낙관적 trace 상한은 +14.48%/+21.17%지만 profiler/client pacing을 포함하므로 unprofiled serving 예측에 사용할 수 없다. 별도 round62 C32 natural에서 목표까지 필요한 throughput 변화는 +31.72%다.

PR 02를 단독 해법으로 간주하지 않는다. 다음 구현은 PR 03 attention adapter의 실제 모델 경로를 우선한다. PR 02의 GPU future token·두 plan 예약·순차 commit은 그대로 남은 범위이며, 완료·성능 승격으로 표시하지 않는다. GPU 연산과 CPU 중첩의 결합 효과는 실제 serving 통합 후 비교한다.


## GPU future-token native 경계

[GPU 전달 검증](../../benchmarks/results/20260913-future-token-native/README.md): 선행 compact 결과 identity 및 연속 replay/progress 검사, row 재배치, descriptor/token-slab 전달, 전체 batch 검증 후 변경을 native prototype으로 구현했다. Device producer→consumer graph의 288개 검사 및 memcheck/racecheck를 통과했다. SM90a/SM100a compile 통과, runtime은 장비 부재 skip이다. 모델의 status 초기화 뒤·embedding 앞에 연결해야 한다는 기존 enqueue 순서도 확인했다.

이는 scheduler 연결 전의 전달 경계다. Canonical reference를 실제 retained authority에서 생성하는 Rust 표현, 두 pending expectation, tentative KV 예약·순차 commit, EOS/cancel 후 drain과 결과 폐기, 실제 모델·serving 검증이 남아 있다. 현재 prototype을 두 iteration overlap 지원으로 표시하지 않는다. FP16 모델 품질 실패 이후 다음 구현 영역을 PR02로 전환하며 앞선 'PR03 우선'은 당시의 순서 기록으로 남긴다.


## Future-token Rust wire와 cookie 수정

[후속 ABI 검증](../../benchmarks/results/20260913-future-token-wire/README.md): 실제 session은 제출마다 cookie를 새로 발급한다. 이에 맞춰 native reference를 source identity + destination cookie로 수정하고 row140 bytes로 확장했다. Rust가 committed replay를 유지한 tentative successor packet을 준비하면서 source/progress·fresh cookie·KV page prefix/ownership을 검증한다. Rust canonical pure/mixed fixtures의 GPU 전달과 memcheck/racecheck를 통과했다. Native v1의 동일-cookie 조건은 폐기한다.

두 scheduler reservation·authority와 runtime expectation queue는 아직 연결하지 않았다. 이번 checked wire 준비를 ahead scheduling 완료로 표시하지 않는다. 후행 실행은 기존 단일 in-flight API에 우회 제출하지 않는다.


## 예약의 두 endpoint와 완료 append 폐기

[KV window 검증](../../benchmarks/results/20260913-reservation-window/README.md): shared reservation의 prefix view와 scheduler용 두 endpoint snapshot을 추가했다. 완료·append-only 쓰기 증거가 있을 때 suffix를 폐기하는 경로는 committed prefix page를 유지하고 tail sidecar를 무효화한다. KV18개·scheduler38개 CPU 검사가 통과했다. 이는 GPU fence나 두 in-flight scheduler 완료가 아니다. 다음은 이 도구를 두 계획 authority·prefix publication·suffix drain에 연결하는 것이다. 기존 전체 commit/terminal reclaim 경로를 후행 GPU가 pending인 상태에 그대로 사용하지 않는다.


## 실제 scheduler two-decode window 연결

[Scheduler window 검증](../../benchmarks/results/20260913-scheduler-decode-window/README.md): 두 decode plan의 공동 최종 예약·prefix/full table, scoped pair authority→future wire, 두 결과 검증 후 prefix/suffix settlement를 구현했다. 선행 stop·cancel의 후행 출력 폐기, OOM rollback, admission fallback, C32 ragged 진행을 host 검사로 확인했다. 기존 single-plan 실행/완료 API로 window를 우회할 수 없다. 이는 더 이상 wire-only prototype은 아니지만 GPU window 실행 adapter와 serving 선택은 미구현이다.

현재 완료 처리는 두 실행 drain 후 공개 방식이다. 다음은 두 staging slot·future-token resolver·source result/sidecar lifetime·graph identity를 실제 모델 경로에 연결하고, output correctness와 EOS/cancel drain을 GPU에서 검증하는 것이다. 새 serving 비교표는 그 통합 milestone에서 작성한다.

## Native model window 실행 연결

[GPU window 검증](../../benchmarks/results/20260913-native-decode-window/README.md): buffered V7 decode graph에 status reset 뒤/embedding 앞 future-token resolver를 연결했다. 첫 device result·두 pinned output·reference scratch의 dependency와 수명을 고정하고, native predecessor ticket 검사·runtime pair expectation/commit·scheduler pair authority adapter를 함께 구현했다. 32개 ragged 요청의 반복 window 생성은 직렬 V7과 exact이며, 선행 stop/cancel의 출력 폐기 및 context allocation 회수도 GPU에서 확인했다.

다음 구현은 HTTP engine의 명시적 window 선택과 기존 sampling/audit/cancel 계약 연결이다. 양쪽 drain 후 공개하는 현재 방식은 serving TPOT/tail 결과를 보고 판단한다. 그 통합 milestone에서 직전 Riley·새 Riley·vLLM 비교표를 작성한다.

## Serving opt-in 연결 — 실행 검증 대기

[Serving window 구현·검사 범위](../../benchmarks/results/20260913-serving-decode-window/README.md): `--decode-window paired-experimental-v1`을 추가하고 GPU greedy 선택, 선행 stop에 대한 후행 GenerationState 수용 억제, 요청·생성 순번별 audit/SSE publication, 재사용 staging을 함께 연결했다. 기본값은 single이다. Host 검사와 별도 복사본의 CUDA-feature Rust typecheck는 통과했으나 원격 SSH가 응답하지 않아 이 serving 연결의 native build·HTTP/GPU 실행·vLLM 비교는 아직 수행하지 못했다.
