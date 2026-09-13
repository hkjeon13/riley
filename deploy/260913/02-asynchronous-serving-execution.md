# PR 02 — 비동기 iteration 실행과 응답 처리 중첩

상태: **구현 진행 중**. native submit/query/wait와 variable session을 연결했다. GPU token 전달·이중 buffer·ahead scheduling은 남아 있다. 공통 계약은 [README](README.md)를 따른다.

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
