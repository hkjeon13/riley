# PR 02 — 비동기 iteration 실행과 응답 처리 중첩

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

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
