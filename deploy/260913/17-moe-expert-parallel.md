# PR 17 — MoE expert dispatch/combine

상태: **계획만 작성 / 미구현**. [공통 계약](README.md)을 따른다.

## 문제와 가설

큰 MoE에서 expert 간 token 이동과 계산을 연결한다. 현재 dense SmolLM2의 성능 개선 수단으로 취급하지 않는다.

## 의존성과 변경 위치

01, 12의 rank·collective 수명 계약. 추가 선행: 지원 MoE 모델 하나의 loader, router, 단일 device reference forward. 이 선행이 없으면 EP를 동작한다고 주장하지 않으며 모델 지원을 별도 PR로 먼저 구체화한다.

예상 위치: runtime execution plan, scheduler/server 경계, CUDA/transport adapter 및 benchmarks. 구체 파일은 실제 checkout에서 확인한다.

## 하나의 optimization batch

1. expert placement와 token→expert routing descriptor를 정의한다.
2. DeepEP adapter 한 경로로 dispatch/combine과 stream event를 연결한다.
3. bounded communication buffer와 backpressure를 구현한다.
4. rank failure·중복/늦은 completion·통신 자원 teardown을 처리한다.

## 범위 경계

모델 로더·router 신규 개발, dynamic expert migration, 다양한 MoE 구조 지원을 이 PR에 합치지 않는다. 대상 topology와 expert routing 형식을 하나로 고정한다.

## Correctness·수명 계약

router의 top-k·weight·token 순서를 reference와 유지한다. token drop을 암묵적으로 허용하지 않는다. rank들이 같은 collective 순서에 참여하며 통신 완료 전 buffer를 재사용하지 않는다.

## 검증과 하드웨어 skip

빈 expert, 특정 expert 쏠림, 중복 routing, capacity 경계, 일부 rank 실패를 검사한다. CPU routing/reassembly 및 build 검증을 현재 실행한다. 4090 한 장에서는 실제 EP 통신·성능을 skip한다. 지원 GPU·topology 확보 후 단일-device reference와 결과 대조 및 같은 GPU 수의 경쟁 엔진 serving 비교를 수행한다.

## 완료·승격 기준

실제 모델·topology의 correctness와 SLO goodput·tail 증거가 있어야 승격한다. rank별 idle·통신 대기·buffer 증가를 같이 기록한다. 모델 선행 미구현은 하드웨어 skip이 아니다.

구현·실장비 검증·성능 승격 상태는 별도로 기록한다. 실패한 테스트를 skip으로 바꾸지 않는다.

## 롤백

신규 요청의 EP 경로를 끄고 진행 중 group을 drain한다. 모델이 기존 장비에 들어가지 않으면 단일 GPU로 무조건 fallback하지 않고 명시적 미지원 오류를 반환한다.

## 연구 근거

[DeepEP](https://github.com/deepseek-ai/DeepEP), [ParallelKittens](https://arxiv.org/abs/2511.13940) — Riley 적용 설계이며 논문의 개선 배수를 예상 성능으로 사용하지 않는다.
