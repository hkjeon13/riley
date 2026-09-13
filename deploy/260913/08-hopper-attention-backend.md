# PR 08 — Hopper 비동기 attention backend

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

Hopper의 data movement·Tensor Core overlap을 활용하는 실행 경로를 마련한다.

## 의존성과 변경 위치

선행: 01, 03.

예상 수정 위치: architecture dispatch, CUDA/C++ attention adapter, build targets, GPU CI. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. FA3 계열 구현의 버전·지원 shape·라이선스를 확인해 adapter를 연결한다.
2. TMA/비동기 MMA pipeline 요구조건과 workspace plan을 명시한다.
3. SM90 전용 build와 runtime capability dispatch를 연결한다.
4. graph·stream·completion 계약 및 기존 fallback을 통합한다.

## 범위 경계

4090용 instruction 모방이나 backward/training kernel 최적화는 제외한다.

## Correctness·수명 계약

필요한 instruction capability를 검사하며 SM 숫자 하나만으로 모든 feature를 허용하지 않는다. dtype·mask·수치 계약을 01/03과 동일하게 적용한다.

## 검증과 하드웨어 skip

현재 가능한 compile-only·CPU dispatch 검사 실행. 4090에서 SM90 kernel 실행 테스트는 이유와 함께 skip. Hopper 확보 시 numerical, sanitizer, graph, 실제 serving 비교를 실행한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

구현 리뷰와 Hopper 검증 완료를 구분한다. 실제 Hopper 결과 없이는 성능 승격하지 않되 다음 기능 구현은 진행할 수 있다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

해당 architecture backend를 비활성화하고 검증된 fallback을 선택한다.

## 연구 근거

[FlashAttention-3](https://arxiv.org/abs/2407.08608), [ThunderKittens](https://arxiv.org/abs/2410.20399). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
