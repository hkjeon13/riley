# PR 09 — Blackwell attention pipeline

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

Blackwell에서 matmul과 softmax·shared-memory 처리량 불균형을 고려한 attention 경로를 제공한다.

## 의존성과 변경 위치

선행: 01, 03; 08은 필수 의존 아님.

예상 수정 위치: Blackwell build/dispatch, attention adapter, GPU validation matrix. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. FA4 계열 inference 구현의 지원 범위를 고정한다.
2. 비동기 MMA와 softmax pipeline·buffer plan을 adapter에 연결한다.
3. 정확한 device feature·toolchain 요구와 fallback을 등록한다.
4. graph 재사용과 shape별 workspace 관리를 통합한다.

## 범위 경계

논문의 backward 성능 재현, 전체 FFN/quantization 동시 변경은 제외한다.

## Correctness·수명 계약

Blackwell이라는 제품군 이름만으로 kernel 호환을 가정하지 않는다. toolchain build 성공과 실제 device 실행을 구분한다.

## 검증과 하드웨어 skip

compile/dispatch 검사 가능 범위 실행. 4090 runtime은 skip. 대응 Blackwell 장비에서 full-model numerical, stream 수명, 같은 장비·모델의 vLLM/TRT-LLM/SGLang serving 비교.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

장비 미확보 시 구현·검증 대기 상태를 명시하고 다음 작업을 진행한다. TFLOPs 수치를 serving 증거로 사용하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

Blackwell 전용 경로를 비활성화한다.

## 연구 근거

[FlashAttention-4](https://arxiv.org/abs/2603.05451). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
