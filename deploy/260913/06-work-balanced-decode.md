# PR 06 — Stream-K·LeanAttention 방식 작업 분할

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

긴 context 또는 output tile 수가 적은 decode에서 GPU 작업 불균형을 줄인다.

## 의존성과 변경 위치

선행: 03.

예상 수정 위치: attention backend plan, kernels reduction/scratch, runtime graph bucket. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. context·batch별 split 작업 계획을 만든다.
2. online softmax partial 결합과 scratch ownership을 구현한다.
3. split/unsplit 선택 및 graph bucket을 연결한다.
4. 실제 serving shape 분포를 평가 harness에 반영한다.

## 범위 경계

GEMM 전체를 동시에 Stream-K로 바꾸거나 짧은 context에도 split을 강제하지 않는다. small-M GEMM은 이 PR 결과 후 별도 후보로 남긴다.

## Correctness·수명 계약

reduction 결합은 수학적 equivalence와 bitwise equivalence가 다르므로 01 수치 계약 적용. padding·빈 split이 결과에 섞이지 않아야 한다.

## 검증과 하드웨어 skip

짧은/긴 context, ragged batch, 최대 길이, split 수 변경의 수치 검증. merge 비용과 scratch 증가 포함. 4090 지원 경로 실행; 장비 특화 경로 조건부 skip.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

long-context serving 이득과 short-context fallback의 회귀 없음이 필요하다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

unsplit backend로 복귀한다.

## 연구 근거

[Stream-K](https://arxiv.org/abs/2301.03598), [LeanAttention](https://arxiv.org/abs/2405.10480). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
