# PR 05 — 메모리 수명 기반 FFN tile 실행

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

gate/up→activation→down 사이 중간 global tensor 이동을 줄인다. launch 수 감소만을 최적화 목표로 삼지 않는다.

## 의존성과 변경 위치

선행: 01.

예상 수정 위치: riley-runtime layer workspace/plan, kernels FFN backend, full-model benchmark. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. FFN DAG와 tensor의 마지막 consumer·수명을 명시한다.
2. 분리 실행과 tile 전달 fusion의 두 plan을 구현한다.
3. 중복 producer 계산·weight read·shared/register 용량을 포함한 후보 선택을 구현한다.
4. shape별 dispatch와 workspace 재사용을 통합한다.

## 범위 경계

범용 GPU compiler 자체 개발, QKV chain까지 동시 재작성, 전 layer persistent scheduler는 제외한다.

## Correctness·수명 계약

부동소수점 연산 순서 변경은 01의 신규 수치 계약을 따른다. 중간 buffer alias는 마지막 GPU 사용 이후에만 허용한다.

## 검증과 하드웨어 skip

작은/큰 M, 전 layer working set, cold/warm 조건을 구별한다. split/merge 오차, spill·재계산, weight traffic을 검증한다. counter가 없으면 논리 byte 추정과 실측을 구분한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

동일 shape micro 결과 이후 full-model·serving 개선을 확인한다. V56의 일부 row 회귀처럼 shape별 손실이 있으면 범위를 제한하거나 reject한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

분리 FFN plan을 기본값으로 되돌린다.

## 연구 근거

[Welder](https://www.usenix.org/conference/osdi23/presentation/shi), [MCFuser](https://arxiv.org/abs/2506.22169), [FLUTE](https://arxiv.org/abs/2407.10960). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
