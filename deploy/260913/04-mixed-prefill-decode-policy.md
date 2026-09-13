# PR 04 — POD 실행과 시간 예산 기반 mixed batching

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

prefill의 compute 자원과 decode의 memory 자원을 함께 활용하면서 decode tail을 제한한다.

## 의존성과 변경 위치

선행: 03.

예상 수정 위치: riley-scheduler, riley-runtime mixed plan, attention adapter, serving workload harness. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. POD mixed execution의 지원 shape·resource plan을 연결한다.
2. 측정된 batch 시간 비용에 따라 prefill chunk와 decode budget을 선택한다.
3. admission·graph bucket·workspace plan을 일관되게 선택한다.
4. 긴 prefill의 starvation 방지와 decode SLO 추적을 추가한다.

## 범위 경계

이미 있는 continuous batching/chunking을 신규 기능으로 재구현하거나 모든 batch를 POD로 강제하지 않는다.

## Correctness·수명 계약

원래 요청 순서·token accounting·KV append 의미 유지. 정책이 요청을 거절하거나 늦추면 offered load, rejection, queue delay에 포함한다.

## 검증과 하드웨어 skip

짧은/긴 prompt·출력 혼합, burst와 지속 도착, starvation, prefill 진행 중 cancellation. POD와 분리 실행을 같은 workload에서 비교한다. 미지원 장비의 POD runtime은 skip.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

정해진 TTFT/TPOT SLO 안 goodput과 P95/P99가 개선되어야 승격한다. 긴 prompt를 굶겨 얻은 decode 개선은 채택하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

기존 chunk/admission 및 분리 attention 실행으로 복귀한다.

## 연구 근거

[POD](https://arxiv.org/abs/2410.18038), [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [NanoFlow](https://arxiv.org/abs/2408.12757). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
