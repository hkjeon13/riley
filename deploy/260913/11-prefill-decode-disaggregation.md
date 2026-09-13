# PR 11 — 고정 P/D worker의 분리 serving

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

prefill과 decode의 자원 간섭을 줄이는 분리 serving을 동일 GPU 예산에서 평가한다.

## 의존성과 변경 위치

선행: 10, 04.

예상 수정 위치: server worker routing, scheduler, KV transfer handoff. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 고정 prefill/decode worker 배치와 request routing을 구현한다.
2. prefill 완료→KV handoff→decode 시작 상태 전이를 연결한다.
3. worker별 queue budget과 backpressure·실패 처리를 구현한다.
4. aggregated 실행과 명시적 선택 설정을 유지한다.

## 범위 경계

동적 autoscaling, TaiChi식 자동 hybrid 전환, 전체 클러스터 제어면은 후속 범위다.

## Correctness·수명 계약

첫 token의 중복/누락이 없어야 한다. handoff 실패 시 retry 여부와 이미 외부로 보낸 token 상태를 일관되게 처리한다. transfer 시간을 TTFT에서 제외하지 않는다.

## 검증과 하드웨어 skip

worker 실패·handoff timeout·cancel·과부하, 같은 GPU 총수와 topology의 aggregated/split 비교. 단일 4090에서는 상태 전이와 local simulation만 검증하고 실제 split GPU 테스트는 skip.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

SLO goodput 향상과 tail·실패율 유지가 필요하다. rejection을 숨긴 성능 비교는 무효. 고정 split의 결과 후에만 adaptive hybrid를 별도 설계한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

신규 요청을 aggregated worker로 보내고 기존 handoff를 drain한다.

## 연구 근거

[DistServe](https://arxiv.org/abs/2401.09670), [TaiChi](https://arxiv.org/abs/2508.01989), [Dynamo](https://docs.dynamo.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
