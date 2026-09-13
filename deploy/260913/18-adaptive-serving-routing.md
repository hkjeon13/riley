# PR 18 — SLO 기반 aggregated/P-D routing

상태: **계획만 작성 / 미구현**. [공통 계약](README.md)을 따른다.

## 문제와 가설

부하에 따라 prefill/decode 간섭과 KV 전송 비용의 균형이 바뀌므로 신규 요청의 배치 경로를 선택한다.

## 의존성과 변경 위치

10, 11; 정책 비용을 정할 고정 aggregated/split 측정 자료. 장비가 없으면 비용 모델을 주입한 상태 전이 검증과 구현은 진행한다.

예상 위치: runtime execution plan, scheduler/server 경계, CUDA/transport adapter 및 benchmarks. 구체 파일은 실제 checkout에서 확인한다.

## 하나의 optimization batch

1. queue·KV locality·예상 prefill/decode 비용을 사용하는 경로 선택을 구현한다.
2. hysteresis와 최소 유지 시간으로 routing 진동을 제한한다.
3. worker capacity와 backpressure를 정책에 반영한다.
4. 선택 이유·전환 비용·SLO miss를 기록하고 고정 정책 fallback을 연결한다.

## 범위 경계

worker autoscaling, 실행 중 요청 migration, 클러스터 재배치는 제외한다. 고정된 worker pool에서 신규 요청의 경로만 선택한다.

## Correctness·수명 계약

기존 요청의 ownership은 변경하지 않는다. cache hit 예상이 틀리면 정상 miss 경로로 실행하며 실패·거절도 전체 요청 분모에 남긴다. 정책 자체가 SLO 달성에 맞춰 benchmark 요청을 선별하면 안 된다.

## 검증과 하드웨어 skip

burst·장문 비중 변화·cache hit/miss·worker 장애·routing 진동의 상태 전이를 검사한다. 실제 다중 GPU 테스트는 장비 부재 시 skip. 확보 후 같은 총 GPU·동일 trace로 고정 aggregated, 고정 split, adaptive를 비교한다.

## 완료·승격 기준

정책 비용과 handoff를 포함한 SLO goodput 및 P95/P99로 채택한다. 한 phase의 개선이 다른 phase의 starvation을 만들면 승격하지 않는다.

구현·실장비 검증·성능 승격 상태는 별도로 기록한다. 실패한 테스트를 skip으로 바꾸지 않는다.

## 롤백

신규 요청을 검증된 고정 routing으로 전환한다. 기존 in-flight 요청은 원래 경로에서 종료한다.

## 연구 근거

[TaiChi](https://arxiv.org/abs/2508.01989), [DistServe](https://arxiv.org/abs/2401.09670), [Mooncake](https://arxiv.org/abs/2407.00079) — Riley 적용 설계이며 논문의 개선 배수를 예상 성능으로 사용하지 않는다.
