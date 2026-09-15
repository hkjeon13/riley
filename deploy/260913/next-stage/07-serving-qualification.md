# PR-N07 — SLO·동시성 평가와 최종 채택 판정

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N06에서 수치·메모리·기본 serving 완료. microbenchmark 개선을 실제 throughput/TTFT/TPOT·tail 개선과 구분하는 최종 단계다. Scheduler 변경은 이 단계에서 관련 비용이 확인된 경우에만 한 묶음으로 수행한다.

## 변경 묶음

1. 고정 arrival trace에서 낮은 부하, 포화 근처, 과부하·회복을 구분한다. client concurrency와 active capacity, shared/unique를 따로 기록한다.
2. 필요할 때만 prefill chunk 예산, decode 보호 및 resident-token admission을 함께 조정한다. 기존 continuous batching을 재구현하지 않는다.
3. 요청의 완료·실패·timeout·취소를 전부 집계하며 P95/P99 산정에 사용한 표본 수와 percentile 정의를 고정한다. client GC/IO 조건은 모든 엔진에 동일하게 적용한다.
4. 개선 batch 전후와 vLLM 비교를 교차 실행하고 전체 raw·반복 변동을 보존한다. timed 실행 중 profiler/build/download를 수행하지 않는다.

## 비교 및 종료 기준

대표 짧은/긴 입력·긴 출력·공유/비공유에서 throughput(req/s, output tok/s), TTFT/TPOT P50/P95/P99, E2E P95/P99, 실패율·GPU peak·품질을 표로 남긴다. 절대 SLO와 표본/반복 수는 실행 전 manifest에 고정한다. P99 표본이 부족한 run은 tail 미검증이다.

같은 조건의 최소 기준은 vLLM throughput 이상 및 TTFT/TPOT 이하, 목표는 throughput +15% 이상 및 TTFT/TPOT 10% 이상 감소다. 높은 concurrency의 tail·안정성 회귀도 통과할 수 없다. 다른 workload의 유리한 지표를 조합하지 않고 각 필수 cell별 판정 및 전체 미달 cell을 공개한다. 변동이 이득과 겹치면 미확정으로 남긴다.

조건부 후보는 적용 가능한 workload를 명시한 opt-in 채택이 가능하지만 이를 전체 목표 달성으로 쓰지 않는다. 충분한 순이득이 없으면 실패 원인과 다음 메커니즘을 한 번 정리하고 작은 knob 탐색을 자동 연장하지 않는다.

## 롤백과 산출물

정책/후보 flag를 이전 native baseline으로 복귀한다. 산출물은 비교표, raw manifest, 채택/보류 목록, 지원 범위와 남은 하드웨어 검사다. 의미 있는 implementation milestone인 N06과 N07에서 비교표를 작성하며 중간 operator PR마다 전체 serving을 반복하지 않는다.
