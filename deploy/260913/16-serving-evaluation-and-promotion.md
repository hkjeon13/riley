# PR 16 — 실제 serving 평가와 후보 승격

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

특정 microbenchmark 승리 대신 동일 조건 serving 목표 충족 여부를 판정한다.

## 의존성과 변경 위치

선행: 01; 후보 PR마다 재사용, 마지막에 전체 조합 평가.

예상 수정 위치: benchmarks/competitive, workload manifests, result analysis; 기존 harness 확장. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 엔진·model·tokenizer·precision·hardware·argv를 고정한 비교 manifest를 확장한다.
2. closed-loop 회귀와 open-loop arrival sweep·혼합 길이·burst·공유 prefix 축을 추가한다.
3. TTFT/TPOT 및 request/token별 tail 정의, SLO goodput·실패/거절·메모리 추이를 기록한다.
4. 교차 순서 반복·불확실성·soak 결과를 근거로 후보 승격 판정을 생성한다.

## 범위 경계

새 kernel 최적화를 이 PR에 넣지 않는다. 지원 안 되는 엔진 조합의 결과를 추정하지 않는다.

## Correctness·수명 계약

출력 길이·EOS·sampling 규칙과 client backpressure를 맞춘다. timed interval에 build/profiling을 섞지 않는다. 요청 실패·rejection·timeout도 분모에 남긴다.

## 검증과 하드웨어 skip

기존 raw fixture로 metric 계산·오류 집계 검증. 실제 같은 장비에서 baseline/candidate/vLLM 및 지원되는 TRT-LLM/SGLang 실행. 지원 안 되는 전용 GPU 테스트는 skip으로 남긴다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

각 batch는 baseline과 vLLM 비교가 필요하다. 최종 throughput ≥vLLM, TTFT/TPOT ≤vLLM과 P95/P99·안정성 유지가 최소 기준; +15% throughput와 −10% TTFT/TPOT 목표를 별도 판정한다. 장비/워크로드 미검증 범위는 완료 주장에 포함하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

성능 회귀 후보는 기존 frozen baseline으로 복귀한다. raw 결과는 삭제하지 않고 rejected/not-promoted 사유를 보존한다.

## 연구 근거

[FlashInfer-Bench](https://arxiv.org/abs/2601.00227), [연구 보고서](../../benchmarks/results/20260912-serving-optimization/RESEARCH_20260913.md). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
