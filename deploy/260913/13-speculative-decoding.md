# PR 13 — Draft-target speculative decoding

상태: **Rust greedy 승인·KV 정산 기반 구현, target-prefix 수치 전제 검증 완료 / 실제 speculative serving 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

큰 target에서 검증 한 번으로 여러 token을 진행해 순차 decode 횟수를 줄인다.

## 의존성과 변경 위치

선행: 01, 03; draft/target 모델 로딩 지원을 착수 전에 확인.

예상 수정 위치: runtime draft/target executor, KV transaction, scheduler token accounting. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 지원되는 draft-target 조합 하나와 최대 draft 길이를 고정한다.
2. batched verification과 greedy acceptance를 구현한다.
3. 승인 prefix commit·거절 suffix KV rollback을 연결한다.
4. acceptance 비용 기록 및 일반 decode fallback을 추가한다.

## 범위 경계

EAGLE 학습 pipeline, 임의 모델 지원, stochastic sampling까지 한 PR에 넣지 않는다. greedy-only 범위를 API에 명시하고 unsupported sampling은 fallback한다.

## Correctness·수명 계약

greedy output은 target 단독과 같아야 한다. EOS·length limit 뒤 token을 승인하지 않는다. 향후 sampling은 target 분포 보존 검증이 별도로 필요하다.

## 검증과 하드웨어 skip

accept all/none/partial, page 경계 rollback, EOS, cancellation, 혼합 batch. 작은 135M뿐 아니라 지원 가능한 큰 target을 별도 축으로 평가. draft 모델 미지원은 하드웨어 skip으로 위장하지 않는다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

draft·verification·rollback을 포함한 serving TPOT/throughput 및 acceptance length를 보고한다. 낮은 concurrency 이득과 높은 concurrency 손실을 분리해 적용 범위를 정한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

draft 경로를 끄고 target 단독으로 복귀한다.

## 연구 근거

[EAGLE-3](https://arxiv.org/abs/2503.01840), [Speculation limits](https://arxiv.org/abs/2601.11580). 논문 성능 배수는 Riley의 예상 개선율이 아니다.


## 첫 구현 batch — 2026-09-14

[기반 구현과 GPU 증거](../../benchmarks/results/20260914-speculative-foundation/README.md): 최대8개 draft의 greedy prefix 승인, replacement/bonus, EOS·길이·취소 처리, 기존 KV reservation의 완료된 prefix 정산을 연결했다. 추가 모델 없는 prompt lookup을 최초 제안기로 구현했으며, 지원되는 큰 target/draft 모델 쌍 도입을 대체 완료하지 않는다. Python은 serving 경로에 들어가지 않는다.

CPU5개 테스트는 KV252조합 및 독립 직렬 oracle11,250조합을 포함해 통과했다. 기존 accepted target의 순차9출력과72개 독립 prefix 요청에서 BF16 logits7,077,888 bytes가 일치했고 full-model memcheck0이다. 이것은 기존 경로의 수치 전제 검사이며, 한 번의 fused verification이나 serving 이득이 아니다.

다음 묶음은 K+1 target prediction 반환, private KV append의 GPU 완료와 승인 prefix 정산, scheduler 출력·길이·취소 accounting 연결, 실제 acceptance·draft·verify·rollback 비용 및 vLLM serving 비교다. 검증된 target 수치 계약을 그대로 사용하며, FP32 분할 attention의 strict 실패를 우회하지 않는다. 미지원 sampling/stop-string/processor 경로는 ordinary decode를 유지한다.

## 다중 위치 target head — 2026-09-14

[GPU 검증 결과](../../benchmarks/results/20260914-verification-head/README.md): 한 packed 모델 실행의 여러 hidden 위치를 기존32행 head에 모으고, 별도 pinned 결과와 Rust completion identity로 읽는 실험 경로를 연결했다. 최대4owners×8positions이며 pending input을 포함하므로 첫 GPU 경로의 draft 상한은7이다. 기존 응답·정확한 target 수치 경로를 유지하며 buffered/compact 경로는 지원하지 않는다.

순차 target과72행 BF16 logits7,077,888 bytes가 일치했고 실제 모델 memcheck 및 bounded selector memcheck/racecheck가 오류0이다. 완료 전·정산 후 읽기 거부도 검사했다. 초기 prompt까지8-token chunk로 처리하는 검증 fixture이므로 총 모델 실행 수는 양쪽10회이며 성능 향상을 주장하지 않는다.

다음 구현 묶음은 GPU argmax의 작은 검증 결과, private speculative append와 greedy 승인/rollback, scheduler 다중 토큰 accounting을 연결하는 것이다. 아직 실제 speculative serving이나 vLLM 비교 완료가 아니다.
