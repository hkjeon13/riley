# PR 04 — POD 실행과 시간 예산 기반 mixed batching

상태: **실험용 wall-time feedback 정책 구현·C32 회귀 확인, 비활성 유지. POD 및 전체 정책 미완료**. 공통 계약은 [README](README.md)를 따른다.

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


## 2026-09-14 calibration과 착수 경계

[측정 및 비용 자료](../../benchmarks/results/20260914-mixed-chunk-calibration/README.md): 현재 V7은 decode 우선·최대4개 prefill·고정 budget512의 mixed scheduler다. 새 prefill FFN paired를 유지한 C32 두 순서에서 chunk512→128은 throughput−9.91%,256은−1.68%였다. 전체 reference가 일치했지만 고정 chunk 축소는 채택하지 않는다.

진단 모드의 bounded shape histogram을 추가해 row/decode/prefill/context 구간별 성공한 ordinary 실행 wall-time과 overflow를 보존했다. 작은 chunk의 mixed iteration 수 및 누적 실행 시간이 증가했다. 이를 순수 GPU predictor로 사용하지 않는다. 다음 구현 batch는 비용 추정과 미관측 fallback, decode 진행·prefill 최소 진행량/aging을 지키는 선택, graph/KV/admission 계약, 긴 prompt와 open-loop SLO 평가를 함께 다룬다. POD native backend와 시간 정책 완료는 여전히 별도 검증이 필요하다.


## 첫 wall-time feedback 후보 결과

[구현과 C32 비교](../../benchmarks/results/20260914-mixed-time-policy/README.md):8개 context/decode class, saturated 성공 샘플8개,20% deadband·64token 조정·128 하한, validated settlement 이후 feedback, decode 우선·기존 KV/graph capacity 계약을 연결했다. CPU scheduler48/CLI34/config7, CUDA build 및 serving stop/cancel/reference 검증을 통과했다.

4ms 목표의 두 순서 C32에서 직전 동일 FFN paired 대비 throughput−8.85%, TTFT/TPOT/P95/P99도 악화했다. 비활성 opt-in으로 유지하며 기본값 승격하지 않는다. coarse reactive controller를 비용 predictor/POD 완료로 취급하지 않는다. 고부하·장기 qualification은 미실행이다. 다음 batch는 threshold 미세 튜닝보다 실제 attention prefill/decode 자원 공유와 수치/지원 계약을 다룬다.

## POD-style native gate (2026-09-14)

[Native 구현과 실험](../../benchmarks/results/20260914-pod-mixed-native/README.md): packed prefill/decode queue, explicit device task body,4-warp virtual 작업, SM 교대/비례 배정 및 static packing 대조군을 구현했다. Audit/release bitwise234개·coverage117회, 소규모 memcheck/racecheck, SM90a/SM100a object compile을 통과했다. 해당 장비 runtime과 multi-GPU는 미검증이다.

대표 mixed native shape는 기존보다30–52% 느리며,163register/24,584B shared의 resource envelope가 남는다. 같은 SM에 역할을 배정한 사실을 동시 resident 실행/serving 이득으로 주장하지 않는다. Native gate 탈락으로 Rust recorder·serving에 연결하지 않았다. PR04 전체 완료도 아니다. 이후는 단순 CTA 상수 변경보다 phase별 resource 및 native precision/backend 계약을 함께 다뤄야 한다.

## Probability storage compaction (2026-09-14)

[Native resource batch](../../benchmarks/results/20260914-compact-mixed-native/README.md): small-query/prefill 모두 exponential FP32 저장을 재사용하고 PV operand load 때 BF16 round를 적용했다. Direct와 POD를 분리 비교했다.234개 bitwise·117 coverage, bounded sanitizer, SM90a/SM100a compile은 통과했다. Direct shared6144→4096B, POD24584→16392B로 감소했다.

Direct native는 대표 shape에서1.6–3.7% 낮지만 긴 prefill은 거의 동률이며, compact POD는159register/3CTA 상한과 주요 mixed 회귀가 남는다. Native 저장 규약 후보만 보존하고 Rust/serving에 연결하지 않았다. 작은 native 이득을 serving 완료로 승격하지 않는다. 다음은 phase별 live register·native backend precision 계약을 함께 다루는 영역이다.

## Cache 결합 후 attention 우선순위 재확인

[Composed profile](../../benchmarks/results/20260914-prefix-cache-profile/README.md): 고유 prompt에서 prefill/mixed는 선택 graph span의83.8%, mapped attention은 kernel 누적시간의31.1%다. 공유 prefix에서도 mapped attention이 최대 단일 kernel family다. Warmup 파일쓰기 공백을 제거해 재측정했고 profiler/client 영향이 남는 gap을 순수 CPU 병목으로 주장하지 않는다. 다음 batch는 query 작업 mapping, K/V reuse, resource-bounded native dispatch를 함께 다룬다. 기존 POD CTA 상수/시간 threshold 반복은 하지 않는다. 실제 register/shared usage·exact causal/page/nonfinite native gate 이후 retained graph·serving 검증으로 진행한다. 상세 수치 계약과 미완료 범위는 profile 문서를 따른다.

## Query reuse native batch

[구현·검증 결과](../../benchmarks/results/20260914-query-reuse-native/README.md): 기존 task map에서 인접8-query 작업을16-query로 묶고 K/V 재사용·compact probability 저장·audit coverage를 연결했다.153 audit+153 graph replay 비교가 bitwise 일치하고 bounded memcheck/racecheck 및 SM90a/SM100a 컴파일이 통과했다. Serving capacity512의 긴 mixed prefill은 compact16이9–11% 빠르나 짧은 suffix는동률, pure prefill은약2%에 그친다. Register/shared envelope가 커져 shape별 효과를 구분한다. 다음은 명시적 retained-model option과 source/catalog identity, 모델·cache·stop/cancel 검증 후 실제 serving 비교다. 현재 production release와 기본값은 변경하지 않았다.

## Query reuse 모델·serving 통합 결과

[모델 통합](../../benchmarks/results/20260914-query-reuse-model/README.md)과 [matched serving](../../benchmarks/results/20260914-query-reuse-serving/README.md): C ABI recorder→Rust retained owner→명시적 serving env 옵션까지 연결했고 source/catalog identity 및 cache·paired 호환성을 유지했다.2,359,296 BF16 logits bytes가 일치하고 serving retained Riley3,072건·stop96건·취소96건·복구96건 검증을 통과했다. 그러나 동일 직전 composed cache-on 대비 throughput은 공유prefix−0.26%,고유prompt+0.96%로 유의미한 개선을 확인하지 못했다. `RILEY_MIXED_QUERY_REUSE`는 기본 비활성이고 native의 유리한 mixed shape로 전체 성공을 대체하지 않는다. 동일 query tile의 작은 변형을 이어가지 않고, pure-prefill 자원 사용과 retained 실행 pipeline 중 전체 workload에 영향을 주는 다음 구조를 선택한다. 기존 POD/수치 quality 실패와 미완료 hardware qualification은 그대로 남는다.
