# PR 19 — Persistent layer 실행의 전 layer·serving 통합

상태: **Post-attention phase DAG 진단 구현·serving screen 완료 / 전체 layer DAG 미완료**. [공통 계약](README.md)을 따른다.

## 문제와 가설

한 layer의 실험 결과를 동적 batch와 streaming을 갖춘 full-model 실행으로 확장해 구조 개선의 실제 효과를 판정한다.

## 의존성과 변경 위치

01, 02, 07. 07의 task·buffer 계약이 구현되어야 한다. 장비 부재로 07 GPU 검증이 skip되어도 비활성 경로 구현은 진행할 수 있으나 기본값 승격은 실제 검증 후다.

예상 위치: runtime execution plan, scheduler/server 경계, CUDA/transport adapter 및 benchmarks. 구체 파일은 실제 checkout에서 확인한다.

## 하나의 optimization batch

1. 지원 dense 모델 하나의 전체 layer 실행 plan과 workspace 수명을 연결한다.
2. iteration마다 batch metadata를 갱신하고 graph/plan identity를 검증한다.
3. 02 ticket에 GPU completion을 연결해 admission·EOS·cancel을 처리한다.
4. 기존 operator graph와 persistent plan의 명시적 dispatch 및 fallback을 통합한다.

## 범위 경계

GPU 내부 request scheduler 전체 이식, 무한 persistent loop, 모든 모델의 자동 compiler lowering을 제외한다. PR07에서 고른 executor를 iteration 단위로 확장한다.

## Correctness·수명 계약

바뀐 active row가 이전 iteration의 KV/activation을 읽지 않아야 한다. CPU commit 전에 owner를 재활용하지 않는다. fallback은 in-flight 실행이 완료된 경계에서만 허용한다. 수치 계약은 PR01을 적용한다.

## 검증과 하드웨어 skip

동적 admission·ragged 길이·EOS·cancel·최대 context·연속 요청 slot 재사용을 full-model로 검사한다. memory/race 검사와 long soak를 수행한다. 4090 미지원 경로는 compile/CPU 검사 후 GPU 실행 skip; 지원 장비에서 operator graph와 동일 workload serving 비교를 수행한다.

## 완료·승격 기준

단일 layer 가속과 별도로 throughput·TTFT/TPOT·P95/P99·실패율을 판정한다. GPU 내부 scheduler 비용이나 workspace가 이득을 상쇄하면 기본 backend로 승격하지 않는다. 최종 목표는 PR16에서 판정한다.

구현·실장비 검증·성능 승격 상태는 별도로 기록한다. 실패한 테스트를 skip으로 바꾸지 않는다.

## 롤백

신규 iteration부터 기존 graph 경로로 복귀하고 persistent ticket을 drain한다. timeout만으로 GPU 종료를 가정하지 않는다.

## 연구 근거

[MPK](https://arxiv.org/html/2512.22219v2), [Ada-MK](https://arxiv.org/html/2605.11581v1) — Riley 적용 설계이며 논문의 개선 배수를 예상 성능으로 사용하지 않는다.

## FFN 단계 DAG 구현·측정

[구현 및 비교 보고](../../benchmarks/results/20260913-persistent-ffn-native/README.md): V7의 gate/down/residual-norm body를 재사용하고 finite cooperative grid의 두 barrier로 연결했다. Occupancy·device 검증, graph replay, inactive row·invalid plan 검증을 포함한다. Native memcheck/racecheck0, SM89 runtime 및 SM90a/SM100a compile 통과(후자 runtime 장비 부재 skip). 별도 모델 source에서30-layer pure-decode FFN을 연결했고 mixed/prefill은 기존 경로다. 생성1,024토큰과 full-logit12,582,912개 값이 bitwise 같고 whole-model memcheck0이다.

C32 natural 두 역순에서 V7/후보/vLLM throughput은10,440.7/10,715.1/11,793.7 tokens/s, median TPOT는2.920/2.850/2.381ms다. V7 대비2.63% 개선, vLLM 대비9.15% 부족이다. 기본값 승격은 보류한다. 아직 attention+MLP 전체 layer DAG, persistent prefill, async ticket 통합, 높은 concurrency·장시간 안정성 검증이 남아 있다. FFN 내부 global intermediate는 유지되므로 on-chip 전체 재사용을 구현했다고 주장하지 않는다. 다음 batch는 attention/projection 경계와 scratch 수명을 함께 다루는 구조 확장 타당성 검토다.

## Projection부터 FFN까지 확장 결과

[Post-attention batch 보고](../../benchmarks/results/20260913-persistent-post-attention/README.md): attention 출력 projection·residual/norm·FFN을 하나의 finite cooperative grid로 연결하고 partial scratch 재사용 전 barrier를 추가했다. Native24 replay와 full-model1,024토큰/12,582,912 logits가 bitwise 일치하며 memcheck/racecheck0이다. SM90a/SM100a compile 통과, runtime 장비 부재 skip이다.

같은 C32 natural 두 역순에서 V7/FFN-only/확장 후보/vLLM은10,482.0/10,610.7/10,711.1/12,029.1 tokens/s다. 후보는 직전 구조보다0.95%, V7보다2.19% 빠르지만 vLLM보다10.96% 느리다. 작은 추가 이득의 통계적 유의성은 입증하지 않았다. 기본값 승격을 보류한다. 다음 영역은 기존 attention의 score/value 연산을 유지하면서 ragged context 작업 분배와 score scratch·grid barrier를 함께 연결하는 구조 확장이다. Attention 자체와 mixed/prefill·async ticket 통합은 여전히 미완료다.
