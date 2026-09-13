# PR 19 — Persistent layer 실행의 전 layer·serving 통합

상태: **계획만 작성 / 미구현**. [공통 계약](README.md)을 따른다.

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
