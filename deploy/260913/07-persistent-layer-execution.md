# PR 07 — persistent task graph의 한 layer 실행 실험

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

operator별 kernel 경계를 넘어 task 의존성과 on-chip 데이터 재사용을 묶는 구조의 실효성을 판정한다.

## 의존성과 변경 위치

선행: 01, 05; 03 attention 계약 재사용.

예상 수정 위치: CUDA task executor, riley-runtime layer plan/ABI, correctness 및 full-model 호출 경계. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. attention+MLP 한 layer를 task DAG로 낮춘다.
2. 정적 task 순서와 bounded device task queue 중 한 방식을 정하고 completion counter를 구현한다.
3. tile buffer 수명·barrier·scratch 예약을 연결한다.
4. 기존 model loop가 선택한 layer backend를 호출하도록 통합한다.

## 범위 경계

GPU 내부 전체 serving scheduler, 모든 모델 자동 lowering, 다중 GPU는 제외한다. Mirage compiler 전체 재작성 대신 기존 artifact 활용 가능성을 먼저 확인한다.

## Correctness·수명 계약

barrier 참여 thread/CTA와 자원 제한을 명시해 deadlock을 막는다. 실패한 persistent kernel 이후 owner를 조기 회수하지 않는다.

## 검증과 하드웨어 skip

한 layer numerical·race/memory 검증, task dependency 오류 검출, 전체 model에서 반복 호출. 지원 GPU별 실행; 4090 미지원이면 compile/CPU plan 검증 후 runtime skip으로 다음 개발 진행.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

이 PR 완료는 architecture feasibility 판정이다. 고정 batch demo나 kernel 가속으로 serving 목표를 완료 처리하지 않는다. 전 layer serving 확장은 [PR19](19-persistent-serving-integration.md)에서 진행한다. 장비 부재로 runtime 검증이 skip되어도 비활성 경로 구현은 진행할 수 있다. 실제 검증 전 기본값으로 승격하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

기존 layer backend 선택으로 복귀. device executor가 drain된 뒤 자원 반환.

## 연구 근거

[MPK](https://arxiv.org/html/2512.22219v2), [Ada-MK](https://arxiv.org/html/2605.11581v1), [ThunderKittens](https://arxiv.org/abs/2410.20399). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
