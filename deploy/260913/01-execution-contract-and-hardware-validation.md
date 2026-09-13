# PR 01 — 실행 backend 계약과 하드웨어 검증 기반

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

장비와 backend가 늘어도 동일한 실행·정확성 계약을 적용하고, 미지원 장비 때문에 개발을 막거나 검증을 통과로 오인하지 않게 한다.

## 의존성과 변경 위치

선행: 없음.

예상 수정 위치: crates/riley-cuda, crates/riley-runtime, benchmarks의 capability·결과 기록 경계. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. device/architecture/instruction capability와 GPU 수·peer topology를 조회하고 backend 요구조건과 대조한다.
2. backend plan identity에 model revision, dtype, KV layout, shape 범위, device 및 workspace 요구를 포함한다.
3. 실행 결과를 pass/fail/skip으로 분리하고 required/observed capability·skip 이유·실제 선택 backend를 기록한다.
4. 현행 exact 경로와 신규 numerical/quantized 경로의 검증 기준을 분리하여 결과를 보기 전에 고정한다.

## 범위 경계

새 attention kernel, 여러 GPU 실제 실행, 모델 지원 확장은 이 PR에 넣지 않는다.

## Correctness·수명 계약

기존 경로의 수치 기준을 약화하지 않는다. capability가 부족하면 명시적으로 거부하거나 검증된 fallback만 선택한다. fallback 성공은 전용 backend 성공이 아니다.

## 검증과 하드웨어 skip

capability 조합 및 fallback 선택, plan identity mismatch, required-GPU CI에서 잘못된 skip을 실패로 처리하는 테스트. 4090 기존 실행 smoke. 없는 SM90/Blackwell/multi-GPU 실행은 skip하고 가능한 compile-only 결과를 별도로 기록한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

공통 계약·검증 결과 스키마가 사용되고 기존 backend 동작이 유지되면 기반 PR 완료. 이 PR 자체에 성능 개선 배수를 요구하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

기존 backend 선택으로 복귀하고 새 결과 필드는 호환 가능한 부가 필드로 유지한다.

## 연구 근거

[FlashInfer API](https://docs.flashinfer.ai/api/attention.html), [TRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/features/attention.html). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
