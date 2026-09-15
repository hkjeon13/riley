# 후속 PR 묶음 — Hopper·Blackwell·multi-GPU

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 착수 조건

1차4090 평가의 필수 선행 조건이 아니다. N02의 native ABI/수치 계약을 재사용한다. 세 하드웨어 변경을 한 PR에 넣지 않고 아래 H01/H02/D01로 각각 구현·검증한다. 장비가 없으면 runtime 검사만 skip하고 미검증을 표시한다.

## H01 — 기존 Hopper FA3의 실제 실행 검증

기존 recorder/CLI/owner 연결을 재사용한다. (1) capability/fixture 고정, (2) SM90a native·graph·수명/오류 및 수치 gate, (3) 같은 Hopper의 모델·serving 비교를 묶는다. 컴파일 통과 및4090 reject를 Hopper 실행 증거로 재사용하지 않는다. 실패 시 기존 native backend로 복귀한다.

## H02 — Blackwell backend 선택과 native artifact

(1) SM100/SM120 등 정확한 target·toolchain·지원 feature 확인, (2) FA4/XQA 등 후보의 AOT/native ABI·workspace·graph 연결, (3) 실제 장비의 수치·모델·serving을 묶는다. Python DSL 사용과 Python runtime 의존을 구분한다. 모든 Blackwell을 동일 capability로 간주하지 않는다. BF16를 먼저 평가하고 FP8/FP4는 별도 수치 계약 PR이다. 미지원 target은 명시적으로 거절하며 범용 native fallback을 보존한다.

## D01 — 두 GPU tensor parallel

(1) topology와 NCCL·shard/collective 계약, (2) rank별 weight/KV/stream 수명 및 오류·취소 동작, (3) 실제 두 GPU correctness·memory·통신 포함 serving을 묶는다. NVLink를 가정하지 않고 PCIe 경로도 식별한다. rank별 peak와 총 메모리를 기록하고 모든 비교 엔진에 같은 GPU 수·topology를 사용한다. 4090의 20GB 단일 GPU 계약을 임의로 다중 GPU 총량 계약으로 해석하지 않는다. 실제 장비 단계에서 예산을 명시한다.

## 완료와 보류

각 PR마다 native/model/serving 증거를 분리한다. 1GPU 결과를 multi-GPU scaling으로 계산하지 않는다. 통신 비용 때문에 TP가 느리면 지원 가능성과 성능 실패를 나눠 기록한다. P/D 분리·통신 연산 중첩은 D01 기준 후 별도 기여도 gate를 통과해야 착수한다.
