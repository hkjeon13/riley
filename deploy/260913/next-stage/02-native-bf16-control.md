# PR-N02 — native BF16 대조군과 첫 가능성 판정

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. **Qwen2.5-3B projection shape의 deterministic synthetic prepared-GEMM control은 추가됐지만, SM89 GPU 실행·native-BF16 품질 gate·full-model 측정은 아직 없다.** strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N01의 로컬 artifact와 controller는 준비됐지만 원격 receipt 및 native-BF16 품질 기준은 아직 보류 상태다. 강한 native 대조군 없이 전용 kernel과 논문 후보만 비교하는 문제를 막는다. 라이브러리 연결은 대표 operator/layer harness까지만 하며 전체 모델 통합은 N06에서 결정한다.

## 변경 묶음

1. 기존 prepared GEMM에 cuBLASLt 또는 지원되는 CUTLASS BF16 경로를 연결한다. weight packing과 workspace 수명을 재사용하며 매 token allocation/heuristic 실행을 피한다.
2. 기존 FlashInfer/native attention adapter에서 SM89·BF16·GQA·paged KV·mask 조합을 확인하고 native profile 대조군을 구성한다. 현행 XQA wrapper의 major 9/10/12 제한 때문에 이를 4090 경로로 강제하지 않는다.
3. Rust/C ABI에서 stream·device·dtype·shape·layout·owner 수명·오류·graph capture 계약을 검증한다. planner와 capture 실행을 분리하고 반복 metadata 갱신 비용은 timed 경로에 포함한다.
4. 실제 reference tensor로 operator 및 제한된 layer 실행을 만들고, kernel 시간과 준비/merge/packing 포함 비용을 각각 출력한다.

## 대표 shape와 범위

3B config 확인 후 QKV K2048/N2560, gate+up K2048/N22016과 attention head_dim128/GQA8을 사용한다. decode M1/8/32 및 작은 prefill chunk128 중 소수 대표값을 선택한다. 135M strict 회귀도 확인한다. 처음부터 전 shape autotuner를 만들지 않는다.

## 검증과 완료

N01 수치 기준, page/mask 경계, 반복·capture 실행, 할당/해제 및 scratch peak를 검증한다. `strict reference / native control`의 차이와 지원 범위를 표로 남긴다. 결과를 135M serving 승리로 해석하지 않는다.

산출물은 immutable library/toolkit/SM artifact 정보, reference/native raw, 비용 비중과 N03~N05 착수 권고다. 아직 full-model timing이 없으면 Amdahl 계산의 입력을 추정/미측정으로 표시한다.

## 중단·롤백

native 경로가 수치 gate를 실패하면 원인이 명확한 수정 1회 후 해당 경로를 보류한다. 기존 strict 반올림을 보정하는 대형 residual 경로를 다시 만들지 않는다. opt-in adapter를 끄면 기존 경로로 돌아가며 대조군 구축 자체는 성능 승격이 아니다.
