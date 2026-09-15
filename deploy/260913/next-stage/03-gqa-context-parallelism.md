# PR-N03 — GQA와 긴 context 작업 분할의 가능성 검증

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N02 native 대조군 통과. [LeanAttention/FlashInfer 조사](../31-method-feasibility-research.md)를 바탕으로 KV의 head 간 재사용과 작은 query의 SM 병렬성을 함께 평가한다. bandwidth 병목이 이미 입증됐다고 가정하지 않는다.

## 변경 묶음

1. 지원되는 native Tensor Core GQA 경로와 기존 native decode를 동일한 paged KV fixture에서 선택 가능하게 한다.
2. library가 제공하는 context split/planning을 우선 사용한다. 분할 없음과 제한된 workload 기반 정책을 비교하고 기존 strict split 실패를 그대로 재포장하지 않는다.
3. FP32 partial state와 stable merge, metadata 갱신 및 graph replay를 하나의 측정 경로로 묶는다. workspace를 재사용하고 작은 context는 대조군으로 fallback한다.

## 제한된 실험

우선 (batch, context)=(1,2048),(1,16384),(8,8192),(32,2048). 전체 모델 상당 KV 예산을 산정하고 초과하면 동일 조건을 줄인다. GQA 선택만 변경, split만 변경, 둘을 결합한 ablation을 포함하되 무제한 tile 탐색은 하지 않는다.

## 검증·진행 기준

수치 기준, 빈/부분 page, 서로 다른 길이, mask, 반복 실행과 scratch 상한을 통과해야 한다. merge 및 host metadata를 포함한 순이득을 판정한다. 작은 shape 회귀는 명시적 fallback으로 제거한다.

operator 이득이 반복 변동보다 크고 전체 비용 감소 가능성이 있으면 N06 후보로 선정한다. 전체 비용 비중이 미측정이면 `operator 유망`까지만 기록한다. 단일 kernel 배수를 serving 예상 배수로 쓰지 않는다.

## 중단·롤백

기본 두 실행 방식과 원인이 명확한 수정 한 차례 후 순이득이 없거나 품질이 실패하면 보류한다. N02 native 경로로 rollback한다. 산출물은 shape별 raw·scratch·오차·기여도 판정이며 새 full-model 구현은 이 PR 범위가 아니다.
