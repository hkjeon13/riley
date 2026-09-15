# PR-N05 — RMSNorm–GEMM 대수적 재배치 검증

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N02 완료 및 norm 중간 tensor/직렬 의존성 제거의 비용 상한이 실험 가치가 있다는 근거. [Mirage 조사](../31-method-feasibility-research.md)의 대수적 변환을 제한된 두 연산 묶음에서 평가한다. 큰 persistent kernel 구현을 재개하지 않는다.

## 변경 묶음

1. `r=sqrt(mean(x²)+epsilon)`에 대해 `((x*g)/r)W=((x*g)W)/r` 구조로 norm 통계와 matmul의 의존성을 분리한다.
2. 입력 fragment의 gamma 적용과 epilogue row scale을 구현 가능한 native 경로에서 평가한다. bias는 scale 뒤, RoPE는 projection 뒤의 모델 계약을 유지한다. SwiGLU 비선형을 넘어 이동하지 않는다.
3. QKV와 gate/up 두 묶음에서 분리 RMSNorm+강한 GEMM 대조군과 비교한다. weight 전체의 GPU 추가 사본을 전제로 하지 않으며 scratch·동기화·register 비용을 합산한다.

## 검증·진행 기준

M1/8/32 및 prefill128 중 대표값, 작은/큰 norm·outlier·near-zero·실제 모델 tensor를 포함한다. 중간 BF16 cast가 달라져 strict bitwise 일치는 요구할 수 없지만 N01 native profile 품질 gate를 통과해야 한다.

norm 제거의 이론적 상한과 실제 순이득을 함께 기록한다. weight 읽기가 지배하여 전체 기여가 작거나, gamma/epilogue 비용이 더 크면 탈락시킨다. 통과 후보만 N06에 연결한다. Mirage compiler 전체 도입은 범위 밖이다.

## 중단·롤백

두 실행 방식과 명확한 수정 한 차례의 범위에서 판정한다. 품질 실패 시 기준을 완화하지 않는다. 분리 norm+native GEMM으로 복귀한다. 산출물은 수치/비용/메모리 표와 채택 또는 보류 근거다.
