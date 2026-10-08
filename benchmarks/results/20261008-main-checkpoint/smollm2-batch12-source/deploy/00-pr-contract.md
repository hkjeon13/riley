# PR 00 — 작업·리뷰 계약

**상태:** Active

**선행 조건:** 없음  
**다음:** [PR 01 — 기준선과 재현성](01-baseline-and-reproducibility.md)

[목차](README.md) | [다음 →](01-baseline-and-reproducibility.md)

## 목적

향후 구현을 작은 PR로 유지하기 위한 공통 규칙을 확정한다. 이 PR은 제품 코드가 아니라 개발 계약과 템플릿만 추가한다.

## PR 크기 원칙

- 하나의 PR은 한 가지 질문에 답한다.
- 권장 hand-written production diff는 약 `200~800`줄이다.
- 테스트 코드는 production code와 비슷하거나 더 커도 된다.
- generated fixture, lockfile, vendored source는 별도 집계한다.
- 파일 수가 많아져도 하나의 interface 변화만 다루면 허용한다.
- 두 개 이상의 독립적 subsystem을 건드리면 분리한다.

크기보다 중요한 것은 **독립적인 검증과 롤백 가능성**이다.

## 모든 PR에 필요한 설명

```markdown
## 문제
현재 무엇이 부족하거나 잘못되었는가?

## 범위
이번 PR이 정확히 무엇을 바꾸는가?

## 비범위
의도적으로 하지 않는 것은 무엇인가?

## 의미 보존 등급
E0, E1, A1, M1 중 무엇이며 왜 그런가?

## 설계 결정
대안과 선택 이유는 무엇인가?

## 검증
정확성, 성능, 메모리, 실패 경로를 어떻게 확인했는가?

## 결과
측정 수치와 artifact는 무엇인가?

## 롤백
문제가 생기면 어떤 flag/commit/interface로 되돌리는가?
```

## 수학적 최적화의 의미 보존 등급

수학적 치환, 확률적 가속, 근사 알고리즘은 동일한 검증 기준을 사용할 수 없다. 관련 PR은 반드시 다음 등급 중 하나를 선언한다.

### `E0` — Exact algebraic transformation

실수 연산에서는 동일하고, 차이는 floating-point 계산 순서에서만 발생하는 변환이다.

예:

- online softmax
- associative partial reduction
- residual과 norm의 합법적인 fusion
- 대각·직교 좌표변환을 quantization 이전 full precision에서 적용

필수 검증:

- reference parity
- 극단값과 여러 reduction 순서
- dtype별 tolerance
- token-level 회귀

### `E1` — Distribution-preserving stochastic algorithm

실행 경로와 RNG 소비량은 달라질 수 있지만 목표 sampling distribution을 보존하는 알고리즘이다.

예:

- rejection sampling으로 보정된 speculative decoding

필수 검증:

- 수락·거절 공식과 residual distribution 구현 검토
- request별 RNG 격리와 snapshot/restore
- greedy 경로 exact match
- sampling distribution에 대한 통계 검정
- 고정 seed 결과의 정의와 문서화

### `A1` — Bounded approximation

출력이 원본과 달라질 수 있으나 명시적인 error budget 또는 품질 예산으로 제어하는 근사다.

예:

- omitted softmax mass 상한을 이용한 KV page pruning
- 저랭크 weight 또는 KV 압축
- quantized inference

필수 검증:

- 근사 파라미터와 단위 공개
- exact fallback
- error/quality와 latency의 curve
- feature flag와 기본값
- 사용자 응답 또는 metric에서 근사 사용 여부 식별

### `M1` — Model-changing method

재학습, distillation, calibration 또는 architecture 변경이 필요한 방법이다.

이 등급은 기본 inference runtime PR에 섞지 않고 별도 연구 트랙으로 둔다.

## Merge 필수 조건

- [ ] `cargo fmt --check`
- [ ] `cargo clippy`에서 새 warning 없음
- [ ] unit/integration test 통과
- [ ] 공개 API 변경 시 문서와 예제 갱신
- [ ] `unsafe` 추가 시 safety invariant 주석
- [ ] CUDA 호출 추가 시 오류와 stream semantics 검증
- [ ] allocation 추가 시 lifetime과 ownership 설명
- [ ] 성능 주장 시 before/after raw result 첨부
- [ ] 최적화 PR은 `E0`·`E1`·`A1`·`M1` 중 하나를 선언
- [ ] `E0`은 reference parity와 수치 안정성 결과 첨부
- [ ] `E1`은 분포 보존 근거와 RNG 검증 첨부
- [ ] `A1`은 error budget, exact fallback, opt-in flag 첨부
- [ ] 범위를 벗어난 후속 과제는 issue 또는 다음 deploy 문서에 남김

## Correctness 우선 원칙

최적화 전에 반드시 느리더라도 명확한 reference path가 있어야 한다.

```text
Reference implementation
        ↓ parity or distribution contract
Optimized backend
        ↓ parity, error budget, or statistical validation
Fused/custom/approximate path
```

다음은 금지한다.

- reference 없이 처음부터 fused kernel 작성
- 허용 오차를 결과에 맞춰 사후 확대
- 근사 알고리즘을 exact optimization처럼 표현
- 분포 보존 알고리즘을 소수 prompt의 token 일치만으로 검증
- 한두 번의 최저 latency만 제시
- 다른 dtype, batch, prompt 길이로 엔진 비교
- warm/cold 결과 혼합

## 성능 보고 계약

End-to-end 성능 주장에는 다음 항목을 기록한다.

- GPU 모델과 compute capability
- CPU, RAM, OS
- NVIDIA driver와 CUDA toolkit/runtime
- model/checkpoint revision
- dtype와 quantization
- batch/concurrency
- prompt/output length
- warm-up 횟수와 측정 횟수
- median, p95, 가능하면 p99
- TTFT, TPOT/ITL, throughput, VRAM
- profiler trace 또는 raw CSV 위치
- 의미 보존 등급
- 근사 또는 speculative parameter
- exact/reference fallback 결과

Version-controlled raw 결과는 `benchmarks/results/<YYYYMMDDTHHMMSSZ>-<implementation-id>-<workload>-<run-id>/`에 append-only로 저장한다. 디렉터리별 필수 파일, end-to-end와 microbenchmark의 scope별 필드, 대형 artifact 보존 방식과 비교 가능성 규칙은 [`benchmarks/README.md`](../benchmarks/README.md)를 따른다.

Microbenchmark는 공통 환경·revision·의미 등급과 operation/shape/layout/backend별 측정값을 기록하고, 적용되지 않는 end-to-end 필드는 `null`과 이유를 남긴다. Microbenchmark 결과만으로 end-to-end 개선을 주장하지 않는다.

## `unsafe` 정책

- CUDA FFI와 raw pointer 조작은 전용 crate/module로 제한한다.
- safe wrapper는 ownership, aliasing, stream ordering을 보장해야 한다.
- device pointer를 임의의 `usize`로 장기간 보관하지 않는다.
- async kernel이 끝나기 전에 buffer가 drop되지 않는다는 근거가 있어야 한다.
- `Send`/`Sync` 수동 구현은 별도 검토 대상으로 취급한다.

## Feature flag와 롤백

새 backend나 최적화는 처음에는 선택 가능해야 한다.

```text
reference
optimized-exact
experimental-distribution-preserving
experimental-approximate
```

회귀 발생 시 runtime flag 또는 compile feature로 reference path를 선택할 수 있어야 한다. 안정화 후에만 이전 path 제거를 검토한다.

`A1` path는 명시적인 정책 결정 전까지 기본값이 될 수 없다. `E1` path도 exact greedy와 분포 검증이 완료되기 전에는 기본값으로 승격하지 않는다.

## 완료 기준

- [x] PR 템플릿 또는 동등한 문서가 저장소에 존재
- [x] benchmark 결과 저장 위치가 합의됨
- [x] unsafe/FFI 검토 규칙이 명시됨
- [x] 의미 보존 등급과 등급별 검증 방식이 합의됨
- [x] 단계 문서의 승인 기준을 merge gate로 사용하기로 합의

구현 artifact는 [PR 템플릿](../.github/pull_request_template.md), [기여 계약](../CONTRIBUTING.md), [benchmark artifact 계약](../benchmarks/README.md)이다. 이 단계는 merge 전까지 `Active`, merge 후 `Complete`로 전환한다.

[목차](README.md) | [다음 →](01-baseline-and-reproducibility.md)
