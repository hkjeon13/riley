# PR 00 — 작업·리뷰 계약

**상태:** Planned  
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

## 설계 결정
대안과 선택 이유는 무엇인가?

## 검증
정확성, 성능, 메모리, 실패 경로를 어떻게 확인했는가?

## 결과
측정 수치와 artifact는 무엇인가?

## 롤백
문제가 생기면 어떤 flag/commit/interface로 되돌리는가?
```

## Merge 필수 조건

- [ ] `cargo fmt --check`
- [ ] `cargo clippy`에서 새 warning 없음
- [ ] unit/integration test 통과
- [ ] 공개 API 변경 시 문서와 예제 갱신
- [ ] `unsafe` 추가 시 safety invariant 주석
- [ ] CUDA 호출 추가 시 오류와 stream semantics 검증
- [ ] allocation 추가 시 lifetime과 ownership 설명
- [ ] 성능 주장 시 before/after raw result 첨부
- [ ] 범위를 벗어난 후속 과제는 issue 또는 다음 deploy 문서에 남김

## Correctness 우선 원칙

최적화 전에 반드시 느리더라도 명확한 reference path가 있어야 한다.

```text
Reference implementation
        ↓ parity
Optimized backend
        ↓ parity
Fused/custom kernel
```

다음은 금지한다.

- reference 없이 처음부터 fused kernel 작성
- 허용 오차를 결과에 맞춰 사후 확대
- 한두 번의 최저 latency만 제시
- 다른 dtype, batch, prompt 길이로 엔진 비교
- warm/cold 결과 혼합

## 성능 보고 최소 항목

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
optimized
experimental-fused
```

회귀 발생 시 runtime flag 또는 compile feature로 reference path를 선택할 수 있어야 한다. 안정화 후에만 이전 path 제거를 검토한다.

## 완료 기준

- [ ] PR 템플릿 또는 동등한 문서가 저장소에 존재
- [ ] benchmark 결과 저장 위치가 합의됨
- [ ] unsafe/FFI 검토 규칙이 명시됨
- [ ] 단계 문서의 승인 기준을 merge gate로 사용하기로 합의

[목차](README.md) | [다음 →](01-baseline-and-reproducibility.md)
