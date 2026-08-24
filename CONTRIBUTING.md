# rustinfer 기여 계약

`rustinfer`는 [단계별 실행 계획](deploy/README.md)을 번호 순서대로 구현한다. 각 PR의 선행 조건과 완료 기준은 선택적인 참고사항이 아니라 merge gate다. 공통 계약의 원문은 [PR 00](deploy/00-pr-contract.md)에 있다.

## PR 범위

- 하나의 PR은 한 가지 질문에 답하고 독립적으로 검증·롤백할 수 있어야 한다.
- hand-written production diff는 대체로 200~800줄을 권장한다. 테스트, generated fixture, lockfile, vendored source는 별도로 설명한다.
- 다음 단계가 필요로 하는 interface는 정의할 수 있지만 다음 단계의 구현을 미리 포함하지 않는다.
- correctness와 aggressive optimization, 또는 서로 다른 두 subsystem은 별도 PR로 나눈다.
- 선행 단계가 merge gate를 통과하지 않았으면 다음 단계 PR을 시작하지 않는다.

모든 PR은 저장소의 템플릿을 사용해 문제, 범위, 비범위, 의미 보존 등급, 설계 결정, 검증, 결과, 롤백을 설명한다. 아직 Cargo workspace가 없는 문서 전용 초기 단계처럼 검증 명령이 적용되지 않으면 체크를 지우는 대신 `해당 없음`의 이유를 남긴다.

## 의미 보존 등급

수학적 변환이나 최적화는 다음 중 하나를 선언한다.

- `E0`: 실수 연산에서 동일한 대수적 재구성. reference parity와 dtype별 수치 안정성을 검증한다.
- `E1`: 목표 분포를 보존하는 확률적 알고리즘. 분포 검증과 request별 RNG 격리·snapshot/restore가 필요하다.
- `A1`: 오차 또는 품질 예산이 있는 근사. exact fallback, opt-in flag, error-quality-latency 결과가 필요하다.
- `M1`: calibration, 학습, architecture 변경 등 모델 의미를 바꾸는 방법. production core와 분리된 연구 트랙에서 다룬다.

inference 의미와 성능 비교에 영향이 없는 문서·빌드 변경은 PR 템플릿의 항목을 생략하지 않고 `N/A`와 근거를 기록한다. allocator, scheduler, launch orchestration처럼 수학적 변환 없이 성능을 비교하는 exact systems 변경은 benchmark의 `semantic_class`를 `reference`로 기록하고, 무엇을 동일하게 보존하는지 PR에 설명한다. `E0`, `E1`, `A1`, `M1`의 상세 검증 계약은 [PR 00](deploy/00-pr-contract.md#수학적-최적화의-의미-보존-등급)을 따른다.

## `unsafe`와 CUDA FFI

`unsafe`와 raw CUDA 호출은 전용 crate 또는 module의 작은 경계에 가둔다. safe wrapper가 외부에 노출되기 전에 다음 invariant를 설명하고 검증한다.

- host/device pointer의 유효 범위, alignment, 크기
- mutable aliasing이 발생하지 않는 근거
- allocation ownership과 drop 책임
- kernel 또는 async copy 완료 전 buffer가 해제되지 않는 근거
- 호출이 사용하는 CUDA stream과 필요한 event/order 관계
- CUDA 오류를 Rust 오류로 전파하는 방식
- thread 간 공유 가능 여부와 수동 `Send`/`Sync` 구현의 근거

각 `unsafe` block에는 호출자가 지켜야 할 조건이 아니라 해당 block이 안전한 이유를 가까운 `// SAFETY:` 주석으로 남긴다. device pointer를 장기 식별자로 `usize`에 보관하지 않고, stream 완료가 보장되지 않은 resource를 `Drop`에서 즉시 해제하지 않는다.

수동 `Send`/`Sync`, 새로운 FFI surface, 새로운 async allocation lifetime은 PR 설명에서 별도 검토 항목으로 표시한다.

## 검증과 결과

관련 항목을 모두 실행한다.

```text
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace
```

CUDA 또는 GPU가 필요한 검증은 CPU-only gate와 분리하고 실행 환경을 결과에 기록한다. 공개 API를 바꾸면 문서와 예제를, allocation을 추가하면 ownership/lifetime 설명을, CUDA 호출을 추가하면 오류와 stream semantics 검증을 함께 제출한다.

성능 주장은 [benchmark artifact 계약](benchmarks/README.md)을 따라 raw 결과와 환경을 남긴다. 최저값 한 번이 아니라 동일 조건의 반복 측정과 median/p95를 기본으로 사용한다.

## 롤백

새 backend와 최적화는 안정화 전까지 reference 또는 exact path로 돌아갈 수 있어야 한다. compile feature, runtime flag, backend interface 중 실제 롤백 경계를 PR에 적는다. `A1`은 기본 비활성이고, `E1`도 greedy 및 분포 계약을 통과하기 전에는 기본 경로가 될 수 없다.
