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

## 첫 릴리스 이후 확장 admission

[PR 17](deploy/17-extension-gates.md)에 열거된 아이디어는 자동 승인 목록이 아니다.
새 quantization, compression, sparse/selected attention, speculative/Jacobi
decode, MoE, SSM, multimodal 또는 그 밖의 확장은 구현 전에 별도의
admission-only PR을 먼저 병합한다. 그 PR은 production code, dependency, kernel,
runtime 동작이나 성능 주장을 추가하지 않고 다음 네 파일만 하나의 계약으로 만든다.
현재 v1 allowlist에 없는 track은 admission과 섞지 않고 먼저 checker/schema/docs를
확장하는 gate-version PR을 병합해야 한다.

- `deploy/extensions/registry.json`의 정렬된 allowlist entry
- `deploy/extensions/proposals/<extension-id>.json`
- `deploy/extensions/plans/<extension-id>.md`
- `benchmarks/extensions/contracts/<extension-id>.json`

registry entry의 `implementation_link_path`는 admission 시 `null`이다. proposal은
`approved-for-implementation` 상태, 닫힌 roadmap track/class 조합, 서로 다른
Git-tracked reference/fallback의 path와 실제 bytes SHA-256, experimental runtime
flag, 기본 off, rollback과 결과 disclosure를 선언한다. PR 17의 승인 질문 10개는
`approval_answers`의 닫힌 field로 모두 답하며, 기존 IR 표현 가능성, memory/운영
복잡도, FLOPs·serial depth·HBM traffic 중 하나 이상의 기대 절감 축과 end-to-end
가설을 생략할 수 없다. benchmark contract는 동일한 계약과 Git-tracked workload
path/hash, single-GPU의 실제 GPU·driver·CUDA·model ID·full 40-hex model
revision·dtype, concurrency, prompt/output 길이, 닫힌 sampling 설정, cold/warm 상태를
동결한다.
class는 exact systems의 `reference`와 `E0`/`E1`/`A1`/`M1`이며, track별 허용
조합과 공통 result schema에 실제 기록 가능한 track별 required
performance/quality metric path를 checker가 닫고 primary가 required set에
포함되는지 확인한다. E0 tolerance dtype은 비교 환경 dtype과 정확히 같고,
query-aware A1 omitted-mass
budget은 `[0, 1)` fraction이어야 한다. `M1`은 production core가 아닌 research
boundary만 허용한다. 현재 result schema에
적합한 quality field가 없는 class/track admission과 multi-GPU track은 schema
version-up 전까지 fail closed다. 자세한 형식과 명령은
[`deploy/extensions/README.md`](deploy/extensions/README.md)를 따른다.

PR CI는 full base SHA를 checker에 전달해 v1 append-only 전이, 기존 entry와 세
admission artifact의 불변성, rename source를 포함한 admission PR의 정확한 네 파일
diff를 검사한다. production crate의 새 experimental flag는 정확히 한 approved
implementation link와 일치해야 한다.
Markdown plan의 의미와 구현의 실제 runtime control flow는 reviewer가 직접 확인한다.

```text
python3 ci/check_extension_gates.py --base-revision <full-lowercase-base-sha>
python3 -m unittest discover -s ci/tests -p 'test_*.py' -v
```

admission은 구현 작업 시작만 승인한다. 후속 구현 PR은 승인된 ID/plan/contract/flag와
tracked source 및 `{path, sha256, test_id}` auto-discovered integration test를
implementation link로 결합한다. test ID는 direct top-level `#[test] fn`이며
hidden/cfg/ignored target은 허용하지 않는다. reviewer는 CPU workspace test가 해당 test를 실행하고 default-off,
flag-on, stable fallback을 실제로 검증하는지 확인한다. v1은 이 immutable
experimental link까지만 지원하며 이 link 자체는 성능 result 통과를 뜻하지 않는다.
stable 승격은 result receipt gate를 포함하는 schema v2 transition을 먼저 도입해야
하고 withdrawal이나 계약 수정도 같은 version-up 경계를 따른다.

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
