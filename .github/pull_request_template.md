## 문제

<!-- 현재 무엇이 부족하거나 잘못되었는지 설명합니다. -->

## 범위

<!-- 이번 PR이 바꾸는 한 가지 목적과 변경 범위를 적습니다. -->

## 비범위

<!-- 의도적으로 하지 않는 일과 후속 PR을 적습니다. -->

## 의미 보존 등급

<!-- reference, E0, E1, A1, M1 중 하나를 선택하고 근거를 적습니다. inference 의미와 성능 비교에 영향이 없는 변경만 N/A와 이유를 적습니다. exact systems 성능 변경은 reference로 분류합니다. -->

- 등급: `N/A`
- 근거:

## 설계 결정

<!-- 검토한 대안, 선택한 접근, dependency/runtime 경계의 영향을 설명합니다. -->

## 검증

<!-- 실행한 명령과 정확성, 성능, 메모리, 실패 경로의 검증 결과를 적습니다. -->

```text
# 실행한 명령과 결과
```

## 결과

<!-- 측정값과 재현 가능한 artifact 위치를 적습니다. 성능 결과는 benchmarks/results/ 아래에 둡니다. -->

## 롤백

<!-- 문제가 생겼을 때 되돌릴 flag, backend, interface 또는 commit 경계를 적습니다. -->

## Merge gate

### 공통

- [ ] 이 PR은 한 가지 질문에 답한다.
- [ ] 관련 `deploy/` 단계의 선행 조건과 완료 기준을 확인했다.
- [ ] 공개 API 변경에 맞춰 문서와 예제를 갱신했다. 해당 없음: <!-- 이유 -->
- [ ] 범위 밖 후속 과제를 issue 또는 다음 `deploy/` 문서에 남겼다. 해당 없음: <!-- 이유 -->
- [ ] `cargo fmt --check`가 통과했다. 해당 없음: <!-- Cargo workspace가 아직 없는 경우 등 이유 -->
- [ ] `cargo clippy --workspace --all-targets`에서 새 warning이 없다. 해당 없음: <!-- 이유 -->
- [ ] 관련 unit/integration test가 통과했다. 해당 없음: <!-- 이유 -->

### Rust `unsafe` / CUDA FFI / allocation

- [ ] 새 `unsafe`마다 검토 가능한 safety invariant를 주석으로 남겼다. 해당 없음: <!-- 이유 -->
- [ ] CUDA 호출의 오류 전파와 stream semantics를 검증했다. 해당 없음: <!-- 이유 -->
- [ ] allocation의 ownership, lifetime, 비동기 완료 조건을 설명했다. 해당 없음: <!-- 이유 -->
- [ ] 수동 `Send`/`Sync` 구현을 별도 검토 대상으로 표시했다. 해당 없음: <!-- 이유 -->

### 수학적 변환 또는 최적화

- [ ] `reference` exact systems 변경은 behavioral/token parity, stable fallback, lifetime/resource regression을 검증했다. 해당 없음: <!-- 이유 -->
- [ ] `E0`은 reference parity, dtype별 tolerance, 수치 안정성을 검증했다. 해당 없음: <!-- 이유 -->
- [ ] `E1`은 분포 보존 근거와 request별 RNG 격리/복원을 검증했다. 해당 없음: <!-- 이유 -->
- [ ] `A1`은 error budget, exact fallback, opt-in flag를 제공한다. 해당 없음: <!-- 이유 -->
- [ ] 성능 주장의 환경, raw result, before/after 비교를 `benchmarks/results/`에 남겼다. 해당 없음: <!-- 이유 -->

### PR 17 후속 확장 admission

- [ ] 이 PR은 새 확장의 admission-only PR이며 구현·dependency·runtime 동작을 추가하지 않는다. 해당 없음: <!-- 확장 PR이 아닌 이유 -->
- [ ] `deploy/extensions/registry.json`에 `approved-for-implementation`, 허용 track/class, `implementation_link_path: null`인 정렬된 단일 entry를 추가했다. 해당 없음: <!-- 이유 -->
- [ ] proposal과 benchmark contract의 ID·등급·path+SHA-256·flag·primary/track-required/quality metric·class gate가 일치하며 reference/fallback/workload는 Git-tracked regular file이다. 해당 없음: <!-- 이유 -->
- [ ] benchmark contract가 `gpu_count: 1`, GPU/driver/CUDA/model ID/full 40-hex model revision/dtype, positive concurrency/prompt/output 배열, 닫힌 sampling, cold/warm 상태를 실제 값으로 동결한다. 해당 없음: <!-- 이유 -->
- [ ] primary metric이 exact track-required set에 포함되고, E0 tolerance는 비교 dtype과 일치하며, query-aware A1 budget은 `[0, 1)` fraction이다. 해당 없음: <!-- 이유 -->
- [ ] PR 17 승인 질문 10개를 닫힌 `approval_answers`로 모두 답했다. 해당 없음: <!-- 이유 -->
- [ ] reference와 exact fallback은 서로 다르고 runtime flag는 `RILEY_EXPERIMENTAL_*`, 기본값은 off다. 해당 없음: <!-- 이유 -->
- [ ] 별도 deploy plan의 scope·선행조건·후속 구현 PR·실패 처리·rollback을 reviewer가 확인했다. 해당 없음: <!-- 이유 -->
- [ ] `python3 ci/check_extension_gates.py --base-revision <full-base-sha>`가 exact four-file admission diff로 통과했다. 해당 없음: <!-- 이유 -->
- [ ] rename source를 포함한 다른 파일 변경이 없고 production crate에 새 experimental flag를 우회 추가하지 않았다. 해당 없음: <!-- 이유 -->

### PR 17 후속 experimental implementation

- [ ] 이미 승인된 registry ID의 null implementation link만 `deploy/extensions/implementations/<id>.json`으로 연결하며 admission metadata를 수정하지 않았다. 해당 없음: <!-- 이유 -->
- [ ] manifest가 승인된 proposal/plan/contract/runtime flag, tracked implementation source, flag source, non-empty `{path, sha256, test_id}` validation test를 교차결합한다. 해당 없음: <!-- 이유 -->
- [ ] validation test가 명시적 workspace member의 top-level auto-discovered integration target이며 `autotests`, harness, feature 조건 때문에 CPU `cargo test --workspace --all-targets --no-default-features` lane에서 빠지지 않는다. 해당 없음: <!-- 이유 -->
- [ ] validation `test_id`는 hidden/cfg/ignored target이나 comment/string이 아니라 정확히 한 direct top-level `#[test] fn`을 가리킨다. 해당 없음: <!-- 이유 -->
- [ ] reviewer가 실제 실행 경로의 default-off, flag-on, stable fallback을 확인했다. flag 문자열이 주석이나 dead code에만 있는 것으로 대체하지 않았다. 해당 없음: <!-- 이유 -->
- [ ] stable promotion, withdrawal 또는 admitted contract 수정이 아니다. 필요하면 먼저 schema-v2 transition을 별도 승인한다. 해당 없음: <!-- 이유 -->
