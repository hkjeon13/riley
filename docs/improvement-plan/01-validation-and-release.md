# W1. 검증·CI·릴리스 계약 복구

작성일: 2026-11-13\
기준·상태·공통 가드레일: [종합 계획](README.md)\
목표: 이후 성능·정확성 변경을 동일한 후보와 재현 가능한 검사로 판정할 수 있게 한다.

진행 상태: W1 구현 및 로컬 CPU 검사 완료. Linux 전용 실행·과거 고정 호스트 검사·CUDA 검증은 별도 확인이 필요하다. 변경 근거와 실행 결과는 [W1 결과·결정 기록](01-validation-result.md)에 정리했다.

## 1. 확인된 문제

| 문제 | 확인 내용 | 영향 |
|---|---|---|
| 포맷 | Rust 1.85.0의 `cargo fmt --all -- --check`가 57개 파일에서 차이를 보고 | 현재 소스가 필수 CI 포맷 단계를 통과하지 못함 |
| Clippy | `--workspace --all-targets --no-default-features -- -D warnings` 실패 | 실제 경고·코드 구조와 CI 정책이 어긋남 |
| 릴리스 소스 결속 | `release_common.py`의 `SERVER_DEFAULTS_SOURCE_SHA256`와 현재 `main.rs` 해시 불일치 | 패키지 생성·검증 테스트가 정상 fixture에서도 먼저 거부됨 |
| 성능 계약 결속 | 성능 파생 검사 소스와 `NATIVE_PROFILE_CONTRACT_SHA256` 불일치 | 정상 근거 판정과 runner drift 테스트가 의도한 단계까지 못 감 |
| Python 환경 | Python 3.12 벤치마크 테스트 182건: 실패 2·오류 23. 3.10은 다른 결과 | 실행 환경을 고정하지 않으면 코드 문제와 도구 환경 문제를 혼동 |
| 부하 실행 실패 전파 | `compare.py`는 부하 자식 프로세스에 `check=False`를 사용하고 뒤에서 `done`·`COMPARE_DONE` 출력 | 완료 문자열만 수집하면 실패·불완전 결과가 성공처럼 보일 수 있음 |

CPU 라이브러리는 네트워크 제한을 해제한 재실행에서 621건 통과·1건 ignored였다. 최초 HTTP 테스트 실패 26건은 포트 바인딩 권한 문제였으므로 엔진 결함과 구분한다. 이번 검사는 GitHub Actions의 현재 실행 상태나 전체 CUDA 검사의 통과를 증명하지 않는다.

## 2. 조사 시작점

- [필수 CPU workflow](../../.github/workflows/production-cpu.yml): 실제 필수 명령의 기준.
- [CPU 기능 조합](../../ci/check_feature_matrix.sh): 서버 기능이 기본 테스트에서 빠지는 범위 확인.
- [릴리스 공통 계약](../../ci/release/release_common.py): 기본값 소스 pin, CUDA·대상 아키텍처, 기본 옵션 metadata.
- [성능 근거 검사](../../benchmarks/scripts/check_release_performance.py): `_require_bound_semantic_policy`, native checker·runner 해시.
- [성능 파생 검사](../../benchmarks/scripts/check_native_profile_pair.py): 원본에서 최종 지표로의 변환.
- [비교 실행기](../../benchmarks/live/compare.py), [부하 생성기](../../benchmarks/live/load.py): 실패·표본·프로세스 수명 경계.
- [CI 설명](../../ci/README.md): Python 3.11+ 등 기존 요구사항. 실제 workflow·코드와 대조할 설명 자료이며 완전한 최신 사실로 가정하지 않는다.

## 3. 문제를 찾고 해결하는 순서

### W1-1. 환경·기준선을 고정한다

작업 시작 시 Git SHA, tracked/untracked 변경, Rust/Cargo, Python, OS를 기록한다. Python 3.11 이상 중 한 버전을 CI와 로컬의 기준으로 명시한다. 3.10을 지원할지는 별도 결정하며, 우연히 테스트가 수집되지 않은 결과를 통과로 취급하지 않는다.

실패를 `제품 코드`, `검사 계약 드리프트`, `환경 제약`, `미실행`으로 나누고 최초 오류·재현 명령·영향 범위를 기록한다. 뒤따르는 수십 개 fixture 실패를 독립 결함 수십 개로 집계하지 않는다.

### W1-2. 포맷·린트를 동작 변경과 분리한다

1. 작업 브랜치에서 `cargo fmt --all`을 적용하고 실제 diff가 포맷에 한정되는지 확인한다.
2. Clippy는 우선순위·불필요한 참조·중복 closure·문서 문제부터 수정한다.
3. 긴 함수 분해는 오류 반환·버퍼 소유권·실행 순서가 바뀌지 않게 하고 해당 기존 테스트를 실행한다.
4. CUDA 비활성 feature에서만 생기는 unused 항목은 선언·사용 조건을 맞춘다. workspace 전체 경고를 일괄 허용하는 방식으로 덮지 않는다.

포맷도 소스 해시를 바꾼다. 해시 갱신은 최종 소스 형태가 정해진 뒤 진행하고, 포맷으로 바뀐 바이트와 의미 변경을 구분해 기록한다.

### W1-3. 계약이 무엇을 보호하는지 먼저 재검토한다

`main.rs`의 과거 pin 커밋과 현재 코드 사이에서 CLI 기본값, 프로파일 해석, 버퍼·KV 설정, 오류 처리, API 동작의 변경을 목록화한다. 현재 파일 해시를 계산해 literal만 바꾸는 작업으로 끝내지 않는다.

성능 검사도 native checker와 runner 변경을 비교해 지표 산식, 임계값, 필수 입력, 원본 근거 재생 규칙이 바뀌었는지 확인한다. 검토한 변경에 맞춰 정책 digest·fixture·설명을 함께 갱신한다. 정상 경로를 복구한 뒤에는 한 바이트 drift·필드 누락·다른 바이너리·다른 모델 근거를 여전히 거부해야 한다.

계약에 남은 CUDA 12.8.1/sm_89 metadata와 0.1.0 A100 CUDA 12.4 측정의 관계도 확인한다. 과거 RC3 패키징과 현재 Docker 배포가 다른 경로라면 각각의 지원 범위·필수 검사를 명시한다. 모든 과거 gate를 현행 릴리스 gate로 오인하거나, 불일치를 이유로 검사를 임의 삭제하지 않는다.

### W1-4. 비교 실행기를 실패 시 명확히 종료하게 한다

1. 자식 프로세스 종료 코드와 출력 JSON의 존재·파싱 가능 여부를 검사한다.
2. 기대 요청 수·실패 수·warm-up 구분·finite 지표·모델 ID·실행 조건을 확인한다. 의도적 429 실험은 사전에 기대 상태를 별도 지정한다.
3. 모든 사전 지정 셀이 유효할 때만 전체 성공을 기록한다. 실패 셀·stderr·부분 결과는 진단용으로 남긴다.
4. 정상·비정상·timeout 모두 자기 자식 프로세스를 정리한다. readiness timeout의 `kill()` 뒤 회수도 확인한다.
5. 현재 실행기의 `ROOT/src`, `ROOT/models`, `ROOT/bin/riley-<suffix>`, vLLM venv·CUDA 경로 가정을 문서화하거나 인자로 받는다. 공백이 든 경로도 argv 목록으로 보존한다.

검증 테스트는 실제 고장 경계에 한정한다. 자식 실패, 잘린 JSON, 누락 셀, NaN, 요청 수 불일치, timeout 이후 자식 잔류를 작은 대체 프로세스로 재현한다. 이 검사는 GPU나 외부 API를 필요로 하지 않는다.

## 4. 재현·검증 명령

저장소 루트에서 실행한다. 아래 `python3.12`가 로컬·CI의 지원 기준이다. Rust는 1.85.0으로 고정했다. `REVIEW_BASE_SHA`에는 PR base 또는 실제 변경 비교 기준 SHA를 사전 지정한다. 실제 실행 결과와 플랫폼별 미실행 항목은 [결과 기록](01-validation-result.md)을 따른다.

```bash
export PYTHONDONTWRITEBYTECODE=1
rustc --version
cargo --version
python3.12 --version
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo clippy --locked -p riley-server --all-targets --no-default-features --features server -- -D warnings
cargo test --locked --workspace --all-targets --no-default-features
cargo test --locked -p riley-server --lib --no-default-features --features server
python3.12 -m unittest discover -s ci/tests -p 'test_*.py' -v
python3.12 -m unittest discover -s ci/release -p 'test_*.py' -v
python3.12 -m unittest discover -s benchmarks/scripts/tests -p 'test_*.py' -v
python3.12 ci/check_workspace_boundaries.py --locked
python3.12 ci/check_serving_source_contract.py
: "${REVIEW_BASE_SHA:?PR base 또는 실제 비교 기준 SHA를 지정하세요}"
python3.12 ci/check_extension_gates.py --base-revision "$REVIEW_BASE_SHA"
python3.12 ci/release/verify_runtime_dockerfile.py
python3.12 ci/release/verify_reconstructed_runtime_assembly_dockerfile.py
python3.12 benchmarks/scripts/validate_contract.py
bash -n ci/run_release_soak.sh
ci/verify_python_free_model_loading.sh
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --no-default-features
ci/check_feature_matrix.sh
ci/check_workspace_without_research_tools.sh
```

`check_extension_gates.py`의 `--base-revision`은 workflow와 같은 의미의 비교 기준이어야 한다. 현재 HEAD를 편의상 전달해 변경 검사를 무력화하지 않는다.

## 5. 완료 조건과 산출물

- [x] W1-1: 지원 Python·Rust 버전과 재현 환경이 명시되어 있다.
- [x] W1-2 로컬: 포맷·기본/서버 Clippy·CPU 테스트·문서·경계·격리 빌드가 통과한다.
- [ ] W1-2 CI: 실제 Linux hosted workflow를 실행하고 Linux 전용 항목의 통과를 확인한다.
- [x] W1-3 로컬: 계약 정상 경로와 drift 거부 경로가 모두 통과한다. 과거 릴리스 근거의 검증 가능 범위가 보존된다.
- [x] W1-4: 부하 실행 실패·누락 JSON·무효 셀이 전체 성공으로 승격되지 않는다.
- [x] [결과·결정 기록](01-validation-result.md)에 변경 전 실패, 수정 근거, 실행한 검사, 미실행 검사를 명시했다.
- [ ] 과거 RC3 고정 호스트 실행이 필요할 때 전용 lane에서 수행한다. 일반 CI의 명시적 skip은 해당 계약의 실행 통과를 뜻하지 않는다.

## 6. 다른 작업과의 관계

W2·W3는 이 작업에서 고정한 후보 ID·측정 조건·유효성 검사를 사용한다. W4는 독립 정확성 판정기를 여기에 추가하되 운영 서버 의존성을 늘리지 않는다. W5는 검증된 필수 명령과 패키징 경로를 CI·사용자 문서에 반영한다.

완료 전 성능 프로파일 조사는 가능하다. 다만 판정 도구의 실패를 해결하지 않은 상태에서 새 후보를 정식 통과로 기록하지 않는다. 회수 시에는 검증 계약과 소스 변경을 한 쌍으로 이전 상태로 돌리고, 실패 거부 기능을 비활성화하지 않는다.
