# W1 결과·결정 기록

작성일: 2026-11-13 (Asia/Seoul)\
작업 기준: `main` `37ede444562b1d6c3718fae0a8db314f81fbba92`\
초기 복구 작업 브랜치: `codex/restore-validation-release-contracts`\
반영 브랜치: `main`\
최신 판정: 로컬·Linux CI, 격리 컨테이너의 과거 RC3 고정 테스트, CUDA 12.4 배포 이미지 빌드·CPU 검사 완료. 배포 이미지의 GPU 추론 검증은 미완료.

[W1 계획](01-validation-and-release.md) · [종합 계획](README.md) · [검사 사용법](../../ci/README.md)

## 1. GPU 서버 필요 여부

이번 포맷·린트·CPU 테스트·계약 검사·비교 실행기 복구에는 GPU 서버가 필요하지 않다. macOS ARM64에서 Rust 1.85.0, Python 3.12.7로 수행했다. HTTP 테스트에 필요한 로컬 포트 바인딩과 자식 프로세스 실행은 허용한 환경에서 검증했다.

다음 확인은 구분한다.

| 확인할 내용 | 필요한 환경 | 초기 로컬 실행 |
|---|---|---|
| CPU 검사·판정 도구 | Rust 1.85.0, Python 3.12 | 완료 |
| Linux `/proc`·파일 디스크립터·renameat2 계약 | Linux CPU 호스트 | 미실행 |
| 과거 RC3 고정 실행 호스트 계약 | 검토된 정확한 `/usr/bin/python3.10`을 가진 Linux 호스트 | 미실행 |
| CUDA 컴파일·링크 | 지원 Linux/CUDA 빌드 환경 또는 Docker, 장치는 없어도 됨 | 미실행 |
| 실제 모델 추론·성능·GPU 자원 수명·soak | 대상 NVIDIA GPU 서버 | 미실행 |

로컬에는 Docker와 CUDA 도구가 없어 컨테이너를 빌드하지 않았다. Dockerfile의 정적 계약 검사는 통과했으며, 이를 이미지 빌드 성공으로 해석하지 않는다. W2·W3 성능 변경과 W4-A 수치 정확성 조사는 GPU 서버가 필요하다.

## 2. 변경한 내용과 판단 근거

### 포맷·린트와 실행 대상

Rust 포맷을 맞추고 Clippy 경고를 수정했다. 문서의 오류 조건·링크, 불필요한 코드, feature별 사용 조건과 스타일을 정리하고, 모델 설정·토크나이저 검증 일부를 작은 helper로 옮겼다. CUDA의 긴 실행·소유권 절차와 기존 FP32 연산은 함수별 근거를 둔 제한적인 lint 허용을 사용했다. workspace 전체 lint 수준을 낮추지 않았다.

호스트 메타데이터의 좁은 정수 변환은 범위를 검사하도록 바꿨고, 공유 KV 블록을 채택할 때 오류 검사를 상태 변경 전에 수행한다. 지수 근사의 `0.693_147_2_f32`는 동일한 FP32 비트의 표준 `LN_2` 상수로 표현했다. 커널, third-party 소스와 Cargo.lock은 변경하지 않았다. CUDA 빌드·수치 동등성의 실행 확인은 남아 있다.

기존 CUDA fault 테스트의 subprocess 코드는 동작과 테스트 이름을 유지해 `tests/support`로 이동했다. production 소스의 외부 프로세스 실행 금지 경계와 테스트를 분리했다. 새로 명시한 GPU 통합 테스트 58개에는 CUDA/probe `required-features`를 달았다. CPU 실행에서 해당 GPU 테스트가 실행된 것처럼 집계하지 않으며, 명시적 CUDA 기능 실행 대상은 보존한다.

### 현행 서빙과 과거 릴리스 계약

과거 RC 패키징의 CUDA 12.8.1/sm_89 메타데이터에 현행 CLI 해시를 덮어쓰지 않았다. 당시 커밋 `21f445f4870a140346509144c36c7294f2f677f3`의 정확한 CLI blob을 [고정 fixture](../../ci/release/fixtures/legacy-server-defaults-21f445f.rs.txt)로 보존했다. 과거 패키지 정상 경로가 통과하고, 현행 CLI를 그 메타데이터로 묶으면 계속 거부하는 테스트를 추가했다.

현행 배포는 CUDA 12.4/sm_80·86·89 경로다. [별도 source 계약](../../ci/serving-source-contract.json)이 CLI·inference profile·배포 Dockerfile·entrypoint 네 파일을 결속하고, [검사기](../../ci/check_serving_source_contract.py)가 byte drift·누락·링크·inventory 변경을 거부한다. 이는 검토한 소스를 식별하는 장치이며 CUDA 바이너리·실제 배포 qualification을 대신하지 않는다.

2026-11-19에 배포 Dockerfile 전용 ignore 파일을 계약에 추가했다. 배포 entrypoint의 `serve --help`와 `serve -h`는 모델 경로 없이 CLI에 전달하도록 수정했다. Linux Actions가 현행 Dockerfile의 빌드와 패키지 검사도 실행한다. [배포 이미지 CI](../../.github/workflows/distribution-image.yml) · [이미지 검사기](../../ci/check_distribution_image.py)

첫 Actions 실행에서 CUDA 12.4 배포 이미지는 빌드됐다. CPU 러너의 CLI 실행은 GPU 런타임이 공급하는 `libcuda.so.1`이 없어서 종료 코드 127로 실패했다. CPU의 기본 이미지에서 이 라이브러리만 미해결인지 확인하고, 도움말·버전 검사 컨테이너에만 같은 CUDA SDK의 스텁을 읽기 전용으로 연결하도록 수정했다. [첫 이미지 빌드와 실행 기록](https://github.com/hkjeon13/riley/actions/runs/44358229097)

성능 파생 검사의 native checker 변경을 검토해 소스·정책 pin을 함께 맞췄다. 진단용 checker의 선택적 image digest가 정식 릴리스로 전파되지 않도록 full/compact 릴리스 경로 모두 image digest를 필수로 검사한다. 기존 성능 임계값을 낮추지 않았다.

### 의존성·확장 경계

[경계 정책](../../ci/check_workspace_boundaries.py)을 이미 존재하는 MiniJinja, ordered JSON, probe 기능 및 개발 의존성의 정확한 버전·기능과 일치시켰다. 캐시된 패키지 checksum, license, MSRV를 확인했으며 신규 패키지 설치나 버전 갱신은 하지 않았다. optional 의존성을 숨기지 않도록 metadata는 모든 feature를 resolve한다. resolve 자체는 CUDA를 컴파일하거나 GPU를 초기화하지 않는다. 최종 검사는 직접 의존성 14개·registry package 28개를 확인한다.

C12 prefix-cache admission은 당시 fallback 해시와 현행 KV 소스가 이미 달랐다. 승인된 proposal·registry·contract는 변경하지 않았다. [확장 검사기](../../ci/check_extension_gates.py)의 좁은 호환 규칙은 정확한 extension/path/과거 해시 조합, 과거 admission Git blob, 검토한 현행 해시를 모두 확인한다. 구현 링크가 생기거나 이후 소스가 다시 바뀌거나 Git 이력이 없으면 거부한다. tenant isolation이나 prefix cache 구현의 qualification을 부여하지 않는다. 향후 구현 admission은 현행 fallback에 맞는 별도 계약이 필요하다.

### 비교·부하 실행의 실패 전파

[비교 runner](../../benchmarks/live/compare_runner.py)는 기존 시나리오와 positional interface를 유지하면서 명시적인 실행 파일·모델·부하 스크립트 경로 인자를 지원한다. 공백·JSON은 argv 원소로 보존하며, evidence label은 새 경로만 허용한다.

자식 비정상 종료, 누락·잘린 JSON, 중복 키·비유한 수치, 다른 모델/조건, 요청 수 불일치, warm-up 실패, 429·요청 실패는 전체 비교 실패로 전파한다. 전 셀이 유효할 때만 `COMPARE_DONE`을 출력한다. 실패 시 로그와 실패 상태의 `run.json`을 남기고, readiness/load timeout에는 자신이 만든 프로세스 그룹을 종료·회수한다. 서버·부하 argv, 셀 결과 해시와 존재하는 script/binary/checkpoint manifest의 해시를 기록한다.

[부하 생성기](../../benchmarks/live/load.py)는 SSE의 terminal finish와 `[DONE]`, 유효한 token usage를 요구하고 연결을 정리한다. 기존 처리량·백분위 계산식은 유지했다. runner와 load.py를 함께 배포해야 하며, 과거 측정 결과는 새 유효성 규칙을 적용한 결과로 바꾸어 부르지 않는다. 새 속도 주장은 양쪽 엔진을 같은 갱신된 부하 생성기로 다시 측정한 뒤 판단한다.

### CI와 플랫폼별 계약

[Production CPU](../../.github/workflows/production-cpu.yml)는 Rust 1.85.0·Python 3.12를 고정하고, 기본/서버 Clippy와 테스트, 현행 source guard를 실행한다. 격리 빌드는 모델·기존 evidence 전체를 복사하는 대신 production 입력·embedded fixture만 복사한다.

Linux 디스크립터 의미론이 필요한 fixture는 macOS에서 명시적으로 skip한다. hosted CI는 `RILEY_REQUIRE_LINUX_CONTRACT_TESTS=1`이므로 Linux 조건이 없으면 실패한다. 기존 순수 parser·정책 검사는 계속 실행한다. JSON 깊이는 Python accelerator의 recursion limit 차이에 의존하지 않고 고정 상한으로 검사한다.

과거 정확한 호스트 Python을 실행하는 30개 테스트는 필요한 환경이 없으면 이유를 기록하고 skip한다. manual `run_legacy_host_tests` 입력으로 `[self-hosted, linux, x64, riley-legacy-rc3]` lane을 선택하면 `RILEY_REQUIRE_LEGACY_HOST_TESTS=1`을 적용해 환경 불일치를 실패시킨다. 이 lane은 GPU를 쓰지 않는다. 실제 runner 등록·과거 환경 제공·실행은 이번에 하지 않았다. 일반 hosted CI의 성공만으로 이 30개의 실행 통과를 주장하면 안 된다.

## 3. 실행 결과와 원본 근거

처음 전체 검사에서 문서 링크 두 개가 실패했다. 링크만 수정한 뒤 해당 문서 검사·포맷·Clippy·벤치마크 검사를 다시 실행했다. 비교 manifest의 부하 argv 기록 보완도 재실행에 포함했다. 첫 실패 로그를 보존하고, 최신 결과를 합쳐 19개 검사 항목을 판정했다.

| 검사 | 실제 결과 |
|---|---|
| 포맷·기본 workspace Clippy·서버 Clippy | 모두 통과, warnings를 오류로 처리 |
| Rust workspace all-targets CPU | 793 passed, 8 ignored, 0 failed |
| 서버 `--features server --lib` | 97 passed, 1 ignored, 0 failed; 별도 실행 수치 |
| CI Python 검사 | 42 passed |
| 릴리스 Python 검사 | 1,314건 수집, 1,150 passed, 164 skipped, 실패·오류 0 |
| 벤치마크 Python 검사 | 196 passed; 비교·부하 고장 경계 12건 포함 |
| dependency/source/extension 경계 | 세 검사 모두 통과 |
| Dockerfile 계약·결과 schema·soak shell syntax | 네 검사 모두 통과 |
| Python-free synthetic model loading | 통과, 실제 모델 추론 없음 |
| Rustdoc `-D warnings` | 수정 후 통과 |
| CPU feature matrix·production 격리 빌드 | 모두 통과 |

릴리스 skip 164건은 Linux `/proc` 조건 132건, 과거 고정 호스트 30건, 기존 renameat2 조건 2건이다. ignored·skip은 실행 통과 수에 포함하지 않았다. Rust 두 명령에는 중복 대상이 있으므로 하나의 고유 테스트 합계로 더하지 않는다.

- [종합 실행 결과](../../benchmarks/evidence/20261113-validation-restoration/summary.json): 19개 검사의 최신 결과, 재실행 여부, 원본 로그 위치.
- [첫 전체 실행](../../benchmarks/evidence/20261113-validation-restoration/local-cpu-r1/results.json): 문서 검사 실패를 포함해 보존.
- [영향 범위 재실행](../../benchmarks/evidence/20261113-validation-restoration/local-cpu-r2/results.json): 수정한 포맷·린트·현행 소스·벤치마크·문서 검사.
- [최종 입력 파일 해시](../../benchmarks/evidence/20261113-validation-restoration/source-manifest.json): 검사 관련 source/config와 새 파일을 포함한 dirty 작업 트리 스냅샷 식별. 원본 결과의 patch 해시는 당시 tracked diff만 포함하며, commit이나 release archive 해시가 아니다.

## 4. 남은 종료 조건과 다음 작업

검사 당시에는 미커밋 작업 트리였다. 2026-11-18에 검증한 변경을 `main`의 `d90959b00`에 커밋했다. 기존 측정 근거와 실행 로그는 보존했다.

푸시 후 별도 `Rust checkpoint preparation` workflow에서 LoRA 도구의 포맷 실패를 발견했다. 포맷과 Clippy 경고 두 건을 고치고, 기존 소문자 SHA-256 변환 함수를 재사용했다. 로컬에서 기본 구성 16건과 `hub` 구성 21건이 통과했다. 빈 PATH에서 테스트 바이너리와 도움말 실행, 기본 구성의 Hub 의존성 부재도 확인했다.

GitHub Actions Linux Production CPU가 `9ca18cb0f`에서 통과했다. 당시 로그에서 `RILEY_REQUIRE_LINUX_CONTRACT_TESTS=1`과 릴리스 테스트 1,314건 중 1,284건 통과·30건 skip을 확인했다. 장문 추적과 네 모델 GSM8K 결과를 포함하는 후속 `main`의 `ed392e579`도 Production CPU를 통과했다. 과거 RC3 고정 호스트 lane의 별도 실행 결과는 아래에 기록했다. [최초 Production CPU 실행](https://github.com/hkjeon13/riley/actions/runs/44221081481) · [후속 main 실행](https://github.com/hkjeon13/riley/actions/runs/44379277897) · [체크포인트 도구 CI](https://github.com/hkjeon13/riley/actions/runs/44221081323)

CUDA 관련 Rust 포맷·문서·host cast 변경도 있으므로 릴리스 전 CUDA compile/link lane과 영향받는 GPU 검증을 실행해야 한다. 기존 커널·임계값·측정 근거가 유지되었다는 사실만으로 GPU 실행 동등성을 확정하지 않는다.

### 과거 RC3 고정 테스트의 실제 실행

2026-11-19 KST에 GitHub Actions의 격리된 Jammy 컨테이너에서 `3.10.12-1~22.04.16` 패키지의 `/usr/bin/python3.10`을 사용했다. 바이너리 해시와 소유권이 계약과 일치했고 체크아웃은 지정 경로에 읽기 전용으로 마운트했다. 다섯 파일의 테스트 34건이 통과했고 실패·skip은 0건이었다. 기존 self-hosted 작업과 같은 실행 스크립트를 쓴다. [성공한 실행](https://github.com/hkjeon13/riley/actions/runs/44367370717) · [패키지 해시](../../ci/release/LegacyHostContracts.sha256) · [결과 및 환경 기록](../../benchmarks/evidence/20261119-rc3-host-contracts/receipt.json)

첫 시도는 Ubuntu 스냅샷 인덱스의 HTTP 500으로 멈췄다. 패키지 SHA-256을 고정해 직접 가져온 뒤에는 Python 검증이 통과했지만, 다른 마운트 경로를 RC3 v2 검사기가 거부했다. 원래 경로를 컨테이너에 재현하고 같은 검사기를 실행했다. 검증 범위는 Linux에서 수행한 CPU 계약 테스트다. 실제 RC3 복구·재생과 배포 자격 검증은 각각의 운영 조건을 따른다.

### 현행 배포 이미지

`96c04dbe4`의 소스 아카이브로 CUDA 12.4 Linux AMD64 이미지를 빌드했다. 비루트 UID/GID, entrypoint, Riley의 버전·서버 도움말, 체크포인트 도구, 16개 프리셋, 이진 파일 해시와 런타임 파일을 검증했다. [성공한 Actions 실행](https://github.com/hkjeon13/riley/actions/runs/44361292764) · [해시가 포함된 검증 보고서](../../benchmarks/evidence/20261119-cu124-distribution-image/report.json)

GPU 없는 runner에서 Riley 바이너리는 NVIDIA가 주입하는 `libcuda.so.1`을 요구했다. CPU CLI 검사는 CUDA 12.4 SDK에서 가져온 스텁 파일을 읽기 전용으로 마운트해 수행했다. 완성 이미지의 드라이버 주입과 모델 추론은 GPU 호스트에서 추가 검증이 필요하다. W4·W5의 네이티브 GPU 실행은 자체 바이너리로 검증했다. [기본 링커 기록](../../benchmarks/evidence/20261119-cu124-distribution-image/runtime-dependencies.txt) · [CPU 검사 링커 기록](../../benchmarks/evidence/20261119-cu124-distribution-image/cpu-cli-dependencies.txt) · [A100 혼합 부하](05-mixed-soak-result.md)

배포 이미지 재빌드에 쓰려던 소스 아카이브의 A100 전송은 자동 승인 검토에서 전송 자료의 범위와 목적지를 이유로 거절됐다. 배포 이미지의 GPU 검증은 승인된 경로에서 동일 이미지와 바이너리의 해시를 확인한 뒤 수행해야 한다.

후속 CUDA 메모리 오류 빌드에서 cuBLAS 진단 기능을 끄면 C ABI 헤더가 그 기능의 내부 타입을 참조하는 오류를 확인했다. 관련 함수 두 개의 선언을 정의와 같은 기능 조건에 넣었다. GPU 없이 C ABI 테스트를 네 가지 기능 조합으로 컴파일하는 단계를 Production CPU에 추가했고, 로컬에서 첫 오류를 재현한 뒤 네 조합의 통과를 확인했다. A100의 CUDA 재빌드와 오류 주입 실행은 등록 상태다. [실패 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/initial-build/build.log) · [CI 검사](../../.github/workflows/production-cpu.yml)

`a21829d0e`의 C ABI 검사와 Production CPU가 Linux에서 통과했다. 이 커밋에서 이미지 CI는 실행되지 않아 `kernels/**`와 `rust-toolchain.toml`을 이미지 검증의 `main` 변경 감지에 추가했다. `120e984c9`의 Production CPU와 CUDA 12.4 이미지 빌드·CPU CLI·런타임 무결성 검사가 모두 통과했다. 새 이미지 ID와 소스 아카이브는 이전 이미지와 달랐고, 설치된 런타임 파일 해시 전체와 Riley 바이너리 해시는 일치했다. GitHub의 증거 ZIP 해시와 로컬에서 다시 계산한 소스 아카이브 해시를 대조했다. [Production CPU](https://github.com/hkjeon13/riley/actions/runs/44395442651) · [이미지 빌드와 검사](https://github.com/hkjeon13/riley/actions/runs/44395442715) · [새 이미지 검증 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/header-fix/receipt.json)

W2·W3는 이 runner의 새 label·실패 판정·실행 argv/해시를 사용한다. W4는 정확성 판정 결과를 같은 후보에 결속한다. W5는 현행 이미지와 네이티브 GPU 실행 근거를 릴리스 문서에 연결한다. M5 판정에는 동일 후보의 패키징·GPU 실행·품질 근거가 필요하다.
