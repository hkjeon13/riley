# PR 02 — Rust Workspace와 CI 뼈대

**상태:** Planned  
**선행 조건:** [PR 01](01-baseline-and-reproducibility.md)  
**다음:** [PR 03 — CUDA Host Runtime](03-cuda-host-runtime.md)

[← 이전](01-baseline-and-reproducibility.md) | [목차](README.md) | [다음 →](03-cuda-host-runtime.md)

## 목적

모델 구현 없이 Rust workspace, 빌드 규칙, lint, CPU-only CI와 CUDA 선택 빌드 구조를 만든다.

## 권장 초기 구조

```text
Cargo.toml
crates/
├── rustinfer-core/
├── rustinfer-cuda/
├── rustinfer-tensor/
├── rustinfer-model/
├── rustinfer-runtime/
├── rustinfer-scheduler/
└── rustinfer-server/
kernels/
├── CMakeLists.txt
├── include/
└── src/
benchmarks/
tests/
deploy/
```

처음부터 모든 crate에 코드를 넣지 않는다. 빈 crate는 책임과 dependency 방향만 정의한다.

## Dependency 방향

```text
server → scheduler → runtime → model/tensor → cuda
                               ↘ core
```

규칙:

- `cuda`는 server나 scheduler를 알지 못한다.
- `model`은 HTTP 타입을 알지 못한다.
- `tensor`는 model architecture를 알지 못한다.
- 공통 오류와 작은 value type만 `core`에 둔다.
- 순환 dependency는 허용하지 않는다.

## CI job

### CPU-only 필수

- format
- clippy
- unit test
- docs build
- dependency/license check
- feature 조합 compile check

### GPU 선택

GPU runner가 준비되기 전에는 로컬 명령과 nightly/manual workflow만 정의한다.

- CUDA compile smoke
- device test
- kernel correctness
- benchmark는 PR merge 필수가 아니라 결과 artifact로 보존

## Feature 설계

예:

```text
default = []
cuda = []
server = []
bench = []
experimental = []
```

CPU-only 환경에서 `cargo test --workspace`가 동작해야 한다. CUDA가 없다는 이유로 전체 workspace import가 실패하면 안 된다.

## 빌드 경계

- CUDA source는 C ABI를 export한다.
- Rust build script는 toolkit discovery와 link만 담당한다.
- toolkit path를 코드에 하드코딩하지 않는다.
- compile target과 runtime device capability를 분리한다.
- build log에 CUDA arch 목록을 출력한다.

## 테스트

- dependency 방향 smoke
- no-default-features compile
- CUDA feature 없는 Linux compile
- invalid CUDA path 오류 메시지
- version 정보 출력

## 비범위

- CUDA device 호출
- tensor allocation
- model loading
- API server 동작

## 완료 기준

- [ ] 새 clone에서 문서화된 명령으로 CPU CI 통과
- [ ] CUDA가 있는 환경에서 빈 kernel library link 성공
- [ ] crate 책임과 dependency가 README에 명시됨
- [ ] release/debug profile 기본값 결정
- [ ] panic 정책과 error type 기본 원칙 결정

[← 이전](01-baseline-and-reproducibility.md) | [목차](README.md) | [다음 →](03-cuda-host-runtime.md)
