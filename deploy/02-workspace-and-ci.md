# PR 02 — Rust Workspace와 CI 뼈대

**상태:** Planned  
**선행 조건:** [PR 01](01-baseline-and-reproducibility.md)  
**다음:** [PR 03 — CUDA Host Runtime](03-cuda-host-runtime.md)

[← 이전](01-baseline-and-reproducibility.md) | [목차](README.md) | [다음 →](03-cuda-host-runtime.md)

## 목적

모델 구현 없이 Rust workspace, native CUDA C++ 빌드 규칙, lint, CPU-only CI와 CUDA 선택 빌드 구조를 만든다. Production workspace와 optional Python/Triton 연구 도구의 dependency 경계를 이 단계에서 강제한다.

상세 정책: [구현 언어·라이브러리와 Runtime Dependency 경계](../docs/implementation-stack-and-runtime-boundaries.md)

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
├── include/                 # C ABI
├── src/                     # CUDA C++
└── cutlass/                 # 필요 시 추가

tools/
├── python/                  # optional offline/reference
└── native/

experiments/
└── triton/                  # optional prototype

benchmarks/
tests/
docs/
deploy/
```

처음부터 모든 crate에 코드를 넣지 않는다. 빈 crate는 책임과 dependency 방향만 정의한다.

## Dependency 방향

```text
server → scheduler → runtime → model/tensor → cuda
                               ↘ core
                                         ↓ C ABI
                                     CUDA C++ library
```

규칙:

- `cuda`는 server나 scheduler를 알지 못한다.
- `model`은 HTTP 타입을 알지 못한다.
- `tensor`는 model architecture를 알지 못한다.
- 공통 오류와 작은 value type만 `core`에 둔다.
- 순환 dependency는 허용하지 않는다.
- production crate는 `tools/python`과 `experiments/triton`을 import, link, invoke하지 않는다.
- Python tool은 production crate의 build script가 될 수 없다.
- Python이 생성한 artifact는 JSON/safetensors 같은 명시적 형식으로만 입력된다.

## 언어·라이브러리 경계

```text
Rust
→ host runtime, scheduler, model IR, memory metadata, server

CUDA C++
→ native production kernels

C ABI
→ Rust와 CUDA C++ 사이의 호출 규약

cuBLASLt
→ dense GEMM 기본 경로

CUTLASS
→ 이후 profiler로 필요성이 입증된 특수 GEMM

Triton
→ experiments 전용; 초기 production dependency 아님

Python
→ reference/offline tools 전용; runtime dependency 아님
```

## CI job

### Production CPU-only 필수

- format
- clippy
- unit test
- docs build
- dependency/license check
- feature 조합 compile check
- `tools/python` 없이 workspace metadata와 build 확인

### Native CUDA 선택

GPU runner가 준비되기 전에는 로컬 명령과 nightly/manual workflow만 정의한다.

- CUDA C++ compile smoke
- C ABI link smoke
- device test
- kernel correctness
- benchmark는 PR merge 필수가 아니라 결과 artifact로 보존

### Python reference 선택

별도 optional job으로만 실행한다.

- pinned Python environment
- PyTorch/Transformers reference fixture 생성
- artifact provenance 검사
- 생성 artifact를 Rust가 Python 없이 읽는 integration test

### Python-free release smoke

Python이 설치되지 않은 container 또는 runner에서 다음을 실행한다.

```bash
cargo build --release --features cuda,server
```

PR 02에서는 아직 추론을 하지 않지만 build와 빈 native link가 Python 없이 완료되어야 한다.

## Feature 설계

예:

```text
default = []
cuda = []
server = []
bench = []
experimental = []
```

`python` 또는 `triton`을 production Cargo feature로 추가하지 않는다. Python/Triton 도구는 독립 환경과 manifest를 사용한다.

CPU-only 환경에서 `cargo test --workspace`가 동작해야 한다. CUDA가 없다는 이유로 전체 workspace import가 실패하면 안 된다.

## 빌드 경계

- CUDA source는 C ABI를 export한다.
- CUDA source는 `.cu`의 CUDA C++로 구현한다.
- Rust build script는 toolkit discovery와 native link만 담당한다.
- toolkit path를 코드에 하드코딩하지 않는다.
- compile target과 runtime device capability를 분리한다.
- build log에 CUDA arch 목록을 출력한다.
- 초기 kernel은 `nvcc` AOT compile을 사용한다.
- NVRTC와 runtime Triton JIT는 초기 범위에서 제외한다.
- CUTLASS는 vendoring/version 정책이 정해질 때까지 선택 dependency로 유지한다.

## 테스트

- dependency 방향 smoke
- no-default-features compile
- CUDA feature 없는 Linux compile
- Python executable이 없는 환경의 production build
- production binary/library dependency inspection
- invalid CUDA path 오류 메시지
- version 정보 출력

## 비범위

- CUDA device 호출
- tensor allocation
- model loading
- API server 동작
- Triton production integration
- CUTLASS operation 구현
- NVRTC

## 완료 기준

- [ ] 새 clone에서 문서화된 명령으로 CPU CI 통과
- [ ] CUDA가 있는 환경에서 빈 native CUDA library link 성공
- [ ] Python 없는 환경에서 production build 성공
- [ ] production Cargo dependency graph에 Python/PyTorch/Transformers/Triton runtime 없음
- [ ] `tools/python`과 `experiments/triton`이 production workspace 경계 밖에 있음
- [ ] crate 책임과 dependency가 README에 명시됨
- [ ] release/debug profile 기본값 결정
- [ ] panic 정책과 error type 기본 원칙 결정

[← 이전](01-baseline-and-reproducibility.md) | [목차](README.md) | [다음 →](03-cuda-host-runtime.md)
