# PR 03 — CUDA Host Runtime과 FFI 경계

**상태:** Active
**선행 조건:** [PR 02](02-workspace-and-ci.md)  
**다음:** [PR 04 — Tensor와 메모리](04-tensor-and-memory.md)

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)

## 목적

Rust에서 CUDA device, context, stream, event와 CUDA C++ kernel launch를 예측 가능하게 제어하는 최소 host runtime을 만든다.

## 언어 경계

```text
Rust
→ ownership, validation, device/context/stream/event lifetime

extern "C" ABI
→ 고정 폭 type과 status code를 사용하는 연결 규약

CUDA C++
→ 실제 smoke kernel과 향후 production GPU operation
```

`extern "C"`는 ABI를 의미하며 kernel 구현 언어가 C라는 뜻이 아니다. 이 단계의 native source는 CUDA C++로 작성한다.

Python, PyTorch extension, Triton JIT를 통해 CUDA를 호출하는 우회 경로는 만들지 않는다.

## 설계 원칙

- CUDA C++와 Rust 사이에는 좁은 `extern "C"` ABI를 둔다.
- raw CUDA error code는 즉시 Rust error로 변환한다.
- global implicit context보다 명시적 device/context ownership을 선호한다.
- default stream 의미에 의존하지 않는다.
- destructor에서 실패를 숨길 수 있는 작업은 explicit close/synchronize API도 제공한다.
- C++ exception이 ABI를 넘어오지 않게 한다.
- Rust panic이 ABI를 넘어가지 않게 한다.
- FFI function은 Python object나 PyTorch tensor를 받지 않는다.

## 최소 타입

```rust
CudaDevice
CudaContext
CudaStream
CudaEvent
CudaModule 또는 KernelHandle
CudaError
DeviceProperties
```

각 타입은 다음을 문서화한다.

- ownership
- thread 이동 가능 여부
- `Send`/`Sync` 여부
- drop 시 동작
- async 작업과의 lifetime 관계

## C ABI 기본 규칙

- opaque handle 또는 CUDA native handle의 소유권을 명시
- `void*`는 device pointer인지 host pointer인지 함수명·문서로 구분
- shape와 byte length는 overflow를 검사할 수 있는 고정 폭 정수 사용
- return은 status code, 상세 오류는 별도 조회 또는 caller buffer 사용
- ABI version과 library build metadata 제공
- struct padding에 의존하는 복잡한 C++ type 노출 금지

예:

```cpp
extern "C" RileyStatus riley_fill_f32(
    float* device_output,
    std::uint64_t element_count,
    float value,
    cudaStream_t stream
);
```

## 구현 범위

1. device enumerate와 property 조회
2. target device 선택
3. stream 생성·동기화
4. event record/wait/elapsed time
5. host callback 또는 동등한 완료 확인 수단
6. CUDA C++ smoke kernel launch
7. kernel launch 직후와 sync 시점 오류 구분
8. native library ABI/build version 조회

Smoke kernel은 성능 목적이 아니다. vector add 또는 buffer fill 정도로 제한한다.

## FFI safety checklist

- [x] null pointer 처리
- [x] length overflow 검사
- [x] host/device pointer 혼동 방지
- [x] stream handle lifetime 보장
- [x] launch parameter 범위 검사
- [x] CUDA error 문자열 포함
- [x] C++ exception이 ABI를 넘지 않음
- [x] panic이 FFI 경계를 넘어가지 않음
- [x] Python 또는 PyTorch object가 API에 없음

## 테스트

### CPU-only

- CUDA 비활성 build
- 오류 enum/format
- invalid device index
- native library가 없을 때 명확한 오류

### GPU

- device 정보 조회
- 두 stream의 event ordering
- async kernel 후 명시적 sync
- 잘못된 launch가 오류로 전달되는지 확인
- 반복 생성/drop 시 resource leak smoke
- Python이 없는 환경에서 동일 테스트 실행

의도적인 illegal-access/assert kernel로 late device fault를 만드는 검사는 이
단계의 7-test smoke에 섞지 않는다. 그런 fault는 CUDA context를 poison하고
`compute-sanitizer`의 zero-error leak gate와 같은 프로세스의 후속 lifecycle
검증을 오염시킨다. PR 03은 정상 fill의 명시적 synchronize와 non-poisoning
invalid launch를 통해 `LAUNCH`/`SYNCHRONIZE` stage 경계를 고정한다. 실제
device-fault injection은 별도 프로세스 격리와 복구 정책을 함께 검증하는
[PR 16 오류 격리 gate](16-reliability-and-release.md#오류-격리)에서 수행한다.

## 비범위

- 범용 tensor
- allocator pool
- cuBLASLt
- CUTLASS
- model operation
- CUDA Graph
- Triton
- NVRTC

## PR 계약 요약

- **문제:** PR 02의 native library는 version ABI만 제공해 device/context/stream/event
  ownership이나 kernel 완료 시점, CUDA 오류 단계를 Rust에서 안전하게 표현할 수 없었다.
- **범위:** additive C ABI v1, primary-context lease, non-default stream와 timing event,
  AOT diagnostic fill kernel, 오류 domain/stage, safe Rust lifecycle API와 CPU/GPU gate를
  하나의 host-runtime 경계로 구현했다.
- **의미 보존 등급:** `N/A`. 모델 연산이나 수학적 최적화를 추가하지 않는다. Smoke
  kernel은 전달된 `f32` bit pattern을 그대로 저장하는 진단 연산이며 성능 주장을 하지
  않는다.
- **설계 결정:** primary context를 retain/release하되 reset하지 않고, 기존 current
  context는 push/pop으로 복원한다. 복원이 모호해지면 runtime을 단조롭게 poison해 이후
  호출을 fail closed한다. 비동기 fill은 originating stream을 소유한 pending token으로
  완료 전 buffer 수명을 보장한다.
- **롤백:** Cargo의 `cuda` feature를 비활성화하면 CPU workspace는 native library를
  build/link하지 않는다. 전체 host-runtime 경계는 선행 snapshot `226445d`로 독립적으로
  되돌릴 수 있으며 PR 04 이상의 tensor/model interface에는 의존하지 않는다.

## 구현 및 검증 evidence — 2026-08-25

검증된 source snapshot은 commit
`ca1cb350614a5fd21a67fa6889300c0daea67c49`이다. 이후 follow-up documentation은 이
evidence 절과 safe lifecycle 사용 예제만 바꾸며 구현 동작을 변경하지 않는다. 로컬에서는
CUDA, GPU, model inference를 실행하지 않고 CPU/model-free 및 정적 gate만 통과했다.

```text
cargo fmt --all -- --check
python3 ci/check_workspace_boundaries.py --locked
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo test --locked --workspace --no-default-features
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --no-default-features
ci/check_feature_matrix.sh                         # CUDA 외 8개 조합
ci/check_workspace_without_research_tools.sh       # 연구 도구 없는 fresh copy
sh -n ci/verify_python_free_cuda.sh ci/verify_python_free_gpu_runtime.sh
C11/C++17 header syntax와 workflow YAML parse
```

CUDA compile/link와 GPU 실행은 `server-4096`에서만 수행했다. Compile image build에는
GPU를 전달하지 않았고, 실제 GPU gate는 `--gpus all --network none`으로 분리했다.
Compile gate는 CUDA feature를 켠 `riley-cuda` all-target Clippy까지 포함한다. Runtime
gate는 Python이 없는 image에서 정확히 일곱 GPU test를 실행하며 model, cuBLAS,
CUTLASS, Triton, NVRTC를 호출하지 않는다.

```text
source commit:       ca1cb350614a5fd21a67fa6889300c0daea67c49
archive command:     git archive --format=tar HEAD
source archive sha:  05569ea0fe9383c69b1be81eeae218eb79273976fb442efd8cd9638032de5c37
Rust image amd64:    rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03
CUDA image amd64:    nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
final image id:      sha256:67a5dd96128bfdab335a56bce8d349a1af2307732c7705d7c4cf5679bd8f973b
rustc / cargo:       1.85.0 / 1.85.0
nvcc:                12.8.93 (AOT architecture 89)
compute-sanitizer:   2025.1.0.0 build 35583870 (동일 image ID의 사후 version query)
host / driver:       Linux 6.8.0-138-generic / 580.173.02
device:              NVIDIA GeForce RTX 4090, compute capability 8.9
device memory:       25,250,627,584 bytes; 128 multiprocessors
driver/runtime API:  13000 / 12080
normal GPU tests:    7 passed, 0 failed; lifecycle iterations 128
normal free memory:  24,594,284,544 -> 24,594,284,544 bytes
sanitizer tests:     7 passed, 0 failed; lifecycle iterations 32
sanitizer memory:    24,481,038,336 -> 24,481,038,336 bytes
sanitizer summary:   0 bytes leaked in 0 allocations; 0 errors
external artifact:   server-4096:/home/psyche/rustinfer-artifacts/pr03/ca1cb350614a5fd21a67fa6889300c0daea67c49/
```

최종 외부 artifact의 integrity는 두 evidence manifest에 대해 `sha256sum -c`로
재검증했다.

```text
docker-build.log:             6bc07b0753c285a7e651d02fc552430ed79f7fb0469633506b341d30323d5c9f
gpu-run.log:                  167383f0a133b50f8b659ab0140cc8a12a770eea9a471d768d0ac6a7ca0beac5
sanitizer-run.log:            b3bd3926e8a372ea16fd313b14b0f31cdfec74b9c516e7b4d8114fa3740caba5
gpu/SHA256SUMS:               47ff59e6118729797a2923899c783a01315d0c1f4ce11737ae70bd34aa120f7e
sanitizer/SHA256SUMS:         b974ab62c7cfea35cf98d98e39a662b5fbd2a6d5da41e2be948dd77727f7b05c
```

실패 이력도 보존했다. Snapshot `5bbe5f9`는 GPU-free compile image에 실제 driver
`libcuda.so.1`이 없다는 ABI metadata smoke 문제를 검출했다. 수정은 해당 smoke 실행
동안에만 toolkit stub의 SONAME alias를 사용하며 실제 GPU verifier는 주입된 driver를
계속 요구한다. Snapshot `cf993a1`은 정상 GPU 7/7과 누수 0을 통과했지만 의도적으로
assert한 invalid-launch API status 두 건을 sanitizer error summary에 포함했다. 이후
`--report-api-errors no`로 sanitizer의 API-error reporting channel을 끈다. Suite는 모든
CUDA API 결과를 test assertion과 process exit로 계속 검사하며 memory/leak 검사는
그대로 유지한다.
Snapshot `6120b78`은 runtime gate 후 별도 feature-on Clippy에서 test lint 다섯 건을
검출했다. 최종 snapshot은 이를 수정하고 동일 Clippy를 compile gate에 편입했다.

### PR 크기 예외

PR 02 tip `226445d` 대비 최종 PR diff는 총 `+4,737/-126`줄이다. 분류하면
production/build/ABI `+3,454/-82`, tests `+310/-0`, docs `+440/-30`, CI/workflow
`+533/-14`다. 권장 hand-written production `200~800`줄을 넘지만, 이 PR은 하나의
질문—Rust가 CUDA host resource와 asynchronous completion을 어떤 ABI와 ownership
계약으로 안전하게 제어하는가—만 다룬다. C header와 CUDA 구현, 동일 layout을 검증하는
Rust FFI, safe wrapper, lifecycle/failure test와 Python-free verifier는 함께 검토하고
롤백해야 한다. 이를 더 쪼개면 safe wrapper 없는 native ABI 또는 검증되지 않은
ownership API가 중간 단계에 남는다. 범용 tensor, allocator, model operation과 PR 04
이후 subsystem은 포함하지 않았다. 따라서 파일 수보다 interface 변화와 독립 검증·롤백을
우선하는 PR 00 예외 조건을 적용한다.

## 완료 기준

- [x] Rust에서 device/stream/event lifecycle이 safe API로 노출됨
- [x] `unsafe`가 FFI module 밖으로 새지 않음
- [x] CUDA C++ smoke kernel 결과가 정확함
- [x] C ABI와 ABI version이 문서화됨
- [x] compute capability와 memory 정보가 benchmark metadata로 출력됨
- [x] Python 없는 환경에서 host runtime test 통과
- [x] CUDA 미설치 환경 오류가 명확함

구현 gate는 통과했다. 이 문서는 선행 PR과 함께 merge되기 전까지 `Active`, merge
후 `Complete`로 전환한다.

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)
