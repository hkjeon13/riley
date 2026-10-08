# PR 04 — Tensor 표현과 GPU 메모리 수명

**상태:** Active
**선행 조건:** [PR 03](03-cuda-host-runtime.md)  
**다음:** [PR 05 — 모델 로딩과 IR](05-model-loading-and-ir.md)

[← 이전](03-cuda-host-runtime.md) | [목차](README.md) | [다음 →](05-model-loading-and-ir.md)

## 목적

모델 연산 전에 device memory, shape, stride, dtype, view와 async lifetime을 명확히 표현한다.

## 핵심 타입

```rust
DType
Shape
Strides
Layout
DeviceBuffer<T 또는 untyped>
PinnedHostBuffer
TensorView<'a>
TensorViewMut<'a>
Workspace
```

## 필수 invariant

- logical element count와 byte length가 일치한다.
- view는 소유 buffer보다 오래 살 수 없다.
- mutable alias를 만들지 않는다.
- kernel 실행이 끝나기 전에 backing storage가 해제되지 않는다.
- reshape는 element order가 유지될 때만 zero-copy다.
- transpose와 contiguous materialization을 구분한다.
- dtype cast가 암시적으로 발생하지 않는다.

## Allocation 단계

이번 PR에서는 단순하고 검증 가능한 allocator를 사용한다.

1. direct device allocation/free wrapper
2. 선택적 pinned host allocation
3. workspace 명시 전달
4. allocation accounting

Caching allocator와 paged KV pool은 아직 구현하지 않는다.

## Layout 표준

초기 canonical layout을 문서로 고정한다.

예:

```text
Hidden: [batch, sequence, hidden]
Q/K/V: [batch, heads, sequence, head_dim]
Weights: checkpoint 원본과 execution packed layout 구분
```

각 kernel은 지원 stride/layout을 capability로 선언한다. 지원하지 않는 layout을 조용히 copy하지 말고 planner가 명시적으로 변환한다.

## Async lifetime 전략

이번 PR은 **execution step에 해당하는 pending copy token이 stream과 두 buffer를
완료까지 보유**하는 전략을 선택한다.

- safe Rust `CudaPendingH2D`/`CudaPendingD2H`가 originating stream,
  device buffer, pinned host buffer를 실제 `&mut`로 borrow한다.
- native copy token도 세 resource에 persistent active-use reservation을 둔다.
- 정상 query/synchronize와 context-stack 복원이 모두 성공한 뒤에만 reservation을
  해제한다.
- pending 값을 `mem::forget`하거나 완료 확인이 실패하면 token, accounting,
  context child lease를 남겨 reuse/free를 영구 거부한다.
- 한 resource에는 동시에 copy 하나만 허용하며 완료 뒤에만 다른 stream으로
  명시적으로 handoff한다.

중요한 것은 Rust borrow가 host lifetime만 보장하고 GPU 실행 완료까지 자동 보장하지 않는다는 점이다.

상세 계약은 [Tensor와 CUDA memory invariant](../docs/tensor-memory-invariants.md)와
[CUDA C ABI v1](../docs/cuda-abi-v1.md)에 고정한다. Stream-ordered allocator,
deferred-free queue와 caching pool은 도입하지 않는다.

## 테스트

- allocation/free accounting 0으로 복귀
- overflow와 zero-sized tensor
- view slicing과 offset
- reshape 가능/불가능 케이스
- non-contiguous transpose 표현
- host↔device copy round trip
- 서로 다른 stream에서 잘못된 조기 free 방지

정상·validation 경로와 Compute Sanitizer는 이번 PR에서 검증한다. 반면 rollback
`cudaFree*`, deferred copy 오류, context restoration 실패를 강제로 만드는 test-only
fault backend는 공유 primary context와 leak gate를 오염시키지 않도록
[PR 16 오류 격리 gate](16-reliability-and-release.md#오류-격리)에서 별도 process로
구현한다. 해당 분기는 이번 PR에서 fail-closed source audit 대상으로 고정한다.

## 비범위

- model-specific tensor
- weight packing
- memory pool 최적화
- KV block allocator
- unified memory

## PR 계약 요약

- **문제:** PR 03의 diagnostic smoke buffer만으로는 model tensor의 dtype/shape/stride,
  backing lifetime, 범용 device/pinned allocation과 async copy 완료를 표현할 수 없었다.
- **범위:** checked tensor metadata와 borrowed view, 명시적 workspace, opaque untyped
  device/pinned buffer, context별 coherent accounting, stream-ordered H2D/D2H completion
  token을 구현한다.
- **의미 보존 등급:** `N/A`. Model operation이나 수학적 최적화를 구현하지 않으며
  copy round trip은 입력 byte를 그대로 보존해야 한다.
- **설계 결정:** reshape는 contiguous zero-copy만, transpose는 metadata-only만
  허용한다. mutable layout은 non-overlap을 보수적으로 증명해야 한다. async 안전은
  Rust borrow와 native active-use token을 함께 사용하고 모호한 실패에서는 leak을
  선택해 fail closed한다.
- **롤백:** `riley-tensor`의 `cuda` feature를 끄면 metadata는 host storage만으로
  동작하고 native CUDA를 build/link하지 않는다. PR 04 전체는 PR 03 evidence commit
  `9428af8` 위의 독립 snapshot으로 되돌릴 수 있다.

## 구현 및 검증 evidence — 2026-08-25

검증된 source snapshot은 commit
`c6c93e2d0e77082c46d15c50401db0f4e181a7f3`이다. 이후 follow-up은 이 evidence 절,
완료 checklist와 PR 16의 deferred fault gate 문서만 바꾸며 구현 동작을 변경하지
않는다. 로컬에서는 CUDA, GPU, model inference를 실행하지 않고 CPU/model-free 및
정적 gate만 통과했다.

```text
cargo fmt --all -- --check
python3 ci/check_workspace_boundaries.py --locked
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo test --locked --workspace --all-targets --no-default-features   # 25 passed
cargo test --locked -p riley-cuda --doc --no-default-features     # 9 compile-fail passed
cargo test --locked -p riley-tensor --doc --no-default-features   # 3 compile-fail passed
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --no-default-features
ci/check_feature_matrix.sh                         # CUDA 외 8개 조합
ci/check_workspace_without_research_tools.sh       # 연구 도구 없는 fresh copy
sh -n ci/verify_python_free_cuda.sh ci/verify_python_free_gpu_runtime.sh
C11/C++17 public header syntax와 workflow YAML parse
```

CUDA compile/link와 GPU 실행은 `server-4096`에서만 수행했다. Compile image build에는
GPU를 전달하지 않았고, 실제 GPU gate는 `--gpus all --network none`으로 분리했다.
Compile gate는 CUDA feature를 켠 `riley-cuda`와 `riley-tensor`의 all-target
Clippy, 두 GPU test binary와 tensor CUDA surface의 no-run build를 포함한다. Runtime
gate는 Python이 없는 동일 immutable image에서 PR 03의 일곱 host-runtime test와 PR
04의 다섯 memory test를 각각 정확한 inventory로 실행하며 model, cuBLAS, CUTLASS,
Triton, NVRTC를 호출하지 않는다.

```text
source commit:       c6c93e2d0e77082c46d15c50401db0f4e181a7f3
archive command:     git archive --format=tar HEAD
source archive sha:  f3b4c83b671b061bce6d9c46b6c493f30531ea66dc0868520523d71797250a61
Rust image amd64:    rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03
CUDA image amd64:    nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
final image id:      sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849
rustc / cargo:       1.85.0 / 1.85.0
nvcc:                12.8.93 (AOT architecture 89)
compute-sanitizer:   2025.1.0.0 build 35583870 (동일 image ID의 사후 version query)
host / driver:       Linux 6.8.0-138-generic / 580.173.02
device:              NVIDIA GeForce RTX 4090, compute capability 8.9
device memory:       25,250,627,584 bytes; 128 multiprocessors
driver/runtime API:  13000 / 12080
normal host tests:   7 passed, 0 failed; lifecycle iterations 128
normal memory tests: 5 passed, 0 failed; accounting marker all-zero exactly once
normal free memory:  24,594,284,544 -> 24,594,284,544 bytes
sanitizer host:      7 passed, 0 failed; lifecycle iterations 32
sanitizer memory:    5 passed, 0 failed; accounting marker all-zero exactly once
sanitizer summary:   각 binary 0 bytes leaked in 0 allocations; 0 errors
external artifact:   server-4096:/home/psyche/rustinfer-artifacts/pr04/c6c93e2d0e77082c46d15c50401db0f4e181a7f3/
```

최종 외부 artifact의 integrity는 normal GPU와 sanitizer evidence manifest 각각에
`sha256sum -c`를 적용해 모든 파일을 재검증했다.

```text
docker-build.log:             aec90a8f0eabf6ac70f59eee8e1b4a140c8a3af6ad88bd3776cfabd8b0bca556
gpu-run.log:                  dace99d5c322c5c929f49845c5cddda7c7e0cf30d898e6ab1aa139b4bebf665f
sanitizer-run.log:            1e461f9cac76d8df5b85503fc2c0941fc31a1342c37637dece7a194bd96e789d
gpu/SHA256SUMS:               0f548aefac55b01016b96aac5155da1c03548a58ea91023aecf5a42fdf43b74b
sanitizer/SHA256SUMS:         26f8e00c400ad6236e013950a3d1cbe7bea8a3c8e28bc706e8efc6630a5813f7
```

실패 이력도 commit별 별도 artifact로 보존했다. Snapshot `a76c68a`는 첫 원격
feature-on build에서 guard가 Rust exhaustiveness 판정에 포함되지 않는 match를
검출했으며 GPU는 실행하지 않았다. Snapshot `64006a4`는 compile/link와 test binary
생성 뒤 feature-on Clippy 이름 lint 세 건을 검출했으며 역시 GPU는 실행하지 않았다.
Snapshot `d63dbec`은 compile gate와 실제 GPU test 7/7 + 5/5를 통과했지만 libtest가
test 이름과 accounting marker를 같은 줄에 출력해 exact-line evidence parser가 이를
0회로 판정했다. 최종 snapshot은 marker를 독립된 한 줄로 출력하고 전체 compile,
normal GPU, sanitizer gate를 새 image와 evidence directory에서 처음부터 통과했다.

### PR 크기 예외

PR 03 evidence commit `9428af8` 대비 검증된 source snapshot diff는 총
`+4,592/-106`줄이다. 분류하면 production/build/ABI `+3,515/-28`, tests
`+489/-2`, docs `+334/-21`, CI/workflow `+254/-55`다. 권장 hand-written production
`200~800`줄을 넘지만, 이 PR은 하나의 질문—model execution이 읽는 tensor view와
asynchronous CUDA buffer를 하나의 ownership·byte-range·completion contract로 어떻게
안전하게 표현하는가—만 다룬다. Checked metadata와 borrowed view, C ABI/native
allocation, safe Rust owner, pending copy token, accounting/failure contract와 Python-free
GPU verifier는 함께 검토하고 롤백해야 한다. 이를 더 쪼개면 backing owner 없는 view나
완료 수명을 증명하지 못하는 async allocation API가 중간 단계에 남는다. Caching
allocator, KV pool, model-specific tensor, weight packing과 model operation은 포함하지
않았다. 따라서 파일 수보다 interface 변화와 독립 검증·롤백을 우선하는 PR 00 예외
조건을 적용한다.

## 완료 기준

- [x] ownership과 async lifetime invariant 문서화
- [x] tensor metadata unit test 완비
- [x] GPU copy round trip 정확성
- [x] allocation 통계 조회 가능
- [x] implicit contiguous/cast가 없음

구현 gate는 통과했다. 이 문서는 선행 PR과 함께 merge되기 전까지 `Active`, merge
후 `Complete`로 전환한다.

[← 이전](03-cuda-host-runtime.md) | [목차](README.md) | [다음 →](05-model-loading-and-ir.md)
