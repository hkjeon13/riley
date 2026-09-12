# G03 MLP chain — local implementation, remote validation pending

## 구현 범위

정규화된 입력과 residual을 staged H2D로 받아 gate/up GEMM → SiLU → multiply → down GEMM → residual add → staged D2H를 한 graph로 capture하는 진단 경로를 추가했다. 실제 선택된 intermediate/down GEMM plan과 알고리즘·정책을 유지한다. 공유 parent는 기존 ledger로 보유하며, capture 종료 또는 실행 완료 여부가 불확실하면 graph·stream·parent 자원을 보유한다.

실제 모델의 완료된 scratch snapshot과 마지막 layer weight를 사용해 독립 eager MLP 출력과 byte parity를 비교하고, zero/original 입력을 교대해 32회 replay하는 audit를 추가했다. 이는 실제 마지막 layer 실행 도중 입력을 연결한 전체 decode graph가 아니다. scratch/pinned 복원, plan 불변성, 오류 시 executor poison을 검사한다.

## 확인된 결과와 한계

- CPU 테스트 391 passed: CUDA crate 84 + graph contract 32, runtime 259 + architecture 15 + inventory 1.
- C ABI syntax check, cargo fmt, git diff --check 및 일반 CPU Clippy 성공. 기존 Clippy 경고는 남아 있다.
- 무관한 model-loader 6개 파일의 전후 SHA-256 일치.
- CUDA native 및 CUDA feature Rust 코드는 아직 빌드하지 못했다. CPU 통과는 새 GPU 경로의 컴파일·정확성을 증명하지 않는다.
- 준비된 native fixture: geometry/alias/ledger/중복 capture 거절, optional workspace, close/Drop 및 capture domain 해제 검사. 미실행.
- 준비된 모델 검사: 5개 case × 4회 × 32 replay = 640회 eager parity와 기존 continuation/logits/KV 회귀. 미실행.
- 전체 decode graph, G02H/G03 qualification, vLLM 대비 성능 향상은 미검증.

## 원격 전송 승인 대상

대상: `ai-assistant:/tmp/riley-g01-native-260910`. 다음 소스 11개만 전송하는 payload다. 모델·비밀정보·배포 파일은 포함하지 않는다. 파일별 전후 해시는 [transfer-scope.json](transfer-scope.json)에 있다.

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/tests/abi_layout.c`
- `kernels/src/ffi_internal.hpp`
- `kernels/src/primitives.cu`
- `kernels/src/gemm.cu`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/batch_executor.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`
- `crates/riley-runtime/src/llama/graph_decode_mlp_audit.rs`

자동 승인 검토가 이전 9개 파일을 넘어선 이번 11개 소스의 전송과 원격 반영에는 명시 승인이 필요하다는 사유로 거절했다. 이번 단계에서 원격 쓰기는 수행하지 않았다. 승인 후 이 payload를 전송하고 CUDA 빌드, native fixture, 실제 모델 eager parity와 기존 C07 회귀를 실행한다. 검증 과정에서 이 11개 파일 내 수정·재전송이 필요할 수 있다.

## 증거

[source-hashes.json](source-hashes.json), [model-preservation.json](model-preservation.json), [cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [abi.log](abi.log), [clippy.log](clippy.log), [format-diff.log](format-diff.log).
