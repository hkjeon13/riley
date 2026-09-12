# G03 post-attention norm + MLP — local implementation

## 구현 범위

기존 MLP graph API를 유지하고 `record_norm_mlp`를 추가했다. 동일 parent ledger와 capture/완료/해제 계약 아래 residual 입력 → RMSNorm → gate/up GEMM → SiLU/multiply → down GEMM → residual add를 연결한다. 기존 staged payload의 첫 hidden 영역은 scratch이며 norm 결과로 덮어쓴다. residual 영역은 매 replay 새로 공급한다.

canonical BF16 norm과 HF SmolLM2 norm은 각각 기존 eager 커널을 사용한다. HF는 hidden 576/epsilon 1e-5로 제한한다. 등록되지 않은 norm weight, 다른 parent와의 alias, 크기 불일치, nonfinite/nonpositive epsilon 및 잘못된 profile을 capture 전에 거절한다. FixedContiguous37Balanced profile은 명시적으로 거절한다.

실제 모델 audit에 독립 eager norm→MLP 결과와 graph 결과 비교를 추가했다. 마지막 layer의 실제 weight와 완료된 scratch snapshot을 사용하며, 기존 MLP-only audit도 유지한다. 이는 실행 중 모든 layer의 입력을 연결한 full decode 경로가 아니다.

## 로컬 검증

- CPU 391 passed: CUDA crate 84 + graph contract 32; runtime 259 + architecture 15 + capture inventory 1.
- C11 ABI syntax, fmt/diff 및 일반 CPU Clippy 성공. 기존 경고는 남아 있다.
- 무관한 model-loader 6파일 전후 SHA-256 일치.
- CUDA feature Rust 및 native CUDA 빌드/실행은 미수행. CPU 테스트 통과는 새 GPU 경로 컴파일·정확성의 증거가 아니다.
- native fixture에 norm weight/epsilon/profile 거절과 norm+MLP close/Drop 검사를 추가했으며 미실행 상태다.
- 실제 모델 norm+MLP 5 case × 4 iteration × 32 replay = 640회 검사를 준비했다. 기존 MLP-only 640회와 C07 회귀도 함께 실행해야 한다.

## 원격 전송 승인 대상

대상 `ai-assistant:/tmp/riley-g01-native-260910`. 변경된 소스 9개만 전송한다. 파일별 변경 전/후 해시는 [transfer-scope.json](transfer-scope.json)에 기록했다.

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/tests/abi_layout.c`
- `kernels/src/ffi_internal.hpp`
- `kernels/src/primitives.cu`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/graph_decode_mlp_audit.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`

원격의 전송 전 9개 파일 해시는 이전 MLP 검증 상태와 일치한다. 자동 승인 검토가 이번 새 정규화-MLP payload에 명시 승인이 필요하다며 전송을 거절했다. 이전 11파일 승인과 후속 진행 요청은 새 구현 전송 승인으로 인정되지 않았다. 이번 단계 원격 쓰기는 수행하지 않았다.

승인 요청 범위는 위 9개 파일 전송, CUDA 빌드·native/model GPU 검증, 그 검증 과정의 해당 파일 내 수정·재전송이다. 모델·비밀정보·배포 파일은 payload에 포함하지 않는다. 승인 후 최종 원격 해시를 확인하고 GPU 20개 회귀와 norm+MLP 640회 정확성을 검증한다.

전체 layer/model DAG 통합, G02H/G03 qualification 및 동일 조건 SmolLM2 vLLM 성능 비교는 남아 있다. 성능 향상을 주장하지 않는다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [clippy.log](clippy.log), [abi.log](abi.log), [format-diff.log](format-diff.log), [model-preservation.json](model-preservation.json), [remote-before.txt](remote-before.txt).
