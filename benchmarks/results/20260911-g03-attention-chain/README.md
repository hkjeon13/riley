# G03 norm/QKV/RoPE/KV → attention — local implementation

## 구현 범위

기존 KV-write graph 뒤에 canonical grouped attention을 연결했다. 기존 eager 경로와 동일하게 query/KV head 비율에 따라 shared-KV GQA 또는 grouped-head kernel을 선택한다. D64/M=1/logical block 0, 하나의 고정 physical block 및 valid prefix에 한정한다. 여러 logical block이나 일반 packed batch admission을 지원하지 않는다.

실제 packed metadata parent의 sequence offsets/physical id/valid prefix/row slot을 매 replay에 고정 binding 값으로 다시 채운다. position은 사용자 payload에서 새로 전달하고 기존 범위 검사를 적용한다. 모든 metadata 영역의 크기·alignment·상호 overlap을 capture 전에 검증한다. 새로운 attention output parent도 같은 ledger에 등록하며 기존 입력·weight·cache와 alias를 거절한다. 결과는 attention output과 K/V byte 배열이다.

모델 audit는 zero replay로 slot 0을 쓴 뒤 original token을 현재 slot에 쓰는 순서를 CPU scatter로 구성하고 기존 eager grouped-attention API로 예상 출력을 계산한다. eager가 변경한 cache를 원래대로 복원한 뒤 graph를 실행한다. nonfinite eager output을 거절한다. attention scratch·전체 cache·metadata 복원 및 이전 operator audit를 유지한다.

## 로컬 검증

- CPU 391 passed: CUDA lib 84 + graph contract 32, runtime lib 259 + architecture 15 + inventory 1.
- C11 ABI syntax, fmt/diff, 일반 CPU Clippy 통과. 기존 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. 원격 기존 9파일 해시가 직전 KV-write 검증 상태와 일치.
- **새 native CUDA와 CUDA feature Rust는 아직 빌드·실행하지 않았다.** CPU 결과는 이 GPU 조건부 경로의 컴파일·정확성을 증명하지 않는다.
- native fixture에 metadata overlap/범위, logical block 0 제한과 attention capture/close 검사를 추가했다. 초기화되지 않은 fixture weights/tables는 실행하지 않는다. 미실행 상태다.
- 실제 모델 attention chain 5 case × 4 iteration × 32 = 640회 eager parity와 cache CPU oracle·복원을 준비했다. 기존 KV/RoPE/QKV/tail/MLP 및 ledger/SwiGLU GPU 회귀도 필요하다. 미실행 상태다.

## 승인 대상

`ai-assistant:/tmp/riley-g01-native-260910`에 아래 소스 9개만 전송하고 CUDA 빌드·GPU 검증을 수행한다. 검증 중 해당 파일 내 수정·재전송을 포함한다. 모델·비밀정보·배포 파일은 포함하지 않는다. [파일별 전후 해시](transfer-scope.json).

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/src/batch_primitives.cu`
- `kernels/src/ffi_internal.hpp`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/graph_decode_qkv_audit.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`

자동 승인 검토가 이번 새 payload와 목적지에 대해 명시 승인이 필요하다며 전송을 거절했다. 이번 단계에서 원격 쓰기를 수행하지 않았다.

## 남은 절차

승인 후 attention eager parity·cache/metadata 복원·기존 회귀를 검증한다. 이후 layer tail과 앞부분을 연결하고 실제 layer 입력, 모든 layer/embedding/final norm/head/output을 통합해야 한다. 동적 block mapping/valid prefix, 오류·완료·수명, bucket/admission/fallback qualification도 남아 있다. 현재 코드는 완료된 scratch 기반 부분 graph 진단이다. 전체 decode/G02H/G03 완료가 아니며 성능 측정은 시작하지 않았다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [abi.log](abi.log), [clippy.log](clippy.log), [format-diff.log](format-diff.log), [remote-before.txt](remote-before.txt), [model-preservation.json](model-preservation.json).
