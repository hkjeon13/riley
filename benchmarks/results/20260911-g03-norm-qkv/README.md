# G03 input norm → Q/K/V — local implementation

## 구현

input norm과 Q/K/V selected GEMM을 단일 retained graph로 capture하는 `record_norm_qkv`를 추가했다. M=1 canonical/HF SmolLM2 BF16 norm을 지원하며 Q와 K/V width 차이를 처리한다. input/norm/Q/K/V/norm weight/QKV weights 및 optional workspace는 동일 native ledger의 실제 parent다. alias, 등록되지 않은 부모, shape, epsilon/profile, selected plan을 capture 전에 검사한다.

공유 native capture 생명주기를 내부 동기 함수 `record_reserved_sequence`로 추출했다. 기존 MLP/norm-MLP/layer-tail과 새 QKV 모두 동일 TLS/domain guard, 종료 불확실 시 자원 보유, instantiate 및 완료 계약을 사용한다. Rust/user callback은 capture 중 호출하지 않는다. 이 추출의 CUDA 회귀는 아직 미검증이다.

replay payload 길이는 Q+K+V 결과 바이트 수이며 첫 hidden 바이트가 새 입력, 나머지는 padding이다. H2D는 실제 input만 복사한다. 완료 후 packed Q/K/V 결과를 읽는다. pinned buffer는 payload의 두 배 이상이어야 한다.

실제 모델에서 완료된 scratch input과 마지막 layer의 actual norm/QKV weight를 사용한 독립 eager byte parity audit를 추가했다. 매회 zero/original 입력을 교대하고 plan 불변성, scratch/pinned 복원을 확인한다. Q/K/V bias 및 FixedContiguous37Balanced profile은 거절한다. RoPE나 attention을 연결한 실제 layer trace가 아니다.

## 로컬 검증과 한계

- CPU 391 passed: CUDA lib 84 + graph contract 32, runtime lib 259 + architecture 15 + capture inventory 1.
- C11 ABI syntax, fmt/diff, 일반 CPU Clippy 성공. 기존 경고는 남아 있다.
- 무관한 model-loader 6파일 전후 SHA-256 일치.
- 원격 기존 7파일 해시는 직전 layer-tail 검증 상태와 일치하며 새 audit 파일은 없다.
- **CUDA native 및 CUDA feature Rust 빌드/실행 미수행.** CPU 통과는 새 GPU 코드 컴파일·정확성의 증거가 아니다.
- 준비된 native GPU fixture: 다른 Q/KV 폭, 잘못된 epsilon/HF geometry/plan, 부모 미등록/alias/짧은 staging, 중복 capture/완료 전 read 거절, optional workspace, close/Drop 및 domain 해제. 초기화되지 않은 weight 실행은 하지 않는다.
- 준비된 실제 모델 norm-QKV: 5 case × 4 iteration × 32 replay = 640회. 기존 layer-tail/norm-MLP/MLP와 C07/ledger/SwiGLU 회귀도 필요하다. 모두 이번 변경에서는 미실행이다.

## 승인 대상

`ai-assistant:/tmp/riley-g01-native-260910`에 아래 소스 8개만 전송하고 CUDA 빌드·GPU 검증을 수행한다. 검증 중 해당 파일 내 수정·재전송을 포함한다. 모델·비밀정보·배포 파일은 포함하지 않는다. [파일별 전후 해시](transfer-scope.json).

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/batch_executor.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`
- `crates/riley-runtime/src/llama/graph_decode_qkv_audit.rs`

자동 승인 검토가 이전 7파일 승인과 포괄적 진행 요청을 새 8파일 payload의 명시 승인으로 인정하지 않아 전송을 거절했다. 이번 단계의 원격 쓰기는 수행하지 않았다.

## 남은 절차

승인 후 CUDA 빌드와 native/model 정확성·기존 회귀를 검증한다. 이후 RoPE→KV write→attention, 실제 layer 앞/뒤 구간 연결, 모든 layer와 embedding/final norm/head/output 통합, fresh metadata·완료/오류·자원 수명·bucket/fallback qualification이 남아 있다. 전체 decode/G02H/G03 완료가 아니며 성능 측정은 시작하지 않았다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [abi.log](abi.log), [clippy.log](clippy.log), [format-diff.log](format-diff.log), [remote-before.txt](remote-before.txt), [model-preservation.json](model-preservation.json).
