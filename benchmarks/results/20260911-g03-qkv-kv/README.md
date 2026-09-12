# G03 norm/QKV/RoPE → KV write — local implementation

## 구현 범위

기존 QKV/RoPE graph 뒤에 rotated K 및 V를 실제 cache parent의 지정 layer/physical block에 쓰는 native BF16 scatter를 연결했다. D64, M=1이다. cache layout은 기존 eager scatter와 동일한 `[layer,physical block,KV head,16 tokens,64 depth]` 순서다.

이번 진단 graph의 logical/physical block mapping과 valid-token prefix는 capture 때 고정한다. native가 parent 등록·alias·cache 전체 크기·overflow·layer/block/prefix 범위를 검사하고, replay 위치가 고정 logical block이나 유효 prefix 밖이면 launch 전에 거절한다. mapping/prefix 변경은 재capture가 필요하다. **일반 packed batch admission이나 block 경계를 넘는 retained decode 지원이 아니다.** host mapping은 실제 rows를 pack한 결과로부터 도출한다. position만 기존 actual packed device parent에 매 replay 갱신한다.

실제 모델 audit는 마지막 layer에서 zero/original 입력을 교대하고, 종료 후 cache parent 전체를 독립 CPU scatter oracle과 비교하도록 추가했다. zero replay와 original replay의 K/V 결과를 각각 적용해 다른 위치·다른 layer 변경을 검사한다. 그 후 전체 key/value cache와 scratch, packed position metadata를 원상복원하고 확인한다. 실패 시 executor를 poison한다.

## 로컬 검증

- CPU 391 passed: CUDA lib 84 + graph contract 32, runtime lib 259 + architecture 15 + inventory 1.
- C11 ABI syntax, fmt/diff, 일반 CPU Clippy 통과. 기존 경고는 남아 있다.
- 무관한 model-loader 6파일 SHA-256 보존. 원격 기존 9파일 해시가 직전 QKV-RoPE 검증 상태와 일치.
- **새 CUDA native 및 CUDA feature Rust는 아직 빌드/실행하지 않았다.** CPU 통과는 새 GPU 코드 컴파일·정확성을 증명하지 않는다.
- native fixture에 cache shape/layer/block/prefix/alias 거절 및 유효 prefix 밖 replay 거절을 추가했다. 초기화되지 않은 fixture weight/table은 실행하지 않는다. GPU 미실행 상태다.
- 실제 모델 5 case × 4 iteration × 32 = 640회 KV 통합 replay와 전체 cache CPU oracle·복원 검사를 준비했다. 기존 QKV/RoPE/tail/MLP 및 ledger/SwiGLU 회귀도 필요하다. 아직 실행하지 않았다.

## 원격 승인 대상

`ai-assistant:/tmp/riley-g01-native-260910`에 아래 소스 9개만 전송한다. CUDA 빌드·GPU 검증 및 해당 파일 내 수정·재전송을 포함한다. 모델·비밀정보·배포 파일은 포함하지 않는다. [파일별 전후 해시](transfer-scope.json).

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/src/batch_primitives.cu`
- `kernels/src/ffi_internal.hpp`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/graph_decode_qkv_audit.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`

자동 승인 검토가 이번 구체적인 payload와 목적지에 대한 명시 승인이 필요하다며 전송을 거절했다. 이번 단계에서 원격 쓰기는 수행하지 않았다.

## 남은 절차

승인 후 CUDA 빌드·KV CPU scatter oracle·복원·기존 GPU 회귀를 검증한다. 이후 attention, layer tail, 실제 layer 입력과 모든 layer/embedding/final norm/head/output을 통합해야 한다. 동적 block mapping/prefix 갱신과 bucket/admission/fallback, 오류/완료·자원 수명 qualification도 남아 있다. 현재 코드는 완료된 scratch 기반 부분 graph 진단이다. 전체 decode/G02H/G03 완료가 아니며 성능 측정은 시작하지 않았다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [abi.log](abi.log), [clippy.log](clippy.log), [format-diff.log](format-diff.log), [remote-before.txt](remote-before.txt), [model-preservation.json](model-preservation.json).
