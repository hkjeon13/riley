# G03 norm/QKV → RoPE — local implementation

## 구현

기존 norm-QKV 경로를 유지하며 Q/K projection 뒤에 기존 BF16 indexed RoPE kernel을 연결했다. D64/full rotary, M=1 범위다. rotated Q/K, 실제 F32 cos/sin 및 packed position parent를 같은 resource ledger에 등록한다. extra parent alias, 크기·alignment·offset·테이블 geometry를 capture 전에 검증한다.

매 replay payload의 첫 padding word에 위치를 전달하고, 실제 packed position parent의 지정 offset에 새로 H2D한다. 테이블 범위 밖 위치는 graph launch 전에 거절한다. 잘못된 replay도 이전 결과를 읽을 수 없도록 completion을 무효화한다. 출력은 rotated Q/K와 V를 이어 붙인 byte 배열이다.

실제 모델 audit는 현재 rows를 pack해 host/device 위치 일치를 확인하고 actual RoPE table을 사용해 독립 eager QKV→RoPE를 계산한다. zero input/position 0과 original input/현재 position을 교대한다. 실제 metadata와 rotated scratch를 저장·복원한다. norm/QKV, tail 및 MLP 기존 audit도 유지한다.

## 로컬 검증

- CPU 391 passed: CUDA lib 84 + graph contract 32, runtime lib 259 + architecture 15 + inventory 1.
- C11 ABI syntax, fmt/diff, 일반 CPU Clippy 통과. 기존 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. 원격 기존 9파일 해시는 이전 검증 상태와 일치.
- **새 CUDA native 및 CUDA feature Rust 코드는 아직 빌드·실행하지 않았다.** CPU 통과는 새 GPU 경로의 컴파일·정확성 증거가 아니다.
- native fixture에 misaligned/out-of-range offset, alias, 테이블 밖 replay 위치, 잘못된 payload 길이, close/Drop/domain 해제를 추가했다. 이 fixture의 초기화되지 않은 weights/tables는 실행하지 않고 잘못된 위치가 launch 전에 거절되는지만 검사한다. 미실행 상태다.
- 실제 모델 QKV→RoPE 5 case × 4 iteration × 32 = 640 replay 및 기존 C07/operator 회귀를 준비했다. 미실행 상태다.

## 승인 대상

대상은 `ai-assistant:/tmp/riley-g01-native-260910`. 아래 소스 9개 전송, CUDA 빌드·GPU 검증, 해당 파일 내 수정·재전송을 포함한다. 모델·비밀정보·배포 파일은 포함하지 않는다. [파일별 전후 해시](transfer-scope.json).

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/src/batch_primitives.cu`
- `kernels/src/ffi_internal.hpp`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/graph_decode_qkv_audit.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`

자동 승인 검토가 이번 새 9파일 payload와 목적지에 대한 명시 승인을 요구하며 전송을 거절했다. 이번 단계에서 원격 쓰기를 수행하지 않았다.

## 남은 범위

이번 구현도 완료된 scratch snapshot에서 마지막 layer weight를 사용한 부분 graph 진단이다. KV write·attention은 아직 이 graph에 연결되지 않았다. CUDA 검증 후 KV write/attention과 layer tail을 연결하고, 실제 layer 입력 및 모든 layer/embedding/final norm/head/output을 통합해야 한다. metadata freshness, 완료·오류·수명, bucket/admission/fallback qualification은 별도 검증이 필요하다. 전체 decode/G02H/G03 완료로 승격하지 않는다. 성능 측정은 시작하지 않았다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [abi.log](abi.log), [clippy.log](clippy.log), [format-diff.log](format-diff.log), [remote-before.txt](remote-before.txt), [model-preservation.json](model-preservation.json).
