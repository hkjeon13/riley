# G03 layer tail — local implementation, remote approval pending

## 요청 및 현재 상태

사용자는 성능 측정 전까지 통합과 검증을 계속하도록 요청했다. 이번 단계에서는 attention context → output projection → attention residual → post-attention RMSNorm → MLP → final residual을 같은 retained graph로 연결했다. **전체 layer/decode graph가 완성되거나 검증된 상태는 아니다. 성능 측정은 시작하지 않았다.**

기존 MLP 및 norm-MLP API는 유지하며 `record_layer_tail`을 추가했다. 선택된 hidden projection GEMM plan을 사용하고 projection bias는 거절한다. projection weight·plan·norm weight의 실제 ledger 등록, shape와 alias를 capture 전에 검증한다. projection output은 norm 이후 최종 output 슬롯으로 재사용하며, context 슬롯도 projection 후 norm output으로 재사용한다. attention residual은 동일 element만 읽고 쓰는 eager kernel의 순차적 in-place 실행을 사용한다.

모델 audit는 독립 eager projection/residual/norm/MLP chain을 계산해 graph output과 비교한다. 실제 hidden_context와 hidden_current의 **완료된 scratch snapshot**을 staged input으로 사용하므로 실제 layer 실행 중의 activation 연결 증거는 아니다. scratch/pinned 및 선택 GEMM plan의 불변성 검사를 유지했다.

## 확인한 것

- CPU 테스트 391 passed: CUDA lib 84, graph contract 32, runtime lib 259, architecture 15, inventory 1.
- C11 ABI syntax, fmt/diff, 일반 CPU Clippy 통과. 기존 경고가 남아 있다.
- 무관한 model-loader 6파일 전후 SHA-256 일치.
- 원격의 기존 7파일 해시가 직전 norm-MLP 검증 상태와 일치한다.
- 새 CUDA native 및 CUDA 활성화 Rust 빌드, GPU 실행은 **미수행**이다. CPU 통과는 이 GPU 조건부 구현의 컴파일·정확성 증명이 아니다.
- native fixture에 projection weight/plan 거절 및 tail capture close/Drop을 추가했다. 모델에 5 case × 4 iteration × 32 = 640 tail replay와 기존 norm-MLP/MLP 회귀를 준비했으나 아직 실행하지 않았다.

## 승인 대상

대상은 `ai-assistant:/tmp/riley-g01-native-260910`이다. 다음 7개 소스만 전송하고 CUDA 빌드·GPU 정확성/반복 실행/회귀 검증을 수행한다. 검증 중 이 파일 범위 내 수정·재전송도 포함한다. 모델·비밀정보·배포 파일은 payload에 포함하지 않는다. 전후 해시는 [transfer-scope.json](transfer-scope.json)에서 확인할 수 있다.

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`
- `crates/riley-runtime/src/llama/graph_decode_mlp_audit.rs`
- `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs`

자동 승인 검토가 이번 변경 payload의 해당 목적지 전송에 명시 승인이 필요하다며 거절했다. 이전 9파일 승인과 “성능 측정 전까지 계속” 요청은 새 7파일 전송 승인으로 인정되지 않았다. **이번 단계에서 원격 쓰기는 수행하지 않았다.**

## 성능 측정 전에 남은 절차

1. 현재 tail payload 전송 승인 후 CUDA 빌드와 실제 모델 tail replay·기존 C07 회귀를 통과시킨다.
2. input norm→Q/K/V GEMM→RoPE→KV write→attention을 단일 owner/ledger 아래 연결한다. 실제 packed metadata와 layer별 KV span을 사용한다.
3. attention 앞부분과 현재 tail을 연결해 실제 layer 입력에서 eager/graph parity를 검증한다. 완료된 scratch 진단과 실행 중 activation 증거를 구분한다.
4. embedding→모든 layer→final norm/head→greedy/output/status까지 하나의 retained model DAG로 통합한다. 홀수/짝수 layer hidden buffer 역할과 매 replay metadata freshness를 검증한다.
5. SmolLM2 단일 모델에서 multi-step continuation, logits/initialized KV, 완료 전 결과 차단, 실패·재시도·자원 수명·복원, 지원 bucket/admission/fallback을 검증한다. G02H/G03 qualification은 이 증거 이후에만 판단한다.
6. 비교 revision/model/workload/GPU 조건과 실행 절차를 고정해 성능 측정 준비 상태를 정리한다. 사용자 요청에 따라 **측정 자체는 시작하지 않는다**.

이 중 2~6은 아직 완료하지 않았다. 새로 정의되는 전송 payload가 자동 승인 검토에서 별도 승인을 요구할 수 있다.

## 증거

[cuda-cpu.log](cuda-cpu.log), [runtime-cpu.log](runtime-cpu.log), [clippy.log](clippy.log), [abi.log](abi.log), [format-diff.log](format-diff.log), [model-preservation.json](model-preservation.json), [remote-before.txt](remote-before.txt).
