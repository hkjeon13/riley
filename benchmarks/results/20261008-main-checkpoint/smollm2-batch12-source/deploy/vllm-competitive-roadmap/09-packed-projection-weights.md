# C09 — Packed QKV와 Gate-Up Projection

**상태:** Planned  
**의미 등급:** `E0`  
**한 가지 목적:** Llama/Qwen의 반복 projection을 packed weight와 단일 projection plan으로 실행하여 GEMM submission과 intermediate 경계를 줄인다.

[이전: C08](08-executable-pattern-registry.md) | [목차](README.md) | [다음: C10](10-transformer-subgraph-fusion.md)

## 1. 가설

현재 Q, K, V와 gate, up은 각각 separate GEMM으로 실행된다. weight를 output dimension 방향으로 concatenate하면 동일한 선형변환을 하나의 larger GEMM으로 표현할 수 있다.

```text
Q = X Wq
K = X Wk
V = X Wv

[Q | K | V] = X [Wq | Wk | Wv]
```

```text
Gate = X Wg
Up   = X Wu

[Gate | Up] = X [Wg | Wu]
```

가설은 GEMM launch/dispatch와 input read를 줄인 이점이 larger-N GEMM의 algorithm 변화와 output split 비용보다 크다는 것이다.

## 2. 범위

### 포함

- packed projection IR/layout revision
- Python-free checkpoint load/packing
- QKV packed weight ownership
- gate-up packed weight ownership
- cuBLASLt plan 생성과 output view/split
- Llama/Qwen geometry
- exact separate fallback
- registry candidate 등록
- numeric/token/performance evidence

### 비범위

- RoPE/KV write fusion
- SwiGLU activation/gate multiply fusion
- down projection/residual fusion
- quantization
- offline Python-only converted checkpoint 강제

## 3. Weight layout

logical order를 ABI/IR에 고정한다.

```text
QKV packed: [Q columns][K columns][V columns]
GateUp packed: [Gate columns][Up columns]
```

각 segment는 offset, logical width, padded width, alignment를 metadata로 가진다. Q/K/V width가 다른 GQA를 지원하고, padding 영역은 실행 결과에 노출되지 않는다.

layout signature:

```text
schema version
source tensor names and hashes
logical dimensions
segment order/offsets
physical stride/alignment
dtype
packing implementation revision
```

## 4. Python-free packing

production model load에서 다음 중 한 경로를 사용한다.

### Host streaming pack

- safetensors slice를 순차 read
- bounded pinned/host staging으로 final packed device buffer의 segment에 upload
- 원본 세 weight를 VRAM에 동시에 유지하지 않음

### Device copy pack

이미 temporary device upload가 필요한 loader 구조라면 bounded temporary 한 개를 재사용해 final packed buffer로 device copy한다.

어느 방식이든 model ready 후 persistent VRAM에는 packed buffer와 필요한 fallback storage 정책만 남는다.

초기 rollout에서는 exact separate fallback을 위해 원본 weight를 영구 중복 보관하지 않는다. fallback은 packed buffer의 segment view로 separate GEMM을 실행할 수 있어야 한다. 이 조건이 불가능하면 VRAM trade-off를 사전 문서화하고 5% 상한을 적용한다.

## 5. GEMM plan

- packed N dimension용 cuBLASLt descriptor/heuristic
- deterministic reduction policy 준수
- active-row bucket별 plan validation
- workspace cap 유지
- unsupported algorithm은 packed candidate 제외 후 separate segment fallback

packed GEMM이 separate GEMM과 다른 reduction/rounding을 사용할 수 있으므로 `E0` numeric tolerance와 multi-step token gate로 검증한다. 결과에 맞춰 tolerance를 완화하지 않는다.

## 6. Output ownership

packed output은 한 buffer에 생성하고 segment `TensorView`로 소비한다.

- Q view
- K view
- V view
- gate view
- up view

view는 storage를 소유하지 않으며 parent buffer lifetime을 넘을 수 없다. C10이 RoPE/KV/activation을 fuse하기 전까지 기존 kernel이 각 view를 사용한다.

## 7. Registry 연결

C08 pattern registry에 다음 candidate를 추가한다.

```text
AttentionProjection:
  packed-qkv-cublaslt-v1
  fallback: separate-q-k-v-cublaslt

MlpInputProjection:
  packed-gate-up-cublaslt-v1
  fallback: separate-gate-up-cublaslt
```

capability는 model family 이름이 아니라 dimensions/layout/dtype/GPU 조건으로 표현한다.

## 8. Correctness

### Low-level

- random small matrix FP32 reference
- BF16 Q/K/V 각 segment tolerance
- odd/non-equal widths
- padding sentinel 보존
- segment offset/alignment
- deterministic repeated execution

### Model

- SmolLM2와 Qwen prefill/decode
- sequence `18,128,1024,4096,near-limit`
- active-row bucket `1,2,4,8,16,32`
- 32/128-step greedy token exact
- graph disabled/enabled compatibility
- packed candidate disable 시 segment fallback parity

### Loader

- source tensor hash/shape/name mismatch fail-closed
- incomplete/duplicate segment 거부
- load failure rollback 후 allocation 0
- model close 후 packed storage 0

## 9. Performance campaign

candidate를 분리해 측정한다.

1. QKV packed only
2. GateUp packed only
3. combined packed

각 candidate가 단독 5% primary 개선을 못하면 independent default 승격하지 않는다. combined enabling value가 있다면 combined candidate가 10% 이상 개선해야 한다.

Metric:

- projection GPU time
- GEMM launch count
- input/output HBM bytes 추정 및 NCU attribution
- TTFT/TPOT/E2E/throughput
- workspace/peak VRAM/model load time
- graph capture/instantiate bytes와 시간

## 10. 사전 gate

- numeric/token mismatch 0 within fixed contract
- projection candidate primary paired improvement `>= 5%`
- combined candidate `>= 10%` 또는 각 candidate 독립 5%
- TTFT/TPOT p95 non-target ratio `<= 1.05`
- c8 throughput ratio `>= 0.95`
- persistent VRAM ratio `<= 1.05`
- model load p95 ratio `<= 1.10`
- hot allocation 0

## 11. 예상 파일 변경

```text
crates/riley-model/src/*                 # packed layout metadata only
crates/riley-runtime/src/cuda_weights.rs
crates/riley-runtime/src/llama/plan.rs
crates/riley-runtime/src/llama/executor/gemm_plan.rs
crates/riley-runtime/src/pattern.rs
crates/riley-runtime/tests/packed_projection_gpu.rs
crates/riley-model/tests/*
benchmarks/results/<campaign>/...
```

## 12. 오류와 fallback

- checkpoint/layout mismatch: model load fail-closed
- packed plan unsupported: segment-view separate fallback
- packed execute pre-dispatch failure: separate fallback 가능
- execute 후 completion/mutation 불명확: 자동 재실행 금지, executor poison 계약
- candidate numeric gate 실패: registry default에서 제외

## 13. 롤백

운영 flag 또는 implementation override로 separate candidate를 선택한다. 코드 rollback 시 packed layout revision과 loader/runtime consumer를 함께 revert한다. packed checkpoint를 별도 artifact로 배포했다면 old runtime이 이를 잘못 읽지 않도록 schema version을 fail-closed한다.

## 14. 완료 정의

원본 weight를 불필요하게 중복 보관하지 않고 QKV와 gate-up을 packed projection으로 실행하며, fixed E0/token gate와 사전 성능·메모리 기준을 통과해 registry 승격 여부가 결정될 때 완료다.
