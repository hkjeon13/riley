# C06 — Graph Signature와 Execution Dispatcher

**상태:** Planned  
**의미 등급:** `E0` systems dispatch  
**한 가지 목적:** workload와 runtime capability에 따라 `full graph | piecewise graph | exact eager`를 선택하고 실패 시 exact fallback하는 bounded dispatcher를 구현한다.

[이전: C05](05-cuda-graph-ownership-abi.md) | [목차](README.md) | [다음: C07](07-decode-graph-buckets.md)

## 1. 배경

CUDA Graph를 global on/off flag로 다루면 dynamic shape, mixed prefill/decode, sampling backend, metadata layout 차이 때문에 잘못된 graph를 재사용하기 쉽다. 반대로 모든 조합을 capture하면 VRAM과 cold-start가 폭증한다.

C06은 graph를 선택하는 identity와 policy를 먼저 고정한다. 실제 Llama decode graph bucket의 성능 구현은 C07 범위다.

## 2. Execution mode

```rust
enum ExecutionMode {
    FullGraph,
    PiecewiseGraph,
    ExactEager,
}
```

- `FullGraph`: iteration의 capture-safe GPU chain 전체 replay
- `PiecewiseGraph`: dynamic boundary 사이의 고정 segment만 replay
- `ExactEager`: 현재 검증된 command-batch path

dispatcher는 unsupported 조건에서 자동 근사하지 않는다. graph가 없거나 invalid하면 exact eager로 돌아간다.

## 3. GraphSignature

최소 key는 다음을 포함한다.

```text
schema version
model architecture family and revision
weight layout revision
GPU compute capability
CUDA/cuBLAS/native ABI version
dtype and compute type
prefill | pure-decode | mixed stage
active-row bucket
head dimension, Q heads, KV heads
KV page size/layout version
metadata layout signature
sampling/output backend
attention implementation ID
projection/MLP implementation IDs
GEMM plan IDs/reduction policy
```

raw pointer와 process-unique address를 persistent signature에 넣지 않는다. pointer stability는 instantiated graph owner가 별도로 검증한다.

## 4. Signature 생성 시점

- model load/cold prepare에서 static 부분 생성
- iteration plan에서 stage, active rows, sampling capability를 추가
- dispatch 직전 runtime capability와 graph cache lookup

signature 생성은 hot path allocation 없이 bounded stack/fixed-array value로 수행한다. 문자열 hash 대신 closed enum과 fixed hash를 사용한다.

## 5. Dispatcher policy

초기 정책은 단순한 closed table이다.

```text
pure decode + supported bucket + GPU greedy + stable layout
  -> FullGraph candidate

prefill 또는 mixed + admitted piecewise segments
  -> PiecewiseGraph candidate

그 외
  -> ExactEager
```

`FullGraph` miss가 곧 capture를 의미하지 않는다. request hot path에서는 capture하지 않고 이미 cold-prepared graph만 사용한다.

## 6. Bounded graph inventory

model instance별 graph inventory는 설정 상한을 가진다.

- maximum graph count
- maximum retained host bytes
- maximum retained device bytes
- bucket allowlist
- full/piecewise별 quota

초기 release에서는 startup에 명시된 bucket만 prepare하며 dynamic LRU capture를 하지 않는다. graph가 quota를 넘으면 cold validation에서 server startup을 실패시키거나 명시적 bucket을 제외하고 exact eager fallback한다. 어떤 정책을 썼는지 startup receipt에 기록한다.

## 7. Fallback reason

metric과 trace에 closed reason을 남긴다.

```text
graph-hit
not-prepared
unsupported-stage
unsupported-shape
unsupported-sampling
layout-mismatch
backend-not-capture-safe
graph-poisoned
launch-failed-recovered
capacity-disabled
operator-capability-unknown
```

request ID를 metric label로 사용하지 않는다.

## 8. Failure policy

### Dispatch 전 mismatch

- device mutation 없음
- exact eager 실행
- graph miss/fallback counter 증가

### Graph launch 전 validation failure

- exact eager 가능
- graph entry를 그대로 유지하거나 signature mismatch로 격리

### Launch 후 completion 확인 + graph-specific failure

- 해당 graph exec를 poison/evict
- stream과 model state가 quiescent하고 KV mutation이 안전한 경우에만 request를 exact eager로 재시도
- 이미 KV/token state가 mutation된 iteration은 이중 실행하지 않고 scheduler failure 계약을 사용

### Completion 미확정

- model executor를 보수적으로 poison
- 같은 request를 자동 재실행하지 않음

## 9. Configuration

```text
--execution-graph-policy disabled
--execution-graph-policy auto
--execution-graph-policy require
--execution-graph-buckets 1,2,4,8,16,32
--execution-graph-max-count N
--execution-graph-max-bytes BYTES
```

- `disabled`: exact eager만 사용
- `auto`: 지원 graph hit, 나머지 eager
- `require`: 요청/iteration이 준비된 graph에 맞지 않으면 fail-closed. benchmark/qualification 용도이며 일반 운영 기본값으로 사용하지 않음

C06에서 production default는 `disabled`다.

## 10. Observability

- mode별 iteration count
- signature lookup latency
- graph hit/miss/fallback reason
- prepared/poisoned/evicted graph count
- retained host/device bytes
- full/piecewise/eager latency
- padding rows와 selected bucket
- launch/recovery failure

metric snapshot은 allocation-free이고 counter degradation이 inference ownership을 깨지 않아야 한다.

## 11. 예상 파일 변경

```text
crates/riley-runtime/src/llama/executor/graph.rs
crates/riley-runtime/src/llama/executor/dispatch.rs
crates/riley-runtime/src/llama/executor/metrics.rs
crates/riley-runtime/src/kernel.rs
crates/riley-server/src/main.rs
crates/riley-server/src/engine.rs
crates/riley-server/src/service.rs
crates/riley-runtime/tests/graph_dispatch_cpu.rs
crates/riley-server/tests/cli.rs
```

C07 전에는 실제 model graph가 없어도 synthetic/dummy graph registry로 dispatcher를 검증할 수 있어야 한다.

## 12. 테스트

- signature equality/hash/version
- 한 field라도 다르면 miss
- unknown capability가 eager로 귀결
- require mode mismatch fail-closed
- bounded inventory overflow
- poisoned graph 재선택 금지
- fallback reason 정확성
- hot lookup allocation 0
- concurrent request가 immutable graph entry를 안전하게 공유
- shutdown 중 graph lookup/launch race
- metric overflow degradation

GPU에서는 작은 fixed kernel graph와 exact eager를 같은 dispatcher를 통해 실행해 output parity와 fallback을 검증한다.

## 13. 승인 기준

- graph 선택이 closed signature에만 의존
- unsupported/unknown이 exact eager 또는 require-mode error로만 종료
- graph miss/poison이 다른 graph entry를 오염시키지 않음
- hot lookup allocation 0
- bounded graph count/bytes 유지
- disabled 모드가 기존 executor behavior/performance를 3% 이내로 보존
- default는 disabled
- C07이 model graph를 registry에 등록할 수 있는 API 고정

## 14. 롤백

CLI와 dispatcher, graph registry를 함께 revert한다. C07 이후에는 graph consumer를 먼저 비활성화하고 `--execution-graph-policy disabled`를 운영 rollback으로 사용한다.

## 15. 완료 정의

각 iteration이 왜 full graph, piecewise graph 또는 eager를 사용했는지 signature와 closed reason으로 설명 가능하고, graph 관련 모든 miss/failure에 exact eager 또는 명시적 fail-closed 경로가 있을 때 완료다.
