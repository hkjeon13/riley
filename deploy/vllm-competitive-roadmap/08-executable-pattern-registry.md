# C08 — Executable Pattern Registry

**상태:** In progress — C08-0는 model-owned closed semantic pattern schema만 고정했다. static executable registry, existing dispatch migration, CUDA parity 및 성능 판정은 C07의 actual native graph ownership·capture/replay parity 결과 뒤에 남아 있다.
**의미 등급:** `reference`  
**한 가지 목적:** canonical model semantics와 실제 CUDA/cuBLASLt 구현 선택을 분리하는 bounded executable pattern registry를 도입한다.

[이전: C07](07-decode-graph-buckets.md) | [목차](README.md) | [다음: C09](09-packed-projection-weights.md)

## 1. 배경

현재 최적화가 Llama executor의 구체 경로에 직접 연결되면 model family, shape, GPU별 candidate가 늘어날수록 조건문과 fallback이 복잡해진다. Riley의 장기 차별점은 model별 클래스를 복제하는 것이 아니라 반복되는 semantic pattern에 여러 implementation을 연결하는 것이다.

이 PR은 새로운 fused kernel을 추가하지 않는다. 기존 reference/exact implementation을 registry를 통해 선택하도록 구조를 바꾸고 observable behavior를 유지한다.

### C08-0 — model-owned closed semantic schema (completed)

C07-22 뒤에는 실행 경계를 바꾸지 않는 CPU-only 준비 단위로 `riley-model`에 schema v1의
`PatternId`와 `SemanticPattern`만 고정한다. ID `1..=8`은 아래 semantic pattern 순서와
동일하고 `0` 및 unknown raw value는 fail-closed다. model 계층에는 numeric identity,
schema version, raw-ID decode만 두며 CUDA/dtype/layout/GPU capability, candidate,
`ImplementationId`, fallback, registry, planner receipt를 넣지 않는다.

이 단위는 full C08 registry migration을 시작하거나 완료한 것이 아니다. 특히 existing direct
dispatch, `riley-runtime::kernel`, executor default, C06/C07 registry decision, CUDA graph
capture/replay 또는 metric은 변경하지 않고 성능 향상도 주장하지 않는다. C05 native owner와
C07 capture/replay parity가 검증된 뒤에만 아래 runtime-owned executable registry migration을
시작한다.

## 2. 두 계층

### Semantic pattern

모델 의미와 tensor contract를 표현한다.

```text
Norm
AttentionPrepare
PrefillAttention
DecodeAttention
MlpSwiGlu
ResidualNorm
LmHead
GreedyOutput
```

### Executable candidate

특정 조건에서 semantic pattern을 실행하는 구현이다.

```text
reference CUDA primitives
cuBLASLt + separate epilogue
cooperative shared-KV attention
future packed QKV
future fused MLP
future LM-head greedy
```

model IR은 semantic pattern을 선택하고, planner가 candidate predicate를 평가한다.

## 3. Core types

```rust
struct PatternId(u16);
struct ImplementationId(u32);

struct PatternRequest<'a> {
    pattern: PatternId,
    model: ModelGeometry,
    workload: WorkloadGeometry,
    dtype: DType,
    gpu: GpuCapability,
    layouts: LayoutSignatureSet,
    semantic_requirement: SemanticRequirement,
}

struct KernelCandidate {
    implementation_id: ImplementationId,
    semantic_class: SemanticClass,
    capability: CapabilityPredicate,
    priority: u16,
    workspace: WorkspaceRequirement,
    graph_capability: GraphCapability,
    fallback_id: ImplementationId,
}
```

모든 identifier는 stable closed value이며 문자열 비교를 hot path에서 사용하지 않는다.

## 4. Capability predicate

predicate가 확인할 축:

- GPU compute capability
- dtype/compute/accumulator type
- batch/active-row bucket
- hidden/intermediate size
- Q/KV heads, head dimension
- prefill/decode/mixed
- KV page/layout version
- weight layout revision
- graph capture requirement
- deterministic reduction requirement

predicate 결과는 `Supported | Unsupported(reason) | Unknown(reason)`이다. Unknown은 선택하지 않는다.

## 5. Selection policy

초기에는 deterministic priority table을 사용한다.

1. semantic requirement를 만족하지 않는 candidate 제외
2. capability `Supported`만 유지
3. workspace/capacity bound를 넘는 candidate 제외
4. explicit implementation override가 있으면 exact match
5. otherwise lowest priority number 선택
6. candidate가 없으면 canonical reference fallback

runtime autotuning은 이 PR 범위가 아니다. C14 이후 별도 contract 없이 실행 중 benchmark를 수행하지 않는다.

## 6. Fallback closure

모든 non-reference candidate는 registry에 존재하는 exact/reference fallback을 가리켜야 한다.

- fallback cycle 금지
- fallback capability가 target workload를 지원해야 함
- disabled/poisoned candidate 선택 시 fallback
- fallback 사용 여부와 reason metric 기록

registry validation이 이 조건을 cold startup에서 확인한다.

## 7. Static registry

production 초기 버전은 compile-time/static Rust registry다.

- dynamic shared library/plugin 없음
- runtime JSON으로 function pointer 주입 없음
- Python-generated code 없음
- unknown implementation ID fail-closed

benchmark/inspection을 위해 machine-readable registry dump를 제공하되 dump가 실행 source of truth는 아니다.

## 8. Planner receipt

model load/cold prepare 시 다음을 출력한다.

```text
pattern ID
selected implementation ID
fallback ID
semantic class
capability evidence
workspace bytes
GEMM/attention plan IDs
graph capability
source/build revision
```

request마다 전체 dump를 만들지 않고 plan ID만 trace/metric에 사용한다.

## 9. 변경 파일 범위

### C08-0 completed files

```text
crates/riley-model/src/pattern.rs
crates/riley-model/src/lib.rs
crates/riley-model/tests/semantic_pattern.rs
```

### C08-1+ runtime migration files (C07 capture/replay parity 이후)

```text
crates/riley-runtime/src/pattern.rs
crates/riley-runtime/src/kernel.rs
crates/riley-runtime/src/llama/plan.rs
crates/riley-runtime/src/llama/executor/dispatch.rs
crates/riley-runtime/src/llama/executor/graph.rs
crates/riley-runtime/tests/pattern_registry.rs
crates/riley-server/src/engine.rs
crates/riley-server/src/service.rs
```

`riley-model`에는 semantic pattern descriptor만 두고 CUDA implementation을 알게 하지 않는다.

## 10. Migration 순서

0. **C08-0 (completed):** model-owned closed `PatternId`/`SemanticPattern` schema v1을 CPU-only로 고정한다. runtime implementation ID나 dispatch migration은 하지 않는다.
1. **C08-1 (C07 capture/replay parity 이후):** runtime-owned `ImplementationId`와 executable schema를 정의한다.
2. 기존 decode attention implementations 등록
3. 기존 norm/MLP/LM-head reference 등록
4. current direct dispatch와 registry dispatch를 test-only dual 실행
5. selected implementation inventory 비교
6. registry path를 default internal dispatch로 전환
7. old ad-hoc conditional 제거

한 번에 모든 operation을 옮기지 않고 pattern별로 전환하되 최종 PR에서 behavior가 닫혀야 한다.

## 11. 테스트

- duplicate pattern/implementation ID 거부
- missing fallback, fallback cycle 거부
- semantic class mismatch 거부
- unsupported/unknown predicate 선택 금지
- priority deterministic selection
- explicit override validation
- GPU/dtype/layout field 하나 변경 시 올바른 candidate/fallback
- registry dump deterministic bytes
- hot selection allocation 0
- poison candidate fallback
- architecture boundary: model crate가 implementation을 import하지 않음

GPU parity는 기존 implementation을 registry 전/후로 선택해 token/KV/output을 비교한다.

## 12. Performance non-regression

registry 도입 자체는 성능 개선 PR이 아니다.

- hot dispatch lookup latency bounded
- c1/c8 TPOT·throughput ratio 3% 이내
- kernel inventory unchanged
- workspace/VRAM unchanged
- allocation 0

hot path에서 linear scan이 문제가 되면 cold prepare 시 pattern request를 resolved plan ID로 변환하고 iteration에는 ID lookup만 수행한다.

## 13. 승인 기준

- 모든 실행 pattern에 reference fallback 존재
- registry validation fail-closed
- model/runtime dependency 방향 유지
- existing token/numeric/KV parity 통과
- performance 3% non-regression
- hot allocation 0
- planner receipt가 실제 선택을 재현 가능
- C09~C11 candidate가 model-specific 조건문 없이 등록될 extension point 확보

## 14. 롤백

registry dispatch와 migration된 pattern을 함께 revert한다. candidate 구현은 아직 추가되지 않으므로 old direct dispatch로 완전 복귀 가능해야 한다.

## 15. 완료 정의

새 kernel implementation을 추가할 때 model forward 코드를 직접 분기하지 않고 capability, semantic class, fallback을 가진 registry entry로 연결할 수 있으며, 기존 경로가 같은 registry의 reference candidate로 실행될 때 완료다.
