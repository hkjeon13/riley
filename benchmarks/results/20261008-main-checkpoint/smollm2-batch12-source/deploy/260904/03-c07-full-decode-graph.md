# PR-G03 — model-owned full pure-decode CUDA Graph slices

**목적:** 검증된 C07 capability를 model-owned full decode graph로 조합한다. cold ownership,
M=1 capture, executor integration, 추가 bucket을 서로 다른 PR로 분리한다.

## 전체 선행 조건

- G02H aggregate가 14 required slot 모두 `Supported + owner-bound`
- exact eager path와 rollback test가 존재
- GPU 단계는 Q05 no-GPU acceptance와 Q06 operation authorization 이후
- candidate/model artifact는 [09](09-common-validation-and-evidence.md)의 receipt를 충족

## PR-G03A — cold graph owner와 immutable recipe

**한 가지 목적:** CUDA launch 없이 full graph가 소유/대여할 resource와 operation order를 cold value로 닫는다.

**구현:** M=1 fixed metadata layout, host/device slab, weight/KV/RoPE/workspace/output owner leases,
14-slot inventory digest, operation DAG, failure/close order를 `executor/graph.rs`에 구성한다.

**비범위:** capture/instantiate/replay, scheduler, CLI, metric, 성능 claim.

**테스트:** wrong owner/layout/device/context/stream, lease overlap, operation reorder, missing slot,
close failure precedence, hot allocation counter의 CPU contract.

## PR-G03B — M=1 cold capture/instantiate 및 isolated parity

**선행:** G03A 및 GPU 승인.

**한 가지 목적:** production-valid sentinel로 M=1 graph를 cold capture/instantiate하고 one-shot eager parity를 확인한다.

**구현:** capture 밖 host validation; fixed H2D node 또는 launch 전 same-stream H2D를 별도 arm으로 유지;
embedding → layer loop → final norm → LM head → GPU greedy → result completion chain; cold receipt.

**비범위:** live executor dispatch, scheduler commit, CLI/default, M>1.

**GPU gate:** 1/2/32/128 replay, token/status/KV byte parity, position/page boundary, owner close allocation 0,
capture/instantiate/launch/close failure injection.

## PR-G03C — executor M=1 replay와 transactional result handoff

**선행:** G03B parity.

**한 가지 목적:** eligible pure-decode iteration만 graph로 실행하고 result completion 뒤 기존 scheduler
commit 경계로 넘긴다.

**구현:** preflight → exact signature → graph replay → one completion → token/status validation → commit.
launch 전 mismatch는 eager; launch 후 completion 미확정은 poison/request failure; commit 실패를 graph
재실행으로 보상하지 않는다.

**검증:** graph/eager alternating history, unsupported sampling fallback, cancellation before/in-flight/
post-result, commit failure, graph poison 후 eager-only continuation.

**성능 gate:** independent 5-pair AB/BA에서 TPOT `>=15%`, M2 TPOT p50 `<=3.58 ms`, TTFT p95
`<=1.05`, peak VRAM `<=1.05`, hot allocation 0. 미달은 `experimental/not-promoted`다.

## PR-G03D — bucket expansion

G03C가 correctness를 통과한 뒤 bucket을 한 묶음씩 별도 PR로 확대한다.

- **G03D1:** M=2,4
- **G03D2:** M=8
- **G03D3:** M=16,32

각 PR은 exact/padded row mapping, output-slot permutation, KV capacity, graph retained bytes,
c1/c4/c8 regression을 독립 측정한다. 한 bucket 실패가 이미 검증된 bucket을 비활성화하지 않되,
실패 bucket은 registry에 등록하지 않는다.

## 변경 표면

```text
crates/riley-runtime/src/llama/executor/{graph,owner,dispatch,metadata,output}.rs
crates/riley-runtime/src/llama/graph_decode_capture_inventory.rs
crates/riley-cuda/src/{graph,primitives}.rs
crates/riley-runtime/tests/llama_graph_gpu.rs (new focused target)
crates/riley-scheduler/tests/llama_iteration_gpu.rs (new focused target)
crates/riley-server/src/benchmark.rs
benchmarks/results/<new immutable campaign>/
```

## 완료 판정

G03C까지 통과하면 M=1 graph는 G04 registry migration 대상으로 갈 수 있다. G03D는 병렬 후속이며
각 bucket이 개별 capability/status를 가진다. full bucket set가 완료되기 전에는 “C07 전체 완료”로
표현하지 않는다.
