# PR-G04 — verified graph runtime registry migration slices

**목적:** GPU parity가 끝난 graph만 runtime registry에서 선택하고, 모든 나머지 request는
deterministic eager fallback을 사용하게 한다. schema, dual-run, production selection, observability를
별도 PR로 분리한다.

## 선행 조건

- G03C의 M=1 graph/eager GPU parity와 lifecycle receipt
- model crate의 semantic pattern schema는 implementation detail을 import하지 않음
- graph policy 기본값은 `disabled`

## PR-G04A — executable implementation schema

**한 가지 목적:** runtime-owned `ImplementationId`, capability predicate, fallback edge, immutable
registry validation과 deterministic serialization을 정의한다.

**등록 대상:** `ReferenceEager`, `PureDecodeGraphRows1`만. 여기서 `Rows1`은 bucket M=1이며
성능 milestone M1과 혼동하지 않는다. future bucket/kernel ID를 미리 활성화하지 않는다.

**테스트:** duplicate ID, missing fallback, fallback cycle, semantic mismatch, unknown capability,
priority ambiguity, deterministic registry bytes, model→runtime dependency 역전 거부.

## PR-G04B — test-only dual selection

**한 가지 목적:** 기존 direct dispatch와 registry selection을 같은 plan에 대해 side-effect 없이 비교한다.

selected implementation, graph signature, fallback reason, workspace bytes, plan ID가 일치하는지만
검사한다. 두 implementation을 GPU에서 동시에 실행하거나 output을 publish하지 않는다.

**완료:** current workload corpus에서 selection mismatch 0; mismatch는 registry default 전환 blocker다.

## PR-G04C — production internal dispatch

**선행:** G04B 및 G03C GPU receipt.

**한 가지 목적:** internal dispatch를 registry-resolved plan ID로 전환한다.

`--execution-graph-policy disabled|auto|require` 계약:

- `disabled`: always reference eager
- `auto`: exact graph hit, otherwise reason-coded eager fallback
- `require`: unsupported/mismatch/poison은 typed reject, silent eager 금지

default는 `disabled`다. graph launch 후 ambiguity는 eager retry가 아니라 existing poison policy다.

**gate:** eager token/KV parity, exact signature hit, one-field mismatch fallback, require reject,
poison 후 auto fallback, c1/c8 latency/throughput 3% non-regression, hot selection allocation 0.

## PR-G04D — bounded observability와 rollout receipt

**한 가지 목적:** selected implementation/fallback reason, graph hit/replay/failure/poison,
registry digest와 active policy를 bounded metric/shutdown receipt로 노출한다.

request ID, model path, prompt/token 원문을 metric label에 넣지 않는다. counter overflow/metric failure는
inference result를 바꾸지 않고 sticky degraded state로 격리한다.

## 변경 표면

```text
crates/riley-runtime/src/{pattern,kernel}.rs (new if the boundary does not already exist)
crates/riley-runtime/src/llama/{plan,executor/{graph,dispatch,graph_registry,graph_metrics}}.rs
crates/riley-server/src/{engine,main,service}.rs
crates/riley-runtime/tests/pattern_registry.rs (new)
crates/riley-server/tests/*graph* (new focused target if absent)
```

## 완료 판정

G04C/D 통과는 `auto`를 release candidate에서 시험할 수 있다는 뜻이다. production default 승격은
B03/B04 Tier C/S, soak, HTTP E2E와 rollback receipt 뒤에만 가능하다. G03D bucket은 검증 완료된
ID만 후속 작은 registry-entry PR로 추가한다.
