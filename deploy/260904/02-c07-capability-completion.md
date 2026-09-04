# PR-G02 — C07 remaining capability completion slices

**목적:** C07의 14개 logical slot 중 현재 `Unknown`인 7개를 “primitive가 존재한다”가 아니라
“현재 executor owner와 exact graph capture가 호환된다”는 기준으로 판정한다. 전체 completion을
한 PR에 넣지 않고 아래 slice를 각각 독립 PR로 진행한다.

## 현재 inventory truth

실제 enum 이름과 순서는 다음과 같다.

```text
MetadataH2d, Embedding, Norm, LayerProjectionGemm, Rope, KvWrite, Attention,
MlpSiluBf16, MlpGatedMultiply, Residual, FinalNorm, LmHead, GpuGreedy,
CompletionBoundary
```

현재 C05 evidence adapter가 이미 **primitive capability 기준으로** `Supported`로 mapping한 slot은 7개다.

```text
MetadataH2d, Norm, Rope, KvWrite, MlpSiluBf16, MlpGatedMultiply, Residual
```

남은 7개는 `Embedding`, `LayerProjectionGemm`, `Attention`, `FinalNorm`, `LmHead`,
`GpuGreedy`, `CompletionBoundary`다. 기존 Supported slot의 primitive를 재구현하지는 않지만,
그 capability가 실제 executor owner/span과 결속됐는지는 별도 audit해야 한다.

## 선행과 병렬성

- G02P와 G02A~G02F source work는 G01 및 Q-track과 병렬 가능하다.
- G02G attention은 G01 GPU parity receipt 이후에만 완료할 수 있다.
- G02H aggregate는 P, A~G와 기존 seven-supported evidence가 모두 닫힌 뒤 실행한다.
- 새 kernel, full graph instantiate, executor hot-path dispatch, CLI/default 변경은 모든 G02에서 금지한다.

각 mapping은 device/context/stream, buffer owner/span, dtype/layout/geometry, mutable control ownership,
completion semantics, exact C05 evidence ID를 비교한다. 하나라도 다르면 `Unknown` 또는 `Unsupported`다.
현재 capability vocabulary에 없는 `Incompatible` 상태를 계획서만으로 새로 만들지 않는다. mismatch의
세부 reason은 별도 typed binding error로 보존하고 inventory에는 fail-closed 상태로 투영한다.

## PR-G02P — existing seven-supported owner-binding audit

primitive capability를 그대로 두고 executor owner binding 여부를 다음 slice별로 확인한다.

- **G02P1:** `MetadataH2d` — 기존 C07 exact slab/H2D owner chain이 selected executor slab과 동일한지 검증
- **G02P2:** `Norm` — per-layer input/output/weight/workspace/span binding
- **G02P3:** `Rope` — Q/K buffer, position metadata, table, layer/head layout binding
- **G02P4:** `KvWrite` — parent KV allocation, layer span, block metadata와 mutation/completion binding
- **G02P5:** `MlpSiluBf16` input/output owner binding
- **G02P6:** `MlpGatedMultiply` gate/up/output owner binding
- **G02P7:** `Residual` attention/MLP residual source/destination alias와 owner binding

각 slice에서 existing binding이 충분하면 characterization test만 추가하고 끝낸다. 부족하면 그 slot의
owner adapter만 구현한다. capability query의 `Supported`를 owner-bound evidence로 자동 승격하지 않는다.

## PR-G02A — Embedding owner/evidence mapping

**한 가지 목적:** C05 embedding validation/status graph를 executor의 token input, embedding weight,
status/output owner와 결속한다.

status D2H가 full decode의 최종 `CompletionBoundary`를 대신한다는 주장은 금지한다.

**완료:** invalid token/status와 valid embedding bytes의 GPU parity receipt가 exact owner digest와 연결된다.

## PR-G02B — LayerProjectionGemm sub-evidence

한 enum slot 안의 모든 projection family가 필요하므로 다음을 각각 별도 PR로 닫는다.

- **G02B1:** Q/K/V projection GEMM geometry와 weight/workspace owner
- **G02B2:** attention output projection
- **G02B3:** gate/up projection
- **G02B4:** down projection
- **G02B5:** B1~B4의 deterministic sub-evidence aggregate를 `LayerProjectionGemm`에 mapping

composite RMSNorm→GEMM evidence를 arbitrary projection에 확장하지 않는다. B1~B4 중 하나라도
Unknown/Unsupported면 B5는 Supported가 아니다.

## PR-G02C — FinalNorm mapping

final norm의 exact rows/hidden width/input/output/weight owner와 C05 norm capture evidence를 별도 binding한다.
per-layer `Norm` slot의 Supported 상태를 그대로 복사하지 않는다.

## PR-G02D — LmHead mapping

LM-head GEMM의 tied/untied weight identity, final-norm input, full logits output owner와 workspace를 binding한다.
canonical GEMM evidence가 exact M/N/K/layout/algo/stream owner와 일치할 때만 Supported다.

LM-head evidence는 `GpuGreedy` evidence가 아니며 full logits lifetime을 completion까지 유지한다.

## PR-G02E — GpuGreedy mapping

row-gather/argmax의 output-slot order, tie, non-finite status, token/status owner와 fixed-address graph evidence를
binding한다. LM-head output layout과 vocabulary length도 identity에 포함한다.

## PR-G02F — CompletionBoundary mapping

token/status D2H query/synchronize/close와 scheduler commit 전 completion dependency를 binding한다.
completion 미확정은 eager 재실행이 아니라 executor poison/request failure다.

## PR-G02G — Attention mapping

G01의 parent-allocation/layer-span owner와 C05-19 GPU receipt를 `Attention` slot에 연결한다.
QH/KVH/D64/page geometry, layer offset/span, metadata layout, stream/context가 exact일 때만 Supported다.

## PR-G02H — aggregate admission

**한 가지 목적:** 14 slot의 immutable status/evidence/owner-binding digest를 deterministic aggregate로 만들고
C06 selection에 read-only로 제공한다.

**테스트:** one Unknown/Unsupported, swapped evidence, composite evidence 재사용, owner digest mismatch,
slot order/version mismatch, nondeterministic serialization을 모두 fail-closed한다.

## 공통 변경 표면

```text
crates/riley-runtime/src/llama/graph_decode_capture_inventory.rs
crates/riley-runtime/src/llama/graph_decode_c05_capture_capability_evidence.rs
crates/riley-runtime/src/llama/executor/{owner,graph,dispatch,output}.rs
crates/riley-cuda/src/graph.rs
crates/riley-runtime/tests/*graph*_cpu.rs
crates/riley-cuda/tests/graph_cpu.rs
```

## 완료 판정

각 slice는 current capability vocabulary의 `Supported`, `Unknown`, `Unsupported`와 별도의 owner-binding
decision을 evidence와 함께 내면 완료다. G02H aggregate는 모든 slot이 `Supported + owner-bound`일 때만
Supported다. 그때만 G03을 시작한다. 누락 primitive가 확인되면
G03으로 넘어가지 않고 그 primitive 하나의 capture/lifecycle/GPU parity 카드를 이 문서에 먼저 추가한다.

GPU evidence는 Q05/Q06 이후에만 수집한다. source-only mapping을 Supported로 승격하지 않는다.
