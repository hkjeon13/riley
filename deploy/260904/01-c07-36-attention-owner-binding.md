# PR-G01 — C07 attention graph owner binding

**목적:** C05-19의 isolated grouped paged-attention graph를 Riley executor의 실제 parent KV
allocation에 안전하게 결속한다. 이 PR은 full decode graph나 성능 개선을 주장하지 않는다.

## 왜 지금 필요한가

executor는 모든 layer K/V를 하나의 parent allocation에 보관하고 `layer_byte_offset` span만
attention에 빌려 준다. 반면 C05-19 graph owner는 한 layer K/V pool allocation을 by-value로
독점한다. 단순히 `Attention=Supported`로 표시하면 graph가 parent allocation의 일부 span을
capture하는데 lifetime/offset/stream 동등성을 증명하지 못하는 false admission이 된다.

## 선행 조건

- C05-19 GPU lifecycle/byte-parity evidence는 존재한다.
- C07 inventory에서 attention은 계속 `Unknown`이다.
- 이 PR은 CPU/ABI contract까지는 즉시 가능하다. GPU acceptance는 Q05 no-GPU acceptance와
  Q06 operation authorization 뒤, 별도 승인된 host에서만 한다.

## 변경 범위

```text
kernels/include/riley_cuda.h
kernels/src/*attention* 또는 graph ownership implementation
crates/riley-cuda/src/graph.rs 및 FFI wrapper
crates/riley-runtime/src/llama/executor/{owner,graph}.rs
crates/riley-cuda/tests/graph_cpu.rs
crates/riley-runtime/tests/*graph* (새 ownership contract test)
```

`riley-model`, scheduler routing, graph registry/default policy, full graph capture, generic per-layer
KV reallocation은 이 PR 범위가 아니다.

## 구현 순서

1. C ABI에 `parent allocation + immutable layer span`을 나타내는 opaque owner/lease를 추가한다.
   raw pointer와 length만 전달하는 API는 금지한다.
2. cold prepare에서 parent device allocation identity, layer offset/length, geometry
   `(QH, KVH, D64, page=16)`, metadata layout digest, stream/device/context를 하나의 typed binding으로 만든다.
3. capture owner는 parent allocation의 lifetime을 보유하고, exact immutable span만 access 가능하게 한다.
   span 밖 access, offset overflow, overlap, wrong context/stream, closed parent는 capture 전 reject한다.
4. Rust wrapper는 graph launch completion 전 parent/lease close 또는 mutable reuse를 type/lifecycle으로 막는다.
5. C07 evidence inventory의 attention slot은 위 binding과 GPU parity receipt가 모두 있을 때만
   `Supported`가 된다. 그 전에는 `Unknown`을 유지한다.

## 필수 테스트

- CPU/ABI: wrong parent, offset/length overflow, overlapping span, wrong device/context/stream,
  parent close, graph close, double release, failure precedence.
- GPU: isolated owner와 parent-span owner가 동일 QH/KVH/D64/page geometry에서 attention output bytes가
  같음; capture/replay 반복; layer A capture가 layer B span을 읽거나 쓰지 않음.
- lifecycle: in-flight graph 동안 parent close/reuse 거부, completion 뒤 owner/resource count 0.

## 완료 판정

다음 모두를 만족하면 `complete`다.

- C07 attention slot에 exact owner binding과 GPU parity receipt가 연결된다.
- parent allocation lifetime과 layer span access가 graph completion까지 보장된다.
- invalid binding은 eager fallback 대상이지 capture admission 대상이 아니다.

GPU 측정을 하지 못하면 CPU contract까지만 `source-complete / gpu-pending`으로 기록한다.
TPOT 수치, graph default, full decode replay는 이 PR 결과에 포함하지 않는다.

종료 시 [STATUS.md](STATUS.md)의 G01 행에 source revision, CPU/GPU decision과 receipt를 기록한다.
