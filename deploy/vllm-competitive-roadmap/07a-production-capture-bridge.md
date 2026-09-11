# C07-A — Production Decode Graph Capture Bridge

**상태:** Planned — C07-35까지의 metadata/C06/C05 primitive evidence를 실제 Llama pure-decode replay owner로 연결하기 전에 필요한 ownership·capture-session 경계를 고정한다.
**의미 등급:** `E0`
**한 가지 목적:** existing eager executor의 실제 stream·resource·completion contract를 보존하면서, 한 cold-owned pure-decode graph가 모든 graph-visible resource를 끝까지 보유하도록 만든다.

[C07 본문](07-decode-graph-buckets.md) | [C05 ownership ABI](05-cuda-graph-ownership-abi.md) | [다음: C08](08-executable-pattern-registry.md)

## 1. 왜 이 bridge가 필요한가

C05-22까지는 fixed-address primitive 각각의 capture/end/instantiate/replay/close와 GPU byte parity를 검증했다. C07-35는 그중 C07 V1 operation과 정확히 동치인 seven primitive의 capability evidence만 inventory에 반영한다. 이 둘은 full Llama decode graph의 resource owner나 production execution path를 만들지 않는다.

현재 executor는 `PreparedLlamaBatchExecutor::execute_packed`에서 caller가 빌려 준 `CudaStream`과 `CudaCommandBatch`를 사용한다. command batch는 per-operation synchronize를 한 completion boundary로 모으지만 native graph begin은 active command batch를 의도적으로 거부한다. 반대로 generic `GraphCapture`는 abort-only이며 graph 내부에서 existing primitive/GEMM calls를 안전하게 실행하거나 used-resource lease를 captured graph까지 유지하는 API가 아니다. C05 one-node owners도 stream과 exact allocations을 by-value로 독점하므로 서로 합성하거나 executor가 빌린 buffer span과 섞을 수 없다.

따라서 C07의 다음 단계는 partial C05 evidence를 `Supported`으로 넓히거나 existing one-node graph를 executor에 억지로 주입하는 것이 아니다. graph capture 동안 제출되는 모든 buffer, pinned buffer, GEMM plan, stream, and completion destination을 하나의 cold owner가 보유하는 새 capture-session contract가 먼저 필요하다.

## 2. 고정된 비목표

- C05 primitive `Supported`를 `Embedding`, `Attention`, `LayerProjectionGemm`, `LmHead`, `GpuGreedy`, 또는 `CompletionBoundary`에 자동 승격하지 않는다.
- C05-19 grouped attention과 C05-20 embedding status graph를 현재 executor에 map하지 않는다. 현재 executor의 layer span, parent KV allocation, token/status completion ownership이 각 owner와 같다는 proof가 아직 없다.
- graph prepare나 capture를 request hot path에서 수행하지 않는다.
- generic raw graph handle, raw pointer, caller-selected capture mode, node update, or unsafe resource escape hatch를 model/server code에 노출하지 않는다.
- graph-ready owner가 없는 policy는 exact eager fallback (`Auto`) 또는 existing required-graph rejection (`Require`)을 유지한다.

## 3. 실행 순서

### C07-36 — cold full-decode resource bundle extraction

`PreparedLlamaBatchExecutor`에서 graph-visible ownership을 독립 cold bundle로 뽑는다. bundle은 graph-mode가 사용할 하나의 owned stream, exact bucket-specific metadata/token/control storage, forward workspace, uploaded weights, KV pool, RoPE tables, GEMM plans, greedy result/status storage를 함께 가진다. graph-mode bundle은 existing eager owner와 동시에 같은 native allocation을 사용하지 않는다.

첫 slice는 raw graph handle이나 capture를 만들지 않는다. 다음을 proof한다.

- bundle construction 후 address-bearing resources가 owner lifetime 동안 이동·close·replace되지 않는다.
- graph candidate가 아닌 shape/sampling/output mode는 bundle을 만들거나 capture하지 않고 current eager path를 유지한다.
- all resource geometry, model revision, device, implementation profile, completion layout이 one `GraphStaticSignature` input으로 exact representation된다.
- prepare/close failure에서 partial bundle, allocation leak, or changed eager behavior가 남지 않는다.

Acceptance은 CPU ownership/source-contract, CUDA build, remote GPU cold prepare/close allocation baseline, and exact eager non-regression이다. 이 단계는 capture/replay performance claim이 아니다.

### C07-37 / C05-23 — owned multi-operation capture session and lease ledger

C05에 a private-by-default multi-operation capture session을 추가한다. It must retain one capture stream plus a fixed, deduplicated native lease ledger for every buffer/pinned buffer/GEMM plan that an admitted operation touches. Capture-mode primitive submission must defer per-operation synchronization without pretending that the existing command-batch owner is active. A capture session cannot coexist with an ordinary command batch on the same stream.

The state transition is:

```text
ColdResources
  -> Capturing(lease ledger frozen)
  -> Captured
  -> Instantiated(resources still retained)
  -> Launching
  -> Instantiated
  -> Closed(resources returned or released)
```

Every failed native transition is one-shot. If native consumption or resource release is uncertain, the session remains terminal/retained; it must not retry, fall back to eager on the same possibly-mutated iteration, or manufacture a recoverable resource bundle.

Before full Llama wiring, C05-23 needs a synthetic multi-operation GPU fixture that uses real existing primitive plus GEMM calls in one capture session, verifies capture/end/instantiate/replay parity, replay at least 64 times, no hot allocations, same-context/nonalias rejection, and close allocation baseline zero. This proves the session/ledger only, not model decode support.

### C07-38 — M=1 canonical pure-decode capture and parity

Using the C07-36 owner and C05-23 session, cold-capture one exact `M=1` canonical pure-decode chain. Its metadata must be produced by the C07 exact layout owner, and its input/control staging must complete before replay. The chain must include the exact execution profile selected in the static signature, not a substitute primitive. Capture-time sample metadata uses a production-valid sentinel and is never reused as a request payload.

Before exposing selection, compare graph and eager with the same model/context/history for generated-token hash, per-step token, KV continuation, inactive storage invariants, output/status record, and cancellation/error behavior. A capture, launch, or completion failure permanently excludes that signature from the prepared registry and preserves eager fallback according to policy.

### C07-39 — buckets, registry publication, and execution selection

After M=1 parity, add only one bucket at a time (`2`, `4`, `8`, `16`, `32`). Each bucket has its own cold resource bundle, complete static signature, C06 registry entry, and close receipt. Registry publication occurs only after capture, instantiate, and one-shot parity complete. The hot path may pack/stage values and resolve an exact entry, but may not allocate, compile, capture, or repair an incompatible request.

The production dispatch order is C07 eligibility → exact signature → C06 policy/registry selection → matching owner resolution → launch/completion → token/status validation → scheduler commit. A non-full decision remains the established eager path. A selected slot/signature mismatch is a terminal owner error rather than a best-effort lookup retry.

### C07-40 — performance decision

Run the C07 primary campaign only after a clean graph candidate passes the correctness matrix. Use the pinned SmolLM2 diagnostic (`c1/p128/o32/greedy`) and five independent-process AB/BA pairs versus the same-revision eager candidate. Preserve all raw arms, including rejected/no-improvement arms. Promote neither default nor vLLM competitive claim unless the C07 M2 and common promotion gates are met.

## 4. Required operation evidence for C07-38

The C07 V1 inventory must be complete for the *exact* owned execution profile. Existing seven C05 facts cover metadata H2D, canonical generic BF16 RMSNorm, indexed BF16 RoPE, ragged BF16 KV write, BF16 SiLU, BF16 gated multiply, and BF16 residual add. The following remain `Unknown` until the C07-36/C05-23 owner proves the same resource and semantic contract:

| C07 operation | Why existing C05 evidence is insufficient |
|---|---|
| Embedding | C05-20 owns a separate table/token/output/report chain; executor token staging and completion ownership differ. |
| Layer projection GEMM | C05-21/22 use one whole allocation and strict plan; executor uses spans, shared workspace, and multiple projections. |
| Attention | C05-19 owns one exact D64 grouped operation; executor chooses implementation and uses parent KV layer spans. |
| Final norm / LM head | No exact full-profile owner exists. |
| GPU greedy / completion | C05 raw gather/argmax/D2H receipts do not establish scheduler token/status commit semantics. |

`Unknown` is intentional and must keep `Auto` eager and `Require` rejected. No test may replace it with a synthetic all-supported inventory in a production path.

## 5. Exit evidence

The bridge is not complete until all of the following are available for each admitted bucket:

- clean commit and cold-preparation receipt bound to model/tokenizer/GPU/driver/toolkit and complete graph signature;
- actual graph capture/end/instantiate/replay/close GPU evidence, not a one-node primitive proxy;
- graph/eager alternating-history parity including generated-token hash and KV continuation;
- no hot host/device allocation, owner-close live allocation zero, and failure/cancellation behavior equal to eager;
- C06 full-graph selection resolves exactly one matching owner and all other inputs retain exact fallback;
- append-only performance raw evidence satisfying the published C07/M2 gate.

Until then C07 remains `In progress`, and the observed vLLM comparison remains an eager-vs-vLLM baseline rather than graph speedup evidence.
