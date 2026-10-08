# Riley Agentic Inference Optimization Review & Roadmap

**상태:** Proposed research/implementation track  
**작성 기준:** `main@a15da11e84c20e9e1d91a819458c9c3f0f5f70ce`  
**외부 조사 기준일:** 2026-08-28  
**한 가지 목적:** agentic·multi-agent workflow의 실제 호출 패턴을 Riley의 scheduler, KV lifecycle, routing, constrained decoding에 반영할 수 있는지 판정하고, 구현 가능한 항목만 독립 PR 단위로 분해한다.

[기존 vLLM 경쟁력 로드맵](../vllm-competitive-roadmap/README.md) | [Scheduler와 Continuous Batching](../13-scheduler-and-batching.md) | [Extension Gate](../17-extension-gates.md)

---

## 1. 결론

**가능하다.** 다만 `agentic workflow 전용 Transformer kernel`이 따로 존재하는 것은 아니다. 모델의 attention·MLP·sampling 연산은 일반 채팅 요청과 본질적으로 같다. 실질적인 최적화 지점은 에이전트 실행기가 알고 있는 다음 정보를 추론 인프라도 제한적으로 이해하도록 만드는 데 있다.

- 여러 turn이 같은 workflow/session에 속한다는 사실
- tool 호출로 인해 다음 turn까지 GPU 계산이 잠시 중단된다는 사실
- 다음 요청이 이전 요청의 긴 prefix를 다시 사용할 가능성이 높다는 사실
- fan-out된 여러 sub-agent가 하나의 workflow budget을 공유한다는 사실
- 짧은 tool-decision turn과 긴 final-synthesis turn의 지연 목표가 다르다는 사실
- JSON/tool-call schema가 반복된다는 사실
- 분산 worker 중 어느 worker가 필요한 KV prefix를 이미 보유하는지

따라서 Riley가 제공할 가치가 큰 기능은 다음 순서다.

1. **Agent execution context와 bounded serving hints**
2. **workflow-aware fairness와 priority scheduling**
3. **tool-gap-aware exact KV retention/eviction**
4. **structured tool-output runtime과 grammar cache**
5. **cancellable speculative/prepared prefill**
6. **KV-aware distributed routing과 이후의 KV offload·prefill/decode 분리**

반대로 Riley가 agent planner, tool executor, retry graph, durable workflow state까지 직접 소유하는 것은 권장하지 않는다. Riley는 agent framework가 아니라 **agent lifecycle signal을 안전하게 소비하는 inference runtime**이어야 한다.

### 최종 판정

| 질문 | 판정 |
|---|---|
| 추론 단에서 agentic workload 최적화가 가능한가? | **예.** 특히 KV reuse/lifetime, scheduler, routing, constrained decoding에서 효과가 있다. |
| 별도 agent 모델 또는 agent 전용 CUDA kernel이 필요한가? | **대체로 아니오.** 기존 model fast path 위에 serving semantics를 추가하는 문제다. |
| 현재 Riley에서 즉시 구현 가능한가? | metadata/benchmark/fair scheduling은 가능하다. KV retention은 C12 exact prefix cache 이후가 안전하다. |
| 분산 agent 최적화까지 즉시 가능한가? | **아니오.** 현재 single-worker 경계를 먼저 닫고, C13 이후 cache directory와 router를 별도 단계로 도입해야 한다. |
| semantic cache나 position-independent KV reuse를 production 기본값으로 둘 수 있는가? | **아니오.** 정확성·모델 의존성·격리 위험 때문에 연구 track으로만 유지한다. |

---

## 2. 용어와 범위 분리

`distributed agent`는 서로 다른 두 문제를 의미할 수 있으므로 섞지 않는다.

### 2.1 Multi-agent workflow

여러 planner, researcher, reviewer, coder 같은 논리적 agent가 fan-out/fan-in 또는 반복 대화를 수행하는 애플리케이션 실행 패턴이다.

```text
orchestrator
  ├─ agent A ─ tool ─ agent A
  ├─ agent B ─ tool ─ agent B
  └─ agent C ─ tool ─ agent C
            ↓
       final synthesis
```

Riley는 이 graph의 의미나 tool을 실행하지 않는다. 다만 동일 workflow에 속한 요청의 공정성, cache locality, lifecycle을 최적화할 수 있다.

### 2.2 Distributed inference serving

여러 Riley worker/GPU가 요청과 KV state를 나누어 처리하는 인프라 문제다.

```text
agent harness
      |
      v
agent-aware router
  ├─ worker A: prefix X 보유
  ├─ worker B: decode 부하 높음
  └─ worker C: idle, prefix 없음
```

여기서는 cache directory, load+locality scoring, KV transfer, prefill/decode 분리가 핵심이다. Multi-agent를 지원한다고 자동으로 multi-GPU serving이 구현되는 것은 아니다.

---

## 3. Agentic workload에서 반복되는 패턴

### 3.1 Growing-prefix multi-turn

일반적인 agent loop는 매 turn마다 이전 대화, system prompt, tool schema, tool result를 다시 포함한다.

```text
turn 1: system + tools + user
turn 2: system + tools + user + assistant(tool call) + tool result
turn 3: turn 2 전체 + assistant(tool call) + tool result
...
```

이때 새 요청 대부분은 이전 요청의 긴 exact prefix를 공유한다. Prefix cache가 없다면 turn이 늘수록 같은 token을 반복 prefill한다.

### 3.2 Tool-induced idle gap

모델이 tool call을 생성한 뒤 네트워크, 검색, 코드 실행, 브라우저, 데이터베이스 응답을 기다리는 동안 해당 session의 KV는 GPU에서 계산하지 않지만 곧 다시 쓰일 가능성이 높다.

- 너무 빨리 evict하면 다음 turn에서 긴 prefix를 다시 계산한다.
- 무조건 pin하면 긴 tool call 하나가 HBM을 점유해 다른 요청을 방해한다.
- 따라서 예상 재사용 가치와 memory pressure를 함께 반영하는 bounded retention이 필요하다.

### 3.3 Fan-out/fan-in burst

하나의 user workflow가 짧은 시간에 다수 sub-agent 요청을 생성할 수 있다. 단순 request-level FCFS에서는 한 workflow가 queue와 KV capacity를 독점할 수 있다.

필요한 정책은 다음과 같다.

- tenant/workflow별 동시 실행 상한
- workflow-level fair share
- interactive turn과 background branch 분리
- final synthesis처럼 critical-path에 있는 요청의 bounded 우선순위
- starvation 방지 aging

### 3.4 Heterogeneous turn shape

같은 workflow 안에서도 shape가 다르다.

| 단계 | 흔한 특성 |
|---|---|
| tool selection | 긴 prompt, 매우 짧은 output, TTFT 민감 |
| observation digestion | 긴 tool output, prefill 지배적 |
| autonomous reasoning | 중간 prompt, 긴 output, decode 지배적 |
| final synthesis | 매우 긴 prefix, 중간~긴 output, workflow JCT의 critical path |

`모든 request가 동일한 비용과 SLO를 가진다`는 가정은 agent workload에서 특히 부정확하다.

### 3.5 Repeated structured output

Tool call은 JSON schema, enum, regex 또는 grammar 제약을 반복 사용한다. 같은 schema를 매 요청 다시 compile하거나 CPU에서 느리게 검사하면 지연과 retry가 증가한다.

### 3.6 Cache-locality-sensitive distributed routing

분산 환경에서 가장 빈 worker가 항상 가장 빠른 worker는 아니다. 이미 긴 prefix KV를 보유한 worker가 약간 바쁘더라도 전체 prefill 비용은 더 낮을 수 있다. 반대로 session sticky routing을 절대 규칙으로 만들면 hot worker가 과부하될 수 있다.

---

## 4. 현재 Riley와의 접점

### 이미 있는 기반

- [`deploy/13-scheduler-and-batching.md`](../13-scheduler-and-batching.md)는 bounded FCFS, decode-first, aging, token budget, prefill chunking, paged-KV ownership을 구현했다.
- [`C12 — Tenant-safe Exact Prefix Cache`](../vllm-competitive-roadmap/12-tenant-safe-prefix-cache.md)는 exact complete KV block reuse, sharing domain, immutable ownership, bounded eviction을 계획한다.
- [`C13 — Restartable Isolated GPU Worker`](../vllm-competitive-roadmap/13-restartable-gpu-worker.md)는 scheduler+model+CUDA를 local worker process로 격리한다.
- [`C08 — Executable Pattern Registry`](../vllm-competitive-roadmap/08-executable-pattern-registry.md)는 향후 constrained decoding 또는 specialized candidate를 model forward의 ad-hoc 분기 없이 연결할 수 있는 기반이다.

### 현재 부족한 계약

현재 `RequestDescriptor`는 사실상 `prompt_token_ids`와 `max_new_tokens`만 알고, transport-independent `GenerationRequest`도 model, prompt, sampling, stop, stream만 가진다. 다음 정보는 아직 없다.

```text
session/workflow/program identity
parent-child relation
turn index
latency class
bounded priority
expected output length
expected resume delay
cache retention intent
session close/release event
```

또한 현재 OpenAI compatibility surface는 `/v1/completions` 중심이며 unknown field를 명시적으로 거절한다. Agent metadata를 추가할 때 임의 field를 조용히 받아들이지 말고, **closed/versioned Riley extension 또는 별도 internal API contract**로 도입해야 한다.

### 기존 로드맵과의 관계

이 문서는 `vLLM보다 빠르게 만들기` 로드맵을 대체하지 않는다.

```text
vLLM competitive roadmap
  └─ request 단위 kernel/runtime/안정성 기반

agentic inference roadmap
  └─ 여러 request를 잇는 lifecycle/cache/scheduling/routing 의미
```

Agentic track은 C03 scheduler invariant, C12 exact prefix cache, C13 worker isolation 결과를 재사용하되 독립적으로 gate한다.

---

## 5. Riley가 소유할 경계

### Riley가 소유할 수 있는 것

- closed agent execution metadata의 normalization과 validation
- request/session/workflow 단위 admission과 fairness
- exact prefix lookup, KV retention, eviction, release lifecycle
- worker-local cache inventory와 안전한 transfer descriptor
- cache locality와 load를 위한 routing signal
- structured output grammar compile/cache/mask execution
- speculative/prepared prefill의 실행·취소·회계
- turn/workflow 수준 serving metrics

### Riley가 소유하지 않을 것

- agent planning/reasoning graph
- tool 선택의 비즈니스 의미
- tool 실제 실행과 credential
- retry/backoff/compensation workflow
- durable checkpoint와 workflow database
- semantic memory/vector database
- agent-to-agent protocol 또는 메시지 broker
- 사용자 요청에서 임의로 지정하는 tenant/cache sharing domain

상위 agent harness는 **무엇을 실행할지** 결정하고, Riley는 **주어진 model execution을 어디서 언제 어떤 KV state로 실행할지** 최적화한다.

---

## 6. 기능별 타당성 판정

| ID | 기능 | 추론 단 타당성 | Riley 적합성 | 선행 조건 | 판정 |
|---|---|---:|---:|---|---|
| A01 | Agentic workload/trace benchmark contract | 높음 | 높음 | 없음 | 즉시 착수 |
| A02 | Passive context + advisory serving hints | 높음 | 높음 | closed schema | 즉시 착수 |
| A03 | Workflow-aware fair scheduling | 높음 | 높음 | A02, C03 invariant 권장 | 우선 구현 |
| A04 | Tool-gap-aware exact KV retention/TTL | 높음 | 높음 | C12-B exact cache | 핵심 기능 |
| A05 | Structured output grammar cache | 높음 | 중~높음 | API/grammar contract, C08 활용 | 별도 track |
| A06 | Cancellable speculative/prepared prefill | 중~높음 | 중간 | A02, C12, token-level API | 실험 후 승격 |
| D01 | KV-aware multi-worker routing | 높음 | 중~높음 | C13, multi-worker control plane | 분산 1단계 |
| D02 | HBM→host/remote KV offload·prefetch | 높음 | 중간 | D01, transfer ABI, topology profiling | 분산 2단계 |
| D03 | Prefill/decode disaggregation | 조건부 높음 | 중간 | multi-GPU, KV transfer, C14 이후 | 후순위 |
| R01 | Collective multi-agent KV sharing | 특정 패턴에서만 높음 | 낮~중간 | synchronized fan-out trace | 연구 |
| R02 | Position-independent/semantic KV reuse | 모델·방법 의존 | 낮음 | 별도 correctness/security 연구 | production 비권장 |
| R03 | Learned KV pruning/intent cache | 품질 trade-off 존재 | 낮음 | model별 quality gate | 연구 전용 |

핵심은 A01~A04다. D01~D03은 Riley가 multi-worker serving으로 확장될 때 가치가 커지지만, 현재 single-GPU 경쟁 기준을 흐리면서 먼저 구현할 이유는 없다.

---

## 7. Agent execution context와 serving hints

### 7.1 내부 계약 제안

```rust
pub struct AgentExecutionContext {
    pub session_id: Option<OpaqueId>,
    pub workflow_id: Option<OpaqueId>,
    pub program_id: Option<OpaqueId>,
    pub parent_program_id: Option<OpaqueId>,
    pub branch_id: Option<OpaqueId>,
    pub turn_index: Option<u32>,
}

pub struct AgentServingHints {
    pub requested_priority: i16,
    pub latency_class: LatencyClass,
    pub step_class: StepClass,
    pub expected_output_tokens: Option<u32>,
    pub expected_resume_after_ms: Option<u64>,
    pub cache_retention: CacheRetention,
    pub speculative_prefill: bool,
}

pub enum LatencyClass {
    Interactive,
    Normal,
    Background,
}

pub enum StepClass {
    General,
    ToolDecision,
    ToolResume,
    BranchWork,
    FinalSynthesis,
}

pub enum CacheRetention {
    Default,
    ReuseLikely { ttl_ms: u64 },
    Ephemeral,
    ReleaseAfterTurn,
}
```

### 7.2 Context와 hint를 분리하는 이유

- `AgentExecutionContext`는 관찰·추적·공정성 grouping을 위한 **수동적 identity**다.
- `AgentServingHints`는 scheduler/cache가 참고할 수 있는 **능동적이지만 advisory인 signal**이다.
- `session_id`가 있다고 자동 sticky placement나 cache sharing을 활성화하지 않는다.
- `requested_priority`는 그대로 적용하지 않고 auth/tenant policy로 계산한 `effective_priority`로 clamp한다.

### 7.3 보안과 cardinality 규칙

- 모든 opaque ID는 길이 상한과 canonical byte encoding을 가진다.
- raw ID를 Prometheus label로 사용하지 않는다.
- raw prompt, tool result, agent name을 scheduler metric에 넣지 않는다.
- session/workflow ID는 **cache equality key가 아니다.** Exact token/model identity가 일치해야만 hit다.
- tenant/sharing domain은 인증된 server policy에서 주입한다. request가 선택하지 못한다.
- unknown enum/field/version은 ignore가 아니라 fail-closed 또는 명시적 fallback이다.
- hint가 누락되면 현재 request-level behavior와 동일해야 한다.

### 7.4 API 노출 순서

A02에서는 먼저 benchmark harness와 internal domain type에만 계약을 추가해 behavior를 바꾸지 않는다. Public wire extension은 별도 API PR에서 다음 중 하나로 고정한다.

1. versioned Riley-native endpoint
2. closed namespaced object
3. trusted gateway가 주입하는 authenticated metadata

현재 `/v1/completions`의 unknown-field rejection을 우회하지 않는다.

---

## 8. Workflow-aware scheduler

### 8.1 목표

현재 bounded request scheduler를 유지하면서, request 선택 시 tenant와 workflow 비용을 함께 본다.

```text
tenant fairness
  -> workflow fairness
    -> latency lane
      -> bounded priority + aging
        -> request cost/shape
```

### 8.2 초기 정책

- `Interactive | Normal | Background`의 bounded lane
- tenant별 active request/KV/token budget
- workflow별 fan-out concurrency 상한
- priority는 작은 closed range로 clamp
- waiting time aging으로 starvation 방지
- expected output length를 decode budget 추정에 사용
- final synthesis 우선순위는 authorized policy 범위에서만 가산
- GPU iteration 경계에서만 선택하며 첫 버전에는 kernel 중간 hard preemption 없음

### 8.3 피해야 할 정책

- 높은 priority 하나로 quota와 admission을 우회
- 한 session을 무조건 같은 worker에 고정
- workflow 전체를 하나의 giant request처럼 lock
- long tool call의 KV를 무제한 hard pin
- sub-agent 수만큼 tenant fair share를 늘려주는 정책

### 8.4 필요한 지표

```text
queue wait by latency class
workflow active/waiting count
per-workflow concurrent branches
workflow fair-share debt
priority clamp/rejection reason
starvation age
estimated vs actual output tokens
workflow completion time
```

Opaque workflow ID 자체는 metric label이 아니라 bounded trace field로만 남긴다.

---

## 9. Tool-gap-aware exact KV lifecycle

### 9.1 C12와의 관계

C12는 exact KV block의 identity, tenant boundary, ownership, publish/lookup/eviction을 먼저 닫는다. Agentic cache policy는 그 위에 얹는다.

```text
C12: 이 block을 재사용해도 정확하고 안전한가?
A04: 안전한 block 중 무엇을 얼마나 오래 남길 것인가?
```

A04가 C12 identity를 완화하거나 session ID만으로 hit를 만들면 안 된다.

### 9.2 제안 상태

```text
Active(request)
  -> ReusableIdle(session, expiry, reuse_score)
  -> Active(next turn)
  -> Cached
  -> Evicting
  -> Free
```

`ReusableIdle`은 hard pin이 아니라 best-effort retention이다. HBM pressure가 높으면 policy에 따라 evict할 수 있어야 한다.

### 9.3 Retention score

첫 버전은 학습 모델 없이 설명 가능한 비용식으로 시작한다.

```text
reuse_value
  = reusable_prefix_tokens
  × measured_prefill_cost_per_token
  × bounded_resume_probability

retention_cost
  = occupied_kv_bytes × memory_pressure
  + expected_eviction_cost_for_other_requests
```

`reuse_value > retention_cost`인 entry를 우선 보존하되 tenant/workflow quota와 TTL 상한을 먼저 적용한다. `expected_resume_after_ms`는 신뢰된 hint 또는 과거 bounded histogram에서만 사용한다.

### 9.4 Lifecycle event

상위 harness가 명시적으로 보낼 수 있는 event를 별도 control contract로 둔다.

```text
ProgramPaused(expected_resume_after_ms)
ProgramResumed
ProgramClosed
WorkflowCancelled
```

- `ProgramClosed`는 해당 private cache retention을 조기 해제할 수 있다.
- close event가 유실돼도 TTL과 quota로 메모리는 bounded여야 한다.
- cancelled/failed/uncommitted block은 C12 규칙대로 publish하지 않는다.

### 9.5 Host offload는 후속 단계

HBM에서 eviction하는 대신 pinned host memory로 옮기는 것은 가능하지만 초기 A04에 섞지 않는다.

- PCIe transfer가 재계산보다 빠른 shape인지 측정해야 한다.
- offload/prefetch queue가 decode를 방해하지 않아야 한다.
- host memory도 tenant quota와 zeroization/lifecycle을 가져야 한다.
- 잘못된 prefetch는 GPU bandwidth와 HBM을 낭비하므로 별도 D02 gate가 필요하다.

---

## 10. Structured tool-output runtime

Structured generation은 agent framework가 아니라 inference runtime이 직접 최적화할 수 있는 영역이다.

### 제안

- JSON schema/grammar의 canonical serialization
- `grammar_hash + tokenizer_revision + compiler_version` cache key
- bounded compiled-grammar cache
- async/cold compile과 request admission 분리
- token step마다 허용 token mask를 생성하는 exact constrained decoding
- C08 executable pattern registry를 통한 CPU/reference/GPU candidate 선택
- schema miss/compile failure 시 명시적 reject 또는 configured exact fallback

### 기대 효과

- invalid JSON/tool call로 인한 retry 감소
- 반복 schema compile 제거
- tool-call correctness의 기계적 검증
- grammar engine CPU bottleneck 감소 가능

단일 request의 raw TPOT가 반드시 개선되는 것은 아니다. 효과는 `invalid-call rate`, `retry-adjusted workflow latency`, `grammar compile/cache hit`, `mask latency`로 판정한다.

---

## 11. Speculative/prepared prefill

Tool 실행 중 다음 prompt의 **확정된 부분**을 미리 prefill하거나, 높은 확률의 다음 branch를 낮은 우선순위로 준비할 수 있다.

### 정확성 계약

- 실제 next-turn token IDs가 준비한 token IDs와 exact match할 때만 KV를 commit한다.
- mismatch, timeout, workflow cancel 시 speculative state를 폐기한다.
- speculative block은 일반 active request보다 낮은 eviction priority를 가진다.
- RNG와 generated token state를 추측 경로와 공유하지 않는다.
- user-visible token을 speculative branch에서 먼저 publish하지 않는다.

### 실행 정책

- GPU idle 또는 prefill 여유 budget에서만 수행
- request/tenant/workflow별 speculative token 상한
- 예상 resume 시간과 prefix 길이로 admission
- 실제 foreground request가 오면 즉시 iteration-level로 양보
- `prepared_tokens`, `used_tokens`, `discarded_tokens`, `wasted_gpu_ms`를 회계

현재 Riley는 completions 중심 API이므로 A06 전에 chat/tool turn을 token-level로 안정적으로 normalize하는 계약이 필요하다.

---

## 12. KV-aware distributed routing

### 12.1 Router cost model

Worker 선택은 단순 round-robin이나 active request 수만으로 하지 않는다.

```text
routing_cost(worker)
  = predicted_queue_delay
  + remaining_prefill_compute
  + kv_transfer_cost
  + overload_penalty
  - reusable_kv_value
```

낮은 cost를 선택한다. Session affinity는 `reusable_kv_value`에 반영되는 한 요소이지 절대 sticky rule이 아니다.

### 12.2 필요한 worker signal

```text
worker epoch/readiness
active prefill/decode tokens
admission headroom
KV capacity and pressure
opaque exact-prefix block inventory summary
transfer bandwidth/topology class
recent queue and execution latency
```

Raw prompt/token 또는 tenant ID를 router metric label에 노출하지 않는다. Cache directory는 versioned block identity와 create/free/move event를 사용하고 stale worker epoch를 거절한다.

### 12.3 소유권 경계

- Riley worker: local block identity, refcount, lifetime, transfer source/destination validation
- Router/control plane: worker discovery, placement score, retryable routing decision
- Deployment layer: autoscaling, node lifecycle, network policy
- Agent harness: workflow graph와 semantic retry

Riley core가 처음부터 Kubernetes scheduler나 범용 message broker가 될 필요는 없다. Engine이 안전한 cache/transfer telemetry와 control ABI를 제공하고, router는 별도 component로 시작하는 편이 낫다.

### 12.4 Prefill/decode disaggregation

Prefill과 decode는 GPU utilization과 latency 특성이 다르므로 별도 pool로 나누는 전략은 agent의 긴 observation prompt에서 가치가 있을 수 있다. 하지만 Riley에서는 다음이 선행돼야 한다.

- multi-GPU topology 승인
- exact KV transfer ABI
- transfer 중 block ownership과 cancellation
- prefill/decode worker epoch
- queue와 network tail latency 포함 benchmark
- non-disaggregated rollback path

분리 자체가 항상 빠른 것은 아니다. 짧은 prompt나 느린 interconnect에서는 transfer overhead가 이득을 상쇄할 수 있으므로 D03에서 workload별로 판정한다.

---

## 13. Multi-agent 전용 최적화의 한계

### Collective KV sharing

여러 agent가 같은 context를 읽고 동시에 각자 decode하는 synchronized round/all-gather 패턴에서는 공통 KV를 한 번 만들고 공유하는 방식이 가능하다. 다만 일반적인 비동기 ReAct workflow는 branch별 token sequence와 진행 시점이 달라 적용률이 낮을 수 있다.

다음 trace 조건이 실제로 확인될 때만 R01 후보로 올린다.

```text
동일 exact prefix
동일 model execution identity
동일 trusted sharing domain
좁은 시간창의 동시 fan-out
각 branch가 prefix 이후에만 분기
```

### Position-independent 또는 semantic reuse

표준 causal attention에서는 position과 앞선 token sequence가 KV 의미에 포함된다. Position-independent reuse는 특정 attention architecture나 별도 변환 계약에 의존할 수 있다. Cross-context reuse는 잘못 설계하면 tenant 격리와 prompt integrity까지 위협한다.

따라서 초기 Riley production은 다음 원칙을 유지한다.

- exact token prefix만 재사용
- model/tokenizer/RoPE/layout/adapter/tenant identity 모두 일치
- position-independent candidate는 model별 연구 extension
- semantic similarity만으로 KV hit 금지
- learned pruning은 quality benchmark와 opt-in 없이는 금지

---

## 14. Agentic benchmark contract

A01은 production code를 바꾸지 않고 workload와 판정식을 먼저 고정한다.

### 14.1 Workload suite

| ID | 패턴 | 주요 축 |
|---|---|---|
| W1 | Serial ReAct | 8/16/32 turns, tool gap 50 ms/1 s/10 s |
| W2 | Coding agent | 4K→32K growing prefix, 큰 tool output, 짧은 decision output |
| W3 | Fan-out/fan-in | 2/4/8/16 branches, final synthesis |
| W4 | Shared schema | 동일 JSON/tool schema 반복, cache cold/warm |
| W5 | Mixed SLO | interactive와 background workflow 혼합 |
| W6 | Cancellation | tool wait/branch/final synthesis 각 상태에서 cancel |
| W7 | Cache pressure | 여러 tenant/workflow와 제한된 KV capacity |
| W8 | Distributed locality | worker별 cache overlap/load 불균형, 후속 D track |

Prompt와 tool result는 synthetic fixture와 승인된 trace-derived shape를 분리한다. 실제 prompt 원문이나 credential을 benchmark artifact에 넣지 않는다.

### 14.2 핵심 metric

| 영역 | metric |
|---|---|
| Workflow | completion/JCT p50/p95/p99, successful workflow/s |
| Turn latency | TTFT, tool-resume TTFT, E2E, TPOT/ITL |
| KV | exact hit rate, reused tokens/blocks, recomputed tokens, retention hit/miss |
| Memory | HBM/host KV bytes, eviction, quota rejection, fragmentation proxy |
| Scheduling | fair-share debt, starvation, priority clamp, branch concurrency |
| Speculation | prepared/used/discarded tokens, wasted GPU ms |
| Structured output | compile/cache latency, invalid output, retry-adjusted latency |
| Distributed | routing decision latency, cache-local route, KV transfer bytes/time |
| Reliability | timeout, cancel completion, duplicate terminal, stale epoch event |
| Security | cross-domain hit, unauthorized priority, raw-ID metric exposure |

### 14.3 Baseline arms

```text
B0: current request-level scheduler, cache off
B1: C12 exact prefix cache, agent hints off
B2: candidate feature one 개만 on
B3: approved feature composition
COMP: 같은 campaign의 current vLLM/SGLang 또는 지정 competitor
```

기능을 한꺼번에 켜고 효과를 합쳐 보고하지 않는다. 각 PR은 한 가지 가설을 독립 arm으로 판정한다.

### 14.4 공통 guardrail

정확한 promotion threshold는 A01 contract에서 결과를 보기 전에 고정한다. 최소 guardrail은 기존 competitive roadmap을 따른다.

- generated token/canonical numeric mismatch 0
- cross-tenant/domain cache hit 0
- request/workflow starvation 0
- duplicate terminal/stale event publish 0
- hint 미사용 baseline의 TTFT/TPOT p95 회귀 3% 이내
- throughput 3% 이상 회귀 시 승격 금지
- hot-loop allocation delta 0
- configured memory/queue/token bound 초과 0
- feature off가 current exact behavior와 동일
- append-only raw evidence와 closed report 생성

Agent-specific primary gain은 `tool-resume TTFT`, `recomputed prefill tokens`, `workflow JCT`, `SLO goodput` 중 해당 PR이 사전에 선택한 지표로 판정한다.

---

## 15. 제안 PR 순서

| ID | 문서/PR 목적 | production 변화 | 선행 조건 |
|---|---|---|---|
| A01 | Agentic workload·trace·metric contract | 없음 | 없음 |
| A02 | Passive context와 advisory hint schema | default behavior 없음 | A01 일부 |
| A03 | Workflow-fair scheduler lanes/budget | scheduler policy, flag off | A02, C03 권장 |
| A04 | Tool-gap exact KV retention/TTL | cache policy, flag off | C12-B, A02 |
| A05 | Structured output grammar compile/cache | constrained decode, flag off | C08, API contract |
| A06 | Prepared/speculative prefill | idle prefill, flag off | C12-B, A02, token API |
| D01 | Worker cache directory와 KV-aware router | multi-worker routing | C13, architecture approval |
| D02 | HBM→host KV offload/prefetch | memory tier | D01, transfer benchmark |
| D03 | Prefill/decode disaggregation | worker pools와 KV transfer | D01~D02, multi-GPU gate |
| R01 | Collective multi-agent KV sharing study | 없음/experimental | real trace evidence |
| R02 | Position-independent/learned reuse study | 없음/experimental | model-specific proposal |

### 권장 착수

```text
지금: A01 -> A02
C03/C12 진행과 병행: A03 설계 및 simulation
C12-B exact cache 이후: A04
API와 model fast path가 안정된 뒤: A05/A06
C13과 single-GPU M4/M5 이후: D01 -> D02 -> D03
```

A01/A02는 기존 vLLM competitive roadmap을 방해하지 않고 시작할 수 있다. A04는 C12의 exactness/lifetime을 우회하지 않는다. D track은 현재 로드맵의 초기 비범위인 multi-GPU를 암묵적으로 확대하지 않고 별도 architecture gate를 거친다.

---

## 16. 예상 파일 경계

### A01/A02

```text
benchmarks/agentic/README.md
benchmarks/agentic/contracts/agentic-workload-v1.json
benchmarks/agentic/contracts/agentic-workload-v1.schema.json
benchmarks/agentic/traces/synthetic-*.jsonl
benchmarks/agentic/scripts/run_campaign.py
benchmarks/agentic/scripts/check_campaign.py
crates/riley-server/src/domain.rs
crates/riley-server/src/service.rs
crates/riley-scheduler/src/scheduler.rs
crates/riley-scheduler/src/metrics.rs
```

A02는 internal contract만 추가할 경우 `openai.rs`를 바꾸지 않는다. Public API exposure는 별도 PR이다.

### A03/A04

```text
crates/riley-scheduler/src/config.rs
crates/riley-scheduler/src/scheduler.rs
crates/riley-scheduler/src/metrics.rs
crates/riley-runtime/src/prefix_cache.rs
crates/riley-runtime/src/paged_kv.rs
crates/riley-scheduler/tests/agentic_simulation.rs
crates/riley-runtime/tests/agentic_cache_lifecycle.rs
```

### A05/A06

```text
crates/riley-runtime/src/pattern.rs
crates/riley-runtime/src/structured_output/*
crates/riley-runtime/src/prefix_cache.rs
crates/riley-scheduler/src/speculative_prefill.rs
crates/riley-server/src/domain.rs
crates/riley-server/src/service.rs
```

### D track

초기에는 production crate 수를 늘리지 않고 `riley-server`의 별도 binary/module 또는 명시적 router binary로 검토한다.

```text
crates/riley-server/src/bin/riley-router.rs
crates/riley-server/src/routing/*
crates/riley-server/src/ipc/*
crates/riley-runtime/src/kv_transfer/*
benchmarks/agentic/distributed/*
```

새 crate 또는 remote protocol은 별도 architecture approval과 versioned schema가 필요하다.

---

## 17. 운영 flag 제안

아래는 이름 제안이며 실제 값과 default는 각 PR contract에서 고정한다.

```text
--agent-context disabled|passive
--workflow-scheduling disabled|fair
--agent-cache-retention disabled|best-effort
--structured-output disabled|enabled
--speculative-prefill disabled|idle-only
--routing-policy load-only|kv-aware
--kv-offload disabled|host
--serving-topology unified|disaggregated
```

모든 첫 implementation은 `disabled` 또는 behavior-neutral `passive`가 default다. Rollback은 단일 flag로 가능해야 하며 cache metadata/worker protocol처럼 schema가 연결된 기능은 관련 component를 함께 rollback한다.

---

## 18. 하지 말아야 할 구현

- Riley 안에 범용 agent orchestrator를 구축
- session ID만 같다는 이유로 KV를 공유
- client가 tenant/cache namespace를 선택
- untrusted priority로 quota 우회
- tool wait 동안 KV를 무기한 hard pin
- session sticky routing으로 overload를 무시
- exact fallback 없이 semantic cache 적용
- 표준 GQA model에 position-independent reuse를 일반화
- 품질 gate 없이 learned KV pruning을 production default로 적용
- single-GPU correctness/competitive evidence 전에 multi-GPU 복잡도를 핵심 runtime에 혼합
- 실제 workload trace 없이 collective multi-agent 전용 primitive를 먼저 구현

---

## 19. 완료 정의

이 roadmap의 첫 production milestone은 다음 조건을 만족할 때다.

1. Agent harness가 closed context/hint schema로 lifecycle signal을 전달할 수 있다.
2. Hint가 없으면 기존 request-level 의미와 성능을 보존한다.
3. 동일 tenant/workflow의 fan-out이 전체 scheduler를 독점하지 않는다.
4. C12 exact identity를 유지한 채 tool-resume turn의 재계산을 줄인다.
5. cancellation, close, TTL expiry, eviction에서 KV ownership과 accounting이 0으로 닫힌다.
6. Raw prompt/session/workflow ID가 metric label이나 crash log에 노출되지 않는다.
7. Agentic campaign이 workflow JCT, resume TTFT, recomputed tokens, SLO goodput을 재현 가능하게 판정한다.
8. 모든 feature가 default-off/passive, exact fallback, bounded memory, 명시적 rollback을 가진다.

분산 milestone은 추가로 cache directory의 stale event 0, cross-worker block ownership 오류 0, routing fallback, worker epoch, KV transfer cancellation, topology별 closed report를 만족해야 한다.

---

## 20. 조사 근거

아래 자료는 agentic serving에서 prefix-heavy, priority-sensitive, tool-gap, program/session identity, KV-aware routing, structured generation, disaggregated serving이 실제 연구·구현 대상임을 확인하기 위해 사용했다. 논문의 수치는 Riley에서 재현되기 전까지 Riley의 성능 주장으로 사용하지 않는다.

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [vLLM Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [NVIDIA Dynamo Agent Hints](https://docs.nvidia.com/dynamo/agents/agent-hints)
- [NVIDIA Dynamo Agentic Inference](https://docs.nvidia.com/dynamo/dev/digest/agentic-inference)
- [Continuum](https://arxiv.org/abs/2511.02230)
- [TokenCake](https://arxiv.org/abs/2510.18586)
- [TokenDance](https://arxiv.org/abs/2604.03143)
- [AgentServeSim](https://arxiv.org/abs/2606.09613)
- [XGrammar](https://arxiv.org/abs/2411.15100)
- [ToolSpec](https://arxiv.org/abs/2604.13519)
- [DistServe — OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [Mooncake — FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)
- [Irminsul](https://arxiv.org/abs/2605.05696)
- [HijackKV](https://arxiv.org/abs/2607.19957)

---

## 21. 한 문장 방향

**Riley는 agent를 실행하는 framework가 아니라, agent의 반복 호출·tool 대기·fan-out·cache locality를 이해해 exact inference를 더 적은 재계산과 더 나은 tail latency로 제공하는 runtime을 목표로 한다.**
