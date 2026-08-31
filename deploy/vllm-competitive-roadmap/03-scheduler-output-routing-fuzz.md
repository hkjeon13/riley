# C03 — Scheduler Output Routing Property Fuzz

**상태:** In progress — C03-A CPU-only reference harness는 C02-P1 source closure 뒤 병렬로
진행할 수 있다. C03-B GPU fixed corpus와 C03의 formal completion은 C02 actual qualification
뒤에만 수행한다.
**의미 등급:** `reference`  
**한 가지 목적:** scheduler plan부터 sampling/commit/terminal event까지 request-token 대응 관계를 model-based property test로 고정한다.

[이전: C02](02-rc3-candidate-qualification.md) | [목차](README.md) | [다음: C04](04-llama-executor-refactor.md)

## 0. 병렬 착수 경계

**C03-A**는 clean committed source와 C02-P1 source closure만을 전제로 하는 CPU-only
reference model, deterministic corpus, seeded property harness다. scheduler의 public host API와
synthetic `IterationResult`만 사용하며 CUDA, server, Docker, root service, Gate E, candidate
freeze, evidence 또는 qualification을 실행하거나 주장하지 않는다. 따라서 C02-P2의
administrator provisioning과 병렬로 구현할 수 있지만, 그 결과는 candidate-bound routing
evidence나 C02 pass가 아니며 이후 actual C02 candidate에 포함해 재검증한다.

**C03-B**의 GPU corpus/fixture source는 same-candidate integration replay를 주장하려면 C02
candidate freeze 전에 그 candidate source archive에 포함되어야 한다. GPU fixed-corpus execution,
integration replay, formal C03 acceptance는 C02 actual qualification 뒤에만 수행한다. C02 뒤에
그 source가 바뀌면 same-candidate 주장을 거부하고 새 C02 candidate와 requalification부터
다시 시작한다. C03-A의 green CPU test는 C03-B, C02, M4/M5 또는 vLLM win을 대체하지 않는다.

## 1. 배경

RC1의 mixed-stage defect는 output slot 집합은 dense했지만 sampling 순서와 request mapping이 달라 token이 잘못된 request에 연결될 수 있음을 보여줬다. 단일 regression fixture만으로는 prefill/decode 순열, cancellation, commit failure의 조합을 충분히 탐색하지 못한다.

이 PR은 production 최적화를 하지 않는다. 현재 state machine과 `IterationPlan`의 observable behavior를 독립 reference model과 비교한다.

## 2. 핵심 invariant

모든 생성 sequence에서 다음이 성립해야 한다.

1. GPU output slot 하나는 정확히 하나의 `(request_id, generation_step)`에 대응한다.
2. request는 하나의 iteration에서 최대 한 개의 sampled output을 받는다.
3. sampled token은 canonical dense slot 순서와 무관하게 원래 request에 정확히 연결된다.
4. terminal event는 request당 정확히 한 번 발생한다.
5. cancel/fail/timeout request는 이후 token을 받지 않는다.
6. reserve된 KV block은 commit, rollback, free 중 정확히 하나로 정산된다.
7. commit 실패 후 downloaded token이 다른 request에 재사용되지 않는다.
8. request 순서 변화가 다른 request RNG state를 오염시키지 않는다.
9. malformed plan은 GPU dispatch 전에 거부된다.
10. close 후 pending request, promised block, completion outbox가 0이다.

## 3. Reference model

테스트 전용 순수 Rust model을 만든다.

```text
ReferenceSchedulerState
  waiting requests
  admitted requests
  request phase and version
  logical token history
  logical KV ownership
  expected output map
  RNG snapshot/version
  terminal event ledger
```

Reference model은 CUDA, tensor pointer, production scheduler 내부 collection을 사용하지 않는다. production operation과 동일한 abstract event만 소비한다.

## 4. 생성할 operation

```text
submit(request, prompt_len, output_len, seed)
admit
plan_iteration(token_budget, prefill_cap)
produce_gpu_outputs(permuted_slots)
commit
abort_pre_dispatch
abort_post_execute
cancel(request)
timeout(request)
client_disconnect(request)
inject_sampling_failure(slot)
inject_commit_failure(request)
finish(request)
shutdown
```

Generator는 valid operation을 주로 만들되, duplicate slot, missing output, stale block-table version, unknown request 같은 invalid plan도 일정 비율로 생성한다.

## 5. 상태 공간

- request count `0..16`
- prompt length `0..64`의 small symbolic token sequence
- output length `1..16`
- prefill/decode 혼합
- token budget `1..32`
- KV page size를 작은 symbolic 값 `2,4,8`로 축소
- cancellation 시점: waiting/admitted/prefill/decode/post-execute/pre-commit
- output slot permutation 전수 또는 random
- deterministic seed와 request insertion order permutation

작은 상태 공간에서는 exhaustive permutation을 사용하고, 큰 조합은 seeded random property run으로 확장한다.

## 6. 테스트 계층

### C03-A — CPU fast property

- PR마다 최소 10,000 generated traces
- nightly 또는 scheduled run에서 1,000,000 traces
- failure 시 source seed, full descriptor, grammar candidate-minimized trace 출력

현재 구현 slice는 CUDA를 쓰지 않는 10,000 valid-feedback permutation trace와 10,000
`FaultAction` microtrace, 10,000 bounded mixed-stage trace, 10,000 bounded operation-sequence
V2 trace다. `FaultAction` microtrace는 deferred cancel 뒤 commit/`NotDispatched` abort,
`DeviceQuiescedMutationUnknown` abort, waiting timeout, stale/missing/unplanned feedback이
output-slot ledger·terminal-once·KV/queue quiescence를 깨지 않는지 확인한다. bounded mixed-stage
trace는 `MixedStageTraceV1 { seed, decoder_max_new_tokens, final_prefill_len, action }`를 seed에서
생성하고, `decoder prime → decoder decode slot 0 + final-prefill slot 1 → optional deferred decoder
cancel → explicit [slot 1, slot 0] feedback/commit → close`를 public scheduler API로 재생한다.

V2는 같은 RC1 mixed prefix를 concrete `Submit`, `Plan`, `Complete`, `Cancel`, `RejectFeedback`,
`AbortNotDispatched`, `Close` operation list로 canonicalize한다. seed가 선택한 최대 9개 operation은
decoder cancel, stale/missing/unplanned feedback의 surface-preserving rejection, reverse feedback
commit에 의한 valid retry 또는 `NotDispatched` abort, consuming close를 조합한다. 각 operation은 public
scheduler API만 호출하며 planning/update/close 전체에서 terminal-once, token/index ledger, rejected
feedback 뒤 surface 불변성, final quiescence를 확인한다. 실패 시 source seed를 포함한 canonical
descriptor JSON과 derived canonical operation list를 출력한다. failure 때는 source seed와 full descriptor를 보존한 채, 같은 V2 failure
predicate가 재현되는 경우에만 optional rejection/cancel을 제거하고 decoder output capacity와
final-prefill length를 더 작은 값으로 바꾸는 greedy shrinker도 실행한다. rejection kind, settlement
(`NotDispatched` abort 또는 reverse commit), prime/mixed prefix와 reverse feedback은 바꾸지 않는다.
report는 원본과 candidate-minimized V2 descriptor JSON 및 canonical operation list를 함께 남긴다. seed만으로
shrunken descriptor를 재생성한다는 뜻은 아니다. predicate는 inner replayer가 panic하는지만 보존하므로
same assertion site, failure signature 또는 root cause를 보존한다고 주장하지 않는다.

V2는 별도 test-only pure `V2RoutingOracle`도 사용한다. 이 oracle은 scheduler나
`IterationPlan`을 보유·순회하지 않고, bounded grammar가 정한 decoder A=`slot 0`,
final-prefill B=`slot 1`, symbolic token/index, deferred cancellation, terminal ledger와 close
결과만으로 feedback과 기대 public update를 만든다. adapter는 public plan의 request binding·stage·input·slot
구조를 oracle 기대와 별도로 비교하고 iteration ID만 oracle feedback에 전달한다. 따라서 production
plan에서 feedback의 request mapping을 다시 유도하지 않으며, reverse feedback의 token/completion은
opaque public request ID에 bind한 oracle ledger와 비교한다. stale/missing/unplanned rejection 뒤에는
oracle phase가 advance하지 않고 기존 public surface non-mutation check도 유지한다. 이는 fixed
two-request V2 grammar의 independent routing/lifecycle oracle일 뿐 admission/aging/KV allocation을
재구현하거나 general multi-request reference scheduler가 완성됐다는 뜻은 아니다.

V2에는 test-only typed `OperationTraceV2DescriptorV1` codec도 있다. `format`,
`format_version`, `trace_kind`, `case_id`, fixed-width lowercase `source_seed`와 bounded
selector를 canonical JSON document로 고정하고, committed fixture를 exact byte round-trip 한 뒤
public scheduler API로 재생한다. `rejected_feedback`는 반드시 명시적인 JSON `null` 또는 known
kind여야 하며, unknown/missing/duplicate field, noncanonical formatting/order, unsupported
format/kind/version, invalid seed/bounds는 decode 단계에서 거부한다. `source_seed`는 provenance일
뿐 shrunken selector의 재생성 recipe가 아니므로, 축약된 failure는 그 전체 canonical document로
재생한다. 이 codec은 current bounded V2 grammar 전용 test artifact이며 arbitrary/general trace나
portable historical runtime configuration을 serialize한다고 주장하지 않는다.

이는 여전히 C03-A의 부분 범위다. unbounded/general mixed-operation generator와 그 전체 grammar의
shrink/global-minimum counterexample, general multi-request independent reference model, scheduled 1,000,000 seed rotation,
post-validation sampling/commit fault injection은 남아 있다. failure-signature/same-assertion preservation,
multi-edit/delta-debugging reduction도 별도 범위다. V2
shrinker는 이 작은 grammar의 greedy local minimum일 뿐 일반 trace shrinker나 globally minimal
counterexample을 주장하지 않는다. 마지막 항목은 현 public scheduler API에 injection seam이 없으므로
별도 test-only seam 계약으로 설계한다.

### Deterministic corpus

과거 defect와 발견된 모든 fuzz counterexample을 JSON 또는 Rust fixture로 영구 등록한다.

현재 C03-A corpus는 Rust `MixedStageTraceV1` entries로 RC1 mixed-stage 최소 재현을 보존한다.
decoder A의 `slot 0`과 final-prefill B의 `slot 1`에 대해 physical feedback vector를 명시적으로
`[slot 1, slot 0]`으로 역순 제출한다. normal commit, deferred decoder cancellation, 그리고 남은
decoder를 `close`가 정확히 한 번 취소해야 하는 output capacity 3/4 케이스를 재생한다. 모두 host-side
synthetic scheduler feedback의 request/token/generation-index mapping·terminal history·quiescence를
검증하며, GPU mixed execution이나 stream cancellation을 실행하거나 주장하지 않는다.

`OperationTraceV2` corpus는
`crates/riley-scheduler/tests/corpus/output-routing/operation-trace-v2/*.json`의 committed canonical
document로 같은 RC1 normal/cancel prefix를 보존하고, unplanned feedback rejection 뒤 valid reverse
retry와 missing feedback rejection 뒤 deferred cancel + `NotDispatched` abort도 등록한다. fixture는
typed strict decode, canonical byte round-trip, V2 pure oracle와 host-side synthetic feedback replay를
모두 통과해야 한다.
이는 GPU fixture나 C02/C03-B evidence가 아니다. `IterationResult::new` 단계에서 막히는 duplicate
slot이나 immutable public plan으로 만들 수 없는 stale block-table version은 scheduler boundary property라고
주장하지 않는다.

### C03-B — GPU integration slice

CPU model이 생성한 대표 trace 중 다음만 실제 CUDA path로 replay한다.

- `C=5`, `C=8` mixed output
- KV boundary
- GPU greedy output
- commit failure와 cancellation

GPU test는 property runner 전체를 실행하지 않고 고정된 최소 corpus만 검증한다.

## 7. Shrinking

현재 `OperationTraceV2`는 failure가 발생하면 같은 property predicate로 bounded candidate를 다시
replay해 optional operation 제거와 output/prefill length 축소 순서의 local minimum을 출력한다. seed와
rejection kind·settlement은 고정하며, exact canonical descriptor JSON과 derived operation list 전체가
shrunken replay 입력이다. 이 minimizer 자체는 deterministic synthetic predicate와 모든 bounded candidate의 grammar
replay로 candidate order·local-minimum 성질을 고정한다. 실제 defect counterexample이나 arbitrary trace를
이 경로가 이미 global minimum으로 축약했다는 뜻은 아니다.

일반 generator가 추가된 뒤에는 다음 순서로 trace를 축약한다.

1. request 수 감소
2. iteration 수 감소
3. prompt/output 길이 감소
4. operation 제거
5. output slot permutation 단순화
6. cancellation/failure 시점 단순화

그 general final report는 source seed와 full replay descriptor, 사람이 읽을 수 있는 globally minimized
operation list를 함께 포함한다. general grammar의 durable format은 V2 codec을 재사용한다고 미리
가정하지 않고 별도 versioned contract로 설계한다. 이는 현재 V2 local shrinker보다 강한 후속 기준이다.

## 8. 예상 파일 변경

```text
crates/riley-scheduler/tests/model_based_routing.rs
crates/riley-scheduler/tests/support/reference_scheduler.rs
crates/riley-scheduler/tests/corpus/output-routing/*.json
crates/riley-scheduler/Cargo.toml   # dev-dependency only, 필요 시
benchmarks/scripts/check_routing_fuzz_receipt.py
.github/workflows/production-cpu.yml
```

새 dependency를 추가할 경우 production dependency graph에 포함되지 않는 dev-dependency여야 하고 `Cargo.lock`을 고정한다.

## 9. Observability assertion

production scheduler metric snapshot도 reference state와 비교한다.

- active/waiting/terminal gauges
- cancellation/failure counters
- KV promised/active/free totals
- completion outbox count
- degraded metric flag

metric recording 실패를 주입한 경우 inference ownership은 유지되고 `metrics_degraded`만 sticky하게 설정되어야 한다.

## 10. Negative properties

다음 behavior가 발생하면 즉시 failure다.

- token이 request history에 두 번 append됨
- unknown/finished request에 token append
- 두 request가 같은 exclusive physical KV block 소유
- terminal event 중복
- cancelled branch RNG 소비가 surviving branch에 반영됨
- malformed plan이 runtime adapter에 도달
- shutdown 후 live state 잔존

## 11. CI 정책

- 기본 CPU workflow: deterministic seed set + 10,000 traces
- scheduled workflow: random seed rotation + 1,000,000 traces
- GPU manual/scheduled: 고정 corpus
- CI failure artifact: seed, minimized trace, scheduler config, Git SHA

flaky retry로 통과시키지 않는다. 동일 seed가 재현되지 않으면 test harness defect로 별도 실패한다.

## 12. 승인 기준

- C03-A: fixed/random CPU seed에서 invariant 위반 0, deterministic corpus와 shrink/replay test 통과
- RC1 최소 재현 fixture가 수정 전 실패/현재 통과하는 contract test로 보존
- shrink된 counterexample serialization/replay test 통과
- production runtime 코드의 semantic change 없음
- CPU test runtime이 일반 PR CI budget 내
- C03-B: C02 actual qualification 뒤 GPU corpus에서 output token/request mapping exact

## 13. 롤백

테스트-only 변경이므로 rollback은 신규 test/support/corpus를 함께 revert한다. 다만 실제 defect를 재현하는 corpus는 원칙적으로 삭제하지 않고, harness 교체 시 새 형식으로 이관한다.

## 14. 완료 정의

C03-A의 formal CPU completion은 scheduler event 순서와 output slot permutation을 deterministic
seeded CPU trace로 생성해 reference model과 production state가 일치하고, mixed-operation
generator·shrink·counterexample corpus/replay·failure seed와 최소 trace까지 갖출 때다. 현재
valid/fault microtrace slice는 이를 향한 partial coverage이며 아직 C03-A 완료 선언 근거가 아니다.
C03의 formal completion은 C03-B가 C02 actual qualification 뒤 GPU corpus에서도 exact mapping을
확인할 때만 선언한다.
