# C03 — Scheduler Output Routing Property Fuzz

**상태:** In progress — C03-A CPU-only reference harness의 일부와 C03-Ax의 deterministic
test-only post-validation fault-containment unit contract, C03-Ay의 scheduled CPU seed-band rotation,
C03-Az의 bounded raw-program stateful local reducer와 in-flight raw-program local reducer, 그리고
`inflight-mixed-program-v2`의 caller-asserted quiesced terminal-abort CPU contract가 구현되었다.
C03-B의 fixed corpus/CPU topology contract source는 C02-P1 source closure 뒤 병렬로 진행할 수 있다.
C03-B의 actual CUDA fixed-corpus execution과 C03의 formal completion은 C02 actual qualification
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

C03-Ax는 완전한 validation 뒤 containment만 검증하는 좁은 예외로, 두 private `#[cfg(test)]`
unit seam을 사용한다. 이는 public API/re-export, Cargo feature, 환경변수·설정, server/CUDA path를
추가하거나 바꾸지 않고 arbitrary trace operation으로 노출되지 않는다.

C03-Az도 bounded raw-program test support와 CPU adapter 안에서만 실행된다. 실제 inner replayer가
panic한 뒤에만 fixed local candidate order를 재생하며, production scheduler semantic, CUDA, GPU,
receipt/checker, 환경변수·설정 또는 C02 qualification을 추가·변경·주장하지 않는다.

in-flight raw-program local reducer도 같은 test-only CPU 경계에 남는다. 그것은 raw descriptor의
label/feedback slot을 rebase하지 않으며, 기존 grammar validation과 strict codec이 허용한 lifecycle
contraction만 actual inner replayer panic 뒤에 다시 재생한다. 별도 `inflight-mixed-program-v2`는
caller가 이미 quiesced라고 단언한 `DeviceQuiescedMutationUnknown` disposition을 public host API로
소비하는 terminal branch만 검증한다. CUDA synchronization, 실제 device mutation, GPU replay 또는
성능 evidence는 이 CPU contract로부터 성립하지 않는다.

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

- PR마다 각 CPU property lane의 deterministic 10,000 generated traces
- nightly 또는 scheduled run에서 최소 1,000,000 traces
- failure 시 source seed와 full descriptor를 출력한다. candidate-minimized trace는 reducer가 명시적으로
  구현된 grammar slice에만 추가한다.

#### C03-Ax — test-only post-validation fault-containment seam

C03-Ax는 public scheduler API로는 도달할 수 없는, 이미 valid한 feedback의 publication 직전
failure containment만 CPU unit test로 고정한다. 이는 general fault-injection grammar나 production
failure mode를 추가하는 작업이 아니다.

- sampling seam은 dense sample count, output-count의 `u32` 변환, vocabulary 검증이 모두 끝난 뒤,
  commit output DTO를 push하기 직전에 한 번만 실패한다. 실패는 owning `IterationCommitFailure`로
  downloaded output을 돌려주며, abort data는 계속 `DeviceQuiescedMutationUnknown`이다.
- scheduler seam은 iteration-result validation, output/completion capacity 확보, completion publication
  prevalidation이 모두 성공한 뒤 각 reservation의 `commit_reservation` 직전에만 동작한다.
  `after_successful_commits = 1`은 첫 reservation을 실제로 commit한 뒤 두 번째 reservation만
  실패시킨다.
- 두 seam은 private `#[cfg(test)]` state/method로만 존재한다. Cargo feature, CLI, 환경변수, config,
  public/reexported API, server/runtime/CUDA 경로를 추가하거나 바꾸지 않는다.
- acceptance는 valid sample이 token DTO로 publish되기 전 owning failure로 되돌아오는지, commit
  containment이 token 0개 publication, affected request 각각 정확히 한 번의 terminal
  `ExecutorFailure`, KV reclaim, iteration metric 부재, completed 0/aborted 1 metric, close 뒤
  ownership gauge 0을 보장하는지 검증한다. terminal tombstone, physical capacity, completion capacity는
  의도적으로 0이 아닌 gauge이므로 이 검사의 대상이 아니다.

C03-Ax는 actual device mutation, CUDA stream fault, multi-event trace grammar, reducer, candidate evidence
또는 C03 formal completion을 주장하지 않는다.

현재 구현 slice는 CUDA를 쓰지 않는 10,000 valid-feedback permutation trace와 10,000
`FaultAction` microtrace, 10,000 bounded mixed-stage trace, 10,000 bounded operation-sequence
V2 trace, 10,000 parameterized two-wave `general-mixed-operation-v1` trace, 10,000
bounded raw `bounded-mixed-program-v1` trace, 10,000 bounded raw in-flight
`inflight-mixed-program-v1` trace, 그리고 10,000 bounded raw in-flight
`inflight-mixed-program-v2` caller-asserted terminal-abort trace다. `FaultAction` microtrace는 deferred cancel 뒤 commit/`NotDispatched` abort,
`DeviceQuiescedMutationUnknown` abort, waiting timeout, stale/missing/unplanned feedback이
output-slot ledger·terminal-once·KV/queue quiescence를 깨지 않는지 확인한다. bounded mixed-stage
trace는 `MixedStageTraceV1 { seed, decoder_max_new_tokens, final_prefill_len, action }`를 seed에서
생성하고, `decoder prime → decoder decode slot 0 + final-prefill slot 1 → optional deferred decoder
cancel → explicit [slot 1, slot 0] feedback/commit → close`를 public scheduler API로 재생한다.

C03-Ay는 기본 PR의 여덟 10,000-trace CPU lane(총 80,000 trace)을 바꾸지 않는다. 전용
c03-routing-fuzz.yml scheduled/manual workflow는 매 run마다 기록 가능한 canonical decimal base를
선택하고, 15개 matrix slot에 서로 겹치지 않는 10,000-index band를 배정한다. 따라서 한 scheduled run은
8 × 10,000 × 15 = 1,200,000 CPU trace를 실행한다. test-only routing_fuzz_rotation helper는
base/slot/slot-count가 모두 absent일 때 기존 seed와 FaultAction 순서를 정확히 유지하고, 부분 설정,
non-canonical decimal, overflow를 fail closed한다. 각 lane의 final seed 또는 descriptor가 재현
identity이며, 기존 fault/general lane이 seed factor를 공유하므로 이것은 1,200,000 trace 실행이지
전역 unique seed 수 주장이 아니다. 이 workflow는 GPU target, CUDA feature, production runtime API 또는
성능 주장을 추가하지 않는다.

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

V1은 V2와 별도 `format="riley.scheduler.general-mixed-operation"`,
`trace_kind="general-mixed-operation-v1"` canonical descriptor를 사용한다. schema는 decoder A
`1..=3`개(모두 prompt 1, max-new 2)의 prime wave와 final-prefill B `1..=3`개(모두 prompt 1,
max-new 1)의 mixed wave를 고정한다. `prime_slot_order`와 `mixed_slot_order`는 각 wave의 정확한
slot permutation이고, `cancel_decoder_index`는 명시적 JSON `null` 또는 A index이며,
`settlement`는 commit 또는 `NotDispatched` abort다. source seed는 provenance일 뿐 selector를
다시 생성하는 recipe가 아니다. 이 별도 V1 descriptor 전체가 replay 입력이다.

`GeneralMixedOperationOracle`은 `Scheduler`나 `IterationPlan`을 import하지 않는다. decoder
A의 prime slot `i`와 mixed slot `i`, final-prefill B의 mixed slot `D + j`, symbolic token/index,
optional deferred cancellation, terminal ledger와 close를 descriptor와 opaque public request ID만으로
계산한다. public adapter는 plan의 request binding, prefill/decode kind, input, canonical slot,
total-token projection을 별도 assert하고 iteration ID만 oracle feedback builder에 넘긴다. 따라서
production plan item을 순회해 feedback mapping을 재구성하지 않는다. V1은 optional one-decoder
cancel, permuted valid commit 또는 pre-dispatch abort, close의 terminal-once/zero-gauge를
검증한다.

이는 parameterized fixed two-wave grammar다. partial prefill, queue/aging, invalid feedback/retry,
`DeviceQuiescedMutationUnknown`, arbitrary raw operation sequence 또는 general shrinker를 구현·주장하지
않는다. 그 영역은 기존 `FaultAction`/V2 또는 후속 general grammar PR의 범위다.

V1에는 이 bounded grammar 전용 `GeneralMixedOperationTrace::shrink_candidates`와
`minimize_general_mixed_operation_trace`가 있다. 후보 순서는 cancellation 제거, 더 작은 cancellation
target, decoder index별 제거와 selector rebase, final-prefill index별 제거와 mixed-slot compaction,
prime/mixed identity permutation, 각 order의 좌→우 adjacent inversion swap이다. candidate는 중복 제거와
grammar validation 뒤 lexicographic `(total request width, cancellation presence, cancellation index,
total inversion count)`를 반드시 낮춘다. source seed와 settlement 및 fixed two-wave topology는 바꾸지
않으며, source와 각 candidate를 exact canonical descriptor JSON으로 strict serialize/parse한 뒤에만
predicate에 전달한다.

local-minimum 주장은 deterministic replay predicate에 한정한다. 실제 V1 replayer가 panic하면 source와
minimized canonical descriptor 및 derived operation list를 test panic diagnostic에 함께 출력한다. 이
predicate는 inner replayer가 panic하는지만 보존하므로 panic site, payload, failure signature 또는 root
cause를 보존하지 않는다.

그 diagnostic에는 별도 versioned `riley.scheduler.routing-fuzz-receipt` v1 artifact contract가
추가됐다. CI가 `RILEY_ROUTING_FUZZ_RECEIPT_DIR`와
`RILEY_ROUTING_FUZZ_SOURCE_REVISION`을 함께 제공한 실제 V1 failure에만, pre-created absolute
run directory 아래 safe case ID와 source seed 기반 create-new leaf를 쓴다. 따라서 concurrent failure가 서로
overwrite하지 않으며 local passing test는 file을 만들지 않는다. write/sync가 실패하면 newly-created partial
leaf 제거를 시도하고, 제거 실패도 원래 replay failure를 가리지 않는 diagnostic으로 남긴다. receipt는 source/minimized canonical
descriptor 원문, 각각의 descriptor-derived `SchedulerConfig`, fixed symbolic KV layout
`(1, 64, 1, 8)`, fixed replay timeline `(0, 1, 2)`, operation spelling, source Git SHA와 explicit
`not_established` boundary를 bind한다. source/minimized seed·settlement drift 또는 rank 증가도 writer와
read-only Python checker가 거부한다.

`benchmarks/scripts/check_routing_fuzz_receipt.py`는 duplicate key, float/non-finite value, noncanonical
outer/nested JSON, unknown/missing/reordered field, symlink/non-regular/oversized file, nonzero SHA, descriptor,
config, layout, timeline, operation, scope drift를 fail closed한다. checker의 successful output은
**structurally valid diagnostic**일 뿐 scheduler failure 재실행, panic site/payload/signature/root cause,
general/global minimum, GPU evidence, C02 qualification 또는 C03-B acceptance를 establish하지 않는다.
전체 workspace test가 다른 이유로 실패해 receipt가 없으면 CI checker는 이를 별도 failure로 오인하지
않고, receipt가 있는 경우에만 엄격 검사·failure-only upload한다.

`bounded-mixed-program-v1`은 V1/V2와 별도 canonical descriptor다. 최대 네 logical label(각 label은
한 번만 submit)과 최대 세 live request에서 prompt 1, `max_new_tokens=1..=2`를 고정하고, raw
`submit`, settled-boundary `cancel`, `plan_commit(feedback_slot_order)`, final `close`만 허용한다.
최대 12 operations, 네 plan commit, 한 settled cancel이며, 최소 두 commit과 하나 이상의
decode+prefill mixed plan을 강제한다. label domain은 `1..=4`이므로 다섯 번째 logical request는
grammar 밖이다. fixed active/budget=3, prefill-chunk=1 configuration이 매 plan에서 모든 live request를
선택하게 하며 queue, timeout, aging override, partial prefill은 범위 밖으로 고정한다. descriptor의
feedback slot order는 해당 plan의 dense slot 전체 순열이다.

seeded sampler는 exhaustive enumerator가 아니다. `seed & 3`으로 singleton-first, two-prefill-first,
three-prefill-first prefix를 고정 분기한 뒤, 두 번째 plan에서 필요한 decode+prefill mixed shape를
만든다. 따라서 10,000 seed set은 첫 plan의 width 1/2/3을 모두 재생하지만, bounded grammar 안의 모든
raw operation vector를 열거한다는 뜻은 아니다.

이 slice의 pure `BoundedMixedProgramOracle`은 `Scheduler`나 `IterationPlan`을 import하지 않는다.
logical label의 submission order·history·terminal ledger만으로 decode/prefill projection과 semantic
slot/token feedback을 만들고, public adapter는 request ID, work kind, input, dense slot, total token을
별도로 assert한 뒤 iteration ID만 전달한다. 따라서 feedback mapping을 production plan에서 다시
추론하지 않는다. settled cancel은 즉시 `Cancelled` completion 하나를 받아야 하며, plan commit과 close는
token/index mapping, length/cancel terminal-once, final zero gauge를 oracle ledger와 비교한다. corpus는
3-slot reverse mixed feedback, first-plan two/three-prefill 뒤 mixed replay, unplanned prefill cancel 뒤
replacement submit, generated-history decoder cancel, two live decoder close를 canonical JSON으로 보존한다.

C03-Az는 이 bounded grammar 전용 stateful local reducer다. candidate order는 (1) settled cancel
제거, (2) logical submit 하나와 그 label의 dependent settled cancel 제거, (3) 각 non-identity
`feedback_slot_order`의 direct identity화, (4) 각 order의 좌→우 adjacent inversion swap이다. 첫 두
변환은 source/candidate descriptor 상태를 함께 재생해 submission order 안에서 decode 먼저, prefill
나중의 semantic label/slot projection을 다시 만들고, surviving label은 원래 feedback 순서를 보존하며
candidate-only live label은 candidate의 canonical semantic slot 순서로 뒤에 붙인다. 따라서 raw slot
번호를 단순 filter/reindex하지 않는다. 각 candidate는 grammar validation과 strict canonical codec
round-trip을 통과해야 하며 seed는 보존된다. rank는 `(submit_count, settled_cancel_presence,
total_feedback_inversions)`이고 모든 accepted candidate는 lexicographically 엄격히 작다.

이 local reducer는 `Close` 제거, label renumber, `PlanCommit` 삭제, `max_new_tokens` 축소, arbitrary
operation/delta debugging, cancellation timing 재배치나 general trace reduction을 하지 않는다. actual
inner replayer panic 뒤에만 panic-only predicate로 greedy replay하며, report는 source/minimized canonical
descriptor와 각 operation list를 남긴다. 이는 해당 fixed candidate order의 local minimum일 뿐 panic
site/payload/failure signature/root cause, general/global minimum 또는 GPU/C02 evidence를 보존·주장하지
않는다. V1 receipt/checker는 재사용하지 않는다.

`inflight-mixed-program-v1`은 위 settled-boundary grammar와도 별도 canonical descriptor다. 최대 네
logical label(각 label은 한 번만 submit), 세 live request, 16 operations, 네 `plan`, 한 deferred
`cancel`, 한 `abort_not_dispatched`를 상한으로 둔다. prompt 1과 `max_new_tokens=1..=2` 아래 raw
`submit`, `plan`, in-flight `cancel`, `complete(feedback_slot_order)`, `abort_not_dispatched`, final
`close`를 명시한다. `plan`은 idle에서만 열리고 `complete` 또는 abort가 정확히 한 pending plan을
settle한다. abort 다음 operation은 반드시 fresh `plan`이며 normal `close`는 pending plan 없이 final에만
허용한다. 최소 두 plan과 하나 이상의 decode+prefill mixed plan을 강제하고, fixed active/budget=3,
prefill-chunk=1가 매 plan의 모든 live request를 선택하게 한다.

seeded sampler는 first-plan width 1/2/3과 세 settlement branch(deferred-cancel complete,
cancel 없는 `NotDispatched` abort/retry, deferred-cancel abort/retry)를 고정 분기한다. sampler는
exhaustive enumerator가 아니며 arbitrary raw operation vector를 모두 생성한다고 주장하지 않는다.
`complete` feedback에는 deferred-cancel label의 dense slot도 반드시 포함하지만, 그 slot은 token event를
만들지 않고 prior history의 `Cancelled` completion 하나로 settle해야 한다. `NotDispatched` abort는
deferred label만 terminal로 만들고 나머지 request의 history/state를 rollback하여 fresh iteration ID의
retry plan으로 다시 검증한다. cancel 없는 abort/retry branch는 final prefill을 `max_new_tokens=2`로
남겨 retry 뒤 live decoder를 normal `close`가 history 보존 `Cancelled` completion으로 settle하는지도 확인한다.

pure `InflightMixedProgramOracle`은 `Scheduler`나 `IterationPlan`을 import하지 않는다. label의
submission order·history·live/deferred 상태·pending semantic slot·terminal ledger만 보유하며, adapter는
public plan의 request ID, work kind, input, dense slot, total token 및 single-inflight contract를 별도로
assert한 뒤 iteration ID만 oracle feedback builder에 전달한다. corpus는 deferred prefill cancel의 reverse
complete, generated-history decoder cancel의 abort/retry, cancel 없는 three-slot abort/retry를 canonical
JSON으로 보존한다. 각 case는 token/index, deferred cancellation priority, terminal-once, final zero gauge를
검증한다. 이 slice의 raw-pending-lifecycle reducer는 actual inner replayer panic 뒤에만 source/minimized
canonical descriptor와 각 operation list를 test panic diagnostic으로 남기며, receipt/checker를 재사용하지
않는다.

`inflight-mixed-program-v1` 자체는 `DeviceQuiescedMutationUnknown`, pending close disposition, stale/missing/unplanned feedback,
settled-boundary cancellation, partial prefill, queue/aging/timeout, multiple deferred cancel/abort,
C03-Ax boundary seam, general reducer/receipt 또는 GPU evidence를 구현·주장하지 않는다.

`inflight-mixed-program-v2`는 V1 acceptance를 넓히지 않는 별도 strict `trace_kind`다. V1과 같은
normal complete 및 `NotDispatched`/fresh-`Plan` retry branch를 보존하면서, prior decoder history가
있는 decode+prefill mixed plan 뒤의 no-cancel·deferred-cancel terminal
`DeviceQuiescedMutationUnknown` abort와 즉시 `Close`를 추가한다. adapter는 caller-supplied
disposition을 public scheduler API에 전달한 뒤 pending request가 token 없이 `ExecutorFailure`로
terminal-once 처리되고 ownership이 quiescent한지만 확인한다. 이는 caller가 실제로 CUDA stream을
synchronize했는지, device mutation 결과가 같은지, GPU에서 replay되는지 또는 성능이 향상됐는지를
검증·주장하지 않는다.

이는 여전히 C03-A의 부분 범위다. arbitrary/unbounded mixed-operation generator와 그 전체 grammar의
shrink/global-minimum counterexample, admission/aging/KV까지 포함하는 general reference scheduler,
generalized multi-event fault-injection grammar는 남아 있다.
failure-signature/same-assertion preservation, multi-edit/delta-debugging reduction도 별도 범위다.
`bounded-mixed-program-v1`도 Plan/Complete
분리, in-flight deferred cancel, abort/retry, stale/missing/unplanned feedback, `DeviceQuiescedMutationUnknown`,
queue/aging/timeout, partial prefill 또는 injection seam을 구현·주장하지 않는다. V1/V2 shrinker는 각각
작은 grammar의 deterministic greedy local minimum일 뿐 일반 trace shrinker나 globally minimal
counterexample을 주장하지 않는다. C03-Ax는 bounded-mixed-program-v1의 grammar를 확장하지 않는
별도 unit-only containment contract다.

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

`GeneralMixedOperationTraceV1` corpus는
`crates/riley-scheduler/tests/corpus/output-routing/general-mixed-operation-v1/*.json`에
2x2 reverse commit, 3x1 permuted cancel commit, 3x3 permuted abort를 canonical descriptor로
등록한다. fixture와 seeded trace는 strict decode, exact canonical round-trip, descriptor-only pure
oracle, public plan projection assertion, host-side replay 및 close quiescence를 모두 통과해야 한다.
이는 V2 codec 확장이나 arbitrary/general trace format의 주장도 아니다.
이는 GPU fixture나 C02/C03-B evidence가 아니다. `IterationResult::new` 단계에서 막히는 duplicate
slot이나 immutable public plan으로 만들 수 없는 stale block-table version은 scheduler boundary property라고
주장하지 않는다.

`InflightMixedProgramTraceV1` corpus는
`crates/riley-scheduler/tests/corpus/output-routing/inflight-mixed-program-v1/*.json`에 deferred
prefill cancel의 reverse complete, deferred decoder cancel의 `NotDispatched` abort/retry, cancel 없는
three-slot abort/retry와 retry 뒤 live decoder close를 committed canonical descriptor로 등록한다. fixture와 seeded trace는 strict
decode/exact canonical round-trip, descriptor-only pure oracle, public plan projection, one outstanding
plan rejection, host-side replay와 close quiescence를 모두 통과해야 한다. 이는 fixed V2 selector grammar의
확장이나 device-executed abort/GPU evidence가 아니다.

`InflightMixedProgramTraceV2` corpus는
`crates/riley-scheduler/tests/corpus/output-routing/inflight-mixed-program-v2/*.json`에 prior decoder
history를 가진 deferred-cancel case와 no-cancel sibling의 decode+prefill mixed plan,
caller-asserted terminal quiesced abort 및 immediate close를 strict canonical descriptor로 보존한다.
V2 fixture와 seeded trace는 strict decode/exact canonical round-trip, descriptor-only oracle, public
host-side abort disposition replay, `ExecutorFailure` terminal-once 및 close quiescence를 통과해야
한다. 이 fixture는 V1 grammar를 확장하지 않으며 actual CUDA synchronization, device mutation parity,
GPU replay 또는 performance evidence가 아니다.

### C03-B — GPU integration slice

CPU model이 생성한 대표 trace 중 다음만 실제 CUDA path로 replay한다.

- `C=5`, `C=8` mixed output
- KV boundary
- GPU greedy output
- commit failure와 cancellation

GPU test는 property runner 전체를 실행하지 않고 고정된 최소 corpus만 검증한다.

첫 source slice는 `gpu-fixed-v1` canonical JSON 세 건을
`crates/riley-scheduler/tests/corpus/output-routing/gpu-fixed-v1/`에 고정한다. pure parser는
case ID별 body drift, unknown/reordered/noncanonical JSON, label alias를 거부하고 scheduler/CUDA를
열지 않는다. `c5-kv15-to17-mixed-greedy`는 decoder 두 개의 prompt 15 / `max_new=3`과 final
prefill 세 개를 사용해 prime `[15]` → mixed decode `[16]` → final decode `[16,1]` KV table을
고정한다. `c8-mixed-deferred-cancel`은 decoder 세 개와 final prefill 다섯 개의 dense 8-slot
mixed plan에서 device download 뒤 deferred cancel을 고정한다. 별도
`c8-mixed-greedy-commit-assembly-failure`는 같은 8-slot device-greedy download 후 public
`into_result` sample-count mismatch가 반환한 `DeviceQuiescedMutationUnknown`만으로 abort하는
fixture다. 이는 private reservation-commit failure injection을 주장하지 않는다.

`gpu_fixed_corpus_contract` CPU test는 같은 descriptor를 public scheduler와 synthetic result로
replay해 plan kind/slot/request/target length/block-table valid-token layout, deferred cancel,
terminal-once, abort quiescence를 확인한다. CUDA target `c03_gpu_fixed_corpus`는 source archive에
포함되지만 `#[ignore]` 상태다. future C02-qualified candidate와 별도 GPU execution approval 뒤에만
다음처럼 수동 실행한다.

```bash
cargo test --locked -p riley-scheduler --features cuda \
  --test c03_gpu_fixed_corpus -- --ignored --test-threads=1
```

이 source/CPU contract 통과는 GPU execution, candidate binding, C02 pass, C03-B acceptance 또는
vLLM 성능 우위를 뜻하지 않는다.

## 7. Shrinking

현재 `OperationTraceV2`는 failure가 발생하면 같은 property predicate로 bounded candidate를 다시
replay해 optional operation 제거와 output/prefill length 축소 순서의 local minimum을 출력한다. seed와
rejection kind·settlement은 고정하며, exact canonical descriptor JSON과 derived operation list 전체가
shrunken replay 입력이다. 이 minimizer 자체는 deterministic synthetic predicate와 모든 bounded candidate의 grammar
replay로 candidate order·local-minimum 성질을 고정한다. 실제 defect counterexample이나 arbitrary trace를
이 경로가 이미 global minimum으로 축약했다는 뜻은 아니다.

`general-mixed-operation-v1`은 fixed two-wave grammar 전용 selector-aware local reducer를 사용한다.
request 제거는 remaining request/slot selector를 compactly rebase하고, cancellation target과
permutation을 다시 validate한다. reducer는 cancellation → request removal → direct identity
permutation → adjacent inversion swap의 고정 순서로, strict codec round-trip 뒤 deterministic
panic-only predicate가 재현되는 첫 candidate를 greedy하게 선택한다. 따라서 출력은 이 candidate
order의 local minimum일 뿐 general shrinker, global minimum, panic site/payload/signature/root-cause
preservation이 아니다. failure report는 사람이 읽는 test panic diagnostic을 유지하면서, CI opt-in
directory가 제공된 경우 source/minimized descriptor·각 replay config·KV layout·timeline·Git SHA를
create-new diagnostic receipt에도 쓴다. receipt checker는 구조적 binding만 확인하고 failure predicate,
panic site/payload/signature/root cause 또는 global minimum을 재검증하지 않는다.

`bounded-mixed-program-v1`은 C03-Az의 fixed stateful local reducer를 사용한다. settled cancel/submit
removal 뒤의 slot projection은 위 semantic rebase로 한정하고, plan deletion·output-capacity 축소·label
renumber·arbitrary multi-edit/delta debugging은 후속 별도 contract다. source와 minimized descriptor는
strict codec을 거쳐 panic-only predicate에만 전달되며, report는 두 canonical descriptor와 operation list를
출력한다. 따라서 출력은 fixed candidate order의 local minimum일 뿐 general/global minimum, failure
signature 또는 root cause preservation을 주장하지 않는다.

`inflight-mixed-program-v1`은 fixed raw-pending-lifecycle local reducer를 사용한다. candidate order는
(1) deferred `Cancel` 하나 제거, (2) 인접한 `AbortNotDispatched`와 그 mandatory fresh `Plan` 제거,
(3) 각 `Submit.max_new_tokens`의 `2 → 1` 축소, (4) 각 non-identity `Complete` feedback order의 direct
identity화, (5) 각 order의 좌→우 adjacent inversion swap이다. raw label과 feedback slot은 그대로 두며,
특히 abort/retry contraction은 original pending `Plan`을 그대로 남긴다. 따라서 cancel/abort 뒤의
later pending-plan 또는 complete arity가 달라진 candidate는 기존 grammar validation과 strict canonical
codec round-trip에서 거절된다. accepted candidate는 seed와 final `Close`를 보존하고
`(operation_count, deferred_cancel_count, total_max_new_tokens, total_feedback_inversions)` rank를
lexicographically 엄격히 낮춘다.

actual inner replayer panic 뒤에만 panic-only predicate로 이 fixed candidate order를 greedy replay하고,
report는 source/minimized canonical descriptor와 각 operation list를 남긴다. 이는 raw vector local
minimum일 뿐 panic site/payload/failure signature/root cause, label/slot rebase, arbitrary operation
deletion, `Plan/Complete` block deletion, deferred-cancel target/timing 변경, general/global minimum,
receipt/checker, GPU 또는 C02 evidence를 보존·주장하지 않는다.

`inflight-mixed-program-v2` reducer는 V1의 `AbortNotDispatched`/fresh-`Plan` retry contraction을
terminal quiesced branch에 적용하지 않는다. V2 candidate는 trace kind와 terminal abort·immediate close
contract를 유지한 strict codec round-trip 뒤에만 replay하며, 이 local reduction도 caller-provided host
disposition의 CPU semantics만 다룬다. CUDA stream synchronization, device mutation parity, GPU replay,
performance 또는 C02 evidence는 보존·주장하지 않는다.

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

## 8. 파일 변경

```text
crates/riley-scheduler/src/execution.rs
crates/riley-scheduler/src/scheduler.rs
crates/riley-scheduler/tests/general_mixed_operation_routing.rs
crates/riley-scheduler/tests/support/general_mixed_operation_trace.rs
crates/riley-scheduler/tests/support/routing_fuzz_receipt.rs
crates/riley-scheduler/tests/bounded_mixed_program_routing.rs
crates/riley-scheduler/tests/support/bounded_mixed_program_trace.rs
crates/riley-scheduler/tests/corpus/output-routing/bounded-mixed-program-v1/*.json
crates/riley-scheduler/tests/inflight_mixed_program_routing.rs
crates/riley-scheduler/tests/support/inflight_mixed_program_trace.rs
crates/riley-scheduler/tests/corpus/output-routing/inflight-mixed-program-v1/*.json
crates/riley-scheduler/tests/corpus/output-routing/inflight-mixed-program-v2/*.json
crates/riley-scheduler/tests/support/routing_fuzz_rotation.rs
benchmarks/scripts/check_routing_fuzz_receipt.py
benchmarks/scripts/tests/test_check_routing_fuzz_receipt.py
.github/workflows/production-cpu.yml
.github/workflows/c03-routing-fuzz.yml
```

새 dependency를 추가할 경우 production dependency graph에 포함되지 않는 dev-dependency여야 하고 `Cargo.lock`을 고정한다.

현재 V1 reducer/receipt와 bounded raw-program/in-flight raw-program slice(C03-Az stateful reducer와 V1/V2 in-flight raw-lifecycle reducer 포함)는 기존 test-only `serde_json` dev-dependency만
사용하며 production dependency graph, scheduler runtime semantic, GPU, C02 qualification을 변경하지
않는다. C03-Ax도 동일하게 private test-only source만 바꾼다. 이후 arbitrary general grammar,
durable cross-version replay contract, failure corpus 자동 등록, delta debugging, arbitrary/multi-event
fault-injection grammar는 별도 PR 범위다.

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

- 기본 CPU workflow: 각 property lane의 deterministic 10,000-trace seed set(총 80,000 trace)
- scheduled workflow: fresh recorded base와 15개 disjoint slot으로 여덟 10,000-trace lane을
  회전해 총 1,200,000 CPU trace; 네 C03 CPU target만 실행하고 GPU/CUDA target은 제외
- GPU manual/scheduled: 고정 corpus
- C03-B source contract: `gpu_fixed_corpus_contract`는 기본 CPU workflow에서 실행하며, ignored
  `c03_gpu_fixed_corpus`는 C02 actual qualification과 explicit GPU operation approval 뒤에만
  candidate-bound environment에서 실행
- V1 failure: source/minimized descriptor, operation list, 각각의 scheduler config, fixed KV/timeline,
  source Git SHA를 diagnostic-only receipt로 create-new 기록
- bounded raw-program failure: C03-Az reducer가 source/minimized canonical descriptor와 각 operation
  list를 test panic diagnostic으로 남기며, V1 전용 receipt/checker를 재사용하지 않음
- in-flight raw-program failure: local reducer가 source/minimized canonical descriptor와 각 operation
  list를 test panic diagnostic으로 남기며 receipt/checker를 재사용하지 않음
- CI: unit-test step의 failure 때 receipt가 존재하면 strict checker를 실행하고 run/attempt-scoped
  temp directory를 14일 artifact로 upload; receipt가 없는 unrelated test failure는 그대로 보존

flaky retry로 통과시키지 않는다. 동일 seed가 재현되지 않으면 test harness defect로 별도 실패한다.

## 12. 승인 기준

- C03-A: fixed/random CPU seed에서 invariant 위반 0, deterministic corpus와 shrink/replay test 통과
- RC1 최소 재현 fixture가 수정 전 실패/현재 통과하는 contract test로 보존
- shrink된 counterexample serialization/replay test 통과
- V1 selector-local receipt/checker의 exact canonical descriptor/config/KV/timeline/SHA binding 및
  create-new/no-overwrite/hostile-input rejection test 통과; 구조적 checker pass는 diagnostic-only다.
- bounded raw-program V1의 strict canonical codec, 3-slot feedback permutation 전수, committed corpus,
  10,000 seeded replay, settled cancel/terminal-once/final quiescence test 통과
- C03-Az bounded local reducer의 stateful cancel/submit slot rebase, candidate dedupe/rank/strict-codec
  replay, deterministic synthetic local-minimum/idempotence, source/minimized failure-report binding test 통과
- in-flight raw-program V1의 strict canonical codec, 3-slot feedback permutation 전수, committed corpus,
  10,000 seeded replay, deferred cancel complete/abort-retry/terminal-once/final quiescence test 통과
- in-flight raw-program V2의 V1과 분리된 strict canonical codec, committed corpus, 10,000 seeded
  replay(두 caller-asserted quiesced terminal-abort branch 포함), deferred cancel을 포함한
  `ExecutorFailure` terminal-once/final quiescence test 통과. 이는 CUDA synchronization, device mutation
  parity, GPU replay 또는 performance evidence가 아님
- in-flight raw-lifecycle V1 reducer의 valid cancel/abort-retry/capacity/permutation contraction,
  candidate dedupe/rank/strict-codec replay, deterministic synthetic local-minimum/idempotence,
  source/minimized failure-report binding test 통과
- in-flight raw-lifecycle V2 reducer의 terminal quiesced abort·immediate-close preservation 및
  V1 `AbortNotDispatched`/retry contraction 미적용 test 통과; caller-asserted host disposition 밖의
  CUDA/GPU semantics는 검증하지 않음
- C03-Ax test-only seam이 valid sample validation 뒤 owning abort-safe failure를 돌려주고, 첫
  reservation commit 뒤 두 번째 forced failure에서 token 미발행, terminal-once, KV reclaim,
  completed 0/aborted 1 및 close ownership quiescence를 통과
- C03-Ay는 all-unset PR environment에서 기존 seed/action window를 유지하고, scheduled all-set
  base/slot/slot-count의 malformed·partial·overflow 입력을 fail closed하며, 15 slot이 총
  1,200,000 CPU trace를 실행
- production runtime 코드의 semantic change 없음
- CPU test runtime이 일반 PR CI budget 내
- C03-B source: strict `gpu-fixed-v1` corpus parse와 CPU topology/KV/deferred-cancel/abort contract 통과
- C03-B: C02 actual qualification 뒤 GPU corpus에서 output token/request mapping exact

## 13. 롤백

테스트-only 변경이므로 rollback은 신규 test/support/corpus를 함께 revert한다. 다만 실제 defect를 재현하는 corpus는 원칙적으로 삭제하지 않고, harness 교체 시 새 형식으로 이관한다.

## 14. 완료 정의

C03-A의 formal CPU completion은 scheduler event 순서와 output slot permutation을 deterministic
seeded CPU trace로 생성해 reference model과 production state가 일치하고, mixed-operation
generator·shrink·counterexample corpus/replay·failure seed와 최소 trace까지 갖출 때다. 현재
valid/fault microtrace, two-wave, bounded raw-program 및 C03-Ax의 두 deterministic post-validation
seam은 이를 향한 partial coverage이며 아직 C03-A 완료 선언 근거가 아니다.
C03의 formal completion은 C03-B가 C02 actual qualification 뒤 GPU corpus에서도 exact mapping을
확인할 때만 선언한다.
