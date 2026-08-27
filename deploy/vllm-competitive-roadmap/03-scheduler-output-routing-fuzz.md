# C03 — Scheduler Output Routing Property Fuzz

**상태:** Planned  
**의미 등급:** `reference`  
**한 가지 목적:** scheduler plan부터 sampling/commit/terminal event까지 request-token 대응 관계를 model-based property test로 고정한다.

[이전: C02](02-rc3-candidate-qualification.md) | [목차](README.md) | [다음: C04](04-llama-executor-refactor.md)

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

### CPU fast property

- PR마다 최소 10,000 generated traces
- nightly 또는 scheduled run에서 1,000,000 traces
- failure 시 seed와 최소 축약 trace 출력

### Deterministic corpus

과거 defect와 발견된 모든 fuzz counterexample을 JSON 또는 Rust fixture로 영구 등록한다.

### GPU integration slice

CPU model이 생성한 대표 trace 중 다음만 실제 CUDA path로 replay한다.

- `C=5`, `C=8` mixed output
- KV boundary
- GPU greedy output
- commit failure와 cancellation

GPU test는 property runner 전체를 실행하지 않고 고정된 최소 corpus만 검증한다.

## 7. Shrinking

실패 시 다음 순서로 trace를 축약한다.

1. request 수 감소
2. iteration 수 감소
3. prompt/output 길이 감소
4. operation 제거
5. output slot permutation 단순화
6. cancellation/failure 시점 단순화

최종 report는 재현 가능한 seed뿐 아니라 사람이 읽을 수 있는 최소 operation list를 포함한다.

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

- 고정 seed와 random seed에서 invariant 위반 0
- RC1 최소 재현 fixture가 수정 전 실패/현재 통과하는 contract test로 보존
- shrink된 counterexample serialization/replay test 통과
- production runtime 코드의 semantic change 없음
- CPU test runtime이 일반 PR CI budget 내
- GPU corpus에서 output token/request mapping exact

## 13. 롤백

테스트-only 변경이므로 rollback은 신규 test/support/corpus를 함께 revert한다. 다만 실제 defect를 재현하는 corpus는 원칙적으로 삭제하지 않고, harness 교체 시 새 형식으로 이관한다.

## 14. 완료 정의

scheduler event 순서와 output slot permutation을 임의로 생성해도 reference model과 production state가 항상 일치하고, 실패 시 최소 재현 trace가 자동 생성될 때 완료다.
