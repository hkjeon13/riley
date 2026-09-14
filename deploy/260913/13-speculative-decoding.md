# PR 13 — Draft-target speculative decoding

상태: **Rust greedy 승인·KV 정산 기반 구현, target-prefix 수치 전제 검증 완료 / 실제 speculative serving 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

큰 target에서 검증 한 번으로 여러 token을 진행해 순차 decode 횟수를 줄인다.

## 의존성과 변경 위치

선행: 01, 03; draft/target 모델 로딩 지원을 착수 전에 확인.

예상 수정 위치: runtime draft/target executor, KV transaction, scheduler token accounting. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 지원되는 draft-target 조합 하나와 최대 draft 길이를 고정한다.
2. batched verification과 greedy acceptance를 구현한다.
3. 승인 prefix commit·거절 suffix KV rollback을 연결한다.
4. acceptance 비용 기록 및 일반 decode fallback을 추가한다.

## 범위 경계

EAGLE 학습 pipeline, 임의 모델 지원, stochastic sampling까지 한 PR에 넣지 않는다. greedy-only 범위를 API에 명시하고 unsupported sampling은 fallback한다.

## Correctness·수명 계약

greedy output은 target 단독과 같아야 한다. EOS·length limit 뒤 token을 승인하지 않는다. 향후 sampling은 target 분포 보존 검증이 별도로 필요하다.

## 검증과 하드웨어 skip

accept all/none/partial, page 경계 rollback, EOS, cancellation, 혼합 batch. 작은 135M뿐 아니라 지원 가능한 큰 target을 별도 축으로 평가. draft 모델 미지원은 하드웨어 skip으로 위장하지 않는다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

draft·verification·rollback을 포함한 serving TPOT/throughput 및 acceptance length를 보고한다. 낮은 concurrency 이득과 높은 concurrency 손실을 분리해 적용 범위를 정한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

draft 경로를 끄고 target 단독으로 복귀한다.

## 연구 근거

[EAGLE-3](https://arxiv.org/abs/2503.01840), [Speculation limits](https://arxiv.org/abs/2601.11580). 논문 성능 배수는 Riley의 예상 개선율이 아니다.


## 첫 구현 batch — 2026-09-14

[기반 구현과 GPU 증거](../../benchmarks/results/20260914-speculative-foundation/README.md): 최대8개 draft의 greedy prefix 승인, replacement/bonus, EOS·길이·취소 처리, 기존 KV reservation의 완료된 prefix 정산을 연결했다. 추가 모델 없는 prompt lookup을 최초 제안기로 구현했으며, 지원되는 큰 target/draft 모델 쌍 도입을 대체 완료하지 않는다. Python은 serving 경로에 들어가지 않는다.

CPU5개 테스트는 KV252조합 및 독립 직렬 oracle11,250조합을 포함해 통과했다. 기존 accepted target의 순차9출력과72개 독립 prefix 요청에서 BF16 logits7,077,888 bytes가 일치했고 full-model memcheck0이다. 이것은 기존 경로의 수치 전제 검사이며, 한 번의 fused verification이나 serving 이득이 아니다.

다음 묶음은 K+1 target prediction 반환, private KV append의 GPU 완료와 승인 prefix 정산, scheduler 출력·길이·취소 accounting 연결, 실제 acceptance·draft·verify·rollback 비용 및 vLLM serving 비교다. 검증된 target 수치 계약을 그대로 사용하며, FP32 분할 attention의 strict 실패를 우회하지 않는다. 미지원 sampling/stop-string/processor 경로는 ordinary decode를 유지한다.

## 다중 위치 target head — 2026-09-14

[GPU 검증 결과](../../benchmarks/results/20260914-verification-head/README.md): 한 packed 모델 실행의 여러 hidden 위치를 기존32행 head에 모으고, 별도 pinned 결과와 Rust completion identity로 읽는 실험 경로를 연결했다. 최대4owners×8positions이며 pending input을 포함하므로 첫 GPU 경로의 draft 상한은7이다. 기존 응답·정확한 target 수치 경로를 유지하며 buffered/compact 경로는 지원하지 않는다.

순차 target과72행 BF16 logits7,077,888 bytes가 일치했고 실제 모델 memcheck 및 bounded selector memcheck/racecheck가 오류0이다. 완료 전·정산 후 읽기 거부도 검사했다. 초기 prompt까지8-token chunk로 처리하는 검증 fixture이므로 총 모델 실행 수는 양쪽10회이며 성능 향상을 주장하지 않는다.

다음 구현 묶음은 GPU argmax의 작은 검증 결과, private speculative append와 greedy 승인/rollback, scheduler 다중 토큰 accounting을 연결하는 것이다. 아직 실제 speculative serving이나 vLLM 비교 완료가 아니다.

### GPU argmax verification output — 2026-09-14

The experimental head can now return 512 bytes of slot-validated GPU argmax
records instead of the 3,145,728-byte auxiliary full-logit buffer. The normal
output transfer remains unchanged. Exact serial target token agreement was
verified at all 72 fixture positions; native memcheck/racecheck and full-model
memcheck passed. SM90a/SM100a compile only; runtime tests skipped without hardware.
Evidence: `benchmarks/results/20260914-verification-greedy/README.md`.

This completes the compact auxiliary result primitive, not speculative serving.
Next integrate private KV append, completed-prefix settlement, cancellation and
multi-token scheduler accounting, with a 7-token draft cap for this 8-input head.
Then run strict serial equivalence and a matched serving milestone against the
current baseline and vLLM before selecting any default policy.

### Host scheduler batch — 2026-09-14

Implemented bounded prompt-lookup plans, borrowed execution authority, private
append reservations, and multi-token completed-prefix settlement with cancellation,
EOS, length, rejection and whole-batch commit-failure containment. This is an
experimental host path; ordinary serving does not select it. Shared-prefix cache
and the GPU verification adapter remain unsupported. Existing progress validation
must gain an explicit verification contract before generated-token chunks can run;
do not disguise speculative inputs as a longer original prompt.

Evidence and limitations: `benchmarks/results/20260914-speculative-scheduler/README.md`.
Next milestone combines GPU binding, exact serial equivalence and serving/vLLM
comparison. Host metadata tests alone are not a throughput result.

### Explicit GPU stage and generation gate — 2026-09-14

Connected host speculation to a native-capability-gated verification stage,
preserving actual prompt/generated/KV progress and checking returned completion
identities. Final 12-request gate: all 384 tokens match serial decode; 144 draft
tokens are accepted, with full-model memcheck clean. A failed zero-acceptance
control revealed first-four-candidate blocking; the fixed selection scans for
eligible owners. Failure evidence is retained.

This is not a serving promotion: model calls rise from 36 to 63 because only four
owners verify at once, and ordinary completion still transfers full logits.
Evidence: `benchmarks/results/20260914-speculative-gpu-stage/README.md`.

The next optimization batch must address the structural constraints together:

1. Widen the query/head mapping to cover the active owner batch, with bounded
   query-count buckets, explicit capacity/ownership checks and unchanged greedy
   numerical policy. Do not silently narrow C32 serving to four active requests.
2. Return compact normal completion plus compact verification tokens; avoid the
   full normal-logit copy and retain an explicit diagnostic full mode.
3. Extend immutable shared-prefix read ownership to verification. Require COW
   for shared writable tails and test off-batch readers and cancellation.
4. Add opt-in greedy serving selection, serial fallback and acceptance/work
   counters. Compare same-model/hardware/workload serving against current Riley
   and vLLM at low and high concurrency, including P95/P99 and stop/cancel cases.

Do not use artificial repetition controls as performance evidence, relax the
strict token gate, or tune draft lengths to compensate for the four-owner cap.

### Wide verification and compact completion — 2026-09-14

Implemented the first two structural changes above: up to 32 owners / 256 query
positions, whole selected decode batch participation, separate M256 verification
capture and 8,192-byte total completion transfer. Ordinary M32 decode stays intact.
The same 12-request gate drops from 63 narrow calls to 33 wide calls (36 serial),
with 384 exact tokens and 144 accepted drafts. C32 gate: 1,024 exact tokens,
581 accepted drafts, 33 wide calls versus 44 serial. Both gates pass memcheck.
Evidence: `benchmarks/results/20260914-speculative-wide/README.md`.

This completes two primitives, not the serving milestone. Repetition controls
are not performance evidence. Fixed M256 costs, shared-prefix ownership/COW,
serving selection, fallback and matched vLLM measurements remain to be resolved.
Do not disable prefix caching to present an unmatched serving comparison as
qualification. No runtime Python or default policy promotion is introduced.

### Immutable shared-prefix verification — 2026-09-14

Connected wide verification to retained immutable prefix ownership. Host checks
include off-batch cache/read leases; native checks enforce logical-position
agreement and no shared writes. Existing full-page cache leaves private tails;
append into a genuinely shared writable page still requires COW and is rejected.
C32 warmed-cache generation: 1,024 exact tokens, 629 accepted drafts, 39 serial
calls versus 33 wide calls; each lane records 52 cache hits / 6,640 reused tokens.
Normal and full-model memcheck pass. This remains a correctness gate.
Evidence: `benchmarks/results/20260914-speculative-shared-wide/README.md`.

The remaining serving integration must stage detokenized output for every accepted
position, truncate KV/output together on stop string or stop token, preserve
cancellation and commit-before-publication, and expose acceptance/work counters.
Do not publish multiple scheduler tokens through the current single-token staging
path unchanged. Compare actual serving after that integration; no new throughput
or vLLM improvement is established by the call counts above.

### Serving integration and measured regression — 2026-09-14

Connected opt-in multi-token staging, stop-token/string truncation, prefix KV
settlement and commit-before-publication, with ordinary request-processor fallback
and counters. Scheduler 72 / runtime policy 7 / server 69 CPU tests pass; CUDA
release builds. Matched C32 serving completes 16 lanes, 9,216 total warmup/retained
requests, exact Riley references and stop/cancel/recovery (96 each).

**Performance gate fails:** speculative throughput versus frozen best prior is
−38.73% shared / −9.00% unique, and versus same-binary unbuffered control is
−28.56% / −1.25%. Default remains disabled. vLLM 0.27.1 comparison and tails are
recorded in `benchmarks/results/20260914-speculative-serving-c32/README.md`.

Four bounded Nsight traces locate 56–60% of verification kernel time in mapped
attention; M256 head GEMM is only approximately 2.5%. Prioritize short multi-query
attention and KV reuse, including one-input owners, with strict token gates and
matched serving remeasurement. Do not treat model-call reduction as speedup or
start head bucketing before addressing the measured attention cost.

### Short-query attention and bounded graph — 2026-09-14

Implemented one CTA per verification owner/head, ordered reuse for 2–8 queries,
unchanged single-query math, and a 256-input verification-only launch bound.
The intermediate attention change improves shared serving throughput 8.39%
versus old speculative; unique is unchanged (−0.05%). Bounded graph follow-up
adds only 1.04% shared / −0.13% unique. Both screens retain exact Riley outputs,
stop/cancel/recovery, native and model sanitizer gates.

Stage-1 Nsight mean verification attention time drops 2.531→1.723ms shared and
2.620→1.885ms unique; graph mixes vary, so these are diagnostic, not serving
speedup claims. Final throughput is 7,018.8 / 3,835.7 tok/s (shared / unique),
below prior 10,341.9 / 4,129.2 and vLLM 0.27.1 12,218.5 / 4,947.2.
Speculative remains disabled by default; performance qualification still fails.

Evidence and full latency tables:
- `benchmarks/results/20260914-verification-attention-serving-c32/README.md`
- `benchmarks/results/20260914-verification-capacity-serving-c32/README.md`

Stop micro-tuning this speculative path after the measured small bound benefit.
Next candidate is ordinary cached-prefix short-query reuse (2–16 inputs), with
split-FFN and rolling decode retained; test exact arithmetic and real serving
before promotion. Hopper/Blackwell compile gates pass, execution and multi-GPU
remain untested. No runtime Python is introduced.
