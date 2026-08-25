# PR 13 — Rust Scheduler와 Continuous Batching

**상태:** Implemented

**선행 조건:** [PR 12](12-qwen-compatibility.md)  
**다음:** [PR 14 — API와 Streaming](14-api-and-streaming.md)

**검증된 실행 소스:** `a303a23633baf465f625154b688cafc4009801b6`

[← 이전](12-qwen-compatibility.md) | [목차](README.md) | [다음 →](14-api-and-streaming.md)

## 목적

여러 요청의 prefill/decode 작업을 한 GPU iteration으로 구성하는 Rust-native scheduler를 구현한다.

## 구현 결과

### Bounded scheduler와 소유권

- FCFS admission, decode-first iteration selection, starvation 방지용 bounded aging,
  iteration token budget과 prefill chunk 상한을 구현했다.
- waiting request/prompt token, active sequence, request 최대 길이, promised KV block,
  metric sample window와 terminal tombstone은 모두 설정된 상한을 가진다. Caller의
  과도한 `Vec` capacity는 exact-length scheduler 소유 buffer로 복사한다.
- 모든 request 상태 전이와 paged-KV reserve/commit/rollback/free는 `Scheduler`가
  단독 소유한다. Cancellation, timeout, failure, shutdown과 close가 request와 block을
  정확히 한 번 정산하도록 completion outbox와 explicit close disposition을 추가했다.
- Queue wait와 scheduler iteration CPU time의 rolling p95, active/waiting gauge,
  batch/token/prefill/decode/KV/rejection/cancellation/GPU idle gap metric을 bounded
  nearest-rank window로 수집한다. Metric 기록 실패는 inference ownership을 깨지 않고
  degradation flag로 노출한다.

### Immutable plan과 runtime adapter

- Scheduler는 GPU pointer를 소유하지 않는 immutable `IterationPlan`을 만든다.
  Prefill/decode work item, versioned block table, dense output slot과 token/position
  metadata를 포함한다.
- Malformed CSR, duplicate request/slot/table, block-table count mismatch, cross-request
  physical block alias, non-dense output, decode output 누락을 dispatch 전에 거절한다.
- `IterationPlan`을 allocation 없이 borrowed `LlamaBatchRow`로 변환하고 executor의
  vocabulary, token, position, fixed-M과 output buffer bound를 다시 확인한다.
- BF16 logits를 dense slot으로 download하고 caller sampling 결과를 allocation-free
  `IterationResult`로 조립한다. Scheduler commit이 실패해도 safe abort ownership과
  downloaded result를 함께 보존한다.
- Executor 호출에 진입한 뒤 발생한 오류는 stream synchronize 후 보수적으로
  `DeviceQuiescedMutationUnknown`으로 분류한다. Synchronize 실패 시 host가 안전한
  rollback을 추정하지 않도록 abort data를 제공하지 않는다.

### Fixed-M continuous-batch CUDA path

- 한 번 준비한 fixed-M workspace, layer-sliced paged KV arena와 host metadata buffer를
  iteration마다 재사용한다. Hot `pack`/`execute` path에는 allocation이나 request별
  serial forward dispatch가 없다.
- indexed BF16 RoPE, row gather, packed CSR metadata, ragged paged-KV scatter와 causal
  ragged D64 GQA attention CUDA primitive를 추가했다. Output은 명시적인 dense slot
  순서로 gather한다.
- Absolute position, block boundary 15/16/17, shuffled physical block, permuted output
  slot, mixed prefill/decode와 zero tail을 reference와 비교했다.

## 의미 보존 등급

`E0`이다. 동일한 dense Llama/Qwen 연산을 ragged row와 paged KV 주소로 재배치하며,
sampling distribution이나 모델 의미를 바꾸는 근사는 없다. 수치 기준은 결과에 맞춰
완화하지 않고 PR 01 E0 v2의 고정 final-logit bound를 그대로 재사용했다.

| Metric | 고정 bound | SmolLM2 worst | Qwen2.5 worst |
|---|---:|---:|---:|
| cosine | `>= 0.9979035305495393` | `0.998494007581` | `0.999260337367` |
| max absolute error | `<= 5.852936458587647` | `0.531250000` | `0.546875000` |
| mean absolute error | `<= 1.151280319263363` | `0.273596887` | `0.094341452` |
| 32-step greedy top-1 mismatch | `0` | `0` | `0` |

초기 Qwen batch 진단에서 step 1 cosine과 step 14 top-1이 실패했다. Ragged attention이
기존 reference contract의 `BF16(QK) → BF16(scale)` staging과 reciprocal multiply를
누락한 것이 원인이었다. Kernel과 CPU oracle을 established BF16 staging으로 수정했고,
tolerance는 변경하지 않았다. 이후 SmolLM2와 Qwen이 모두 위 gate를 통과했다.

## 구현된 첫 정책

처음에는 단순하고 설명 가능한 정책을 사용한다.

- FCFS
- starvation 방지용 waiting-time aging
- iteration별 token budget
- prefill chunk 상한
- decode request 우선 또는 명시적 비율
- KV block 부족 시 admission 거절

cache-aware priority, speculative decoding, offload는 제외한다.

## 상태 모델

```rust
Waiting
Admitted
Prefilling
Decoding
Finished
Cancelled
Failed
```

모든 전이는 한 곳에서 검증한다. request가 어떤 실패 경로에서도 두 번 free되지 않아야 한다.

## Batch plan

Scheduler는 GPU tensor를 직접 계산하지 않고 immutable한 실행 계획을 만든다.

```rust
IterationPlan {
  prefill_items,
  decode_items,
  token_count,
  block_tables,
  output_slots,
}
```

Runtime은 계획을 실행하고 결과를 scheduler에 반환한다.

## Backpressure

- 최대 waiting requests
- 최대 active sequences
- 최대 total tokens/KV blocks
- admission timeout 또는 즉시 reject 정책
- cancellation 우선 처리

## 테스트

- deterministic scheduler simulation
- mixed prompt lengths
- decode starvation 방지
- prefill chunking
- KV OOM admission
- request cancellation at every state
- output slot mapping
- random event property test
- long-running allocation accounting

실제 결과는 다음과 같다. 모든 CUDA compile, checkpoint load와 inference는 로컬이
아닌 `server-4096`의 RTX 4090에서 실행했다.

| Gate | 결과 |
|---|---|
| workspace all-features strict Clippy | 통과 |
| workspace all-targets/all-features | 237 passed, 0 failed, 65 ignored |
| workspace all-features doctests | 13 passed, 0 failed |
| scheduler unit + deterministic simulation | 21 + 14 passed |
| low-level indexed/ragged batch GPU | 4 passed |
| SmolLM2 batch parity | 5 passed, prefill byte exact, decode top-1 32/32 exact |
| Qwen2.5 batch parity | 5 passed, prefill byte exact, decode top-1 32/32 exact |
| scheduler → CUDA → greedy sampler → commit | prefill/decode 통과, close 뒤 block/allocation 0 |
| long-running accounting | 실제 1,001 iteration 통과, close 뒤 allocation 0 |
| Compute Sanitizer memcheck | 0 errors |
| Compute Sanitizer racecheck | 0 hazards, 0 errors, 0 warnings |

## 지표

- queue wait
- scheduler CPU time
- iteration batch size
- batched tokens
- prefill/decode 비율
- KV utilization
- rejection/cancellation
- GPU idle gap

## 비범위

- HTTP server
- prefix caching
- priority tenant/SLA
- multi-GPU routing
- offload

## 원격 provenance와 증거

- authoritative source commit:
  `a303a23633baf465f625154b688cafc4009801b6`
- 원격 append-only evidence root:
  `/home/psyche/rustinfer-artifacts/pr13/a303a23633baf465f625154b688cafc4009801b6/run-20260825T150407Z`
- source archive SHA-256:
  `854c784fd5620446e2fbfce81ea219c034a8b60b5a9b05ff5cf9b6a5e48d2eb3`
- 원격 `SHA256SUMS` SHA-256:
  `0af9140e388ca386b989b4825cca039178253bae72c379ec2ea8d1d4978b9604`
- container: `rustinfer-native-cuda:pr04-c6c93e2`
  (`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`)
- GPU/runtime: NVIDIA GeForce RTX 4090, compute capability 8.9, driver
  580.173.02, CUDA 12.8.1, nvcc 12.8.93
- 실행 격리: source/checkpoint read-only mount, `--network none`, Cargo
  `--locked --offline`
- version-controlled evidence index:
  [PR 13 continuous batching evidence](../benchmarks/results/20260825T150407Z-rustinfer-continuous-batching-pr13-run001/README.md)

## 제한

- Functional correctness gate이며 latency/throughput 개선을 주장하지 않는다. 성능
  profile과 목표는 PR 15 범위다.
- 실제 model batch gate는 최대 2 requests, fixed token capacity `M=8`을 검증했다.
- Synthetic CUDA device-loss/kernel-fault injection은 수행하지 않았다. Adapter의
  pre-dispatch/commit failure는 unit test로 검증했고, 모든 post-execute failure는
  stream quiescence를 요구하는 보수 계약을 사용한다.
- GPU kernel을 즉시 중단하지 않는다. 실행 중 cancel은 iteration 종료와 stream
  quiescence 뒤에 정산한다.
- Runtime allocation accounting은 rustinfer 소유 CUDA buffer 기준이며 process peak
  VRAM이나 CUDA driver 내부 allocation을 의미하지 않는다.
- Single GPU/sm89만 검증했다. multi-GPU routing, prefix cache, tenant priority,
  speculative decoding과 offload는 포함하지 않는다.

## Rollback

Scheduler crate, batch executor/adapter와 ragged CUDA primitives를 PR 13 commit 범위로
함께 revert한다. 부분 rollback으로 scheduler plan schema와 runtime batch ABI를 섞지
않는다. Rollback 후에는 PR 12의 SmolLM2/Qwen single-request generation과 reference
forward gate를 재실행한다. 새 driver/toolkit/model에서 fixed E0 parity, sanitizer,
1,001-iteration accounting 또는 close ownership gate가 실패하면 continuous-batch
rollout을 중단하고 기존 single-request path를 사용한다.

## 완료 기준

- [x] concurrency 1 결과가 단일 요청 path와 동일
- [x] 여러 요청 결과가 독립 실행과 일치
- [x] cancellation에서 block 누수 없음
- [x] scheduler simulation이 재현 가능
- [x] p95 queue/iteration 지표 수집
- [x] overload 시 bounded memory 유지

[← 이전](12-qwen-compatibility.md) | [목차](README.md) | [다음 →](14-api-and-streaming.md)
