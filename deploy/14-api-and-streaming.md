# PR 14 — OpenAI 호환 API와 Streaming

**상태:** Implemented

**선행 조건:** [PR 13](13-scheduler-and-batching.md)  
**다음:** [PR 15 — Profiling과 최적화](15-profiling-and-optimization.md)

**검증된 실행 소스:** `8782121068909cc4fc6ff8d13f97e629aaabe271`

[← 이전](13-scheduler-and-batching.md) | [목차](README.md) | [다음 →](15-profiling-and-optimization.md)

## 목적

runtime과 scheduler 위에 얇은 Rust API server를 추가하고, client 연결 상태를 request
cancellation과 backpressure에 연결한다.

이 단계가 Gate D다.

## 구현 결과

### Protocol과 DTO 경계

- `std::net` 기반 bounded HTTP/1.1 connection-close server를 구현했다. Request 하나만
  처리하며 `Content-Length` JSON을 요구한다. Transfer-Encoding, upgrade, trailer,
  pipelining과 framing ambiguity는 fail closed한다.
- Request target, header count/bytes, body bytes, prompt, stop sequence, output token과
  sampling numeric field를 allocation 전에 명시적 상한으로 검사한다.
- OpenAI completions wire DTO는 validation/normalization 뒤 internal
  `GenerationRequest`로 변환된다. Runtime과 scheduler crate는 OpenAI type에 의존하지
  않는다.
- 단일 string prompt와 `temperature`, `top_p`, `seed`, `stop`, `max_tokens`, `stream`을
  지원한다. Token-array/multi-prompt와 구현하지 않은 parameter는 조용히 무시하지 않고
  sanitized 400으로 거절한다.

### Bounded service와 inference engine

- Listener는 bounded connection queue와 고정 worker thread 수를 사용한다. Engine은
  bounded command queue와 request별 bounded event channel을 사용한다.
- 단일 engine worker만 scheduler, model/tokenizer, generation state, prepared batch
  executor, CUDA context/stream을 소유한다. HTTP worker는 GPU/runtime state를 직접
  만지지 않는다.
- Scheduler commit이 성공한 token만 외부 event로 publish한다. Event receiver drop,
  network disconnect, timeout, server shutdown과 slow consumer는 atomic cancellation
  capability로 연결된다.
- Engine submission/live request, scheduler waiting/active sequence, prompt/output/context,
  KV block, connection/request/event queue와 observation memory가 모두 설정된 상한을 가진다.
- `rustinfer serve --model PATH` CLI를 추가했다. Bind address, device, active/waiting,
  context/output, batch/prefill와 KV/weight bound를 명시적으로 설정할 수 있고,
  `--shutdown-on-stdin`으로 portable graceful shutdown을 수행한다.

### Streaming과 cancellation

- SSE 순서는 `token delta* → finish 또는 sanitized error → [DONE]`으로 고정했다.
  Terminal 뒤 delta, 중복 terminal과 `[DONE]` 선행을 encoder가 거절한다.
- Response head 전 오류는 HTTP JSON으로, head 뒤 오류는 SSE error frame과 `[DONE]`으로
  표현한다.
- Waiting/prefill/decode request drop은 safe scheduler transition으로 회수된다. 실행 중
  GPU kernel을 즉시 중단하지 않고 iteration 종료와 stream quiescence 뒤에 정산한다.
  현재 iteration의 미commit token은 cancellation snapshot과 client stream에 포함하지
  않는다.
- Aggregate response는 event를 무한 저장하지 않고 maximum response byte bound를
  검사한다. Streaming write와 periodic aggregate probe가 disconnect를 감지하며 event
  channel full은 slow-consumer cancellation로 전환된다.
- 실제 GPU gate는 192-token prompt를 1-token prefill chunk로 실행해 첫 token 전
  disconnect를 고정했다. `TTFT=None`, generated token 0과 active/waiting 0을 확인한다.
  Decode gate는 첫 non-empty delta 뒤 disconnect하고 TTFT와 완전 회수를 확인한다.

### Shutdown 경합 수정

첫 원격 SmolLM2 gate에서 service event wait와 backend shutdown 경합이 드러났다.
`stopping=true` 직전에 recv에 들어간 뒤 backend의 `Finished::Cancelled`가 먼저 도착하면
503 대신 408이 반환됐다. Recv 성공 뒤에도 shutdown state를 우선하도록 non-streaming은
503, 이미 head를 보낸 streaming은 sanitized shutdown SSE error와 `[DONE]`을 반환한다.
동일 경합을 `CancelOnShutdown` CPU regression fixture로 고정했고 두 checkpoint에서
active shutdown과 engine drain을 재검증했다.

## 초기 endpoint

| Method | Endpoint | 결과 |
|---|---|---|
| `GET` | `/healthz` | process health와 backend 상태 |
| `GET` | `/readyz` | readiness를 HTTP status와 JSON으로 분리 |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/v1/models/{id}` | exact configured model metadata |
| `POST` | `/v1/completions` | streaming SSE 또는 aggregate completion |

## Layering

```text
bounded HTTP framing
  ↓
OpenAI wire DTO validation/normalization
  ↓
Internal GenerationRequest
  ↓
bounded inference command/event channels
  ↓
Scheduler immutable IterationPlan
  ↓
prepared Runtime/CUDA executor
```

## 서비스 오류

| Class | HTTP 정책 |
|---|---|
| malformed/invalid/unsupported request | 400 또는 framing별 4xx |
| unsupported method | 405 |
| request/admission timeout or cancellation before response | 408 |
| connection/backend overload | 429 |
| model unavailable/server shutdown | 503 |
| internal CUDA/model/encoding failure | sanitized 500 |

내부 pointer, filesystem path, tokenizer input, CUDA stack detail은 외부 응답에 노출하지
않는다. 이미 시작한 SSE의 오류는 동일한 sanitized public code만 보낸다.

## 관측성

Bounded in-process observation snapshot은 다음 필드만 보존한다.

- request ID, model ID
- queue wait, TTFT
- tokens generated, finish reason
- status와 error class
- active/waiting count

Prompt와 generated text는 observation/log에 저장하지 않는다. Persistent exporter와
profiling용 token-level timestamp는 PR 15 범위다.

## 테스트

모든 CUDA compile, checkpoint load와 inference는 로컬이 아닌 `server-4096`의 RTX
4090에서 수행했다.

| Gate | 결과 |
|---|---|
| workspace all-targets/all-features strict Clippy | 통과 |
| workspace all-targets/all-features | 275 passed, 0 failed, 66 ignored |
| workspace all-features doctests | 14 passed, 0 failed |
| local server CPU tests | lib 36 + CLI 3 passed |
| feature matrix/boundary/Python-free workspace | 통과 |
| SmolLM2 real CUDA HTTP lifecycle | 1 passed, 6.23 s |
| Qwen2.5 real CUDA HTTP lifecycle | 1 passed, 21.35 s |
| streaming/non-streaming parity와 SSE terminal order | 두 model 통과 |
| prefill/decode disconnect reclaim | active/waiting 0 |
| bounded concurrency | 3 clients, deadlock 없음 |
| active graceful shutdown | HTTP 503, 완전 drain |
| release CLI smoke | health/models/completion/SSE/stdin shutdown 통과 |

## 원격 provenance와 증거

- authoritative source commit:
  `8782121068909cc4fc6ff8d13f97e629aaabe271`
- 원격 append-only evidence root:
  `/home/psyche/rustinfer-artifacts/pr14/8782121068909cc4fc6ff8d13f97e629aaabe271/run-20260825T160859Z`
- source archive SHA-256:
  `d30bcec86fa0df57fc5aa492cb32e0261bf79d7a8e25b02077ab497e65d00766`
- release binary SHA-256:
  `6cad437c640d32cd05eeac9bdd612671a6742de2c83b2c3023a37d8a6601de74`
- 원격 `SHA256SUMS` SHA-256:
  `f79fc20db0de4c3e8f73d5047d6778c7389188ee8e72304f49405951f4ec5e73`
- container: `rustinfer-native-cuda:pr04-c6c93e2`
  (`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`)
- GPU/runtime: NVIDIA GeForce RTX 4090, compute capability 8.9, driver
  580.173.02, CUDA 12.8.1, nvcc 12.8.93
- 실행 격리: source/checkpoint/Cargo cache read-only mount, `--network none`, Cargo
  `--locked --offline`
- version-controlled evidence index:
  [PR 14 API/streaming evidence](../benchmarks/results/20260825T160859Z-rustinfer-api-streaming-pr14-run001/README.md)

## 제한

- Functional correctness gate이며 latency/throughput 개선을 주장하지 않는다. Canonical
  workload, repeatability와 profiler 측정은 PR 15 범위다.
- Completions만 제공한다. Chat completions, TLS, authentication, tenant quota, HTTP/2,
  keep-alive, metrics exporter와 multi-GPU routing은 포함하지 않는다.
- CLI의 portable graceful path는 stdin 기반이다. OS signal integration, runtime-only
  artifact와 operational release packaging은 PR 16 범위다.
- GPU kernel을 즉시 interrupt하지 않고 safe iteration boundary에서 취소한다.
- Single GPU/sm89와 기록된 SmolLM2/Qwen checkpoint만 검증했다.

## Rollback

Server domain/protocol/OpenAI DTO, engine/service/CLI와 remote `api_gpu` gate를 PR 14 commit
범위로 함께 revert한다. 부분 rollback으로 wire DTO와 engine event/scheduler contract를
섞지 않는다. Rollback 후에는 PR 13 scheduler→CUDA integration, batch parity와 allocation
accounting gate를 다시 실행한다. Sanitized error, bounded queue, disconnect reclaim, active
shutdown 503 또는 two-model streaming parity가 실패하면 API rollout을 중단한다.

## 완료 기준

- [x] streaming/non-streaming 결과 일치
- [x] disconnect/cancel에서 자원 회수
- [x] bounded queue와 overload 응답
- [x] graceful shutdown
- [x] API DTO가 runtime crate로 침투하지 않음
- [x] concurrency test에서 deadlock 없음

[← 이전](13-scheduler-and-batching.md) | [목차](README.md) | [다음 →](15-profiling-and-optimization.md)
