# PR 14 API and streaming evidence — run001

`8782121068909cc4fc6ff8d13f97e629aaabe271`에서 bounded Rust HTTP service,
OpenAI-compatible completions DTO, SSE streaming, inference worker와 scheduler/CUDA
adapter를 RTX 4090/sm89에서 검증했다. CUDA compile, checkpoint load와 model inference는
모두 `server-4096`의 network-disabled container에서 수행했다. 로컬에서는 source 편집과
CUDA를 사용하지 않는 검사만 수행했다.

## 결론

PR 14 Gate D를 통과했다. `/healthz`, `/readyz`, `/v1/models`,
`/v1/models/{id}`와 `/v1/completions`의 streaming/non-streaming 경로가 제공된다.
HTTP framing, connection/request/event queue, prompt/output/context와 observation buffer는
모두 명시적 상한을 가진다. 내부 DTO는 OpenAI wire type과 분리되어 runtime/scheduler
crate로 API schema가 침투하지 않는다.

SmolLM2와 Qwen2.5 실제 checkpoint에서 greedy streaming/non-streaming text와 finish
reason이 일치했고, SSE는 delta 뒤 finish frame과 `[DONE]` 순서를 보존했다. 긴 prefill을
첫 token 전에 끊은 경우와 첫 decode delta 뒤 끊은 경우 모두 `Cancelled`로 관측되고
active/waiting request가 0으로 회수됐다. 세 동시 client가 deadlock 없이 완료됐고,
active request 중 shutdown은 HTTP 503을 반환한 뒤 engine을 drain했다.

## 구현 계약

- HTTP/1.1 request 하나만 처리하고 `Content-Length` JSON만 허용한다. Transfer-Encoding,
  upgrade, trailer, pipelining과 framing ambiguity는 실행 전에 거절한다.
- Request target/header count/header bytes/body bytes와 OpenAI numeric/string/list field는
  bounded validation 후 internal `GenerationRequest`로 정규화한다.
- Engine command queue와 request별 event channel은 bounded `sync_channel`이다. Engine
  worker만 scheduler, prepared executor, CUDA context/stream과 generation state를 소유한다.
- Scheduler commit 뒤에만 token event를 외부로 publish한다. Kernel 실행 중 cancellation은
  iteration 종료와 stream quiescence 뒤에 정산하며 현재 미commit token은 노출하지 않는다.
- Slow consumer, receiver drop, network disconnect, timeout과 shutdown은 동일 cancellation
  capability로 연결된다. Event channel이 가득 차면 request를 취소해 무한 buffering을 막는다.
- Streaming은 response head 전 오류는 HTTP JSON으로, head 뒤 오류는 sanitized SSE error와
  `[DONE]`으로 표현한다. 내부 path, CUDA detail, pointer와 tokenizer input은 외부 오류에
  포함하지 않는다.
- Request observation은 request/model ID, queue wait, TTFT, token count, finish/error class와
  active/waiting count만 bounded buffer에 보존한다. Prompt/generated text는 기록하지 않는다.

## Shutdown 경합 회귀

첫 원격 SmolLM2 gate가 service의 25 ms event wait와 backend shutdown 사이 경합을
재현했다. `stopping=true` 직전에 recv에 들어간 뒤 backend의 `Finished::Cancelled`가 먼저
도착하면 HTTP 503 대신 408이 반환됐다. Service가 recv 성공 뒤에도 shutdown state를
우선하도록 수정했고, `CancelOnShutdown` CPU fixture로 경합을 고정했다. 수정 후 두 실제
checkpoint 모두 active shutdown 503과 완전 drain을 통과했다.

## 검증 결과

| Gate | 결과 |
|---|---|
| workspace all-targets/all-features strict Clippy | pass |
| workspace all-targets/all-features | 275 passed, 0 failed, 66 ignored |
| workspace all-features doctests | 14 passed, 0 failed |
| SmolLM2 real CUDA HTTP lifecycle | 1 passed, 6.23 s |
| Qwen2.5 real CUDA HTTP lifecycle | 1 passed, 21.35 s |
| streaming/non-streaming greedy parity | 두 model 모두 pass |
| SSE terminal ordering | finish immediately before `[DONE]` |
| prefill/decode disconnect | cancellation 관측, active/waiting 0 |
| bounded concurrency | 3 clients pass, deadlock 없음 |
| active graceful shutdown | HTTP 503, engine ready/accepting false, 완전 drain |
| release CLI smoke | health/models/non-stream/SSE/stdin shutdown pass |

## 환경과 고정 artifact

- Host: `server-4096`, Ubuntu 22.04.5, Linux 6.8.0-138-generic
- CPU/RAM: Intel Core i7-13700K, 24 logical CPUs, 67,185,598,464 bytes RAM
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- Driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- Image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- SmolLM2 revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16
- Qwen2.5 revision: `7ae557604adf67be50417f59c2c2f167def9a775`, BF16
- Container network disabled; source/checkpoints/Cargo cache read-only; target/evidence만 writable
- Release binary SHA-256:
  `6cad437c640d32cd05eeac9bdd612671a6742de2c83b2c3023a37d8a6601de74`

주요 Cargo command는 모두 `--locked --offline`로 실행했다.

## Artifact와 provenance

전체 stdout, raw HTTP response, exact source tar, release binary, normalized command,
host/GPU/toolchain/checkpoint provenance와 checksum은 다음 append-only 원격 root에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr14/8782121068909cc4fc6ff8d13f97e629aaabe271/run-20260825T160859Z
checksum payload lines: 23
regular files including SHA256SUMS: 24
regular-file bytes: 59,868,341
artifact-root apparent bytes: 59,876,533
SHA256SUMS sha256: f79fc20db0de4c3e8f73d5047d6778c7389188ee8e72304f49405951f4ec5e73
source tar sha256: d30bcec86fa0df57fc5aa492cb32e0261bf79d7a8e25b02077ab497e65d00766
```

`logs/cuda-clippy.log`는 Cargo cache mount를 빠뜨린 첫 setup 시도를 투명하게 보존한
파일이다. Source failure가 아니며, read-only cache를 명시한 authoritative 재실행 결과는
`logs/cuda-clippy-pass.log`에 있다.

## 제한

- Gate D functional/correctness evidence이며 latency 또는 throughput 개선을 주장하지
  않는다. 고정 workload, warmup, 반복성과 profiler 계약은 PR 15 범위다.
- 공개 inference endpoint는 completions 하나이며 chat completions, TLS, authentication,
  tenant quota, metrics export와 multi-GPU routing은 포함하지 않는다.
- Server는 bounded blocking thread pool과 HTTP/1.1 connection-close framing을 사용한다.
  HTTP/2, keep-alive, chunked request body와 general-purpose web framework 기능은 없다.
- CLI의 portable graceful path는 `--shutdown-on-stdin`이다. Process signal integration과
  release packaging은 PR 16 범위다.
- GPU kernel은 즉시 중단하지 않으며 safe iteration boundary와 stream quiescence 뒤에
  취소한다. 단일 GPU/sm89와 기록된 두 checkpoint만 검증했다.
- Observation은 bounded in-process snapshot이다. Persistent metrics/log exporter는 없다.

## Rollback

PR 14의 server protocol/domain/engine/service/CLI와 `api_gpu` target을 함께 revert한다.
부분 rollback으로 OpenAI DTO와 engine event contract를 섞지 않는다. Rollback 뒤에는
PR 13 scheduler→CUDA integration, batch parity와 allocation accounting gate를 다시
실행한다. Sanitized error, bounded queue, disconnect reclaim, active shutdown 503 또는
두-model streaming parity가 실패하면 API rollout을 중단한다.
