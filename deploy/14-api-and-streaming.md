# PR 14 — OpenAI 호환 API와 Streaming

**상태:** Planned  
**선행 조건:** [PR 13](13-scheduler-and-batching.md)  
**다음:** [PR 15 — Profiling과 최적화](15-profiling-and-optimization.md)

[← 이전](13-scheduler-and-batching.md) | [목차](README.md) | [다음 →](15-profiling-and-optimization.md)

## 목적

runtime과 scheduler 위에 얇은 Rust API server를 추가하고, 클라이언트 연결 상태를 request cancellation과 backpressure에 연결한다.

이 단계가 Gate D다.

## 초기 endpoint

최소 범위:

- health/readiness
- model metadata
- OpenAI-compatible completions 또는 chat completions 중 하나
- streaming SSE
- non-streaming response

API 호환성을 위해 내부 구조를 OpenAI request type에 직접 종속시키지 않는다.

## Layering

```text
HTTP DTO
  ↓ validation/normalization
Internal GenerationRequest
  ↓
Scheduler
  ↓
Runtime
```

## Streaming 규칙

- event ordering 보장
- token delta와 finish event 구분
- client disconnect 감지
- bounded channel
- slow consumer backpressure
- UTF-8/token decode 오류 처리
- 이미 전송한 응답 이후 오류 표현 방식 정의

## Cancellation

다음 시점 모두 테스트한다.

- waiting queue
- prefill 전
- prefill 중
- decode 중
- finish 직전
- network disconnect

GPU kernel 자체를 즉시 중단하기 어렵다면 iteration 종료 후 안전하게 회수하는 정책을 명시한다.

## 서비스 오류

- invalid request: 4xx
- overload: 429 또는 명시 정책
- unsupported model params
- internal CUDA/model failure
- timeout
- server shutdown

내부 pointer, filesystem path, CUDA stack detail을 외부 응답에 노출하지 않는다.

## 관측성

- request ID
- model ID
- queue wait
- TTFT
- tokens generated
- finish reason
- status/error class
- active/waiting count

prompt와 generated text logging은 기본 비활성 또는 별도 privacy policy를 따른다.

## 완료 기준

- [ ] streaming/non-streaming 결과 일치
- [ ] disconnect/cancel에서 자원 회수
- [ ] bounded queue와 overload 응답
- [ ] graceful shutdown
- [ ] API DTO가 runtime crate로 침투하지 않음
- [ ] concurrency test에서 deadlock 없음

[← 이전](13-scheduler-and-batching.md) | [목차](README.md) | [다음 →](15-profiling-and-optimization.md)
