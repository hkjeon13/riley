# C13 — Restartable Isolated GPU Worker

**상태:** Planned  
**의미 등급:** `reference` systems architecture  
**한 가지 목적:** CUDA context poison, native abort, release `panic=abort`가 HTTP/API process 전체 장애로 전파되지 않도록 model+scheduler+CUDA를 restartable worker process로 격리한다.

[이전: C12](12-tenant-safe-prefix-cache.md) | [목차](README.md) | [다음: C14](14-multi-model-hardware-matrix.md)

## 1. 목표 구조

```text
riley-server
  HTTP/SSE, auth, request validation, readiness
       |
       | bounded local IPC
       v
riley-engine-worker
  scheduler, model runtime, KV, sampling, CUDA context
       |
       v
GPU
```

scheduler는 worker 안에 둔다. 매 token마다 API process와 scheduler가 왕복하지 않도록 request admission/control과 generated token event만 IPC로 전달한다.

## 2. 범위

### 포함

- worker binary/process mode
- versioned bounded IPC protocol
- request/stream/cancel/shutdown control
- generated token/status/terminal event
- worker epoch와 exactly-once terminal 처리
- worker crash detection/restart/circuit breaker
- transactional model load/readiness switch
- fault injection과 performance overhead gate
- in-process rollback mode

### 비범위

- remote/distributed worker
- multi-GPU routing
- external message broker
- request 자동 replay로 중복 생성 숨기기
- persistent KV across worker crash

## 3. Process mode

```text
--engine-process-mode in-process
--engine-process-mode isolated
```

초기 default는 `in-process`다. S1과 overhead gate 통과 후 운영 profile별 isolated default 승격을 검토한다.

## 4. IPC protocol

초기 transport는 Unix domain socket의 length-delimited binary frame 또는 동등한 local IPC다. protocol은 다음 특성을 가져야 한다.

- schema/version handshake
- maximum frame/request/token sizes
- bounded send/receive queue
- request ID + server generation + worker epoch
- checksum 또는 strict frame validation
- unknown message fail-closed
- partial frame/peer close 처리

대량 tensor/model data를 IPC로 전달하지 않는다. model artifact는 worker가 read-only path에서 직접 load한다. generated event는 token ID, sequence index, status, timing metadata 정도로 제한한다.

shared-memory ring은 UDS overhead가 C13 gate를 넘을 때만 별도 후보로 검토한다. 첫 구현부터 복잡한 zero-copy IPC를 강제하지 않는다.

## 5. Message set

```text
Hello / HelloAck
LoadModel / ModelReady / ModelLoadFailed
SubmitRequest / RequestAccepted / RequestRejected
CancelRequest / CancelAck
TokenEvent
TerminalEvent
MetricsSnapshot
Drain / Shutdown / ShutdownAck
Heartbeat
WorkerFatal
```

모든 request는 server와 worker에서 상태 전이를 검증한다. TokenEvent가 TerminalEvent 뒤에 오면 protocol violation이다.

## 6. Worker epoch와 exactly-once

worker restart마다 monotonic/random unique epoch를 부여한다.

```text
RequestKey = server_generation + public_request_id + worker_epoch
```

server는 현재 epoch의 event만 수락한다. 이전 worker의 buffered/stale event는 폐기한다.

terminal ledger는 request당 다음 중 하나만 허용한다.

```text
finished
cancelled
failed-retriable
failed-terminal
rejected
```

worker crash 시 in-flight request를 자동으로 새 worker에서 재실행하지 않는다. client가 이미 token을 받았을 수 있으므로 중복 생성 위험이 있다. 명시적 retriable failure를 보내고 상위 client 정책이 새 request ID로 재시도한다.

## 7. Worker lifecycle

```text
Starting
-> LoadingModel
-> Warming
-> Ready
-> Draining
-> Stopped

모든 상태 -> Failed
```

readiness는 worker가 다음을 완료한 뒤에만 true다.

- artifact/hash validation
- model load
- CUDA context/backend prepare
- warmup
- golden probe
- IPC event loop ready

## 8. Transactional model reload

기존 ready worker를 유지한 채 새 worker를 별도 epoch로 기동한다.

```text
new worker load
-> warmup/golden probe
-> Ready
-> server atomic routing switch
-> old worker Drain
-> old worker shutdown
```

새 worker 준비가 실패하면 기존 worker traffic을 유지한다. 동일 GPU에서 두 model을 동시에 올릴 VRAM이 부족하면 drain-and-replace mode를 별도 명시하고 rollback artifact를 유지한다.

## 9. Crash/restart 정책

- peer EOF, heartbeat timeout, child exit 감지
- exit code/signal/core metadata 수집
- current epoch in-flight request terminal failure 처리
- readiness false
- bounded exponential backoff
- N회 연속 실패 시 circuit open
- operator/manual reset 또는 cooldown 후 probe

무한 restart loop로 GPU/host를 소모하지 않는다.

## 10. Backpressure

server와 worker 양쪽 queue는 bounded다.

- submit queue full: admission rejection
- token event queue full: slow client policy에 따라 request cancel/fail
- server disconnect: worker cancel 전송
- worker control queue stuck: health failure

worker가 server보다 빠르게 token을 생성할 때 memory가 무한 증가하지 않도록 per-request/max-total pending event 상한을 둔다.

## 11. Security

- socket directory/file permission 최소화
- peer credential 확인 가능한 platform에서는 UID/PID 검증
- raw prompt/token을 worker crash log에 기본 출력하지 않음
- environment/CLI secret 전달 최소화
- core dump 정책 문서화
- IPC frame에 arbitrary path/command 실행 capability 없음

## 12. 예상 crate/file

production crate 7개 고정 원칙을 깨지 않도록 `riley-server` 내부 추가 binary 또는 기존 crate의 bin target으로 시작한다.

```text
crates/riley-server/src/bin/riley-engine-worker.rs
crates/riley-server/src/ipc/*.rs
crates/riley-server/src/supervisor.rs
crates/riley-server/src/engine.rs
crates/riley-server/src/service.rs
crates/riley-server/tests/isolated_worker.rs
crates/riley-server/tests/worker_faults.rs
ci/run_worker_fault_campaign.sh
```

새 production crate가 정말 필요하면 기존 crate boundary 문서와 별도 architecture approval이 선행되어야 한다.

## 13. Fault tests

- worker 정상 종료
- SIGKILL/abort
- model load panic/abort subprocess
- synthetic CUDA poison/fatal status
- IPC malformed/partial frame
- heartbeat loss
- server shutdown과 worker token event race
- cancel 중 worker crash
- terminal 전/후 crash
- repeated restart/circuit breaker
- old epoch stale token injection

검증:

- server process 생존
- readiness 정확성
- request terminal exactly once
- stale token publish 0
- orphan worker/socket/shared memory 0
- restart 후 golden generation 정상

## 14. Performance gate

same binary/config에서 in-process와 isolated를 비교한다.

- engine-only GPU latency는 worker 내부에서 별도 측정
- HTTP E2E TTFT/ITL에는 IPC overhead 포함
- c1/c8/c32
- slow client/backpressure

사전 기준:

```text
TTFT p95 ratio <= 1.03
TPOT p95 ratio <= 1.03
throughput ratio >= 0.97
CPU utilization ratio <= 1.10
```

초기 UDS가 기준을 넘으면 protocol batching 또는 shared-memory event ring을 별도 candidate로 측정한다. 안정성을 이유로 무제한 latency 회귀를 허용하지 않는다.

## 15. Stability promotion gate

- worker crash 100회 주입에서 server crash 0
- request duplicate terminal/token 0
- restart 성공/실패가 circuit policy와 일치
- process/socket/file descriptor/RSS leak 0
- transactional load 실패 시 old worker serving 유지
- qualification soak에서 worker restart 후 정상 recovery
- in-process rollback mode 유지

## 16. Observability

- worker epoch/state/PID
- restart/crash/circuit counters
- IPC queue depth/bytes/latency
- event lag
- in-flight request by epoch
- model load/warmup/recovery duration
- stale event rejected
- readiness transitions

request ID를 metric label로 넣지 않는다.

## 17. 롤백

즉시 운영 rollback은 `--engine-process-mode in-process`다. isolated mode의 protocol/schema와 supervisor를 함께 revert하며, worker process가 남지 않도록 shutdown/kill cleanup을 수행한다.

## 18. 완료 정의

worker abort/CUDA fatal을 반복 주입해도 API process가 생존하고 모든 in-flight request가 정확히 한 번 실패 처리되며, 새 worker가 golden probe 후 readiness를 회복하고 E2E overhead가 3% gate 안에 있을 때 완료다.
