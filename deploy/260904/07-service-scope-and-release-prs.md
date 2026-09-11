# PR-S — 서비스 범위와 release 승격 후속 카드

이 트랙은 short c1 성능 claim을 확장하기 위한 별도 PR들이다. PR-G04와 RUN-B의
candidate evidence가 없으면 implementation을 시작하지 않는다.

여기서 `RUN-B`는 [05](05-candidate-gpu-and-competitive-campaign.md)의 B01~B04를 뜻한다.
core M4/M5 이후 새 S 기능이 runtime/default에 들어가면 기존 report는 그대로 보존하되 B01~B04를
새 candidate로 다시 실행한다.

## PR-S01 — C12-A prefix cache admission contract

**목적:** prefix cache를 구현하기 전에 tenant/model/tokenizer/RoPE/dtype/layout/adapter/cache-domain을
포함한 exact identity와 disabled-by-default 정책을 고정한다.

**범위:** admission schema, bounded cache budget/config, rejection reason, CPU property tests. KV reuse,
scheduler lookup, cache-on benchmark는 넣지 않는다.

**변경 표면:** `crates/riley-runtime/src/prefix_cache.rs`(신규 가능), paged-KV identity types,
scheduler/server config schema와 CPU tests. 새 production crate는 architecture approval 없이 추가하지 않는다.

**완료:** cross-tenant/cross-model/token-one-difference/stale-generation hit가 모두 fail-closed이며
cache disabled는 current exact behavior와 동등하다.

## PR-S02 — C12-B exact prefix cache implementation

**선행:** S01 승인.

**범위:** complete block만 publish, generation/version/refcount lifecycle, scheduler bounded lookup,
shutdown/unload cleanup. tenant sharing domain 외 재사용은 금지한다.

**gate:** cross-domain hit/refcount error 0, miss TTFT p95 `<=1.03`, 90% controlled hit에서
TTFT 또는 prefill work `>=20%`, c8/c32 goodput non-regression, cache disable rollback 검증.

## RUN-S02P — C12-C controlled promotion campaign

**선행:** S02 correctness와 candidate re-freeze.

cache-off, unique-prompt miss, exact 50% hit, exact 90% hit을 서로 다른 cells로 실행한다.
prefix lengths 128/1024/4096, c1/c8/c32에서 lookup/publish/eviction latency, reused tokens/blocks,
cache bytes와 effective KV capacity를 기록한다. cross-tenant sharing은 별도 negative cell이다.

S02의 gate를 통과해도 stable/default promotion은 이 campaign과 B01/B03/B04 재실행 뒤에만 결정한다.

## PR-S03 — C13 isolated GPU worker protocol contract

**목적:** restartable worker 전에 IPC frame, worker epoch, exactly-once terminal, backpressure,
transactional reload, in-process rollback을 CPU fault model로 고정한다.

**범위:** protocol/state machine/fault tests만. process-mode default, real GPU worker, shared-memory ring은 제외.

**변경 표면:** server-owned protocol/supervisor module과 tests. new crate가 필요하면 이 PR에서 구현하지
않고 architecture-boundary proposal만 별도 카드로 추가한다.

**완료:** stale epoch token, partial/malformed frame, cancellation/crash race, duplicate terminal이
CPU property/fault test에서 fail-closed한다.

## PR-S04A — C13 minimal UDS worker implementation

**선행:** S03 및 candidate/GPU authority.

**범위:** UDS 기반 최소 worker process, epoch/heartbeat/circuit breaker, transactional load와
`--engine-process-mode in-process|isolated`. shared-memory optimization은 넣지 않는다.

**gate:** crash 100회에서 API crash 0, duplicate terminal/token 0, orphan resource 0,
TTFT/TPOT p95 `<=1.03`, throughput `>=0.97`, in-process immediate rollback.

## PR-S04B — IPC overhead optimization, conditional

**시작 조건:** S04A correctness/stability는 통과했지만 사전 latency/throughput gate만 실패하고,
profile이 UDS copy/wakeup을 원인으로 지목함.

shared-memory event ring 또는 protocol batching 중 하나만 선택한다. epoch/ownership/backpressure와
in-process rollback을 유지하고 S04A fault/soak 및 performance gate를 다시 실행한다. 조건이 없으면
이 PR은 `not-needed`로 닫는다.

## RUN-S05 — C14 additional hardware/context expansion

**목적:** B03에서 끝낸 RTX 4090 0.5B/1~3B/7~8B matrix를 반복하지 않고, 승인된 H100/sm90 및
후속 32K context 등 추가 hardware/capability lane으로 범위를 확대한다.

**선행:** B03/B04 RTX 4090 closed report, target hardware authorization, hardware별 current vLLM pin.

**완료:** model/hardware별 immutable lane, warm/cold policy, capability exclusion, M4/M5 report를
분리해 기록한다. 지원하지 않는 cell은 zero/failure로 숨기지 않고 `unavailable`로 표기한다.

## release/default 승격 공통 gate

각 S-track 또는 graph/kernel 최적화가 default가 되려면 같은 frozen candidate에서 다음 모두 통과해야 한다.

- c1/c4/c8, `p128/p1024/p4096`, mixed prefill/decode, cold start
- HTTP streaming, cancellation, disconnect, backpressure
- Llama/Qwen representative parity, Python-free/runtime lifecycle, candidate-bound soak
- deployed HTTP E2E, active default receipt, `disabled`/in-process/reference rollback

이 gate 전까지 모든 새 path는 opt-in/experimental이다.

S01~S05는 서로 독립적으로 모두 시작할 수 있는 목록이 아니다. S02/RUN-S02P는 S01,
S04A/S04B는 S03, RUN-S05는 B03/B04와 별도 hardware 승인 뒤에만 진행한다.
