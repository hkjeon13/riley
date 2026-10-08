# C12 — Tenant-safe Exact Prefix Cache

**상태:** Planned track  
**의미 등급:** `reference`  
**한 가지 목적:** 동일한 model execution identity와 허용된 sharing domain 안에서만 immutable KV block을 exact 재사용한다.

[이전: C11](11-lm-head-sampling-fusion.md) | [목차](README.md) | [다음: C13](13-restartable-gpu-worker.md)

## 1. PR 분리

기존 [`deploy/17-extension-gates.md`](../17-extension-gates.md)의 규칙을 따른다.

### C12-A — Admission-only PR

정확히 다음 네 파일만 추가한다.

```text
deploy/extensions/registry.json
deploy/extensions/proposals/prefix-cache-v1.json
deploy/extensions/plans/prefix-cache-v1.md
benchmarks/extensions/contracts/prefix-cache-v1.json
```

상태는 `approved-for-implementation`, `default_enabled=false`, `stable_default=false`, `implementation_link_path=null`이다.

### C12-B — Implementation PR

- experimental implementation manifest 연결
- production source와 direct integration test 추가
- default off
- exact fallback 유지

### C12-C — Stable promotion

현재 extension schema v1이 stable promotion을 지원하지 않으므로 먼저 schema v2 transition이 필요하다. C12-B 결과만으로 stable default를 선언하지 않는다.

## 2. 보안/정확성 identity

cache key는 token IDs만으로 만들지 않는다.

```text
model ID and exact revision
weights/config hashes
tokenizer revision and token IDs
RoPE configuration/scaling
attention/KV layout revision
dtype and KV storage format
adapter/LoRA identity or none
prompt-processing feature identity
tenant sharing domain
cache schema/hash algorithm version
```

하나라도 다르면 miss다. unknown identity field는 공유하지 않는다.

## 3. Tenant 정책

기본은 tenant-private다.

```text
sharing_domain = tenant_id 또는 explicit private namespace
```

cross-tenant 공유는 운영자가 명시적으로 같은 trusted sharing domain을 부여한 경우에만 가능하다. request body가 임의 domain을 선택하게 하지 않는다.

로그/metric에는 raw token, prompt text, tenant ID를 label로 노출하지 않는다.

## 4. Block identity

KV page/block 단위 exact prefix를 저장한다.

```text
PrefixBlockKey {
  execution_identity_hash,
  sharing_domain_hash,
  parent_prefix_hash,
  token_block_hash,
  logical_token_count,
}
```

cryptographic hash collision만 믿지 않고 hit 시 logical token count와 저장된 compact token identity를 추가 확인한다. 원문 prompt를 저장하지 않고 token ID bytes 또는 strong digest chain을 사용한다.

부분 block은 초기 버전에서 공유하지 않는다. complete immutable block만 publish한다.

## 5. Ownership 상태

```text
Free
Reserved(request)
Active(request)
Cached(cache_entry)
Shared(active_refs, cache_ref)
Evicting
```

보존식:

```text
free + reserved + active_exclusive + cached_only + shared_physical = total physical blocks
```

logical refcount와 physical block count를 혼동하지 않는다.

- request ref
- cache ownership ref
- lookup transient pin
- eviction pin

각 ref class를 metric/accounting에서 분리한다.

## 6. Publish와 lookup

### Publish

- scheduler commit이 성공한 complete block만 cache candidate
- cancelled/failed/uncommitted block은 publish 금지
- token/block identity와 KV version이 일치할 때 immutable 전환
- publish failure가 request correctness를 실패시키지 않고 cache miss로 격리 가능해야 함

### Lookup

- longest complete block prefix 탐색
- lookup 동안 entry pin
- request block table에 shared ref로 attach
- attach/plan validation 성공 후 pin을 active ref로 전환
- 실패 시 정확히 rollback

cache miss는 current exact prefill path다.

## 7. Eviction

초기 정책은 bounded LRU 또는 CLOCK 중 하나로 단순하게 시작한다.

선택 기준:

- hot path allocation 없음
- lock contention bounded
- pinned/active entry eviction 금지
- deterministic simulation 가능
- tenant/domain별 quota 적용 가능

전체 cache bytes/block count와 domain quota를 설정한다. cache가 KV admission을 고갈시키면 scheduler가 cache eviction을 먼저 요청하되 active request block을 침범하지 않는다.

## 8. Privacy와 lifecycle

- domain 삭제/rotation 시 해당 domain entry invalidate
- model unload 시 model identity entry 전체 invalidate
- adapter unload 시 adapter identity entry invalidate
- server shutdown에서 active ref drain 후 cache ref free
- stale entry가 block ID 재사용 후 hit하지 않도록 generation/version 포함
- cache debugging dump에 prompt/token 원문 없음

## 9. Scheduler integration

cache lookup은 admission/prefill plan 전 bounded 단계에서 수행한다.

- matched token count
- reused block count
- remaining prefill tokens
- cache lookup latency
- miss/fallback reason

scheduler priority에 cache hit를 바로 가산하는 것은 C12 범위가 아니다. 먼저 exact reuse와 lifetime을 닫는다.

## 10. Correctness 테스트

- identical identity hit
- token 한 개 차이 miss
- model/tokenizer/RoPE/dtype/layout/adapter 차이 miss
- tenant domain 차이 miss
- hash collision test double에서 token confirmation miss
- complete block만 publish
- cancellation/commit failure publish 금지
- active shared block eviction 금지
- generation/version stale entry 거부
- concurrent lookup/eviction
- shutdown/model unload accounting 0
- prefix hit on/off generated token exact

property test로 random publish/lookup/attach/cancel/evict sequence에서 보존식과 refcount를 검증한다.

## 11. Performance matrix

### Miss overhead

unique prompts, hit 0%:

- TTFT p95 ratio `<= 1.03`
- throughput ratio `>= 0.97`
- cache lookup CPU/allocation bound

### Controlled hit

- 50% exact prefix hit
- 90% exact prefix hit
- prefix lengths 128/1024/4096
- c1/c8/c32

Metric:

- TTFT/throughput/SLO goodput
- reused tokens/blocks
- lookup/publish/eviction latency
- cache bytes and effective KV capacity
- exact fallback/miss rate
- domain quota rejection

## 12. Promotion gate

C12-B experimental gate:

- token mismatch/cross-domain hit/refcount error 0
- miss workload TTFT p95 ratio `<= 1.03`
- 90% hit primary TTFT 또는 prefill work `>= 20%` 개선
- c8/c32 goodput 개선 또는 non-regression
- peak VRAM은 configured cache budget 안
- cache disabled가 current exact behavior와 동일
- close/model unload 후 live blocks 0

stable promotion threshold와 운영 default는 schema v2에서 별도 고정한다.

## 13. 예상 구현 파일

```text
crates/riley-runtime/src/prefix_cache.rs 또는 별도 bounded module
crates/riley-runtime/src/paged_kv.rs
crates/riley-scheduler/src/*
crates/riley-server/src/main.rs
crates/riley-server/src/engine.rs
crates/riley-runtime/tests/prefix_cache.rs
crates/riley-scheduler/tests/prefix_cache_simulation.rs
benchmarks/extensions/results/*
```

## 14. Configuration

```text
--prefix-cache disabled|private|shared-domain
--prefix-cache-max-blocks N
--prefix-cache-max-bytes BYTES
--prefix-cache-domain-quota-blocks N
```

C12-B default는 `disabled`다. request payload가 arbitrary sharing domain을 지정하지 못한다.

## 15. 오류와 rollback

cache metadata corruption/identity mismatch는 entry를 invalidate하고 exact prefill로 fallback한다. KV physical ownership이 불명확한 오류는 cache subsystem 또는 model executor를 poison하고 자동 재사용하지 않는다.

운영 rollback은 `--prefix-cache disabled`다. 코드 rollback은 cache metadata와 scheduler integration을 함께 revert하되 paged KV reference correctness test를 다시 실행한다.

## 16. 완료 정의

허용된 sharing domain에서만 exact complete KV block이 재사용되고, miss/disable 경로가 기존 의미를 보존하며, random concurrency/eviction/cancellation에서도 block 보존식과 privacy boundary가 깨지지 않을 때 C12-B가 완료다. Stable 완료는 schema v2 promotion 이후다.
