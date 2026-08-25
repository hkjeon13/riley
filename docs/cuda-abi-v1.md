# rustinfer CUDA C ABI v1

이 문서는 `kernels/include/rustinfer_cuda.h`가 노출하는 ABI version 1의
binary contract를 정의한다. 이 경계는 Rust ownership API와 CUDA C++ 구현을
연결하기 위한 것이며 Python, PyTorch object, model tensor를 받지 않는다.
ABI metadata 함수 이외의 호출은 CUDA driver 또는 runtime을 초기화할 수 있다.

## 호환성과 type 규칙

- `rustinfer_cuda_abi_version()`은 `RUSTINFER_CUDA_ABI_VERSION`, 현재 `1`을
  반환한다. Rust wrapper는 native library의 값이 자신이 기대하는 값과 같은지
  확인한 뒤 다른 symbol을 사용한다.
- 모든 exported function은 C linkage와 `noexcept`를 사용한다. C++ exception은
  ABI 밖으로 전파되지 않아야 한다.
- ABI의 integer는 `<stdint.h>`의 고정 폭 type만 사용한다. status는 `int32_t`,
  byte/count는 `uint64_t`, 완료 여부는 `uint8_t`의 `0` 또는 `1`이다.
- v1에 field나 function을 추가할 때는 기존 field의 순서, 폭, offset, 의미를
  바꾸지 않는다. 새 struct field는 끝의 reserved 영역을 소비하거나 새
  `struct_size`-gated version을 사용하고, 새 symbol은 기존 symbol을 유지한 채
  추가한다. 기존 layout, numeric value, ownership 또는 함수 의미를 바꾸는
  변경은 ABI version을 올린다.
- v1 caller는 output struct의 `struct_size`를 자신의 `sizeof(struct)`로
  초기화한다. `RustInferCudaDeviceProperties`가 v1 크기보다 작으면
  `INVALID_ARGUMENT/VALIDATION`이며 output을 신뢰하면 안 된다. error buffer는
  선택 사항이므로 `RustInferCudaErrorInfo`가 `NULL`이거나 너무 작으면 상세
  진단만 쓰지 않고 operation status는 그대로 반환한다. reserved field는
  입력에서 `0`으로 두고 출력에서도 의미를 부여하지 않는다.

## 고정 layout

v1은 CUDA가 지원되는 64-bit host C ABI를 대상으로 하며 native build의
`static_assert`가 다음 크기와 핵심 offset을 검증한다.

### `RustInferCudaErrorInfo`

| offset | C type | field | 의미 |
| ---: | --- | --- | --- |
| 0 | `uint32_t` | `struct_size` | caller가 제공한 buffer contract |
| 4 | `int32_t` | `native_code` | 원래 CUDA code, validation/internal 오류는 `0` |
| 8 | `uint32_t` | `domain` | 오류를 만든 subsystem |
| 12 | `uint32_t` | `stage` | 오류를 관측한 lifecycle 지점 |
| 16 | `char[256]` | `message` | NUL-terminated best-effort 진단 |

전체 크기는 272 bytes다. message는 operation과 CUDA 오류 문자열을 포함하되
capacity에 맞게 잘릴 수 있다. status와 숫자 field가 기계 판정의 기준이며
message 문자열은 stable parsing interface가 아니다. caller가 error pointer를
`NULL`로 전달해도 operation의 status 반환은 동일하다.

### `RustInferCudaDeviceProperties`

| offset | C type | field |
| ---: | --- | --- |
| 0 | `uint32_t` | `struct_size` |
| 4 | `int32_t` | `ordinal` |
| 8 | `uint64_t` | `total_memory_bytes` |
| 16 | `uint32_t` | `compute_capability_major` |
| 20 | `uint32_t` | `compute_capability_minor` |
| 24 | `uint32_t` | `multiprocessor_count` |
| 28 | `uint32_t` | `warp_size` |
| 32 | `uint32_t` | `max_threads_per_block` |
| 36 | `int32_t` | `driver_version` |
| 40 | `int32_t` | `runtime_version` |
| 44 | `uint32_t[5]` | `reserved` |
| 64 | `char[256]` | `name` |

전체 크기는 320 bytes다. `name`은 NUL-terminated device name이다. 조회가
실패하면 partially populated output을 사용하지 않는다.

### `RustInferCudaAllocationStats` (PR 04 additive)

| offset | C type | field |
| ---: | --- | --- |
| 0 | `uint32_t` | `struct_size` |
| 4 | `uint32_t` | `reserved` |
| 8 | `uint64_t` | `device_live_bytes` |
| 16 | `uint64_t` | `device_live_allocations` |
| 24 | `uint64_t` | `pinned_host_live_bytes` |
| 32 | `uint64_t` | `pinned_host_live_allocations` |

전체 크기는 40 bytes다. context 내부의 짧은 non-throwing lock 아래 네 값을 함께
갱신하고 snapshot하므로 concurrent `CudaContext::allocation_stats`도 서로 다른
시점의 bytes/count를 섞지 않는다. 정상 explicit close가 모두 성공하면 네 값이
0으로 돌아간다. zero-byte logical handle은 allocation count 1, bytes 0이다.
destructive `cudaFree*`가 오류를 반환해 실제 해제 여부를 확정할 수 없으면 해당
handle은 single-shot으로 소비하되 logical live accounting과 context child lease를
남긴다. 따라서 불확실한 native allocation을 0으로 거짓 보고하거나 context를
release하지 않고 fail-closed leak으로 드러낸다. Create가 실패해 caller-visible
handle을 반환하지 못한 경우에도 rollback `cudaFree*`를 확정하지 못하면 해당 byte와
allocation count 및 child lease를 영구 보존한다. context close는 live-child 검사와
별도로 이 네 accounting 값도 lock 아래 확인하며 하나라도 0이 아니면 primary lease
release를 거부한다.

## status, domain, stage

모든 operation은 `RustInferCudaStatus`를 반환한다. `SUCCESS (0)`만 성공이고
나머지는 다음 stable 분류다.

| 값 | status | 의미 |
| ---: | --- | --- |
| 1 | `INVALID_ARGUMENT` | null, 잘못된 크기 또는 CUDA가 거부한 인자 |
| 2 | `INVALID_DEVICE` | 존재하지 않거나 음수인 device ordinal |
| 3 | `OUT_OF_RANGE` | count/size/launch 범위 초과 |
| 4 | `NOT_READY` | 비차단 query 대상이 아직 완료되지 않음 |
| 5 | `OUT_OF_MEMORY` | host 또는 device allocation 실패 |
| 6 | `DRIVER_ERROR` | 별도 mapping이 없는 CUDA Driver API 오류 |
| 7 | `RUNTIME_ERROR` | 별도 mapping이 없는 CUDA Runtime API 오류 |
| 8 | `INVALID_STATE` | handle ownership 또는 lifecycle 불일치 |
| 9 | `INTERNAL_ERROR` | native invariant 위반 |

error domain은 `NONE (0)`, `VALIDATION (1)`, `DRIVER (2)`, `RUNTIME (3)`,
`INTERNAL (4)`이다. stage는 `INITIALIZE (1)`, `VALIDATION (2)`, `CREATE (3)`,
`LAUNCH (4)`, `SYNCHRONIZE (5)`, `QUERY (6)`, `RECORD (7)`, `COPY (8)`,
`CLOSE (9)`이다. 하나의 CUDA native code가 여러 operation에서 나올 수
있으므로 Rust 오류는 status뿐 아니라 domain, stage, native code, operation과
owned message를 함께 보존한다.

`stream_query`와 `event_query`가 아직 완료되지 않았으면 `NOT_READY`를
반환하고 `out_complete`는 `0`이다. 이는 성공적인 `false`가 아니라 호출자가
재시도할 수 있는 비차단 상태다. query가 `SUCCESS`이면 `out_complete`는 `1`이다.
단, query 자체가 `NOT_READY`였더라도 호출 thread의 context-stack 복원에
실패하면 복원 오류가 우선한다. Rust wrapper는 정상적으로 복원된
`NOT_READY`만 `Ok(false)`로 변환하며 context poison을 완료 미상태로 숨기지 않는다.

## handle ownership과 context

`RustInferCudaContext`, `RustInferCudaStream`, `RustInferCudaEvent`,
`RustInferCudaSmokeBuffer`, `RustInferCudaDeviceBuffer`,
`RustInferCudaPinnedHostBuffer`, `RustInferCudaCopy`는 incomplete C type인 opaque
handle이다. caller는 그 주소의 내부 layout을 읽거나 복사하지 않는다.

- `*_create` 성공은 caller에게 handle 하나의 소유권을 넘긴다. 실패 시 output
  handle은 `NULL`이다.
- `*_close`는 handle의 주소를 받는다. `*handle == NULL`인 close는 성공하는
  idempotent no-op이고, 바깥 pointer 자체가 `NULL`이면
  `INVALID_ARGUMENT/CLOSE`다. context의 poison/live-child 검사, stream/event의
  context 진입, smoke buffer의 context 진입과 in-flight synchronize처럼
  destructive CUDA API를 호출하기 전의 검사가 실패하면 handle은 non-`NULL`로
  유지된다.
- `cuDevicePrimaryCtxRelease`, `cudaStreamDestroy`, `cudaEventDestroy` 또는
  `cudaFree`를 한 번 시도한 뒤에는 그 결과 status와 관계없이 opaque handle을
  single-shot으로 소비하고 `NULL`로 바꾼다. CUDA API가 실제 side effect 뒤에
  앞선 asynchronous 오류를 반환했는지, genuine native failure로 resource가
  남았는지를 ABI에서 안전하게 구별할 수 없기 때문이다. 전자는 재시도하면
  double release/destroy/free가 될 수 있고, 후자는 native resource 또는 lease를
  안전하게 누수시키더라도 재시도하지 않는다. 파괴 뒤 context-stack 복원이나
  child-counter 검사가 실패한 경우도 non-success와 `NULL`이 함께 반환될 수 있다.
- close가 non-success일 때는 갱신된 handle도 반드시 확인한다. handle이 그대로
  non-`NULL`인 precheck 실패에만 retry 또는 상위 cleanup 정책을 적용하며,
  `NULL`이면 절대 재시도하지 않는다. 모든 명시적 close 오류는 처리하거나
  기록해야 한다. Rust `Drop`은 unwind하지 않는 best-effort fallback일 뿐,
  close/synchronize 오류 관측을 대신하지 않는다.
- stream, event, smoke buffer는 생성에 사용한 context를 참조한다. 모든 child를
  close하고 비동기 작업을 완료하기 전에는 context를 close하지 않는다.
  native context는 live child 수를 추적하며 child가 남은 context close를
  `INVALID_STATE/CLOSE`로 거부한다. 서로 다른 context가 소유한
  stream/event/buffer의 조합도 `INVALID_STATE`다.
- v1 native handle 자체는 임의 alias를 통한 concurrent mutation을 보장하지
  않는다. safe Rust wrapper가 ownership과 공유 수명을 보존하며, raw handle을
  꺼내거나 임의의 `Send`/`Sync`를 가정하지 않는다.

## PR 04 범용 allocation과 copy token

PR 04 symbol은 ABI version 1에 additive하게 추가되었다. 기존 status 숫자,
struct layout, PR 03 symbol 의미는 바꾸지 않는다.

- `device_buffer_create`와 `pinned_host_buffer_create`는 `uint64_t byte_len`을 받는
  untyped byte allocation이다. 0-byte도 non-NULL owning logical handle을 반환하고
  allocation count를 올리지만 CUDA allocation call은 하지 않는다. raw allocation
  pointer를 반환하는 symbol은 없다.
- pinned host CPU access는 `pinned_host_buffer_write/read`만 제공한다. caller slice
  pointer는 synchronous call 동안만 빌리며 offset+length를 overflow 없이 검사한다.
  active copy token이 있으면 빈 access를 포함해 `INVALID_STATE`로 거부한다.
- `copy_h2d_async`/`copy_d2h_async`는 device buffer, pinned buffer, 명시적 non-default
  stream이 같은 opaque context owner인지 확인한다. non-zero copy는 세 resource의
  active-use flag를 예약하고 owning `RustInferCudaCopy` token 하나를 반환한다.
  resource당 동시 copy token은 하나만 허용한다. zero-byte copy는 successful
  no-op이고 token은 NULL이다.
- `cudaMemcpyAsync` 호출을 실제 시도한 뒤 관측한 submission/context-restoration
  오류는 copy token에 owned error로 저장한다. submit ABI는 token을 성공적으로
  넘겨 caller lifetime을 계속 묶고, `copy_query` 또는 `copy_synchronize`가 stream
  완료를 확정한 뒤 저장된 오류를 반환한다. pre-attempt 오류만 output NULL과 함께
  submit에서 즉시 반환한다.
- query/synchronize는 stream operation이 `cudaSuccess`이고 caller thread의 context
  stack 복원도 성공한 경우에만 세 active-use flag를 해제하고 `out_complete=1`로
  commit한다. `NOT_READY`, stream 오류, context restoration 오류에서는 flag와
  token을 그대로 유지한다. completion이 확정된 token은 deferred submission
  status가 있더라도 resource guard를 해제하고 그 status를 반환한다.
- incomplete `copy_close`는 originating stream을 synchronize한다. completion을
  확정하지 못하면 token을 non-NULL로 유지하며 buffer access/free, 새 copy와 stream
  close가 계속 실패한다. token 자체를 raw caller 또는 Rust `mem::forget`으로
  영구 분실하면 이 busy state와 allocation/context accounting도 영구 유지된다.
  이는 UAF나 DMA data race 대신 의도적인 fail-closed leak이다.
- safe Rust `CudaPendingH2D`/`CudaPendingD2H`는 `&mut CudaStream`,
  `&mut CudaDeviceBuffer`, `&mut CudaPinnedHostBuffer`를 실제 보유한다. 정상 lexical
  lifetime에서는 compile-time borrow가 조기 access/close를 막고, forget/unwind
  경로에서는 native active-use state와 Rust busy bit가 같은 규칙을 보강한다.
  buffers와 pending token은 `Send`지만 `!Sync`이며 clone/raw-pointer API가 없다.

Caching allocator, stream-ordered memory pool, unified memory, pageable-host async
copy, model-specific tensor operation은 이 additive 경계의 범위가 아니다.

## PR 09 연속 KV cache와 decode partial-state ABI

PR 09는 ABI version을 올리지 않고 다음 네 symbol과 각 parameter struct를
additive하게 추가한다.

- `rustinfer_cuda_kv_cache_write_execute`
- `rustinfer_cuda_decode_attention_reference_execute`
- `rustinfer_cuda_decode_attention_execute`
- `rustinfer_cuda_decode_partial_state_reduce_execute`

`RustInferCudaKvCacheWriteParams`,
`RustInferCudaDecodeAttentionReferenceParams`,
`RustInferCudaDecodeAttentionParams`,
`RustInferCudaDecodePartialStateReduceParams`의 전체 크기는 64-bit ABI에서 각각
272, 328, 344, 176 bytes다. C11, C++와 Rust의 독립된 compile-time assertion이
크기와 핵심 offset을 고정한다. 모든 입력 `reserved` field는 0이어야 하며 기존
v1 symbol과 struct의 layout이나 의미는 바뀌지 않는다.

### 연속 cache layout과 publish 규칙

K와 V는 서로 다른 BF16 allocation이며 각 allocation의 논리 layout은
`[layer, key_value_head, maximum_token_count, head_size]`다. ABI 호출 하나에는
한 layer view `[key_value_head, maximum_token_count, head_size]`가 전달된다.
projection 결과는 token-major `[source_token_count, key_value_head, head_size]`이고
cache-write symbol이 지정한 token interval로 bit-preserving scatter한다.

cache span은 실제로 읽는 logical prefix보다 큰 전체 fixed-stride capacity를
선언해야 한다. 호출자는 모든 layer의 write와 후속 연산이 성공하기 전에는
logical length를 publish하지 않는다. reset은 logical metadata만 0으로 되돌려도
된다. 다음 prefill이 publish할 prompt prefix를 전부 덮어쓰고 decode가 committed
prefix만 읽으므로 capacity tail의 오래된 byte는 관측할 수 없다.

### `DecodePartialState` version 1

`RUSTINFER_CUDA_DECODE_PARTIAL_STATE_VERSION == 1`인 device storage는 F32 packed
array이며 shape과 offset은 다음과 같다.

```text
shape = [partial_state_capacity, query_head_count, head_size + 2]
offset(partition, query_head) =
    (partition * query_head_count + query_head) * (head_size + 2)
row = [m, l, n[0], ..., n[head_size - 1]]
```

KV logical partition `p`는 고정 partition size `P`에 대해
`[p * P, min(logical_token_count, (p + 1) * P))`를 뜻한다. Storage slot 순서는
cache의 physical 주소나 향후 page allocation 순서가 아니라 이 logical KV 순서를
따른다. PR 10의 paged producer도 page table을 따라 K/V를 읽을 수는 있지만, reducer에
넘기는 slot `p`에는 logical partition `p`의 상태를 써야 한다. 따라서 contiguous와
paged producer는 같은 standalone reducer와 storage ABI를 공유할 수 있다.

한 범위 `C`의 state는 정규화 전 FP32 accumulator다.

```text
m_C = max(score in C)
l_C = sum(exp(score - m_C))
n_C = sum(exp(score - m_C) * value)
```

빈 범위와 fully-masked 범위의 canonical 표현은 `m=-inf`, `l=0`, `n=0`이다.
Reducer는 `l=0` state를 identity로 취급한다. 두 state `A`, `B`는 다음처럼
merge하며 부분 output을 먼저 normalize하지 않는다.

```text
m = max(m_A, m_B)
l = exp(m_A - m) * l_A + exp(m_B - m) * l_B
n = exp(m_A - m) * n_A + exp(m_B - m) * n_B
```

모든 logical state를 ascending 또는 descending 순서로 merge한 뒤 마지막 한 번만
`n/l`을 계산해 BF16 output을 쓴다. `partial_state_count == 0`은 유효하며 zero
output을 쓴다. Capacity와 query/head dimension은 0일 수 없다. 순서가 달라도 실수
수학의 결과는 같지만 FP32 연산 순서 차이는 허용 tolerance로 검증한다.

현재 optimized producer는 `head_size == 64`인 query-length-one MHA/GQA를 지원하고
active partition만 쓴다. Capacity tail은 수정하지 않는다. Standalone reducer는
positive `head_size` 전반을 지원하므로 PR 10 producer가 D64 제한과 독립적으로 같은
ABI를 사용할 수 있다. 이 경로가 "exact"라는 말은 모든 committed KV를 읽고
online-softmax 공식을 적용한다는 뜻이며, warp reduction과 FP32 recurrence가
materialized BF16 reference와 bit-exact하다는 뜻은 아니다.

네 호출은 device allocation을 하지 않고 explicit non-default stream과 모든 distinct
buffer를 exclusive-use로 잡는다. Writable span은 다른 touched span과 겹칠 수 없다.
호출은 stream synchronize와 context-stack 복원까지 성공한 뒤에만 반환하고 active-use
guard를 해제한다. 완료를 확정하지 못하면 기존 copy-token 규칙처럼 UAF 대신
fail-closed busy/accounting leak을 선택한다.

## PR 10 exact paged KV ABI

PR 10은 기존 ABI version과 PR 09 struct를 바꾸지 않고 `U16` dtype discriminant
`5`와 다음 세 symbol을 additive하게 추가한다.

- `rustinfer_cuda_paged_kv_cache_write_execute`
- `rustinfer_cuda_paged_decode_attention_reference_execute`
- `rustinfer_cuda_paged_decode_attention_execute`

`RustInferCudaPagedKvBlockTableV1`, `RustInferCudaPagedKvCacheWriteParams`,
`RustInferCudaPagedDecodeAttentionReferenceParams`,
`RustInferCudaPagedDecodeAttentionParams`의 전체 크기는 64-bit ABI에서 각각
168, 432, 480, 488 bytes다. C11, CUDA C++와 Rust assertion이 크기와 핵심
offset을 함께 고정한다.

### Block table version 1

`RUSTINFER_CUDA_PAGED_KV_BLOCK_TABLE_VERSION == 1`이고 block size는 고정 16이다.
K와 V는 별도 BF16 pool이며 한 layer view의 layout은 다음과 같다.

```text
[physical_block_count, key_value_head_count, 16, head_size]
```

table의 U32 `block_ids[block_count]`는 logical block 순서로 physical block을
지정한다. 따라서 값은 연속일 필요가 없고 allocation/free 결과에 따라 섞여도 된다.
U16 `valid_tokens[block_count]`는 마지막 block만 1..16일 수 있으며 그 앞 block은
항상 16이다. `block_count == ceil(logical_token_count / 16)`이고 각 ID는
`physical_block_count`보다 작아야 한다.

safe Rust wrapper는 device array와 함께 immutable host mirror를 요구한다.
`PagedKvBlockTableHostV1` 생성 때 version, 두 배열 길이, valid-token 값,
out-of-pool ID, duplicate ID를 allocation 없이 검사한다. prepared runtime은 cold-path에서
physical-block당 1 byte scratch를 만들고 매 실행의 host mirror를 O(physical blocks +
logical blocks)에 검증한다. 이후 layer별 append/decode는 이 validated type invariant를
신뢰해 검사를 반복하지 않는다. generation/cookie stale-handle 검증, host/device mirror의
upload 완료와 lifetime은 상위 runtime block manager가 별도로 소유한다. Raw C caller가
잘못된 device content를 전달해도 kernel은
pool 밖을 역참조하지 않지만, 성공적인 의미 결과를 얻으려면 같은 table invariant를
외부에서 지켜야 한다.

`metadata_kind == RUSTINFER_CUDA_PAGED_KV_METADATA_NONE`이고
`metadata_version == 0`만 v1 exact path에서 허용한다. 즉 sidecar가 없는 경로가
기본이며 주소 변환 외 metadata branch가 없다. 알 수 없는 kind/version은 launch 전
`NOT_SUPPORTED/VALIDATION`으로 거부한다. 이후 page 통계나 offload metadata는 이
주소 table layout을 바꾸지 않는 별도 versioned capability로 추가한다.

### Scatter와 exact attention

Paged cache write는 RoPE 이후 token-major BF16 `[T,KVH,D]` K/V를 logical
destination interval에 bit-preserving scatter한다. table은 write 완료 뒤 publish할
logical length를 기술하며, 모든 layer가 성공하기 전 SequenceState의 committed
length를 갱신하지 않는다.

Paged materialized reference는 PR 09와 같은 QK, scale, stable softmax, AV 네
stage와 BF16 staging 지점을 유지하되 모든 K/V access를 block table로 변환한다.
따라서 동일 logical K/V와 query에 대해 contiguous reference와 BF16 bit parity가
correctness oracle이다.

D64 optimized producer는 logical block 하나마다 packed F32
`DecodePartialState(m,l,n[D])` 하나를 쓴다. State slot은 physical ID가 아니라
logical block ordinal이다. 그 뒤 PR 09 standalone reducer를 그대로 사용해 logical
순서로 merge하고 마지막 한 번만 normalize한다. 모든 page를 읽으므로 exact이며
page selection/pruning은 하지 않는다. Materialized reference와 FP32 online recurrence
사이에는 기존 허용 오차가 적용되고 bit-exact를 주장하지 않는다. Reference와 online
호출은 각각 kernel 4개와 2개이며 device/host heap allocation 없이 caller가 미리 잡은
workspace를 사용한다.

context 생성은 target device의 CUDA **primary context에 대한 공유 lease**를
`cuDevicePrimaryCtxRetain`으로 얻는다. 프로세스에 독점 context를 만들거나
primary context를 reset하지 않는다. close는 해당 lease의 release를 정확히 한
번만 시도한다. 같은 ordinal에서 context를 여러 번 생성하면 underlying
`CUcontext`는 공유하지만 각 retained lease와 opaque owner identity는 별개다.
따라서 서로 다른 owner에서 만든 stream/event/buffer 조합은 underlying
`CUcontext`가 같아도 `INVALID_STATE`로 거부된다.

context-dependent call은 먼저 `cuCtxGetCurrent`로 호출 thread의 current context를
snapshot한다. target primary context가 이미 current이면 caller의 stack을
건드리지 않고 그 context를 잠시 borrow한다. 그렇지 않으면 target을 push하고
호출 종료 전에 자신이 만든 pop debt를 pop하여 이전 current context를 복원한다.
process-global implicit current context나 `cudaSetDevice` 상태를 ownership 근거로
사용하지 않는다.

snapshot/push/pop의 결과나 pop된 context identity가 모호하면 context owner에
monotonic restoration-failed poison을 기록한다. push가 오류를 반환했더라도
target이 current가 된 것이 확인되면 pop debt를 끝까지 복원한다. 복원을 확정할
수 없으면 이후 context 진입과 primary-lease close를 `INVALID_STATE`로 거부하고
wrapper와 lease를 보존한다. 이 catastrophic path는 불확실한 current context를
release해 stale context를 만들기보다 fail-closed leak을 선택한다.

primary context가 공유되므로 `CudaContext::synchronize`의
`cudaDeviceSynchronize`는 같은 device의 다른 retained lease나 다른 library가
enqueue한 작업에도 영향을 줄 수 있다. unfinished `CudaPendingFill`의 `Drop`도
먼저 launch stream을 synchronize하고, buffer fallback close가 필요하면
context-wide device synchronize까지 수행할 수 있다. 독립적인 정상 완료 관측은
가능하면 명시적 stream/event synchronize와 `CudaPendingFill::finish`를 사용한다.

safe Rust wrapper의 thread/lifetime 계약은 다음과 같다.

- `CudaRuntime`, `CudaDevice`, `DeviceProperties`는 owned/cached host metadata다.
- `CudaContext`는 `Send + Sync`인 primary-context lease다. child는 내부 `Arc`를
  보유하므로 context native handle이 먼저 파괴될 수 없다. `CudaContext::close`
  는 자신을 소비하며 child 또는 공유 reference가 남아 있으면 Rust-side
  `InvalidState/Validation`을 반환한다.
- `CudaStream`과 `CudaEvent`는 host thread 사이로 이동할 수 있는 `Send`지만
  의도적으로 `!Sync`다. ordering/state를 바꾸는 operation은 `&mut self`를
  요구한다. C ABI의 `NOT_READY`는 두 safe `query` method에서 오류가 아닌
  `Ok(false)`로 변환되고, 완료는 `Ok(true)`다.
- `CudaKernel`은 context에 묶인 immutable AOT smoke-kernel handle이며
  `Send + Sync`다. 범용 module/kernel loader가 아니다.
- `CudaPendingFill`은 launch stream을 완료까지 exclusive borrow한다. `finish`가
  synchronize, host copy, explicit buffer close의 오류를 반환하는 정상 경로다.
  unfinished value의 `Drop`은 stream synchronize와 buffer close를 best-effort로
  수행하지만 그 오류를 호출자에게 보고할 수 없다. borrowed `CudaStream`과 함께
  host thread 사이로 이동할 수 있는 `Send`지만 shared mutation을 허용하는
  `Sync`는 아니다.
- `CudaDeviceBuffer`와 `CudaPinnedHostBuffer`는 clone/raw-pointer가 없는 owning
  opaque byte buffer다. 둘 다 `Send + !Sync`이며 close와 copy는 exclusive borrow를
  요구한다. `CudaPinnedHostBuffer`의 read/write/to_vec도 active Rust/native token을
  확인한다.
- `CudaPendingH2D`와 `CudaPendingD2H`는 originating stream, device buffer, pinned
  buffer를 모두 exclusive borrow하는 `Send + !Sync` completion owner다. 명시적
  `synchronize`가 오류 보고 경로이고 Drop은 best-effort다. Drop completion을
  확정하지 못하거나 값을 forget하면 busy/accounting을 해제하지 않는다.

## stream, event와 비동기 완료

- 모든 stream은 `cudaStreamNonBlocking`으로 명시 생성한다. null/default stream을
  API로 노출하지 않으며 legacy default-stream ordering에 의존하지 않는다.
- event는 timing-enabled 상태로 생성된다. ordering은 `event_record`와
  `stream_wait_event`로 명시하고 elapsed time은 같은 context의 두 event에만
  요청한다.
- launch 함수의 성공은 enqueue와 즉시 launch 검사가 성공했다는 뜻이지 device
  실행 완료를 뜻하지 않는다. 실행 중 발생한 late/asynchronous 오류는
  `stream_synchronize`, `event_synchronize` 또는 `context_synchronize`에서
  `SYNCHRONIZE` stage로 관측될 수 있다.
- `smoke_copy_to_host`는 host output capacity를 element 단위로 검증하고, launch에
  사용한 non-default stream의 완료를 먼저 확인한 뒤 synchronous device-to-host
  copy를 수행한다. 따라서 caller-owned host pointer를 ABI 반환 뒤 보유하지 않으며
  완료된 결과만 반환한다. 범용 allocation/copy API가 아니라 PR 03 진단용
  buffer에만 적용된다.
- resource를 닫기 전 사용 중인 stream을 동기화하고, event/buffer/stream을
  context보다 먼저 닫는다. destructor가 암묵적으로 모든 비동기 오류를
  보고한다고 가정하지 않는다.

PR 03의 7-test GPU smoke는 정상 fill의 명시적 완료와 context를 poison하지 않는
invalid-launch 진단만 사용한다. illegal access나 device assert 같은 late fault는
공유 primary context를 poison하여 다른 lease/library의 후속 synchronize와 close,
resource-leak 관측, compute-sanitizer zero-error gate까지 오염시킬 수 있다. 그런
fault injection은 같은 프로세스의 lifecycle smoke에 섞지 않고, 별도 process와
복구 정책을 갖춘 [PR 16 오류 격리 gate](../deploy/16-reliability-and-release.md#오류-격리)에서
검증한다.

## 오류와 언어 경계

native 함수는 null pointer, struct size, device ordinal, element-count 산술과
context ownership을 사용 전에 검사한다. CUDA API 오류는 즉시 stable status로
mapping되고 원래 code와 CUDA 문자열은 caller-owned error buffer에 복사된다.
kernel enqueue 직후의 launch 오류는 `LAUNCH`, 완료 시 드러난 오류는
`SYNCHRONIZE`로 구분한다. C++ exception과 Rust panic은 이 ABI를 통과하지
않는다.

Rust API는 native error buffer를 호출 중에만 빌려 쓰고 반환 전에 owned
`CudaError`로 복사한다. `CudaErrorKind`는 stable high-level 분류이며
`CudaErrorDomain`, `CudaErrorStage`, `native_code`, `operation`, `message`가
진단 context를 보존한다. `cuda` feature가 꺼진 build에서 초기화 시도는 native
symbol을 찾거나 dynamic loading을 시도하지 않고 `Unavailable/Rust/Initialize`
오류와 `cuda` feature를 켜라는 진단을 반환한다.

## PR 03 진단 경계와 PR 04 확장

PR 03의 allocation과 kernel은 lifecycle 및 error propagation을 검증하는
`RustInferCudaSmokeBuffer`와 fill smoke operation으로 제한한다. PR 04는 opaque
untyped allocation, pinned staging, copy token과 accounting만 additive하게 더한다.
Tensor shape/layout/view metadata는 Rust tensor crate가 맡으며 C ABI에 tensor
object나 raw pointer를 추가하지 않는다. allocator pool, unified memory, model
loading/operation, cuBLASLt, CUTLASS, NVRTC, Triton, CUDA Graph도 아직 범위 밖이다.
