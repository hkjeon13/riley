# PR 10 — KV export/import와 공유 prefix 소유권

상태: **구현 진행 중 — host identity·공유 import·COW, local CUDA event/D2D 및 공유 batch/wire/native 구조 검증 완료, captured model/serving 미연결**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

GPU·worker 사이 KV 이동과 공유를 지원하되 stale page 또는 진행 중인 page 재사용을 막는다.

## 의존성과 변경 위치

선행: 01; 02의 ticket 수명과 호환.

예상 수정 위치: riley KV manager, runtime KV layout/identity, transport adapter. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. model/revision/dtype/layout/position 정보를 포함한 KV descriptor를 정의한다.
2. immutable prefix page의 참조·generation과 copy-on-write append를 연결한다.
3. 비동기 export/import 및 완료 event 기반 ownership 이전을 구현한다.
4. 하나의 transport adapter와 검증용 local transport를 연결한다.

## 범위 경계

라우터·autoscaler·SSD 계층 전체를 구현하지 않는다. 압축 KV는 별도 PR이다.

## Correctness·수명 계약

수신 완료와 identity 검증 이전에 page를 decode에 노출하지 않는다. 공유 page는 마지막 consumer와 transfer가 끝난 뒤 반환한다. cache key는 서로 다른 모델/position 표현을 혼합하지 않는다.

## 검증과 하드웨어 skip

부분 transfer 실패·중복/늦은 완료·cancel·prefix append COW·layout mismatch. 4090 local 경로 검증; peer/network 요구 테스트는 장비 부재 시 skip. 실제 transport 없이 network 성능을 주장하지 않는다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

full-model 결과와 page 수명 검증 통과. cache-hit workload의 TTFT 이득뿐 아니라 cache-miss overhead·메모리 상한을 보고한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

cache/transfer를 끄고 local KV를 사용한다. in-flight transfer drain 이후 cache 참조 해제.

## 연구 근거

[Mooncake](https://arxiv.org/abs/2407.00079), [NIXL](https://github.com/ai-dynamo/nixl), [RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/). 논문 성능 배수는 Riley의 예상 개선율이 아니다.

## 착수 방향 — 2026-09-14

[Adaptive projection 결과](../../benchmarks/results/20260914-adaptive-decode-serving/README.md)는 낮은 concurrency에서만 약4% 개선되고 고부하 격차는 남는다. 다음 구조 영역으로 본 PR을 선택한다. 현재 `paged_kv.rs`의 block 소유권은 sequence 단위이므로, frontend 응답 재사용으로 우회하지 않고 descriptor identity·immutable page 참조/generation·COW·완료 후 publication을 기존 reservation/commit/reclaim과 함께 연결해야 한다.

현재 serving 기준은 vLLM prefix caching이 명시적으로 꺼져 있다. 새 cache-hit 및 cache-miss 실험에서는 양쪽 caching 설정, KV memory 상한과 workload를 맞추고 기존 cache-off 결과를 별도 유지한다. 세 개 반복 prompt만으로 일반화하지 않고 공유 prefix·고유 suffix 및 고유 prompt를 포함한다. Network/peer 성능은 실제 transport 및 장비 검증 전에는 주장하지 않는다.

## 재개: host page 수명 기반

`KvBlockPool::lease_prefix`는 pending/poison 상태를 거부하고, committed prefix의 generation-bound page에 non-cloneable `KvReadLease`를 발급한다. Sequence close/reset/orphan reclaim은 sequence 소유권을 해제하되 마지막 read lease가 끝나기 전까지 page와 sidecar를 반환하지 않는다. Allocated count에는 이렇게 보류된 page도 포함된다. 중복 lease 해제는 no-op이며 다른 pool에 대한 해제 실패는 원래 token을 보존한다.

Lease가 있는 page에 쓰는 append는 `ImmutableBlock`으로 거부한다. Full-page prefix 뒤 새 page append는 허용하며, truncate로 partial page가 된 경우에도 보호한다. 아직 COW를 실행하는 기능은 아니다. 이 host token 자체는 CUDA event가 아니므로 acquisition은 device write 완료 후, release는 모든 reader의 device 작업 완료 후 호출해야 한다. Token을 버려도 자동 해제하지 않아 미완료 reader의 page가 재사용되지 않는다.

이 read lease 기반 작업만으로 PR10 완료나 serving 성능 개선을 주장하지 않는다. 이후 identity·공유 import·COW 진행 상황은 아래에 기록한다. 기본 serving 경로에는 lease를 발급하는 호출을 아직 넣지 않았다.

Host 검증: `cargo test -p riley-runtime --lib` 334 passed / 1 ignored / 0 failed. 이후 overflow/stale-completion 원자성 테스트를 추가한 최종 `cargo test -p riley-runtime --lib paged_kv --quiet`는 23 passed / 0 failed. `cargo test -p riley-scheduler --lib --quiet`는 48 passed / 0 failed. 새 테스트는 lease 회수 지연·중복 완료·foreign pool·partial/full-page append·truncate/reset·generation 재사용·overflow 실패 원자성을 확인한다. 이 단계에서는 새 GPU/serving 측정을 실행하지 않았다.

## 다음 구현 batch: identity·공유 import·COW

`crates/riley-runtime/src/paged_kv/prefix.rs`에 아래 host 계약을 연결했다.

1. `PrefixDescriptor`: model revision/content, numerical profile, position encoding/RoPE, tensor/pipeline partition fingerprint 및 BF16 page format을 token digest·시작 position·길이와 함께 대조한다. 누락된 fingerprint, 빈 prefix, position overflow를 거부한다. Pool physical capacity는 page 표현 identity와 분리한다. 실제 loaded model 및 sequence token과의 binding은 앞으로 실행 adapter가 공급해야 한다.
2. `SequenceState::import_prefix`: 검증된 export의 **같은 physical page**를 빈 sequence에 연결한다. Pool은 각 page의 sequence owner들을 추적하고 source 종료·cache eviction·consumer 종료 순서와 관계없이 마지막 owner/read lease 이후에만 반환한다. 공유 owner의 추가 host allocation capacity는 pool metadata 통계에 포함한다.
3. `begin_copy_on_write`/`complete_copy_on_write`: partial tail의 source lease와 별도 staging sequence를 ticket이 보유한다. 복사 중 target table을 publish하지 않는다. 성공 완료 후에만 마지막 page의 ownership/table을 교체한다. 실패, reset 이후 지연 완료, 새 reservation, orphan 취소 및 중복 완료를 구별한다. Enqueue 성공이나 cancel 요청을 copy 완료로 취급해서는 안 된다.
4. `copy_regions`: 모든 layer/head의 유효 token만 복사하는 K/V byte region을 allocation 없이 열거한다. Host buffer 검증에서 별도 dense-layout oracle로 주소와 bytes를 확인하고 unused tail guard를 보존했다. **이 iterator는 아직 CUDA transport adapter가 아니다.**

검증: 최종 `cargo test -p riley-runtime --lib paged_kv --quiet` 31 passed; `cargo test -p riley-scheduler --lib --quiet` 48 passed; `cargo test -p riley-runtime --lib llama::batch:: --quiet` 8 passed. 초기 offset API 인자 수 컴파일 오류는 수정 후 재실행했다. 이 단계에서 GPU 실행·full-model 재검증·serving 성능 측정은 하지 않았다.

Batch 연결 방향: 기존 cross-sequence page alias 일괄 거부를 단순히 제거하지 않고 authoritative shared-owner ledger와 committed read-only range로 읽기 공유와 쓰기 충돌을 구별한다. 이 검증의 구현 결과는 아래에 기록한다. Scheduler cache lookup/publication/eviction 및 model identity 연결도 남아 있으며, 아직 serving 승격 대상이 아니다.

## Local CUDA 전송 검증

이후 `CudaPendingCow` → `CudaPendingKvCopy` → C ABI의 실제 D2D/event 경로를 연결했다. Enqueue 이후 실패도 native token이 양쪽 device buffer·stream·context 수명을 보유하며, 실제 event 완료와 context restoration 확인 전에는 page를 publish/reclaim하지 않는다. 실패는 재시도해도 유지하고, 취소는 완료 이후 discard한다. 상세 결과는 [local CUDA COW gate](../../benchmarks/results/20260914-kv-prefix-cuda/README.md)에 보존했다.

4090에서 4 GPU tests, 기존 H2D/D2H 7 regressions, Memcheck 0 errors/0 leaks(정상·취소·부분 실패 8 cases)를 확인했다. 완료 불명확 fault는 별도 child에서 source/staging 2 pages와 native holds를 보존하는 것으로 검증했다. Trace는 D2D 85 operations/6,736 bytes 및 event 생성·record·파괴 각 8회를 확인한다. 이 수치는 serving benchmark가 아니다.

Nsight 원본은 환경 정보를 담을 수 있으므로 저장소에 넣지 않는다. Runner는 최소 환경으로 profile하고 원본을 원격 private directory에 보관하며, 공개 evidence에는 numeric CUDA activity와 public API 이름만 새 DB로 추출한 SQLite를 넣는다. 이 경로를 포함한 최종 v4 gate가 통과했다.

현재 local adapter는 idle K/V buffer용이다. Captured graph는 `RileyCudaGraphResources`가 parent의 active-use를 장기 보유하므로 그 ledger의 권한·in-flight 상태와 결합한 전송 seam이 추가로 필요하다. 공유 prefix를 일반 serving에 켜기 전에 위 batch/native alias 검사, 모델 identity/정확도, scheduler cache 정책과 함께 연결해야 한다. 아직 실제 model/serving 개선이나 PR10 승격을 주장하지 않는다.

## 공유 batch/wire/native 검증

명시적 shared-prefix metadata 설정에서 committed page의 같은 logical position 읽기만 허용하고, append 겹침·잘못된 위치·중복 owner pair를 거부한다. Wire 검사는 batch 밖 consumer를 포함한 authoritative ledger로 공유 page 쓰기도 거부한다. 실제 pool export/import/reservation을 이용한 host 검증도 통과했다. Native V7에는 기본값 false인 capability를 추가했으며 현재 captured graph 호출은 계속 false다. Packet 자체로 capability나 ownership을 주장할 수 없다.

[검증 기록](../../benchmarks/results/20260914-shared-prefix-validation/README.md): runtime 347 passed / 1 기존 timing diagnostic ignored, wire 18 passed, batch 11 passed, scheduler 48 passed. Linux ASan/UBSan에서 정상·거부 사례 및 61,568 byte × 2 mode mutation traversal을 통과했다. GPU/model/serving 측정은 이 단계에서 실행하지 않았다.

다음 통합 단위는 retained model owner의 capability와 전체 owner ledger 공급, captured K/V COW 권한·drain, scheduler cache lookup/publication/eviction 및 full-model parity다. 이 serving 연결이 완료되면 동일 조건의 cache-hit/cache-miss 및 cache-off regression을 vLLM과 비교하여 표로 보고한다. 검증 helper 단계마다 serving benchmark를 반복하지 않는다.

## Scheduler 실행 authority 연결

일반 iteration과 paired decode window의 plan/authority에 실제 공유 sequence를 연결했다. Scheduler 내부 plan은 같은 위치의 committed page 읽기를 허용하며 쓰기 alias는 거부한다. Pool의 전체 generation/owner 검사와 append 범위의 lease/shared-owner 재검사를 dispatch authority에 연결했다. 두 실행 경로의 owner ledger를 통합하고 physical ID 중복 여부를 누적 목록에서 반복 검색하던 경로를 제거했다. 예약 capacity는 physical page 수 대신 전체 sequence owner/page 수를 사용한다.

실제 export/import 후 두 consumer 실행과 off-batch consumer, NotDispatched 재시도, paired future-token authority, 취소 후 전체 회수를 검증했다. [검증 기록](../../benchmarks/results/20260914-shared-prefix-scheduler/README.md): scheduler 전체 150 passed, runtime library 348 passed / 1 기존 timing diagnostic ignored. 이 테스트의 출력 token은 host fixture이며 model 정확도 증거가 아니다.

현재 ledger는 request-owned sequence를 포함한다. Scheduler cache-only owner가 도입될 때 해당 owner도 추가해야 한다. 자동 cache 정책과 native model capability는 아직 연결하지 않았으므로 serving에서 prefix reuse가 켜진 상태는 아니다. 다음 단계는 위 retained-model/COW/cache 통합과 실제 GPU·serving 검증으로 유지한다.
