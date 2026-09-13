# PR 10 — KV export/import와 공유 prefix 소유권

상태: **구현 진행 중 — host read lease 기반 추가, serving 미연결**. 공통 계약은 [README](README.md)를 따른다.

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

남은 필수 범위는 identity descriptor, shared-prefix sequence import, partial-page COW, CUDA completion과 transport ticket 연결, local adapter, 모델/serving 연결 및 cache-hit/cache-miss 비교다. 이 기반 작업으로 PR10 완료나 serving 성능 개선을 주장하지 않는다. 기본 serving 경로에는 lease를 발급하는 호출을 아직 넣지 않았다.

Host 검증: `cargo test -p riley-runtime --lib` 334 passed / 1 ignored / 0 failed. 이후 overflow/stale-completion 원자성 테스트를 추가한 최종 `cargo test -p riley-runtime --lib paged_kv --quiet`는 23 passed / 0 failed. `cargo test -p riley-scheduler --lib --quiet`는 48 passed / 0 failed. 새 테스트는 lease 회수 지연·중복 완료·foreign pool·partial/full-page append·truncate/reset·generation 재사용·overflow 실패 원자성을 확인한다. 이 단계에서는 새 GPU/serving 측정을 실행하지 않았다.
