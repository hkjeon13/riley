# PR 16 — 신뢰성, Soak Test, 첫 Release Gate

**상태:** Planned  
**선행 조건:** [PR 15](15-profiling-and-optimization.md)  
**다음:** [PR 17 — 확장 Gate](17-extension-gates.md)

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)

## 목적

짧은 benchmark가 아니라 장시간 서비스에서 메모리, 오류, 취소, 과부하와 성능 회귀를 검증해 첫 release candidate를 만든다. 최적화의 의미 보존 등급별로 기본값, 표기, fallback과 운영 안전성을 확인한다.

## 첫 Release의 의미 정책

첫 stable release의 기본 경로는 다음만 허용한다.

- reference path
- 검증을 완료한 `E0` exact optimization

다음은 기본 비활성 또는 release 범위 밖이다.

- `E1` distribution-preserving experimental path
- `A1` approximate path
- `M1` model-changing path

`E1` 또는 `A1`을 preview로 포함한다면 다음이 필요하다.

- 명시적인 feature flag
- exact fallback
- request/run metadata에 사용 여부 기록
- 별도 benchmark와 correctness/quality report
- 운영 중 즉시 비활성화 가능한 configuration

## 신뢰성 시나리오

- 장시간 steady load
- burst 후 idle 반복
- 짧은 요청과 긴 요청 혼합
- 높은 cancellation 비율
- client disconnect
- 잘못된 요청 반복
- KV capacity 근접
- model load/unload 반복
- server graceful shutdown/restart
- CUDA 오류 주입 가능한 범위
- exact backend 간 runtime switch
- experimental flag on/off 반복
- fallback을 유발하는 unsupported shape/metadata

## 검증 항목

### 메모리

- host RSS 추이
- VRAM high-water mark
- free block 복귀
- fragmentation
- pinned memory
- metadata leak
- optional KV sidecar lifetime
- fallback 전환 후 workspace 회수

### 동시성

- deadlock/livelock
- channel backlog
- scheduler starvation
- shutdown race
- duplicate finish event
- RNG state가 다른 request 사이에서 공유되지 않음
- backend fallback 중 stream ordering 유지

### 오류 격리

한 요청의 오류가 다음을 유발하지 않아야 한다.

- process abort
- 다른 요청 cache 손상
- 다른 요청 RNG 오염
- stream 영구 오류
- block pool 불일치
- stale optional metadata 재사용
- 민감 정보 로그 노출

### 의미 보존 등급별 검증

#### `E0`

- reference와 장시간 token-level 회귀
- 여러 partition, graph bucket, backend fallback
- extreme logits와 긴 context

#### `E1` preview

- acceptance와 residual correction invariant
- request-local RNG snapshot/restore
- distribution 검증 artifact
- target/draft 오류 시 exact target path 복귀

#### `A1` preview

- error budget configuration validation
- exact fallback 비율
- approximation 사용 여부 metric
- epsilon 또는 rank/bit-width별 quality regression
- 잘못되거나 stale한 summary에서 exact path 전환

## Release artifact

- supported GPU/CUDA matrix
- supported model family와 제약
- build instructions
- benchmark report
- semantic class별 feature table
- default enable/disable 상태
- approximation/error-budget configuration
- known limitations
- configuration reference
- operational metrics
- upgrade/rollback guide

## 성능 회귀 gate

허용 threshold는 PR 01 기준선에 상대적으로 정의한다.

예:

- TTFT p50/p95
- TPOT p50/p95
- throughput
- peak VRAM
- CPU utilization
- exact fallback overhead
- optional metadata overhead

noise보다 작은 threshold를 두지 않는다. 환경 편차를 먼저 측정한다.

## 비범위

- 새로운 모델 family
- quantization stable enablement
- MoE
- multi-GPU
- speculative decoding stable enablement
- approximate attention의 기본 활성화
- 기능 추가를 통한 benchmark 개선

## 완료 기준

- [ ] 정해진 soak scenario 통과
- [ ] host/VRAM 누수 징후 없음
- [ ] overload와 cancellation 안정
- [ ] release build 재현 가능
- [ ] stable 기본 경로가 reference 또는 검증된 `E0`로 제한됨
- [ ] experimental feature의 flag, 표기, exact fallback 검증
- [ ] supported/unsupported 범위 명확
- [ ] rollback 절차 검증
- [ ] Gate E 승인 후 첫 tag 생성 가능

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)