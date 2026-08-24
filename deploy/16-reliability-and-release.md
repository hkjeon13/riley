# PR 16 — 신뢰성, Soak Test, 첫 Release Gate

**상태:** Planned  
**선행 조건:** [PR 15](15-profiling-and-optimization.md)  
**다음:** [PR 17 — 확장 Gate](17-extension-gates.md)

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)

## 목적

짧은 benchmark가 아니라 장시간 서비스에서 메모리, 오류, 취소, 과부하와 성능 회귀를 검증해 첫 release candidate를 만든다.

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

## 검증 항목

### 메모리

- host RSS 추이
- VRAM high-water mark
- free block 복귀
- fragmentation
- pinned memory
- metadata leak

### 동시성

- deadlock/livelock
- channel backlog
- scheduler starvation
- shutdown race
- duplicate finish event

### 오류 격리

한 요청의 오류가 다음을 유발하지 않아야 한다.

- process abort
- 다른 요청 cache 손상
- stream 영구 오류
- block pool 불일치
- 민감 정보 로그 노출

## Release artifact

- supported GPU/CUDA matrix
- supported model family와 제약
- build instructions
- benchmark report
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

noise보다 작은 threshold를 두지 않는다. 환경 편차를 먼저 측정한다.

## 비범위

- 새로운 모델 family
- quantization
- MoE
- multi-GPU
- 기능 추가를 통한 benchmark 개선

## 완료 기준

- [ ] 정해진 soak scenario 통과
- [ ] host/VRAM 누수 징후 없음
- [ ] overload와 cancellation 안정
- [ ] release build 재현 가능
- [ ] supported/unsupported 범위 명확
- [ ] rollback 절차 검증
- [ ] Gate E 승인 후 첫 tag 생성 가능

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)
