# PR 15 — Profiling, Fusion, CUDA Graph 최적화

**상태:** Planned  
**선행 조건:** [PR 14](14-api-and-streaming.md)  
**다음:** [PR 16 — 신뢰성과 Release](16-reliability-and-release.md)

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)

## 목적

작동하는 서비스의 trace를 기준으로 가장 큰 병목 하나씩만 최적화한다. 이 문서는 하나의 거대 PR을 의미하지 않으며, 아래 항목은 각각 독립 PR 후보이다.

## 먼저 측정할 것

- scheduler CPU time
- GPU idle interval
- kernel launch count
- host↔device copy
- transpose/contiguous copy
- allocation 횟수
- prefill attention
- decode attention
- RMSNorm/elementwise launch
- logits/sampling
- CUDA API synchronization

## 최적화 후보 우선순위

### 후보 A — Residual + RMSNorm

조건:

- norm/elementwise launch가 TPOT에서 유의미
- reference parity 확보
- register pressure가 허용 범위

### 후보 B — RoPE + KV write

조건:

- 중간 Q/K tensor write 또는 layout conversion이 병목
- standard/partial RoPE capability를 구분
- paged block offset 정확성 검증

### 후보 C — GPU Sampling

조건:

- logits copy와 CPU sampling이 작은 batch latency에서 유의미
- deterministic RNG contract 유지

### 후보 D — CUDA Graph decode fast path

조건:

- batch/shape bucket이 안정적
- graph memory reservation 대비 KV capacity 손실 측정
- fallback eager path 유지

### 후보 E — CPU/GPU overlap

- iteration N GPU 실행 중 N+1 batch metadata 준비
- async block table copy
- unnecessary sync 제거

## 각 최적화 PR의 필수 구조

1. profiler evidence
2. 가설
3. reference implementation
4. optimized implementation
5. correctness 결과
6. microbenchmark
7. end-to-end 결과
8. regression range
9. runtime flag와 rollback

## 금지

- profiler 없이 fusion 선택
- 여러 fusion을 한 PR에서 동시 적용
- 평균만 보고 p95/p99 악화 무시
- throughput 향상을 위해 TTFT 목표를 암묵적으로 변경
- graph capture 때문에 지원 shape를 조용히 축소

## 완료 기준

이 단계 전체의 종료 조건:

- [ ] target workload의 top bottleneck이 설명됨
- [ ] 적용한 각 최적화가 end-to-end에서 유효
- [ ] performance regression suite 존재
- [ ] fallback path parity 유지
- [ ] 최적화별 memory trade-off 기록
- [ ] baseline engine과 동일 조건 비교 갱신

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)
