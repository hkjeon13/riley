# PR 15 — Profiling, Fusion, CUDA Graph 최적화

**상태:** Planned  
**선행 조건:** [PR 14](14-api-and-streaming.md)  
**다음:** [PR 16 — 신뢰성과 Release](16-reliability-and-release.md)

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)

## 목적

작동하는 서비스의 trace를 기준으로 가장 큰 병목 하나씩만 최적화한다. 이 문서는 하나의 거대 PR을 의미하지 않으며, 아래 항목은 각각 독립 PR 후보이다.

모든 후보는 [PR 00](00-pr-contract.md)의 의미 보존 등급을 선언한다. `E0`과 `A1`을 같은 PR에서 섞지 않는다.

## 먼저 측정할 것

- scheduler CPU time
- GPU idle interval
- kernel launch count
- host↔device copy
- transpose/contiguous copy
- allocation 횟수
- prefill attention
- decode attention
- score matrix materialization과 HBM traffic
- RMSNorm/elementwise launch
- logits/sampling
- CUDA API synchronization
- context 길이별 KV bytes read
- page 또는 chunk partial-state merge overhead

## 최적화 후보 우선순위

### 후보 A — Residual + RMSNorm (`E0`)

조건:

- norm/elementwise launch가 TPOT에서 유의미
- reference parity 확보
- FP32 reduction semantics 유지
- register pressure가 허용 범위

검증:

- standalone residual과 norm 결과 비교
- hidden size와 batch/token shape별 tolerance
- multi-step greedy token 회귀

### 후보 B — RoPE + KV write (`E0`)

조건:

- 중간 Q/K tensor write 또는 layout conversion이 병목
- standard/partial RoPE capability를 구분
- paged block offset 정확성 검증
- dynamic/multimodal RoPE는 unsupported 또는 별도 path로 명시

### 후보 C — GPU Sampling (`E0` 또는 sampling contract 보존)

조건:

- logits copy와 CPU sampling이 작은 batch latency에서 유의미
- PR 11의 logits processing 순서 유지
- request별 deterministic RNG contract 유지
- CPU reference와 probability 및 token 결과 검증

### 후보 D — CUDA Graph decode fast path (`E0`)

조건:

- batch/shape bucket이 안정적
- graph memory reservation 대비 KV capacity 손실 측정
- fallback eager path 유지
- cancellation과 dynamic block table update 방식 명확

### 후보 E — CPU/GPU overlap (`E0`)

- iteration N GPU 실행 중 N+1 batch metadata 준비
- async block table copy
- pinned metadata buffer reuse
- unnecessary sync 제거
- event dependency를 trace로 검증

### 후보 F — Online softmax split-range 최적화 (`E0`)

PR 08·09에서 정의한 `(max, exp_sum, weighted_value_sum)`을 실제 prefill/decode backend의 병렬 분할에 적용한다.

조건:

- score matrix materialization 또는 긴 KV serial scan이 병목
- partition 수 증가가 merge overhead보다 이득
- all-masked partition과 extreme score가 검증됨
- partition order 변화가 dtype별 tolerance 내에 있음

측정:

- partition 수별 latency
- partial-state workspace
- HBM traffic
- occupancy
- end-to-end TTFT/TPOT

### 후보 G — Query-aware KV page pruning (`A1`, experimental)

이 후보는 긴 context decode에서 **KV page read bandwidth가 실제 top bottleneck**일 때만 검토한다. 첫 release의 필수 항목이 아니며 기본 비활성이다.

#### Page summary

PR 10의 optional sidecar에 page별 key dimension의 최소·최대값을 저장할 수 있다.

현재 query `q`에 대해 page `B`의 모든 key score 상한을 다음처럼 계산한다.

```text
U_B = sum over j:
      q_j * key_max[B, j]  if q_j >= 0
      q_j * key_min[B, j]  if q_j < 0
```

그러면 page 안의 모든 key `k`에 대해 `dot(q, k) <= U_B`가 성립한다. head/group, scale, positional transform과 quantization을 반영한 동일한 좌표계에서 summary를 계산해야 한다.

#### Omitted softmax mass bound

선택한 page는 exact partial state로 처리하고, 아직 읽지 않은 page의 최대 가능 softmax 질량을 상한으로 계산한다.

```text
missing_mass_upper
  = sum over unloaded pages B:
      valid_tokens[B] * exp(U_B - current_max)

missing_probability_upper
  = missing_mass_upper / (seen_exp_sum + missing_mass_upper)
```

`missing_probability_upper <= epsilon`일 때만 중단한다.

이 값은 누락된 attention probability mass의 상한이다. attention output의 norm 오차까지 보증하려면 page별 value norm upper bound를 추가하여 별도 output bound를 계산해야 한다.

#### 구현 순서

1. exact full-page scan reference 유지
2. page summary 생성과 stale metadata 검증
3. query별 page upper-bound 계산
4. 가장 가능성이 큰 page부터 exact scan
5. omitted mass bound 계산
6. epsilon 만족 시 중단, 아니면 계속 scan
7. unsupported mask/layout/metadata는 exact fallback
8. epsilon별 quality-latency curve 생성

#### 필수 안전장치

- feature flag 기본값 `off`
- request 또는 metric에 epsilon과 사용 여부 기록
- invalid/stale summary에서 exact fallback
- epsilon 0 또는 명시 모드에서 full exact scan
- page selection overhead까지 end-to-end에 포함
- 고정 top-k page 수만 사용하는 구현은 별도 근사로 분류

## 각 최적화 PR의 필수 구조

1. profiler evidence
2. 의미 보존 등급
3. 가설
4. reference implementation
5. optimized implementation
6. correctness 또는 error-budget 결과
7. microbenchmark
8. end-to-end 결과
9. regression range
10. runtime flag와 rollback

## 금지

- profiler 없이 fusion 선택
- 여러 fusion을 한 PR에서 동시 적용
- `A1` 결과를 exact optimization으로 표현
- omitted mass bound를 곧바로 output absolute error라고 표현
- 평균만 보고 p95/p99 악화 무시
- throughput 향상을 위해 TTFT 목표를 암묵적으로 변경
- graph capture 때문에 지원 shape를 조용히 축소
- 근사 path에서 exact fallback 제거

## 완료 기준

이 단계 전체의 종료 조건:

- [ ] target workload의 top bottleneck이 설명됨
- [ ] 적용한 각 최적화가 end-to-end에서 유효
- [ ] 각 최적화의 의미 보존 등급이 기록됨
- [ ] performance regression suite 존재
- [ ] exact fallback path parity 유지
- [ ] `A1` 후보는 기본 비활성이고 error-quality-latency curve가 존재
- [ ] 최적화별 memory trade-off 기록
- [ ] baseline engine과 동일 조건 비교 갱신

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)