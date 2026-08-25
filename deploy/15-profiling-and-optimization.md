# PR 15 — Profiling, Fusion, CUDA Graph 최적화

**상태:** Implemented
**선행 조건:** [PR 14](14-api-and-streaming.md)  
**다음:** [PR 16 — 신뢰성과 Release](16-reliability-and-release.md)

**검증된 최적화 소스:** `1826d09fd28825582f22a2d49390f6ed047562ea`

**production 기본값 승격 소스:** `c6e1c9140753c34de27156a107e106a1672a82e3`

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)

## 목적

작동하는 Python-free native 서비스의 trace를 기준으로 가장 큰 병목 하나씩만 최적화한다. 이 문서는 하나의 거대 PR을 의미하지 않으며, 아래 항목은 각각 독립 PR 후보이다.

모든 후보는 [PR 00](00-pr-contract.md)의 의미 보존 등급을 선언한다. `E0`과 `A1`을 같은 PR에서 섞지 않는다.

## 구현 결과

### 측정하고 선택한 병목

첫 후보 A인 residual-add + RMSNorm CUDA C++ fusion은 raw BF16 logits와 16-step
greedy token이 정확히 일치했지만, 5쌍 paired gate에서 primary 개선 중앙값이
`1.6115%`로 사전 선언한 `5%`를 넘지 못했다. Arm 중앙값 개선도 `1.7141%`뿐이었다.
따라서 fused path는 명시적 비교/rollback 후보로 남기고 production 기본값을
`separate`로 되돌렸다.

그 다음 trace와 source boundary를 함께 조사해 fixed-M iteration의 각 primitive가
거의 매번 `cudaStreamSynchronize`로 완료를 기다리는 host-side completion 경계를
가장 큰 병목으로 선택했다. NCU의 한 decode iteration은 양 arm 모두 517개 kernel을
실행한다. Kernel graph를 바꾸지 않고 이 수백 개의 primitive-local completion을
iteration 단위로 묶는 후보 E를 구현했다.

### Iteration command batch (`E0`)

- native C ABI에 stream command batch `begin/end`를 추가했다. `begin`은 stream을
  단일 thread owner에 lease하고, `end`는 한 번 synchronize한 뒤에만 lease를
  해제한다.
- stream마다 고정 1,024-entry resource ledger를 준비해 batch가 참조한 device buffer와
  GEMM plan을 중복 없이 retain한다. Hot path heap allocation은 없으며 ledger overflow,
  nested begin, non-owner access와 batch 중 query/synchronize/wait/event/close는 fail
  closed한다.
- completion 또는 CUDA context 복구가 확인되지 않으면 resource lease를 보수적으로
  유지한다. 부분 KV write 뒤 발생할 수 있는 모든 iteration-batch 오류는 executor와
  forward owner를 poison한다.
- Rust API는 `CudaCommandBatch`와 underlying `CudaStream`을 노출하지 않는
  `CudaCommandStream` proxy를 제공한다. Proxy는 `!Send + !Sync`이고 sealed
  `CudaExecutionStream`만 구현한다. 외부 trait 구현, stream coercion, `mem::swap` 또는
  교체는 compile-fail gate로 고정했다.
- metadata upload는 batch 시작 전에 끝내고, embedding부터 fixed graph와 output
  row-gather까지 한 completion boundary 안에 enqueue한다. Embedding token validation의
  host error report만 안전상 내부 synchronize를 유지하고 iteration 끝에 최종
  synchronize한다.
- 기존 direct `CudaStream` 호출과 explicit per-operation path는 그대로 유지한다.
  Server rollback은 `--execution-completion per-operation`이다.

### 기술 선택

Kernel fusion, CUDA Graph, CUTLASS 변경, Triton 또는 NVRTC를 추가하지 않았다. 기존
cuBLASLt GEMM과 CUDA C++ primitive의 symbol, launch shape, register/shared-memory
inventory가 이미 동일했으며, 더 단순한 completion-boundary 제거가 목표 workload에서
충분한 end-to-end 효과를 냈다. 따라서 kernel/graph 기술을 추가할 근거가 없다.

### Correctness와 ownership gate

SmolLM2-135M BF16에서 `per-operation + separate`와
`iteration-batch + separate`를 16 decode step 비교했다.

| Gate | 결과 |
|---|---|
| raw logits | 모든 iteration byte exact, mismatch 0 |
| greedy token | 16/16 exact |
| generated IDs | `4052, 2025, 284, 965, 6497, 288, 1492, 418, 260, 16438, 30, 198, 198, 504, 16438, 314` |
| hot allocation | 두 arm 모두 delta 0 |
| close accounting | owner close 뒤 allocation 0 |
| lifecycle | one-shot finish와 Drop 뒤 stream 재사용 통과 |
| validation failure | enqueue된 chain 완료, ledger 해제, buffer/stream 즉시 재사용 통과 |

Correctness gate ID는 `pr15-iteration-command-batch-exact-v1`, report SHA-256은
`de5c7c1564290e2ea16cb05f24501f508fd84fd563b2527887ebd6b731bbce39`다.

### Paired end-to-end 결과

동일한 clean commit, 동일 release executable, checkpoint, GPU와 workload로 5쌍을
AB/BA 교차 실행했다. 각 arm은 warmup 5회와 measured 30회이며 workload는
SmolLM2-135M, concurrency 1, prompt 128, output 32, greedy다.

| Metric | Baseline per-operation | Candidate iteration-batch | 결과 |
|---|---:|---:|---:|
| primary `aggregate.host.execute_ns` arm median | 8,328,279,797 | 6,779,500,596 | `18.5966%` 개선 |
| paired primary improvement median | — | — | `18.5881%` (`>=5%`) |
| output tokens/s median | 113.5428 | 139.0239 | `+22.4419%` |
| TTFT p95 ratio | — | — | `0.772982` (`<=1.05`) |
| TPOT p95 ratio | — | — | `0.817256` (`<=1.05`) |
| e2e median | 281.6990 ms | 230.0223 ms | `-18.3446%` |
| failures / dropped trace | 0 / 0 | 0 / 0 | 통과 |

Closed checker status는 `passed`이고 report SHA-256은
`804f6ee39d3aada4bfb8853eaa4772941b7dfc545c9def7816c9d711a894e060`다.
이에 따라 `rustinfer serve`의 production CLI 기본값을 `iteration-batch`로 승격했다.
저수준 executor config의 보수적 기본값과 명시 rollback path는 `per-operation`으로
유지한다. Rejected fused residual path를 사용할 때도 `per-operation`을 명시해야 한다.

### NCU attribution

각 arm에서 output 1-token과 2-token sentinel을 실행했고, 두 번째 trace의 후반 517개
row를 한 decode iteration으로 비교했다.

- 양 arm의 517/517 kernel symbol 순서, count, block/grid, register, shared memory와
  occupancy-limit inventory가 exact match했다.
- Kernel 구성은 GEMM 211, RMSNorm 61, residual-add 60, indexed RoPE 60, paged KV write
  30, ragged attention 30, SiLU 30, gated multiply 30, embedding chain 4, row-gather 1이다.
- Incremental GPU kernel time은 per-operation `2.370176 ms`, iteration-batch
  `2.368192 ms`로 `-0.084%`이며 사실상 같다. 따라서 end-to-end 개선은 GPU work 감소나
  occupancy 변화가 아니라 host completion-boundary 제거에서 온다.
- NCU raw DRAM total은 275.094 MiB 대 272.538 MiB였지만 single run이고 uncontrolled
  cache/clock 경고가 있으므로 구조적 memory-traffic 절감으로 주장하지 않는다.

### Runtime dependency와 memory trade-off

Python, PyTorch, Triton, NVRTC와 새 runtime library dependency를 추가하지 않았다.
고정 ledger는 stream당 약 8 KiB host memory를 사용한다. Device workspace와 KV capacity,
kernel launch count, register pressure와 static shared memory는 바뀌지 않는다.

## 기술 선택 Escalation Gate

최적화 기술은 복잡한 것부터 선택하지 않는다.

### Dense GEMM

```text
cuBLASLt baseline
→ algorithm/layout/epilogue tuning
→ CUTLASS prototype와 비교
→ end-to-end 이점이 있을 때 CUTLASS production
→ universal custom GEMM은 원칙적으로 제외
```

### Non-GEMM/Fusion

```text
분리된 CUDA C++ reference
→ profiler evidence
→ fused CUDA C++ 또는 CUTLASS epilogue
→ native exact fallback 유지
```

### Triton

Triton은 `experiments/triton/`에서 prototype과 성능 상한 탐색에 사용한다.

```text
Triton prototype
→ correctness
→ Nsight/benchmark
→ CUDA C++ 또는 CUTLASS port
→ production integration
```

초기 production binary가 Triton Python compiler나 PyTorch를 요구하도록 만들지 않는다. Triton을 production에 유지하려면 별도 PR에서 Python-free AOT artifact, version pin, cache, cold-start, CUDA Graph와 stream semantics를 검증한다.

### NVRTC

NVRTC runtime specialization은 다음이 모두 확인되기 전에는 도입하지 않는다.

- 미리 compile할 shape/kernel 조합이 운영상 과도함
- specialization 이점이 end-to-end에서 큼
- compile timeout과 failure fallback 존재
- cubin/PTX cache provenance와 invalidation 정책 존재
- release 환경 compiler dependency가 승인됨

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
- backend/library cold-load 및 dispatch overhead

## 최적화 후보 우선순위

### 후보 A — Residual + RMSNorm (`E0`)

조건:

- norm/elementwise launch가 TPOT에서 유의미
- reference parity 확보
- FP32 reduction semantics 유지
- register pressure가 허용 범위

구현 기본값은 CUDA C++다. CUTLASS epilogue로 옮기려면 GEMM 경계와 실제 이점이 확인되어야 한다.

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

Production 구현은 CUDA C++와 C ABI 경계를 따른다.

### 후보 C — GPU Sampling (`E0` 또는 sampling contract 보존)

조건:

- logits copy와 CPU sampling이 작은 batch latency에서 유의미
- PR 11의 logits processing 순서 유지
- request별 deterministic RNG contract 유지
- CPU reference와 probability 및 token 결과 검증

Python sampling은 reference일 뿐 운영 fallback이 아니다.

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
3. technology choice와 더 단순한 대안 검토
4. 가설
5. reference implementation
6. optimized implementation
7. correctness 또는 error-budget 결과
8. microbenchmark
9. end-to-end 결과
10. runtime dependency 변화
11. regression range
12. runtime flag와 rollback

## 금지

- profiler 없이 fusion 선택
- 여러 fusion을 한 PR에서 동시 적용
- cuBLASLt 비교 없이 복잡한 CUTLASS GEMM 도입
- Triton prototype을 자동으로 production dependency로 승격
- Python/PyTorch fallback을 운영 경로에 추가
- `A1` 결과를 exact optimization으로 표현
- omitted mass bound를 곧바로 output absolute error라고 표현
- 평균만 보고 p95/p99 악화 무시
- throughput 향상을 위해 TTFT 목표를 암묵적으로 변경
- graph capture 때문에 지원 shape를 조용히 축소
- 근사 path에서 exact fallback 제거

## 검증과 provenance

모든 CUDA compile, checkpoint load, inference, paired benchmark와 NCU 실행은 로컬이
아닌 `server-4096`의 RTX 4090에서 수행했다. 컨테이너는 `--network none`, Cargo는
`--locked --offline`을 사용했다.

| Gate | 결과 |
|---|---|
| local workspace CPU all-targets test / Clippy | 통과 |
| native profile checker unit | 20 passed |
| remote Python-free CUDA compile/C ABI/strict Clippy | 통과 |
| remote workspace all-targets/all-features | 통과, 실패 0 |
| command batch lifecycle GPU | 1 passed |
| multi-primitive ledger/validation/reuse GPU | 1 passed |
| SmolLM2 16-step exact parity GPU | 1 passed |
| 5-pair performance gate | passed, paired primary `18.5881%` |
| four NCU sentinels | checksum과 HTTP fixed-token contract 통과 |
| promoted production CLI source CUDA compile-only | 통과 |

- authoritative optimization source:
  `1826d09fd28825582f22a2d49390f6ed047562ea`
- append-only correctness/performance/NCU evidence root:
  `/home/psyche/rustinfer-artifacts/pr15/1826d09fd28825582f22a2d49390f6ed047562ea/run-20260825T184555Z`
- source archive SHA-256:
  `39ce9b9898defc9cbc3b0a346739d0743de3833e189f65434985b38606abb86d`
- profile executable SHA-256:
  `60650fa3af1d8a761432672862d1c464db86f2080d1acd5d81dccc6d1f7c9be8`
- evidence `SHA256SUMS` SHA-256:
  `f25393c891a78cfcf661e525192e1c24b2f472440221b49cc0820815acc22d20`
- promoted production default source:
  `c6e1c9140753c34de27156a107e106a1672a82e3`
- promotion compile evidence root:
  `/home/psyche/rustinfer-artifacts/pr15/c6e1c9140753c34de27156a107e106a1672a82e3/run-20260825T185910Z`
- promotion source archive SHA-256:
  `913093e20a0564d8abbd3d4c13200ab8465d2759138eb4e73238209fdc3a5ad9`
- promotion compile log / evidence-manifest SHA-256:
  `ad3a881fd7b7d068d51b5bd6aa3db5bbdb0aa5f1cafcb032a91159a4865faba4` /
  `159eaa56e7f920e8d090e9efb90b90f3439430570a4ec4519e3e5b8beecedd81`
- container image:
  `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- GPU/runtime: NVIDIA GeForce RTX 4090, compute capability 8.9, driver
  580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93, cuBLAS 12.8.4.1
- rejected candidate A evidence:
  `/home/psyche/rustinfer-artifacts/pr15/45b31e212f7b56c6ff6d4f89567485ac04685a1d/run-20260825T175524Z`
- version-controlled evidence index:
  [PR 15 profiling/optimization evidence](../benchmarks/results/20260825T184555Z-rustinfer-profiling-pr15-run001/README.md)

## 제한과 rollback

- 성능 승격 범위는 기록된 single-GPU sm89, SmolLM2 BF16, c1/p128/o32 target
  workload다. Qwen, c8, long-context와 다른 driver/toolkit에서는 regression suite를
  다시 실행한다.
- NCU는 kernel attribution용 single run이고 benchmark latency로 사용하지 않는다.
  Kernel count와 static inventory만 구조적 증거로 사용하고 cache/clock-sensitive DRAM
  delta는 기술적 관찰로만 남긴다.
- Command batch ledger capacity는 1,024 unique resource다. Capacity 또는 owner invariant
  위반은 fallback하지 않고 해당 iteration을 fail closed한다.
- Batch 중 embedding validation report를 위해 한 번의 내부 sync가 남아 있다. 이를
  비동기화하려면 별도 error-report ownership 설계와 correctness gate가 필요하다.
- `--execution-completion per-operation`이 production rollback이다. 새 GPU/toolkit/model에서
  exact parity, zero-allocation, ownership 또는 5-pair regression gate가 실패하면 이 flag로
  즉시 되돌린다. Rejected residual fusion은 기본 비활성이며
  `--residual-rmsnorm fused --execution-completion per-operation`으로만 명시 비교한다.

## 완료 기준

이 단계 전체의 종료 조건:

- [x] target workload의 top bottleneck이 설명됨
- [x] 적용한 각 최적화가 end-to-end에서 유효
- [x] cuBLASLt/CUTLASS/custom CUDA 선택 근거가 기록됨
- [x] 각 최적화의 의미 보존 등급이 기록됨
- [x] performance regression suite 존재
- [x] exact native fallback path parity 유지
- [x] `A1` 후보를 도입하지 않았고 기존 experimental 후보는 기본 비활성
- [x] production path가 Python-free로 유지됨
- [x] Triton/NVRTC를 사용하지 않기로 한 dependency 결정이 기록됨
- [x] 최적화별 memory trade-off 기록
- [x] baseline engine과 동일 조건 비교 갱신

[← 이전](14-api-and-streaming.md) | [목차](README.md) | [다음 →](16-reliability-and-release.md)
