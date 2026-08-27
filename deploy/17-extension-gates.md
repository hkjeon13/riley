# PR 17 — 후속 확장 진입 Gate

**상태:** Active — fail-closed metadata/transition gate; bootstrap은 empty-by-default

**선행 조건:** [PR 16](16-reliability-and-release.md)  
**다음:** 없음

[← 이전](16-reliability-and-release.md) | [목차](README.md)

## 목적

첫 release 이후 기능 요청이 core를 무질서하게 확장하지 않도록, 각 기능의 진입 조건과 독립된 작업 축을 정의한다. 수학적 치환이나 근사 기법도 [PR 00](00-pr-contract.md)의 의미 보존 등급에 따라 별도 deploy 문서와 benchmark contract를 만든 뒤 시작한다.

이 단계가 구현하는 것은 확장 자체가 아니라 **확장을 시작하기 위한 닫힌
allowlist**다. v1 bootstrap artifact는 빈 registry로 landing하며, 이후 현재 승인
집합은 [`extensions/registry.json`](extensions/registry.json)이 유일하게 결정한다.
아래 로드맵에 이름이 있다는 사실만으로 구현을 시작할 수 없다.

PR 16의 이번 순차 구현은 명시적인 soak waiver로 다음 단계 진행만 허용했다. 이
gate의 구현은 PR 16 soak 통과, release qualification, release tag 승인을 의미하지
않으며 기존 release checker나 공통 benchmark result schema를 변경하지 않는다.

## 구현된 admission boundary

확장을 구현하기 전에 admission-only PR로 다음 네 항목을 함께 등록한다.

```text
deploy/extensions/registry.json
deploy/extensions/proposals/<extension-id>.json
deploy/extensions/plans/<extension-id>.md
benchmarks/extensions/contracts/<extension-id>.json
```

[`ci/check_extension_gates.py`](../ci/check_extension_gates.py)는 Python 표준
라이브러리만 사용하며 다음 drift를 fail closed로 거부한다.

- duplicate JSON key, extension ID, path, runtime flag
- schema에 없는 field와 알 수 없는 semantic class
- registry에 없는 proposal/plan/benchmark contract 또는 누락된 등록 파일
- absolute path, `..`, 비정규 POSIX path, symlink, 허용 디렉터리 이탈
- registry, proposal, benchmark contract 사이의 ID·등급·경로·flag·metric 불일치
- `approved-for-implementation` 외 상태, roadmap에 없는 track, 승인 질문 답변 누락
- admission 시점의 enabled/stable default
- 의미 보존 등급별 필수 검증 계약 누락
- reference/fallback/workload의 Git 미추적, `.git` control path, SHA-256 bytes drift
- track과 semantic class의 닫힌 조합, 공통 result schema 밖 metric path
- single-GPU 비교 환경, sampling/cold-warm를 포함한 동일조건 실제 값 누락

빈 registry는 명시적으로 성공한다. 이는 “모든 확장 허용”이 아니라 “허용된 확장
0개”를 뜻한다. 형식, 별도 PR 경계와 실행 명령은
[`extensions/README.md`](extensions/README.md), benchmark 반복·환경 계약은
[`benchmarks/extensions/README.md`](../benchmarks/extensions/README.md)를 따른다.

PR CI는 full base SHA를 전달한다. checker는 bootstrap-empty, v1 append-only,
기존 registry entry와 proposal/plan/contract bytes 불변, 새 admission 한 개의 정확한
four-file diff를 rename source까지 포함해 검사한다. bootstrap 이후 production
crate에 새 `RUSTINFER_EXPERIMENTAL_*` flag를 넣으려면 정확히 한 approved
implementation link가 필요하다. Markdown plan의 의미는 PR template reviewer
check로 닫는다.

## 공통 진입 규칙

모든 확장 제안은 먼저 다음을 선언한다.

```json
{
  "status": "approved-for-implementation",
  "track": "one closed roadmap track",
  "semantic_class": "reference | E0 | E1 | A1 | M1",
  "reference_path": "existing/repository/file",
  "reference_sha256": "64 lowercase hex characters",
  "fallback_path": "existing/repository/file",
  "fallback_sha256": "64 lowercase hex characters",
  "primary_metric": "closed common-result performance/resource path",
  "required_metrics": ["closed track-specific metric set"],
  "quality_or_error_metric": "closed class/track quality path",
  "runtime_flag": "RUSTINFER_EXPERIMENTAL_*",
  "default_enabled": false,
  "stable_default": false
}
```

같은 admission의 registry entry는 `implementation_link_path: null`로 시작한다.
reference와 fallback은 서로 다른 Git-tracked regular file이어야 하며 path와 실제
bytes SHA-256이 proposal/contract에서 일치한다. workload도 label이 아니라
Git-tracked `{path, sha256}` object다. 비교 환경은 `gpu_count: 1`과 정확한
GPU/driver/CUDA/model ID/full 40-hex model revision/dtype, unique positive
concurrency/prompt/output 배열, 닫힌 sampling config, `cold`와 `warm`을 동결한다.

track/class 조합은 다음처럼 닫는다: quantization=`{E0,A1}`,
low-rank-weight-compression=`{A1}`, kv-compression=`{A1}`,
prefix-cache=`{reference}`, kv-offload-prefetch=`{reference}`,
query-aware-kv-selection=`{A1}`, moe=`{E0,M1}`, mamba-ssm=`{E0}`,
multimodal=`{E0,M1}`, speculative-decoding=`{E0,E1}`,
jacobi-lookahead=`{E0,A1,M1}`. Multi-GPU는 current common result contract의
`gpu_count: 1`과 맞지 않아 v1 allowlist에서 제외한다.

primary와 quality metric은 서로 달라야 한다. primary는 공통 result schema의
track별 performance/resource scalar allowlist에 속하고 각 track의 닫힌
`required_metrics` 집합에도 포함되어야 한다. speculative track은 acceptance,
target-call, draft/verify latency, rejected suffix, rollback, lookahead와 end-to-end
latency/throughput 지표를 모두 요구한다. E0 tolerance는 실제
비교 환경 dtype 하나와 정확히 일치하고 `atol`/`rtol`은 각각 1 미만이다. v1
quality path는 `reference`/`E0`의
`failure_count`와 query-aware A1의 `sparse_attention.omitted_mass_bound`만 현재
admission 가능하며, omitted-mass error budget은 `fraction` 단위와 `[0, 1)` 범위다.
`sparse_attention.exact_fallback_rate`는 운영 metric이지
quality/error budget이 아니다. 나머지 A1/E1/M1은 적합한 quality field를 갖는 공통
result schema version-up 전까지 fail closed다.

v1 admission은 구현 작업 시작만 승인한다. 후속 구현 PR은 null link를 한 번만
experimental implementation manifest로 연결해 approved ID/plan/contract/flag와
tracked source, `{path, sha256, test_id}` auto-discovered workspace integration
test를 결합한다. test ID는 hidden/cfg/ignored target이 아닌 direct top-level
`#[test] fn`이어야 한다. checker의 정적 검사는 test 본문의 runtime semantics
증명이 아니므로 reviewer가 실제 default-off, flag-on,
stable fallback control flow와 CPU workspace test 포함을 확인한다. v1은 stable
promotion, performance result receipt 판정, withdrawal, admitted contract 수정을
지원하지 않으며 먼저 schema v2 transition이 필요하다.

---

## Quantization과 등가 좌표변환

**의미 보존 등급:** 변환-only는 `E0`, quantization 적용 후는 `A1`.

### 진입 조건

- BF16/FP16 path 안정
- weight packing interface 존재
- dequant/GEMM backend 후보 비교
- 정확도 평가 corpus 존재
- transform-only와 quantized path를 분리할 수 있음

### 대각 스케일링

Linear가 다음과 같을 때 가역 대각행렬 `D`를 이용할 수 있다.

```text
Y = XW
Y = (X D^-1) (D W)
```

full precision에서는 동일한 함수다. activation outlier를 weight 쪽으로 이동하여 activation quantization을 쉽게 만드는 데 사용할 수 있다.

구현 단계:

1. calibration corpus에서 channel scale 후보 계산
2. full-precision transformed checkpoint 생성
3. transform-only logits/token parity 검증
4. scale을 인접 norm 또는 GEMM prologue/weight에 fold
5. quantization 적용
6. bit-width별 quality-latency-memory curve 작성

주의점:

- residual branch와 공유 activation에 임의로 scale을 삽입하지 않음
- tied weight와 tensor-parallel shard에 동일 변환 적용
- scale overflow/underflow와 accumulator dtype 검증
- 변환이 exact라는 사실과 quantization이 approximate라는 사실을 분리해 보고

### 직교 또는 Hadamard 회전

직교행렬 `R`에 대해 다음이 성립한다.

```text
R R^T = I
XW = (X R) (R^T W)
```

outlier가 특정 channel에 집중된 경우 회전으로 여러 차원에 분산하여 low-bit quantization을 개선할 수 있다.

구현 단계:

1. 고정 Hadamard와 offline learned rotation을 별도 후보로 비교
2. transformed weight를 checkpoint load 시 생성하거나 사전 변환
3. activation rotation의 runtime cost 측정
4. norm/GEMM/attention 경계에 legal하게 fuse 가능한 위치 분석
5. transform-only parity
6. quantized end-to-end benchmark

필수 지표:

- rotation 자체 latency
- 추가 kernel launch와 temporary bytes
- weight/activation/KV quantization error
- perplexity 또는 task 품질
- TTFT/TPOT/throughput/VRAM

### 분리할 PR

1. quantization metadata IR
2. diagonal transform-only pipeline
3. orthogonal/Hadamard transform-only pipeline
4. checkpoint loader와 transformed weight cache
5. GEMM backend 한 format
6. activation quantization
7. KV quantization은 별도
8. end-to-end 품질·성능 보고

INT8, FP8, INT4, FP4, AWQ/GPTQ 등을 한 PR에 넣지 않는다.

---

## Low-rank weight 또는 KV compression

**의미 보존 등급:** 일반적으로 `A1`.

SVD 기반 rank-`r` 근사는 다음과 같다.

```text
W ≈ U_r S_r V_r^T
```

원래 `m × n` Linear의 단순 곱셈 수는 `m*n`, 분해 후는 대략 `r*(m+n)`이다.

```text
필요조건: r * (m + n) < m * n
```

이는 실제 속도 향상의 충분조건이 아니다. 한 번의 큰 GEMM이 두 번의 작은 GEMM과 intermediate write로 바뀌므로 tensor-core utilization과 launch overhead를 포함해야 한다.

### 진입 조건

- target layer의 singular-value decay와 effective rank가 충분함
- 두 GEMM의 실제 latency가 원래 GEMM보다 낮음
- layer별 sensitivity 분석 가능
- quality regression budget이 정의됨

### Weight compression 단계

1. layer별 singular spectrum 수집
2. rank별 reconstruction/logit error 측정
3. 가장 유망한 layer만 선택
4. 두 GEMM execution plan과 workspace 구현
5. quantization과 결합하기 전 단독 효과 측정
6. end-to-end quality-latency curve

### KV compression 단계

Weight와 별도 연구로 둔다.

- K/V hidden dimension의 저랭크 projection
- cache에 저장할 representation 정의
- attention 시 reconstruction 또는 latent-space 계산
- cache bytes와 decode bandwidth 측정
- full attention 대비 오차와 긴 생성 안정성 검증

모든 layer에 일률적으로 SVD를 적용하지 않는다.

---

## Prefix Cache

진입 조건:

- paged KV reference count 안정
- block hash와 token identity 정의
- eviction 정책 benchmark
- multi-tenant privacy 경계

prefix sharing은 단순 성능 기능이 아니라 lifetime과 보안 기능이다.

---

## KV Offload/Prefetch

진입 조건:

- GPU cache pressure가 실제 bottleneck
- PCIe/NVLink bandwidth 측정
- async copy와 stream ordering 검증
- head-of-line blocking 측정

CPU offload, SSD, remote cache를 단계별로 분리한다.

---

## Error-bounded Query-aware KV Page Selection

**의미 보존 등급:** `A1`.

고정 `top-k pages`보다 해석 가능한 **omitted softmax mass upper bound**를 사용하여 긴 context decode의 KV read를 줄이는 연구 트랙이다.

### 진입 조건

- 긴 context에서 decode가 KV memory bandwidth bound임
- PR 10 block table과 optional metadata sidecar가 안정됨
- PR 09의 exact partial-state merge가 사용 가능
- page summary 생성 비용과 bytes가 측정됨
- exact full-page fallback 유지

### Page score upper bound

page `B`의 key dimension별 `key_min`, `key_max`를 저장한다. query `q`에 대해 다음 값을 계산한다.

```text
U_B = sum over dimension j:
      q_j * key_max[B, j]  when q_j >= 0
      q_j * key_min[B, j]  when q_j < 0
```

동일 scale과 positional 좌표계에서 계산했다면 page 내부 모든 key score는 `U_B` 이하이다.

### Adaptive stopping

1. page를 `U_B`가 큰 순서로 선택
2. 선택 page는 exact attention partial state로 처리
3. 읽지 않은 page의 최대 가능 softmax mass를 계산
4. 다음 상한이 `epsilon` 이하가 될 때 중단

```text
missing_mass_upper
  = sum valid_tokens[B] * exp(U_B - current_max)

missing_probability_upper
  = missing_mass_upper / (seen_exp_sum + missing_mass_upper)
```

이 값은 **누락 probability mass의 상한**이다. output vector norm의 오차 상한이 필요하면 page별 `value_norm_max`를 추가한다.

```text
output_error_norm_upper
  <= missing_probability_upper * global_or_page_value_norm_bound
```

실제 구현에서는 더 타이트한 bound를 연구할 수 있지만, 상한의 전제와 단위를 문서화해야 한다.

### 분리할 PR

1. page key min/max summary 생성과 검증
2. query별 score upper-bound CUDA kernel
3. CPU reference page ordering
4. GPU exact partial-state scan과 adaptive stop
5. value norm metadata와 output bound
6. scheduler/cost model 연계
7. epsilon별 품질·성능 보고

### 안전장치

- default off
- epsilon과 실제 selected/total page 수 기록
- stale/unsupported metadata에서 exact fallback
- mask, GQA group, RoPE variant, quantized key 좌표계를 정확히 반영
- epsilon 0 또는 exact mode는 모든 page 처리
- bound 계산 자체가 full scan보다 비싸면 자동 비활성화 가능

고정 page 수만 선택하고 bound를 계산하지 않는 방법은 별도 근사로 보고한다.

---

## MoE

분리할 축:

1. MoE IR/router semantics
2. top-k routing correctness
3. dispatch metadata
4. grouped expert GEMM
5. weighted combine
6. expert parallel communication

Mixtral류와 DeepSeek류 router 차이를 semantic parameter로 보존한다.

Inference router는 모델에 정의된 softmax/sigmoid와 exact top-k 의미를 유지한다. Gumbel-Softmax는 학습용 relaxation이므로 기본 inference MoE 구현 범위에 포함하지 않는다.

---

## Mamba/SSM과 Associative Scan

**의미 보존 등급:** sequential recurrence와 동일한 associative formulation은 `E0`.

필요 primitive:

- causal depthwise Conv1d
- conv state update
- selective scan
- selective state update
- recurrent cache type

attention의 특수 case로 구현하지 말고 `MixerSpec`의 별도 variant로 둔다.

### Affine recurrence의 associative composition

다음 recurrence를 생각한다.

```text
h_t = A_t * h_(t-1) + b_t
```

step을 pair `(A, b)`로 표현하면 두 step의 합성은 다음과 같다.

```text
(A2, b2) compose (A1, b1)
  = (A2*A1, A2*b1 + b2)
```

이 합성이 associative하면 parallel prefix scan으로 sequence 방향의 병렬 깊이를 줄일 수 있다.

### 구현 순서

1. 단순 sequential recurrence reference
2. pair composition unit test
3. CPU tree scan reference
4. CUDA block-local scan
5. block summary의 hierarchical scan
6. selective parameter, gate, `D` skip과 cache update 통합
7. prefill scan과 single-token state update 분리

### 검증

- sequence length와 partition 수별 parity
- zero/large/near-one transition 값
- FP32 accumulator 필요성
- 마지막 recurrent state 일치
- chunked prefill와 full prefill 일치
- Jamba처럼 attention/SSM layer가 섞인 cache slot mapping

전체 work가 줄지 않고 병렬 깊이만 줄 수 있으므로 end-to-end latency와 memory traffic으로 판단한다.

---

## Multimodal

순서:

1. modality encoder interface
2. projector/merger
3. token packing
4. multimodal position IDs
5. shared decoder 연결

Qwen-VL 전체를 model-specific monolith로 추가하지 않는다.

---

## Multi-GPU

진입 조건:

- single GPU 병목과 목표 명확
- tensor/expert/pipeline parallel 중 하나 선택
- NCCL failure와 timeout 정책
- process topology와 rank lifecycle
- 단일 GPU regression 방지

Online softmax partial state를 rank별로 merge할 경우 `E0` parity와 collective ordering을 별도로 검증한다.

---

## Speculative Decoding

**의미 보존 등급:** 올바른 rejection correction을 사용하는 sampling은 `E1`; greedy verification은 deterministic exact path로 검증.

### 진입 조건

- baseline scheduler와 KV rollback 안정
- verify mode attention 지원
- draft/target tokenizer compatibility
- PR 11 RNG snapshot/restore/fork 구현
- acceptance와 target-call metric 정의
- rejected suffix의 KV rollback 검증

### Phase 1 — Fixed-length reference

먼저 고정 draft 길이 `k`만 구현한다.

Draft distribution을 `q`, target distribution을 `p`라고 할 때 draft token `x`의 수락확률은 다음과 같다.

```text
accept_probability(x) = min(1, p(x) / q(x))
```

거절 시 correction sample은 positive residual에서 뽑는다.

```text
residual(x) proportional to max(p(x) - q(x), 0)
```

구현 단계:

1. draft model 또는 self-draft가 `k`개 token과 각 `q` probability를 생성
2. target이 한 번에 draft suffix를 verify
3. 왼쪽부터 acceptance test
4. 최초 rejection에서 residual distribution sample
5. rejected KV와 RNG branch rollback
6. 모두 수락되면 target의 추가 token sample

작은 vocabulary toy distribution에서 모든 결과 확률을 exhaustive하게 검증한다.

### Phase 2 — Dynamic lookahead와 Optimal Stopping

고정 `k`가 안정된 뒤에만 동적 길이를 추가한다.

각 위치의 예상 조건부 acceptance를 `a_i`라고 하면 `j`개 prefix가 모두 수락될 확률은 다음처럼 추정할 수 있다.

```text
P(accepted_prefix >= j) ≈ product from i=1 to j of a_i
```

기대 수락 token 수:

```text
E[accepted_tokens(k)]
  ≈ sum from j=1 to k of product from i=1 to j of a_i
```

scheduler는 측정된 draft와 verify 비용을 사용해 다음과 유사한 utility를 최대화한다.

```text
utility(k)
  = expected_accepted_tokens(k)
    / estimated_total_time(k)
```

구현 순서:

1. request/model/position별 acceptance metric 수집
2. 단순 exponential moving average predictor
3. entropy, token probability, recent rejection 등 feature 추가 여부 검토
4. marginal accepted-token benefit가 marginal cost보다 작을 때 stop
5. min/max lookahead와 starvation guard
6. fixed-length policy와 A/B benchmark

Predictor는 출력 품질을 바꾸지 않고 성능만 바꿔야 한다. correction sampling을 생략하거나 threshold로 임의 수락하면 `E1`이 아니다.

### Draft source 후보

별도 PR로 분리한다.

- 외부 작은 draft model
- 같은 model의 intermediate/self-speculative head
- MTP head
- prompt/cache 특화 lightweight draft

### 필수 지표

- draft length
- acceptance rate
- accepted tokens per verification
- target calls per output token
- draft/verify/correction latency
- rollback token과 KV block 수
- TTFT/TPOT/throughput
- concurrency별 scheduler 영향

### 실패와 fallback

- draft 오류 또는 timeout 시 target-only decode
- tokenizer mismatch 시 시작 거부
- unsupported sampling processor에서 target-only fallback
- request cancellation 시 draft/target state 모두 회수
- target distribution correction에 필요한 probability를 사용할 수 없으면 sampling speculative 비활성

---

## Jacobi/Lookahead Decoding

**의미 보존 등급:** 검증된 prefix만 확정하는 경우 `E0` 또는 exact-by-verification 실험으로 취급한다. 검증 없이 fixed-point 추정 token을 출력하면 `A1` 또는 `M1`이다.

별도 draft model 없이 미래 token 위치에 초기 추정값을 두고 여러 위치를 병렬 갱신한다.

```text
x_i^(t+1) = F_i(x_1^t, ..., x_(i-1)^t)
```

### 진입 조건

- low-concurrency에서 일반 decode가 GPU를 충분히 채우지 못함
- verify mode와 KV rollback이 안정됨
- 추가 FLOPs를 감당할 여유가 있음
- target-only path 대비 serial step 감소가 예상됨

### 구현 단계

1. deterministic greedy toy/reference
2. lookahead window와 initial guess 정의
3. 한 forward에서 여러 candidate position 계산
4. 왼쪽부터 target consistency 검증
5. 일치한 prefix만 commit
6. 나머지 state rollback 또는 재사용 규칙 정의
7. fixed window와 adaptive window 비교

### 필수 지표

- forward당 committed tokens
- 추가 FLOPs
- GPU utilization
- target calls per output token
- low/high concurrency별 손익분기
- exact greedy sequence 일치

이 방법은 일반 continuous batching이 GPU를 이미 채우는 상황에서는 손해일 수 있으므로 low-concurrency fast path로만 검토한다.

---

## 확장 승인 질문

새 기능마다 다음을 답한다. proposal schema의 닫힌 `approval_answers` object가 열
질문을 기계적으로 요구하며, semantic class, reference/fallback, implementation
boundary 같은 공통 field와 함께 checker가 교차 검증한다.

1. 실제 사용자 workload에서 어떤 병목을 해결하는가?
2. `reference`, `E0`, `E1`, `A1`, `M1` 중 무엇이며 track에 허용되는 조합인가?
3. 기존 IR로 표현 가능한가?
4. core에 넣어야 하는가, plugin/backend로 분리 가능한가?
5. correctness reference는 무엇인가?
6. error budget 또는 distribution contract는 무엇인가?
7. memory와 operational complexity는 얼마인가?
8. 실패 시 기존 stable path로 되돌릴 수 있는가?
9. 추가 계산이 줄이는 것은 FLOPs, serial depth, HBM traffic 중 무엇인가?
10. microbenchmark가 아니라 end-to-end에서도 이득인가?

## 완료 기준

PR 17의 완료 범위는 확장 구현이 아니라 범위 통제 장치다. 다음 항목을 구현했다.

- empty-by-default closed registry와 proposal/registry/implementation JSON schema
- 별도 extension benchmark contract schema와 공통 result schema 비변경 원칙
- registry completeness, Git tracking/hash, path/symlink containment, cross-file,
  track/class/metric/environment별 fail-closed 검사
- `reference`·현재 표현 가능한 `E0`·query-aware `A1`의 정상/실패, 공통 quality
  field가 없는 A1/E1/M1의 version-pending 거부, 빈 registry를 검증하는 CPU unit test
- base-revision의 bootstrap-empty, append-only admission, immutable artifact,
  exact four-file diff와 experimental implementation link 검사
- production CPU CI, PR template, 기여 계약 연결

각 확장은 앞으로도 admission-only PR을 먼저 병합한 뒤 별도 deploy plan의 작은 PR로
구현한다. bootstrap/default registry는 비어 있고, landing 이후 현재 승인 수는
registry 내용에서 동적으로 판정한다. stable promotion과 withdrawal은 v1 완료 범위가
아니며 schema v2 transition을 먼저 설계해야 한다.

[← 이전](16-reliability-and-release.md) | [목차](README.md)
