# PR 08 — Attention Backend와 Prefill 경로

**상태:** Implemented — remote validated at `73cbdad`
**선행 조건:** [PR 07](07-llama-reference-forward.md)  
**다음:** [PR 09 — 단일 요청 Decode](09-single-request-decode.md)

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)

## 목적

correctness-first attention을 교체 가능한 backend interface로 만들고, full-sequence prefill에서 검증된 고성능 **native CUDA backend**를 사용한다. 최적화된 경로는 score matrix 전체를 HBM에 materialize하지 않는 **online softmax 기반 `E0` 변환**을 우선한다.

Production attention path는 Python, PyTorch extension 또는 Triton Python JIT를 요구하지 않아야 한다.

## 구현 결과

- BF16 contiguous dense BSHD `[B,S,H,D]`용 `PrefillAttentionRequest`, immutable
  capability, cold selector, execution params와 selection trace를 구현했다.
- Materialized reference는 기존 staged-BF16 `QK → scale/mask → softmax → AV`를
  interface 뒤에서 유지한다. Caller-owned `[QH,S,S]` BF16 workspace 하나를 batch 간
  재사용한다.
- Optimized backend `rustinfer.cuda.online-gqa-prefill.bf16.d64@1`은 D64 MHA/GQA의
  causal·causal-local prefill을 8 warps/CTA, warp당 query row 하나, K/V 32-token tile로
  실행한다. Full score/probability matrix를 global memory에 쓰지 않는다.
- QK 결과와 scaled score는 reference 계약과 같은 위치에서 BF16으로 반올림하되 register에만
  둔다. Max, denominator, numerator `(m,l,n)`은 FP32로 유지하고 최종 context만 BF16으로
  저장한다.
- Default Llama prefill은 optimized를 선택한다. PR 07 probability trace와 golden 경로는
  `.with_reference_attention()`으로만 요청할 수 있고, online에서 probability trace를
  요구하면 정직하게 실패한다.
- Prepared plan은 선택에 사용한 정확한 CUDA context owner에 묶인다. Hot `execute`는 이미
  고정된 backend를 layer마다 한 번 호출하며 backend 재선택, allocation, retry 또는
  post-launch fallback을 하지 않는다.
- Trace는 implementation/version, native dependency, 실제 AOT architecture set, device
  ordinal/compute capability, selection reason, score materialization, workspace와 layout-copy
  byte를 보존한다.

## Backend 구현 우선순위

```text
1. score-matrix correctness reference
2. 검증된 native CUDA attention backend adapter
3. target shape·cache·latency 공백이 확인된 경우 custom CUDA C++
4. CUTLASS/CuTe 구성은 명확한 이점이 있을 때
```

Triton은 prototype과 비교 실험에 사용할 수 있지만 초기 production backend가 아니다. Triton 결과를 채택하려면 Python 없는 AOT loading, stream/graph semantics와 배포 전략을 별도 승인해야 한다.

## Interface 요구사항

```rust
AttentionMode::Prefill
Q/K/V views
head counts
head dimension
causal/local mask
scale
workspace
stream
output view
```

backend는 capability를 선언한다.

- supported GPU arch
- dtype
- head dimension
- causal 여부
- variable sequence 지원
- non-contiguous layout 지원 여부
- CUDA Graph capture 가능 여부
- online reduction 지원 여부
- split-K 또는 partial-state merge 지원 여부
- native runtime dependency
- implementation ID와 version

## Online softmax의 정확한 부분합

한 query row의 score 구간 `C`를 처리할 때 다음 상태만 유지한다.

```text
m_C = max score in C
l_C = sum exp(score - m_C)
n_C = sum exp(score - m_C) * value
```

두 구간 `A`, `B`는 전체 score를 다시 보지 않고 병합할 수 있다.

```text
m = max(m_A, m_B)
l = exp(m_A - m) * l_A + exp(m_B - m) * l_B
n = exp(m_A - m) * n_A + exp(m_B - m) * n_B
output = n / l
```

이 재구성은 실수 연산에서 일반 softmax attention과 동일하다. 실제 GPU에서는 reduction 순서가 달라져 rounding 차이가 생길 수 있으므로 `E0` tolerance와 token-level 회귀를 적용한다.

실제 optimized kernel은 reference의 두 score quantization boundary를 재현하지만 bit-exact
reference 구현은 아니다. QK는 D64 serial FMA 대신 warp-tree reduction이고, reference가
normalized probability를 BF16 matrix로 저장하는 것과 달리 online path는 FP32 recurrence와
numerator를 유지한다. 아래의 `E0` 판정은 이 floating-point order 차이에 대한 pinned
numeric/token evidence이며 PR 01의 31-case full-corpus gate 재인증을 뜻하지 않는다.

## 구현 순서

1. 기존 score-matrix reference backend를 interface 뒤로 이동
2. `OnlineSoftmaxState`의 reference CPU 또는 단순 CUDA C++ 구현 작성
3. 두 부분합의 merge unit test 작성
4. 검증된 native CUDA attention backend 연결
5. backend가 online/tiled softmax를 사용하는지 capability와 문서로 확인
6. unsupported shape는 native reference로 fallback
7. backend 선택 이유, dependency와 score materialization 여부를 trace에 기록
8. output parity 검증
9. Python 없는 release-like 환경에서 backend load/execute 확인

처음부터 universal custom FlashAttention을 작성하지 않는다. 외부 backend가 target shape를 충분히 지원하면 해당 구현을 사용하고, custom kernel은 profiler가 증명한 공백에만 작성한다.

이 PR의 repository에는 Python 없이 AOT-load 가능한 외부 attention dependency가 없었고,
기존 native reference는 BF16 `[QH,S,S]`를 반드시 materialize했다. 따라서 범용 kernel이
아닌 target 전용 D64 dense GQA/MHA kernel을 추가했다. Capability가 맞지 않으면 이를
확장해 억지로 실행하지 않고 cold selector가 reference로 fallback하거나 두 경로 모두
지원하지 못하는 요청을 launch 전에 거부한다.

## Split-K와 병렬 merge

긴 sequence에서 K/V 범위를 여러 CTA 또는 partition으로 나눌 수 있다. 각 partition은 `(m, l, n)`을 만들고 최종 reducer가 결합한다.

필수 조건:

- partition 수와 순서가 결과 의미를 바꾸지 않음
- empty 또는 fully masked partition 처리
- `-inf` score와 all-masked row 처리
- accumulator dtype 명시
- workspace 없이 가능한 경로와 workspace 경로 구분

PR 08에서 split-K 최적화 자체는 필수가 아니지만, backend interface는 partial-state merge를 막지 않아야 한다.

현재 native capability의 `partial_state_merge=false`이며 `key_partition_count=1`만 online으로
실행한다. CPU `OnlineSoftmaxState`는 empty/all-masked, `-inf`, tied `+inf`, NaN reject,
fallible allocation과 여러 partition order/parenthesization을 검증한다. Partition 요청은
cold selector에서 reference가 만족할 수 있으면 fallback하고, causal-local처럼 reference가
지원하지 않는 조합이면 명시적으로 실패한다.

## Shape matrix

- sequence 1, 128, 1K, 4K
- batch 1, 2, 4
- MHA와 target GQA shape
- padding 없는 packed 또는 dense input 중 선택한 형식
- target head dimension
- 극단적으로 큰 양수·음수 score
- causal mask의 첫·중간·마지막 row

Remote suite는 S1/7/8/9/31/32/33 tile boundary, target S128/1024/4096,
B1/2/4, D64 MHA와 QH/KVH=9/3 GQA를 실행했다. Long-shape test는 Q=K=0인
prefix-average O(S) oracle을 사용해 S4096에서도 quadratic CPU reference를 만들지 않았다.
Causal-local window 0/1/3/9, fully masked zero row, 큰 finite positive/negative score gap의
첫·중간·마지막 row와 다른 context owner 거부도 포함했다. Llama integration은 현재 B1이다.

## 검증

- score-matrix reference와 online softmax parity
- 하나의 구간과 여러 구간 결과 비교
- partition merge 순서를 바꾼 결과 비교
- fully masked row의 정의된 동작
- FP32 accumulator와 target output dtype 확인
- logits와 greedy next token 회귀
- native backend load 실패 시 native exact fallback
- Python/PyTorch/Transformers가 없는 환경의 execution test

### 정확성 결과

Pinned SmolLM2 S=7에서 online 대 HF BF16 golden last logits는 cosine
`0.999985581919`, max abs `0.312500000`, mean abs `0.058640185`였다. Greedy token
`4052`와 top-10 set은 exact이고 첫 causal row는 materialized/online 간 byte-exact였다.
100회 hot execution 각각의 full logits도 byte-exact였으며 allocation accounting은 변하지
않았다. Online 대 materialized는 cosine `0.999958345718`, max abs `0.540039062`,
mean abs `0.216633855`였다.

SmolLM2 S=128 model regression은 reference/online top-1 `6354`, cosine
`0.999901240213`, max abs `0.406250000`, mean abs `0.111404342`였다. Direct S=128
GQA kernel parity는 cosine `0.999996006219`, max abs `0.001953125`였다. PR 07 explicit
reference golden trace도 그대로 통과했다.

이 pinned trace에는 PR 01 E0 v2에서 미리 정한 final-logits cosine/max/mean threshold 세
개만 재사용했다. FP32 comparator, relative-error metric과 31-prompt worst-case corpus를
재실행하거나 전체 PR 01 gate를 활성화했다고 주장하지 않는다.

## 성능 판단

- attention kernel latency
- end-to-end prefill latency
- TTFT 영향
- workspace와 peak VRAM
- layout conversion 비용
- score matrix materialization bytes
- estimated/observed HBM traffic
- partial-state merge overhead
- backend cold-load overhead
- fallback 비율

backend 자체는 빨라도 앞뒤 transpose/copy 때문에 전체가 느려질 수 있으므로 trace로 확인한다.

### 원격 성능 결과

RTX 4090/sm89, BF16 dense BSHD B1/QH9/KVH3/D64에서 synchronized native execute를
측정했다. 표의 reference는 correctness-first 4-stage native baseline이다.

| S | reference median | online median | speedup | reference score/workspace | online score/workspace |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.131209 ms | 0.050771 ms | 2.584× | 294,912 B | 0 B |
| 1,024 | 2.804055 ms | 0.848067 ms | 3.306× | 18,874,368 B | 0 B |
| 4,096 | 43.465677 ms | 10.008649 ms | 4.343× | 301,989,888 B | 0 B |

S128/1K/4K measured sample 수는 각각 10/5/2이며 raw sample 순서를 보존했다. S4096
median은 구현된 upper-middle 정의에 따라 두 sample 중 큰 값이다.

Pinned SmolLM2 S=128 prepared full-forward `execute + stream.synchronize` proxy는
reference median `7.565736 ms`, online `5.125984 ms`로 `1.476×`였다. Tokenization,
model load, decode와 sampling은 제외했으므로 serving TTFT가 아니라 prefill execute proxy다.

Nsight Compute에서 같은 shape의 한 call을 profile한 total DRAM counter 합은 다음과 같다.
Profiler replay duration은 raw latency로 사용하지 않았다.

| S | reference DRAM | online DRAM | traffic 감소 |
|---:|---:|---:|---:|
| 128 | 1,167,488 B | 260,096 B | 77.72% |
| 1,024 | 58,658,304 B | 1,980,544 B | 96.62% |
| 4,096 | 2,330,735,104 B | 7,878,784 B | 99.66% |

Online은 한 kernel, reference는 QK/mask/softmax/AV 네 kernel이다. Online trace의
`materialized_score_bytes`, `workspace_bytes`, `layout_copy_bytes`는 모두 0이며 profiler에도
conversion kernel이 없다. DRAM traffic 감소율을 latency speedup이나 peak-VRAM 감소율로
해석하지 않는다. CUDA allocation accounting은 live bytes이지 observed peak VRAM이 아니다.

## 비범위

- decode attention
- paged cache
- variable-request continuous batching
- query-aware page pruning
- random-feature 또는 Nyström 근사 attention
- custom universal FlashAttention
- 모델 재학습이 필요한 attention 교체
- Triton production runtime
- Python/PyTorch attention fallback

현재 production online capability는 BF16 dense contiguous BSHD, D64, compatible AOT CUDA
architecture, single-partition prefill로 제한된다. CUDA Graph capture, native split-K merge와
non-contiguous view는 `false`로 선언된다. Online은 causal-local을 지원하지만 다른 capability
때문에 online이 거절된 causal-local 요청은 reference가 대신할 수 없어 명시적으로 실패한다.
Native code는 Rust crate에 AOT link되며 backend별 runtime `dlopen`은 없다. Linked availability
부재는 cold selector의 fallback/error 계약으로 다룬다.

결과는 `RUSTINFER_CUDA_ARCHITECTURES=89`, RTX 4090, CUDA 12.8.93, driver 580.173.02에
귀속된다. 다른 architecture/toolkit은 동일 gate를 다시 실행해야 한다. NaN/±Inf와 empty row
동작은 API/CPU merge contract에 정의했지만 model-level E0 evidence는 finite input 범위다.

## 검증 provenance와 artifact

- Executable source snapshot: `73cbdad9e0f6b04dd46a9e719be33ae050aa4836`, clean tree
- Source bundle SHA-256:
  `185bb513206ccee51c7c1de5aadf134d9de278e2fc6494b55953cefe5680d6fe`
- Container image ID:
  `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- Remote raw evidence:
  `server-4096:/home/psyche/rustinfer-artifacts/pr08/73cbdad9e0f6b04dd46a9e719be33ae050aa4836`
- Remote `SHA256SUMS` SHA-256:
  `e4dd63012044b84532990bc8150e000bb61201a6deedb296a47e4a4605411212`
- Version-controlled result:
  [PR 08 online prefill run001](../benchmarks/results/20260825T053620Z-rustinfer-online-prefill-pr08-run001/README.md)

Remote container는 `--network none`, Cargo `--locked --offline`, read-only source로 실행했고
Python/Python3가 없음을 기록했다. Workspace all-target/all-feature test, strict Clippy,
direct GPU test 8개, 기존 materialized attention regression 2개와 PR 07 golden regression이
통과했다. Compute Sanitizer memcheck는 direct/local-mask/SmolLM2 100회 경로에서 error 0,
leak 0이고 racecheck는 error/warning 0이다.

`0ac1864…`, `5edd00c…`, `771a75a…`의 실패 로그는 각각 별도 append-only evidence root에
보존했고 소급 통과시키지 않았다. 마지막 실패는 강화된 top-10 gate가 score rounding boundary
누락을 검출한 것이며, QK와 scale 뒤 register-only BF16 rounding을 복원한 뒤 새 snapshot에서
전체 gate와 성능을 다시 실행했다.

## 완료 기준

- [x] reference와 online/optimized backend parity
- [x] `OnlineSoftmaxState` merge test 통과
- [x] target prefill shapes 모두 실행
- [x] optimized path에서 score matrix 전체를 HBM에 materialize하는지 여부가 측정됨
- [x] unsupported 조합이 native fallback 또는 명시적 실패로 처리됨
- [x] baseline 대비 prefill 수치가 raw result로 보존
- [x] hidden copy/contiguous 비용이 profiler에 표시됨
- [x] production backend가 Python 없는 환경에서 실행됨

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)
