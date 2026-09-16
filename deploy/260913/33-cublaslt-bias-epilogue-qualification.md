# PR-N02B — cuBLASLt BF16 bias epilogue qualification

상태: P0/V2b가 Qwen2.5-3B P2048 Q/K/V에서 Riley의 strict staged bias를 exact하게 검증했고, N02B는 cuBLASLt fused BIAS contract가 같은 HF actual module output과 BF16 exact함을 확인했다. candidate는 여전히 **별도 native qualification surface**다. 기본 serving selector, CUDA Graph integration, HTTP scheduler는 바꾸지 않았다.

## 목표

Q/K/V projection의 `GEMM → row_bias_add` 두 GPU launch와 BF16 output read/write를 cuBLASLt `CUBLASLT_EPILOGUE_BIAS` candidate 하나로 줄일 수 있는지 확인한다. cuBLASLt BIAS는 D의 packed row-length bias를 broadcast하고 final postprocessing 전 적용한다. Riley의 row-major logical output은 기존 column-major TN mapping으로 cuBLASLt D rows가 projection output width가 되므로, bias 방향을 바꾸지 않는다. [cuBLASLt epilogue API](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltepilogue-t), [bias pointer contract](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltmatmuldescattributes-t)를 따른다.

이 연산은 strict `BF16 GEMM → BF16-to-FP32 add → BF16 RNE`와 다른 rounding을 할 수 있다. strict path를 교체하거나 fused result를 strict-equivalent라고 표시하지 않는다.

## Serving 비교표 보고 계약

의미 있는 최적화 batch가 끝날 때마다 아래 표를 갱신한다. 표에는 실제 serving의 matched
comparison만 넣는다. 즉 Riley baseline, Riley candidate, vLLM이 **같은 model
revision·checkpoint, GPU/driver/toolkit, workload, prompt/output distribution,
concurrency, server option과 warmup 정책**에서 실행되고, backend 순서를 AB/BA로
교차한 반복 run이어야 한다. 각 수치는 반복 run의 median과 dispersion (IQR 또는
사전 선언한 95% bootstrap interval)을 함께 기록한다. operator event, checkpoint
load를 포함한 ignored test, correctness diagnostic의 wall time은 이 표의 성능 수치가
아니다.

shared-host I/O PSI는 각 run의 공변량으로 함께 남긴다. PSI는 sample을 filter·weight·보정하거나
재시도 결과를 선택하는 데 절대 쓰지 않는다. quality gate를 통과하지 못한 profile은
성능 비교 대상이 아니며 빈 칸을 추정값으로 채우지 않는다.

현재는 **유효한 vLLM 수치 비교가 없다.** P9는 strict raw-Q가 PyTorch의
`reduced_precision_reduction=false` control과 byte-exact함을 분리해 보였지만,
full-sequence projection·full-forward·serving quality gate를 통과시키지는 않는다.
cuBLASLt candidate도 P7에서 같은 eligibility를 얻지 못했다. 아래의 이전 operator AB receipt는
end-to-end serving AB/BA가 아니므로
throughput·TTFT·TPOT·tail latency 수치로 옮기지 않는다.

| Batch | Quality gate | Riley baseline | Riley candidate | vLLM throughput | TTFT | TPOT | P95 | P99 | Failure rate | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| P6: P2051 layer-stage discriminator | Blocked — strict/fused 모두 `layer0.q_proj.last`부터 non-exact | strict staged: serving eligible 아님 | cuBLASLt BIAS: serving eligible 아님 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | P2051 projection-boundary exactness를 먼저 판정; 통과 전 vLLM 성능 비교 보류 |
| P7: P2051 full-sequence projection-boundary | Blocked — strict raw Q가 HF no-bias shadow부터 non-exact이고, strict+row-bias/fused Q/K/V의 다섯 endpoint 모두 non-exact | strict raw/row-bias: serving eligible 아님 | cuBLASLt BIAS Q/K/V: serving eligible 아님 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | M=2051 geometry/alignment discriminator를 먼저 판정; 통과 전 vLLM 성능 비교 보류 |
| P8 r2: P2051 raw-Q geometry | Blocked — F2048/F2051/F2052/F2080/F2176/F2304의 raw Q prefix가 모두 HF no-bias shadow와 non-exact | strict raw-Q geometry: serving eligible 아님 | cuBLASLt BIAS: P7 기준 serving eligible 아님; P8의 대상 아님 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 테스트한 M/padding/anchor 범위의 geometry 가설은 refuted; input/weight/layout/arithmetic을 다음에 판별하고 selector 변경·vLLM 성능 비교는 보류 |
| P9: P2051 raw-Q BF16 arithmetic policy | Diagnostic pass — current strict raw Q가 두 `reduced_precision_reduction=false` control과 exact, P7 default control과 non-exact | strict raw-Q: full-forward/serving eligible 아님 | 없음 — selector 변경 없음 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | P7 default의 reduced-precision-reduction-on 결과를 재현하는 native BF16 candidate를 full-sequence·full-forward에 qualification한 뒤에만 vLLM AB/BA를 실행 |
| P10 r2: P2051 direct cuBLAS BF16 feasibility | Diagnostic pass — direct cuBLAS default가 P7 default와 exact이고, disallow-reduced control이 Riley/P9 reduced-off와 exact; repeat exact | strict raw-Q: reduced-off control exact, full-forward/serving eligible 아님 | direct cuBLAS default raw-Q: P7 default exact, diagnostic only; full-forward/serving eligible 아님 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 동일 native arithmetic candidate를 Q/K/V·full-forward quality gate로 확장한 뒤에만 serving selector와 vLLM AB/BA를 평가 |
| P10 r3: Riley native direct cuBLAS raw-Q | Diagnostic pass — feature-gated native/Rust probe가 P7 default raw-Q와 BF16 exact `0 / 4,200,448`; repeat·allocation accounting pass | strict raw-Q: reduced-off control exact, full-forward/serving eligible 아님 | direct cuBLAS default raw-Q: Riley native/Rust boundary까지 exact, diagnostic only; full-forward/serving eligible 아님 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 없음 — matched AB/BA 미실행 | 동일 profile을 raw Q/K/V, bias boundary, full-forward quality gate에 순차 확장한 뒤에만 selector integration과 vLLM AB/BA를 실행 |

P7은 full-sequence projection correctness receipt이며 vLLM serving result가 아니다.
그 뒤의 각 의미 있는 최적화 batch는 quality gate 결과와 동일 조건의 Riley
baseline/candidate/vLLM AB/BA receipt가 모두 갖춰진 경우에만 이 행을 수치로
갱신한다.

## P9 raw-Q BF16 arithmetic-policy result — 2026-09-16

P9는 serving path 밖에서 같은 P7 `layer0.input_norm`, checkpoint의
`layer0.q_proj.weight`, `M=2051/N=2048/K=2048` raw no-bias Q projection만 묶었다.
Python은 immutable offline oracle artifact를 만들고 재검증하는 데만 사용했다. Rust
consumer는 같은 bytes를 `CudaPreparedGemm`으로 두 번 실행해 반복 BF16 hash, allocation
accounting, P7 binding, checkpoint Q-weight binding을 검증한 뒤 세 PyTorch policy output과
bytewise 대조했다. 기본 serving selector, CUDA Graph, scheduler, HTTP path는 바꾸지 않았다.

최종 Rust receipt는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-bf16-arithmetic-rust-consumer-r3-20260916T065628Z/`에 있다.
source revision은 `7aa5744afccde53cf5141b1e98702db43d49bfa0`, receipt SHA-256은
`8fff47dc65fe966eb368fd0dab574fb83f24204b7751b6df18ca7c5ff6ad2642`다. P9 oracle manifest와
sidecar는 각각 `3079d4b1e6e3e654b92431d69cc0378ece76e87f50bb736710f95c487d61c0ed`,
`11fe03272a5cbfcf2aed5ea7adee3232641aa181d1eb92f10736897c15d3ac08`로 고정했다.
`SHA256SUMS`의 모든 파일을 재검증했다. single ignored test는 249.29초였지만 checkpoint
load와 artifact validation을 포함한 correctness diagnostic이므로 throughput·TTFT·TPOT으로
해석하지 않는다.

| P9 oracle policy | PyTorch backend / flags | Riley strict raw-Q BF16 comparison | Strict output SHA-256 |
|---|---|---|---|
| P7 default | Cublas, reduced reduction on, split-K on | non-exact: 1,087,133 / 4,200,448; max abs 0.0625 | `5ddaf741ef847ffb997041b2b4cfed9811fdacc64d7fcca9a3ca27dd48166eb5` |
| reduced control | Cublas, reduced reduction off, split-K on | exact: 0 / 4,200,448 | `5ddaf741ef847ffb997041b2b4cfed9811fdacc64d7fcca9a3ca27dd48166eb5` |
| full-reduction control | CublasLt, reduced reduction off, split-K off | exact: 0 / 4,200,448 | `5ddaf741ef847ffb997041b2b4cfed9811fdacc64d7fcca9a3ca27dd48166eb5` |

RTX 4090/SM89에서 strict plan은 deterministic algorithm id 21, tile 15, stages 12,
`split_k=1`, `reduction_scheme=0`, workspace 0 B였다. P9가 보이는 범위는 **이 raw-Q
operand와 이 GPU에서 strict output이 reduced-off oracle과 대응한다**는 사실이다. 이는
global root cause, full-forward numerical correctness, fused selector 승격, serving stability,
또는 Riley-vLLM 성능 우위를 뜻하지 않는다. 따라서 다음 meaningful batch는 P7 default의
reduced-precision-reduction-on 결과를 재현하는 native BF16 candidate를 strict default와 분리해
full-sequence projection과 full-forward quality gate까지 확장하는 것이다. 그 batch가 통과하면 동일 model/checkpoint/workload로
Riley baseline·candidate·vLLM AB/BA serving receipt를 만들어 위 표의 빈 성능 열을 갱신한다.

## P10 r2 direct cuBLAS BF16 feasibility result — 2026-09-16

P10 r2는 P9와 동일한 immutable P2051 raw no-bias Q operand만 사용해 direct cuBLAS의
두 arithmetic policy를 분리한 diagnostic이다. artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-cublas-bf16-feasibility-r2-20260916T071002Z/`에
있다. P9 sidecar SHA-256은
`11fe03272a5cbfcf2aed5ea7adee3232641aa181d1eb92f10736897c15d3ac08`로 다시 고정했고,
probe output SHA-256은
`fe43ba8e447e430b971ea7c49962db5f9c84ad05ec02d58c03d7a6d0aff935f6`다. source와 runner
SHA-256은 각각
`498a8f7d3911729bfc3f1ac76f2da5d4c50b4c9e6097426d4ff6b203abe8fb65`,
`c91e3c414eeb75fa1de121a0f9d336b99f0954f098b0dec28f2312ab10001595`로 기록했다.

| direct cuBLAS policy | Reference endpoint | BF16 byte comparison | 반복 실행 |
|---|---|---:|---|
| default reduced-precision-reduction on | P7 default raw-Q | exact: `0 / 4,200,448` unequal | exact |
| disallow reduced-precision reduction | Riley strict / P9 reduced-off raw-Q | exact: `0 / 4,200,448` unequal | exact |
| crossed controls | 반대 policy endpoint | non-exact: `1,087,133 / 4,200,448`; max abs `0.0625` | 해당 policy의 반복 hash는 exact |

이 결과는 native direct cuBLAS policy가 이 pinned raw-Q endpoint에서 P7 default와
P9 reduced-off control을 구분해 재현할 수 있음을 보이는 **diagnostic**이다. Q/K/V 전체,
bias, full-forward, CUDA Graph, scheduler, selector 승격, 안정성, throughput, TTFT, TPOT,
P95/P99, failure rate 또는 vLLM 우위를 검증하지 않았다. 따라서
`performance_claim_eligible=false`와 `vllm_comparison_eligible=false`를 유지하며, 표의
vLLM 열은 matched same-model/hardware/workload/concurrency AB/BA serving receipt가 생길
때까지 수치로 채우지 않는다.

## P10 r3 Riley native direct-cuBLAS raw-Q result — 2026-09-16

P10 r3는 P10 r2의 standalone feasibility 결과를 Riley의 feature-gated native C ABI와
Rust ownership boundary에서 재현한 one-shot quality diagnostic이다. `cuda-cublas-gemm-probe`
feature가 켜진 build에만 direct `cublasGemmEx` plan과 `libcublas` link를 넣었다. 기본 CUDA
archive, serving selector, CUDA Graph, command batch, scheduler, HTTP path에는 연결하지 않았다.
native plan은 `OP_T(Wc)`, `OP_N(Xc)`, BF16 I/O, FP32 compute,
`CUBLAS_DEFAULT_MATH`, host scalar pointers, atomics disabled로 실행하고 매 call을 같은 stream에서
synchronize한다.

원격 RTX 4090/SM89, CUDA runtime `12080`, cuBLAS `120804`에서
`451687a6bf61c6f1f0535069a42856bcb0ef9a82`를 빌드하고 immutable P9 manifest/sidecar를
입력으로 실행했다. receipt는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-cublas-rust-consumer-r1-20260916T074841Z/`에
있다. receipt SHA-256은
`60fcd6661ecdf2e4354c203da4daf36362f7752d0838996c45ac503c86883a1e`, integrity manifest는
`0817a769cfa237651e2a363108351512f11554378bf9b787c54692a4430c0ab0`이며, 생성 뒤
`SHA256SUMS`의 모든 입력 파일을 재검증했다.

| Gate | 결과 |
|---|---|
| P7 default raw-Q BF16 comparison | exact: `0 / 4,200,448`; output SHA-256 `9353470bc0d4110218b9c4584ea782257d5a59888db5a3c0479131e5ba46ec8f` |
| repeated output | byte-exact |
| allocation accounting | unchanged, plan/context close 뒤 zero |
| native metadata | backend `3`, `(M,N,K)=(2051,2048,2048)`, math/pointer/atomics modes 모두 `(0,0)` |
| GPU memory | 시작/종료 모두 `335 MiB` |

I/O PSI `some/full avg10`은 시작 `42.76/37.35`, 종료 `32.16/29.28`로 같이 남겼다. 이는 shared-host
상태의 공변량이며 test 실행·결과 선택·보정·재시도에 사용하지 않았다. ignored test의 9.14초에는
artifact validation과 transfer가 포함되므로 throughput, TTFT, TPOT, P95/P99로 해석하지 않는다.

따라서 P10 r3는 **pinned raw-Q arithmetic correspondence만** 확정한다. raw K/V, bias,
full-forward, model output quality, selector integration, concurrency stability와 vLLM serving 성능은
아직 검증되지 않았고, 이 표의 vLLM 성능 칸은 의도적으로 비워 둔다.

## 변경 묶음

1. native C ABI에 기존 `RileyCudaGemmPlan`과 분리된 `RileyCudaBiasGemmPlan`을 추가한다. create는 `epilogue=BIAS`만 받아 cold phase에서 descriptor, deterministic heuristic, `cublasLtMatmulAlgoCheck`, bounded workspace를 준비한다. current strict GEMM ABI는 `epilogue=NONE`만 계속 허용한다.
2. Rust에 `CudaPreparedBiasEpilogueGemm`과 `BiasGemmParams { input, weight, bias, output, workspace }`를 추가한다. hot path에서는 allocation, descriptor creation, heuristic selection을 하지 않는다. foreign context, short span, overlap, changed shape/dtype, poisoned plan을 fail closed 한다. execute failure 뒤 strict fallback을 같은 request에서 시도하지 않는다.
3. Qwen representative shape GPU qualifier를 추가한다. Q `[M, 2048, 2048]`, K/V `[M, 256, 2048]`, `M=1/8/32/2048`를 사용한다. actual Qwen weight/bias와 V2b artifact의 HF actual output을 이용한 layer gate는 native plan이 준비된 뒤 별도 ignored GPU test로 둔다.
4. 다음 단계에서 native event AB receipt를 strict pair와 fused candidate에 대해 같은 stream, workspace cap, warmup/repeat/order로 기록한다. 이는 operator receipt이며 serving 성능 결과가 아니다. 이 문서의 current receipt는 numerical/lifecycle gate만 포함하고 timing claim은 하지 않는다.

## 수치와 lifecycle gate

- V2/V2b strict artifact hash·metrics와 existing row-bias tests는 변하지 않아야 한다.
- fused candidate의 Q/K/V output은 사전에 선언한 HF **actual module output** BF16 gate를 통과해야 한다. 결과를 본 뒤 tolerance를 완화하지 않는다. 실패하면 fused profile은 보류한다.
- each shape/bias pointer에서 repeated BF16 hash, deterministic algorithm identity, zero allocation growth over hot repeats, plan/context close 뒤 zero accounting을 확인한다.
- input/weight/bias/output/workspace의 overlap, bad bias length, invalid dtype, foreign context, unsupported heuristic, `AlgoCheck` rejection은 fail closed 한다. cold prepare의 unsupported case만 strict caller fallback이 가능하고, execute failure는 plan poison/error다.
- run receipt에는 GPU, compute capability, CUDA/cuBLASLt version, descriptor fields, selected algorithm identity, workspace cap, shape, alignment, and fallback reason을 적는다.

## 원격 correctness receipt — 2026-09-15

`b30fd821847da0889175fddfd1d026d65792bc10`에서 RTX 4090 (SM89, CUDA 12.8.1/cuBLASLt 12.8.04)으로 actual-HF gate를 실행했다. offline safetensors sidecar의 `layer0.input_norm`과 unmodified HF `q_proj`/`k_proj`/`v_proj` output을 읽고, Rust `LoadedModel`이 같은 pinned checkpoint의 Q/K/V weight·bias binding을 검증해 올린 뒤 `CudaPreparedBiasEpilogueGemm`을 직접 실행했다. 실행 중 Python은 호출하지 않았다.

외부 create-only artifact는 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-bias-epilogue-hf-module-20260915T201212Z/`다. source revision은 `b30fd821847da0889175fddfd1d026d65792bc10`, result JSON SHA-256은 `1966c5e24eab16f300674dd2c3572a98b8db92395b5ef37b195e6bc2e29e2b3b`, integrity manifest SHA-256은 `33a8f7b43726a3331a0ad026280ad7cac3d23843d69f48eec2c3f5276bd1039f`다. retained log의 single ignored test는 90.41초에 통과했지만 checkpoint load와 sidecar verification을 포함한 correctness diagnostic 시간이므로 latency나 throughput으로 해석하지 않는다. `SHA256SUMS`의 10개 file hash도 다시 검증했다.

| Projection | Shape `[M, N, K]` | HF module BF16 exact | Unequal / total | Selected algorithm | Workspace | Repeated output / allocation |
|---|---:|---:|---:|---|---:|---|
| Q | `[2048, 2048, 2048]` | yes | 0 / 4,194,304 | id 5, tile 20, stages 7 | 0 B | byte-identical / unchanged |
| K | `[2048, 256, 2048]` | yes | 0 / 524,288 | id 6, tile 15, stages 17 | 0 B | byte-identical / unchanged |
| V | `[2048, 256, 2048]` | yes | 0 / 524,288 | id 6, tile 15, stages 17 | 0 B | byte-identical / unchanged |

모든 선택은 deterministic, `split_k=1`, `reduction_scheme=0`이고 plan/context close 뒤 allocation accounting은 zero였다. GPU memory는 시작/종료 모두 335 MiB였다. I/O PSI는 시작 `some/full avg10=4.36/4.06`, 종료 `16.85/15.29`로 기록했다. 이 값은 공유 host 상태를 설명하는 공변량일 뿐 결과를 걸러내거나 보정하지 않는다.

따라서 이 정확한 Qwen revision·P2048 geometry에서만 `candidate_selector_eligible=true`가 되었다. 이는 arithmetic·lifecycle gate의 통과를 뜻하며, default selector 변경, CUDA Graph, full-model quality, TTFT/TPOT/throughput, vLLM 비교는 아직 증명하지 않는다.

## 원격 operator AB receipt — 2026-09-15

`f3fccb279e941369de6a28d2ef5fa859f2477c18`에서 같은 RTX 4090/SM89와 16 MiB workspace cap으로 strict `prepared GEMM → row-bias`와 fused BIAS를 비교했다. H2D, D2H, cold prepare는 event 구간 밖에 두고, 각 backend를 한 command batch에 넣어 CUDA event로 측정했다. 각 child process는 backend별 warmup 8회 뒤 ABBA 24 paired round를 실행했고, 이를 독립 process 5회로 반복했다. strict/fused는 의도적으로 다른 rounding contract이므로 두 결과끼리의 equality는 요구하지 않았다. separate actual-HF gate가 fused contract의 correctness를 맡는다.

외부 create-only artifact는 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-bias-epilogue-ab-20260915T202405Z/`다. summary SHA-256은 `317135d2399877a19c7365d8b6a804ec3a40f03af622df5b7d2d821f79f9ffa4`, manifest SHA-256은 `c86301f4514f5469c045c0e9b20f79046e2aae1faa1ba02920812206e67d96de`이며 `SHA256SUMS`의 43개 file hash를 재검증했다. 모든 child에서 deterministic repeated output과 zero allocation delta를 확인했다.

아래 speedup은 **strict median / fused median**이다. 각 값은 5개 child-run median의 median이며 bracket은 fixed-seed 10,000 paired bootstrap의 child-median 95% interval이다. 각 case에는 총 120 ABBA paired round가 남아 있다.

| Projection | `M=1` | `M=8` | `M=32` | `M=2048` |
|---|---:|---:|---:|---:|
| Q `[M, 2048, 2048]` | 1.117× [1.111, 1.137] | 1.097× [1.075, 1.113] | 1.066× [1.056, 1.070] | 1.099× [1.098, 1.100] |
| K `[M, 256, 2048]` | 1.226× [1.212, 1.233] | 1.102× [1.092, 1.111] | 1.076× [1.062, 1.077] | 1.125× [1.120, 1.130] |
| V `[M, 256, 2048]` | 1.219× [1.213, 1.231] | 1.102× [1.097, 1.106] | 1.078× [1.075, 1.082] | 1.125× [1.120, 1.131] |

대표 P2048 GPU-event median은 Q strict/fused `0.121880/0.110992 ms`, K `0.024200/0.021520 ms`, V `0.024224/0.021504 ms`다. GPU observed memory는 각 run 시작/종료 모두 335 MiB였고, utilization은 0–4%였다. I/O PSI `some/full avg10`은 run 사이 대략 `4.03/3.71`에서 `4.93/4.66` 범위였으며 이를 filter·weight·보정·재시도 선택에 사용하지 않았다.

이 operator evidence는 fused candidate를 selector-integration **검토 단계**로 올린다. 그러나 full-model quality, current graph interaction, scheduler/HTTP behavior, throughput, TTFT/TPOT, P95/P99, failure rate와 vLLM 대비는 전혀 측정하지 않았다.

## P1 historical server smoke and cache-reference correction — 2026-09-16

`43981337`의 실제 server selector를 같은 Qwen2.5-3B P2048/O128 조건에서
`strict-staged-v1`과 `cublaslt-bias-epilogue-experimental-v1`로 각각 실행했다.
두 run 모두 graph disabled, canonical reduction, native D128 paged attention,
GPU greedy, one request, fixed 128-token output을 사용했다. 시작 receipt는 각
모드가 fallback 없이 실제 선택됐음을 확인했고 HTTP/SSE, token count, usage 및
graceful shutdown도 정상이다.

서버의 기존 vLLM token reference는 첫 token이 `374`인 반면, 같은 immutable
checkpoint를 local-only HF eager BF16 container에서 직접 token ID로 실행한
full-prefix cache-off reference는 `304`로 시작했다. 이 HF artifact는 serving
경로에 Python을 넣은 것이 아니라 checkout 밖 create-only numerical artifact다.

후속 검토에서 historical artifact의 `generation.cache_on` row 1이 prefill 뒤
길이 2048 cache에 `Y0`를 logical position `2049`에서 소비한 것을 확인했다.
정확한 위치는 `2048`이어야 한다. 따라서 이 cache-on 행렬은 cached decode
reference가 아니며, 이전의 cache-on/cache-off token hash·text parity와
`HF internal cache parity` 판정은 철회한다. immutable artifact와 당시 smoke
결과는 재작성하지 않고 보존한다. cache-off 행렬은 각 row가 `prompt + teacher[:i]`
full prefix를 사용하므로 selected-token teacher reference로만 계속 사용한다.

| 비교 대상 | historical cache-off와 첫 선택 token 불일치 | 같은 위치 token 일치 / 128 | 현재 해석 |
|---|---:|---:|---|
| Riley fused Q/K/V bias epilogue | 8 | 108 | smoke 관측값; corrected cached reference 전에는 quality gate가 아님 |
| Riley strict staged bias | 3 | 19 | smoke 관측값; 별도 수치 계약의 quality gate가 아님 |
| existing vLLM workload token reference | 0 | 해당 없음 | Riley/HF correctness golden으로 사용 불가; 별도 원인 판정 필요 |

HF artifact는 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-hf-eager-generation-oracle-r2-20260915T213902Z/`에 create-only로 남겼다.
`generation-oracle.json` SHA-256은
`2129428106fca8e34a2f3883e58291c317a9a1fb14510998bfde4be9b4434df4`이고,
`SHA256SUMS` 전체 재검증을 통과했다. strict serving artifact는
`qwen3b-strict-qkv-serving-smoke-r4-20260915T212000Z`, fused artifact는
`qwen3b-fused-qkv-serving-smoke-r3-20260915T211041Z`에 보존했다. 이 실행은
shared-host I/O pressure 중 model materialization을 포함하므로 throughput,
TTFT, TPOT, latency 또는 vLLM 성능 비교가 아니다.

판정: fused selector는 default로 승격하지 않고, N06-A performance campaign도
실행하지 않는다. 다음 correctness batch는 corrected HF teacher-forced
cache-on/cache-off raw-logit artifact를 만들고, Riley scheduler-committed
prefill/decode logits와 대조하는 것이다. strict와 fused는 각각 독립 numerical
profile로 유지하며, corrected artifact에 대해 사전 선언한 profile-specific
quality gate를 통과한 경우에만 동일 profile 내부의 serving ABBA 및 vLLM
비교로 진행한다.

## P2 scheduler-committed native-D128 teacher-forced trace — 2026-09-16

`fc792554cf40f43daca5ba38b2e390dc32f85c5d`에서 real `Scheduler`와
`execute_llama_iteration_timed`를 사용하는 ignored CUDA test를 RTX 4090/SM89에서
실행했다. artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-native-d128-teacher-forced-trace-r2-20260915T222011Z/`에
create-only로 보존했다. canonical `trace.json` SHA-256은
`e5039137c2ad850de166d72bb87054a10f73dac76aefa807851017a7d43508d5`, source
stdout log SHA-256은
`1a220166da08f8b81753c69de9b74ba7b66365a1ac5827534d43d26749badd8b`이며
`SHA256SUMS`의 모든 항목을 재검증했다.

test는 C8/M32 capacity configuration을 사용하지만 실제로는 request 1개와
scheduled batch 1개만 제출했다. P2048은 32-token chunk 64회로 prefill하고,
그 뒤 1-token decode 8회를 scheduler commit까지 수행해 final prefill 포함 9개
teacher-forced row를 남겼다. HTTP, continuous-concurrency throughput, latency
timing은 측정하지 않았다. 총 223.39초에는 checkpoint load와 초기화가 포함되어
TTFT·TPOT·throughput으로 해석할 수 없다.

old cache-on 행렬은 사용하지 않았다. 각 row는 immutable historical artifact의
valid full-prefix `cache-off` selected token을 제출 뒤 scheduler가 실제 commit한
값과 비교한다. top-32 순서는 observation이며 tie-sensitive correctness gate가
아니다.

| profile | selected token 일치 / 9 | raw argmax 일치 / 9 | 첫 selected-token 불일치 step | top-32 set overlap | full raw BF16 logit hash |
|---|---:|---:|---:|---:|---:|
| `strict-staged-v1` | 8 / 9 | 8 / 9 | 3 | 20–31 | 0 / 9 |
| `cublaslt-bias-epilogue-experimental-v1` | 8 / 9 | 8 / 9 | 8 | 25–31 | 0 / 9 |

fused 후보는 이 한정된 full-prefix teacher trace에서 strict보다 더 오래 selected
token을 유지했지만, 어느 profile도 HF와 raw BF16 logits exact이 아니다. 따라서
이는 fused numerical trajectory가 더 가깝다는 diagnostic evidence일 뿐 full-model
correctness, cached KV correctness, selector 승격, serving performance 또는 vLLM
우위의 근거가 아니다. GPU observed memory는 시작/종료 모두 335 MiB, utilization은
0%였다. I/O PSI `some/full avg10`은 시작 `4.30/3.84`, 종료 `8.74/8.44`로 함께
기록했으며 공유 host 상태를 설명하는 값일 뿐 결과를 filter·weight·보정하지 않는다.

## P3 corrected HF cache-on/cache-off teacher-forced artifact — 2026-09-16

`6cfb27b234cee1515431f002f55f6c533caab8dc`에서 기존 cache-on의 off-by-one
position을 제거한 offline HF eager BF16 artifact를 만들었다. artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z/`에
create-only로 보존한다. manifest
`teacher-forced-oracle.json` SHA-256은
`35d5d9b153d73b4badc67e5eedf2e8226ca2b55c7e30eb0970cfb741031edd35`이다.
이 경로는 serving hot path가 아니다. Python은 immutable checkpoint로 numerical
sidecar를 만드는 offline oracle에만 쓰고, artifact를 만든 뒤 validator는 GPU 없이
source·workload·safetensors binding을 다시 확인한다.

producer의 post-write sidecar 검증과 별도 GPU-free validator replay가 모두
통과했고, `SHA256SUMS` closure도 재검증했다. 두 sidecar는 각각 BF16
`[128, 151936]`, raw payload `38,895,616 B`이며 full-file SHA-256은 다음과 같다.

| reference | file | full-file SHA-256 |
|---|---|---|
| full-prefix cache-off control | `cache-off-logits.safetensors` | `4ecc126c122f040e5a92068ab471f57fad2c7a76977e72d3b69ca51382bc5d4b` |
| DynamicCache cache-on scheduler reference | `cache-on-logits.safetensors` | `d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39` |

teacher token stream은 cache-off addressable greedy selection에서만 유도되며,
SHA-256은 `6030daf20e588f490e0d37ab69c93df8e19851d5c8114499e8388968c9347496`이다.
첫 16 ID는 `304, 279, 198, 13, 41233, 13874, 3989, 16, 17, 13, 715, 220, 220, 16, 17, 13`이다.
기존 workload에 고정되어 있던 vLLM 첫 ID `374`와 다르므로, 그 workload를 Riley/HF
quality golden으로 쓰거나 N06-A serving 비교를 시작할 수는 없다.

| 비교 | selected token 일치 / 128 | raw BF16 row hash 일치 / 128 | 해석 |
|---|---:|---:|---|
| cache-off vs cache-off teacher | 128 / 128 | 해당 없음 | teacher stream의 유일한 생성 근거 |
| corrected cache-on vs cache-off teacher | 128 / 128 | 1 / 128 | cached trajectory token은 일치하지만 raw logits exact parity를 뜻하지 않음 |

cache-on row 0은 P2048 prefill (`cache 0 → 2048`, position `0..2047`)이고,
row 1은 teacher `304` 한 token을 position `2048`에서 소비한다. 마지막 row 127은
teacher `220`을 position `2174`에서 소비해 cache length `2175`가 된다. 따라서
Riley scheduler trace의 primary reference는 cache-on이고, cache-off는 같은 logical
context의 full-prefix control로 남긴다. cache-on selected token이 미래 artifact에서
teacher와 달라도 그 사실을 기록하며, scheduler는 여전히 teacher stream을 입력으로
사용해야 한다.

GPU observed memory는 생성 시작/종료 모두 335 MiB였다. shared host I/O PSI
`some/full avg10`은 시작 `20.99/16.97`, 종료 `25.91/23.51`로 남겼다. 이는 artifact
생성의 host 상태 설명일 뿐 selected-token 결과를 filter·weight·보정·재시도하는
근거가 아니다. artifact 자체도 `performance_claim_eligible=false`이므로
throughput, TTFT, TPOT, P95/P99 또는 vLLM 우위를 주장하지 않는다.

## P4 vLLM prefill diagnostic과 scheduler-committed native trace — 2026-09-16

corrected HF artifact를 기준으로 vLLM과 Riley의 첫 prefill 선택 및 Riley의
teacher-forced scheduler trajectory를 분리해 확인했다. vLLM diagnostic artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-vllm-prefill-diagnostic-r3-20260916T011004Z/`에,
native trace artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-native-d128-trace-r1-20260916T012534Z/`에
create-only로 보존한다. 전자의 `run.json` SHA-256은
`4225a18900ce7c328559ce965b31ef174397f720d68c6c74bc780640f4fd9efe`,
후자의 `trace-artifact.json` SHA-256은
`4cdcfbf733b528a42620e03dc07c45cfc8852d753815c003325345288b968bef`이며,
각 artifact의 `SHA256SUMS` closure를 재검증했다.

vLLM request body는 text 재-tokenization이 아니라 P2048 canonical prompt의 정확한
`prompt_token_ids` 목록을 직접 전달했다. response에 `prompt_token_ids` echo가 없어
collector의 `input_ids_match=false`가 되었지만, 이는 mismatch 증거가 아니라 response
field 부재다. 두 diagnostic response의 `usage.prompt_tokens`는 모두 `2048`이다.
실제로 prompt를 32-token block으로 나누는 D0은
`--max-num-batched-tokens 32 --enable-chunked-prefill`과 `max-model-len=2049`를,
full prefill D1은 `--max-num-batched-tokens 2049 --no-enable-chunked-prefill`을
사용했다. 따라서 아래 D0/D1 차이는 flag만 바꾼 비교가 아니라 실제 split되는
prefill budget을 포함한 path diagnostic이다.

| correctness diagnostic | first selected token | teacher-forced rows / raw BF16 full-row hash | 현재 판정 |
|---|---:|---:|---|
| HF eager BF16 cache-off/cache-on reference | `304` | canonical reference | baseline |
| vLLM D0, actual 32-token chunked prefill | `374` | first prefill row만 수집 | HF 첫 선택과 다름 |
| vLLM D1, 2049-token full prefill | `304` | first prefill row만 수집 | HF 첫 선택과 일치 |
| Riley `strict-staged-v1` | `304` | selected `8 / 9`; full raw BF16 hash `0 / 9` | step 3에서 `16`, teacher `13` |
| Riley `cublaslt-bias-epilogue-experimental-v1` | `304` | selected `8 / 9`; full raw BF16 hash `0 / 9` | step 8에서 `15`, teacher `17` |

Riley trace는 source revision
`0cc873cb80b100031d4c7bc3ab3126f2b00ea3bc`에서 real scheduler와 native D128 paged
attention으로 P2048을 32-token prefill 64회로 commit한 뒤, HF teacher token을
강제로 입력한 8개 decode를 commit했다. 그래서 첫 token mismatch 뒤의 error가
cascade하지 않으며 strict의 step 3과 fused의 step 8을 각각 독립적으로 국소화할 수
있다. 두 profile 모두 raw BF16 full-row hash가 reference와 `0 / 9`이고 top-32 exact
row도 `0 / 9`이므로, selected token `8 / 9`만으로 quality gate를 통과했다고 해석하지
않는다. D0/D1 결과도 vLLM과 Riley의 matched serving 성능 또는 일반적인 chunked-prefill
correctness 우열을 뜻하지 않는다.

native ignored test의 292.38초에는 checkpoint load와 초기화가 포함돼 latency,
TTFT, TPOT, throughput으로 해석할 수 없다. vLLM diagnostic의 I/O PSI
`some/full avg10`은 시작 `38.11/34.39`, 종료 `48.20/36.40`이었고 native trace는
시작 `4.97/4.77`, 종료 `25.15/23.01`이었다. 공유 host 상태를 기록한 값이며 sample
filter·weight·보정·선별 재시도에는 사용하지 않았다. 이 절에는 performance table이나
vLLM 대비 성능 claim이 없다.

## P5 decode shape 및 cache-free outer control — 2026-09-16

P4의 strict step 3 selected-token 불일치가 inactive decode row padding 또는 paged
KV/scheduler에만 국한되는지 확인하기 위해, 서로 다른 두 correctness discriminator를
실행했다. 둘 다 serving benchmark가 아니라 pinned checkpoint와 canonical P2048
teacher-forced input을 쓰는 ignored GPU diagnostic이다. checkpoint load, artifact
검증, 초기화가 포함된 실행 시간은 latency·TTFT·TPOT·throughput으로 해석하지 않으며,
이 절에는 vLLM 성능 비교가 없다.

shape discriminator의 create-only artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-native-d128-shape-discriminator-r2-20260916T015601Z/`다.
source revision은 `c5126749e631762427ec1f73c8015d73c8bca8a1`, canonical
`trace-artifact.json` SHA-256은
`c0053dfb6d3fec720fc681d3302407230136f268571aa9820cf0e4574dfcb5fa`,
`run.json` SHA-256은
`0d3def7f6da95556fe52de7af86e0d890c8e794a49ad5503d0a1e9f08d6d5d06`다.
`SHA256SUMS` closure를 다시 검증했고 ignored test는 403.55초에 통과했다.

fixed policy는 prefill과 decode를 모두 `M=32`로 유지하고, active-row policy는
prefill만 `M=32`, single-request decode를 실제 `M=1`로 실행한다. 두 정책의 strict
selected token은 모두 `304, 279, 198, 16, 41233, 13874, 3989, 16, 17`이며, step 3은
HF cache-off teacher `13`과 다르다. strict M32와 strict M1은 9개 row 모두 raw BF16
logit hash, BF16 top-32, selected token이 정확히 같았다. 따라서 이 P2048/C8
single-request trace에서 inactive-row padding 또는 dense GEMM row shape는 step 3
불일치의 주된 설명으로 우선순위를 낮춘다. 이것이 모든 shape 또는 concurrency에서
padding 영향이 없다는 일반화는 아니다.

cache-free outer control의 create-only artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-native-d128-dense-control-r1-20260916T021256Z/`다.
source revision은 `3bf886de7e81ec46b5ce97a1d478193de1775463`, canonical
`trace-artifact.json` SHA-256은
`3c8f0ff13617f91e298e12eb88b5447fea8dbd2854b0b6f679bf8891f2d2b8d7`,
`run.json` SHA-256은
`9b03ec6d15140a287409ffcb789e3bd98d554871be1d5841c931caec55ae522a`, retained
source stdout log SHA-256은
`e4ddabc988691bc08f18c16fa047570e393ec928165a8d7dbff5859f21a12f09`다.
모든 `SHA256SUMS` 항목을 재검증했고 test는 425.54초에 통과했다.

outer control은 scheduler replay가 아닌 별도 materialized GQA prefill backend
`riley.cuda.materialized-gqa-prefill.bf16`를 사용했다. strict staged projection과
cache-free `PreparedLlamaForward::with_reference_attention()`으로 step 3의 정확한
입력 `P2048 + teacher[:3]` (S2051, input SHA-256
`850a1cb46f8fa5e98af1445a95d77af6621705afc14cb740ec6cb24a638f9c2c`)을 한 번
실행했다. receipt는 `scheduler_executor_replay=false`, `use_cache=false`,
`same_scheduler_engine=false`를 명시한다.

| correctness control | 실행 경로 | step 3 selected / HF cache-off | 관측된 범위 |
|---|---|---:|---|
| strict fixed M32 | native D128 paged scheduler, decode M32 | `16 / 13` | mismatch 유지 |
| strict active M1 | native D128 paged scheduler, decode M1 | `16 / 13` | M32와 raw BF16/top-32/selected 9 / 9 exact |
| strict dense cache-free outer control | materialized GQA prefill, cache 없음, scheduler와 별도 engine | `16 / 13` | raw hash false, top-32 exact false, top-32 set overlap 22 |
| fused M32 scheduler trace | native D128 paged scheduler, decode M32 | step 3 `13 / 13`; step 8 `15 / 17` | profile-specific mismatch는 별도로 유지 |

dense control의 Riley raw argmax와 selected는 모두 `16` (selected logit `10.3125`)이고
top-5는 `16, 271, 13, 198, 382`였다. HF cache-off의 raw argmax와 selected는 `13`이다.
그러므로 step 3 불일치는 paged KV, cache state, scheduler replay 또는 inactive-row
shape에만 배타적으로 존재한다고 볼 수 없다. 반면 materialized dense path와 scheduler
path는 서로 다른 engine이므로, 이 결과만으로 하나의 CUDA primitive나 layer를 원인으로
확정하지 않는다.

shape run의 I/O PSI `some/full avg10`은 시작 `27.23/24.85`, 종료 `32.24/27.16`이고,
dense control은 시작 `56.95/50.02`, 종료 `66.24/60.38`이었다. 모두 공유 host 상태를
설명하기 위해 기록한 공변량이며 sample filter·weight·보정·재시도 선택에는 사용하지
않았다.

## P6 결과: cache-free P2051 layer-stage discriminator

P6는 완료했다. offline HF eager cache-off oracle과 Rust의 cache-free materialized
reference attention이 동일한 `P2048 + teacher[:3]` 입력(S2051)을 실행했고, 각
decoder layer의 마지막 token row를 비교했다. 이 run은 scheduler, paged KV, continuous
batching과 분리된 correctness control이며 성능 claim에는 사용할 수 없다.

정상 artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-layer-stage-r2-20260916T033748Z/`다.
source revision은 `8b657f6f0dce76a15190a195ae5defe7c0767495`, HF manifest SHA-256은
`5c83bf794a50a554f87dc227ed12f6baa3cd567f401cd81cf7a364a9ec4b7577`, HF BF16
sidecar SHA-256은 `9ada5579bc8a3eeaadc67aef07e306bee8cf736533be127adad317321c74cb5f`다.
Rust comparison SHA-256은
`f948429b51f590299e11b607cf76686b87f2648d75f0e3c4d81747801917e2cb`이고, 전체
artifact integrity manifest SHA-256은
`020c22b47c5a0117ca1850bad2398b4f904ac999b1d1048369c53de8b8db6c7f`다. 21개
`P2051_INTEGRITY_SHA256SUMS` 항목을 재검증했다.

초기 wrapper는 Docker가 sidecar를 `root:root 0600`으로 만들어 Rust consumer가 읽지
못해 실패했다. 파일 내용은 SHA-256으로 검증한 뒤 Docker root에서 `a+r`만 추가했고,
동일 HF artifact를 대상으로 Rust-only rerun을 수행했다. rerun receipt는
`rust-p2051-layer-stage-permission-rerun-v2.json`이며 exit 0, test duration은
211.38초다. 이는 수치 결과를 고르기 위한 retry가 아니라 artifact 접근 권한 결함을
수정한 실행 기록이다.

| projection profile | exact stage / 52 | first non-exact stage | Q / K / V unequal elements | final norm max abs | logits max abs |
|---|---:|---|---|---:|---:|
| strict staged | 2 | `layer0.q_proj.last` | `1122 / 2048`, `94 / 256`, `179 / 256` | `9.25` | `4.4921875` |
| cuBLASLt bias epilogue | 2 | `layer0.q_proj.last` | `1028 / 2048`, `80 / 256`, `171 / 256` | `2.5625` | `1.8984375` |

두 profile에서 `embedding.last`와 `layer0.input_norm.last`만 BF16 exact였다. fused
profile은 downstream error magnitude를 줄였지만 P2051에서 Q/K/V 어느 것도 exact로
만들지 못했으며, RoPE와 attention 이전인 `layer0.q_proj.last`가 첫 관측 불일치다.
따라서 paged KV, cache, scheduler, attention softmax, MLP를 다음 병목으로 삼을 근거는
아직 없다. 다음 판별은 M=2051의 actual HF Q/K/V bias-boundary output과 cuBLASLt
algorithm/descriptor receipt를 직접 대응시켜 projection geometry·epilogue contract를
분리해야 한다.

I/O PSI `some/full avg10`은 시작 `61.91/57.54`, 종료 `34.31/30.54`였다. 공유 host
상태를 설명하는 공변량일 뿐 어떤 결과도 filter·weight·보정·재시도 선택에 사용하지
않았다.

## P7 결과: P2051 full-sequence projection-boundary discriminator

P7은 P6의 마지막-row 관측을 full `M=2051` projection boundary로 확장했다. offline
HF eager BF16 oracle은 `layer0.input_norm` 전체와 actual Q/K/V module output, 그리고
Q weight만 적용한 no-bias shadow endpoint를 고정했다. Rust는 같은 pinned checkpoint의
weight/bias를 binding한 뒤 strict raw GEMM, strict GEMM 뒤 row-bias, cuBLASLt BIAS
Q/K/V를 직접 실행했다. Python은 이 immutable oracle artifact의 producer/validator에만
쓰였고 serving hot path에는 들어가지 않았다.

정상 create-only artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-projection-boundary-r1-20260916T043652Z/`다.
source revision은 `3e4f6f1ead8727cd01e52a30e4673d1c83953606`, HF manifest
SHA-256은 `d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7`,
HF BF16 sidecar SHA-256은
`fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96`다. Rust
comparison SHA-256은
`e4a6c2f5110965580d09d82f0539b391d8a2140eb48c4880466f9a425282a727`, run
receipt SHA-256은
`91ac94338ec973a986e471cfd14c174d74731627a5bdb426218e629fc6e0d5da`다. artifact의
모든 checksum entry를 재검증했다.

| Direct endpoint | HF BF16 reference | Unequal / total | Max abs | BF16 exact |
|---|---|---:|---:|---|
| strict raw Q | `q_proj` no-bias shadow | 1,087,133 / 4,200,448 | 0.0625 | no |
| strict Q + row-bias | actual `q_proj` module output | 1,959,078 / 4,200,448 | 0.125 | no |
| cuBLASLt BIAS Q | actual `q_proj` module output | 1,809,317 / 4,200,448 | 0.125 | no |
| cuBLASLt BIAS K | actual `k_proj` module output | 166,116 / 525,056 | 0.25 | no |
| cuBLASLt BIAS V | actual `v_proj` module output | 203,242 / 525,056 | 0.0078125 | no |

다섯 endpoint 모두 non-exact이고, 첫 불일치는 bias를 적용하기 전 strict raw Q에서
나왔다. 따라서 이번 artifact가 관측한 차이는 row-bias, RoPE, attention 이전의
projection arithmetic boundary에 이미 있다. 이것은 projection 내부의 단일 원인을
확정하거나 성능 개선을 뜻하지 않으며, paged KV·scheduler·attention을 이 결과만으로
원인에서 제외하는 주장도 아니다.

모든 endpoint는 repeated output이 deterministic했고 hot repeat 중 allocation invariant를
지켰다. receipt는 RTX 4090 (SM89), CUDA runtime `12.8.0`, cuBLASLt `12.8.4`, selected
algorithm id `21`, tile `15`, stages `12`, `split_k=1`, reduction scheme `0`, workspace
`0 B`를 기록한다. I/O PSI `some/full avg10`은 시작 `26.37/24.01`, 종료
`26.27/24.04`였다. 이 값은 공유 host 상태의 공변량일 뿐 결과를 filter·weight·보정하거나
재시도 선택에 사용하지 않았다.

P7은 vLLM serving result가 아니며 throughput, TTFT, TPOT, P95/P99 또는 vLLM 대비
수치를 제공하지 않는다. P8이 이 다음 geometry discriminator의 결과를 기록한다.

## P8 r2 결과: P2051 raw-Q geometry discriminator

P8 r2는 P7의 immutable full-sequence raw-Q no-bias shadow를 재사용해 strict
cuBLASLt BF16 raw-Q GEMM의 physical `M`, explicit BF16-zero padding, tail data,
256-byte span offset, 그리고 heuristic anchor를 분리했다. 이 실행은 serving hot
path에 Python을 넣지 않은 correctness diagnostic이며 selector를 바꾸지 않았다.

artifact는
`/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2051-raw-q-geometry-r2-20260916T064500Z/`에
보존되어 있다. source revision은 `19d40a2a`이고, P7 manifest/sidecar와 checkpoint
binding의 provenance도 다시 검증했다. `SHA256SUMS`의 모든 entry가 재검증을 통과했다.
그 closure에는 artifact receipt, run script, Rust log, GPU snapshot, CPU/I/O/memory PSI
snapshot이 포함되며, 별도 run receipt도 해당 source revision과 Rust trace exit `0`을
기록한다.

| Fresh case | Physical M | Compared prefix | Unequal / total | Max abs | BF16 exact |
|---|---:|---:|---:|---:|---|
| F2048 | 2048 | 2048 rows | 1,085,440 / 4,194,304 | 0.03125 | no |
| F2051 | 2051 | 2051 rows | 1,087,133 / 4,200,448 | 0.0625 | no |
| F2052 | 2052 | first 2051 rows | 1,087,133 / 4,200,448 | 0.0625 | no |
| F2080 | 2080 | first 2051 rows | 1,087,133 / 4,200,448 | 0.0625 | no |
| F2176 | 2176 | first 2051 rows | 1,087,133 / 4,200,448 | 0.0625 | no |
| F2304 | 2304 | first 2051 rows | 1,087,133 / 4,200,448 | 0.0625 | no |

F2080의 explicit-zero tail과 repeated-last-row tail, O2051의 256-byte
input/weight/output span offset, 그리고 `A2051←F2048` 및
`A2052/A2080/A2176/A2304←F2051` anchored-plan identity control은 모두 invariant를
만족했다. 따라서 이번 결과는 **이 여섯 M, 해당 padding, tail/offset,
anchor controls에서만** geometry/alignment 가설이 raw-Q BF16 exactness를 회복시키지
못했다는 반증이다. 다른 shape, device, toolkit, weight layout 또는 arithmetic
implementation 전체에 대한 일반화는 하지 않는다.

P8도 vLLM serving result가 아니며 Riley/vLLM throughput, TTFT, TPOT, P95/P99를
제공하지 않는다. quality gate가 막혀 있으므로 다음 diagnostic은 serving selector나
benchmark가 아니라 pinned input/weight byte binding, logical/physical layout, 그리고
BF16 GEMM arithmetic/accumulation contract를 직접 분리해 조사해야 한다.

## hardware scope

Ada SM89에서 first qualification을 실행한다. Hopper, Blackwell, multi-GPU는 static architecture allow-list로 자동 enable하지 않는다. device·toolkit·cuBLASLt version·descriptor·shape·alignment·workspace 별 heuristic과 `AlgoCheck` receipt가 있을 때만 candidate가 준비된다. CUDA 12.8.1 release notes의 Blackwell small-`M` fixed issue를 고려해 Blackwell decode `M=1`은 12.8.1 미만에서 skip하고, 지원 toolchain에서도 same artifact gate를 다시 실행한다. [CUDA 12.8.1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).

## 다음 PR과 롤백

corrected HF teacher-forced artifact, scheduler trace schema binding, M32/M1 shape
control, cache-free dense outer control, P2051 layer-stage discriminator, full-sequence
projection-boundary qualification, raw-Q geometry r2 qualification은 P3–P8에서 완료했다.
P8의 다음 PR은 **projection input/weight/layout/arithmetic discriminator**다. 같은
immutable HF full-sequence endpoint를 유지하고 pinned input/weight byte binding,
logical/physical matrix layout, BF16 GEMM compute/accumulation contract를 독립적으로
대조한다. 목적은 raw-Q mismatch가 입력·weight·layout·arithmetic 중 어느 boundary에서
처음 생기는지 좁히는 것이며, selector 변경이나 성능 최적화 claim은 포함하지 않는다.
Python은 artifact producer/validator에만 남기며 serving hot path에 Python, persistent
copy 또는 scheduler fallback을 넣지 않는다.

input/weight/layout/arithmetic gate가 projection endpoint를 설명하거나 BF16 exact profile을
만든 뒤에도 cache-free dense stage와 scheduler trace가 다를 때에만 `PackedBatchV1`을
`PagedKvBlockTableV1`로 연결하는 paged-KV reference adapter를 별도 PR로 평가한다. 이
순서는 scheduler native D128 path를 dense reference attention으로 임의 교체하지 않는다.

그 discriminator와 필요한 correction 뒤 profile-specific quality gate를 통과하기 전까지
N06-A의 AB/BA throughput, TTFT, TPOT, P95/P99, failure rate 및 vLLM 비교는 blocked다.
graph capture는 여전히 별도 PR로 분리한다.

corrected quality gate 또는 operator AB가 실패하면 fused plan/feature는 opt-in으로
남긴다. strict path는 무변경이므로 rollback은 experimental selector를 비활성으로
두는 것으로 끝난다. strict arithmetic을 보존한 launch/host overhead 후보는
GEMM→row-bias composite CUDA Graph 경로로 새 PR에서 평가한다.
