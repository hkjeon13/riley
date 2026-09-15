# PR-N02B — cuBLASLt BF16 bias epilogue qualification

상태: P0/V2b가 Qwen2.5-3B P2048 Q/K/V에서 Riley의 strict staged bias를 exact하게 검증했고, N02B는 cuBLASLt fused BIAS contract가 같은 HF actual module output과 BF16 exact함을 확인했다. candidate는 여전히 **별도 native qualification surface**다. 기본 serving selector, CUDA Graph integration, HTTP scheduler는 바꾸지 않았다.

## 목표

Q/K/V projection의 `GEMM → row_bias_add` 두 GPU launch와 BF16 output read/write를 cuBLASLt `CUBLASLT_EPILOGUE_BIAS` candidate 하나로 줄일 수 있는지 확인한다. cuBLASLt BIAS는 D의 packed row-length bias를 broadcast하고 final postprocessing 전 적용한다. Riley의 row-major logical output은 기존 column-major TN mapping으로 cuBLASLt D rows가 projection output width가 되므로, bias 방향을 바꾸지 않는다. [cuBLASLt epilogue API](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltepilogue-t), [bias pointer contract](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltmatmuldescattributes-t)를 따른다.

이 연산은 strict `BF16 GEMM → BF16-to-FP32 add → BF16 RNE`와 다른 rounding을 할 수 있다. strict path를 교체하거나 fused result를 strict-equivalent라고 표시하지 않는다.

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

## P1 server generation discriminator — 2026-09-16

`43981337`의 실제 server selector를 같은 Qwen2.5-3B P2048/O128 조건에서
`strict-staged-v1`과 `cublaslt-bias-epilogue-experimental-v1`로 각각 실행했다.
두 run 모두 graph disabled, canonical reduction, native D128 paged attention,
GPU greedy, one request, fixed 128-token output을 사용했다. 시작 receipt는 각
모드가 fallback 없이 실제 선택됐음을 확인했고 HTTP/SSE, token count, usage 및
graceful shutdown도 정상이다.

서버의 기존 vLLM token reference는 첫 token이 `374`인 반면, 같은 immutable
checkpoint를 local-only HF eager BF16 container에서 직접 token ID로 실행한
cache-on/cache-off reference는 모두 `304`로 시작했다. HF reference는 padded
vocabulary tail을 selection 전에 mask하고, eager cache-on와 full-prefix
cache-off 128 token ID hash와 text가 동일한 경우만 qualified로
기록했다. 이는 serving 경로에 Python을 넣은 것이 아니라 checkout 밖
create-only numerical artifact다.

| 비교 대상 | HF cache-on과 첫 불일치 | 같은 위치 token 일치 / 128 | 판정 |
|---|---:|---:|---|
| HF eager cache-off | 없음 | 128 | HF internal cache parity 통과 |
| Riley fused Q/K/V bias epilogue | 8 | 108 | HF보다 가까우나 full generation exact 불통과 |
| Riley strict staged bias | 3 | 19 | 현 strict arithmetic profile의 별도 수치 계약 |
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
실행하지 않는다. 다음 correctness batch는 HF per-step logit/top-k artifact와
Riley prefill/decode logits를 step 8에서 대조해 최초 차이가 Q/K/V projection
뒤 attention/KV/decode 중 어디에서 생기는지 분리한다. strict와 fused는
각각 독립 numerical profile로 유지하며, exact generation gate 또는 사전
선언한 profile-specific quality gate를 통과한 경우에만 동일 profile 내부의
serving ABBA 및 vLLM 비교로 진행한다.

## hardware scope

Ada SM89에서 first qualification을 실행한다. Hopper, Blackwell, multi-GPU는 static architecture allow-list로 자동 enable하지 않는다. device·toolkit·cuBLASLt version·descriptor·shape·alignment·workspace 별 heuristic과 `AlgoCheck` receipt가 있을 때만 candidate가 준비된다. CUDA 12.8.1 release notes의 Blackwell small-`M` fixed issue를 고려해 Blackwell decode `M=1`은 12.8.1 미만에서 skip하고, 지원 toolchain에서도 same artifact gate를 다시 실행한다. [CUDA 12.8.1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).

## 다음 PR과 롤백

이 PR이 native quality와 operator AB gate를 통과해도 serving default를 바꾸지 않는다. 다음 PR만 `LlamaProjectionBiasMode::{StrictStagedV1, CublasLtBiasEpilogueExperimentalV1}`로 Q/K/V prefill·decode dispatch에 제한적으로 연결하고, graph capture는 또 다른 PR로 분리한다. 동일 모델·revision·workload·concurrency로 ABBA serving 반복을 수행해 throughput, TTFT, TPOT, P95/P99, failure rate, quality를 vLLM과 함께 기록한 뒤에만 승격한다.

fused gate 또는 operator AB가 실패하면 plan/feature를 비활성으로 남긴다. strict path는 무변경이므로 rollback은 selector 연결을 하지 않는 것으로 끝난다. strict arithmetic을 보존한 launch/host overhead 후보는 GEMM→row-bias composite CUDA Graph 경로로 새 PR에서 평가한다.
