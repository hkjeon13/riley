# Riley 추론 성능 개선: 라이브러리 구현 조사와 적용 계획

작성일: 2026-10-19 (KST)

Riley 조사 기준: `164ac43a60bb388995bea3bdddc0a3ab4fcc235c`

상태: 최초 설계·조사 스냅샷. 이후 사용자 요청으로 W0 실측과 Gate/Up opt-in 개선을 진행했다. 최신 결과·제한·다음 우선순위는 [A100 실행 보고서](a100-execution-path-and-bottleneck-20261019.md)를 참조한다. 아래 라이브러리 조사는 그 실행 이전의 근거다.

## 1. 결론과 권장 순서

vLLM 등은 하나의 기법으로 성능을 확보한 것이 아니다. 요청의 토큰을 효율적으로 묶는 linear 연산, 여러 요청의 KV를 처리하는 Attention, shape에 맞춘 커널 선택, CUDA Graph, CPU/GPU 작업 중첩을 결합한다. Riley는 continuous batching·paged KV·shape bucket·Graph·일부 배치 커널을 이미 갖고 있다. 따라서 이 기능들을 처음부터 다시 구현하는 계획은 적절하지 않다.

Riley에 권장하는 순서는 다음과 같다.

1. **현재 바이너리에서 실제 선택되는 연산 경로와 병목을 다시 확정한다.** 10월 17일 프로파일을 10월 19일 통합 소스 전체에 일반화하지 않는다.
2. **미완료 SM80 후보의 실행 경로 감사를 끝내고 기존 수치 보존 배치를 확장한다.** 호출 감소와 처리량 증가를 별도로 검증한다.
3. **가중치 재사용을 늘리는 projection과 요청 간 병렬 Attention을 개선한다.** 기존 reduction 순서와 BF16 중간 반올림을 보존하는 범위부터 진행한다.
4. **기존 Graph의 실제 적용 범위를 넓힌 뒤, 남은 CPU bubble이 클 때 scheduler overlap을 도입한다.**
5. **vLLM/SGLang의 batch-invariant 구현을 별도 연구 대상으로 비교한다.** 배치 불변성은 HF와의 비트 일치가 아니므로 기존 수치 계약을 대체하지 않는다.

현재 자료로 모든 모델에서 2배 개선을 약속할 수는 없다. 이 계획의 목표는 모델·동시성별 원인을 측정하고, 정확성을 유지하면서 실제 병목을 순서대로 제거하는 것이다.

## 2. 비교 전에 구분해야 할 수치 계약

| 계약 | 의미 | 이 계획에서의 취급 |
|---|---|---|
| 같은 실행의 재현성 | 동일한 전체 입력·배치·환경을 반복하면 같은 결과 | 기본 조건 |
| 배치 불변성 | 다른 요청의 수·순서·배치 구성이 바뀌어도 해당 요청의 결과가 같음 | 최적화 후보의 별도 검증 항목 |
| 기존 Riley 연산 보존 | 고정 환경에서 개선 전후의 full-vocabulary logits/컴포넌트 BF16 값이 같음 | 현재 strict 경로의 필수 게이트 |
| 특정 HF 참조와 일치 | 고정 GPU·라이브러리·checkpoint·입력에서 독립 HF 참조와 같음 | teacher logits와 생성 결과를 구분해 기록 |
| 의미적 품질 | 태스크 정답률·사용자 관점의 품질 | 토큰 일치율로 대신하지 않음 |

batch-invariant Tensor Core 커널은 내부적으로 일관된 reduction을 사용하면서도 HF의 M=1 GEMV와는 다른 값을 낼 수 있다. 반대로 일부 프롬프트의 생성 토큰이 같아도 전체 logits의 동등성이 증명되는 것은 아니다. GPU나 cuBLAS 버전을 바꾼 HF 결과도 동일한 비트 정답으로 취급하지 않는다.

수치 계약을 완화하는 새 프로파일은 이 문서에서 제안하는 **연구 선택지**다. 기존 strict 경로에 자동 적용하거나 허용 오차를 늘려 실패를 통과시키지 않는다. batch-invariant 방식이 strict 게이트까지 통과하는 geometry만 기존 경로의 적용 후보가 된다. [외부 배치 불변성 원리: S8]

## 3. Riley의 현재 구현과 증거

### 3.1 이미 구현된 기능

| 영역 | 현재 소스에서 확인한 내용 | 계획상 의미 |
|---|---|---|
| Continuous batching | scheduler가 prefill/decode iteration을 구성하고 KV 예약·commit·abort를 관리 | scheduler 부재로 단정하지 않음 |
| Paged KV | block pool, sequence reservation, block table, rollback/reclaim | 메모리 관리와 실제 Attention 계산 효율을 구분 |
| Shape bucket | 활성 토큰 수에 따른 준비된 실행 shape 선택 | 선택 shape와 실제 연산 행 수를 함께 관측 |
| 수치 보존 projection | request row group별 GEMM, singleton 배치, 제한된 small-row 경로 | 이미 있는 커널·dispatcher를 확장 |
| Graph | `segments-v1`, 일부 actual-row key 및 한정된 whole-body 경로 | “Attention은 항상 eager”라는 과거 설명을 현 소스 전체에 적용하지 않음 |
| 수치 보존 모드 | `lt-mlp-v2`, `exact-r2-v3`, `exact-r2-global-v4`, `exact-smallrows-v5` 파싱·구현 존재 | 파싱 가능 여부와 해당 GPU에서의 성능 검증을 구분 |
| 빠른 native 프로파일 | `native-bf16-v1`: merged projection, fused 연산, paged attention, decode Graph 등 | owned strict 경로의 결과와 섞지 않음 |

조사한 직접적인 수정 지점:

- [`crates/riley-runtime/src/llama/forward.rs`](../crates/riley-runtime/src/llama/forward.rs): `execute_gemm_with_row_groups`, `execute_projection_with_row_groups`.
- [`crates/riley-runtime/src/llama/executor/dispatch.rs`](../crates/riley-runtime/src/llama/executor/dispatch.rs): eligible batch 판정, layer segment/whole-body Graph, Attention dispatch.
- [`crates/riley-runtime/src/llama/executor/config.rs`](../crates/riley-runtime/src/llama/executor/config.rs), [`owner.rs`](../crates/riley-runtime/src/llama/executor/owner.rs): 명시적 모드, cold preparation, geometry·환경 제한.
- [`kernels/src/cublas_gemm_probe.cu`](../kernels/src/cublas_gemm_probe.cu): 현재 실제 row-group·small-row·exact Attention 커널과 선택 조건. 진단 이름이어도 실행 경로에 사용되므로 관측 대상에 포함한다.
- [`crates/riley-scheduler/src/scheduler.rs`](../crates/riley-scheduler/src/scheduler.rs): 단일 `inflight` iteration과 reservation 수명 관리.
- [`crates/riley-runtime/src/paged_kv.rs`](../crates/riley-runtime/src/paged_kv.rs), [`crates/riley-server/src/inference_profile.rs`](../crates/riley-server/src/inference_profile.rs).

### 3.2 성능 증거의 범위

10월 17일 RTX4090/Qwen2.5-3B 후보의 고정 workload에서는 C1 118.25 대 vLLM 130.16 tok/s, C8 153.25 대 701.79, C32 186.94 대 1499.53이었다. 같은 실험의 Graph 없는 C8 trace에서 GEMV 계열은 kernel 시간 합계의 85.6%였다. 이 분류에는 linear와 Attention QK/PV가 함께 포함되므로 이를 전부 projection 시간으로 해석하지 않는다. 동일 소스 대조 대비 kernel 수 36.52% 감소와 처리량 약 8.38% 개선도 서로 다른 지표다. [R1]

10월 16일 별도 A100/Qwen3-1.7B C1 trace에서는 per-operation→iteration completion 변경으로 GPU span이 671.47→347.03ms, stream sync 호출이 26,086→454회로 줄었다. 이는 당시 경로의 동기화 병목 증거이며 현재 모든 모델이 같은 상태라는 뜻은 아니다. API wait 시간은 GPU 실행과 겹치므로 kernel 시간에 더하지 않는다. [R2]

10월 19일 체크포인트에는 SM89 small-row/Graph 관련 통합 코드가 추가로 있다. SM80 v225 Gate/Up 후보는 제한된 수치·Graph 검증을 통과했지만, v235 QKV 후보는 실제 호출 감사에서 Lt/GemmEx 기록을 찾지 못했다. 이는 수치 불일치 판정도, 빠른 경로 실행 증명도 아니다. 후보 patch는 현재 runtime에 적용되지 않았다. Mistral 새 후보 전체 모델 검증과 heterogeneous 성능 비교도 미완료다. [R3, R4]

따라서 사용자가 관찰한 “여러 모델 평균 약 50%”는 이번에 재측정한 값이 아니다. 과거 한 모델의 결과나 다른 프로파일의 승리로 그 관찰을 설명 완료했다고 보지 않는다.

## 4. 다른 라이브러리는 어떻게 구현했는가

### 4.1 vLLM: 합친 linear, 배치별 Graph dispatch, 수치 정책 분리

`QKVParallelLinear`와 `MergedColumnParallelLinear`는 관련 projection 출력을 합친 linear 구조를 제공한다. `CudagraphDispatcher`는 배치 descriptor와 backend 적격성에 따라 Graph 실행 모드를 선택한다. 요청 수만 늘리는 scheduler와 실제 GPU 연산의 배치화를 함께 구현한 구조다. [S1, S2]

최신 batch-invariant 소스는 persistent matmul을 사용한다. 튜닝된 BF16 경로는 `(N,K)`별 K 타일을 고정하고 M bucket에 따라 M/N 타일과 launch parameter를 선택한다. 즉 배치 크기마다 K 누적 구조를 바꾸지 않으면서 실행 효율을 조정한다. 이것이 HF의 기존 M=1 결과를 그대로 보존한다는 보장은 아니다. [S3]

**Riley 적용:** persistent/tiled weight reuse와 수치 축약 구조 분리, backend별 Graph dispatcher를 참고한다. Python의 `torch.compile` 자체를 Rust runtime에 옮길 필요는 없다. vLLM의 최신 기능이 과거 비교 버전 0.27.1에 모두 존재한다고 가정하지 않는다.

### 4.2 SGLang: CPU/GPU overlap과 캐시·배치 정책의 결합

`event_loop_overlap`은 다음 GPU batch를 실행하는 흐름과 이전 결과를 처리하는 queue를 분리한다. future token/metadata를 연결해 GPU 작업과 CPU 준비·결과 처리를 겹친다. radix cache는 prefix 검색·재사용을 관리한다. [S4]

별도 deterministic mode는 batch-invariant 연산을 사용한다. 공식 문서는 FlashInfer/FA3/Triton의 기능 조합이 다르며 FlashInfer deterministic 조합에서는 radix cache가 지원되지 않는다고 명시한다. 이 호환성 제한을 모든 backend의 공통 기능처럼 취급하면 안 된다. [S5]

**Riley 적용:** 우선 metadata preparation·결과 처리의 overlap을 참고한다. 단일 inflight 및 KV commit 계약을 무시하고 단순 비동기 스레드만 추가할 수는 없다. prefix cache는 반복 prefix workload의 별도 개선으로 평가한다. [설계 배경: S6]

### 4.3 TensorRT-LLM: packed token, in-flight batching, backend heuristic

IFB는 context와 generation 요청을 함께 처리하며 입력을 padding 없이 packing한다. GPT Attention 문서는 context의 FMHA, generation의 masked Attention/XQA 계열, 적격성·성능 heuristic을 구분한다. `DecoderXQARunner::shouldUse`도 특정 커널의 지원과 실제 선택을 분리한다. [S7, S9]

**Riley 적용:** prefill/decode의 연산 특성을 구분하고 geometry·dtype·GPU별 적격성/선택 이유를 명시한다. Riley에 이미 있는 packing·chunked prefill은 재구현하지 않는다. TensorRT 엔진 전체 도입보다 dispatcher·workspace·fused 실행 설계를 참고하는 편이 현재 Rust/CUDA 구조에 맞는다. SM90/SM100용 경로를 SM80/SM89에 그대로 적용하지 않는다.

### 4.4 FlashInfer: paged KV를 직접 받는 batch Attention과 plan/run

batch decode wrapper는 KV page metadata를 받아 여러 요청을 처리하고 준비 단계와 실행 단계를 분리한다. FA2의 `fixed_split_size`는 고정 크기의 KV 분할로 merge reduction의 배치 의존성을 제어한다. 소스상 이 옵션은 Tensor Core decode에 제한되며, KV 길이에 따라 CTA 수가 달라져 Graph 호환성이 자동 보장되지 않는다. `disable_split_kv`도 별도 선택지다. [S10]

**Riley 적용:** 기존 page table을 kernel이 직접 읽는 batched Attention, cold planning, 고정 scratch를 우선 참고한다. FlashInfer 커널을 붙이는 것은 대안 backend 실험이며 HF의 materialized softmax/BF16 경계 보존이 자동 성립하지 않는다. 채택 전 native adapter·workspace 비용과 수치 검증이 필요하다.

### 4.5 llama.cpp: 작은 M 전용 커널과 조건부 dispatch

CUDA backend는 dtype·shape·GPU에 따라 float/quantized matvec, matmul, cuBLAS를 선택한다. Graph 호환성 검사와 update/reinstantiate 로직도 따로 둔다. 작은 M에서 항상 큰 GEMM이나 Graph가 유리하다고 가정하지 않는 구현이다. [S11]

**Riley 적용:** C1/R2/small-row와 큰 batch를 나눠 측정하고, 기존 small-row kernel 선택을 확장한다. quantized matvec의 구조적 아이디어와 BF16 strict 경로의 실제 연산은 구분한다. GGUF 양자화 수치를 BF16 vLLM 비교에 섞지 않는다.

## 5. 적용 가능성 및 우선순위

아래 평가는 소스와 과거 계측을 근거로 한 **계획 판단**이며 개선율 측정 결과가 아니다.

| 항목 | Riley 상태 | 수치 보존 가능성 | 기대 효과의 조건 | 우선순위 |
|---|---|---|---|---|
| 실제 fast-path·fallback 관측 | 경로/옵션 존재, 미완료 호출 감사 있음 | 연산 변경 없음 | 잘못된 비교·비활성 최적화 식별 | P0 |
| Iteration completion·staged upload | 기존 구현 있음 | 동일 plan/버퍼 순서 유지 시 높음 | 현재 per-operation/과도한 전송 대기가 활성일 때 | P0 확인 |
| SM80 bounded projection 후보 | patch 보존, runtime 미적용 | 부분 증거, 전체 검증 미완료 | 지원 geometry의 실제 배치 연산 선택 | P1 |
| 기존 small-row weight reuse 확장 | SM89의 제한된 구현 있음 | 동일 instruction/reduction 재현 필요 | C8 이상 projection 비용·가중치 트래픽이 클 때 | P1 |
| 요청 간 병렬 exact Attention | 한정된 구현 존재 | 기존 수치 단계 보존 시 후보 | 요청별 launch·KV gather가 클 때 | P1 |
| Graph 범위·key 개선 | 부분/한정 whole-body 존재 | arithmetic 변경 없이 가능 | launch/CPU gap·capture miss가 클 때 | P2 |
| Scheduler overlap | 단일 inflight 계약 변경 필요 | 산술 보존 가능, 수명·종료 처리 복잡 | host 준비/후처리가 GPU gap을 만들 때 | P2 조건부 |
| 합친 QKV/Gate-Up Tensor Core GEMM | native fast 경로 존재 | strict에서는 미보장 | 큰 M에서 weight reuse가 충분할 때 | 별도 수치 연구 |
| FlashInfer/FA2 backend | strict 대체 검증 안 됨 | 배치 불변≠기존 HF 일치 | Attention 중심·긴 문맥 workload | 별도 backend 연구 |
| Prefix cache·chunk 정책 튜닝 | paged KV/chunk 기반 존재 | chunk/cache 경계 검증 필요 | 실제 반복 prefix/긴 prefill | P3 |
| FP8/INT4·speculation·TP/PD 분리 | 이번 단일 GPU BF16 원인 분석 밖 | 별도 계약·검증 필요 | 해당 workload·장치·모델이 있을 때 | 후순위 |

Paged KV를 갖췄다는 사실만으로 weight reuse나 efficient Attention이 확보되지는 않는다. 마찬가지로 launch 수 감소, Graph hit 증가, GPU 사용률 상승만으로 처리량 개선을 판정하지 않는다.

## 6. 실행 작업 계획

### W0 — 현재 기준선 및 실행 경로 확정

**산출물:** 모델×GPU×profile별 baseline manifest, 연산 경로 표, profiler 요약.

- 실행 binary/source SHA, checkpoint/tokenizer revision, GPU UUID/SM, CUDA/cuBLAS·드라이버, compile feature, 실제 argv·환경·로드된 라이브러리를 고정한다.
- strict/native fast와 각 owned mode를 분리한다. 모델 이름이 아니라 `(dtype, SM, library, M/N/K, head/kv geometry, bias, context)`로 지원 여부를 기록한다.
- projection별 실제 M/N/K, API/algorithm, 제출 수, 사용 kernel, Graph capture/replay/fallback 이유를 수집한다. 요청 수와 padding bucket의 행 수를 구분한다.
- Nsight Systems로 H2D/D2H·host wait·CPU scheduling·kernel gap을 분리한다. GEMV를 linear QKV/O/Gate/Up/Down/LM head와 Attention QK/PV로 다시 분류한다.
- 병목 상위 kernel만 Nsight Compute로 DRAM bytes·L2 hit·achieved bandwidth·Tensor Core 사용·occupancy를 본다. 전체 계측 overhead를 성능 측정과 분리한다.

**진행 조건:** 대표 C1/C8에서 측정 시간의 주요 비용을 설명하고 fast path가 실제 실행됐음을 확인한다. 비활성 경로를 발견하면 먼저 설정·dispatch 문제를 해결한 뒤 기준선을 새로 고정한다.

### W1 — 기존 SM80 후보의 미완료 검증 해소

**대상:** 보존된 v235 patch와 v225/v234 증거. 현재 main에 곧바로 적용하지 않는다.

- 호출 감사기가 실제 바이너리의 cuBLASLt/GemmEx 호출과 Graph 내부 노드를 포착하는지 known-call 대조군으로 검증한다. 감사기가 실패한 상태를 수치 실패로 분류하지 않는다.
- Qwen3-4B와 Mistral7B에서 선택된 Gate/Up·QKV plan 및 실제 모든 row key를 확인한다. 일반 입력, zero/permuted/changed input, mixed prefill/decode를 포함한다.
- 이미 완료된 작업은 source/binary hash와 receipt가 일치하면 재사용한다. source가 바뀌거나 검증이 비어 있는 부분만 다시 실행한다.
- component→full-vocabulary teacher→generation→실제 serving 성능 순으로 진행한다. SM80 결과로 SM89를 승인하지 않는다.

**수정 지점:** [보존 후보](../benchmarks/candidates/sm80-bounded-projections-v235/README.md), `forward.rs`, CUDA plan/row-group dispatcher.

**완료 조건:** 실제 배치 경로 실행 증거, 개선 전후 BF16 일치, 독립 HF 참조 검사, 유한 workload의 처리량·tail 결과가 모두 존재한다. call count 64→4는 성능 배율로 쓰지 않는다.

### W2 — 정확성을 보존하는 projection weight reuse 확장

**원리:** 여러 요청을 동시에 계산하면서 weight tile의 load를 공유하되 각 output의 product·누적·BF16 저장 순서는 기존 경로와 맞춘다. 이미 있는 R2/small-row 커널을 출발점으로 삼는다.

- W0 상위 projection geometry부터 M=1/2/3/4/8/16/32를 측정한다. singleton batch submission과 실제 tile reuse를 구분한다.
- input/output stride, alignment, bias epilogue, accumulation tree, split-K, 중간 cast, tail mask를 기록해 기존 산술의 재현 가능성을 판단한다.
- cuBLASLt의 검증된 bounded recipe와 native small-row 커널을 각각 비교한다. algorithm ID만 같다는 이유로 다른 tile/stage/library 환경을 승인하지 않는다.
- 수치 계약과 device별 적격성에 맞는 cold-prepared policy table을 구성한다. 측정 구간에서 heuristic 탐색·autotune을 수행하지 않는다.

**수정 지점:** `kernels/src/cublas_gemm_probe.cu`, `forward.rs`, executor `gemm_plan.rs`/`owner.rs`.

**완료 조건:** component/full-model byte gate와 배치 순서·입력 갱신 검사를 통과하고, DRAM traffic 또는 kernel time 및 실제 throughput 개선이 확인된다. 동일 reduction을 재현할 수 없는 Tensor Core 후보는 strict 경로에서 제외한다. FP32 누적 사용만으로 일치를 가정하지 않는다.

### W3 — 기존 exact Attention의 요청 간 병렬성 확대

**원리:** 각 요청의 수학 연산은 유지하면서 CTA/work item을 여러 요청·head에 배분하고, 가능한 KV layout 변환과 gather를 줄인다.

- 현재 exact Attention의 지원 geometry·길이와 Graph 적격성을 먼저 목록화한다. 이미 지원하는 Smol/다른 제한 경로를 다시 구현하지 않는다.
- 첫 후보는 기존 QK→scale/mask→softmax→BF16→PV 순서를 유지한다. QK/PV reduction과 중간 저장 경계를 보존한 상태에서 요청 축만 병렬화한다.
- paged KV 직접 접근은 별도 후보로 구현해 기존 gather 경로와 비교한다. page 경계, partial page, 비연속 slot, heterogeneous context, cancellation 직후 block 재사용을 검사한다.
- FlashInfer식 online softmax/fixed split-size는 독립 후보로 평가한다. fixed split-size의 결정성은 기존 materialized Attention과의 비트 일치 증명이 아니다.

**수정 지점:** executor `dispatch.rs`, `crates/riley-cuda/src/decode.rs`, `kernels/src/attention_cublaslt.cu`, `kernels/src/cublas_gemm_probe.cu`, 관련 softmax helper.

**완료 조건:** D64/D128·실제 지원 GQA·page 경계·길이 경계·prefill→decode 전환의 byte gate와 whole-model gate 통과. 지원 최대 길이를 넘는 시험은 기존 거부/회귀 조건으로 취급하며 임의로 context 지원을 늘리지 않는다.

### W4 — 기존 Graph coverage와 cache policy 개선

- actual rows, row layout, context capacity, pointer allocation generation, stream, 수치 policy를 key/적격성에 반영한다. 기존 resource ledger와 poison 규칙을 재사용한다.
- 입력/metadata와 workspace를 cold 단계에서 준비하고 replay 시 내용만 갱신한다. shape가 같아도 context/active slots가 달라지는 경우를 검증한다.
- Graph hit 외에 replay 범위, eager 구간, 새 capture 수, cache bytes, eviction/fallback 비용을 기록한다.
- Smol의 한정 whole-body 경로와 wide-model actual-row 경로를 기준으로 확장한다. 전체 모델/전 geometry가 이미 capture된다고 가정하지 않는다.

**수정 지점:** executor `graph.rs`, `graph_registry.rs`, `dispatch.rs`, CUDA graph owner.

**완료 조건:** eager와 수치 일치, 입력 갱신·주소 재사용·slot 변경·cache 포화·실패 회수 통과. C1/C8/C32에서 graph off/기존/new를 비교한다. C32에서 Graph가 느려지는 기존 관측을 반영해 무조건 기본 활성화하지 않는다.

### W5 — 측정 근거가 있을 때만 overlap 도입

**착수 조건:** W0/W2~W4 이후에도 CPU 준비·후처리로 인한 GPU gap이 유의하게 남는다. GPU kernel 자체가 대부분을 차지하면 이 작업의 우선순위를 내린다.

- 1단계는 다음 iteration의 입력과 무관한 host 준비·metadata staging·완료 응답 직렬화부터 겹친다.
- 2단계에서 device token 결과를 다음 입력으로 연결하는 future slot/event를 검토한다. `inflight` 하나를 두 개로 바꾸는 단순 수정으로 처리하지 않는다.
- logical commit, GPU completion, KV reclaim을 구분한다. EOS·cancel이 확인되기 전 추가 제출한 step의 결과는 사용자에게 publish하지 않으며 물리적 사용 완료 전 block을 재활용하지 않는다.
- 두 iteration의 buffer alias, failed completion, shutdown, backpressure와 제한된 speculative extra-step 자원을 설계한다. 기존 poison/abort 처리를 보존한다.

**수정 지점:** scheduler `scheduler.rs`/`execution.rs`, executor `buffers.rs`/`output.rs`, server 실행 loop.

**완료 조건:** Nsight에서 gap 감소와 실제 throughput/TTFT/ITL 개선이 함께 확인된다. EOS 이후 token 누출, usage 증가, 중복 completion, 조기 KV 재사용이 없어야 한다. overlap off로 기존 동작 복귀가 가능해야 한다.

### W6 — batch-invariant Tensor Core/외부 Attention 연구

**목적:** 기존 M=1 산술을 보존하는 방식의 성능 한계를 확인하고 대체 커널의 가능성을 측정한다. 기존 strict 계약 변경을 승인받았다는 뜻은 아니다.

- vLLM의 persistent matmul 및 `(N,K)`별 고정 K 타일, M bucket별 실행 설정을 native CUDA/CUTLASS 후보 또는 격리된 연구 harness로 비교한다.
- RMSNorm·Attention까지 배치 및 chunk 경계 불변성을 검사한다. matmul 한 개의 결정성만으로 모델 전체를 판정하지 않는다.
- 결과표는 “기존 Riley bytes”, “HF full logits”, “HF generation”, “배치 불변”, “태스크 품질”, “속도”를 각각 기록한다.
- FlashInfer 직접 연동은 같은 buffer/stream을 받는 얇은 native adapter 가능성부터 확인한다. Python runtime 종속·새 allocator를 기본 경로에 추가하지 않는다. 도입 시 ABI, workspace, GPU 지원, 라이선스/NOTICE를 확인한다.

**완료 조건:** strict gate까지 통과하면 geometry 제한 후보로 승격한다. 배치 불변성만 통과하면 별도 연구 결과로 남기며, 새로운 수치 프로파일 채택 판단을 분리한다.

### W7 — prefill·prefix workload 개선

W0에서 실제 반복 prefix 또는 긴 prefill 비중이 확인될 때 수행한다. prefix hit rate, 절감한 prefill tokens, TTFT, memory/eviction을 함께 본다. cache off 결과와 분리한다. chunk 정책 변경이 kernel shape·logits를 바꿀 수 있으므로 chunk 경계의 수치 게이트를 통과해야 한다. cold load·HTTP 직렬화 개선은 해당 비용이 측정된 경우 별도 항목으로 진행한다.

## 7. 검증 및 벤치마크 계약

### 7.1 단계별 진행과 비용 제한

1. **정적·단위 검증:** 변경한 계층의 계약, workspace/lifetime, 지원·거부 조건.
2. **component gate:** 모델 유래 weight/activation의 고정 참조와 BF16 비교. zero/permuted/tail/changed inputs 및 실제 호출 증거.
3. **모델 gate:** 같은 GPU의 고정 HF teacher full-vocabulary logits와 기존 Riley before/after, 긴 생성·자연 EOS·배치 전환.
4. **진단 screen:** 대표 모델 C1/C8에서 짧은 profiler+비계측 실행. 실패 후보는 여기서 중단한다.
5. **확대 비교:** 통과 후보만 모델·동시성·workload matrix로 확대한다.

초기 screen은 구성당 32 warmup+48 측정 요청, C1/C8 및 AB/BA 두 순서로 제한하는 제안이다. 확대 비교는 셀당 5개 fresh-process AB/BA pair, arm당 32 warmup+192 측정 요청을 기본 제안으로 두고 실행 전에 corpus·request 수·예산을 manifest에 고정한다. 이 횟수를 이미 수행한 것으로 표현하지 않는다. 장시간 soak는 이번 계획의 성능 검증에 포함하지 않는다.

### 7.2 모델·하드웨어·workload

| 축 | 초기 범위 | 확대 조건 |
|---|---|---|
| 모델 | Qwen2.5-3B, Qwen3-1.7B/4B, full-attention Mistral7B, SmolLM2-135M | 지원된 revision·tokenizer·VRAM 예산 확인; 같은 architecture·geometry 중복은 대표부터 |
| 장치 | SM80 A100 / SM89 RTX4090을 독립 lane으로 취급 | 실행 시 사용 권한·프로세스 점유를 새로 확인; 자동 대체 없음 |
| 동시성 | C1/C8 screen → C1/C8/C16/C32 | 실제 active rows 1..32 전환·비연속 slot 검사 병행 |
| 요청 | 고정 token-budget 진단 + 자연 EOS + mixed prompt/context | 짧은/긴 입력과 heterogeneous 배치, 모델 지원 한도 안에서만 |
| 기능 | strict baseline, 기존 opt-in, 단일 변경, 최종 조합 | Graph/cache/quantization/profile 차이는 별도 arm |

서로 다른 모델을 **같은 요청 배치의 서로 다른 prompt 길이**와 혼동하지 않는다. 한 GPU에 여러 모델을 동시에 상주시켜 자원을 경쟁시키는 테스트는 이번 기본 비교에 포함하지 않는다.

### 7.3 엔진 대조군

- **1차:** 같은 장치·checkpoint·BF16·KV/token budget의 Riley before/after와 unmodified stock vLLM.
- **2차:** 지원을 확인한 고정 버전의 vLLM batch-invariant mode와 SGLang default/deterministic mode. 성능과 수치 계약을 함께 비교한다.
- **선택:** TensorRT-LLM/llama.cpp는 해당 BF16 checkpoint 변환·지원이 확인될 때만 end-to-end 수치 대조군으로 추가한다. 이번 구현 조사와 실측을 구분한다.
- 최신 조사 커밋을 과거 설치 버전과 혼합하지 않는다. version, source SHA, attention backend, compile/Graph mode, 실제 loaded libraries를 남긴다.
- event-spin 등 수정 vLLM은 별도 진단군이다. 공식 vLLM 성능으로 보고하지 않는다. 엔진마다 GPU-memory-utilization 비율만 맞추지 말고 실제 weights/KV/workspace/Graph 및 peak/steady memory를 기록한다.

### 7.4 합격 기준과 중단 조건

- **정확성:** 현재 strict 계약의 byte·generation 게이트를 통과해야 한다. 배치 크기/순서·chunk·Graph 전환별 결과를 기록하고 mismatch를 평균 일치율로 숨기지 않는다.
- **개선 채택:** 셀별 paired throughput과 TTFT/TPOT/ITL/E2E p50/p95/p99를 함께 보고한다. 기본 승격 대상 셀에서 반복적으로 tail이 악화되면 기본 적용하지 않고 제한된 opt-in으로만 남긴다.
- **vLLM 동급 선언:** 사전에 정한 필수 matrix의 모든 셀에서 throughput≥vLLM, latency quantile≤vLLM, 수치/자원 게이트 통과가 필요하다. 평균 1.0이나 일부 승리로 선언하지 않는다. 반복 오차로 판정이 불명확하면 “미확정”으로 둔다.
- **시간 정의:** aggregate output tok/s와 요청별 token 속도를 구분한다. TPOT=(마지막-첫 token)/(출력 token 수-1)는 응답 평균이며, 개별 streaming event의 ITL 분포와 다르다. 출력 1 token은 TPOT N/A로 처리한다. SSE가 여러 token을 묶으면 event 간격을 token ITL로 오인하지 않는다.
- **EOS:** 자연 EOS의 길이 차이는 응답 수·output tokens·finish reason과 함께 남긴다. 동등 작업량의 kernel/token-budget 진단과 사용자 관점 자연 종료 결과를 별도로 제시한다.
- **자원:** 오류·중복 completion 없음, 종료 후 active/waiting/KV 회수, Graph/workspace budget 준수. 외부 GPU 작업이 관측되면 해당 run을 오염 자료로 보존하고 채택 근거에서 분리한다.
- 작은 finite sample의 p99는 기술 통계다. 독립 process 반복과 범위·분산을 제시하고, 한 프로세스의 연속 window를 독립 반복이라고 부르지 않는다.

## 8. 구현 순서와 의사결정

| 단계 | 작업 | 다음 단계 결정 |
|---|---|---|
| A | W0 경로 감사·기준선 | per-op sync/비활성 fast path면 기존 옵션·dispatch부터 수정 |
| B | W1 후보 검증 → W2 상위 projection → W3 Attention | 실제 kernel 시간/traffic이 줄지 않으면 해당 후보 중단 |
| C | W4 Graph 범위 개선 | launch gap이 남을 때만 확장; 느린 shape는 eager 유지 |
| D | W5 overlap | host bubble 측정 근거와 KV 수명 설계가 갖춰질 때 착수 |
| E | W6 독립 수치 연구, W7 workload 개선 | strict 통과 여부와 실제 workload에 따라 선택 |

GPU 가용성과 반복 실측 비용을 확인하지 않은 상태에서 달력 일정을 약속하지 않는다. 단계 A 산출물로 나머지 작업량과 실행 lane을 정한다. 각 후보는 “기준선→단일 변경→검증→조합”으로 진행하고 검증되지 않은 여러 최적화를 한 번에 켜지 않는다.

최종 산출물은 모델·장치별 지원표, 재현 가능한 plan/argv, before/after/vLLM 요청 기록, 수치 비교, profiler 요약, fallback/rollback 조건을 묶은 receipt다. 원시 로그·실패 후보를 보존하고 민감한 환경 값은 저장하지 않는다. 현재 중지된 controller나 heartbeat는 이 계획서 작성으로 재개되지 않는다.

## 9. 근거 자료

### Riley 로컬 근거

- **R1:** [Owned Graph/배치 최적화 및 2026-10-17 RTX4090 측정](owned-graph-batch-optimization.md), [최종 수치 요약](../artifacts/qkv-attention-opt-20261017/final-summary-v6.json), [동일 소스 incremental trace](../artifacts/qkv-attention-opt-20261017/profile-incremental-summary-v6.json).
- **R2:** [Inference completion 최적화](inference-completion-optimization.md), [동기화 profile 비교](../artifacts/inference-opt-20261016/evidence/profile-comparison.json).
- **R3:** [2026-10-19 통합 체크포인트](optimization-checkpoint-20261019.md).
- **R4:** [미적용 SM80 v235 후보와 검증 범위](../benchmarks/candidates/sm80-bounded-projections-v235/README.md), [실제 호출 감사 실패 원본](../benchmarks/evidence/20261019-optimization-checkpoint/v237-independent-call-audit-error.txt).
- [native fast 프로파일의 별도 수치·성능 증거](hf-native-performance.md). modified-vLLM screen을 stock 비교로 재사용하지 않는다.

`artifacts/`의 원시 자료는 ignored local 파일이므로 새 clone에서는 없을 수 있다. tracked 문서·checkpoint receipt와 구분한다. 이번 조사에서 원격 서비스 상태나 현재 GPU 점유는 확인하지 않았다.

### 외부 1차 자료 및 소스 고정

아래 SHA는 2026-10-19 조사 시 공식 저장소 HEAD를 조회해 고정한 **조사 스냅샷**이다. Riley와 함께 성능 검증된 의존성 버전이라는 뜻은 아니다. 링크된 소스 파일은 직접 내려받아 관련 구현을 확인했다. 공식 `latest` 문서는 이후 바뀔 수 있으므로 구현 작업에서는 이 SHA와 실제 채택 버전의 차이를 다시 확인한다.

| 저장소 | 조사 SHA |
|---|---|
| vllm-project/vllm | `852989f580820849113d67d13fd8053e279d0444` |
| sgl-project/sglang | `a90e9c5dd8973cdb84af4ff6154c7b4958083b31` |
| NVIDIA/TensorRT-LLM | `066332abd55602c00fa1156b4e058ed4caa8bc0a` |
| flashinfer-ai/flashinfer | `d8e13c06d575cd6ce0746d835b6f7fc42abe46ab` |
| ggml-org/llama.cpp | `f3fb2fcdbc93a8f677e893b8dbd36462b3190ffd` |

- **S1:** [vLLM linear: MergedColumnParallelLinear / QKVParallelLinear](https://github.com/vllm-project/vllm/blob/852989f580820849113d67d13fd8053e279d0444/vllm/model_executor/layers/linear.py#L686).
- **S2:** [vLLM CUDA Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/), [CudagraphDispatcher 소스](https://github.com/vllm-project/vllm/blob/852989f580820849113d67d13fd8053e279d0444/vllm/v1/cudagraph_dispatcher.py#L233).
- **S3:** [vLLM Batch Invariance 공식 문서](https://docs.vllm.ai/en/latest/features/batch_invariance/), [persistent matmul](https://github.com/vllm-project/vllm/blob/852989f580820849113d67d13fd8053e279d0444/vllm/model_executor/determinism/batch_invariant.py#L284), [M bucket과 고정 K 설정](https://github.com/vllm-project/vllm/blob/852989f580820849113d67d13fd8053e279d0444/vllm/model_executor/determinism/batch_invariant_configs.py#L1142).
- **S4:** [SGLang overlap loop](https://github.com/sgl-project/sglang/blob/a90e9c5dd8973cdb84af4ff6154c7b4958083b31/python/sglang/srt/managers/scheduler.py#L2011), [RadixCache](https://github.com/sgl-project/sglang/blob/a90e9c5dd8973cdb84af4ff6154c7b4958083b31/python/sglang/srt/mem_cache/radix_cache.py).
- **S5:** [SGLang Deterministic Inference 공식 문서](https://docs.sglang.io/docs/advanced_features/deterministic_inference), [고정 SHA 문서 소스](https://github.com/sgl-project/sglang/blob/a90e9c5dd8973cdb84af4ff6154c7b4958083b31/docs/docs/advanced_features/deterministic_inference.mdx).
- **S6:** [SGLang 개발팀의 Zero-Overhead Scheduler 설명](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/). 해당 글의 타 하드웨어 개선율을 Riley 예상치로 쓰지 않는다.
- **S7:** [TensorRT-LLM Paged Attention, IFB, Scheduling](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html).
- **S8:** [Thinking Machines Lab: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). 고정 reduction과 batch invariance의 원리 참고. 논문의 속도 수치는 Riley 예상치가 아니다.
- **S9:** [TensorRT-LLM GPT Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html), [DecoderXQARunner](https://github.com/NVIDIA/TensorRT-LLM/blob/066332abd55602c00fa1156b4e058ed4caa8bc0a/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h#L34), [overlap 설계](https://github.com/NVIDIA/TensorRT-LLM/blob/066332abd55602c00fa1156b4e058ed4caa8bc0a/docs/source/torch/features/overlap_scheduler.md).
- **S10:** [FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html), [paged decode plan/run 및 fixed-split 제한 소스](https://github.com/flashinfer-ai/flashinfer/blob/d8e13c06d575cd6ce0746d835b6f7fc42abe46ab/flashinfer/decode.py#L1969).
- **S11:** [llama.cpp CUDA matmul dispatch](https://github.com/ggml-org/llama.cpp/blob/f3fb2fcdbc93a8f677e893b8dbd36462b3190ffd/ggml/src/ggml-cuda/ggml-cuda.cu#L1956), [Graph 검사·갱신](https://github.com/ggml-org/llama.cpp/blob/f3fb2fcdbc93a8f677e893b8dbd36462b3190ffd/ggml/src/ggml-cuda/ggml-cuda.cu#L2699).


후속 W2 실행: [Qwen SM80 projection batching v2](qwen-sm80-projection-batching-v2-20261019.md). Q/O/Down/LM head의 수치 보존 배치와 C1/C8 실측을 완료했으며 확대 release matrix 및 기본 승격은 별도다.

후속 실행: [Qwen C16/C32 확대 및 native K/V 개선](qwen-c16-c32-expansion-20261019.md). mode7의 5회 C16/C32 대조와 native-order K/V mode8의 별도 수치·서빙 검증을 구분해 기록한다.

후속 tail 회복 및 operator 개선: [Qwen C16 tail·KV gather·Gate/Up prefill](qwen-c16-tail-and-attention-20261019.md). Mode10에서 C16/C32의 3회 전체 latency quantile을 개선했고, BF16 exact 범위만 요청 간 prefill GEMM으로 묶었다. Down·QK/PV의 잔여 비용과 다음 native-order 검증 순서를 기록한다.
