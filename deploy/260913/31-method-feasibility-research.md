# 새 실행 방법의 도입 가능성 조사 — 2026-09-15

상태: 문헌·공식 문서·소스 조사와 계획 작성 완료. 이번 조사에서 구현, GPU 실행, 새로운 성능 측정은 하지 않았다. 아래 순위는 Riley의 실측 speedup 순위가 아니라 **검증할 가치와 적용 비용에 대한 판단**이다. [현재 종료 결과](29-serving-closeout.md)와 [다음 단계 계획](30-next-stage-roadmap.md)을 함께 사용한다.

## 결론과 착수 순서

새 방법을 시험할 여지는 있다. 특히 135M·짧은 출력에서 드러나지 않은 긴 context와 GQA는 실행 구조를 바꿀 후보가 있다. 다만 공개 kernel을 채택하는 것만으로 같은 kernel을 활용하는 vLLM/SGLang보다 빨라지지는 않는다. **표준 native BF16 대조군을 확보하고, 제한된 operator/layer 실험으로 기여 가능성을 판정한 뒤, 살아남은 후보만 full-model에 연결**한다.

| 순위 | 후보 | 바꾸는 비용/구조 | 4090 1차 판정 | 기존 Riley와 다른 점 |
|---|---|---|---|---|
| P0 필수 대조군 | native BF16 GEMM + attention | 전용 부분합·반올림 계약 밖의 실행 기준 | 기존 C ABI 자산 재사용, SM89 artifact 확인 | 새 알고리즘이라기보다 다른 후보를 공정하게 평가할 기준 |
| P1 | Tensor Core GQA + 부하 균형을 고려한 context 분할 | KV 재사용, 작은 decode query의 SM 병렬성 | 3B의 2K/8K/16K context 대표 shape | strict recurrence를 유지한 과거 split 재시도가 아닌 별도 BF16 profile |
| P2 조건부 | Cascade식 공유 prefix attention | 같은 prefix KV의 요청 간 읽기/계산 재사용 | 긴 공유 prefix와 여러 owner가 있을 때 | 기존 prefix cache 및 두 owner의 원래 recurrence 재사용과 구분 |
| P3 탐색 | RMSNorm–GEMM 대수적 재배치 | 중간 tensor 및 norm→GEMM 직렬 의존성 | QKV와 gate/up 두 연산 묶음만 | 큰 cooperative kernel로 합치는 대신 연산 의존성을 변경 |
| 보류 | XQA 현행 wrapper, FA4, MPK, 근사 attention | 장비 전용 실행 또는 더 큰 수치/설계 변경 | 현행 XQA 경로는 SM89 제외; 나머지는 아래 조건 참조 | 기존 실패를 이름만 바꿔 재도입하지 않음 |

첫 단계는 P0와 P1의 작은 실험 설계다. P2는 실제 workload에 긴 공유 prefix가 있을 때, P3는 제거 가능한 비용의 상한이 충분할 때만 진행한다. 네 가지를 동시에 구현하지 않는다.

## 공통 계약

- 1차 모델 후보는 Qwen2.5-3B-Instruct, 연결 확인은 SmolLM2-1.7B, 기존 strict 회귀는 135M이다. 모델/config/tokenizer 지원과 수치 reference를 먼저 고정한다. 가중치 로딩 지원 자체를 성능 개선으로 계산하지 않는다.
- BF16 weight/KV, 전체 GPU peak **20,000,000,000 bytes 이하**. [계획30의 메모리 계산](30-next-stage-roadmap.md)을 적용한다. workspace, FP32 partial state, graph pool, packing 중복, 외부 사용량도 포함한다. offload/양자화로 조건을 바꾸지 않는다.
- Rust→C/C++ ABI→CUDA 실행. 논문의 Python DSL이나 라이브러리 Python frontend는 바로 runtime에 연결하지 않는다. 필요하면 오프라인 생성/빌드 후 native artifact를 사용하되 ABI, stream, lifetime, capture, 오류 전파를 검증한다.
- strict 경로와 별도 native-BF16 profile을 유지한다. 누적 dtype, cast 위치, softmax/exp, reduction 순서, bias/RoPE 위치를 기록한다. 실수에서 동치인 식도 BF16에서 bitwise 동치는 아니다.
- 성능을 보기 전에 허용할 layer/logit 오차, teacher-forced 분포 차이, 고정 corpus의 perplexity/업무 품질 허용폭, 자유 생성 회귀 기준을 명시한다. 숫자는 reference와 평가셋을 확정하는 N02에서 사전 등록하며, 결과를 본 뒤 완화하지 않는다. NaN/Inf·mask·page 경계·cancel/COW 오류는 허용하지 않는다.
- 4090은 GDDR6X 장비다. HBM 논문의 메모리 계층 아이디어는 검토하되 HBM 장비의 수치를 4090 예상치로 옮기지 않는다. 출처: [NVIDIA Ada architecture, RTX4090 사양](https://images.nvidia.cn/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

## P0 — 강한 native BF16 대조군

vLLM과 SGLang은 모델·dtype·head shape·장비에 따라 attention backend를 선택한다. 따라서 비교표에는 엔진 버전만 아니라 실제 선택된 prefill/decode backend와 graph 설정도 남겨야 한다. 자료: [vLLM backend registry 기반 지원표](https://docs.vllm.ai/en/latest/design/attention_backends/), [SGLang attention backend 안내](https://docs.sglang.io/docs/advanced_features/attention_backend).

제안하는 작은 묶음:

1. 기존 prepared GEMM/native attention adapter에서 새 profile을 선택하고, 대표 shape에 대해 cuBLASLt 또는 지원되는 CUTLASS/FlashInfer 경로를 연결한다. 처음부터 범용 실행 엔진을 재작성하지 않는다.
2. 계획/packing/할당은 가능하면 준비 단계에 두고, steady 실행과 실제 반복되는 metadata 갱신 비용을 분리 기록한다. capture할 수 없는 planner를 CUDA Graph 안에 넣지 않는다.
3. 같은 입력·weight·mask를 사용한 reference, native library, 후보 결과를 비교한다. 기존 strict와의 차이는 별도 보고한다.

[cuBLAS 공식 API](https://docs.nvidia.com/cuda/cublas/index.html)는 native 호출 기반이다. 문서 최신 버전과 실제 장비의 toolkit 버전이 같다고 가정하지 않고 빌드 시 고정한다. [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html)는 GQA용 Tensor Core 선택과 planning 제약을 명시한다. Riley의 기존 SM89 adapter를 출발점으로 쓰되 해당 dtype/shape/graph 조합의 실행 지원은 다시 확인한다.

### XQA를 4090 즉시 도입 후보에서 제외한 이유

조사 시점 [FlashInfer `jit/xqa.py` main](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/jit/xqa.py)의 NVCC 설정은 `supported_major_versions=[9, 10, 12]`이다. SM89의 major 8은 포함되지 않는다. BF16 입력 지원과 장비 지원은 별개의 조건이다. 이것은 **현행 FlashInfer wrapper 경로**에 대한 판정이며 모든 과거 TensorRT-LLM XQA 구현이 Ada에서 불가능하다는 주장은 아니다. architecture guard를 제거해 강제로 빌드하지 않는다. 별도의 SM89 구현을 선택하려면 지원 소스와 runtime 증거가 선행해야 한다. 현행 main URL은 변하므로 실제 실험 전에 commit SHA를 고정한다.

## P1 — 긴 context의 GQA 및 작업 분할

[LeanAttention (2024)](https://arxiv.org/abs/2405.10480), [본문 v1](https://arxiv.org/html/2405.10480v1)은 decode의 작은 query와 긴 KV를 대상으로, associative softmax state와 Stream-K식 작업 분배로 GPU 작업량을 나눈다. 논문의 긴 문맥 operator 결과는 최신 vLLM serving이나 Riley의 예상 향상률이 아니다.

**적용 가설:** Qwen 3B의 query-head/KV-head 비율과 긴 context에서는 KV를 여러 query head가 재사용하고, batch가 작아도 KV 구간을 여러 SM에 배분하는 것이 유리할 수 있다. 이 가설은 현재 Riley에서 bandwidth 포화가 측정됐다는 뜻이 아니다.

최소 실험 묶음:

1. 별도 native-BF16 대조군에서 일반 decode와 Tensor Core GQA를 비교한다.
2. 지원되는 library의 split/planning을 우선 사용하고, 분할 없음 대조군과 workload에 맞춘 제한된 분할 정책을 비교한다. kernel이 이미 제공하는 기능을 새로 만들지 않는다.
3. partial-state merge, metadata 갱신, graph replay까지 포함한 layer 비용과 scratch peak를 기록한다.

대표 범위는 batch 1/4/8/32, context 2K/8K/16K 중 메모리 예산에 맞는 작은/긴 대표 조합이다. 처음부터 모든 조합을 실행하지 않는다. FP32 partial state의 용량은 batch×query heads×split 수×head dimension에 비례하므로 split 증가를 무료로 보지 않는다.

통과: 지원 shape·수치 gate를 만족하고 merge를 포함해 강한 native 대조군보다 빠르며, 아래 기여도 gate를 통과한다. 탈락: 짧은 shape의 dispatch 회귀가 fallback으로 분리되지 않거나 merge/launch 비용으로 이득이 사라짐. 과거 strict split 실패에 허용오차만 붙여 재사용하지 않고 native profile의 일관된 prefill/decode 결과를 검증한다.

## P2 — 공유 prefix를 분리 계산하고 attention state 병합

[FlashInfer Cascade 설명과 C++ API (2024)](https://flashinfer.ai/2024/02/02/cascade-inference.html)은 공유 prefix와 개별 suffix를 따로 계산한 뒤 attention state를 병합한다. 공개 실험은 A100/H100 및 오래된 비교 조건이므로 그 배수를 현재 serving 목표로 사용하지 않는다.

출력 `o_p`, `o_s`와 각각의 log-sum-exp `l_p`, `l_s`에 대해 `m=max(l_p,l_s)`라 두면 실수 연산에서:

`o = (exp(l_p-m)·o_p + exp(l_s-m)·o_s) / (exp(l_p-m)+exp(l_s-m))`

이 성질을 사용해 공유 prefix의 KV를 여러 요청이 함께 처리한다. BF16 cast와 reduction 순서가 바뀌므로 기존 strict 출력은 자동 보존되지 않는다. 빈 prefix/suffix와 전부 masked인 상태는 별도 처리해야 한다.

최소 실험 묶음:

1. 기존 prefix 소유권/page metadata로 공유 구간만 묶는다. 새 radix cache나 중복 KV 사본을 먼저 만들지 않는다.
2. 공유 prefix 다중 query 실행 + 개별 suffix 실행 + state merge를 연결한다.
3. 비공유/작은 그룹은 기존 native 경로로 보내며, 그룹 구성 비용과 COW/page lifetime을 포함한다.

대표 범위: 공유 owner 1/4/8/16, prefix 0/2K/8K, suffix 128/1K에서 대조군을 포함한 소수 조합. 통과는 긴 공유 입력의 end-to-end layer 이득과 비공유 fallback 회귀 없음이다. suffix가 길어질 때 이득이 줄어드는 경계도 기록한다. 기존 두 owner KV 재사용 실패와 달리 **prefix/suffix를 별도 state로 계산하고 합치는 구조**를 검증한다. 공유가 없는 workload의 범용 개선으로 주장하지 않는다.

## P3 — RMSNorm과 GEMM의 대수적 재배치

[Mirage superoptimization 논문 (2024), 본문 v3](https://arxiv.org/html/2405.05751v3)의 RMSNorm/matmul 사례는 normalization의 행별 분모를 matmul 뒤로 옮기는 의존성 변경을 다룬다. 논문의 대수 검증과 실제 부동소수점 수치 안정성 검증은 별개다.

행 `x`, scale `g`, 행렬 `W`에 대해 `r=sqrt(mean(x²)+epsilon)`이면 실수에서:

`((x ⊙ g) / r) W = ((x ⊙ g) W) / r`

**Riley 적용 가설:** norm 통계와 GEMM을 독립적으로 진행하고 epilogue에서 행별 scale을 적용하면 중간 normalized tensor와 직렬 의존성을 줄일 수 있다. 단순 kernel 결합과 달리 DAG 자체가 달라진다. 그러나 원래 BF16 normalized activation의 반올림이 사라지므로 별도 profile의 품질 gate가 필수다.

최소 실험 묶음:

1. norm 통계를 계산하고 GEMM 입력 fragment에서 `g`를 적용하는 후보를 설계한다. weight 전체의 추가 GPU 사본이나 무조건적인 BF16 weight prescaling을 전제로 하지 않는다.
2. GEMM epilogue의 row scale을 연결한다. bias가 있으면 scale 뒤에 적용하고, RoPE 위치와 epsilon을 보존한다. 이 변환을 SwiGLU 비선형을 넘어 임의로 확장하지 않는다.
3. QKV와 gate/up 두 묶음에 한해 분리 norm+좋은 native GEMM 대조군과 비교한다. atomic·동기화·추가 scratch를 포함한다.

3B 후보 shape는 QKV K=2048/N=2560, gate+up K=2048/N=22016이며 모델 config 고정 후 확인한다. M은 decode 1/8/32와 작은 prefill chunk 128에서 대표값을 선택한다. Mirage 전체 compiler 도입은 선행 조건이 아니다. 작은 M에서 weight 읽기가 지배하면 norm 제거의 상한이 작으므로 그 경우 중단한다. 기존 attention→FFN persistent serving 회귀를 대형 fusion 확장의 근거로 삼지 않는다.

## 다른 문헌에서 얻은 판단과 보류 조건

| 자료 | 얻은 통찰 | 이번 결정 |
|---|---|---|
| [Stream-K (2023)](https://arxiv.org/abs/2301.03598) | GEMM의 출력 tile 수 외에 전체 inner-loop 작업량을 기준으로 분배 | P0에서 native library와 비교할 조건부 후보. 작은 M이면 무조건 유리하지 않으며 reduction/scratch 비용 포함 |
| [FlashDecoding++ (2023/MLSys2024)](https://arxiv.org/abs/2311.01282) | decode 실행·GEMM·파이프라인을 함께 다룸 | 기존 double buffering을 새 기능으로 재제안하지 않음; 다른 shape의 native 기준이 우선 |
| [The I/O Complexity of Attention (2024)](https://arxiv.org/abs/2402.07443) | 특정 2-level memory 모델에서 attention I/O 하한과 cache 크기의 관계 | 임의의 행렬곱 치환으로 메모리 비용이 사라진다는 주장을 배제. paged decode/4090 최적성의 증명은 아님 |
| [Fast Attention Requires Bounded Entries (2023)](https://arxiv.org/abs/2302.13214) | bounded input 등 조건 아래 attention 근사 알고리즘과 복잡도 경계 | 실제 Q/K 범위·scale·다항식 차수·오차가 확보되지 않아 1차 BF16 후보에서 보류. 수학적 근사가 원래 모델 정확성을 보장하지 않음 |
| [Memory-Bound but Not Bandwidth-Limited (2026)](https://arxiv.org/html/2605.30571v1) | 메모리 비용의 하한과 실제 bandwidth 포화, launch 비용을 구분 | 다른 GPU·7–8B·batch1의 preprint 결과. Riley의 counters 부재를 보충하는 측정 증거로 쓰지 않음; 기존 CUDA Graph 재도입 근거도 아님 |
| [FlashAttention-4 (2026)](https://arxiv.org/abs/2603.05451) | Blackwell 실행 자원과 pipeline을 활용 | 추후 Blackwell/AOT-native 가능성 조사. 현재 4090 도입과 구분하며 DSL의 Python frontend를 runtime에 연결하지 않음 |
| [Mirage Persistent Kernel (2025)](https://arxiv.org/abs/2512.22219) | SM task graph와 분산 scheduling | 기존 cooperative fusion과 같지는 않으나 구현 범위가 커 후순위. 작은 후보가 실패했다는 이유만으로 곧바로 megakernel 재작성하지 않음 |

위상수학을 포함한 추상 수학의 이름 자체보다, attention state의 결합 법칙·정규화의 대수 이동·작업량 분해·I/O 하한처럼 실제 코드 변경과 검증 조건이 연결되는 아이디어를 우선했다. 이번 조사에서 원래 dense attention의 품질을 유지하면서 바로 쓸 수 있는 위상수학 기반 대체법은 선정하지 않았다. 전체 분야에 그런 방법이 없다는 결론은 아니다.

Speculation은 이미 serving에서 실패한 구현이 있으므로 새 방법 목록에서 제외한다. 새 draft 모델/검증 방식과 acceptance·메모리·verify 비용 근거가 생겨야 재진입한다. Sparse/quantized attention은 BF16 dense 계약 변경을 요구하므로 1차 밖이다. P/D 분리는 실제 multi-GPU 자원과 통신 비용을 포함한 후속 문제다. FA3는 기존 연결을 재구현하지 않고 Hopper 장비에서 미검증 실행을 확인한다.

## 대규모 구현 전 가능성 판정

### R0 — 소스·수치·용량 판정

N02의 모델/수치 계약과 함께 각 후보의 immutable 소스, 라이선스, SM89 target, native ABI, dtype, layout, graph/capture, scratch 상한을 적는다. 미지원이면 이유와 향후 장비 조건을 기록하고 해당 후보를 보류한다. 컴파일 성공을 runtime pass로 표시하지 않는다.

### R1 — 제한된 operator/layer 실험

P0 대조군과 후보의 2~5개 관련 변경을 한 묶음으로 측정한다. 최대 두 실행 방식과 원인이 분명한 수정 한 차례를 기본 상한으로 두고, 끝없는 tile 탐색을 하지 않는다. 단독 kernel 시간뿐 아니라 실제 필요한 plan 갱신·merge·packing·동기화·graph 실행을 합산한다. 고정 작은 shape와 긴 shape를 포함해 유리한 shape만 선택하지 않는다.

기여도는 먼저 `T_new = T_unchanged + T_candidate + T_added`로 계산한다. 단순 직렬 모델에서 대상 비중 `f`, 해당 부분 speedup `s`, 기존 전체 시간 대비 추가 비용 `h`이면:

`S = 1 / (1 - f + f/s + h)`

예를 들어 10% 부분이 2배 빨라져도 추가 비용이 없을 때 전체 speedup은 약 1.053이다. +15%에는 `f(1-1/s)-h ≥ 0.130435`가 필요하다. 이는 계산 예시이며 serving throughput 예측 모델이 아니다. 실제 batching/overlap에서는 바뀐 critical path를 확인해야 한다. 후보 이득을 단순 합산하지 않는다.

통과 기준은 **수치/메모리 계약 충족 + 반복 측정 변동보다 큰 순이득 + 목표 workload의 의미 있는 전체 비용 감소 가능성**이다. 전체 비용 비중이 아직 없으면 operator 유망 판정까지만 하고 성능 목표 통과로 표시하지 않는다. 초기에는 제한된 layer timing으로 판정하며 광범위한 profiling을 반복하지 않는다.

### R2 — 선택한 후보만 모델·serving 검증

살아남은 후보만 N03/N04 full-model 경로에 연결한다. 우선 3B의 짧은 비공유, 긴 비공유, 긴 공유 workload를 고정하고 기존135M strict 회귀를 확인한다. 수치 gate를 통과한 뒤 엔진별 동일 revision·길이·EOS·memory·부하 조건으로 비교한다. vLLM은 그 시점 안정 버전을 고정하고 실제 선택 backend를 남긴다. SGLang/TRT-LLM은 모델·장비 지원이 되는 경우 같은 조건의 비교군으로 추가한다.

| milestone | 모델/입출력/공유 | 엔진·버전·backend | req/s·output tok/s | TTFT P50/P95/P99 | TPOT P50/P95/P99 | E2E P95/P99 | 실패율 | 전체 GPU peak | 품질 gate |
|---|---|---|---|---|---|---|---|---|---|
| 실행 후 기록 | 동일 조건 | Riley 기준/후보/vLLM/지원 비교군 | 원본+차이 | 원본+차이 | 원본+차이 | 원본+차이 | 요청 실패·OOM | bytes | pass/fail |

표는 의미 있는 실행 경로가 완성된 milestone마다 작성한다. operator마다 vLLM serving 전체를 재실행하지 않는다. 무부하/낮은 부하 latency와 높은 부하 throughput·tail을 구분하고 실행 순서를 교차해 환경 변동을 확인한다. +15% throughput과 TTFT/TPOT 목표는 서로 다른 유리한 조건에서 따로 가져와 합격 처리하지 않는다.

## 계획30에 반영할 변경

N01/N02는 제한된 실험에 필요한 범위로 진행한다. **R0/R1이 N03/N04의 대규모 구현 선행 gate**다. P1이 유망하면 native attention 경로에 우선 반영하고, P2/P3는 각 조건이 충족될 때 별도 PR로 분리한다. 실패하면 후보별 소스·수치·성능·용량 사유를 남기고 다른 메커니즘으로 이동한다. 실패를 근거로 strict 경로의 작은 knob 탐색을 자동 재개하지 않는다.

Hopper/Blackwell/multi-GPU는 계획30의 H01/H02/D01을 유지한다. 하드웨어가 없을 때 실행 테스트를 skip하되 미검증 상태를 남긴다. 이번 문서는 그 기능들의 구현 완료나 장비 실행 결과를 의미하지 않는다.

## 조사 범위와 재현 메모

기존 [행렬·하드웨어 조사24](24-matrix-hardware-research.md), PR03/08/13/19와 종료29의 중복·실패 여부를 대조했다. 신규 문헌뿐 아니라 적용 가능한 2023~2024 방법을 함께 검토했다. 주요 논문은 본문 중 해당 알고리즘/수치 구간을 확인했고, FA4·MPK·FlashDecoding++ 및 복잡도 논문은 초록/핵심 주장 수준의 선별 자료다. 논문 전체 재현이나 exhaustive literature review는 아니다.

검색 축: `LLM decode GQA tensor core split KV load balancing LeanAttention`, `HBM attention I/O complexity cache lower bound`, `CUDA small M GEMM Stream-K decode`, `shared prefix Cascade attention state merge`, `RMSNorm matmul reorder Mirage numerical stability`, `bounded entries polynomial approximation softmax attention`, `Blackwell FlashAttention 4 BF16`, `vLLM SGLang TensorRT-LLM attention backend compute capability`.

공식 API/main 소스는 2026-09-15 조회 자료다. 미래 실험에는 정확한 tag/SHA를 남긴다. 공개 논문의 배수, 다른 GPU 결과, compile-only 결과를 Riley serving 성능으로 변환하지 않았다.
