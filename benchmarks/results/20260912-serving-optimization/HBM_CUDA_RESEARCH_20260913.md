# HBM·CUDA 연산·GPU compiler 중심 연구

2026-09-13. 프레임워크 이름을 제외한 검색을 추가했다. 4090 지원 여부로 후보를 배제하지 않는다. 이 문서는 문헌 선별과 적용 제안이며 구현 또는 성능 검증 결과가 아니다.

## 검색 축과 선별 원칙

- `GPU HBM IO complexity attention efficient memory`
- `CUDA efficient GEMM Stream K work centric decomposition`
- `GPU kernel fusion compiler memory traffic Welder`
- `ThunderKittens tile primitives efficient GPU kernels`
- `GPU memory fusion 2025`, `Lean Attention`

arXiv 원문 페이지와 USENIX·저자 연구실·공식 구현 문서를 사용했다. 최신 연구와 기초 원리를 함께 포함했다. 연산 단위 논문의 개선 배수를 serving 속도 향상으로 옮기지 않는다. HBM이라는 검색어는 off-chip GPU memory 접근 연구의 출발점이며, 현재 4090의 물리 메모리가 HBM이라는 뜻은 아니다.

## 핵심 후보

| 연구 | 핵심 원리 | Riley에 가져올 방향 | 도입 시 확인할 점 |
|---|---|---|---|
| FlashAttention | attention 중간 행렬을 off-chip에 모두 저장하지 않는 IO-aware 계산 | QK·softmax·PV의 데이터 수명을 함께 설계 | 수학적 exact attention과 특정 BF16 구현의 bitwise 일치는 다름 |
| Welder | tile-graph와 메모리 계층별 traffic cost model | 연산 내부 재사용과 연산 사이 재사용을 하나의 계획에서 비교 | fusion으로 register/shared-memory 압력이 커지는 경우 포함 |
| MCFuser | shape 때문에 memory-bound가 된 compute 연산 체인의 fusion | 작은 M의 projection/FFN을 독립 GEMM 목록으로만 취급하지 않음 | kernel 연구 결과를 model·serving까지 재검증 |
| Stream-K | output tile 수 대신 전체 K-loop 작업을 균등 분배 | shape별 GEMM 병렬 분해·partial merge 정책 | 추가 reduction·scratch 비용, rounding 순서 변화 |
| LeanAttention | online softmax reduction과 Stream-K식 attention 분할 | 긴 KV에서 decode 작업량을 SM에 고르게 분배 | 작은 context에서 merge 비용이 더 클 수 있음 |
| ThunderKittens | tile abstraction과 비동기 warp/block 실행 template | GPU backend의 tile·pipeline 구현 기반 | 라이브러리 도입 자체가 serving 개선은 아님 |
| ParallelKittens | tile 수준 multi-GPU 통신과 계산 중첩 | tensor/expert/sequence parallel 연산의 pipeline | topology, 통신 자원 사용, correctness·동기화 계약 |

각 행의 근거와 적용 판단은 아래에 구분했다.

### 1. 접근량 자체를 줄이는 알고리즘

FlashAttention의 출발점은 FLOPs 감소보다 HBM↔SRAM IO 감소다. 따라서 “더 빠른 load”와 “그 load가 필요 없게 만드는 계산 순서”를 구별해야 한다. [FlashAttention](https://arxiv.org/abs/2205.14135).

Riley 제안: decode의 global score scratch와 값 계산 사이의 경계를 없애는 backend, mixed attention에서 반복 KV read를 줄이는 계획을 검토한다. 이미 일부 fused attention을 사용하므로 새로 FlashAttention을 발명하듯 접근하지 않는다. 한 layer의 실행 DAG에서 어떤 tensor를 언제 생성·소비·폐기하는지 정하고, 실제 저장 필요성과 recomputation 비용을 함께 비교한다.

### 2. 연산 체인 전체의 메모리 계획

Welder는 tile 수준 데이터 관리를 통해 메모리 계층별 최적화 공간과 연산 안팎의 재사용을 다룬다. Riley에 유효한 관점은 “kernel 하나를 얼마나 빠르게 만들까”에서 “어떤 중간 tensor를 off-chip까지 보내야 하는가”로 질문을 바꾸는 것이다. [Welder, OSDI 2023](https://www.usenix.org/conference/osdi23/presentation/shi).

MCFuser는 일반적으로 compute 연산으로 분류되는 연산도 shape에 따라 memory-bound가 될 수 있다는 점에서 출발해 체인 fusion을 탐색한다. 논문은 SC24 연구이며 arXiv 등록은 2025년이다. [MCFuser](https://arxiv.org/abs/2506.22169).

Riley 제안: QKV projection→RoPE/KV write와 FFN gate/up→activation→down projection을 각각 영역 단위로 평가한다. 무조건 모두 fusion하지 않는다. tile 수명, 중복 계산, shared-memory 사용량, register spill, kernel 간 buffer traffic을 포함해 분리·부분 fusion·persistent 실행을 비교한다. 기존 V56처럼 launch 감소만으로 효과를 예측하지 않는다.

### 3. 병렬 분해와 작업량 균형

Stream-K는 GEMM의 output tile 배치가 GPU 처리 자원 수에 잘 맞지 않을 때 전체 inner-loop 작업을 균등하게 나누는 방법이다. 특정 tile 크기 조정과 다른 수준의 변경이다. [Stream-K](https://arxiv.org/abs/2301.03598).

LeanAttention은 decode의 긴 context 계산을 online-softmax reduction으로 나누고 Stream-K 방식의 작업 분할을 활용한다. 여기서 softmax 결합의 수학적 성질은 부동소수점의 임의 reduction 순서가 bitwise 동일하다는 보장이 아니다. [LeanAttention, 2025 개정](https://arxiv.org/abs/2405.10480).

Riley 제안: shape-aware 실행 plan에 split 여부, 작업량, merge scratch, graph bucket을 함께 넣는다. small-M GEMM과 long-context attention을 각각 검토한다. “메모리 효율”은 접근량 감소뿐 아니라 균형 잡힌 병렬 작업으로 실효 대역폭을 사용하는 문제도 포함한다. [CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md).

### 4. 비동기 pipeline을 표현하는 구현 기반

ThunderKittens는 warp의 tile 연산, block 내 비동기 작업 중첩, grid 수준 실행 비용을 다루는 abstraction을 제공한다. [논문](https://arxiv.org/abs/2410.20399). 저자들의 TK 2.0 업데이트도 backend 구현 기반을 평가할 자료다. [2026 공식 연구실 글](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2).

Riley 제안: Rust는 request·KV ownership·execution plan을 유지하고, CUDA/C++ backend는 tile/pipeline library 또는 생성된 kernel을 호출하는 구조를 검토한다. C++ template를 Rust로 직접 재작성하는 것이 목적은 아니다. compile time, binary 크기, 지원 GPU, stream·buffer ABI, 유지보수 비용도 채택 기준에 포함한다.

ParallelKittens는 같은 관점을 여러 GPU의 compute/communication 중첩으로 확장한다. 단일 GPU 메모리 최적화와 네트워크 통신을 같은 byte 비용으로 취급하지 않고, 어떤 자원에서 대기가 생기는지 구분해야 한다. [ParallelKittens](https://arxiv.org/abs/2511.13940).

## 연구 결과를 적용할 batch 제안

### 본문 검토로 구체화한 fusion 선택 기준

Welder §4는 단순 tensor byte 수 외에 비연속 접근의 transaction 비용, 큰 tile로 인한 병렬성 부족, 메모리 용량 초과를 cost model에 반영한다. shared-memory fusion 시에는 주소 및 block/thread mapping과 동기화도 처리한다. 이 때문에 producer의 store와 consumer의 load를 기계적으로 삭제하는 것만으로 구현이 끝나지 않는다. [Welder 본문 §4](https://www.usenix.org/system/files/osdi23-shi.pdf).

MCFuser §III–IV는 loop dependency에 따라 중복 load를 이동하고, 불가능하거나 동등한 tiling 후보를 제거한 뒤 성능 모델로 추린 후보를 실제 측정한다. 모델이 benchmark를 대체하는 구조는 아니다. [MCFuser 본문](https://arxiv.org/html/2506.22169v1).

Riley에는 아래 기준을 적용하도록 제안한다. 이는 논문의 수치를 옮긴 것이 아니라 두 연구에서 얻은 설계 원칙이다.

1. **연산 DAG와 tensor 수명:** 각 결과가 어느 consumer에 필요한지, 최종 사용 이전에 재사용할 수 없는 buffer가 무엇인지 명시한다. KV처럼 iteration을 넘어 유지되는 상태와 일시적인 activation을 구분한다.
2. **연결 위치 선택:** global buffer, shared-memory tile 전달, register 전달을 각각 후보로 둔다. 전체 tensor를 on-chip에 보관해야 한다는 전제를 두지 않는다.
3. **형태별 비용 비교:** 줄어드는 global transaction뿐 아니라 추가 weight load, producer 재계산, reduction·barrier, spill 및 낮아진 병렬성을 비교한다. cache hit를 확인하지 않은 논리 byte 수를 HBM 실측값으로 쓰지 않는다.
4. **명시적 fallback:** 좁은 shape의 fused plan을 모든 batch에 강제하지 않는다. shape·dtype·backend별로 계획을 선택하며 선택 비용도 serving에 포함한다.

예를 들어 현재 FFN intermediate를 BF16으로 전체 보관하면 `batch × intermediate_size × 2` bytes다. intermediate1536에서 batch32는 96 KiB, batch64는 192 KiB이며, 이는 해당 tensor 하나의 크기일 뿐 weight tile·pipeline buffer·accumulator 자원을 포함하지 않는다. 이 숫자는 특정 GPU에서 fusion이 가능/불가능하다는 판정이 아니라, 전체 tensor fusion과 tile 전달을 구분해야 하는 이유다.

반면 tile 단위로 down projection에 연결하면 consumer output tile 여러 개가 같은 producer 결과를 요구할 수 있다. 이때 producer를 반복 계산하거나 같은 weight를 더 읽게 되면 저장을 줄인 이득을 잃을 수 있다. 따라서 다음 architecture 실험은 **분리 실행 / tile 전달 fusion / persistent task 실행** 세 계획을 같은 layer DAG에서 비교하는 형태가 적절하다. full-model·serving 검증 전에는 어느 계획도 채택하지 않는다.

| batch | 함께 구현·평가할 변경 2~5개 | 필요한 성공 증거 |
|---|---|---|
| Memory-aware layer execution | tensor 수명 계획, fusion boundary, workspace 재사용, shape dispatch | 대표 layer뿐 아니라 전체 model·serving에서 개선, spill/중복계산 회귀 없음 |
| Work-balanced attention/GEMM | split 계획, partial merge, shape별 backend 선택, graph 연동 | 작은/큰 batch와 짧은/긴 context에서 correctness·전체 latency 비교 |
| Asynchronous pipeline | load/compute 역할 분리, double buffering, event/barrier 수명, backend ABI | 전송·계산 중첩의 trace 증거와 serving throughput·tail 개선 |

첫 번째와 두 번째는 기존 FlashInfer/MPK 연구와 연결된다. 세 번째는 Hopper/Blackwell 전용 기능뿐 아니라 각 GPU에 맞는 구현을 둘 수 있다. 여기서 제시한 순서는 architecture 후보를 구체화하는 단위이며, 당장 세 batch 모두 구현하라는 결정은 아니다.

## 평가 시 구별할 값

### 추가 조사: 저정밀 저장과 연산의 공동 설계

검색에 `KV cache quantization memory bandwidth`, `weight dequantization fused GEMM`, `lookup table quantized CUDA`를 추가했다. 다음 후보는 BF16 실행의 정확성 조건을 그대로 만족하는 대체 구현으로 분류하지 않는다. 별도 정밀도 모드와 품질 검증이 필요한 연구다.

| 연구 | 메모리·연산 관점의 핵심 | Riley 적용 판단 |
|---|---|---|
| [KIVI, ICML 2024](https://arxiv.org/abs/2402.02750) | K는 per-channel, V는 per-token으로 다르게 양자화하는 KV 표현 | 긴 context·높은 concurrency의 KV 용량 및 읽기 비용을 겨냥한다. page layout, scale metadata, append 시 압축 비용을 함께 설계해야 함 |
| [QServe, 2025 개정](https://arxiv.org/abs/2405.04532) | W4A8KV4와 dequantization 비용을 함께 설계하고 weight 재배열·register 병렬성 활용 | bit 수 감소만으로 빨라지지 않는다는 직접적인 근거. GEMM·attention·저장 포맷을 하나의 backend 영역으로 평가 |
| [FLUTE, 2025 개정](https://arxiv.org/abs/2407.10960) | LUT weight의 offline 재배열과 lookup vectorization으로 unpack·shared-memory 비용 감소 | weight-only 저정밀 경로 후보. small-batch GEMM 결과를 높은 concurrency serving 개선으로 일반화하지 않음 |
| [TurboQuant, 2025](https://arxiv.org/html/2504.19874v1) | random rotation과 scalar quantization, inner-product 오차를 위한 residual 보정 | online KV 압축 후보. 낮은 distortion과 품질 평가 결과가 기존 BF16 출력 일치나 serving 가속을 보장하지 않음 |

QServe에서 특히 가져올 원칙은 **메모리에서 절약한 시간보다 unpack·scale·변환 연산이 비싸지 않아야 한다**는 것이다. 압축 데이터를 먼저 전체 BF16 tensor로 복원해 global memory에 쓰고 다시 읽는 설계와, 소비 kernel에서 tile 단위로 복원하는 설계를 구분한다. 후자도 register 압력과 instruction 비용 때문에 자동으로 유리하지는 않다. [QServe 본문](https://arxiv.org/html/2405.04532v3).

추가로 [Fast-TurboQuant, 2026-06](https://arxiv.org/abs/2606.21448)의 multiplier-free online vector quantization을 후속 후보로 찾았다. 현재는 초록 수준 선별이며 구현 성숙도·본문 실험 조건·Riley 적합성 검토를 완료한 후보와 구분한다.

제안하는 평가 batch는 **압축 page/weight 포맷 + fused decode/GEMM + metadata·workspace 관리 + dtype별 dispatch**다. 동일 모델의 동일 양자화 설정을 기준 엔진에도 적용한 비교와, BF16 대비 품질·성능 tradeoff를 각각 보고한다. 기존 BF16 기준 엔진과의 목표 달성 여부를 저정밀 결과로 대체하지 않는다. 실제 할당량에는 scale·zero point·잔여 고정밀 데이터·padding을 포함하고, throughput뿐 아니라 긴 문맥 품질과 tail latency까지 평가한다.

메모리 용량, 이동 byte 수, 실효 bandwidth, 요청 처리 지연은 서로 다른 지표다. buffer 크기가 줄었다고 traffic이 줄었다고 단정하지 않으며, bandwidth 사용률이 높아졌다고 serving이 빨라졌다고도 단정하지 않는다. 중복 메모리 접근이 늘어 bandwidth만 높아질 수 있다.

Roofline의 `min(연산 처리 한계, 실효 bandwidth × arithmetic intensity)`는 후보의 한계를 검토할 틀이다. dtype별 처리량, 어느 메모리 계층의 byte인지, cache hit 여부, 실제 workload shape를 맞춰야 한다. host 대기·kernel launch·불균형·동기화를 제외한 단순 모델을 전체 serving 예측으로 사용하지 않는다.

측정 설계는 cold/warm cache와 전 layer working set을 구분하고, 실제 shape 분포를 반영한다. 최종 판정은 같은 모델·GPU·workload·정확성 조건의 serving throughput, TTFT/TPOT, P95/P99, 실패율로 유지한다. hardware counter가 없으면 추정 traffic과 관측값을 명확히 구별한다.

현재 [전체 연구 보고서](RESEARCH_20260913.md)의 architecture 후보에 이 메모리·연산 관점을 추가한다. 이번 추가 조사에서는 kernel 구현이나 GPU 측정을 수행하지 않았다.
