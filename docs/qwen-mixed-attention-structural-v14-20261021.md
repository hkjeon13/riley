# Qwen R2~16 Gate/Up·mixed attention·host 실행 구조 v14

2026-10-21 KST. 구조 후보4개와 여러 attention/kernel 대안을 구현·검증했지만 **유의미한 전체 처리량 개선을 얻지 못했다.** 마지막 D 후보는 직전 검증된 v13B 대비 C16 처리량 중앙값 +0.76%, C32 -0.27%였고, 반복별2% 초과 tail 플래그도3개였다. 따라서 D를 기본 코드에 채택하지 않는다. 기본 소스는 v13B를 유지하고, [적용하지 않은 재현 패치](../benchmarks/candidates/sm80-structural-v14/prototype.patch)와 실행 증거를 보존한다.

D의 original-native/HF full-vocabulary BF16 바이트 일치, native SSE 출력 계약 및 GPU6/7 holder context 검사는 성능 판정과 별도로 수행한다. 정확도가 통과했다는 이유로 성능 개선을 주장하지 않는다.

## 설계와 후보 선택

vLLM의 [CUDA Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/)는 batch composition과 attention backend 지원 여부로 full/piecewise/eager를 구분한다. FULL_AND_PIECEWISE는 uniform decode 전체 Graph와 다른 batch의 piecewise를 함께 사용한다. 이 원칙을 참고했으며, Riley가 vLLM의 전체 scheduler·compiler·attention 구현을 이식했다는 뜻은 아니다. 설치된 stock의 소스 해시·버전과 실제 시작 로그를 별도로 보존한다.

| 후보 | 구조 | 결정 근거 |
|---|---|---|
| A | mixed logical iteration을 decode full-body Graph pass와 prefill pass로 분리 | C16 screen 처리량 681.3205→639.6094 tok/s, 약6.12% 악화. 제외 |
| B | A + embedding 중간 report 대기를 완료 시점으로 이동 + 모든 R의 merged Gate/Up | C16 screen 683.7374→645.1523 tok/s, 약5.64% 악화. 제외 |
| C | 단일 batch 유지, embedding 완료 시 검증, R2~16 merged Gate/Up, R1 dynamic attention와 전체 decode Graph | C16 exploratory5회만 보존. 긴 context의 기존 MLP fallback을 보완한 D로 검증을 진행했으며 C를 채택하지 않음 |
| D | C의 R1 fallback 보완 + 선택적 softmax/PV 융합 | CPU·native/HF 정확도 통과. 정상 처리량·tail 판정 미달로 미채택 |

A/B의 64 warm /64 keep 단기 결과는 최종 5회 비교와 섞지 않는다. 두 pass 사이의 추가 완료·출력 복사 비용을 전체 Graph 재사용 이득으로 상쇄하지 못했다. 전체 모델의 수치 검사를 통과하는 것과 성능 후보 채택은 구별한다. B의 staggered mixed 1,024개 full-vocabulary 출력은 기존 native/HF와 일치했지만 성능 악화 때문에 분리 구조를 채택하지 않는다.

다른 라이브러리에서 참고한 범위는 다음과 같다. 라이브러리 자체를 benchmark 중 추가 설치하거나 stock의 backend를 바꾸지 않았다.

| 근거 | 적용·검증한 원칙 | 이번에 완료하지 않은 범위 |
|---|---|---|
| [vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) | batch composition/attention 적격성에 따른 전체·piecewise 경로 구분, R1 전체 Graph 확대 | vLLM compiler·scheduler 전체 이식 |
| [FlashInfer paged decode API](https://docs.flashinfer.ai/api/attention.html) | graph용 device metadata buffer의 주소·수명과 batch key를 명시적으로 유지 | FlashInfer의 attention arithmetic을 strict native 경로에 그대로 대입 |
| [TensorRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6.post1/features/attention.html) | 낮은 batch와 높은 batch에서 CTA 병렬성과 shared-memory 사용을 따로 검증 | online softmax·multi-block merge 등 다른 누적 경로의 무검증 도입 |

## 실험 D의 실행 구조

C를 기반으로 다음 구조를 구현했다. D는 별도 source copy와 빈 target에서 다시 빌드하고, D 자체를 전체 모델·서버 검사했다. 다음 내용은 기본 코드에 적용되지 않은 실험 패치의 동작이다.

- **R24~32 dynamic paged attention**: QK의 병렬 실행은 유지한다. softmax가 BF16 QK를 읽고 기존 BF16 scale/FP32 softmax tree를 수행한 뒤 BF16 probability를 shared memory에 저장한다. PV도 원래 BF16 입력과 FMA/WMMA 누적 순서로 계산해 BF16 출력을 쓴다. 기존 세 launch를 두 launch로 줄인다.
- PV의 기존 WMMA 가족인 context 길이8의 배수는4 query heads, 기존 SIMT 가족은2 query heads를 공유한다. 요청별 context가 달라도 device metadata를 읽어 분기한다. 지원 범위는 원래 SM80 native 가족의 context24~216이며, 이 범위를 넓혔다고 주장하지 않는다.
- R1~23은 기존 별도 softmax/PV를 유지한다. R16 구성요소에서 일관된 이득이 없었기 때문이다. 요청 수·hardware 적격성으로 선택하며 특정 prompt/token/fixture 값을 캐시하지 않는다.
- R1 전체 decode Graph는 지원 context에서 사용한다. 긴 context, 출력 형태 등 전체 Graph 적격성 밖에서는 기존 MLP piece fallback을 유지한다. 전체 Graph를 지원하면서 이미 있던 fallback을 제거하지 않는다.
- embedding 중간 report는 iteration 완료 시 기존 pinned report 경로에서 확인한다. CPU token admission은 KV 변경 전 유지하며, report/CUDA 실패 시 owner를 poison하고 출력을 publish하지 않는다. scheduler의 EOS·취소·KV reservation commit 시점은 바꾸지 않는다.

전체 QK/softmax/PV를 한 CTA에 모은3종은 각각37,056개의 구성요소 조합에서 바이트가 일치했지만 병렬성이 줄어 C16에서 느렸다. softmax/PV만 결합한3종을 더 비교한 뒤 위의 요청 수·PV 가족 구분을 선택했다. 이 최종 hybrid도 깨끗한 GPU7에서 R1~32·context24~216·6종 finite state(3종은 요청별 이종 context), 총37,056조합을 바이트 비교했다. 구성요소 검사를 전체 모델의 수치 보장으로 대체하지 않는다.

hybrid의 구성요소 교차3회 중앙값이다. 원래 QK/softmax/PV 합계와 새 QK+softmax/PV 합계를 같은 GPU7에서 Graph64×8 replay로 비교했다. 서버 처리량 개선율이 아니다.

| context | R24 기존→hybrid µs | 감소 | R32 기존→hybrid µs | 감소 |
|---:|---:|---:|---:|---:|
| 128 | 39.753→36.529 | 8.11% | 47.909→40.806 | 14.83% |
| 129 | 36.844→33.317 | 9.57% | 45.021→39.805 | 11.59% |
| 144 | 42.536→39.059 | 8.17% | 50.690→42.788 | 15.59% |
| 159 | 41.611→37.589 | 9.66% | 54.485→45.560 | 16.38% |
| 192 | 54.366→48.874 | 10.10% | 69.948→58.104 | 16.93% |
| 213 | 55.633→51.554 | 7.33% | 80.655→72.063 | 10.65% |

## R2~16 Gate/Up 대안

- 기존 누적 두 chain을 유지하도록 K축을 재배열하고 split-K2를 사용하는 대안은 일치하는 recipe가 있었지만 R16 약91.029µs로 기존 약82µs보다 느렸다.
- 원래 BF16 비트를 그대로 복원하는 13-bit 저장·raw fallback 방식도 검사 입력에 정확했지만, 직접 unpack 비용 때문에 R16 약217µs로 더 느렸다. dtype이나 정확도를 낮추는 방법으로 전환하지 않았다.
- 기존 merged Lt recipe31/tile15/stage17/splitK0/reduction0/swizzle0/custom0을 새 GPU7 실행에서 모든 R1~32·6종 finite 입력으로 재검사했다. 192조합 모두 native BF16 결과와 일치했다. R16의 세 교차 측정은 기존82.385/82.338/82.409µs, merged81.760/81.711/81.725µs였다. 이 작은 구성요소 차이를 서버 처리량 개선율로 사용하지 않는다.

## 정확도와 GPU 사용 범위

모델·스크립트·독립 source copy·빈 target 빌드·벤치마크 scratch는 EFS에 유지한다. 이전에 허용된 네 C02 테스트의 private 임시 fixture만 `/tmp`에서 재검사하고 제거한다.

GPU6에는 사용자가 공존을 허용한 작은 fresh-demand 예측 작업이 실행 중이다. 같은 UID, 현재 살아 있는 고정 부모 PID/start ticks, 정확한 script directory의 `*_predict.py` 자식, 최대2GiB 조건으로 제한한다. 다른 작업을 종료하지 않는다. 정상 C16·C32 비교는 **둘 다 깨끗한 물리 GPU7에서 순차 실행**한다. GPU6은 정확도·구성요소 검사에 사용한다. 서버 전체의 CPU/IO가 완전히 격리된 환경이라는 주장은 하지 않는다.

원래 holder GPU6 PID2027914/start_ticks6083215119, GPU7 PID2027915/start_ticks6083215121을 유지한다. idle20GiB, active64MiB anchor 전환 시 context를 해제하거나 재시작하지 않는다. controller와 finite observer가 장치 전체22GiB 한도, 실제 자식 PID/start ticks 및 소유권을 검사한다. 프로세스별 NVML 메모리가0으로 보고되는 경우에도 장치 전체 사용량을 검사한다.

## 정상 서버 비교와 판정

C16·C32 모두 동일한 물리 GPU7에서 v13B/D/stock vLLM을 각각5개 독립 새 프로세스로 실행했다. 각 arm은64 warm /192 keep이며 반복별 실행 순서를 바꿨다. Qwen3-4B BF16, token budget576, 최대 sequence1024, KV blocks=C×64, GPU greedy, Riley 양쪽 모두 decode Graph 사용, stock은 prefix cache off 및 기본 compile/Graph 경로다. 정상30 arms에는 profiler/inspection preload가 없다.

측정 workload는 prompt115~130tokens인12개 fixed 요청을 반복한다. 생성 상한8/16/24/32tokens가 균등하게 섞여 각 arm의 실제 생성량은3,840tokens다. 별도 correctness에는 natural16/128/397 prompt도 포함한다. 아래는 지표별5회 중앙값이며 percentile을 요청 전체에 다시 pooling하지 않았다. TPOT는 요청별 평균 생성 토큰 지연으로 token ITL percentile과 다르다.

| 지표 | C16 v13B | C16 D | C16 stock | C32 v13B | C32 D | C32 stock |
|---|---:|---:|---:|---:|---:|---:|
| output tok/s | 717.594 | 723.023 | 1149.627 | 933.101 | 930.614 | 1674.932 |
| TTFT p95 ms | 88.177 | 87.253 | 83.390 | 270.162 | 270.486 | 163.738 |
| TTFT p99 ms | 177.466 | 177.557 | 110.877 | 364.730 | 364.608 | 216.655 |
| TPOT p95 ms | 20.994 | 20.720 | 12.380 | 33.801 | 33.512 | 17.755 |
| TPOT p99 ms | 22.242 | 21.652 | 13.076 | 40.752 | 40.010 | 21.634 |
| E2E p95 ms | 689.143 | 683.791 | 414.511 | 1063.540 | 1062.172 | 559.684 |

D의 처리량 변화는 C16 **+0.76%**, C32 **-0.27%**다. stock 대비 처리량 비율은 같은 새 측정 안에서 C16 **62.42%→62.89%**, C32 **55.71%→55.56%**다. 큰 격차가 줄었다고 판단할 수 없다. 과거 v13 캠페인의 stock 수치와 섞지 않는다.

## Tail 판정과 초기 cohort 관측

D의 TPOT p99 중앙값은 C16 -2.65%, C32 -1.82%였다. 그러나 반복별2% 초과 악화는 다음3개이며, **모든 반복2% 이내라는 기존 수용 기준은 통과하지 못했다.**

| concurrency / 반복 | 지표 | 악화 |
|---|---|---:|
| C16 /0 | TPOT p99 | +3.53% |
| C32 /2 | TPOT p99 | +4.74% |
| C32 /4 | TTFT p95 | +3.76% |

C16 반복0의 p99 보간 index는189.09다. 기준선 rank189는 `logic`8tokens/TPOT21.363ms, rank190은 `summary`8tokens/26.217ms였다. D의 rank189는 `python`16tokens/22.200ms, rank190은 `summary`8tokens/26.298ms였다. 서로 다른 요청이 percentile bracket에 들어왔다.

같은 ordinal1 `python` 요청의 TTFT는88.545→42.521ms, E2E는388.584→376.079ms로 줄었지만 TPOT는19.993→22.200ms로 늘었다. 같은 worst `summary` 요청의 E2E는227.244→227.187ms였다. 첫 토큰이 더 일찍 나온 경우 요청별 평균 TPOT가 커지는 관계와 bracket 변화를 관측한 것이다. 이 자료만으로 단일 커널 저하나 특정 scheduler 정책을 원인으로 확정하지 않는다. tail 플래그를 없애거나 실패 표본을 제외하지 않았다.

모든10개 before/after pair의 percentile bracket, 같은 ordinal 요청의 TTFT/TPOT/E2E 및 batch-shape snapshot을 보존한다. shape snapshot 사이에는 correctness·disconnect recovery·warm·measurement가 함께 있으므로 측정 window만의 kernel trace로 취급하지 않는다.


## 구성요소 이득이 서버 개선으로 이어지지 않은 이유

추가 cache-displacement 진단은64MiB streaming write 뒤 한 iteration Graph를 event로 측정했다. write 시간은 비교 시간에서 제외한다. 이 검사는 GPU6의 허용된 공존 작업이 있는 별도 구성요소 진단이며, DRAM/L2 counter로 완전한 cache flush를 입증한 것은 아니다.

D와 같은 unpartitioned hybrid에서 이종 context115~162의 R24는62.034→63.928µs(**3.05% 악화**), R32는73.442→72.276µs(**1.59% 감소**)였다. clean GPU7에서 동일 길이·같은 buffer를 반복한 앞선 구성요소 결과7~17%가 모든 pipeline 상태에 유지되는 것은 아니다. cache 상태·context 구성·동시 작업이 다른 진단을 합쳐 하나의 개선율로 만들지 않는다.

출력 depth를64/32로 나누는 추가2종도37,056조합씩 바이트가 일치했다. 그러나 이종 context 진단의 R32에서64-slice는73.356→116.132µs,32-slice는73.472→150.454µs로 느렸다. 동질 context만의 좋은 수치를 보고 이 대안을 채택하지 않는다.

정상 비교와 별도로64 warm /64 keep의 Nsight node trace를 수집했다. 아래는 각1회의 진단이며 정상5회 처리량을 대신하지 않는다. profiling 및 interposer가 batch 구성과 host 실행을 바꿀 수 있다. CPU API 대기 시간과 GPU kernel 시간은 겹치므로 더하지 않는다.

| 별도 trace 항목 | C16 v13B→D | C32 v13B→D |
|---|---:|---:|
| GPU kernel 수 | 103951→108021 | 79966→78962 |
| GPU kernel 시간 합 ms | 1712.785→1733.231 | 1396.635→1370.216 |
| kernel union / span | 81.28%→82.63% | 71.02%→75.62% |
| `cudaLaunchKernel` 호출 | 49423→49965 | 47686→42377 |
| `cudaGraphLaunch` 호출 | 1256→1256 | 1184→975 |
| `cudaStreamSynchronize` 호출 | 306→204 | 195→132 |

동기화 호출 수는 약1/3 줄었지만 C16의 전체 GPU kernel 시간은 약1.19% 늘었다. R2~16의 이전 fused Gate/Up1kernel을 Lt+SiLU2kernels로 바꾼 경로를 포함하며, 실제 batch 구성 차이도 있어 kernel 증가를 전부 한 변경으로 귀속하지 않는다. 동기화 횟수가 줄었다는 사실을 그만큼 서버가 빨라졌다는 주장으로 바꾸지 않는다.

D C32 trace의 새 softmax/PV kernel은1,116회/36.073ms였다. dynamic decode QK2,340회 중47.69%에 대응하며, 새 kernel 시간은 전체 GPU kernel 합1,370.216ms의2.63%다. 나머지 decode 구간·prefill·projection·MLP를 모두 최적화한 것이 아니다. Nsight의 실제 grid도 X16/Y24~32/Z1, block128로 확인했다. C16과 v13B에는 이 kernel이 없었다.

동일 GPU7의 새 stock C32 trace는32,396 kernels/845.413ms, kernel union/span87.70%였다. stock 시작 로그에서 `FLASH_ATTN`, FlashAttention version2 및 `FULL_AND_PIECEWISE`를 확인한다. stock은 수치·출력 경로도 다르므로 총합 차이를 특정 연산의 인과 효과로 해석하지 않는다.

Graph 내부 cuBLAS 호출의 NVTX 상관관계가 사라지는 문제가 남아 있다. D의 `other-unattributed`는 C16 **703.478ms**, C32 **468.320ms**다. 이 시간을 Gate/Up이나 Down으로 임의 분류하지 않는다. 직접 분류된 C32 K/V167.919ms, Gate/Up155.593ms, QK/PV108.271ms, Down87.161ms도 전체 해당 연산 시간의 완전한 합계가 아니다. 추가 Graph-aware attribution과 실제 pipeline의 DRAM/L2·SM counter 없이는 memory-bound 등의 결론을 내리지 않는다.

## 정확도·실행 경로·자원 감사

D는 독립 source copy와 빈 target에서 빌드했다. 일반 회귀521개와 허용된 private `/tmp` C02 fixture4개를 합쳐 **525개 통과,40개 ignored**다. 원래 EFS 상위 경로 소유권 때문에 발생하는 네 실패를 보안 검증 완화로 우회하지 않았다. 네 테스트의 임시 트리는 제거했다.

D의 original-native/HF full-vocabulary BF16 파일 **2,464개**와 별도 mixed 출력 **2,016개**가 모두 바이트 단위로 일치했다. mixed는32개 natural128 요청을 tick마다 하나씩 도착시켜63tick/31mixed 단계를 만들고 입력·출력 슬롯을 역순으로 배치했다. 두 번째 variant는 prefill에 output slot을 주지 않아992개 출력을 검사한다. 잘못된 prefill token이 들어간 batch는 GPU 작업 전에 거부되며 shape·MLP stats·owner 재사용 가능성이 유지됐다. hot CUDA allocation 증가는 없었다.

저장된 전체 SSE **6,759개**를 독립 재검사했다. 정상30 arms의6,042개 중 Riley4,032개는 원래 native token IDs·text·finish reason·EOS 유무·usage·prompt IDs와 일치했다. stock2,010개는 자기 SSE와 usage를 검증했으며 stock의 다른 수치 경로를 Riley 허용 기준으로 사용하지 않았다. 정상 Riley의 첫 arm별2회 disconnect 후 KV·active/waiting gauge가0으로 회수되고 allocation이 유지되며 재시도 응답도 native와 일치했다. 표본의 finish는 length이며 positive early-EOS 완료를 새로 검증했다는 주장은 하지 않는다.

정상 Riley 양쪽은 listen 전에 body Graph를 준비했다. v13B의 body keys는 R2~C, D는 R1~C였다. D의 MLP captures는 C16 576/C32 1152로 시작 seal과 종료 통계가 같았다. 서빙 중 추가 capture·eager fallback·stream reset은0이었다.

별도 call probe에서 실제 N19456/K2560 cuBLASLt 호출의 R1~32 모두 status0 및 recipe31/15/17/0/0/0/0을 확인했다. 첫 호출/행 수와 host API 호출 수는 startup/capture를 포함하며 GPU Graph replay 횟수가 아니다. 최초 r2 진단은 라이브러리 복사 누락으로 loader가 preload를 무시해 실제 호출 증거가 없었다. 이 기록은 보존하고, 라이브러리를 고정한 r3에서 재검증했다. 정상30 arms에는 이 라이브러리를 사용하지 않았다.

정상 arm의 장치 전체 사용량 최고는 **18,265MiB**로22GiB 한도 이내였다. 측정 중 GPU7에 다른 GPU PID는 없었다. CPU와 IO가 서버 전체에서 격리됐다는 뜻은 아니다. 종료 후 원래 GPU6/7 PID/start ticks와 idle20GiB를3개의 새 표본으로 확인했다. GPU6의 허용된 작은 외부 작업은 종료하지 않았다.

## 기본 코드 유지와 재현

**기본 소스는 v13B로 유지한다.** 이 소스935개가 원래 v13B manifest와 모두 일치함을 확인했고, 기존 v13의 compiled CPU test artifacts를 다시 실행해 **524개 통과,40개 ignored**를 확인했다. 같은 v13 binary가 이번 정상 비교의10개 before arm에서도 원래 native/SSE·자원 검사를 통과했다. D의525개 결과를 기본 코드의 새 검사 수로 혼동하지 않는다.

- 유지한 v13B binary SHA256: `9dd87720ceb5c9f99532b8b01851557598032a6fe5ba037316e327eba1505566`
- 유지한 source935 manifest SHA256: `474ebd145037fc91a023dcc8a4adc996be1d20eaead886a5058d9086e59a6fbe`
- 미채택 D binary SHA256: `c46ee3eb264091de3b78e21d6f02187dfb7090b6a44f7162565668230b3ac71a`
- 미적용 D 재현 patch SHA256: `4491e3a49ce20f7440c780938fb9980713cc12587ede7022056a6733075061ee`

별도 디렉토리에서 v13B에 patch를 적용해 재구성한 D의935개 소스도 실험 manifest와 모두 일치했다. 재현 패치는 기본 코드에 적용하지 않는다. 이번 커밋은 마지막 Git checkpoint 이후 미커밋 상태였던 검증된 v13까지의 누적 코드와, 이 v14 실험의 문서·재현 patch·압축 증거를 함께 보존한다. 성능 표의 before는 Git parent가 아니라 위 해시의 v13B binary다.

최종 원격 실험 root는 `/mnt/efs/nlp/psyche/riley-structural-v14d-20261021`이고 A/B/C 탈락 후보와 연구도 별도 EFS root에 유지한다. full BF16·원래 Nsight trace·실행 ELF는 EFS에 두고 파일별 SHA256 index로 연결한다. 압축 증거는11,119,006bytes/1,920entries이며 SHA256은 `62edbb7f085b6131b2d3953b8bcdeb58123e39700880f8f50565fee958c65a48`다. 다운로드 후 모든 ZIP entry의 크기·SHA256과 source 재구성을 로컬에서 재검사했다. heavy index는4,498개 원격 BF16/trace/ELF 파일을 연결하며 원본을 Git에 넣지 않는다.

[압축 증거](../benchmarks/evidence/20261021-structural-v14.zip), [감사·통계 index](../benchmarks/evidence/20261021-structural-v14/README.md), [기본 v13 증거](../benchmarks/evidence/20261020-piecewise-v13.zip).


## 다음 큰 작업의 우선순위

추가 launch 설정이나 단일 구성요소 속도를 전체 개선으로 해석하는 접근은 중단한다. 먼저 Graph-aware kernel attribution과 실제 pipeline의 cache/SM counter를 확보하고, native BF16 경계를 유지하는 prefill attention fusion 및 metadata·worker overlap을 별도 큰 실험으로 검증한다. mixed 실행을 나누더라도 중간 완료·pinned buffer 재사용·EOS/취소/KV 회수 계약이 먼저 충족돼야 한다. 이번 D의 미채택 결과를 이러한 구조 변경의 성능 증명으로 사용하지 않는다.
