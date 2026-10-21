# Qwen mixed-batch piecewise Graph·decode Gate/Up v13

2026-10-20 착수, 2026-10-21 KST 검증. mixed MLP piecewise Graph와 선택적 merged decode Gate/Up을 구현했다. 직전 v12B 대비 C16·C32 처리량 중앙값은 각각1.40%·2.37% 증가했다. C16 TPOT p95·p99 중앙값은 각각2.00%·3.29% 줄었고, 5회 반복에서 2% 초과 tail 악화는 없었다. vLLM과의 큰 처리량 격차는 남아 있다. 실행 경로 진단과 최종 자원 감사는 아래에 기록한다.

## 이번 변경의 범위

직전 v12B를 직접 기준선으로 사용한다. v12의 C16 tail 문제를 해결했다는 가정을 두지 않으며, C16·C32에서 기준선·후보·stock vLLM을 각각 새 프로세스로 5회 재측정한다. 과거 승인된 v11E 결과와 합쳐 이번 개선율을 만들지 않는다.

- mixed batch의 decode 행을 안정적으로 앞에 모으고, 해당 prefix의 MLP를 레이어·행 수별 고정 주소 Graph로 실행한다. 호출자가 정한 출력 슬롯, KV block 경계와 요청별 attention은 유지한다. prefill suffix는 기존 native arithmetic family를 사용한다.
- 순수 decode R2 이상은 기존 전체 본체 Graph를 유지한다. R1은 새 MLP piece를 사용한다. mixed attention과 모든 projection을 포괄하는 Graph 구현까지 완료했다는 뜻은 아니다.
- 시작 시 MLP 조각을 미리 capture한 후 cache를 봉인한다. 서빙 중 cache miss는 eager로 실행하며 capture하지 않는다. key에는 stream, 실제 행 수, 입력·가중치·scratch·출력의 allocation identity/offset/length가 포함된다. fixture별 prompt cache는 만들지 않는다.
- decode Gate와 Up의 가중치를 원본 BF16 바이트 그대로 결합한다. 기존 prefill 가중치도 유지해 약3.59GB의 추가 device storage가 필요하다. cold cuBLASLt plan에 고정 recipe를 준비하고, 구성요소 이득이 분명한 R1과 R17~32에만 사용한다. R2~16은 기존 fused kernel을 유지한다.
- 원래 Gate·Up의 BF16 저장 경계, BF16 SiLU 결과와 BF16 multiply 결과를 유지한다. tolerance, dtype, EOS 기준을 바꾸지 않는다. 일반 `merged_gate_up` 옵션을 켜거나 strict HF validation을 완화하지 않는다.

vLLM의 [CUDA Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/)는 uniform decode full Graph와 mixed/prefill piecewise를 별도로 선택한다. 이 실행 단위 구분을 적용했다. 설치된 stock은 `0.27.1+cu129`, Torch `2.13.0+cu129`이며 해당 설치 소스와 실제 시작 로그를 별도 증거로 남긴다. 최신 문서의 모든 기능이 설치 버전에 있다는 주장은 하지 않는다.

## Gate/Up 설계 선택과 거절한 후보

실제 checkpoint layer0 Gate/Up 가중치를 사용했다. 같은 GPU에서 원래 native M1 두 번과 BF16 SiLU/multiply 결과를 기준으로 모든 R1~32와 6종 finite input state를 검사했다. 구성요소 검사는 실제 전체 모델 검사를 대신하지 않는다.

| 설계 | 정확도 검사 | 측정 및 결정 |
|---|---|---|
| Warp-local paired WMMA epilogue 6종 | native BF16 일치 | 기존 실제 fused kernel보다 느려 제외 |
| K128 pipeline 5종 | native BF16 일치 | 느려 제외 |
| CTA 순서·stage 조정 6종 | native BF16 일치 | 느려 제외 |
| merged Lt + 기존 BF16 SiLU | 192개 조합에서 일치 | R1 및 R17~32만 통합 |

선택한 recipe는 id31/tile15/stage17/splitK0/reduction0/swizzle0/custom0이다. SM80·108SM·CUDA runtime12040·cuBLAS/Lt120405 및 변경되지 않은 workspace/SM 환경에서만 준비한다. 모든 행 수에 `AlgoCheck`를 적용하고 필요한 workspace는 cold allocation한다. [NVIDIA 문서](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgocheck)의 `AlgoCheck`는 descriptor 지원과 workspace 검사이며, BF16 수치 동등성의 증명은 별도의 실행 검사로 수행한다.

Graph64 nodes ×8 replay, 세 번 교차 측정한 구성요소 중앙값이다. R2 이상 기준선은 실제 v12 fused kernel이며 R1은 원래 native M1 두 번+SiLU다.

| 활성 decode 행 | 기존 µs | merged µs | 감소 |
|---:|---:|---:|---:|
| 1 | 86.698 | 78.907 | 8.99% |
| 16 | 82.413 | 81.703 | 0.86%, 미적용 |
| 17 | 86.393 | 81.856 | 5.23% |
| 24 | 86.831 | 83.033 | 4.37% |
| 32 | 90.012 | 83.581 | 7.12% |

이 표는 서버 처리량 개선율이 아니다. 처음 시도한 A 통합의 active-row span 오류와 B의 최초 compile 오류도 EFS에 보존한다. 수정된 최종 실행과 실패 표본을 섞지 않는다.

## 정확도와 실행 자원

독립 source copy와 빈 target에서 빌드했다. 회귀 검사 **524개 통과, 40개 ignored**다. 이 중 기존 EFS 상위 경로 소유권 때문에 실패하는 C02 네 테스트만 사전 허용된 private `/tmp` fixture로 재검사했고, 임시 트리 제거도 확인했다. 모델·실행·빌드·벤치마크 scratch는 EFS에 유지했다.

기존 full-vocabulary BF16 파일 **2,464개**는 원래 native와 HF 바이트에 일치했다. 별도 mixed 검사에서도 **1,024개 출력**이 일치했다. 32개 natural128 요청을 한 tick에 하나씩 도착시켜 63단계를 실행하고, 31개 mixed 단계에서 decode R1~31와 prefill을 함께 처리했다. 격 tick마다 입력 순서를 뒤집고 명시적 출력 슬롯도 역순으로 지정했다. 이 검사에서 hot CUDA allocation 증가는 없었다. 이 결과를 모든 가능한 BF16 입력에 대한 수학적 동등성 증명으로 확대하지 않는다.

## C16·C32 정상 서버 재비교

같은 물리 GPU에서 각 concurrency별 v12B / v13B / stock vLLM을 **각각5개의 독립 새 프로세스**로 실행했다. 각 arm은64 warm /192 keep이다. 실행 순서를 반복별로 바꿨다. Qwen3-4B BF16, budget576, max sequence1024, KV blocks=C×64, GPU greedy이며 기준선과 후보 모두 기존 pure decode Graph를 켰다. stock은 prefix cache off, 기본 compile/Graph 경로다. 정상 성능 표본에는 profiler·inspection preload가 없다.

지표별 5회 중앙값이다. 요청 전체를 합쳐 percentile을 다시 계산하거나 component 시간을 서버 개선율로 사용하지 않았다. TPOT는 **요청별 평균 생성 토큰 지연**의 percentile이며 개별 토큰 ITL percentile과 다르다.

| 지표 | C16 v12B | C16 v13B | C16 stock | C32 v12B | C32 v13B | C32 stock |
|---|---:|---:|---:|---:|---:|---:|
| output tok/s | 703.620 | 713.472 | 1160.244 | 907.068 | 928.542 | 1676.065 |
| TTFT p95 ms | 91.730 | 86.729 | 83.111 | 277.191 | 270.619 | 164.010 |
| TTFT p99 ms | 183.911 | 176.938 | 110.187 | 374.744 | 363.354 | 216.976 |
| TPOT p95 ms | 21.452 | 21.022 | 12.374 | 34.588 | 33.523 | 17.709 |
| TPOT p99 ms | 23.094 | 22.334 | 13.247 | 41.777 | 40.107 | 21.496 |
| E2E p95 ms | 705.317 | 689.009 | 413.842 | 1094.143 | 1068.827 | 556.430 |

처리량 개선은 **C16 +1.40%, C32 +2.37%**다. vLLM 대비 처리량 비율은 같은 새 측정 안에서 C16 **60.64%→61.49%**, C32 **54.12%→55.40%**다. 과거 캠페인의 stock 처리량과 섞지 않는다. 큰 격차를 해소했다고 판단하지 않는다.

정상30 arms의6042 SSE 응답을 독립 재검사했다. Riley4032개는 원래 native의 token IDs·text·finish reason·EOS·usage·prompt IDs와 일치했다. stock2010개는 자기 SSE 응답과 usage를 검증했으며, 이 중1802개가 HF token sequence와 일치했다. stock의 다른 수치 경로를 Riley 정확도 허용 기준으로 사용하지 않는다. 처리량은 각 엔진이 실제 생성한 토큰 수와 관측 wall time으로 재계산했다.

## C16 TPOT tail 판정

직전 v12의 과거 outlier에서는 pure R16 평균 GPU 시간이 약11.62→11.60ms로 늘지 않은 반면 mixed M256 횟수는66→84, pure M16 횟수는194→177로 바뀌었다. 이 집계만으로 특정 커널을 원인으로 확정할 수 없다. 이번에는 mixed/R1 구간의 host launch와 MLP 실행 단위를 바꾸고, 동일 입력을 새로 반복해 tail을 확인했다.

| 반복 | C16 TPOT p95 변화 | C16 TPOT p99 변화 | C32 TPOT p95 변화 | C32 TPOT p99 변화 |
|---:|---:|---:|---:|---:|
| 0 | -2.31% | -7.44% | -3.63% | -4.44% |
| 1 | -1.92% | -2.52% | -4.20% | +1.14% |
| 2 | -1.59% | +1.64% | -3.77% | -3.30% |
| 3 | +0.33% | -3.64% | -3.40% | -1.47% |
| 4 | -3.39% | -1.23% | -1.77% | -3.83% |

C16 TPOT p95·p99 중앙값은 **-2.00%·-3.29%**다. C16·C32 모두 TTFT/TPOT p95·p99의 반복별 **2% 초과 악화 플래그는0개**다. 모든 반복에서 p99가 줄었다는 주장은 하지 않는다. 이번 workload와 5회 반복에서 이전 tail 수용 기준을 통과했다는 의미이며, 과거 특정 스케줄 순서를 완전히 재현해 단일 원인을 제거했다는 뜻은 아니다. 요청별 percentile 보간 bracket과 각 arm의 batch histogram을 증거에 보존했다.

## 실제 실행 경로와 남은 병목

정상 서버 성능과 별도로64 warm /64 keep의 NVTX·Nsight node trace를 수집했다. mixed MLP Graph를 적용한 후보의 `cudaGraphLaunch` 수는 C16 **65→1291**, C32 **36→1115**였다. C16 pure R2~16의 기존 fused Gate/Up은 여전히 그대로 실행된다. C32의 R17~32는 기존 fused Gate/Up 대신 Lt+BF16 SiLU를 사용하므로, 개별 kernel 수가 늘어도 GPU 시간이 줄 수 있다.

| 별도 진단 항목 | C16 v12B→v13B | C32 v12B→v13B |
|---|---:|---:|
| 전체 kernel 수 | 105394→104276 | 77478→79899 |
| GPU kernel 시간 합 ms | 1777.878→1721.548 | 1386.813→1379.731 |
| `cudaLaunchKernel` 호출 수 | 56347→50255 | 47526→45774 |
| GPU kernel union / span | 76.76%→77.42% | 72.25%→68.55% |

C32에서 GPU 시간 감소는 약0.51%이고 union/span 비율은 오히려 낮았다. Graph를 붙였다고 host 실행 병목이 해결됐다고 판단하지 않는다. 진단은 한 번의 별도 표본이고, node trace·inspection 비용과 batch 구성 차이가 있다. profiler의 HTTP 처리량을 정상5회 표에 섞지 않는다.

후보 C16 trace의 fused R2~16 Gate/Up은3564회/307.774ms로 여전히 가장 큰 단일 kernel 집단이다. 이 구간은 native 누적 순서를 유지하는 대안17종이 더 느렸고, merged Lt도 구성요소 차이가 작아 기존 커널을 유지했다. 따라서 **C16의 주된 Gate/Up 연산 자체를 크게 줄인 변경이 아니다.**

새 MLP Graph 안의 Lt·Down은 eager 호출의 NVTX 상관관계를 잃는 경우가 있어, 후보 C32에서384.948ms가 `other-unattributed`에 남았다. 직접 분류된 Gate/Up이362.946→236.346ms로 줄었다는 것만으로 전체 Gate/Up이 그만큼 빨라졌다고 주장하지 않는다. 그 밖의 직접 식별된 C32 K/V172.230ms, QK/PV138.184ms와 softmax23.923ms도 남아 있다. `cudaStreamSynchronize` 시간은 GPU 완료 대기와 겹치므로 kernel 시간에 더하지 않는다. DRAM/L2 counter가 없어 memory-bound라는 판정도 하지 않는다.

별도 instrumentation을 켠 finite profiler child에서 실제 `cublasLtMatmul`의 N19456/K2560 호출과 algorithm metadata를 읽었다. **R1과 R17~32의17개 행 수 모두 status0, recipe31/15/17/0/0/0/0**이었다. 이 진단을 정상 성능 표본에 포함하지 않았다. 최초 non-profile probe는 runner의 inspection 보호 조건에 의해 서빙 전에 거부됐으며, 이후 profiler 진단으로 실행했다. 새 runtime binary를 바꿔 증거를 만든 것은 아니다.

같은 GPU7의 새 stock64 warm /64 keep trace는 **32325 kernels /847.570ms**, kernel union/span **84.26%**였다. Riley 후보는 **79899 /1379.731ms /68.55%**다. 실제 stock 시작 로그에서 `FLASH_ATTN`, FlashAttention version2, `FULL_AND_PIECEWISE`가 확인됐다. 다른 batch 구성과 수치·출력 경로를 포함한 합계이며 stock의 익명 GEMM을 특정 연산으로 임의 분류하지 않았다.

따라서 다음 큰 개선의 우선순위는 **아직 그대로인 R2~16 Gate/Up**, prefill을 포함한 선형·attention 실행 범위, 그리고 GPU metadata·비동기 실행을 함께 다루는 구조다. 이번에 async scheduler나 전 범위 mixed Graph를 구현했다고 주장하지 않는다. native BF16 누적 경계를 바꾸지 않는 것과 EOS·취소·KV 회수의 완료 시점을 보존하는 것이 후속 작업의 필수 조건이다.

## 최종 감사와 증거

935개 source snapshot이 현재 로컬 파일과 일치한다. v12B 대비 변경은8개 source 파일이다. 정상30 arms와 별도 진단의 총6679 SSE 응답을 독립 재검사했다. 모든 정상 Riley arm은 시작 전 body warmup을 마쳤고, 후보 MLP captures는 C16 **576**, C32 **1152**로 prelisten seal과 종료 통계가 같았다. 서빙 중 추가 MLP capture·eager fallback·stream reset은0이다. 기존 body Graph도 listen 이후 새 capture가 없었다.

측정된 정상 arm의 장치 전체 사용량 최고는 **18255MiB**로22GiB 한도 안이었다. 프로세스별 NVML 수치가0으로 보고되는 경우가 있어, 프로세스 합산만으로 메모리를 판단하지 않고 장치 전체 사용량도 검사했다. 독립 PID/start_ticks·소유권 검사는 유지했다. 완료 시 active/waiting/KV block gauge가0으로 회수됐다.

추가 가중치와 cold Graph 준비에는 비용이 있다. readiness 중앙값은 C16 **53.108→56.867초**, C32 **54.611→58.381초**였다. 이 EFS 실행 환경의 프로세스 시작~readiness 시간이며 정상 요청 TTFT에 합산하지 않는다.

- 기준 v12B binary SHA256: `e4a4f4cafe1d005b80e4a0dde9aed249fca3b42f2d721408895e0ce4cdca32c9`
- 최종 v13B binary SHA256: `9dd87720ceb5c9f99532b8b01851557598032a6fe5ba037316e327eba1505566`
- source935 manifest SHA256: `474ebd145037fc91a023dcc8a4adc996be1d20eaead886a5058d9086e59a6fbe`

압축 증거 SHA256: `4df7f88552b077d01105f169cb0a8bc6a53c4258bcb32783825675dc11ad5945` (6,338,800bytes,714 entries). 다운로드 후 모든 ZIP entry 해시와 source935개를 로컬에서 재검사했다.

로컬 증거: [감사·측정·소스 증거](../benchmarks/evidence/20261020-piecewise-v13/), [압축 증거](../benchmarks/evidence/20261020-piecewise-v13.zip). 상세 실행 기록은 EFS에 보존한다. full BF16과 원래 Nsight trace는 EFS heavy evidence index의 파일별 SHA256으로 연결하고, 원래 실패 표본도 유지한다.

최종 원래 holder의 PID/start_ticks, idle20GiB 및 외부 GPU PID 부재를3회의 새 표본으로 확인했다. holder context를 해제하거나 재시작하지 않았다.

EFS 실행 root: `/mnt/efs/nlp/psyche/riley-piecewise-v13b-20261020`. 원래 GPU6/7 holder PID2027914/2027915 및 start_ticks6083215119/6083215121을 유지한다. 모든 GPU 작업은 원래 controller와 finite observer를 통과한다. idle20GiB 홀드, active64MiB context anchor를 유지하며 합산22GiB 한도를 검사한다. CUDA context를 중단·재시작하거나 다른 사용자의 프로세스를 종료하지 않는다.
