# Qwen3-4B: native-order Down·paged QK/PV 개선과 stock vLLM 재검증

상태: 구현·정확도·C16/C32 비교·보조 검사·독립 감사 완료. 증거 압축파일을 내려받아656개 파일의 SHA256을 재검증했다. 구성요소 수치와 서버 전체 성능을 구분한다.

## 유지한 기준

- 기존 same-A100 native 및 기존에 생성한 fresh same-A100 HF 기준과 전체 vocab BF16 바이트 일치. 허용오차를 추가하지 않는다.
- 생성 token IDs·text·prompt·usage·finish reason과 원래 EOS를 유지한다. vLLM의 출력 일치율은 별도로 보고한다.
- 물리 GPU6·7만 사용한다. 기존 holder PID2027914/2027915와 CUDA context를 유지하며, 실행 중64MiB·대기 중20GiB로 크기만 변경한다. 테스트 종료 후 해제하지 않는다.
- 모델·빌드·벤치마크·증거는 EFS에 둔다. 사용자 승인 범위대로 C02 테스트4개의 임시 fixture만 private `/tmp`를 사용하고 제거한다.

## Down: native split-K 산술을 유지한 Tensor Core 재사용

SM80/cuBLAS12.4의 실제 native M1은 `64x64_sliced1x2` BF16 Tensor Core kernel과 BF16 split-K reduction을 사용했다. 단순 WMMA와 operand transpose는 일치하지 않았다. 두 split 각각에서 K32 단위로 교차하는 FP32 accumulator2개를 유지하고, FP32 sliced 합→BF16 partial→FP32 split 합→BF16 출력 경계를 재현하자 비트 차이가0이 됐다.

K64 타일을4단계 `cp.async`로 미리 읽고16개 요청이 가중치를 재사용한다. 기존 private Lt workspace 안에 bounded partial을 저장하며 추가 replay lease를 생략하지 않는다. 실제 행 수12..32에만 적용한다. 1..11행 및 prefill은 기존 경로를 유지한다.

구성요소는 실제 L0 Down 가중치·입력을 포함한6상태, R1..32의192회 비교에서 BF16 값8,110,080개 모두 일치했다. 같은 프로세스에서 baseline/candidate 순서를 교대한3회 CUDA-event 측정의 중앙값:

| 실제 행 수 | 기존 Lt4 반복 µs | 후보 µs | 변화 |
|---:|---:|---:|---:|
| 4 | 45.72 | 113.78 | 느려서 제외 |
| 8 | 91.42 | 114.26 | 느려서 제외 |
| 12 | 137.07 | 116.42 | -15.1% |
| 16 | 182.67 | 114.69 | -37.2% |
| 24 | 274.09 | 126.85 | -53.7% |
| 32 | 365.34 | 133.87 | -63.4% |

## Attention: 요청별 호출을 native-family 묶음으로 변경

CUPTI runtime와 driver callback으로 T1..512의 실제 QK/PV kernel family를 확인했다. SM89용 reduction을 그대로 확대하지 않았다. 적용 범위인 T24..216에서는 QK의8-lane contiguous16/tree,16-lane stride16/serial, native CUTLASS WMMA family를 구분한다. PV는2-lane interleaved SIMT와 native CUTLASS WMMA family를 구분한다.

요청별로 gather→QK→softmax→PV를 반복하는 대신, 서로 다른 T를 가진 독립 singleton 요청2..32개를 family별로 묶는다. QK kernel 최대3개→기존 BF16 scale/softmax1개→PV kernel 최대2개를 실행한다. Paged KV를 직접 읽고 원래 BF16 score/probability 경계를 유지한다. Sequence slot·offset·page ID를 검사하고 invalid page의 기존 NaN 동작을 유지한다. Singleton, prefill, T범위 밖 요청은 기존 경로로 돌아간다.

새 cold chunk scratch는 최대3MiB다. 기존 per-request scratch는 fallback/prefill용으로 계속 보유한다. 따라서 기존 mode10 대비 새 scratch가 추가되는 것이며,256MiB를 실제로 회수했다고 주장하지 않는다.

개별 요청에서 QK/PV를 한 커널·두 커널·세 커널로 바꾸는 후보는 일부 길이에서 느려 제외했다. 묶음 후보의 첫 T24..304 검사는 PV의 T240·304에서 값8개가 달라 채택하지 않았다. 원본 로그와 재현 코드를 보존했고, 정확도 허용오차를 완화하지 않았다. 적용 범위를 T24..216으로 제한하고 dense probability·invalid metadata를 포함한6상태와 R1..32를 재검사해576회, BF16 값37,048,320개의 차이0을 확인했다. 전체 범위의 single-request operator 검사와 full-model 검증도 별도로 수행한다.

## 실행 범위와 검증 프로토콜

- A100-SXM4-40GB,108SM, CUDA12.4, cuBLAS/cuBLASLt120405, native SM80 image 및 unmodified workspace/SM 환경으로 제한한다.
- 기본 실행 모드는 바꾸지 않는다. 기존 opt-in `native-prefill-sm80-v5`를 확장하고 startup에 Down/paged v6 선택을 표시한다.
- 최종 v6 binary SHA256은 `6e1c2a46227629110068ebcb7ce1c85185ace0f343b1cce1b28d75d91a16d574`이며,935개 source manifest와 결합한다.
- Riley baseline은 이전에 검증된 mode10 binary SHA256 `b704f82694a32edc791cdf035c6bb203c7cf3b79932c51ea713d32545821bab5`다.
- Riley 전후와 stock vLLM 모두 budget576, BF16, max sequence1024, KV block16·blocks=C×64, prefix cache off, 같은 prompt/cap/sampling/EOS를 사용한다.
- C16/GPU6·C32/GPU7,8 CPU affinity씩, fresh process3회,64 warm+192 measured requests/arm. 세 엔진의 실행 순서를 교대한다. C1·C8 보조 검사는 GPU7에서 별도 fresh process로 수행한다. GPU6에 새 `fresh-demand` PID1001995(1436MiB)가 나타나 기존 보조 검사를 C8 시작 전에 중단했으며, 해당 실패 campaign은 최종 통계에서 제외한다. 외부 프로세스와 기존 holder를 중단하지 않았다. 별도 C32 Nsight profile도 수행한다.
- Stock vLLM0.27.1+cu129의 기본 compile/Graph를 유지한다. Event-spin 패치나 cuBLAS inspection preload를 성능 측정에 사용하지 않는다.
- Raw SSE의 전체 값을 원래 native oracle과 대조하고 p50/p95/p99와 throughput을 재계산한다. TPOT는 response-mean이며 token ITL이 아니다. HTTP 수치는 loopback finite closed-loop workload의 결과다.
- Owned `/v1/config`의503은 기록한다. 실제 argv·startup·binary/source SHA와 kernel trace로 실행 경로를 확인한다.

## 최종 결과

### C16: fresh process3회 중앙값

| 지표 | 기존 mode10 | 개선 v6 | stock vLLM | Riley 변화 |
|---|---:|---:|---:|---:|
| 처리량 tok/s | 392.155 | 514.119 | 1104.710 | +31.10% |
| TTFT p50 ms | 95.654 | 79.371 | 50.040 | -17.02% |
| TTFT p95 ms | 121.298 | 121.772 | 82.415 | +0.39% |
| TTFT p99 ms | 260.691 | 252.312 | 109.514 | -3.21% |
| response-mean TPOT p99 ms | 40.718 | 31.736 | 14.012 | -22.06% |
| E2E p99 ms | 1348.925 | 1035.296 | 477.094 | -23.25% |

개선본 처리량은 같은 측정의 stock vLLM 대비 46.5%다.

| 반복 | TTFT p95 after/before | 처리량 after/before | stock token ID 일치 |
|---:|---:|---:|---:|
| 0 | 1.018721 | 1.306079 | 175/192 |
| 1 | 1.006512 | 1.320168 | 177/192 |
| 2 | 0.935060 | 1.294884 | 173/192 |

### C32: fresh process3회 중앙값

| 지표 | 기존 mode10 | 개선 v6 | stock vLLM | Riley 변화 |
|---|---:|---:|---:|---:|
| 처리량 tok/s | 422.366 | 622.558 | 1685.334 | +47.40% |
| TTFT p50 ms | 145.761 | 98.318 | 52.682 | -32.55% |
| TTFT p95 ms | 426.459 | 392.345 | 163.336 | -8.00% |
| TTFT p99 ms | 612.840 | 538.200 | 216.193 | -12.18% |
| response-mean TPOT p99 ms | 78.337 | 62.916 | 21.583 | -19.69% |
| E2E p99 ms | 2591.716 | 1868.427 | 678.039 | -27.91% |

개선본 처리량은 같은 측정의 stock vLLM 대비 36.9%다.

| 반복 | TTFT p95 after/before | 처리량 after/before | stock token ID 일치 |
|---:|---:|---:|---:|
| 0 | 0.921357 | 1.473167 | 171/192 |
| 1 | 0.913153 | 1.487253 | 170/192 |
| 2 | 0.912817 | 1.473094 | 172/192 |

Riley before/after는 각 반복의 측정192개가 모두 기존 native 기준과 일치했다. vLLM은 모든 arm에서3840 output tokens를 처리했지만 token ID 일치율은 위 표처럼 다르다. 이는 별도로 보고하는 수치 경로 차이이며, 품질평가 점수로 해석하지 않는다.

**C16 p95는 개선을 주장하지 않는다.** 중앙값은121.298→121.772ms로0.474ms(+0.39%) 높아졌다. 반복0·1은 소폭 악화했고 반복2는 개선했다. 측정에서 초기 요청을 빼거나 tail 지표를 바꾸지 않았다. C16의 p95 경계 요청과 초기 prefill 묶음을 별도로 기록한다. 이번 변경은 decode Down 및 singleton Attention에 집중돼 있으며 prefill Down은 그대로 남는다. 따라서 엄격한 모든 분위수 무악화 성능 gate를 통과했다고 주장하지 않는다.

기존 native/HF 정확도 gate는 유지한다. 기본 모드를 전역 승격하거나 성능 수치를 이유로 허용오차를 추가하지 않는다.


## 최종 정확도·회귀·자원 검증

- 고정된 source935개가 최종 EFS snapshot 및 현재 로컬 파일과 모두 일치한다.6개 변경 파일의 layer patch는 현재 소스에서 reverse-check·whitespace 검사를 통과했다.
- 회귀 기본 실행에서517개 통과,40개 ignored였다. EFS 상위 경로 소유권 때문에 실패한 C02 fixture4개만 사용자 승인대로 private `/tmp`에서 같은 테스트 binary로 재실행해 모두 통과했다. 합계521개 통과이며 보안 검증 코드를 완화하지 않았다. 임시 트리 제거도 확인했다.
- C1 eager 및 실제 R1..32의 segments Graph 각각104개 HF full-vocabulary row,208개 BF16 파일이 기존 native와 HF에 바이트 단위로 일치했다. Graph capture/성공 첫 replay는 각각2304회였다.
- 별도의 fused SiLU+bounded MLP Graph 호환성 검사도 R1..32·208개 파일이 모두 일치했다.1152 captures/성공 첫 replay,2448 hits, eager fallback0, stream reset0이며 모든 allocation이 닫혔다.
- 최종 선택한29개 HTTP/profile arm의 완료 SSE 값4971개를 원본 스트림에서 재구성해 검사했다. Quantile·throughput·binary/runner SHA·실제576 budget·startup 선택·quiescent metric도 재검산했다.10개 arm에서 disconnect/retry 후 allocation 및 KV 자원 회복을 검사했다.
- 선택한 측정 구간의 foreign GPU PID는0개였다. 측정 중 own+holder peak는14212MiB로22528MiB 한도 안이었다. GPU6의 외부 점유로 중단한 이전 보조 campaign은 raw로 보존하고 최종 통계에서 제외했다.
- C1·C8 보조 검사는 GPU7,2회·32 warm+96 measured requests/arm이다. 아래 C1의 관측 감소를 숨기거나 전역 성능 개선으로 포장하지 않는다.

| 보조 동시성 | 처리량 before→after tok/s | TTFT p95 before→after ms |
|---:|---:|---:|
| C1 | 88.834→87.007 (-2.06%) | 22.910→23.180 |
| C8 | 314.365→360.867 (+14.79%) | 89.588→86.872 |

C16 후보의3회 p95 interpolation 경계6개는 모두 측정 시작 후0.53~1.09ms에 접수된 초기 요청이었다. 기존 모드는 반복0에서 초기 요청, 반복1·2에서는1.38~4.59초의 refill 요청이 경계였다. 후보의 잔여 p95와 초기 prefill 묶음을 분리해 후속 검증해야 하며, 이 관측만으로0.47ms 차이의 원인을 단정하지 않는다. 모든192개 요청을 headline 지표에 포함했다.

## 실제 C32 profile과 남은 vLLM 격차

일반 HTTP 측정과 별도의64 warm+64 measured request profile이다. GPU7·budget576, 각 arm1280 output tokens이며 Riley 두 arm은 원래 native 결과와 일치한다. Nsight2026의 CUDA/NVTX/node Graph trace를 사용했다. EFS 임시 경로 경고는 보존하고 `/tmp`로 옮기지 않았다. Profile의 span·커널 합을 일반 API 처리량으로 쓰지 않는다. DRAM/L2 hardware counter는 수집하지 않았다.

| GPU kernel 진단 | 기존 mode10 | 개선 v6 | 변화 |
|---|---:|---:|---:|
| kernel count | 293049 | 112193 | -61.7% |
| kernel duration 합 ms | 2863.992 | 2067.877 | -27.8% |
| kernel span ms | 3419.996 | 2354.471 | -31.2% |
| Down ms | 666.512 | 397.089 | -40.4% |
| QK/PV ms | 441.870 | 202.767 | -54.1% |
| KV gather ms | 190.248 | 9.178 | -95.2% |
| softmax ms | 138.355 | 26.256 | -81.0% |
| Gate/Up ms | 505.958 | 508.352 | 거의 그대로 |
| K/V projection ms | 227.292 | 227.893 | 거의 그대로 |

새 Down·paged QK/PV kernel의 실제 실행을 이름별 trace count로 확인했다. Down397.089ms 중 새 native-sliced body201.266ms, reduction4.259ms, 기존 prefill139.888ms, 작은 R의 기존 경로51.675ms였다. 새 chunk QK/PV body 합은158.635ms였으며 전체 QK/PV202.767ms의 나머지는 기존 경로다. After의 Gate/Up은508.35ms(24.6%), Down397.09ms(19.2%), K/V projection227.89ms(11.0%), QK/PV202.77ms(9.8%), O172.45ms(8.3%), norm161.68ms(7.8%)였다. 요청별 산술은 유지하지만 scheduler의 실제 묶음 구성과 iteration 수가 약간 달라질 수 있어, 이 표를 고정 단일 연산의 microbenchmark와 혼동하지 않는다.

Stock vLLM trace는32672 kernels, kernel 합848.220ms, span939.729ms였고 `cudaGraphLaunch`41회가 관측됐다. Stock에는 W0 cuBLAS inspection preload를 넣지 않았으므로 익명 GEMM 시간을 임의로 Gate/Up·Down에 배분하지 않는다. 설치된 Qwen3→Qwen2MLP의 merged Gate/Up·QKV 및 SiluAndMul 경로는 source SHA로 별도 기록한다. 실제 backend/compile/Graph 설정은 저장한 startup log와 trace로 확인한다.

다음 병목은 별도 정확도 qualification이 필요하다.

1. **Gate/Up 및 QKV**: merged projection과 더 큰 row tile을 재검증한다. 이전 N19456 단순 merge가 비트 일치하지 않았다는 이유로 tolerance를 완화하지 않는다.
2. **Prefill·작은 R의 Down**: 현재 R12..32 이득과 분리해 native M128 산술을 유지하는 recipe를 찾는다. C16의 초기 prefill p95를 여기서 다시 확인한다.
3. **긴 context Attention**: T>216은 기존 경로다. T240/304의 PV 반례를 해결하고 독립 검증하기 전까지 범위를 넓히지 않는다.
4. **실행 구조**: 요청 묶음 전체의 fused QK/softmax/PV 및 native Down의 더 큰 row tile/shared layout은 추가 후보로 남긴다. 이전의 단일-request fusion 측정만으로 묶음 fusion의 성능을 단정하지 않는다.  Riley는 선택한 Graph-off recipe에서 여전히 많은 launch를 수행한다. Stock Graph/compile 및 fused Attention과의 차이는 남지만, source와 trace만으로 전체 격차를 특정 기능 하나에 귀속하지 않는다. CUDA12.4와 stock cu129 stack 차이도 이 실험에서 별도로 분리하지 않았다.

## 증거 위치

- 연구·실패 원본: `/mnt/efs/nlp/psyche/riley-down-qkpv-v6-20261020`
- 최종 source/build/검증: `/mnt/efs/nlp/psyche/riley-down-paged-v6-20261020`
- 이전 mode10 결과: [C16 tail·Attention 보고서](qwen-c16-tail-and-attention-20261019.md)


## 완료 시점 홀드와 검증 산출물

기존 GPU6 PID2027914/start ticks6083215119, GPU7 PID2027915/start ticks6083215121이 모두 살아 있고 각각 allocation/reserved20GiB, phase idle인 것을 확인했다. Context/PID를 종료·재시작하거나 홀드를 해제하지 않았다. 증거 봉인 snapshot에서 GPU6에는 별도1436MiB 프로세스2개가 공존했고 GPU7에는 기존 holder만 있었다. 외부 프로세스는 중단하지 않았다. 전달 직전 live 재검사에서는 두 GPU 모두 외부 점유0개, 기존 PID·start ticks·idle20GiB가 유지된 것을 다시 확인했다. 홀드를 GPU 전체의 독점 소유라고 주장하지 않는다.

- [전체 SSE·성능·자원 감사](../benchmarks/evidence/20261020-down-qkpv/final/independent-audit.json)
- [전체 vocab 및 Graph 감사](../benchmarks/evidence/20261020-down-qkpv/final/teacher-audit.json)
- [선택적 MLP Graph 호환성 감사](../benchmarks/evidence/20261020-down-qkpv/final/mlp-compatibility-audit.json)
- [C32 profile 집계](../benchmarks/evidence/20261020-down-qkpv/final/profile-summary.json)
- [C16 p95 경계 요청 감사](../benchmarks/evidence/20261020-down-qkpv/final/c16-tail-audit.json)
- [완료 홀드 receipt](../benchmarks/evidence/20261020-down-qkpv/final/final-hold-receipt.json)
- [6파일 변경 patch 및 적용 기준](../benchmarks/candidates/sm80-down-paged-v6/README.md)
- [압축 증거](../artifacts/down-qkpv-20261020/compact-evidence.zip):2,031,022bytes, SHA256 `a6598a94a4fe1bc414110a5ff3b8a2818199af41523de62517b2f2b1f4e99afd`.

압축파일의 hashlisted656개 파일과 archive SHA를 로컬에서 재검증했다. 원본 SSE·BF16·Nsight·binary·weights는 EFS에 유지한다. 별도 raw index2229개 파일의 크기·SHA를 포함하며, 실패한 실험과 중단한 보조 검사도 보존한다. 최종 결과는 완료한 선택 campaign만 사용했다.

후속: [C16 p95 및 prefill Gate/Up·Down v7 검증](qwen-c16-prefill-mlp-v7-20261020.md).
