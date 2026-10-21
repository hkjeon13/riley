# Qwen prefill attention·host 실행 구조 v15B

2026-10-21 KST. prefill 구조 변경과 host resource index·embedding report 완료 경계를 합친 후보를 독립 빌드하고 C16/C32에서 v13B·stock vLLM과 각각5회 비교했다. 처리량 중앙값은 **+3.06%, +5.45%**였다. 큰 격차를 해소하거나 계획의 첫 묶음 +10% 목표를 달성한 결과는 아니다.

**이 checkpoint에서 v15B는 기본 코드에 채택하지 않는다.** C32 반복2의 TPOT p99+2.40% 플래그를 보존한다. 또한 일반 BF16 query/output span이2-byte offset일 때 기존 재배치 경로를 유지하는 정렬 조건을 최종 후보에서 보완한다. 측정된 서버 내부 span은 정렬돼 있었지만 그 결과를 범용 span 전체의 보장으로 확대하지 않는다. 기본 코드는 검증된 v13B이며 [미적용 재현 패치](../benchmarks/candidates/sm80-prefill-host-v15b/prototype.patch)를 보존한다.

## 구현

- prefill M>1/T<=2048/QH32/KVH8/D128에서 compact GQA K/V staging, row-major Q/context pointer-batched GEMM, native BF16 경계의 causal softmax 융합을 사용한다. Q/K/V/output의 수치와 mask·softmax reduction 순서를 유지한다. [prefill 단독 v15A](qwen-prefill-host-v15a-20261021.md)에서156 구성요소 조합과 full-vocabulary gate를 먼저 검사했다.
- command-batch resource 등록의 반복 선형 검색을2048-slot 고정 포인터 index로 바꾼다. 원래1024-entry ledger는 acquisition/release의 권위를 유지한다. owner/thread 검사, capacity와 실패 후 lease 유지 규칙은 보존하며, 확인된 완료와 모든 resource release 뒤에만 index를 비운다. 추가 비용은 stream마다 cold host16KiB이며 hot allocation이 없다.
- mode10 staged lane에서 CPU token admission을 유지하고 embedding device report를 iteration 완료 시 pinned staging에서 확인한다. GPU 제출 중간의 host report 대기를 제거한다. report/실행 실패는 출력 publish와 scheduler commit 전에 전파한다. EOS·취소·KV reservation의 commit 시점은 바꾸지 않는다.
- R2~16 Gate/Up은 기존 native fused kernel이다. v14의 all-R merged Gate/Up, 두 physical mixed passes, 선택적 decode softmax/PV 후보를 다시 합치지 않았다.

CPU 실제 native 등록 함수 진단(2백만 중복 등록,1024 resources)은296.070→7.777ns/호출이었다. foreign thread/stream, capacity 초과 시 미획득, 중복 lease 및 index clear/reuse4세대도 검사했다. 이 값은 서버 개선율이 아니다.

## 정상5회 비교

물리 GPU7에서 before/after/stock 순서를 교대하며 매번 새 서버를 시작했다. Qwen3-4B BF16, GPU greedy, budget576, max sequence1024, KV blocks=C×64,64 warm/192 keep,12 fixed prompts115~130 tokens와8/16/24/32 출력 상한이다. 각 arm은 실제3,840 output tokens다. 정상30 arms에는 profiler·inspection preload·host trace가 없다. 지표별 반복 중앙값이며 percentile을 pooling하지 않는다.

| 지표 | C16 v13B | C16 v15B | C16 stock | C32 v13B | C32 v15B | C32 stock |
|---|---:|---:|---:|---:|---:|---:|
| output tok/s | 719.189 | 741.221 | 1125.205 | 933.316 | 984.202 | 1671.802 |
| TTFT p95 ms | 88.291 | 82.279 | 83.717 | 272.284 | 245.556 | 165.122 |
| TTFT p99 ms | 176.772 | 164.013 | 110.692 | 366.823 | 333.062 | 218.655 |
| TPOT p95 ms | 20.867 | 20.269 | 12.758 | 33.471 | 31.645 | 18.123 |
| TPOT p99 ms | 22.400 | 21.712 | 13.406 | 40.443 | 37.692 | 21.643 |
| E2E p95 ms | 691.977 | 670.040 | 427.521 | 1061.863 | 1010.308 | 559.312 |

stock 대비 처리량은 C16 **63.92%→65.87%**, C32 **55.83%→58.87%**다. stock은0.27.1+cu129/Torch2.13.0+cu129의 기본 FA2/FULL_AND_PIECEWISE, prefix cache off이며 실제11개 startup log를 검사했다. stock 수치 경로를 Riley의 정확도 기준으로 사용하지 않는다.

TTFT p95 중앙값은 C16 -6.81%, C32 -9.82%, TPOT p99 중앙값은 -3.07%, -6.80%였다. C16의 반복별2% 초과 tail 플래그는0이다. C32 반복2 TPOT p99는38.144861→39.059459ms(+2.3977%)였으므로 모든 pair 비악화라는 판정을 하지 않는다.

## tail 표본 확인

C32 반복2의 p99 보간 index는189.09다. before rank189/190은 `logic` ordinal8/`summary` ordinal0, after는 `comparison` ordinal4/`logic` ordinal8이었다. 같은 `logic`의 TTFT는135.157→38.300ms, E2E는399.454→336.009ms였지만 요청 평균 TPOT는37.744→42.370ms였다. `summary`의 E2E도388.337→337.729ms였다.

TPOT는 첫·마지막 토큰 사이의 요청별 평균이며 token ITL percentile과 다르다. 첫 토큰 시점, 전체 완료 시점과 percentile bracket이 함께 달라진 관측이다. 같은 ordinal의 서버 completion ID도 달라졌으며 처리 순서가 같았다고 가정하지 않는다. 이 자료만으로 단일 kernel 저하나 TTFT 변화만을 인과 원인으로 확정하지 않는다. 플래그를 제거하지 않고 최종 후보의 추가 paired 검증으로 확인한다. 정렬 가드는 호환성 보완이며 이 tail의 해결 원인으로 주장하지 않는다.

## 실제 prefill 경로와 host 진단

정상 측정과 분리한192-keep Nsight node trace다. 모든 CUDA API 수집을 명시했다. 이 Nsight 버전에서는 driver 호출도 `CUPTI_ACTIVITY_KIND_RUNTIME`에 함께 저장돼 별도 DRIVER table의 유무로 호출 수집 여부를 판단할 수 없다. Graph node 순서와 source의 layer/residual/PV/SiLU 경계를 함께 분류한다.

| 별도 진단 | C16 v13B→v15B | C32 v13B→v15B | C32 stock |
|---|---:|---:|---:|
| GPU kernel 수 | 291306→259301 | 213753→184813 | 70687 |
| GPU kernel 시간 합 ms | 4939.702→4763.737 | 3844.251→3679.475 | 2211.901 |
| kernel union/span | 81.82%→84.15% | 73.19%→78.23% | 81.74% |

C16·C32 after 각각 compact prefill gather6,912회와 pointer-batched GEMM13,824개 NVTX ranges를 확인했다. 즉192 요청×36 layers에서 두 GEMM이 실행됐다. 기존 Q gather/output scatter/scale-mask tile/BF16 scatter는 after trace에서0이었다. before에는 compact gather와 pointer-batched ranges가0이었다. 실제 M/N/K와 leading dimensions를 증거에 보존한다.

C16 `cudaLaunchKernel` 호출은152,679→120,167, `cudaStreamSynchronize`는810→540이었다. API duration은 GPU 실행 및 queue 대기와 겹치므로 더하거나 순수 CPU 연산 시간으로 해석하지 않는다. 같은 workload라도 실제 batch composition이 달라질 수 있어 전체 kernel 감소를 한 연산에 모두 귀속하지 않는다.

C16 after의 `cuCtxGetCurrent`는487,675회/30.193ms였고 before의 context pop은102,430회/8.870ms였다. 이 진단에서 context push/pop 생략은 큰 개선의 우선 근거가 아니므로 generic context 복원 계약을 바꾸는 추가 구조를 도입하지 않는다. 이전 baseline 재분류에서는 Gate/Up+Down이 GPU 시간 합계의 약41~43%였고, 이번에도 더 큰 GPU 작업이 남는다. DRAM/L2/SM hardware counter 없이 memory-bound 판정을 하지 않는다.

## 정확도·자원과 실패 기록

독립 source/빈 target 빌드에서 ordinary CPU520개와 기존 승인된 private `/tmp` C02 fixture4개, 합계 **524 pass/40 ignored**다. 최초 aggregate command의 EFS 소유권 실패4개와 재검사를 보존하며 임시 트리는 제거했다.

원래 native/HF의 literal full-vocabulary BF16 gate를 유지했다. 기본 검사2,464 files(104 C1+104 all-R shrink+1,024 natural128 R1~32 rows), mixed 두 종류2,016 files(1,024+992)가 일치했다. 역순 input/output slots, prefill output slot 없음, 잘못된 prefill token의 GPU 제출 전 거부, owner/shape/stat 재사용과 hot CUDA allocation 불변을 검사했다. R1~32 row 검사와 body Graph key2~32는 구분한다. positive early-EOS 완료를 새로 검증한 것은 아니다.

초기 finite observer는 process별 메모리 합만 저장해 NVML이0으로 보고하는 모델 process의 장치 전체 cap을 입증하지 못했다. 해당 자원 감사는 완료로 처리하지 않았다. 동일 binary로 새 경로의5개 검사를 반복하고 원래 native/HF bytes와 장치 전체22GiB를 다시 확인했다. 첫 fast component observer의 exit/PID 관측 race도 실패로 보존하고, 실행 파일·UID·환경 marker·start ticks·thread group을 살아 있을 때 먼저 묶은 observer로 재검사했다. 이미 확인한 PID가 사라진 NVML 잔여 row는 GPU 메모리 합에는 계속 포함하며 완전한 teardown을 기다린다. 미확인 live process를 own으로 간주하거나 종료하지 않는다.

초기 추가 GPU integration test build는 오래된4-argument `execute_row_groups` 호출 때문에 실패했다. B에서 해당 검사가 통과했다고 주장하지 않는다. 최종 후보에서 현재 API의 grouping selector0을 명시하고 별도 실제 GPU 계약 검사로 확인한다. B의 정상 성능·native byte gate와 이 추가 test build 실패는 구분한다.

정상·진단35 arms의 저장된 **7,239 full SSE**를 native/self-SSE·usage·종료·자원 회수로 독립 감사했다. 장치 전체 최고는 **18,267MiB**였다. GPU6의 승인된 작은 외부 작업은 보존하고 정상 비교는 모두 깨끗한 GPU7에서 했다. 원래 holder PID2027914/2027915, ticks6083215119/6083215121은 유지했다. 최종 idle20GiB 검사는 후속 작업 종료 때 별도로 확인한다.

## 재현

- binary SHA256: `7d9e89a57f6d8ed248a853ba693b3a77e502b941ee03d16ddd9b3d454b6662a2`
- qualified source936: v13B935 files와 새 host index header. 재현 patch는 기본 코드에 적용하지 않는다.
- EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15b-20261021`.
- [압축 증거](../benchmarks/evidence/20261021-prefill-host-v15b.zip), [감사·통계 index](../benchmarks/evidence/20261021-prefill-host-v15b/README.md).
- 범용 span의 정렬 보완 근거: [cuBLAS12.4 pointer-batched BF16 alignment](https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html#cublasgemmbatchedex).

중간 source/runner snapshot과 이후 생성한 진단 스크립트는 구분한다. 정상30 arms는 각 plan의 실제 binary/runner hash를 대조하고, 완료 후 source936과 binary가 동일함을 다시 검사했다. 원본 BF16·Nsight·ELF는 EFS의 heavy SHA index로 연결한다.
