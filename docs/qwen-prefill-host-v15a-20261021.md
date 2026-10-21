# Qwen prefill attention 구조 실험 v15A

2026-10-21 KST. Git 기준선은 `40087684`의 검증된 v13B다. 이 단계는 **prefill 단독 구조 후보의 정확도 검사와 선별 비교**이며, 기본 코드 채택 판정이나 5회 vLLM 재비교 결과가 아니다. [미적용 재현 패치](../benchmarks/candidates/sm80-prefill-v15a/prototype.patch)를 보존하고 host 구조 변경을 합친 다음 후보를 별도로 검증한다.

## 이전 실험의 한계와 재확인한 병목

v14D는 일부 decode softmax/PV와 R2~16 Gate/Up 경로만 바꿨다. 새 softmax/PV kernel은 D C32 trace의 GPU kernel 시간 합계 중 2.63%였고, hot-buffer 구성요소 이득은 cache-displacement/이종 context 진단에서 유지되지 않았다. R2~16에서는 기존 fused Gate/Up 1 kernel이 Lt+SiLU 2 kernels로 바뀌었다. 동기화 수 감소도 전체 GPU 작업 감소와 같지 않았다. [원래의 부정적 결과](qwen-mixed-attention-structural-v14-20261021.md)를 유지한다.

기존 **v13B baseline** trace를 Graph node 단위로 다시 분류했다. 첫 실행의 node 순서, 36 layers의 72 residual 경계, QKV pack·PV·SiLU 경계와 source 실행 순서를 함께 사용한다. kernel 이름이나 grid만으로 GEMM의 역할을 추정하지 않는다. node별 근거를 JSON으로 보존한다. 이 trace에는 별도 driver API table이 없으므로 아래 재분류 이득은 Graph node 매핑에서 나온다. 후속 진단에는 모든 CUDA API 수집을 명시해 driver 호출 비용을 따로 확인한다.

| GPU kernel 시간 합계, ms | C16 baseline | C32 baseline |
|---|---:|---:|
| 전체 | 1712.785 | 1396.635 |
| Gate/Up | 481.752 | 369.208 |
| Down | 255.074 | 202.785 |
| K/V projection | 191.250 | 171.100 |
| QK/PV | 158.204 | 137.248 |
| O projection | 139.282 | 107.074 |
| 재분류 전 other | 407.503 | 375.545 |
| 재분류 후 other | 96.909 | 89.720 |

Gate/Up+Down은 각각 약43.0%, 41.0%다. Q/K/V/O의 Graph 내부 일부는 QKV aggregate로 따로 남기며 중복 합산하지 않는다. 위 수치는 각 1회의 **기존 profiler 진단**이다. 정상 HTTP 처리량이나 prefill 전용 비율이 아니다. GPU kernel union/span은 각각81.28%, 71.02%였으며 빈 구간 전체를 CPU 계산 시간이라고 부르지 않는다. API wait와 GPU 실행은 겹친다.

서버에는 `ncu`가 PATH에 없고 드라이버 `RmProfilingAdminOnly: 1`이다. DRAM/L2/SM hardware counter를 확보했다는 주장이나 memory-bound 판정은 하지 않는다.

## prefill 실행 구조 변경

SM80으로 cold-qualified된 Qwen D128/QH32/KVH8 경로의 `M>1`, `T<=2048`에 한정한다. 다른 hardware/geometry와 긴 context는 기존 경로를 유지한다.

- K/V를 query-head 수로 4배 확장하는 대신 KV-head 수 그대로 모은다. 기존 K scratch의 남는 영역에 6×32개의 device pointer table을 둔다. 새 device allocation이 없다.
- Q와 context는 처음부터 row-major buffer를 사용한다. pointer-batched GEMM의 leading dimension으로 head 간격을 표현해 별도 Q gather와 output scatter를 없앤다. 각 head의 원래 M/N/K는 유지하고 padding·head 간 누적 병합은 하지 않는다.
- BF16 QK → BF16 scale → 기존 FP32 warp softmax reduction → BF16 probability 경계는 유지한다. causal mask, scale, softmax, BF16 cast를 한 kernel에 모아 FP32 중간 buffer 왕복과 타일별 launch를 제거한다. online softmax나 FlashAttention의 다른 누적 순서를 도입하지 않는다.
- 기존 span/alias/owner 검사와 command-batch lease, completion 실패 처리, scheduler의 EOS·취소·KV commit 계약을 유지한다.

기존 prefill은 request/layer마다 Q gather, expanded K/V gather, QK, 128-row 타일별 scale/mask·softmax·cast, PV, output scatter를 제출했다. 새 대상 경로는 compact gather+pointer setup, QK, fused softmax, PV의 **4개 제출 단계**다. cuBLAS 내부 kernel 수가 항상4개라는 뜻은 아니다.

## 정확도와 자원

- 독립 source copy/빈 target에서 CUDA12.4·SM80 빌드. 일반 CPU 회귀520개와 기존에 허용된 private `/tmp` C02 fixture4개를 합쳐 **524 pass /40 ignored**. 최초 aggregate command의 EFS 소유권 실패4개와 재검사 기록을 모두 보존한다. 첫 자동 재검사 스크립트의 허용 flag 누락도 보존하고 올바른 flag로 다시 실행했다.
- 원래 native 및 HF와 full-vocabulary BF16 **2,464 files** 일치: C1 eager104 rows, all-R shrink Graph104 rows, natural128 R1~32의1,024 rows. 서로 다른 검사 범위를 Cartesian coverage로 합치지 않는다.
- 구성요소에서는26개 `(M,T)`와6종 finite 입력의 **156조합**을 검사했다. M2~2048, T2~2048, 비정렬/128·256·512 경계와 비연속 page 순서를 포함하며 QK·probability·최종 context의 바이트 불일치는 모두0이었다. full-model gate를 대체하지 않는다.
- 저장된 baseline 진단 및 선별5 arms의 **816 full SSE**를 독립 검증했다. 원래 native IDs/text/finish/EOS 유무/usage/prompt IDs, disconnect recovery, KV/active/waiting 회수와22GiB 제한을 확인했다. 이 fixture의 finish는 length이며 positive early-EOS 검사를 새로 입증한 것은 아니다.
- 원래 GPU6 PID2027914/ticks6083215119, GPU7 PID2027915/ticks6083215121을 유지한다. idle20GiB/active64MiB로 전환할 때 holder context를 종료·재시작하지 않는다. GPU6의 허용된 작은 외부 작업은 보존한다. 정상 비교는 깨끗한 물리 GPU7에서 순차 수행했다.

## 단독 후보 선별 결과

| 선별 | v13B→v15A output tok/s | 변화 | TTFT p95 ms | TPOT p99 ms |
|---|---:|---:|---:|---:|
| C16, 64 warm /64 keep, 1 pair | 671.024→703.088 | +4.78% | 170.130→157.124 | 22.085→22.611 |
| C32, 64 warm /192 keep, 1 pair, host trace on | 940.562→989.527 | +5.21% | 271.242→245.829 | 39.865→38.014 |

C16 TPOT p99는 약2.38% 악화했다. 실행 길이와 진단 설정이 서로 다르므로 두 선별 결과를 하나의 정상 캠페인이나 5회 반복 결과로 합치지 않는다. vLLM 비율도 계산하지 않는다.

C32 host trace의 두 번째128-step 구간에서 submit wall time은9647.6→7015.5µs/step, completion wait는17781.5→19188.3µs/step이었다. GPU와 겹치는 시간이며 첫 번째 구간에는 correctness/warm/idle 경계도 포함된다. submit 감소를 같은 크기의 CPU 절감이나 처리량 증가로 주장하지 않는다.

다음 후보는 여기에 고정 크기 resource index와 embedding report의 completion 경계 검증을 합친다. 기존 ledger는 등록 때마다 최대1,024개 포인터를 선형 검색했다. CPU의 실제 등록 함수 진단에서는2백만 번 중복 등록이296.070→7.777ns/호출로 줄었지만, 이것은 서버 개선율이 아니다. foreign thread/stream, capacity 초과 시 미획득, 중복 lease와 index 재사용도 별도 검사했다.

## 재현과 후속 판정

EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15a-20261021`. 원본 BF16와 실행 ELF는 SHA256 heavy index로 연결한다. source935·binary·runner 해시와 정상 선별 원문은 [증거 index](../benchmarks/evidence/20261021-prefill-host-v15a/README.md)에 보존한다. 기본 코드 채택과 README의 최신 정상 성능표는 후속 후보의 C16/C32 5회 before/after/stock 결과, 기존 정확도와 tail gate로 결정한다.

참고: [cuBLAS pointer-batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex), [vLLM full/piecewise Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/). 설치된 cuBLAS는12.4.5이며 최신 문서의 새 API를 사용한 것은 아니다.
