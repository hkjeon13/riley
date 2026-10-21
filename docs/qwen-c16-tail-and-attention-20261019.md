# Qwen3-4B C16 tail 회복과 SM80 인퍼런스 개선

후속 Down·QK/PV 개선 및 stock vLLM 재검증: [2026-10-20 보고서](qwen-down-qkpv-and-vllm-20261020.md).

2026-10-19~20 KST. aws-a100의 물리 GPU6·7에서 기존 HF/native BF16 exactness를 유지했다. 이전 mode8은 [앞선 C16/C32 보고서](qwen-c16-c32-expansion-20261019.md)를 참고한다.

## 최종 결과

**선택한 조합은 mode10 `native-prefill-sm80-v5` + GPU greedy + token budget576, Graph off / separate SiLU다.** C16·C32 모두 fresh-process 3회에서 TTFT·응답 평균 TPOT·E2E의 p50/p95/p99가 매 반복 개선됐다. 첫 요청 파형을 제거하지 않고 전체192요청으로 계산했다.

| 항목 | 기존 mode8 | 최종 mode10 | 변화 |
|---|---:|---:|---:|
| C16 GPU6 throughput tok/s | 342.73 | 389.29 | +13.6% |
| C16 TTFT p95 ms | 140.20 | 121.37 | -13.4% |
| C16 TTFT p99 ms | 286.47 | 260.97 | -8.9% |
| C16 응답 평균 TPOT p99 ms | 46.33 | 40.72 | -12.1% |
| C16 E2E p99 ms | 1525.94 | 1351.92 | -11.4% |
| C32 GPU7 throughput tok/s | 366.01 | 422.27 | +15.4% |
| C32 TTFT p95 ms | 497.35 | 425.08 | -14.5% |
| C32 TTFT p99 ms | 744.30 | 613.94 | -17.5% |
| C32 응답 평균 TPOT p99 ms | 87.01 | 78.64 | -9.6% |
| C32 E2E p99 ms | 3115.33 | 2612.87 | -16.1% |

중앙값은 반복별 지표의 중앙값이다. C16의 paired p95 비율은 0.8655/0.8613/0.8844, C32는 0.8589/0.8541/0.8481이었다. 응답 평균 TPOT p99는 응답별 평균 TPOT의 p99이며 개별 token interval의 p99와 구분한다.

비교 조건은 Qwen3-4B BF16, C16 CPU24-31 / C32 CPU32-39, 각64 warm +192 측정, 출력합3840토큰/arm, max sequence1024, KV block16·개수C×64, greedy·중립 penalty·원래 EOS, prefix cache off다. AB/BA 순서로 반복했다. **기준선은 budget512, 후보는576**이므로 최종 표는 코드와 스케줄러 설정을 합친 개선이다. 같은 budget576인 별도 프로파일로 Gate/Up 효과를 분리했다. 이전 GPU7의5회 결과, 64요청 profile, 과거 vLLM 결과와 이 표를 합치지 않았다.

추가 C1/C8 검사는 GPU6에서 각32 warm +96 측정, 출력합1920토큰/arm, AB/BA 2회다.

| 동시성 | 기존→최종 tok/s | 기존→최종 TTFT p95 ms |
|---|---:|---:|
| C1 | 87.18→88.40 (+1.4%) | 23.019→23.011 |
| C8 | 286.65→313.61 (+9.4%) | 93.07→89.94 |

C8은 두 반복의9개 latency quantile이 모두 개선됐다. C1은 r0 TTFT p50 +0.071%, r1 p95 +0.022%·p99 +1.269%가 남았다. 따라서 **전체 동시성·모든 반복의 성능 무회귀를 주장하지 않고 기본값 승격도 하지 않는다.** C16/C32용 명시적 최적화 조합으로 제공한다.

## C16 p95 악화 원인

이전 mode7→8의192요청 원본을 다시 정렬했다. 첫16요청 파형이 상위20개 중8~12개를 차지했고, 최초 파형은 약140/213/285/330ms의 계단을 만들었다. 첫 파형 이후176요청의 p95는 r0 130.30→120.18ms, r1 133.74→122.86ms, r2 128.17→129.88ms였다. 이 분리는 진단용이며 채택 기준은 전체 p95다.

스케줄러는 decode 행을 먼저 예산에서 빼고 owned HF prefill은 요청 전체를 선택한다. 프롬프트 네 개가512에 가까우면 decode 몇 행 때문에 네 번째 요청이 다음 iteration으로 밀린다. 빨라진 decode가 후속 HTTP 유입·종료 구성을 바꾸면서 p95의 순위 경계도 달라졌다. 여기에 프리필 Gate와 Up이 같은 weight를 요청마다 별도 GEMM으로 처리하는 비용이 남아 있었다. 이번 수정은 단순히 표본 수나 quantile 계산을 바꾸는 대신 해당 프리필 비용을 줄였다.

Budget1024 대조는 p95를 악화해 제외했다. 이 고정 workload의 프롬프트는115~130토큰이므로576이면32 decode + 네 개의130토큰 prompt까지 들어간다. 다른 prompt 분포에서도576이 최적이라고 일반화하지 않는다.

## 구현과 수치 범위

### Mode9: KV gather와 softmax load

- 16바이트 BF16 bit copy 한 번으로 같은 GQA 그룹의 네 query head에 쓴다. Expanded K/V scratch 및 QK/PV cuBLAS의 batch count·크기·stride를 유지한다. Invalid page는 기존 NaN bit pattern을 유지한다.
- Singleton Q32/KV8/D128, T≤2048에서 scale의 BF16 반올림을 softmax 입력 로드에 합친다. QK/PV 연산 순서와 BF16 score/probability 경계는 유지한다.
- T=1..2048 ×3상태에서18,432개 비교, gather51,564,773,376개 BF16 값과 softmax201,424,896개 값의 bit 차이가0이었다. 무작위 BF16 bit pattern·zero/변형 score·invalid page를 포함한다.

### Mode10: 별도 Gate/Up의 요청 행 배치

실제 SM80 weight/input에서 M64..576을 native M128 산술과 대조했다. Gate와 Up 각각1539개 비교·4,790,845,440개 BF16 값에서 **M64..130 및 M161..576만** native M128과 같았다. M131..160은 달라 제외했다.

Mode10은 각각의 Gate와 Up weight 및 N9728/K2560을 유지한다. 인접한 **원래 요청 M64..130**을 **합계 M161..576**으로 묶는다. 원래 요청 경계는 Attention/KV/출력 메타데이터에 남는다. 허용 합계가 없거나 다른 크기·decode 행을 만나면 기존 요청별 GEMM을 사용한다. 추가 row copy나 scratch buffer가 필요하지 않다. 이는 N19456으로 gate|up weight를 합쳤다가 실패한 후보와 다른 변환이다.

Decode의 Q≤32, Gate/Up≤16, O≤8, Down≤4, LM head≤16, K/V native-order reuse 규칙은 유지했다. Down은 native M128과 같은 구간이 M64..163에 한정됐고 큰 묶음은 달랐다(전체 비교340,278,138개 BF16 차이). **Down 프리필을 합치거나 허용오차로 통과시키지 않았다.**

SM80/108SM, CUDA12.4/runtime12040, cuBLAS/Lt120405, Qwen3-4B `(2560,32,8,128,9728)`, 최대32요청, unmodified workspace/MPS 환경 guard를 유지한다. 다른 모델·GPU에 이 수치 증거를 확대하지 않는다. 기존 mode0..9의 산술과 기본 선택값은 유지한다.

### 별도 MLP Graph 후보

기존 M1 MLP Graph를 그대로 켜서 배치를 잃는 대신, mode9/10에서 Gate/Up≤16·Down≤4 exact 레시피를 capture/replay하도록 구현했다. 실제 행 수·stream·8개 operand의 allocation identity/offset/extent를 key로 쓰고, Down의 private Lt workspace도 replay resource ledger에 등록한다. Graph를 먼저 닫고 plan을 닫으며, layer segment와 MLP Graph의 nested capture를 막는다.

같은 budget576·fused SiLU의 profile에서는 CUDA runtime kernel launch가259641→244090으로 줄었고 Graph launch1332회로 대체됐다. GPU span은3367.76→3316.91ms(-1.5%)였다. Graph kernel의 원래 cuBLAS NVTX attribution이 없을 수 있으므로 그 시간을 임의로 Down/Gate-Up에 배분하지 않는다.

이 조합의 비계측 C16 대조는 세 반복 중 두 번 p95가 약0.7~0.9ms 악화했다. **최종 C16 조합에는 MLP Graph/fused SiLU를 켜지 않았다.** 수치 검증은 통과한 선택 옵션으로 남긴다.

## 실제 비용 변화와 남은 병목

모두 GPU7 C32,64 warm +**64 측정** 요청, 출력합1280토큰의 Nsight 진단이다. cuBLAS NVTX/CUDA timeline에 근거하며 DRAM/L2 hardware counter를 확보한 것으로 표현하지 않는다. 위192요청 HTTP 성능 숫자와 합치지 않는다.

| Attention 분리 대조 | mode8 | mode9 |
|---|---:|---:|
| kernel count | 340073 | 296225 |
| kernel sum ms | 3336.70 | 2954.70 |
| GPU kernel span ms | 3820.70 | 3404.67 |
| KV gather ms | 460.02 | 190.52 (-58.6%) |
| scale/mask ms | 126.60 | 16.46 (-87.0%) |

같은 budget576인 mode9→10 대조에서는 Gate/Up kernel 시간이 **583.60→516.43ms(-11.5%)**, 해당 kernel 수11880→9576이었다. 실제 NVTX correlation에서 N9728/K2560의 M245/246/250/256/258/373/491/499/504/506/511/512를 확인했다. 따라서 단순히 모드 이름만 선택된 것이 아니라 요청 간 행 배치가 실행됐다. 이 count는 kernel 수이며 cuBLAS 함수 호출 수와 구분한다.

최종 profile의 kernel sum은2882.13ms, span3344.57ms였다. 주요 잔여 비용은 다음과 같다.

| 잔여 연산 | kernel ms | 전체 kernel sum 비율 |
|---|---:|---:|
| Down | 668.23 | 23.2% |
| Gate/Up | 516.43 | 17.9% |
| QK/PV | 441.97 | 15.3% |
| KV gather | 190.86 | 6.6% |

QK/PV 본체와 Down kernel은 그대로 남는다. Down stage17/tile18 exact 대안은16행 chunk 합 기준183.1us보다185.3/195.6us로 느려 제외했다. Stage17은 종료 관측 receipt도 실패해 채택 증거로 쓰지 않는다. Tile18은 정상 관측·bit 차이0이었지만 느렸다. N19456 merged gate|up 후보는 BF16 값20365개가 달라 제외했다. 실패 원본과 raw counterexample을 보존했다.

과거 stock vLLM과의 차이가 닫혔다고 주장하지 않는다. 이번 turn에서 stock vLLM을 다시 측정하지 않았으며 과거 stock 결과는 앞선 보고서에 별도로 남긴다.

## 정확도와 자원 검증

- 최종 source 독립 빌드: 회귀 **521개 통과**(EFS517 + 이미 허용된 네 C02 fixture만 private `/tmp` 재검사), 기존 ignore40. 원래 EFS C02 실패/cargo exit101과 rerun 로그를 함께 보존하며 보안 로직을 완화하지 않았다. 네 fixture tree는 검사 후 삭제했다.
- 전체 모델: HF full-vocabulary104행, 실행별208개 BF16 파일이 원래 native 및 HF와 byte-exact. Eager와 layer segment Graph 실제R1..32에서 동일했고 Graph capture/첫 replay 각각2304개를 확인했다.
- Mode10 + 선택적 MLP Graph 조합도 별도104행/208파일 검사를 통과했다. 실제R1..32에서 capture/첫 replay1152개, cache hit2448회, fallback/stream reset0, 모든 allocation closure를 확인했다.
- 최종 HTTP22 arms에서 저장된3620개 전체 SSE 값을 원래 native oracle과 독립 재대조했다. Prompt IDs·출력 IDs·text·finish reason·usage·`[DONE]`을 검증했고 EOS/출력 cap/penalty를 바꾸지 않았다. 처리량 및9개 quantile을 원본 timing에서 재계산했다.
- Metrics의 active/waiting/KV allocation0, disconnect/retry의 allocation 복구, UID/marker/PID start tick과 GPU own+holder22GiB 상한을 검사했다. Owned `/v1/config`는503 `config_unavailable`이므로 성공한 config 조회로 표현하지 않는다. 실제 argv·startup mode·binary SHA·kernel 경로로 선택을 검증했다.
- Attention 단계28 arms/5696개 값, MLP 단계22 arms/8808개 값도 별도 audit했다. 세 단계 합계72 arms/18124개 저장 SSE 값이다. 서로 다른 workload/window의 성능 지표를 합산하지 않는다.

## 재현과 증거

최종 binary: `/mnt/efs/nlp/psyche/riley-sm80-prefill-v5-20261019/riley-sm80-prefill-v5`

SHA256: `b704f82694a32edc791cdf035c6bb203c7cf3b79932c51ea713d32545821bab5`

최종 source935파일은 해당 root의 `prefill-source-manifest.json` 및 로컬 checkout과 일치한다. [candidate patch](../benchmarks/candidates/sm80-native-prefill-v5/candidate.patch)는 frozen mode9 위7파일 layer이며, [mode8부터의 combined patch](../benchmarks/candidates/sm80-native-prefill-v5/combined-mode8-to-mode10.patch)는9파일 layer다. 현재 checkout에서 두 patch 모두 reverse-check한다. Commit/push는 하지 않았다.

실제 전체 argv는 [C16](../benchmarks/evidence/20261019-c16-tail-v5/prefill/prefill-c16.argv.json), [C32](../benchmarks/evidence/20261019-c16-tail-v5/prefill/prefill-c32.argv.json)에 보관한다. 새 output 경로로 재현하려면 원격에서 다음처럼 실행한다. 기존 GPU holder controller와 resource monitor가 그대로 적용된다.

```sh
ssh aws-a100 python3 - <<'PY'
from pathlib import Path
import json, subprocess, time
root=Path('/mnt/efs/nlp/psyche/riley-sm80-prefill-v5-20261019')
argv=json.loads((root/'prefill-c16.argv.json').read_text())
argv[argv.index('--out')+1]=str(root/'runs'/('reproduce-c16-'+str(time.time_ns())))
subprocess.run(argv,check=True)
PY
```

원격 raw/source/binary는 다음 root에 보존한다. 모델·벤치마크·script 및 컴파일 scratch는 EFS를 사용했다.

- Attention: `/mnt/efs/nlp/psyche/riley-c16-tail-v4-20261019`
- MLP Graph: `/mnt/efs/nlp/psyche/riley-sm80-mlp-graph-v4-20261019`
- 최종 prefill: `/mnt/efs/nlp/psyche/riley-sm80-prefill-v5-20261019`
- [Compact 증거](../artifacts/c16-tail-v4-20261019/compact-evidence.zip): SHA256 `a9bb0097e1df41e1159929972891d69e55b9d888e192af01e88ef7f9d4e4213d`. `benchmarks/evidence/20261019-c16-tail-v5/`에1035개 hash-listed 파일을 검증·추출했고 raw index는2812개 파일을 가리킨다. 단계별 audit·profile·CPU/HF 증거, scripts, argv, source manifest를 포함한다. 무거운 SSE/NVML/Nsight/SQLite/BF16/binary/weight 원본은 EFS에 둔다.

GPU6 holder PID2027914/start6083215119, GPU7 PID2027915/start6083215121을 유지했다. 모든 테스트 종료 후 각20GiB allocation/reservation, idle, 다른 compute process 없음, heartbeat와 원래 PID/start tick을 재확인했다. 테스트 중에는64MiB reservation으로 조절하며 CUDA context를 계속 유지했다. 자동 점유 해제나 compute spin은 하지 않는다. GPU0..5 및 외부 job은 건드리지 않았다.

## 다음 개선 순서

1. **Down native-order 구현**: 현재 decode M>4와 큰 prefill은 native anchor와 BF16 reduction이 다르다. SM80/cuBLAS12.4의 실제 M1/4 reduction·split-K·BF16 partial 경계를 CUPTI/SASS로 확인하고, 같은 순서를 유지하는 weight-reuse kernel을 먼저 component로 검증한다. 일반 큰 GEMM이나 tolerance 완화로 대체하지 않는다.
2. **Paged GQA Attention**: 현재 gather의 네 배 expanded 쓰기와 QK/PV 약442ms가 남는다. SM80의 T별 native QK/PV reduction을 독립 검증한 후 paged operand와 fused staging/softmax를 확장한다. SM89의 family table을 그대로 SM80에 적용하지 않는다. BF16 score/probability 경계와 원래 mask/EOS를 유지한다.
3. **Gate/Up decode 확장**: 이번 큰 GEMM 동등성은 prefill 구간이다. Decode native M1의 M17..32 또는 N19456 동등성을 얻은 것으로 간주하지 않는다. 별도 native-order kernel/고정 recipe가 exact gate를 통과한 경우에만 확대한다.

각 단계는 component의 full BF16 값, HF104행, Graph R1..32, 전체 SSE·자원 복구와 C16/C32 matched 반복을 거친다. C1의 작은 tail 회귀까지 해소되기 전에는 전역 기본값으로 승격하지 않는다.
