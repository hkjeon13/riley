# Model-owned K-major Gate/Up backend v19A

2026-10-22. 구조 변경을 실제 서버와 decode Graph에 연결했다. **C16은 +2.49%, C32는 -0.18%이며, C32 paired TPOT p99 두 건이 기준을 넘어 전체 구성의 기본 경로로 채택하지 않는다.** 기존 v15C가 이 체크포인트의 accepted runtime이다. 성능이 확인된 용량에서만 선택하는 [v19B 최종 비교](qwen-weight-layout-v19b-20261022.md)도 완료했으며, B 역시 반복별 tail 기준을 통과하지 못해 미채택했다.

## 바뀐 구조

checkpoint의 Gate/Up을 cold upload에서 `[K64][N32][Gate32|Up32][K64]`로 순열 배치한다. 모델 owner가 한 레이어당 한 사본을 보관하고 모든 shape/GEMM plan이 공유한다. per-shape 복사나 hot packing은 없다. 3GiB 사본 예산으로 Qwen36개 레이어 중32개를 지원하며 실제 추가 저장량은3,187,671,040B(2.96875GiB)다. 기존 row-major/merged 사본은 prefill과 R17..32에 그대로 쓴다.

새 WMMA backend는 R2..16의 원래 두 FP32 누산 체인과 BF16 projection·SiLU·multiply 경계를 유지한다. mixed singleton groups와 pure-decode MLP Graph 양쪽에 연결했다. packed allocation identity가 Graph key에 들어가고, capture/replay의 command-batch lease와 전체 모델 parent binding에도 포함된다. 기존4-span API는 유지하고5번째 packed span을 받는 경로는 크기·dtype·context·alias·alignment를 검증한다. Down과 R17..32의 Lt 경로는 바꾸지 않았다.

[선행 연산 실험](qwen-weight-layout-v19-screen-20261022.md)은 실제3레이어의 R2..16에서4.28~5.07% 낮은 지연을 확인했다. Down packing과 전치 Lt는 이득이 없었다. [별도 무손실 포맷](qwen-lossless-projection-v19-20261022.md)도 수치는 같지만 느려 제외했다.

## 정상 HTTP 비교

같은 physical A100 GPU7에서 엔진별 fresh process5회,64warm+192keep, C16/C32를 측정했다. Qwen3-4B BF16 greedy, budget576, T1024, KV=C×64, prefix cacheoff,12fixed prompts115..130tokens 및 균형 output8/16/24/32다. 각 정상 arm은3840output tokens다. baseline은 accepted v15C, stock은 vLLM0.27.1/cu129, FA2/FULL_AND_PIECEWISE/version2다.

| C | Matched v15C tok/s | v19A tok/s | 변화 | vLLM tok/s | v19A/vLLM |
|---|---:|---:|---:|---:|---:|
| 16 | 732.961 | 751.208 | +2.49% | 1140.859 | 65.85% |
| 32 | 979.213 | 977.470 | -0.18% | 1667.614 | 58.61% |

5개 run-level metric의 중앙값이며 pooled percentile이 아니다. C16의 각 throughput pair 변화는 +6.44/+2.50/+1.31/+2.96/+2.85%다. C32는 -0.63/-1.79/+0.31/+0.11/-0.06%다. 완료된 정상 arm을 제외하거나 다시 측정하지 않았다.

| C | 지표 ms | v15C | v19A | stock |
|---|---|---:|---:|---:|
| 16 | TTFT p95 / p99 | 81.250 / 162.209 | 79.915 / 161.778 | 83.402 / 110.817 |
| 16 | TPOT p95 / p99 | 20.356 / 21.595 | 20.005 / 21.242 | 12.731 / 13.472 |
| 32 | TTFT p95 / p99 | 247.164 / 333.515 | 246.619 / 333.200 | 163.204 / 216.428 |
| 32 | TPOT p95 / p99 | 32.063 / 38.141 | 31.861 / 37.037 | 17.893 / 21.187 |

C16의 paired TTFT/TPOT p95/p99 >2% flag는0개다. C32는 r0 TPOT p99 **+6.26%**, r2 **+4.42%**다. 중앙값이 좋아진 지표라도 이 pair들을 지우지 않는다. TPOT는 request-average token latency의 percentile이며 token-level ITL percentile이 아니다.

GPU6 정상 측정은 외부 PID가 시작 전 발견돼0arm에서 중단했다. GPU7 전환 후 측정기 복사의 허용 GPU 목록 오류도 서버 실행 전 수정했다. 이 준비 실패는 정상 표본으로 계산하지 않는다. 이후 C16의 첫 Riley2arm이 끝난 뒤 stock이 누락된 EFS IPC parent 때문에 요청 전에 실패했다. IPC를 만들고 **기존2arm을 그대로 유지한 채 나머지13arm만 동일 runner로 이어갔다**. 원본 오류·partial result·continuation argv와 SHA를 보존했다. C32는 연속15arm이다.

## 별도 프로파일의 실제 실행

정상30arm과 분리된64warm64keep before/after4개 window다.

| C | before 전체 kernel sum ms | after 전체 kernel sum ms | 새 packed 실행수 | 그중 Graph | packed kernel sum ms |
|---|---:|---:|---:|---:|---:|
| 16 | 1684.048 | 1637.021 | 3168 | 3168 | 248.648 |
| 32 | 1302.837 | 1333.107 | 800 | 800 | 61.645 |

새 커널은 두 동시성 모두 실제 실행됐으며 모든 해당 호출이 Graph 안에 있었다. C32에서는 이 window 전체 kernel sum 중 packed 커널이 약4.62%여서 적용 범위가 작다. 이 비율은 정상 HTTP 개선율이 아니다. 호출수와 배치 구성이 달라 aggregate 전후 차이를 동일 연산의 speedup으로 계산하지 않는다. opaque mixed GEMM을 모두 Gate/Up 또는 Down으로 강제 귀속하지 않았다. R17..32·prefill Gate/Up, 원래 Lt Down, attention·host 비용은 남는다.

## 정확도·메모리·홀드

원래 native/HF BF164480파일, serving-capacity576 추가2048파일, canonical output1024파일을 모두 literal equality로 검증했다. GPU7에서 같은 바이너리의 Graph208파일을 추가 확인해 총 **7760파일**이다. reference/tolerance는 그대로이며 stock vLLM은 numerical oracle이 아니다. CPU524pass/40ignored, 기존 GPU계약5개와 새 packed검사1개, 별도 packed Graph identity검사1개가 통과했다. 기존120 alignment/view 사례도 유지했다.

새 invalid-span 테스트의 첫 시도는 올바른 거부 이후 poisoned plan의 close를 성공으로 기대해 실패했다. 기존 close 계약대로 오류를 요구하도록 테스트만 수정했다. 실행 소스7개와 서버 바이너리는 바꾸지 않았으며 실패 로그와 amendment를 남겼다. 승인된4개 C02 fixture만 private/tmp에서 재검사했고 삭제했다. 다른 모든 실행 데이터·스크립트·빌드는 EFS다.

qualified source937개와8-file patch를 검증했다. 유한 검증 whole-device peak22205MiB, 정상/별도 profile peak21339MiB로 기존22528MiB 상한 이하다. original holder6/7 PID2027914/2027915, ticks6083215119/6083215121 및 같은 CUDA context를 유지했다. 최종3개 fresh sample에서 유휴20GiB를 확인했다. 외부 GPU6 작업은 건드리지 않았다. 정상/profile GPU7 samples의 외부 PID는0이며 CPU/I/O 전체 독점을 주장하지 않는다.

[모든 반복·통계](../benchmarks/evidence/20261022-weight-layout-v19a/final-statistics.json) · [실행·자원 증거](../benchmarks/evidence/20261022-weight-layout-v19a/resource-audit-final.json) · [정확한 패치](../benchmarks/candidates/weight-layout-v19a/candidate.patch) · [archive](../benchmarks/evidence/20261022-weight-layout-v19a.zip). Archive7,023,216B/917entries, SHA256 `160cdf63e7ceb42039b653a59a785a6dbb67031adc2f8339770fc2bd5a77f29f`. 서버 SHA256 `9d2e280d320e868eae87a515cfc533406d7239b4f1b333ddcb4a035308fb05ec`. Local검사에서 ZIP CRC·모든 manifest SHA·8개 source의 patch reconstruction을 확인했다. 대형 BF16/ELF/SQLite/Nsight는 archive의 EFS SHA/size index로 보존한다.34arm 및6490stored SSE value structures를 독립 검증했다.
