# Capacity-aware projection backend v19B

2026-10-22. 기존 tail/처리량 채택 기준을 통과하지 못해 기본 runtime은 v15C를 유지한다. B안의 구현은 검증된 미적용 패치로 보존한다. 큰 처리량 개선 목표는 여전히 미달이다.

## 구현과 적용 범위

[v19A](qwen-weight-layout-v19a-20261022.md)의 model-owned K-major Gate/Up 구조를 유지하며, **설정한 최대 sequence capacity가16 이하일 때만** cold packing/새 backend를 활성화한다. A→B production 변경은 owner의 capacity guard 한 곳이다. capacity32에서는 packed 사본을 만들지 않으며 기존 projection을 그대로 실행한다. 이는 실제 활성 R<=16 여부와 구분된다. 따라서 C32의 일시적인 작은 R에도 새 경로를 쓰지 않는다.

C16은36개 layer 중32개에 총3,187,671,040B(2.96875GiB)의 추가 packed 사본을 공유하며, hot packing/per-shape 복사 없이 R2..16의 native fused Gate/Up에 사용한다. R1·R17..32·prefill·Down은 원래 경로다. 두 FP32 누산 체인과 BF16 중간 경계를 그대로 유지하고 packed allocation을 Graph key와 replay/command-batch lease에 포함한다. C32의 추가 packed 저장량은0이다. kernel/layout은 선행 연산 검사와 동일하며 이 B에서 새 operator speedup을 주장하지 않는다.

[연산 비교](qwen-weight-layout-v19-screen-20261022.md)에서 실제 Qwen3레이어 Gate/Up의 지연은4.28~5.07% 감소했지만 Down packing·전치 Lt는 느렸다. [무손실 BF16 포맷](qwen-lossless-projection-v19-20261022.md)도 저장량은 줄었으나 실행이 느려 제외했다. CUTLASS 라이브러리를 통합했다는 뜻은 아니며 독립 WMMA 구현이다.

## 사전 등록 판정

B의 성능 측정 전에 정책과 SHA를 동결했다. C16은 중앙값과5개 paired 처리량이 모두 양수여야 한다. C32는 개선을 주장하지 않는 기존 경로 비교로 중앙값 회귀2% 이하, packed allocation/실행0이어야 한다. 두 C 모두 **어느 paired TTFT/TPOT p95/p99든2% 초과 악화가 있으면 채택하지 않는다**. 원래 literal BF16 정확도 기준과22528MiB 상한은 유지한다. 사전 등록 정책은 [adoption-policy.json](../benchmarks/evidence/20261022-weight-layout-v19b/adoption-policy.json)에 있고 normal-input SHA에 포함돼 있다.

## fresh 정상 HTTP 비교

같은 physical A100 GPU7에서 baseline v15C/B/stock vLLM0.27.1을 순차 실행했다. 엔진·동시성마다 fresh process5회,64warm+192keep, 각 arm3840outputtokens다. Qwen3-4B BF16 greedy, budget576,T1024,KV=C×64,prefixoff,12fixed prompts115..130tokens,output8/16/24/32균형이다. stock은 FA2/FULL_AND_PIECEWISE/version2다. 아래는5개 run-level 값의 중앙값이며 pooled percentile이 아니다.

| C | Matched v15C tok/s | v19B tok/s | 변화 | vLLM tok/s | v19B/vLLM |
|---|---:|---:|---:|---:|---:|
| 16 | 749.718 | 758.596 | +1.18% | 1147.495 | 66.11% |
| 32 | 980.833 | 976.130 | -0.48% | 1688.063 | 57.83% |

| C | 지표 ms | v15C | v19B | stock |
|---|---|---:|---:|---:|
| 16 | TTFT p95 / p99 | 79.808 / 163.501 | 81.099 / 162.441 | 84.171 / 111.642 |
| 16 | TPOT p95 / p99 | 19.999 / 21.457 | 19.774 / 21.082 | 12.671 / 13.328 |
| 32 | TTFT p95 / p99 | 250.153 / 334.932 | 253.244 / 340.307 | 165.031 / 218.328 |
| 32 | TPOT p95 / p99 | 31.664 / 38.560 | 31.715 / 38.285 | 17.341 / 22.687 |

C16 throughput pairs: +2.031% / -1.005% / +3.117% / +0.977% / +1.651%.

C32 throughput pairs: -0.486% / +1.001% / +0.493% / -1.246% / -0.809%.

| C | Repeat | Tail metric | 악화 |
|---|---|---|---:|
| 16 | r1 | ttft_ms_p95 | +4.80% |
| 16 | r1 | ttft_ms_p99 | +3.19% |
| 16 | r1 | tpot_ms_p99 | +2.51% |
| 16 | r3 | ttft_ms_p95 | +2.50% |
| 32 | r2 | ttft_ms_p95 | +3.71% |
| 32 | r2 | ttft_ms_p99 | +2.33% |
| 32 | r2 | tpot_ms_p95 | +2.12% |
| 32 | r2 | tpot_ms_p99 | +3.48% |
| 32 | r4 | ttft_ms_p95 | +4.12% |
| 32 | r4 | ttft_ms_p99 | +2.71% |

완료된 정상 arm30개를 제외하거나 재측정하지 않았다. TPOT는 request-average output-token latency의 percentile이며 token-level ITL이 아니다. C32의 차이는 기존 경로 fallback의 fresh 측정 결과이며 새 kernel speedup으로 해석하지 않는다.

## 실제 실행과 남은 격차

정상 측정과 분리한64warm64keep before/after4개 Nsight window다.

| C | before 전체 kernel sum ms | after 전체 kernel sum ms | packed calls | 그중 Graph | packed sum ms |
|---|---:|---:|---:|---:|---:|
| 16 | 1679.968 | 1657.153 | 3168 | 3168 | 248.877 |
| 32 | 1331.600 | 1338.057 | 0 | 0 | 0.000 |

모든 C16 candidate 정상5개와 profile1개 로그에서32-layer packed allocation을 확인했고 C32의6개 로그에서는 allocation이 없음을 확인했다. C32 trace의 새 kernel도0이다. Aggregate kernel sum은 배치 구성/호출수가 달라 동일 연산 speedup과 같지 않다. Opaque mixed vendor kernel의 Gate/Up·Down 귀속을 강제하지 않는다. 새 경로는 전체 decode의 일부인 R2..16 Gate/Up만 개선하며 Down·R17..32·prefill·attention·host 비용이 남는다. 연산4~5% 개선을 서버 전체의 큰 개선으로 환산할 수 없고, vLLM과 큰 격차는 해소되지 않았다.

## 정확도·자원·재현

원래 capacity32 suite7552파일(4480+bounded5762048+canonical1024)을 그대로 통과했다. 새 경로를 실제 쓰는 capacity16/budget576의 body1024+reversed mixed512+no-output496+canonical512를 추가해 **10096파일이 원래 native/HF BF16 바이트와 일치**한다. teacher 토큰·수치 기준·원래 검사를 바꾸지 않았다. CPU524pass/40ignored, 기존 GPU5개+packed1개 및 별도 packed Graph identity1개, 기존120 alignment/view 사례도 통과했다. 승인된 C02 테스트4개의 fixture만 private/tmp에서 재검사했고 나머지 모델·빌드·실행 자료는 EFS다. stock은 수치 oracle이 아니다.

finite whole-device peak19133MiB, 정상/별도 profile peak18835MiB로22528MiB 이하다. original GPU6/7 holder PID2027914/2027915와 ticks6083215119/6083215121,CUDA context를 유지했다. 최종3fresh sample에서 둘 다 idle20GiB를 확인했다. GPU6 외부 작업은 건드리지 않았으며 GPU7 normal/profile samples에는 외부 PID가 없다. CPU/I/O 전체 독점을 주장하지 않는다.

937source hash와8-file patch, 서버 SHA256 `6d19091a3ea8d879730a2ced7c2a7c9a174d5d5fe35e6dd578732c97732c698f`를 동결했다. [archive](../benchmarks/evidence/20261022-weight-layout-v19b.zip)는6569950B/788entries, SHA256 `2a2d851e0e90a5e643f309c14d7d9cdad4d826667931e713167bc42cb65d9472`다. ZIP CRC·모든 manifest SHA와 정확한8개 source patch reconstruction을 로컬에서 검사했다. 대형 BF16/ELF/SQLite/Nsight는 EFS SHA/size index로 보존한다. 정상30+별도 profile4arm의 저장 SSE structures6490개를 독립 검증했다.

[모든 반복·tail](../benchmarks/evidence/20261022-weight-layout-v19b/final-statistics.json) · [자원·정확도](../benchmarks/evidence/20261022-weight-layout-v19b/resource-audit-final.json) · [채택 판정](../benchmarks/evidence/20261022-weight-layout-v19b/adoption-decision.json) · [정확한 패치](../benchmarks/candidates/weight-layout-v19b/candidate.patch).
