# Bounded BMM 후보194 실제 서버 검증

후보194는 미반영이다. 주 소스는 통합164를 유지한다. 수치 검증은 통과했지만 실제 서버 이득이 없어 같은 후보를 반복해 유리한 결과를 고르지 않는다. 광범위한 HF 모델에서 stock vLLM 동급 이상이라는 목표는 미달이다.

## 같은 RTX4090 서버 비교

각 셀마다 현재164/후보194/stock vLLM0.27.1을 새 프로세스로 비교1회, natural 요청96건씩이다. 단위 출력 tok/s. 표의 각 실행 출력토큰7,168개는 같다. 반복검증 중앙값·일반 성능 우위로 해석하지 않는다.

| 모델/동시성 | 현재164 | 후보194 | 변화 | stock vLLM | 후보/stock |
|---|---:|---:|---:|---:|---:|
| qwen4b C16 | 597.19 | 596.82 | -0.06% | 1125.80 | 53.0% |
| qwen4b C32 | 746.36 | 746.16 | -0.03% | 1690.18 | 44.1% |
| mistral C32 | 542.49 | 541.21 | -0.24% | 1056.00 | 51.3% |

| 모델/동시성 | TTFT p95 ms 현재→후보 | TTFT p99 ms 현재→후보 | TPOT p99 ms 현재→후보 |
|---|---:|---:|---:|
| qwen4b C16 | 176.22→189.58 | 321.44→319.66 | 25.30→25.10 |
| qwen4b C32 | 587.56→590.89 | 768.68→772.84 | 40.43→39.69 |
| mistral C32 | 1078.44→1070.64 | 1321.87→1315.32 | 59.99→60.07 |

세 셀 모두 모든 p50/p95/p99 지연 무회귀 조건을 통과하지 못했다. 일부 지연은 개선했지만 처리량은 모두 거의 같거나 낮았다.

같은 GPU UUID,22GiB 예산/context1024/토큰예산512/KVblock16 및64×동시성/prefixoff/CPUaffinity/greedy/원래 EOS를 유지했다. Native CUDA12.8, 설치된 stock CUDA13 차이는 남아 있다. stock HF 일치는 원본 별도 필드로 기록하며 속도로 품질 동등성을 주장하지 않는다. TPOT p99는 요청별 평균 token interval의 백분위이며 개별 token ITL p99가 아니다.

## 수치와 실제 실행 검증

- CPU517개 통과, 일반 GPU40개 제외. 별도 실제 GPU 검증과 구분한다.
- 전체T1~512 × 실제R2~32 및6개 mixed 모드:18,944조건, 캡처18,944Graph/변경입력37,888상태에서 context/probability BF16 및 scaled score F32 변경0, 추가할당0. 미래KV NaN 및 역순slot/page 통제. 별도 smoke629조건도 통과.
- smoke629Graphs 중371개에서 실제 pointer pack 및 확대된 native BMM batch count를 Driver API로 확인.
- Qwen/Mistral 각각 기존 같은 GPU HF104 full-vocabulary행과 C1/고정Graph/실제1~32행 변경context가 바이트 일치. 실제 uncaptured pointerpack 실행 관측. Driver metadata에서 각 row1~32의 pre/post Graph를 각각36/32개 확인했다. Replay는 source-bound once-per-entry flag와 capture/first-replay 동일 개수로 전 Graph coverage를 확인했으며 개별 replay timestamp를 직접 기록한 것은 아니다.
- 긴 native trajectory256 full-vocabulary행/model에서 현재164C1 eager와 후보194 C32고정/가변이 모두 일치. 각 after의 Qwen38,895,616개/Mistral8,388,608개 BF16값. 추가 긴 구간은 native 비교이며 새 HF logits 검증은 아니다.
- legacy2/3/4 및 Smol/Qwen2.5 C1/C8 Graph 전후 일치. SmolHF104 일치, Qwen2.5 기존 HF16 기준 차이는 유지하며 해결했다고 주장하지 않는다.
- 서버 응답3,248개를 tokens/text/prompt/finish/usage 전체로 native eager 기준과 감사했다. 소유 PID/22GiB/할당·KV 복구/연결중단·재시도 통과, 측정 중 신규Graph캡처0.

## 부품 이득과 실제 경로의 차이

이전 부품192의 약1.75~1.94배는 원래 라이브러리 QK와 PV 두 역할을 함께 비교한 수치다. 실제 소스는 T397/415/511의 QK를 이미 native F8 커널로 처리한다. 따라서 그 부품 비교의 QK 배치 확대 이득이 해당 실제 경로에는 적용되지 않는다. 초기 가설에서 긴 홀수T의 QK/PV가 둘 다 라이브러리라고 본 QK 부분은 잘못됐고 정정했다. 실제native QK지원은T1~512 중420개 길이다. PV 긴 홀수 구간은 여전히 라이브러리다. 실제 equal-T cohort 빈도와 잔여 시간 비중은 별도 진단이 필요하며 소스만으로 단정하지 않는다.

이 때문에 균일한 합성 부품 이득을 서버 개선으로 확대하지 않았다. 후보194는 원래 연산을 보존하는 안전한 bounded BMM을 구현·검증했으나 실제 효과가 없어 채택하지 않는다.

## 다음 진행

196 timed arms와 감사가 모두 끝난 뒤197 PV arithmetic probe를 시작했다. 과거74는 QK만 수집했으며 PV_covered0은 PV 실패를 뜻하지 않는다. 새197은512개T, R32,3개 seed×5상태에서 contiguous ceil(T/lanes)와 여러 정확한 reduction 순서를 원래 request별 PV와 비교한다. 일부 변형의 불일치는 보존하고, 일치하는 가족만 다음 후보로 검토한다. 아직 API·HF·모델·성능 검증이 아니다.

그 외 Ktile224/shared occupancy와 독립 Gate/Up Graph 병렬화, 완전 GPU-native wide Attention 및 whole Graph는 미구현 가설로 분리했다. 어떤 것도 성능 개선으로 주장하지 않는다.

A100 GPU2/3 외부 wckim 작업을 보존했고, 각 유한 GPU 작업 후 재확인했다. 이번 결과는 RTX4090만이며 A100과 합산하지 않았다. 배포·커밋·푸시·외부 작업 종료·장시간 부하 실행을 하지 않았다.

근거: [전체 서버 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v196-pilot-independent-audit.json), [수치 표](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v196-verified-performance-table.json), [전체 CAPI 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v194-independent-CAPI-audit.json), [Graph key 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v194-independent-live-Graph-audit.json), [원래 경로 정정](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/post196-native-recipe-source-audit.json).


197의 유한 arithmetic search는 완료됐다.12개 변형 중 적어도 하나가159개 T에서 원래 PV와 일치했다. 총512T ×3seed ×5상태, R32이며, T397/415/511은8개 contiguous ceil(T/8) 부분합과 descending tree가 변경0이었다. T122/124/130/134는4개 부분합 descending tree가 변경0이었다. 이 결과는 입력이 주어진 PV 산술 검증이며 GQA8·heterogeneous metadata·Graph·HF·모델·속도 증명은 아니다. 다음 단계는 실제 남은 PV만 대상으로 coalesced batched kernel을 작성해 기존 request별 라이브러리 경로와 비교하는 것이다.
