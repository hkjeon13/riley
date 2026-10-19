# Shared-KV Attention 후보183 검증 결과

이번 후보는 미반영이다. 현재 주 소스는 통합164를 유지한다. 기존 QKV·Gate/Up 배치 개선 위에 QK·HF Softmax·PV를 결합하고, 두 query head가 K/V 타일을 공유하도록 구현했다. 기존 FP32 누산·리덕션 순서와 BF16 반올림 경계를 유지한다. 실험에서 선택된 부품은 약1.08~2.14배 빨랐지만, 전체 서버 개선은 아래와 같이 작았다.

## 같은 RTX4090 실제 서버 비교

단위 출력 tok/s. 각 셀은 새 프로세스의 현재164/후보183/stock vLLM0.27.1 비교 1회, 각각 natural 요청96개이다. 반복 중앙값·일반화된 성능 개선으로 해석하지 않는다.

| 모델 / 동시성 | 현재164 | 후보183 | 변화 | stock vLLM | 후보/vLLM | TTFT p99 ms 현재→후보 |
|---|---:|---:|---:|---:|---:|---:|
| qwen4b C16 | 596.66 | 597.52 | +0.14% | 1123.81 | 53.2% | 326.32 → 329.69 |
| qwen4b C32 | 745.81 | 756.38 | +1.42% | 1691.94 | 44.7% | 781.57 → 781.65 |
| mistral C32 | 541.72 | 547.84 | +1.13% | 1056.47 | 51.9% | 1313.09 → 1391.25 |

22GiB 예산, context1024, 토큰 예산512, KV block16/블록64×동시성, prefix off, 같은 CPU affinity, 원래 EOS와 greedy를 유지했다. native CUDA12.8과 설치된 stock CUDA13의 차이는 남아 있다. 각 측정 실행은 출력7,168토큰으로 동일했다. stock의 HF 일치는 별도 원본 필드이며 속도 동등성이 수치 동등성을 뜻하지 않는다.

세 셀 모두 모든 p50/p95/p99 지연 무회귀 조건을 통과하지 못했다. 이 작은 효과를 채택하거나 같은 후보를 반복해 좋은 결과만 선택하지 않았다. vLLM과의 성능 격차 해결은 아직 미완료다.

## 수치 및 실행 검증

- CPU517개 통과. 일반 GPU 테스트40개 제외와 실제 GPU 검증은 구분한다.
- CAPI 112개 smoke 및7,168개 전체 조건, T1~512. 캡처 Graph7,168개/변경 입력 상태14,336개. raw QK BF16, scaled score F32, probability BF16, context BF16 모두 변경0. 사용하지 않는 미래 KV는 NaN으로 채웠다. hot allocation0.
- Qwen/Mistral 각각 기존 HF full-vocabulary104행, C1 및 실제1~32행의 변경 context Graph 일치. 실제 fused kernel 실행을 확인했다.
- 긴 trajectory 각각256 full-vocabulary행: 현재164 C1 eager 대비 후보183 C32 고정 및 가변 배치가 byte-exact. 각 후보 arm에서 Qwen38,895,616개, Mistral8,388,608개 BF16값. 긴 추가 구간은 native 전후 비교이며 새로운 HF logits 검증으로 주장하지 않는다.
- 서버 전체 응답3,248개를 native eager 기준과 비교했다. 토큰/텍스트/프롬프트/종료/사용량 모두 일치. 소유 PID,22GiB 예산, 할당 안정성, 연결 취소·재시도 복구도 통과했다. 측정 중 신규 Graph 캡처0.
- 이전 모드2/3/4와 기존 Smol/Qwen2.5 모델 회귀 검사 통과. Qwen2.5의 기존 HF 차이는 그대로 기록했다.

## 후속 진단과 실험

Graph 수준 CUDA trace를 현재164와 후보183에서 각각32개의 유한 요청으로 수집했다. 응답1,216개는 기준과 완전히 일치했다. 프로파일러가 붙은 시간은 앞 성능 표에 섞지 않는다.

현재164의 GPU 활동 합집합3,232.80ms 중 Graph 구간은1,756.98ms(54.3%)였다. Graph 밖 custom Attention은408.35ms였고, 후보183에서는 남은 custom320.55ms+fused48.85ms였다. Graph 밖 라이브러리 시간은631.81→632.40ms였다. 이 라이브러리 합에는 prefill/linear/Attention이 섞여 있어 전부 Attention이라고 단정하지 않는다. GPU 빈 구간400.58→391.85ms에는 계측 오버헤드가 포함되므로 production CPU 병목의 증명으로 사용하지 않는다.

이에 QKV·Gate/Up 등 큰 배치 projection에32행 two-column 타일186을 새로 시험했다. 각 모델6개 실제 가중치/128개 activation행,3개 타일 변형, 모델당5,292조건이 모두 exact였으나 시험한 warm/postflush 속도가 모두 기존보다 느려 미반영했다. 기존32행 C4 column-major를 새 후보로 다시 시험한 것은 아니며,32행 row-major/C8 구성이 대상이었다.

다음으로 pointer-array cuBLAS BMM의 head batch32를32×요청 수로 늘린187은2,304조건 중549조건, 변경 Graph120상태 중10상태가 불일치했다. random만으로는 드러나지 않고 실제 activation 유래/큰 상쇄 입력에서 드러났다. 원본 출력·실패·반례는 보존했다.

188의 두 유한 반례 trace에서 다음 변화를 확인했다.

| 반례 | 원래 batch32 | 확대한 배치 | 출력 차이 |
|---|---|---|---|
| T1/R2 QK | gemvx variant6 / block4×16 | variant5 / block4×8 |18 BF16값, 일부27→28 |
| T397/R12 PV | gemvx variant9 / block64×8 | variant6 / block32×2 |2 BF16값, 예:0.67578125→0.6796875 |

행렬·입력·m/n/k·dtype·compute32F는 같고 batchCount에 따라 내부 커널/스레드 구성이 달라졌다. 이것은 부동소수점 누산·리덕션 재배치 가능성의 직접적인 실행 근거이다. 각 커널의 상세 FMA 순서를 새로 역공학했다는 주장은 하지 않는다.

189는 원래 native 함수·block·gridXY·shared 크기를 유지하고 독립 행렬의 gridZ만 늘어나는 구간까지만 묶도록 구현했다. 각 T와 역할 QK/PV별로1부터 연속으로 같은 구성이 유지되는 cap을 Driver Graph에서 확인하고, 선택이 달라지면 더 큰 배치는 원래 크기로 분할한다. 이 후보189는 전체65,536조건에서126조건이 불일치했다. T122의 실제 반례190/191에서 native 함수·스레드는 같지만120바이트 인자의 offset112 정수가1→4로 바뀐 것을 확인했다. 내부 필드의 정확한 의미를 추측해 이름 붙이지 않았다. 주 소스·API에는 아직 반영하지 않았다. 수치 허용오차는 완화하지 않았다.

근거: [Graph 시간 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v185-trace-audit.json), [projection186](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v186-controller-result.json), [BMM 반례 근거](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v188-counterexample-diagnosis.json).

A100과 RTX 결과는 합산하지 않는다. 각 유한 GPU 실행 후 A100 GPU2/3 점유자 확인을 유지한다. 외부 작업 종료·배포·커밋·푸시는 수행하지 않았다.

근거: [전체 서버 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v184-pilot-independent-audit.json), [수치 포함 표](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v184-verified-performance-table.json), [CAPI](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v183-CAPI-controller-result.json), [긴 context](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v184b-long-context-controller-result.json).

## 전체 kernel argument까지 고정한 후보192

선택 조건을 강화하여 native 함수·block·gridXY·shared와 **전체 kernel argument 바이트**가 같은 범위에서만 BMM을 합치고, 다른 경우 원래 호출 크기로 분할했다. 원본 값에 맞추기 위한 오차 허용이나 출력 보정은 하지 않았다.

모든T1~512 × 실제R1~32 × random/큰 상쇄/실제 activation 유래/zero의65,536조건에서 QK/PV BF16 변경0이었다. 별도42개 캡처 Graph의168실행 상태에서도 변경0이었다. 양쪽 Graph를 새 원래 eager 출력과 비교했고 pointer 순서 변경을 포함했다.1024개 T/역할 cap의 전체 인자 동일성을 독립적으로 감사했다.

| 문맥T / 배치32 | GPU 부품 warm 속도비 |512MiB 쓰기 후 속도비 |
|---|---:|---:|
|397 |1.94× |1.87× |
|415 |1.92× |1.87× |
|511 |1.82× |1.75× |

이는 원래 라이브러리 QK/PV 호출 두 역할을 묶은 부품 비교이다. 기존 production의 custom native QK/PV/WMMA가 적용되는 경로는 이 비교에 포함하지 않았다. 후보 시간에는 추가 pointer pack이 포함된다.512MiB 쓰기는 하드웨어 cache miss의 증명이 아니다. 전체 Attention·모델·서버 개선으로 확대하지 않는다.

주 소스164에서 분리된4파일 후보193/194의 CPU 빌드가 완료됐고194 CPU 회귀517개 통과/일반 GPU40개 제외를 확인했다. GPU 검사는 이 제외 목록과 별개로 실행했다.194는16바이트 정렬 및 유효한 남은 라이브러리 그룹이 있을 때만 pointer pack을 켠다.193 GPU는 실행하지 않았다.194 실제 paged Attention smoke629건과 T1~128의4,736건이 변경0으로 통과했고, smoke Graph629개 중371개에서 실제 pointer pack/native BMM batch 확대를 Driver API로 확인했다. 나머지 T129~512와 모델·긴 문맥·기존 모드 검증은 진행 중이며 새 서버 성능 비교는 아직 수행하지 않았다.183의 미반영 fusion은 이 새 후보에 포함하지 않는다.

근거: [전체 독립 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v192-independent-audit.json), [내부 인자 차이](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v191-parameter-difference-summary.json), [분리된194 delta](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v194-backend-delta-plan.json).


194 smoke 종료 후 SSH 연결만 남았던 장애는 remote 완료 기록과 GPU 프로세스 종료를 확인한 뒤 소유한 SSH만 닫아 복구했다. 원래 실패 관측 기록은 보존했고 GPU smoke는 반복하지 않았다. 이후 컨트롤러는 SSH와 별개로 유한 실행 완료 기록을 확인한다.
