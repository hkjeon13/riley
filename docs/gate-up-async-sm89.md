# Gate/Up 비동기 병합 후보 — 아직 미채택

검증 장치: ai-assistant의 RTX4090, SM89/128 SM, CUDA 12.8.1, cuBLAS 120804. A100 결과와 합산하지 않는다. 기존 채택 소스는 actual164이며 이번 후보는 연구용 독립 CUDA 프로그램이다.

설치된 stock vLLM 0.27.1은 Qwen3에서 `QKVParallelLinear`, Qwen3가 재사용하는 Qwen2MLP에서 `MergedColumnParallelLinear`와 `SiluAndMul`을 사용한다. 참고한 핵심은 같은 입력에 대한 독립 projection의 작업을 함께 노출하는 것이다. vLLM의 GEMM 크기/Attention 연산을 그대로 바꾸면 원래 Riley의 reduction 및 BF16 반올림 경계와 달라질 수 있어 별도 검증한다.

후보는 원래 가중치 배열을 유지하며 Gate와 Up의 column 작업을 하나의 grid로 묶는다. K=128 타일의 가중치를 cp.async로 다음 공유 메모리 버퍼에 미리 읽고, 현재 타일의 기존 FP32 FFMA와 겹친다. FFMA 순서, reduction tree, BF16 저장 경계는 유지한다. R16/C8의 공유 메모리는 두 버퍼 합계 26624 B이며 host/device가 같은 constexpr를 사용한다. 초기 단일 버퍼 크기 할당 오류였던 v209는 보존했고 v210에서 수정했다.

두 모델의 실제 서로 다른 Gate/Up 가중치와 동일 입력으로 R1..32, 원래 입력/0/복원 상태를 검증했다. fresh 같은 장치 M1 출력과 384개 조건의 BF16 값이 전부 일치했고 128개 Graph의 변경 입력 replay 256개 상태도 일치했다. 이 결과는 모델 전체의 logits/생성/API 검증을 대신하지 않는다.

메모리 및 레이스 검사는 R25/32, 두 모델, 새 async 커널만 대상으로 했다. 도구마다 48개 조건, 16개 Graph, 32개 변경 상태를 검사해 memcheck 오류 0, racecheck 오류·hazard·warning 0이었다. 다른 row tile이나 전체 Riley의 안전성을 포괄하는 결과는 아니다.

## 실제 기존 dispatch와 비교한 컴포넌트 속도

다음은 512 MiB device write 이후 두 projection의 CUDA event 측정이다. 하드웨어 캐시가 완전히 비었다는 주장은 아니다. 4회 순서 교대 측정은 한 프로세스 안의 컴포넌트 측정이며 HTTP 처리량이나 독립 재시작 통계가 아니다. 양수는 기존 대비 speedup 증가율이다.

| 모델 | 배치 R | speedup 증가율 범위 |
|---|---:|---:|
| qwen4b | 1 | -19.87% ~ -19.87% |
| qwen4b | 2–8 | +0.67% ~ +2.60% |
| qwen4b | 9–16 | -11.64% ~ -8.82% |
| qwen4b | 17–24 | +11.92% ~ +14.63% |
| qwen4b | 25–32 | +7.82% ~ +10.05% |
| mistral | 1 | -12.54% ~ -12.54% |
| mistral | 2–8 | -3.96% ~ +2.81% |
| mistral | 9–16 | -0.31% ~ +0.57% |
| mistral | 17–24 | +11.39% ~ +13.22% |
| mistral | 25–32 | +5.05% ~ +6.25% |

R1과 일부 작은 배치에서 회귀한다. 전체 범위에 일괄 적용하지 않는다. 큰 배치의 이득도 실제 서버의 처리량과 TTFT/TPOT를 통과하기 전에는 채택하지 않는다. 이전 Attention PV 후보 v199 역시 수치 검증은 통과했지만 실제 서버의 작은 처리량 이득과 지연 회귀 때문에 미채택이다.

## 후속 상태

v213은 기존 custom projection 커널의 인수·grid·공유 메모리를 유지하고 Graph DAG에서 독립 Q/K/V 및 Gate/Up의 의존성만 fork/join하는 사전 실험이다. CPU 빌드만 완료했으며 GPU 실행 전 admission에서 A100 2·3번이 비어 있음을 확인하여 RTX 실험을 시작하지 않았다. 실제 병렬 실행이나 성능 향상을 주장하지 않는다.

A100에서는 actual164의 고정된 935개 소스를 CUDA12.4/SM80으로 새로 빌드했다(v214, exit0). v215는 실제 가중치 범위와 입력을 옮겨 fresh A100 M1과 독립적으로 비교하는 컴포넌트 qualification이다. SM89의 결과를 SM80 지원 증거로 사용하지 않는다. 기존 SM89 custom dispatch를 복제한 비교 arm도 실제 A100의 현재 runtime dispatch로 표시하지 않는다.

원시 증거: `artifacts/multimodel-parity-20261017/v211-controller-result.json`, `v212-controller-result.json`, `gateup-allrows-v212.cu`, `v213-prototype-plan.json`, `v214-a100-build-receipt.json`, `v215-handoff.json`. 배포·commit·push는 하지 않았다.

## A100에서 별도로 확인한 결과

새 A100 M1은 `ampere_bf16_s16816gemm_bf16_128x64_sliced1x2_ldg8_f2f_stages_64x6_tn`을 사용한다(nsys v218, 두 출력 모두 이전 fresh M1과 일치). SM89의 SIMT 순서를 그대로 옮긴 후보는 Qwen Gate 10개/Up 8개 값이 달랐다. CPU에서 16-lane FFMA 순서를 재현하면 후보와 전 값이 일치하여, 비동기 복사 오류와 native 기준 커널 선택 차이를 구분했다. 이 포트는 미채택이다.

대신 각 projection의 GEMM을 최대 16행으로 제한한 A100 후보는 총 R1..32 모두에서 fresh M1과 일치했다(v221): 두 모델 512개 조건, 64개 Graph, 256회 replay, 원래 입력→0→복원→변형 입력의 변경 상태 192개, BF16 값 203,292,672개 중 차이 0개. 실제 서로 다른 Gate/Up 두 행렬을 사용한 단일 레이어 컴포넌트 검증이며 모델 전체·HF·실제 API·성능 검증은 남아 있다.

이 경로를 기존 소스의 owned GEMM 실행 지점에 추가한 private v220을 별도 빌드하고 있다. SM80/108SM/runtime12040/cuBLAS120405, 원래 workspace/MPS 환경, explicit mode5, Gate/Up의 두 검증된 N/K 조합에서만 진입한다. 원래 buffer/plan/stream 등록과 workspace 소유권을 유지한다. 주 작업 소스에는 적용하지 않았으며 전체 모델 검증과 같은 장치 vLLM 비교 후 채택 여부를 판단한다. 현재 진행 상태는 `v220-handoff.json`과 `checkpoint.json`에 보존한다.

현재 A100 상태: 독립 mode6/private v225를 구현했고 두 모델의 전체 logits/배치1~32/Graph 검증이 완료됐다. 주 작업 소스 채택과 실제 API 성능 비교는 남아 있다. 상세 결과는 [A100 Gate/Up 검증](bounded-gateup-sm80.md)에 정리했다.
