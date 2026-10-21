# Qwen SM80 projection batching v2

2026-10-19 KST. 이전 [TTFT 개선](qwen-ttft-p95-investigation-20261019.md)의 Gate/Up + GPU greedy 설정을 기준으로, 실제 큰 비용을 차지하는 projection을 추가 배치했다. 기존 HF/native BF16 수치 기준을 유지한다.

## 결과

Qwen3-4B, aws-a100 물리 GPU7, BF16, Graph off, iteration-batch completion, 동일 GPU greedy 설정으로 C1/C8 각각 두 fresh-process AB/BA 반복을 비교했다. arm당 32 warmup + 96 측정 요청, 출력은 각 1,920 tokens다. 표는 두 arm 수치의 중앙값이며 과거 48-request screen과 합치지 않았다.

| C8 항목 | 기존 Gate/Up + GPU greedy | projection v2 + GPU greedy | 변화 |
|---|---:|---:|---:|
| output tok/s | 158.07 | 259.08 | **+63.9%** |
| TTFT p50 ms | 106.45 | 72.14 | -32.2% |
| TTFT p95 ms | 121.52 | 97.17 | **-20.0%** |
| TTFT p99 ms | 145.60 | 129.59 | -11.0% |
| 응답 평균 TPOT p99 ms | 49.80 | 29.73 | -40.3% |
| E2E p99 ms | 1604.62 | 990.40 | -38.3% |

두 반복의 처리량 비율은 **1.6423배, 1.6358배**다. C8의 TTFT/응답 평균 TPOT/E2E p50/p95/p99 모두 두 반복에서 개선됐다. C1 처리량은 85.64 → 85.80 tok/s로 사실상 유사하다. C1 일부 quantile의 작은 변동까지 전체 무회귀로 선언하지 않는다. 두 반복은 확대 release matrix를 대신하지 않으며 기본값은 유지한다.

중간 Q/O/Down-only 후보도 별도로 측정해 158.53 → 231.60 tok/s(+46.1%)를 확인했다. 이후 LM head 배치를 추가한 최종 조합을 별도의 두 쌍으로 다시 대조했다. 서로 다른 단계의 비율을 곱해서 최종 개선율을 만들지 않았다.

## 구현

새 명시적 mode7 `bounded-projections-sm80-v2`를 통합했다. 기존 mode0..6과 기본값은 유지한다. merged QKV로 강제 전환하지 않고 현재 separate-QKV 실행 경로에서 Q만 묶는다.

| 연산 | N/K | 한 GEMM의 최대 요청 행 | 실행 방식 |
|---|---|---:|---|
| Q | 4096/2560 | 32 | 기존 cuBLAS GemmEx |
| Gate/Up | 9728/2560 | 16 | 기존 mode6 유지 |
| O | 2560/4096 | 8 | 고정 Lt21/tile15/stages18/split2/F32 reduction/swizzle0 |
| Down | 2560/9728 | 4 | 고정 Lt31/tile15/stages18/split2/BF16 output reduction/swizzle1 |
| LM head | 151936/2560 | 16 | 기존 cuBLAS GemmEx |
| K/V | 1024/2560 | 1 | 기존 경로 유지 |

Down의 BF16 partial reduction은 기존 native M=1이 가진 반올림 경계를 재현하는 선택이다. 정확성 계약을 완화하거나 새로운 저정밀 허용오차를 도입하지 않았다. 일반 M=8 Down과 17행 이상 LM head는 실제로 값이 달라져 채택하지 않았다. 각 최대 행을 넘는 실제 batch는 검증한 크기로 나누며 남은 1행은 원래 M=1을 사용한다. 다중 token prefill request extent와 Attention의 산술·KV 표현은 유지한다.

O/Down Lt descriptor·layout·algorithm과 workspace는 cold prepare에서만 준비한다. workspace는 plan이 소유하는 ledger-tracked CUDA buffer이며 원래 command-batch use ledger에 등록하고 Graph보다 늦게 해제한다. workspace check는 최대 4MiB를 넘으면 거부한다. 실행/capture 중 자체 CUDA buffer를 새로 할당하지 않는다. Graph 옵션은 기존 layer segment 방식이며 동적 Attention을 무조건 whole-body capture하는 변경은 하지 않았다.

지원 조건은 Qwen3-4B geometry, SM80/108SM, CUDA runtime12040, cuBLAS/Lt120405, 원래 workspace/SM 설정, 최대 32 sequences다. 다른 환경은 cold prepare에서 거부한다. 이 mode7의 Mistral 지원을 선언하지 않으며 Mistral의 기존 mode6는 유지한다.

## 실제 비용 감소

GPU6에서 Nsight Systems 2026.5.1로 같은 12개 요청/출력 240 tokens의 prefill+decode 구간을 계측했다. 두 arm 모두 report/SQLite export exit 0이다. 아래는 계측 kernel 시간이며 비계측 처리량이 아니다.

| kernel 분류 | 기존 ms | v2 ms |
|---|---:|---:|
| Gate/Up | 202.50 | 202.87 |
| Down | 389.55 | 144.43 |
| Q | 131.76 | 45.02 |
| O | 146.96 | 52.64 |
| LM head | 130.34 | 27.43 |
| K/V | 140.56 | 140.96 |
| 전체 kernel 시간 합 | 1467.06 | 938.74 |
| kernel span | 1672.99 | 1113.27 |

실제 NVTX/cuBLAS correlation에서 Down M=4, O M=8, Q M=8 및 LM head M=8 호출을 확인했다. Gate/Up과 K/V 비용은 거의 같고 바꾼 연산에서 큰 감소가 발생했다. 요청별 같은 weight를 읽고 호출하던 일을 묶은 효과다. DRAM bytes/L2 hit 하드웨어 카운터를 확보한 것으로 표현하지 않는다.

## 정확성과 자원 검증

- Q/O/Down: 실제 weight, 실제 입력·0·permutation/scale 변형 입력, 1..32행을 합쳐 **288개 비교, BF16 14,598,144개 값이 native M=1과 일치**했다.
- LM head: 같은 조건의 1..32 실제 행을 최대16행씩 나눠 **96개 비교, BF16 240,666,624개 값이 일치**했다.
- 전체 모델: HF **full-vocabulary 104 teacher rows**, 원래 native M=1의 208개 BF16 원시 파일과 byte-exact 비교했다. 실제 Graph key **1..32**, 변경 입력 순서·context에서 capture 2,304회와 첫 replay 2,304회를 확인했다. 이는 검증한 유한 조합이며 104 rows와 모든 shape의 Cartesian 전수 검사라는 뜻은 아니다.
- CPU 회귀 **518 통과, 기존 ignore 40개**다. cold-prepared variant를 `mut`로 준비하면서 기존 소스 문자열 검사 하나가 실패한 것은 같은 검사 조건을 유지하며 두 선언 형태를 인식하도록 수정했다. 이후 전체 검사를 다시 통과했다. C02 4개만 앞서 사용자가 승인한 private `/tmp` fixture 예외로 재검사했고 임시 트리를 삭제했다. 나머지 실행·모델·자료는 EFS다.
- 원시 HTTP/SSE 값·timing·22GiB 예산·KV 회수를 독립 재검사했다. 중간 후보/최종 성능 비교/최종 serving/profile를 합친 19개 arm의 전체 응답 값 **2,016건**을 확인했다. 이 합계를 최종 바이너리만의 요청 수로 표현하지 않는다. 성능 표의 최종 비교는 8개 arm, 768개 측정 요청이며 correctness/retry까지 합치면 900건이다.

최종 teacher C1 검사는 성공 종료 뒤 감시기가 이미 종료된 자신의 0MiB NVML context를 외부 PID로 오인하는 경합이 있었다. 원래 실패 receipt를 보존했고 PID/start ticks·전체 샘플·성공 종료·allocation/context close·BF16 파일을 독립 확인해 reconciliation receipt를 추가했다. 감시기는 검증된 자신의 PID가 `/proc`에서 사라지고 NVML이 0MiB일 때만 teardown으로 처리하도록 수정했다. 완료한 C1 수치 검사를 반복하지 않고 남은 Graph 검사만 수행했다.

## 소스 및 바이너리 결합

원격 root: `/mnt/efs/nlp/psyche/riley-sm80-projections-v2-20261019`.

최종 qualified 바이너리 SHA256은 `d339fe5d39b04983f09cbb36dce32b22c4ef45d77db9b2a8604ab1a38e22f884`다. mode7 통합·검증 시점의 935개 파일이 `guarded-source-manifest.json`과 모두 일치했다. 9개 변경 파일에는 명시적 CLI/runtime 모드, CUDA/FFI/GEMM 연결, cold workspace 소유·정리, 소스 계약 검사의 mut 인식이 포함된다.

성능 표는 LM 배치가 들어간 variant의 두 fresh-process 쌍이다. 최종 qualified 빌드와 그 variant의 차이는 **cold cuBLASLt120405 버전 검사 추가뿐**이며 실행 중 산술·dispatch는 같다. 최종 빌드 자체도 HF104/Graph1..32 검증과 별도 실제 C8 serving 48개 측정 요청을 통과했고 253.51 tok/s였다. 이 48개 짧은 구간을 96개 측정 arm과 합산하지 않는다. 최종 빌드의 실제 연산 감소도 위 Nsight 자료로 확인했다.

패치·적용 조건은 [candidate README](../benchmarks/candidates/sm80-bounded-projections-v2/README.md), compact proof는 [evidence](../benchmarks/evidence/20261019-sm80-projections-v2/)에 보존한다. 원시 SSE/SQLite/BF16/binary는 EFS에 남긴다. commit/push는 수행하지 않았다.

## 사용과 남은 범위

검증한 기존 owned-HF/iteration-batch 실행 인자에 아래 설정을 명시한다.

```text
--owned-decode-batch bounded-projections-sm80-v2
--sampling-backend gpu-greedy
--max-active-sequences 8
```

rollback은 `--owned-decode-batch bounded-gateup-sm80-v1`이다. 두 fresh-process screen은 C1/C8/C16/C32의 5회 확대 release matrix를 대신하지 않는다. 기본 승격과 vLLM parity를 선언하지 않는다. 이전 stock vLLM C8 약656.92 tok/s와 비교하면 이번 약259 tok/s도 격차가 남지만, 이전 약158 tok/s보다 그 차이를 크게 줄였다. 해당 stock 수치는 이번에 새로 paired 재측정한 값은 아니다.

GPU6/7은 모든 자신의 실험을 끝낸 뒤 기존 holder PID/context를 유지하며 각각 20GiB 대기 예약으로 복귀했다. 외부 작업은 종료하지 않았다. 최종 상태는 `final-hold-receipt.json`을 기준으로 한다.

후속 C16/C32 확대 검증과 native-order K/V mode8은 [확대 검증 보고서](qwen-c16-c32-expansion-20261019.md)를 기준으로 한다. mode7의 위 수치와 소스 manifest는 당시 variant의 기록이다.
