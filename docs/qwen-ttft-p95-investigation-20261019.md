# Qwen TTFT p95 원인 확인 및 GPU greedy 개선

2026-10-19 KST. 앞선 [W0 보고서](a100-execution-path-and-bottleneck-20261019.md)의 Qwen C8 TTFT p95 악화를 별도로 조사했다. 기존 수치 기준은 유지하고 GPU 6/7 holder는 종료하지 않는다.

## 기존 악화 수치의 범위

이전 GPU7 단독 실행은 arm당 32 warmup + 48 측정 요청이었다. 기준선 `lt-mlp-v2` 대비 Gate/Up 후보에서 TTFT p95 중앙값이 153.70 → 157.91ms로 약 2.7% 악화했다. 그러나 요청별 원본을 재대조하면 p95를 결정한 요청들은 두 반복 모두 최초 8개 요청에 몰려 있었다. 최초 8개를 분리한 이후 요청들의 TTFT p95는 다음과 같다. 이 분리는 원인 진단이며 전체 p95를 대체하는 채택 기준이 아니다.

| 반복 | 기준선 이후 요청 p95 ms | Gate/Up 이후 요청 p95 ms |
|---|---:|---:|
| 0 | 145.37 | 125.82 |
| 1 | 150.58 | 127.55 |

최초 8개 요청은 client thread의 HTTP 도착 순서가 달랐다. 원본 SSE의 서버 request ID와 prompt token 수를 보면 반복 0의 기준선은 최초 prefill 묶음에 499 tokens가 들어간 것으로 추정되고, 후보는 512 tokens가 정확히 들어간 것으로 추정된다. iteration token 예산은 512이고 이미 실행 중인 decode가 먼저 예산을 사용한다. atomic prefill은 남은 예산에 안 들어가는 prompt를 잘라 넣지 않고 다음 iteration으로 넘긴다. 따라서 동일한 8개 입력도 도착 순서에 따라 후속 prefill 묶음이 3개 또는 4개가 된다. 실제 iteration trace에서 이 구성을 별도로 확인한다.

기존 반복 0에서는 가장 느린 최초 응답이 기준선 229.06ms, 후보 158.44ms였다. 후보에서 더 많은 요청이 두 번째 prefill로 이동하면서 최댓값은 줄었지만 p95가 걸린 순위의 값은 약 4ms 커졌다. 따라서 이 수치를 같은 prefill 구성의 kernel 자체가 느려졌다는 근거로 사용할 수 없다. 짧은 closed-loop workload의 초기 burst·도착 순서·유한 표본의 순위 효과가 섞인 지표다.

## 개선한 실행 설정

Gate/Up 후보의 BF16 연산과 기존 기본값을 유지하고, 이미 구현돼 있는 **`--sampling-backend gpu-greedy`를 명시적으로 사용**했다. 런타임 연산 코드를 추가로 바꾸거나 sampling 정확성 기준을 완화하지 않았다. temperature 0, repetition penalty 1, finish-token mask 등 기존 eligibility 검사를 유지한다. 지원하지 않는 요청은 기존 CPU 선택 규칙을 따른다. 이번 결과는 검증한 greedy 요청 범위에 한정한다.

같은 바이너리 `844dde2bac465c19f7540c6546575808a6389466ffed9813915c345be2cd4391`에서 Gate/Up mode6 + CPU 선택과 mode6 + GPU 선택을 비교했다. 원격 실행 root는 `/mnt/efs/nlp/psyche/riley-qwen-ttft-20261019`다. GPU7 단독, 동일 affinity 32-39, BF16, Graph off, iteration-batch completion, C1/C8 각각 AB/BA 두 fresh-process 반복이다. arm당 32 warmup + **96 측정 요청**, output tokens는 각 1,920으로 같다. 이전 48개 실행과 합치거나 그 개선율로 직접 환산하지 않는다.

| 항목 | mode6 + CPU | mode6 + GPU greedy | 변화 |
|---|---:|---:|---:|
| C8 throughput tok/s | 153.52 | 158.57 | +3.3% |
| C8 TTFT p50 ms | 109.27 | 105.86 | -3.1% |
| C8 TTFT p95 ms | 124.63 | 121.47 | -2.5% |
| C8 TTFT p99 ms | 148.84 | 149.62 | +0.5% |
| C8 응답 평균 TPOT p99 ms | 52.29 | 49.81 | -4.8% |
| C8 E2E p99 ms | 1653.24 | 1599.41 | -3.3% |
| C1 throughput tok/s | 84.87 | 85.45 | +0.7% |
| C1 TTFT p95 ms | 23.237 | 23.186 | -0.2% |
| C1 TTFT p99 ms | 23.293 | 23.447 | +0.7% |

표는 각 arm quantile의 두 반복 중앙값이다. C8 TTFT p95는 반복별 **127.73 → 124.40ms**, **121.52 → 118.54ms**로 두 번 모두 감소했다. 처리량은 각각 +3.63%, +2.96%다. 다만 C8 TTFT p99는 반복 0에서 +4.66%, 반복 1에서 -3.53%였고 C1 p99도 일관된 개선이 아니다. **p95 개선을 확인한 제한적 opt-in 설정이며, 모든 tail 무회귀 또는 기본 승격을 선언하지 않는다.** 두 반복은 확대 release matrix를 대신하지 않는다.

## 정확성과 실제 경로 검증

측정 요청 768개와 별도 correctness 요청 120개의 token/text/prompt/finish/usage 전체를 기존 같은-A100 eager 참조와 대조했다. correctness에는 원래의 natural-EOS 사례도 포함하며 token budget이나 EOS를 바꾸지 않았다. disconnect 후 재시도 12개까지 포함해 **전체 값 900건**을 독립 재대조했다. 8개 arm 모두 active/waiting/KV가 0으로 회수됐고 allocation 불변을 통과했다. holder를 포함한 NVML process memory 최대치는 10,712MiB로 22GiB 예산 이하였다. 외부 GPU PID는 없었다. 원본 요청에서 tok/s와 quantile을 독립 재계산했다.

HF full-vocabulary 104 rows/model 및 Graph 수치 증거는 이전 Gate/Up 소스에 대한 그대로의 증거다. GPU greedy가 모든 모델·sampling·Graph 조합에서 새로 full-vocabulary 검증됐다는 뜻으로 확대하지 않는다. 이번 변경은 기존 forward 뒤 token 선택 경로를 선택하는 옵션이며, Graph off의 전체 생성 값 일치는 이번 실제 HTTP 경로에서 별도로 확인했다.

현재 문서의 비계측 수치는 GPU7 단독 arm만 사용한다. GPU6 진단에는 사용자가 허용한 작은 작업이 공존했고, 식별되지 않은 별도 CUDA PID 등장 시 진단 arm만 중단했다. 외부 작업은 종료하지 않았으며 실패 결과도 보존한다. 계측 처리량은 위 표에 포함하지 않는다.

## 적용 및 rollback

검증한 SM80/108SM, CUDA 12.4/cuBLAS 120405, Qwen3-4B 환경에서 다음 옵션 조합을 명시적으로 사용한다. 나머지 검증 조건은 위 benchmark plan/argv와 동일해야 한다.

```text
--numerics-profile owned-hf-eager-bf16-v1
--execution-completion iteration-batch
--owned-decode-batch bounded-gateup-sm80-v1
--sampling-backend gpu-greedy
--max-active-sequences 8
--max-sequence-tokens 1024
--max-output-tokens 128
--kv-blocks 512
```

`--sampling-backend cpu`로 되돌리면 이번 선택 경로 개선을 끌 수 있다. 기존 기본값은 바꾸지 않는다. 홀드 controller의 `run`으로 실행해 테스트 중 예약만 줄이고 종료 후 20GiB 예약을 복구한다. holder PID/context는 계속 유지한다.

## 선택 경로의 실제 효과

GPU6에서 Nsight Systems 2026.5.1로 같은 12개 요청/출력 240 tokens를 계측하고 두 arm 모두 SQLite export exit 0을 확인했다. 이 자료에는 승인된 작은 외부 작업이 공존하므로 성능 채택용 시간으로 사용하지 않고 경로·복사량 확인에 사용한다.

| 계측 항목 | CPU 선택 | GPU greedy |
|---|---:|---:|
| D2H 복사 횟수 | 100 | 100 |
| D2H 복사 bytes | 72,930,880 | 3,520 |
| GPU BF16 argmax kernel | 없음 | 50회 |

GPU 선택에서는 전체 vocabulary logits 대신 선택 결과를 읽어 오며 D2H bytes가 약 99.995% 줄었다. H2D는 약 195.74MB로 거의 같았고 모든 복사가 제거된 것은 아니다. 이는 실제 GPU argmax 경로가 실행됐다는 근거이며 D2H 절감율을 전체 속도 개선율로 해석하지 않는다. 전체 성능 개선율은 위 GPU7의 비계측 비교로만 판단한다.

## 원인 교차 확인과 종료 상태

GPU7의 단독 iteration trace에서도 동일한 8개 입력의 도착 순서에 따라 기준선은 **4+4 prefill**, Gate/Up 후보는 **4+3+1 prefill**로 나뉘었다. 이 진단 실행은 앞선 48개 실행과 다른 도착 순서이며 구성 효과의 재현 근거다. 기준선은 503 rows(4 prefill), 이어 511 rows(4 prefill+4 decode)로 실행됐고 두 번째 step의 GPU 시간은 92.24ms였다. 후보는 501 rows, 388 rows(3 prefill+4 decode), 132 rows(1 prefill+7 decode)로 나뉘었고 후속 GPU 시간은 72.46ms와 55.95ms였다. 여기에서도 두 arm의 quantile은 서로 다른 iteration 구성의 영향을 받는다. 계측 수치는 비계측 성능 표와 합치지 않는다.

따라서 기존 p95 악화의 직접적인 설명은 **HTTP 도착 순서에 따른 atomic prefill packing 차이와 초기 burst가 quantile 순위에 미친 영향**이다. 현재 자료로 Gate/Up의 동일 구성 kernel 자체가 느려졌다고 단정하지 않는다. 초기 burst·steady state·p95·p99를 따로 보존하고, 개선안은 같은 96개 조건으로 새로 대조했다.

최종적으로 모든 자신의 실험 controller는 terminal 상태이며, GPU6/7 holder는 원래 PID **2027914/2027915**와 context를 유지한 채 각 **20GiB 예약**으로 복귀했다. GPU6의 기존 작은 작업은 그대로 보존했다. 원시 자료는 EFS 실행 root에 남겼고 compact receipt는 [TTFT evidence](../benchmarks/evidence/20261019-qwen-ttft/)에 보존한다. 새 산술 후보를 적용하지 않았으므로 제품 런타임 소스와 독립 빌드의 935-file manifest는 이전 검증 그대로이며 CPU 회귀 518개를 다시 실행한 것으로 표현하지 않는다.


후속으로 실제 projection 배치를 확대한 [SM80 v2 보고서](qwen-sm80-projection-batching-v2-20261019.md)에 더 큰 개선과 최종 소스 검증을 기록했다. 이 문서의 약3% 옵션 개선과 구분한다.
