# Decode QK·PV v16B 전체 모델·C16/C32 검증

2026-10-21 KST. 기준 runtime은 v15C(`57502e7a0`)다. **정확도는 통과했지만 처리량 향상이 작고 paired tail 경고가 남아 후보를 채택하지 않는다.** v16A의 큰 Gate/Up·Down·KV 구조 후보 탈락 결과와 함께 보존한다. 후속 v16C는 이 attention 후보 위에 별도 prefill Gate/Up 통합을 검사하며, 이 결과만으로 기존 runtime을 바꾸지 않는다.

## 같은 캠페인의 정상 서버 비교

각 concurrency에서 기존 v15C·후보·stock vLLM을 순서를 교차해 각각5개 새 프로세스로 실행했다. 아래는 각 run metric의 중앙값이며, 서로 다른 캠페인의 수치를 합치지 않았다.

| C | v15C tok/s | 후보 tok/s | 변화 | vLLM tok/s | 후보/vLLM |
|---:|---:|---:|---:|---:|---:|
| 16 | 729.296 | 738.407 | +1.25% | 1130.970 | 65.29% |
| 32 | 973.861 | 986.968 | +1.35% | 1672.321 | 59.02% |

| C | TTFT p95 기존→후보 | TPOT p95 기존→후보 | TPOT p99 기존→후보 | E2E p95 기존→후보 |
|---:|---:|---:|---:|---:|
| 16 | 80.565→81.736 | 20.572→20.207 | 21.659→20.874 | 677.880→668.380 |
| 32 | 252.431→249.485 | 31.863→31.408 | 38.130→38.557 | 1013.074→1001.141 |

단위ms. TPOT은 요청별 평균 output-token 지연의 percentile이며 token-level ITL percentile이 아니다. paired TTFT/TPOT p95/p99 +2% 초과 경고는 다음과 같다. 유리한 중앙값으로 이 반복들을 숨기지 않는다.

- C16: r0 ttft_ms_p95 +2.13%; r1 ttft_ms_p95 +4.42%; r4 ttft_ms_p95 +4.74%; r4 ttft_ms_p99 +3.83%.
- C32: r1 ttft_ms_p95 +3.57%; r1 ttft_ms_p99 +2.39%; r4 tpot_ms_p99 +11.52%.

[전체5회 수치](../benchmarks/evidence/20261021-mlp-decode-v16b/final-statistics.json)와 압축 안의 raw SSE/명령/자원 표본을 보존한다. C16 stock r1의 낮은 처리량·높은 TTFT 표본도 제외하지 않았다. 완료한 정상30 arms 중 제외0개.

## 적용한 후보와 정확도 범위

QK native serial16/tree8 누적 순서를 유지하며 shared reduction/barrier를 warp shuffle로 바꿨다. PV는 기존 depth tile을 맡은4개 warps를 CTA128로 묶고 warp별 private staging을 사용했다. softmax·BF16 경계·page layout·컨텍스트 범위·launch3개는 유지했다. 실제 서버 dispatch proof는 profile 원자료에 따로 남긴다. 구성요소 hot Graph의 약11~26% 개선이 서버의 같은 비율 개선을 뜻하지 않는다.

독립 source937/빈 target 빌드 후 원래 native/HF full-vocabulary BF16 4,480파일의 literal equality, CPU524 pass/40ignored, 실제 GPU 계약5개(기존 alignment120경우 포함)를 통과했다. eager104HF rows, all-R Graph104rows, natural128 body1024rows, mixed output1024rows, no-output prefill992rows 범위다. 모든104HFrows×32R Cartesian 증명이나 새 early-EOS coverage로 확장해 주장하지 않는다. GPU runtime observer는 원래 v18 SHA를 그대로 사용했다.

4개 C02 테스트의 private 임시 fixture만 사전 허용한 `/tmp`에서 재검사하고 삭제했다. 모델·스크립트·벤치마크·빌드·scratch는 EFS에 유지했다.

## 비교 조건과 보존

Qwen3-4B BF16 greedy, token budget576, max sequence1024, KV blocks=C×64, prefix cache off. 12 fixed prompts115..130tokens, output limits8/16/24/32가 균형을 이룬64warm+192keep, arm당3840 output tokens. Riley는 owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1; stock은 vLLM0.27.1+cu129/Torch2.13+cu129 FA2/FULL_AND_PIECEWISE다. stock은 correctness reference가 아니다. 서버/부하 CPU pinning을 분리했지만 노드 전체 CPU·I/O 격리는 아니다. 일부 CPU compile이 초기 정상 비교와 겹쳤다는 한계를 남긴다.

GPU6에 새로 나타난 허용 범위 밖의 외부 작업은 중단하거나 controller 예외로 허용하지 않았다. 모든 실제 유한 GPU 작업은 GPU7에서 직렬 실행했다. GPU6·7의 원래 holder PID2027914/2027915와 start ticks6083215119/6083215121은 유지하며 해제·재시작하지 않았다.

초기 GPU6 진입 거절, controller 중복 실행 거절, 잘못된 freeze baseline 경로로 child 실행 전 실패한 시도는 raw 로그에 그대로 보존한다. 정상 비교는 명시적으로 v15C binary SHA233c3a6a… baseline을 고정한 r4이며, 최종 source/binary/normal argv/runner hash를 재검사했다.

[압축 원자료](../benchmarks/evidence/20261021-mlp-decode-v16b.zip) · [증거 목록·검증](../benchmarks/evidence/20261021-mlp-decode-v16b/README.md) · [v16A 구조 후보](qwen-mlp-decode-v16a-20261021.md). runtime patch는 v16A prototype 및 압축 안 candidate.patch로 재현한다.

## 구성요소 개선이 서버에 작게 나타난 이유

별도 C32 before/after trace의 kernel-duration 합계는1322.983/1329.129ms다. 기준 trace에서 Gate/Up358.624ms, Down193.363ms, KV projection171.511ms로 합계54.69%이며 attention QK/PV139.662ms는10.56%다. 후보 trace도 Gate/Up367.326ms, Down199.565ms, KV172.606ms로55.64%, attention128.224ms로9.65%다. 작은 attention 부분만 개선하는 후보이므로 전체 처리량을 같은11~26%만큼 높일 수 있다는 주장은 근거가 없다.

실제 candidate PV는grid(2,16,R)/block(128,1,1), QK/softmax/PV 각각2232회로 관측됐다. 후보 미실행이 아니다. 다만 before의 decode QK/PV는2196회였고 Gate/Up 등의 count도 다르다. 서버의 도착·배치 구성에 따라 실제 work가 달라지므로 이 profile aggregate 차이를 고정 작업량의 kernel speedup이나 정상 HTTP 속도로 계산하지 않는다. trace는 정확도 검사와64warm 이후 별도64keep 구간만 Nsight start/stop으로 포착한 진단이다. 정상192keep×5회와 분리하며 profiling overhead를 포함한다. DRAM/L2/SM hardware counter는 확보하지 못했으며 메모리 bandwidth 포화라고 확정하지 않는다.

v16A에서 KV의 exact FP32 expansion·전치·행 병렬화는 느렸고 Tensor Core zero-inserted 재현은 BF16 원소1개 차이로 탈락했다. mixed Gate/Up은 각 native 누적 family를 보존했지만 큰 CTA와 staging 비용으로 느렸다. Down 긴 K pipeline도 이득이 없었다. 따라서 동일한 작은 tile 변경을 더 겹치기보다 실제 큰 비중의 prefill Gate/Up을 library merged GEMM으로 줄이는 v16C를 따로 검증한다. 새 수치 허용오차를 도입하지 않는다.
