# Qwen3-4B: C16 TTFT p95와 prefill Gate/Up·Down v7

후속 decode Gate/Up·Down 개선과 fresh v7/v8/stock 비교는 [v8 보고서](qwen-decode-mlp-v8-and-vllm-20261020.md)에 기록했다.

상태: 구현·기존 정확도·C16/C32 비교·profile·보조 검사·독립 감사 완료. 내려받은 증거625개의 SHA256과 소스935파일을 재검증했다.

기존 native/HF 전체 vocab BF16 바이트 일치, 생성 token/text/usage/finish reason 및 원래 EOS를 유지한다. GPU6·7은 기존 holder PID2027914/2027915와 CUDA context를 유지한다. 테스트 중64MiB, 종료 후20GiB로 크기만 변경하고 해제하지 않는다. EFS 실행 원칙과 C02 fixture4개만 private `/tmp`를 허용한 예외를 유지한다.

## 변경

이전 v6는 decode Down·paged Attention을 개선했지만, prefill Down은 요청별 native GEMM을 반복했다. Gate/Up 병합도 native heuristic을 그대로 사용했다. 직전 C16 p95121.298→121.772ms는 개선되지 않았으며 후보의 p95 경계 요청6개는 초기 prefill이었다. 이는 추가 조사 근거이며 0.474ms 차이의 단독 원인 증명은 아니다.

실제 L0 가중치·입력으로 cuBLASLt 후보를 탐색하고 native M128 산술에 맞는 고정 recipe를 찾았다. 일반 Down GEMM을 큰 M에 적용하면 기존 BF16 기준이 달라지므로 그 경로를 사용하지 않는다.

| 경로 | 적용 범위 | 고정 recipe |
|---|---|---|
| prefill Down | original64..130 요청들을 merged161..576행으로 병합 | algo6/tile24/stages15/splitK5/reduction4/swizzle1/custom0 |
| prefill Gate/Up | 같은 original 범위, merged257..512행 | algo21/tile20/stages11/splitK0/reduction0/swizzle0/custom0 |

단독 긴 prefill은 기존 native 경로를 유지한다. Attention/KV/logits의 요청별 metadata를 병합하지 않는다. 기본 모드를 바꾸지 않고 opt-in `native-prefill-sm80-v5`만 확장한다. 내부 selector11은 public decode mode가 아니다.

Descriptor·workspace는 cold prepare에서 만들고 행별 AlgoCheck·최대16MiB workspace 제한을 확인한다. Dispatch는 입력/가중치/출력과 private workspace를 같은 사용 ledger에 등록한다. 기존 context·span·alias·capture 규칙과 close를 유지한다. CUDA12.4/cuBLASLt120405/SM80·108SM 및 unmodified workspace/SM 환경으로 제한한다.

## 구성요소 정확도

Actual L0 세 projection, M64..576,6상태의9234회 비교에서 BF16 값21,684,879,360개가 native M128과 모두 일치했다. 병합 전 originalM64..130의 native 경로도6상태로 별도 검사했다. 입력은 원본 seed,0, 변형 seed, dense signed 값, sparse 값, 다른 signed 변형 seed다. 유한 입력 검증이며 임의 모델·임의 입력의 수학적 동치 증명으로 표현하지 않는다.

## 서버 검증

이전 v6·후보 v7·stock vLLM을 같은 Qwen3-4B BF16, budget576, max sequence1024, 같은 prompt/cap/greedy/EOS, prefix cache off로 비교한다. C16/GPU6, C32/GPU7, fresh process3회,64 warm+192 measured requests/arm이며 실행 순서를 교대한다. Stock의 기본 compile/Graph를 유지한다. 전체 요청을 headline p95에 포함하고 초기 cohort를 제외하지 않는다.


## 비교 결과

같은 측정의 fresh process3회 중앙값이다. Riley baseline은 직전 v6 binary이며 과거 측정값을 섞지 않는다.

### C16

| 지표 | 이전 v6 | 후보 v7 | stock vLLM | Riley 변화 |
|---|---:|---:|---:|---:|
| 처리량 tok/s | 510.527 | 522.967 | 1142.452 | +2.44% |
| TTFT p50 ms | 78.351 | 78.426 | 42.341 | +0.10% |
| TTFT p95 ms | 119.382 | 107.558 | 82.035 | -9.90% |
| TTFT p99 ms | 251.083 | 224.360 | 109.287 | -10.64% |
| response-mean TPOT p99 ms | 32.102 | 30.154 | 13.305 | -6.07% |
| E2E p99 ms | 1046.705 | 984.856 | 437.992 | -5.91% |

| 반복 | TTFT p95 after/before | 처리량 after/before | stock measured token IDs 일치 |
|---:|---:|---:|---:|
| 0 | 0.913305 | 1.026040 | 172/192 |
| 1 | 0.887147 | 1.024368 | 174/192 |
| 2 | 0.885800 | 1.014294 | 174/192 |

후보 처리량은 같은 측정의 stock vLLM 대비45.8%다.

### C32

| 지표 | 이전 v6 | 후보 v7 | stock vLLM | Riley 변화 |
|---|---:|---:|---:|---:|
| 처리량 tok/s | 621.046 | 639.820 | 1668.236 | +3.02% |
| TTFT p50 ms | 99.546 | 94.582 | 53.235 | -4.99% |
| TTFT p95 ms | 391.974 | 353.399 | 164.621 | -9.84% |
| TTFT p99 ms | 538.664 | 486.766 | 217.897 | -9.63% |
| response-mean TPOT p99 ms | 61.163 | 56.037 | 21.568 | -8.38% |
| E2E p99 ms | 1849.249 | 1765.859 | 671.256 | -4.51% |

| 반복 | TTFT p95 after/before | 처리량 after/before | stock measured token IDs 일치 |
|---:|---:|---:|---:|
| 0 | 0.901589 | 1.030230 | 174/192 |
| 1 | 0.912933 | 1.034081 | 176/192 |
| 2 | 0.908552 | 1.023732 | 171/192 |

후보 처리량은 같은 측정의 stock vLLM 대비38.4%다.

C16·C32 모두3회씩 TTFT p95가 감소했다. C16 p50은78.351→78.426ms(+0.075ms)로 거의 같으며, 모든 분위수 무악화 gate를 통과했다고 표현하지 않는다. 처리량 증가는2.44%/3.02%로 제한적이다. 이번 목표인 tail 개선과 남은 decode 비용을 구분한다.

Riley는 각 반복의 측정192개가 모두 기존 native/HF 기준과 일치했다. Stock은 각 arm3840 output tokens를 처리했으며 ID 일치율은 표처럼 별도 보고한다. 이를 품질 점수 또는 stock 정확도 실패로 해석하지 않는다. TPOT는 response-mean이며 token ITL이 아니다.

이번 C16 전후3회의 p95 interpolation 경계 요청12건은 모두 측정 시작 후0.48~1.28ms에 접수된 초기 요청이었다. 이 workload의 C16 p95는 초기 prefill burst 비용에 민감하다. 요청별 Down 반복과 큰 prefill Gate/Up의 비용을 줄여 해당 tail을 낮췄으며, 이전0.474ms 차이의 유일 원인을 확정한 것으로 표현하지 않는다.

## Profile

일반 HTTP와 별도의64 warm+64 measured requests, 각 arm1280 output tokens다. C16/GPU6와 C32/GPU7의 CUDA/NVTX trace를 수집했다. Inspection preload는 Riley profile에만 사용하며 일반 HTTP·stock 성능 비교에는 사용하지 않는다. 아래 kernel 합은 CPU 시간과 별개이며 GPU span과도 다르다.

| Profile 지표 | C16 v6 | C16 v7 | C32 v6 | C32 v7 |
|---|---:|---:|---:|---:|
| GPU kernel 수 | 138327.000 | 136635.000 | 112041.000 | 109593.000 |
| GPU kernel 합 ms | 2420.534 | 2365.168 | 2072.218 | 2005.206 |
| GPU kernel span ms | 2744.591 | 2785.352 | 2372.978 | 2358.963 |
| prefill Gate/Up kernel 합 ms | 202.625 | 179.519 | 190.803 | 168.846 |
| prefill Down kernel 합 ms | 140.374 | 102.371 | 140.084 | 97.140 |
| 전체 Gate/Up kernel 합 ms | 522.795 | 499.480 | 511.467 | 489.018 |
| 전체 Down kernel 합 ms | 531.963 | 493.871 | 396.129 | 352.126 |

Prefill Down의 kernel 합은 C16 -27.1%, C32 -30.7%이며 prefill Gate/Up은 각각 -11.4%, -11.5%다. 같은64개 요청에서도 도착·완료 시점에 따라 실제 batch 구성이 달라지므로 profile 차이를 단일 kernel의 고정 shape 속도 향상률로 해석하지 않는다. 별도 구성요소의 고정 shape 비교와 함께 판단한다.

C16 profile span은2744.591→2785.352ms로1.49% 늘었다. Instrumented profile 전체 구간이 개선됐다고 주장하지 않는다. 일반 HTTP192요청3회에서는 C16 p95가 매회 감소했고 처리량도 매회 증가했다. 측정 구간 크기와 도구 영향을 포함한 별도 관측이다. DRAM/L2 counter는 수집하지 않았다.

## 정확도와 실행 소스

- 범위 내 회귀 검사는 기본517개 통과·40개 ignored, EFS 상위 경로 소유권 때문에 실패한 C02 fixture4개만 승인된 private `/tmp`에서 재검사해 통과했다. 합계521개 통과이며 임시 트리 제거도 확인했다. 검증 코드를 완화하지 않았다.
- C1 eager 및 R1~32 segments Graph 각각104개 HF 전체 vocab row·208개 BF16 파일이 기존 native/HF와 바이트 단위로 일치했다. R1~32 Graph 실행의 capture/첫 replay는 각각2304회다.
- fixture의128-token prefix를 복제한512행 병합은 새 Gate/Up·Down을,130-token prefix의520행 병합은 새 Down과 Gate/Up fallback을 검증한다.16·397-token prefix도 기존 경로를 유지한다.
- fused SiLU+bounded MLP Graph 조합도 R1~32·208개 BF16 파일이 모두 일치했다.1152 captures/첫 replay,2448 cache hits, allocation close를 확인했다.
- 빌드 입력935파일이 최종 로컬/EFS 소스와 일치한다. v6 위의6파일 layer patch는 reverse-check·whitespace 검사를 통과했다. 기존 dirty/untracked 작업을 보존했고 commit/push하지 않았다.
- v7 binary SHA256: `e3fc21feb8bed3d5e63440e75bbc1c8a159147b8ad8109eef9a9cbf00c04176d`. 이전 v6 baseline: `6e1c2a46227629110068ebcb7ce1c85185ace0f343b1cce1b28d75d91a16d574`.
- stock 설치 파일을 read-only로 확인했다. vLLM0.27.1+cu129·Torch2.13.0+cu129이며 Qwen3→Qwen2MLP의 merged Gate/Up·QKV 및 SiluAndMul 구현을 기록했다. CUDA12.4 Riley와의 소프트웨어 차이를 이번에 분리 실험하지 않았으며 격차의 특정 비율을 버전에 귀속하지 않는다.

## 보조 검사·독립 감사·GPU 홀드

C1/GPU7 및 C8/GPU6를 각2회·32 warm+96 measured requests/arm으로 확인했다. 각 동시성 안에서는 전후에 같은 GPU·CPU affinity를 사용했다.

| 동시성 | 처리량 v6→v7 tok/s | TTFT p95 v6→v7 ms |
|---:|---:|---:|
| C1 | 90.281→90.727 (+0.49%) | 22.819→22.624 |
| C8 | 364.305→365.477 (+0.32%) | 86.683→82.665 |

C8 처리량은 반복0에서 -0.86%, 반복1에서 +1.50%였다. C1·C8의 작은 변화로 전역 성능 개선을 주장하지 않는다. 선택한31개 HTTP/profile arm의 raw SSE5195개를 전체 response 값으로 재구성하고 usage·finish reason·token/text 및 분위수·throughput을 재검산했다. Riley는 원래 native oracle의 전체 값과 일치했다. Stock은 자체 response 구성·prompt token 수·보고된 ID 일치율을 별도 검사했다.12개 arm의 disconnect/retry 이후 allocation·KV 자원 회복과 quiescent gauge를 확인했다.

선택한 측정 구간의 foreign GPU PID는0개이며 own+holder peak14236MiB로22528MiB 한도 안이었다. 모든 GPU 작업 종료 후 GPU6 PID2027914/ticks6083215119 및 GPU7 PID2027915/ticks6083215121이 각각20GiB allocation의 idle 홀드로 복귀했다. CUDA context를 중단·재시작하지 않았고 외부 작업을 종료하지 않았다. 홀드는 물리 GPU의 배타 잠금이 아니며 새 외부 프로세스가 나타나면 admission/관측 gate를 따른다.

## 남은 vLLM 격차와 우선순위

C32 stock profile은32986 kernels, kernel 합855.173ms, span967.780ms, cudaGraphLaunch41회였다. Riley v7은109593 kernels,2005.206ms,2358.963ms다. Stock의 익명 GEMM을 kernel 이름만으로 Gate/Up·Down에 배분하지 않는다. Stock source의 merged linear·SiluAndMul과 실제 Graph trace는 확인했지만 각 기능이 전체 격차의 몇 %인지 분리한 실험은 아니다.

C32 v7의 잔여 kernel 합은 Gate/Up489.018ms(24.4%), Down352.126ms(17.6%), KV projection226.937ms(11.3%), QK/PV206.984ms(10.3%)다.

1. **Decode Gate/Up:** prefill을 제외해도320.172ms가 남는다. 현재 최대16행 native GEMM을 반복한다. 같은 SM80 trace의 decode family는 `64x64_sliced1x2 ... stages_64x5_tn`이다. 다음 후보는 실제17~32행을 한 번에 처리하면서 native sliced/rounding 순서를 유지하는 고정 Lt 또는 전용 kernel이다. 단순 큰-M/merged-N 전환을 정확도 검증 없이 적용하지 않는다.
2. **Decode Down·긴 prefill:** native-order decode kernel은201.884ms가 남는다. 현재16행 tile의32행 재사용 개선이 후속 후보이며 미측정이다. 이번 prefill Down은 original64..130만 병합한다.397행 같은 단독 긴 prefix와 범위 밖 입력은 native 그대로이므로 별도 arithmetic-family qualification이 필요하다.
3. **KV·Attention·launch 구성:** KV projection과 QK/PV가 여전히 합21.6%다. 기존 paged chunk의 BF16 score/scale/probability/context 경계를 지키며 묶음 launch·주소 재사용을 추가로 평가해야 한다. 이번 layer에서 해당 산술을 바꾸지 않았다.

이번 변경은 C16 p95 악화를 해결하고 prefill Gate/Up·Down 비용을 줄였지만, stock 처리량과의 격차는 남는다. 기본 모드 승격이나 정확도 허용오차 완화는 하지 않았다.3회 finite loopback 결과를 임의 모델·입력 또는 production SLO의 보장으로 해석하지 않는다.

## 검토·재현 자료

- [6파일 layer patch](../benchmarks/candidates/sm80-prefill-v7/candidate.patch), [layer manifest](../benchmarks/candidates/sm80-prefill-v7/source-layer-manifest.json), [적용 범위](../benchmarks/candidates/sm80-prefill-v7/README.md). Clean HEAD가 아니라 검증된 v6 snapshot 위의 layer다.
- [전체 SSE 독립 감사](../benchmarks/evidence/20261020-prefill-v7/independent-audit.json), [profile 요약](../benchmarks/evidence/20261020-prefill-v7/profile-summary.json), [C16 p95 경계 요청](../benchmarks/evidence/20261020-prefill-v7/c16-tail-audit.json). 초기 cohort를 제외하지 않았다.
- [구성요소 감사](../benchmarks/evidence/20261020-prefill-v7/component-audit.json), [전체 vocab 감사](../benchmarks/evidence/20261020-prefill-v7/teacher-audit.json), [MLP Graph 조합 감사](../benchmarks/evidence/20261020-prefill-v7/mlp-compatibility-audit.json).
- [source/binary binding](../benchmarks/evidence/20261020-prefill-v7/source-binary-binding.json), [설치 stock source](../benchmarks/evidence/20261020-prefill-v7/stock-installed-source-proof.json), [GPU 홀드 증거](../benchmarks/evidence/20261020-prefill-v7/terminal-hold-proof.json), [raw EFS index](../benchmarks/evidence/20261020-prefill-v7/raw-artifact-index.json).
- [625파일 SHA256](../benchmarks/evidence/20261020-prefill-v7/checksums.json). EFS raw root: `/mnt/efs/nlp/psyche/riley-prefill-v7-20261020`. Full raw SSE도 compact evidence에 포함하며, 큰 BF16·Nsight·모델·binary는 EFS에 유지한다.

압축파일5308685B의 SHA256은 `83c077afaecef1dcd9ef6990fd55f3282201bcaa255e5712a889f3791b969524`다. GPU 실행은 기존 `control.py --lease-dir .../leases-v2 --index 6|7 run -- ...`를 사용하며, 각 campaign의 실제 argv·runner SHA·binary SHA·CPU affinity·startup·raw SSE·GPU sample과 재현 helper를 증거에 보존했다.
