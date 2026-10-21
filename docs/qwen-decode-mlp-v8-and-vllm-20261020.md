# Qwen3-4B: decode Gate/Up·Down v8와 stock vLLM 재검증

후속 초기 prefill·Down·launch/동기화 개선과 fresh v8/v9/stock 비교는 [v9 보고서](qwen-initial-prefill-and-launch-v9-20261020.md)에 기록했다.

상태: 구현·기존 정확도·C16/C32 fresh3회 비교·C1/C8 보조 검사·profile·독립 감사 완료. 증거680파일과 현재 소스935파일의 SHA256을 재검증했다. 구성요소 시간과 서버 전체 처리량을 구분한다.

기존 native/HF 전체 vocab BF16 바이트 일치와 token IDs·text·usage·finish reason·원래 EOS를 유지한다. GPU6·7의 원래 holder PID2027914/2027915 및 CUDA context를 유지하며, 실행 중64MiB·대기 중20GiB로 크기만 변경한다. 모델·빌드·벤치마크·증거는 EFS에 유지하고, 기존에 승인된 C02 fixture4개만 private `/tmp`에서 검사 후 제거한다.

## 실제 병목과 선택

이전 v7 C32 profile에는 decode Gate/Up320.172ms, native-order decode Down201.884ms가 남았다. 이번 변경은 해당 decode 경로를 대상으로 한다. 이전 v7 prefill Gate/Up·Down 및 paged Attention의 산술을 바꾸지 않는다.

Gate/Up의 일반 cuBLASLt 탐색900후보/투영에서는 R32에서 native M1과 일치한 후보가 없었다. 이는 모든 가능한 Lt recipe에 대한 불가능 증명이 아니다. 전용 WMMA 경로에서 native의 K32 sliced 누적과 BF16 epilogue를 유지하며 여러 독립 행을 한 launch에 처리한다.

| 경로 | 적용 | tile·공유 메모리·pipeline | 유지한 산술 |
|---|---|---|---|
| decode Gate/Up | 실제17~32행 | N64/R16, leading80, K64 두 stage | K32 교차 FP32 accumulator2개→FP32 합·native epilogue→BF16 |
| decode Down | 실제5~32행 | N64/R16, leading72, K64 네 stage | split-K2 각각 K32 교차 FP32 accumulator2개→BF16 partial→FP32 split 합→BF16 |

32행 tile도 측정했지만 선택한16행 tile보다 느려 제외했다. Gate/Up1~16행과 Down1~4행은 기존 경로를 유지한다. Shared memory는 Gate25600B·Down46080B로48KiB 안이다. Down은 기존 private workspace와 use/replay ledger를 재사용하며 새 device buffer를 추가하지 않는다.

Cold prepare에서 선택한 native SM80 image를 확인하고 private flag를 설정한다. 기존 SM80·108SM·CUDA12.4·cuBLAS120405·unmodified workspace/MPS 및 context·span·alias·alignment 검사를 유지한다. 기본 모드는 바꾸지 않고 opt-in `native-prefill-sm80-v5`의 startup에 v8 선택을 표시한다. 이번 layer는 v7 snapshot 위의 커널과 startup 표시2파일이다.

## 구성요소 정확도와 속도

Actual L0 가중치·원본 seed, R1..32와6상태를 비교해 Gate/Up 각각30,818,304개, Down8,110,080개, 총69,746,688 BF16 값의 차이0을 확인했다. 총576회 비교이며 허용오차를 추가하지 않았다. 원본 seed·0·signed 변형·deterministic finite BF16·큰 signed 값·작은 signed 값을 포함한다. 유한 입력 검증이며 임의 모델/입력에 대한 동치 증명으로 표현하지 않는다.

실제 v7 dispatch와 같은 프로세스에서 baseline/candidate 순서를 교대한3회 CUDA-event 중앙값:

| 경로/행 수 | 이전 v7 µs | v8 후보 µs | 선택 |
|---|---:|---:|---|
| Gate16 | 44.445 | 50.193 | 기존 유지 |
| Gate17 | 88.750 | 51.043 | 적용 |
| Gate32 | 88.981 | 51.816 | 적용 |
| Up32 | 88.769 | 51.700 | 적용 |
| Down4 | 46.022 | 58.514 | 기존 유지 |
| Down5 | 91.332 | 59.459 | 적용 |
| Down8 | 91.863 | 58.837 | 적용 |
| Down16 | 114.250 | 59.304 | 적용 |
| Down32 | 133.201 | 68.418 | 적용 |

초기 Down 연구 helper의 pageable H2D는 default stream, 계산은 nonblocking stream이어서 입력 업로드 순서를 보장하지 않았다. Zero 입력 진단에서 일부 이전 값이 남는 현상을 확인했고 해당 실험을 정확도·성능 결론에서 제외했다. 받아들인 helper는 모든 H2D를 계산 stream에 명시적으로 enqueue한다. [NVIDIA의 API 동기화 문서](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)는 pageable H2D 호출 반환 시 DMA가 끝나지 않을 수 있음을 설명한다. Riley 제품의 업로드 경로를 수정한 것은 아니며, invalid 로그와 ordered 재검사 모두 보존한다.

## 서버 검증 조건

- Qwen3-4B BF16, A100-SXM4-40GB, same GPU의 v7·v8·stock vLLM을 비교한다.
- 세 엔진 모두 budget576, max sequence1024, KV block16/blocks=C×64, prefix cache off, 같은12개 prompt·8/16/24/32 cap·greedy sampling·원래 EOS다.
- C16/GPU6 및 C32/GPU7, CPU24~31/32~39, fresh process3회·64 warm+192 measured requests/arm, 실행 순서를 교대한다.
- Stock의 기본 compile/Graph를 유지하고 event-spin·inspection preload를 성능 비교에 사용하지 않는다. Native 원본 oracle은 기존 eager binary에 묶고, 새 성능 baseline만 v7로 바꾼다.
- 전체 요청을 p95에 포함하고 초기 cohort를 제거하지 않는다. TPOT는 response-mean이며 token ITL이 아니다. Profile은 별도의64 warm+64 measured 진단 구간이다.

## C16/C32 fresh 비교 결과

각 지표는 같은 lane에서 새로 실행한3회 결과의 중앙값이다. 이전 보고서의 수치를 baseline으로 재사용하지 않았다. 단위는 처리량 output tok/s, 지연 ms다. percentile을 합쳐 다시 계산한 pooled percentile은 아니다.

| 동시성 | 엔진 | 처리량 | TTFT p50/p95/p99 | TPOT p50/p95/p99 | E2E p50/p95/p99 |
|---|---|---:|---|---|---|

| C16 | v7 | 522.831 | 76.735/109.152/226.392 | 27.200/29.301/30.154 | 534.580/946.412/1002.589 |
| C16 | v8 | 566.788 | 72.884/109.186/223.797 | 24.989/27.080/28.228 | 487.851/872.038/925.726 |
| C16 | stock vLLM | 1121.457 | 45.760/82.698/109.883 | 11.707/12.838/13.394 | 260.741/428.137/457.699 |
| C32 | v7 | 643.552 | 94.620/359.498/493.964 | 45.169/47.967/58.447 | 831.138/1528.126/1742.489 |
| C32 | v8 | 718.490 | 85.246/345.379/470.136 | 40.445/43.565/52.653 | 749.671/1369.778/1611.935 |
| C32 | stock vLLM | 1676.630 | 51.568/165.421/219.050 | 15.736/17.463/21.575 | 338.529/555.857/666.922 |

- C16: 처리량 **+8.41%**, stock 대비 46.62%→**50.54%**. TTFT p95 109.152→109.186ms (+0.031%).
- C32: 처리량 **+11.64%**, stock 대비 38.38%→**42.85%**. TTFT p95 359.498→345.379ms (-3.927%).

C16 p95는 **개선으로 판정하지 않는다**. 반복별 v7→v8은109.152→108.697,114.226→110.863,107.509→109.186ms로2회 개선·1회 악화했다. 중앙값 차이+0.034ms는 매우 작지만, 통계적 동등성을 증명한 것도 아니다. C32는359.498→345.379,351.848→341.591,359.938→346.329ms로3회 모두 개선했다. 초기 cohort를 포함한 전체192개를 그대로 집계했다.

C16 p95 interpolation 양쪽 요청은6개 arm 모두 측정 시작 후 약0~1.2ms 안에 들어온 초기 cohort이며 prompt125~130 tokens다. 이는 남은 tail이 초기 prefill·대기 구간에 걸쳐 있음을 보여주며, 이번 decode 처리량 개선만으로 p95가 내려간다고 볼 수 없다. 초기 요청을 제거하거나 별도 steady-state percentile을 headline으로 쓰지 않았다.

### 반복별 처리량과 TTFT p95

| 동시성·반복 | v7 tok/s·p95 ms | v8 tok/s·p95 ms | stock tok/s·p95 ms |
|---|---:|---:|---:|
| C16 · 0 | 527.164 · 109.152 | 566.977 · 108.697 | 1121.457 · 82.698 |
| C16 · 1 | 518.786 · 114.226 | 566.788 · 110.863 | 1143.188 · 83.335 |
| C16 · 2 | 522.831 · 107.509 | 563.410 · 109.186 | 1110.224 · 82.011 |
| C32 · 0 | 646.234 · 359.498 | 714.008 · 345.379 | 1681.888 · 164.227 |
| C32 · 1 | 642.124 · 351.848 | 719.546 · 341.591 | 1676.630 · 167.999 |
| C32 · 2 | 643.552 · 359.938 | 718.490 · 346.329 | 1674.691 · 165.421 |

## 프로파일에서 확인한 남은 격차

별도의64 warm+64 measured 진단 trace다. 아래 시간은 trace 안의 kernel-duration 합이며 전체 HTTP 지연이나 순수 decode 시간과 같지 않다. 독립 실행의 batch 배치와 kernel 수가 달라 정확한 절감률은 정상 HTTP3회 결과로 판단한다. CPU API 대기와 GPU kernel 시간은 겹치므로 더하지 않는다.

| 진단 kernel 분류 | C16 v7→v8 ms | C32 v7→v8 ms |
|---|---:|---:|
| Gate/Up 전체 | 502.717→505.185 | 488.789→395.435 |
| Down 전체 | 494.555→313.437 | 352.331→239.257 |
| K/V projection | 298.405→294.952 | 228.019→228.886 |
| QK/PV | 240.997→244.625 | 206.709→208.620 |
| RMSNorm | 177.899→179.435 | 163.653→163.238 |
| KV gather | 8.637→8.649 | 8.966→9.117 |

C16 trace에 새 Down3240회, C32에 새 Down1944회·Gate/Up2664회가 실제 등장했다. Before/stock에는 새 커널이 없다. C16에서 Gate/Up 새 경로가 안 나오는 것은17행 이상만 선택하기 때문이다. R1..16 native Gate/Up과 R1..4 Down은 현재 후보보다 빨라 남겨 두었다.

C32 v8의 Gate/Up395.435ms+Down239.257ms는 kernel 합1806.763ms의35.13%로 여전히 우선 병목이다. K/V228.886ms, QK/PV208.620ms, RMSNorm163.238ms도 남아 있다. 단순 KV gather9.117ms는 약0.50%라 이를 제거하는 것만으로 큰 처리량 격차를 설명할 수 없다. DRAM/L2 counter를 수집하지 않았으므로 대역폭 병목이라고 단정하지 않는다.

Stock C32 trace는32290 kernels/844.120ms, Riley v8은107189/1806.763ms다. Stock에는 실제 CUDA Graph launch38회와 FlashAttention split-KV가 보인다. Riley는 일반 성능 모드에서 Graph off이며 stream synchronize816회, 직접 kernel launch107189회다. 이는 launch·동기화·융합 범위의 격차가 남았다는 근거다. Stock의 익명 GEMM을 특정 Gate/Up·Down에 추측으로 배정하지 않는다. 계측 자체와 실행 배치 차이가 있어 이 값으로 단일 원인의 처리량 기여도를 계산하지 않는다.

추가 개선 순서는 다음과 같다.

1. Native 누적 순서를 유지하는 R1..16 Gate/Up의 tile·가중치 재사용 개선. 현재 R16 후보44.445→50.193µs는 느려서 배포하지 않았다.
2. Down의 native split·BF16 경계를 유지하며 CTA 병렬도를 늘리는 후보 검증. 추가 slice-parallel 연구 helper는 빌드됐지만 lane 사용 중 admission이 거부되어 GPU 실험하지 않았으며 성능/정확도 성과로 세지 않는다. 제품에는 반영하지 않았다.
3. 검증된 v8 경로까지 포함하는 bounded Graph 확장과 decode launch/동기화 감소. 현재 optional Graph 호환성 검증은 기존 mode9이며 일반 HTTP 비교의 Graph 성능 개선으로 계산하지 않는다.
4. Native/HF full-vocab 기준을 통과하는 QK/PV·norm·projection 결합을 별도 후보로 검증. Stock의 FlashAttention·fused kernel을 그대로 교체해도 exactness가 유지된다고 가정하지 않는다.

## 정확도·보조 회귀·홀드

- CPU 회귀는 기본517개 통과·40개 ignored, EFS 상위 경로 소유권으로 실패한 C02 fixture4개를 승인된 private `/tmp`에서만 재검사해 모두 통과했다. 합계521개 통과이며 fixture 제거 증거를 남겼다. 보안 검증 코드는 완화하지 않았다.
- 전체 모델은 C1 eager 및 실제 R1..32 segments Graph 각각208개 BF16 파일·104개 HF full-vocabulary 행에서 original native/HF와 바이트 일치했다. All-R에서 Graph capture/first replay 각각2304회이며 모든 allocation을 닫았다. Prefix16/128/130/397과 활성 행 수·순서 변화를 포함한다. 구성요소 검사와 별개다.
- Optional fused SiLU+기존 bounded MLP Graph 조합도208파일·104 HF행 일치, capture1152/first replay1152/cache hit2448, all-R allocation 회수를 확인했다. 이 경로는 기존 mode9 MLP 커널을 사용하므로 v8 decode Graph 가속 성과로 계산하지 않는다.
- 독립 감사31개 arm의 전체 SSE5195응답을 재구성했다. Riley3844응답은 original native oracle의 token IDs·text·usage·finish reason·prompt IDs 및 HF token/text와 모두 일치했다. 원래 EOS를 유지했다. Recovery·최종 active/waiting/KV0·실제 argv·startup·runner/binary SHA 및 지연 percentile/처리량 재계산을 통과했다.
- Stock1351응답은 raw SSE 자체 재구성·동일 prompt token 수·보고된 ID/text 일치 여부를 검증했다. Stock가 original native와 모두 같았다는 뜻은 아니다. Main 측정192개 중 native token 일치는 C16에서171/173/174개, C32에서171/174/169개다. Riley main은 두 동시성 모두 각192/192/192개다. Riley의 정확도 기준을 stock 수준으로 낮추지 않았다.

C1/GPU6 및 C8/GPU7은 각각2회,32 warm+96 measured requests/arm의 fresh before/after 보조 검사다. 각 동시성 안에서 GPU와 affinity를 맞췄다.

| 동시성 | v7→v8 tok/s | TTFT p95 ms | TTFT p99 ms | TPOT p95 ms |
|---|---:|---:|---:|---:|
| C1 | 88.622→90.876 | 22.989→22.782 | 23.435→22.961 | 10.931→10.617 |
| C8 | 366.905→391.154 | 81.618→81.088 | 109.534→111.584 | 20.364→19.219 |

C1의 새 decode 경로는 선택되지 않으므로 약2.5% 차이를 새 커널의 인과적 성과로 주장하지 않는다. C8 p99는 약2.05ms 높아졌으며 반복1에서만 악화했다. C16 p95와 함께 남은 tail 제한으로 기록한다.

선택한 감사 구간에서 foreign GPU PID0개, own+holder peak14236MiB로22528MiB 한도 안이었다. 모든 GPU 작업 종료 후 GPU6 PID2027914/ticks6083215119 및 GPU7 PID2027915/ticks6083215121은 각각20GiB idle allocation으로 복귀했다. 원래 holder PID와 CUDA context를 중단·재시작하지 않았으며 홀드는 계속 유지한다. 외부 작업을 종료하지 않았다. 이 메모리 홀드는 물리 GPU의 배타 잠금이 아니므로 외부 프로세스가 나타나면 admission/관측 gate를 따른다.

## 검토·재현 자료

- [2파일 layer patch](../benchmarks/candidates/sm80-decode-mlp-v8/candidate.patch), [적용 범위](../benchmarks/candidates/sm80-decode-mlp-v8/README.md), [layer manifest](../benchmarks/candidates/sm80-decode-mlp-v8/source-layer-manifest.json). Clean HEAD가 아닌 immutable v7 snapshot 위의 layer다. Reverse apply·whitespace 검사와 현재 소스935파일 일치를 확인했다. 기존 다른 변경은 유지했고 commit/push하지 않았다.
- [독립 SSE·지표·자원 감사](../benchmarks/evidence/20261020-decode-v8/independent-audit.json), [C16 결과](../benchmarks/evidence/20261020-decode-v8/runs/final-c16/result.json), [C32 결과](../benchmarks/evidence/20261020-decode-v8/runs/final-c32/result.json), [C16 p95 인접 요청](../benchmarks/evidence/20261020-decode-v8/c16-tail-audit.json).
- [구성요소576회 검사](../benchmarks/evidence/20261020-decode-v8/component-audit.json), [전체 vocab](../benchmarks/evidence/20261020-decode-v8/teacher-audit.json), [optional MLP Graph](../benchmarks/evidence/20261020-decode-v8/mlp-compatibility-audit.json), [C02 fixture 제거](../benchmarks/evidence/20261020-decode-v8/c02-removal-proof.json).
- [5개 trace 요약](../benchmarks/evidence/20261020-decode-v8/profile-summary.json), [실제 v8 kernel 선택](../benchmarks/evidence/20261020-decode-v8/v8-kernel-selection.json), [stock 설치 소스 증거](../benchmarks/evidence/20261020-decode-v8/stock-installed-source-proof.json).
- [source/binary binding](../benchmarks/evidence/20261020-decode-v8/source-binary-binding.json), [원본 reference provenance](../benchmarks/evidence/20261020-decode-v8/reference-provenance), [GPU 홀드](../benchmarks/evidence/20261020-decode-v8/terminal-hold-proof.json), [raw EFS index](../benchmarks/evidence/20261020-decode-v8/raw-artifact-index.json), [증거680파일 체크섬](../benchmarks/evidence/20261020-decode-v8/checksums.json).

원본·빌드·Nsight/BF16 raw는 `aws-a100:/mnt/efs/nlp/psyche/riley-decode-v8-20261020`에 유지한다. 선택 binary `riley-decode-v8` SHA256은 `5b43383a92d067f492d77114f0cbcfcbff2a3f87ea767e0dcf6a3d01d83e2bcb`다. 기존 v7 binary와 original eager oracle은 변경하지 않았다.

압축 증거5162129B의 SHA256은 `e7c66b7c5fafd320d0af29bab594c6158a70ec2f5d0a4d1a35fa03f99254bb5e`다. 내려받은 archive의 크기·SHA·680개 파일 SHA·중복/안전 경로를 검사했다. 모든 GPU 실행은 기존 `control.py --lease-dir .../leases-v2 --index 6|7 run -- ...`를 사용한다. 실제 argv·startup·SSE·GPU sample·실행 helper를 증거에 보존한다.
