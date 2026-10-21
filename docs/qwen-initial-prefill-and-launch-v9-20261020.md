# Qwen3-4B: 초기 prefill TTFT·decode Down·launch/동기화 v9

상태: 최종 구현·기존 정확도·C16/C32 fresh3회 비교·C1/C8 보조 검사·5개 profile·독립 감사 완료. 압축 증거871파일과 현재 소스935파일의 SHA256을 재검증했다.

기존 original native/HF full-vocabulary BF16 바이트 일치, token IDs·text·usage·finish reason·원래 EOS 기준을 유지한다. GPU6 PID2027914/ticks6083215119 및 GPU7 PID2027915/ticks6083215121의 CUDA context를 유지한다. 대기20GiB·실행64MiB로 크기만 바꾸며 holder를 중단·재시작하지 않는다. 모든 모델·스크립트·빌드·벤치·scratch는 EFS에 두고, 기존에 승인된 C02 fixture4개만 private `/tmp`에서 검사 후 제거한다.

## 최종 변경과 선택 근거

최종 layer는 immutable v8 위의7파일이다. Default 실행 모드는 유지하며 비교 후보는 mode10 + staged uploads + fused SiLU multiply다. Graph/coalescing 후보는 사전 비교에서 제외하고 해당 소스 변경도 되돌렸다.

- **Decode Down:** native split-K2의 각 K32 accumulator chain을 별도 CTA로 계산한다. 두 FP32 chain의 합→BF16 partial, 두 partial의 FP32 합→BF16 output 경계를 유지한다. R5..16은 N128/R16/shared stride40·4stage, R17..32는 N64/R32/stride48·4stage다. R1..4는 기존 경로다. 선택한 SM80 native image를 cold prepare에서 확인한다. 기존 workspace/use/replay ledger를 사용하되 cold Down workspace 최소 크기를327680→1310720B로 늘린다. 이 준비 함수를 공유하는 다른 opt-in SM80 projection 모드에도 workspace 증가가 적용되며 기본 모드는 바꾸지 않는다.
- **Staged 업로드:** 기존 pinned staging과 command-batch copy 경로를 mode10의 owned HF 조합에 한정해 허용한다. Synchronous metadata 및 iteration-batch 조건, dtype·span·수명 검증을 유지하며 기존 owned HF 모드의 차단은 유지한다.
- **남아 있던 attention 메타데이터 업로드:** 기존 row-prefix 인코딩을 공유하고, valid-prefix와 inactive-row zero-tail payload를 기존 pinned 버퍼의 겹치지 않는 영역에 준비한다. 용량·checked-add 검사 후 쓴다. GPU 복사는 iteration command batch 안에 enqueue해 별도 동기 업로드2개를 없앤다. 새 H2D active-command-batch 메서드는 기존 typed copy와 같은 context·range·use 검사 및 native ledger를 사용한다. 새 buffer/FFI ABI는 없다. Shared transport를 쓰는 다른 staged+materialized 조합에도 이 복사 경로가 적용되며 성능·전체 모델 검증 범위는 이 보고서의 Qwen3-4B/A100이다.
- **SiLU·곱셈 융합:** 기존 fused path를 새 Down과 함께 선택한다. SiLU 결과의 BF16 중간 반올림을 유지하며 layer당 activation/multiply2launch를1개로 줄인다. 실제 전체 모델 일치를 다시 확인한다.
- **Gate/Up:** K32 slice 병렬화21개 변형과 원래 WMMA의8-warp 추가 변형은 채택할 속도 이점이 없어 제외했다. v8의 R17..32 전용 kernel과 R1..16 native 경로를 유지한다.

### 제외한 실험

아래는 최종 fresh3회 비교와 구분한 사전 설정 비교다. 정확도 통과만으로 느린 후보를 제품에 남기지 않았다.

| 사전 비교 | before→after tok/s | TTFT p95 ms | 판단 |
|---|---:|---:|---|
| v8 C16 all-R segment Graph,32warm96keep | 548.783→457.541 | 202.770→169.521 | 처리량 악화·측정 중72capture로 제외 |
| prototype C16 staged→hot16/32 Graph+cohort512/500µs,64warm192keep | 575.89→580.63 | 105.287→124.750 | 초기 tail 악화로 제외 |
| C32 v8→prototype hotGraph+cohort | 720.20→755.69 | 340.173→345.983 | p95 개선 없음 |
| C16 prototype staged→hotGraph only | 570.75→571.26 | 107.674→107.936 | 이점이 미약해 제외 |
| C32 v8→prototype Down+bulk staged only | 715.75→715.83 | 343.586→341.837 | 처리량 변화 미미; 이후 남은 attention2업로드와 SiLU를 개선 |

마지막 사전 후보는 attention 메타데이터2개가 여전히 동기 경로였다. 최종 버전은 이를 command batch로 옮겼다. 초기 요청을 모으는 대기나 Graph shape 제한은 최종 제품 patch에 없다. 원래 scheduler/Graph 동작을 유지하며 실험 source/binary와 실패 로그를 별도로 보존한다.

## 구성요소 증거

Down actual L0 가중치와 seed, R1..32×6상태의192회 비교에서 BF16 값8110080개가 native M1과 모두 일치했다. 허용오차를 추가하지 않았으며 임의 모델/입력에 대한 수학적 동치 증명은 아니다. 모든 H2D는 계산 stream에 명시적으로 enqueue했다. Baseline은 R1..4의 native/Lt와 R5..32의 v8 kernel을 그대로 사용한다.

| 실제 행 | v8 µs | v9 µs |
|---|---:|---:|
| 4 | 45.731 | 45.715 |
| 5 | 59.771 | 49.280 |
| 8 | 58.914 | 49.320 |
| 12 | 59.730 | 50.619 |
| 16 | 59.733 | 51.501 |
| 17 | 69.583 | 56.275 |
| 24 | 68.860 | 56.963 |
| 32 | 69.327 | 57.724 |

각각3회 alternating CUDA-event 중앙값이며 서버 처리량으로 해석하지 않는다. R4는 변경하지 않은 경로이므로 작은 차이를 개선 성과로 세지 않는다.

## 검증 조건

이전 v8·v9 후보·설치 stock vLLM0.27.1을 같은 A100-SXM4-40GB GPU lane, Qwen3-4B BF16, budget576, max sequence1024, KV block16/blocks=C×64, prefix cache off, 같은12prompts와 cap8/16/24/32·greedy sampling·원래 EOS로 비교한다. C16/GPU6·C32/GPU7 각각 fresh3회,64 warm+192 measured requests/arm으로 실행 순서를 교대한다. 초기 cohort와 cold capture 비용을 제거하지 않는다. Stock의 기본 compile/Graph를 유지하고 일반 HTTP와 stock profile에 inspection preload/event spin을 사용하지 않는다. Native 원본 oracle은 기존 eager baseline이며 v8은 새 성능 baseline일 뿐이다.

일반 HTTP 측정과 별도64 warm+64 measured profile을 구분한다. TPOT는 response mean이며 token ITL이 아니다. 결과는 지표별3회 중앙값이며 pooled percentile이 아니다.

## Fresh C16/C32 결과

같은 GPU의 v8·v9·stock을 새로3회 실행한 지표별 중앙값이다. 이전 보고서의 수치를 baseline으로 재사용하지 않았다. 처리량 단위는 output tok/s, 지연은 ms다. Percentile을 합쳐 재계산한 pooled percentile은 아니다.

| 동시성 | 엔진 | 처리량 | TTFT p50/p95/p99 | TPOT p50/p95/p99 | E2E p50/p95/p99 |
|---|---|---:|---|---|---|
| C16 | v8 | 566.949 | 70.885/108.628/222.694 | 24.931/26.756/28.189 | 501.339/876.299/917.204 |
| C16 | v9 | 574.188 | 70.014/105.854/220.887 | 24.800/26.418/27.766 | 497.341/862.944/913.481 |
| C16 | stock vLLM | 1126.664 | 49.331/83.666/110.866 | 11.876/12.893/13.386 | 261.842/432.189/446.530 |
| C32 | v8 | 717.308 | 83.758/341.738/468.512 | 40.523/43.139/55.134 | 757.230/1369.138/1595.424 |
| C32 | v9 | 735.891 | 83.263/339.077/461.062 | 39.466/41.840/54.049 | 731.648/1339.934/1559.662 |
| C32 | stock vLLM | 1671.550 | 52.237/164.610/217.645 | 15.703/17.778/21.536 | 342.454/555.179/657.116 |

- C16: 처리량 **+1.28%**, stock 대비 50.32%→**50.96%**. TTFT p95 108.628→105.854ms (-2.55%).
- C32: 처리량 **+2.59%**, stock 대비 42.91%→**44.02%**. TTFT p95 341.738→339.077ms (-0.78%).

C16 p95는 중앙값이 낮아졌지만 **안정적으로 해결됐다고 판정하지 않는다**. 반복1은107.704→111.133ms로 악화했고, 반복0·2가 개선했다. C32도 반복2는341.738→340.975ms로 차이가 작다.3회의 유한 비교이며 작은 처리량 차이의 통계적 유의성을 증명한 것은 아니다. Stock C16 반복1은1035.89tok/s로 다른 두 회1127.89/1126.66보다 낮았으며 제외하지 않고 중앙값·반복표에 포함했다.

| 동시성·반복 | v8 tok/s·TTFT p95 | v9 tok/s·TTFT p95 | stock tok/s·TTFT p95 |
|---|---:|---:|---:|
| C16 · 0 | 566.949 · 108.628 | 574.188 · 105.854 | 1127.892 · 83.666 |
| C16 · 1 | 570.296 · 107.704 | 576.650 · 111.133 | 1035.891 · 95.419 |
| C16 · 2 | 565.765 · 109.170 | 570.626 · 105.223 | 1126.664 · 83.119 |
| C32 · 0 | 716.981 · 339.839 | 739.894 · 338.361 | 1663.051 · 164.610 |
| C32 · 1 | 717.308 · 345.253 | 732.691 · 339.077 | 1671.550 · 166.992 |
| C32 · 2 | 720.089 · 341.738 | 735.891 · 340.975 | 1697.908 · 163.062 |

## 초기 tail과 실제 남은 병목

C16 p95 interpolation 양쪽 요청은6개 arm 모두 측정 시작 후0.43~1.14ms 안에 들어온 초기 cohort이며 prompt125~130tokens다. 초기 요청을 제외하거나 steady-state percentile로 바꿔 개선처럼 보이게 하지 않았다. 중앙값 개선2.774ms와 반복1의 악화를 함께 유지한다.

별도64warm+64measured 진단에서 projection은 실제 NVTX GEMM M과 row-bounded kernel 이름으로 구분했다. 고정 측정 prompt는125~130tokens이므로 M<=32는 decode row-group projection이다. 다른 커널에는 같은 phase 가정을 적용하지 않는다. 아래 시간은 각각의 trace kernel-duration 합이며 정상 HTTP 지연·단일 요청 시간·기여도 비율이 아니다. Batch 배치와 padding이 달라질 수 있어 같은 커널의 전체 합이 늘었다고 곧 회귀로 단정하지 않는다.

| kernel 구분 | C16 v8→v9 ms | C32 v8→v9 ms |
|---|---:|---:|
| decode Gate/Up | 319.948→319.698 | 223.837→223.818 |
| prefill Gate/Up | 175.807→185.627 | 157.702→167.767 |
| decode Down | 209.939→180.797 | 143.282→123.498 |
| prefill Down | 100.122→100.388 | 92.606→94.947 |
| K/V projection 전체 | 300.880→295.884 | 226.044→228.090 |
| QK/PV 전체 | 242.703→244.884 | 208.717→206.595 |
| RMSNorm 전체 | 176.405→179.485 | 162.093→162.902 |
| SiLU+multiply 전체 | 68.645→48.639 | 56.850→42.714 |
| KV gather 전체 | 8.646→8.316 | 8.944→8.602 |

실제 v9 Down은 C16 primary kernel3240회+reduce3240회, C32 primary kernel1944회+reduce1944회가 등장했다. Fused SiLU는 C16 3672회, C32 2340회이며 before에는 없다. Gate/Up 투영 코드는 v8과 같고, profile에서도 decode 시간이 거의 같다. Prefill 투영 자체도 이번에 빨라진 것으로 보이지 않는다. 초기 p95를 안정적으로 해결했다고 하지 않는 이유다.

| 진단 지표 | C16 v8→v9 | C32 v8→v9 |
|---|---:|---:|
| cudaStreamSynchronize 호출 수 | 1208→306 | 807→195 |
| 위 API CPU 대기 합 ms | 1122.685→1105.675 | 889.997→874.071 |
| GPU kernel 수 | 134187→131271 | 106209→103761 |
| GPU kernel 합 ms | 2176.597→2136.592 | 1779.873→1762.015 |
| GPU kernel span ms | 2551.911→2382.929 | 2058.823→1988.096 |

동기화 호출 수는 약75% 줄었지만 기다리는 시간의 상당 부분은 GPU 작업과 겹친다. API CPU 대기와 GPU kernel 시간을 더하거나, 호출 수 감소율을 처리량 개선율로 쓰지 않는다. 정상 HTTP의 처리량 개선은 C16 1.28%·C32 2.59%로 작다.

Stock C32 trace는32433kernels/846.620ms이며 기본 CUDA Graph launch38회와 FlashAttention split-KV kernel이 실제 등장한다. Riley v9는103761kernels/1762.015ms이며 정상 설정에서 Graph off다. Stock의 익명 GEMM을 Gate/Up·Down에 추측으로 배정하지 않았고, DRAM/L2 counter가 없어 대역폭 포화라고 단정하지 않는다. 이 계측은 남은 launch·융합 범위와 계산량 차이를 보여주지만 각각의 처리량 기여도를 분리해 증명한 것은 아니다.

다음 우선순위는 native 누적 순서를 유지하는 Gate/Up·prefill projection의 batch/recipe 범위 확장, 그 뒤 K/V·QK/PV·norm이다. 현재 Gate/Up 후보와 Graph 변형이 실제로 빨라지지 않았으므로 단순히 켜지 않았다. 새 후보는 같은 BF16 전체 vocab 기준과 정상 HTTP 비교를 먼저 통과해야 한다.

## 정확도·보조 검사·GPU 홀드

- CPU 회귀는517개 통과·40개 ignored, EFS 상위 경로 UID 소유권으로 실패한 C02 fixture4개만 승인된 private `/tmp`에서 재검사해 통과했다. 합계521개 통과이며 fixture 트리 제거를 확인했다. 보안 검증 코드는 완화하지 않았다.
- 최종 staged metadata+fused SiLU+새 Down 조합의 C1 eager와 실제 R1..32 Graph 각각208 BF16파일·104 HF full-vocabulary 행이 original native/HF와 바이트 일치했다. All-R Graph capture/first replay 각각2304회, allocation 회수를 확인했다. Normal HTTP는 Graph off이며 이 숫자를 성능 성과로 세지 않는다.
- 독립 감사는 사전 성공 비교를 포함한41arm, 전체 SSE7403응답을 재구성하고 처리량·p50/p95/p99를 다시 계산했다. Riley6052응답의 token IDs·text·usage·finish reason·prompt IDs가 original native oracle과 같고 HF token/text 일치도 모두 통과했다. 원래 EOS를 유지했다. 미완료/실패 실험은 accepted 비교에 섞지 않고 raw에 남겼다.
- Stock1351응답은 raw SSE 자체 재구성·동일 prompt token 수·보고된 ID/text 일치 여부를 검증했다. Original native와 모두 같다는 뜻은 아니다. Main192개 중 stock native token 일치는 C16 175/175/173개, C32 171/170/174개다. Riley main은 각각192/192/192개다. Riley 정확도를 stock 수준으로 낮추지 않았다.

C1/GPU6와 C8/GPU7도 같은 동시성 안에서 before/after GPU·affinity를 맞춰2회,32warm+96measured/arm을 검사했다.

| 동시성 | v8→v9 tok/s | TTFT p95 ms | TTFT p99 ms | TPOT p95 ms |
|---|---:|---:|---:|---:|
| C1 | 88.818→90.201 | 23.020→22.549 | 23.353→23.034 | 10.988→10.743 |
| C8 | 390.854→397.086 | 80.838→79.767 | 109.449→110.256 | 19.168→18.811 |

C1은 R1이므로 새 Down 경로가 선택되지 않는다. C8 p99 중앙값은 약0.807ms 높다. C1·C8은 보조2회 비교이고 모든 반복의 tail 비악화를 보장하지 않는다.

감사 구간의 foreign GPU PID는0개, own+holder peak14250MiB로22528MiB 한도 안이었다. 모든 GPU 작업 종료 후 GPU6 PID2027914/ticks6083215119 및 GPU7 PID2027915/ticks6083215121이 각각20GiB idle allocation으로 복귀했다. CUDA context를 중단·재시작하지 않았고 홀드는 계속 유지한다. 외부 작업을 종료하지 않았다. 메모리 홀드는 물리 GPU의 배타 잠금이 아니므로 새 외부 프로세스에는 admission/관측 gate를 따른다.

## 검토·재현 자료

- [최종7파일 patch](../benchmarks/candidates/sm80-latency-v9/candidate.patch), [적용 범위](../benchmarks/candidates/sm80-latency-v9/README.md), [layer manifest](../benchmarks/candidates/sm80-latency-v9/source-layer-manifest.json). Immutable v8 위의 layer이며 clean HEAD가 아니다. Reverse apply·whitespace 검사와 현재 소스935파일 일치를 확인했다. 기존 다른 변경은 보존했고 commit/push하지 않았다.
- [독립 SSE·지표·자원 감사](../benchmarks/evidence/20261020-latency-v9/independent-audit.json), [C16 결과](../benchmarks/evidence/20261020-latency-v9/runs/final-c16/result.json), [C32 결과](../benchmarks/evidence/20261020-latency-v9/runs/final-c32/result.json), [C16 p95 인접 요청](../benchmarks/evidence/20261020-latency-v9/c16-tail-audit.json).
- [최종 전체 vocab](../benchmarks/evidence/20261020-latency-v9/final-teacher-audit.json), [새 Down의6상태 검사](../benchmarks/evidence/20261020-latency-v9/component-audit.json), [실제 최종 설정](../benchmarks/evidence/20261020-latency-v9/selected-configuration.json), [C02 fixture 제거](../benchmarks/evidence/20261020-latency-v9/c02-removal-proof.json).
- [5개 trace 요약](../benchmarks/evidence/20261020-latency-v9/profile-summary.json), [실제 Down·융합 SiLU·동기화 감소](../benchmarks/evidence/20261020-latency-v9/profile-selection-audit.json), [projection row-group 구분](../benchmarks/evidence/20261020-latency-v9/projection-row-phase-audit.json), [v9 kernel 선택](../benchmarks/evidence/20261020-latency-v9/v9-kernel-selection.json).
- [source/binary binding](../benchmarks/evidence/20261020-latency-v9/source-binary-binding.json), [현재 설치 stock source 증거](../benchmarks/evidence/20261020-latency-v9/stock-installed-source-proof.json), [GPU 홀드](../benchmarks/evidence/20261020-latency-v9/terminal-hold-proof.json), [raw EFS index](../benchmarks/evidence/20261020-latency-v9/raw-artifact-index.json), [증거871파일 체크섬](../benchmarks/evidence/20261020-latency-v9/checksums.json).

설치 source 증거 helper에 이전 output root가 남아 JSON 증거1개를 덮어쓴 것을 발견했다. 새 read-only metadata 증거를 v9에 옮기고 helper를 parent-relative로 수정했으며, 이전 증거는 보존된 v8 archive와 동일 SHA로 복원했다. 모델·측정 binary·설치 vLLM 코드는 바꾸지 않았다. [경로 수정·원본 복원 증거](../benchmarks/evidence/20261020-latency-v9/stock-proof-path-correction.json)를 남겼다.

원본·빌드·heavy BF16/Nsight raw는 `aws-a100:/mnt/efs/nlp/psyche/riley-latency-v9-20261020`에 유지한다. 최종 binary `riley-latency-v9-final` SHA256은 `d9a1e6042861e432e08f4ac4b533e0f2dcc0a45a95f13b8713f2969941734424`다. Prototype binary와 source archive를 보존했고 최종 source와 구분한다. 기존 v8 binary 및 original eager oracle은 변경하지 않았다.

압축 증거7233579B SHA256은 `8b2dc53f73a2cd3729fcb1da608bea75be4ecc49de6f5be450f6f4a4249e1343`다. 내려받은 archive의 크기·SHA·871파일 SHA·중복/안전 경로와 소스935파일을 검증했다. 모든 GPU 실행은 기존 `control.py --lease-dir .../leases-v2 --index 6|7 run -- ...`를 사용했다. 실제 argv·startup·SSE·GPU sample·실행 helper를 증거에 보존한다.
