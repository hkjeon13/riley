# Qwen 배치 projection·attention 및 decode 본체 Graph 재설계 v11

작성일: 2026-10-20. 상태: 최종 E 구현·기존 정확도·실제 GPU 경로·C16/C32 fresh5회·원시 응답/자원 감사 완료.

## 결과

C16 처리량 중앙값은 immutable v10 대비 **+17.93%**, TTFT p95는 **-2.66%**, TPOT p95는 **-15.40%**였다. vLLM 대비 처리량 비율은 51.72%→**60.99%**로 높아졌다.

C32 처리량 중앙값은 immutable v10 대비 **+17.59%**, TTFT p95는 **-1.92%**, TPOT p95는 **-14.48%**였다. vLLM 대비 처리량 비율은 44.83%→**52.72%**로 높아졌다.

기존 정확도와 GPU6·7의 원래 holder context를 유지했다. 개별 상수 조정 대신 실제 배치 행·weight 재사용·attention 병렬성·Graph 실행 범위를 바꿨다. 두 동시성 모두 처리량 +20% 목표에는 미달했고 vLLM과의 격차는 남아 있다. [종합 계획](inference-gap-comprehensive-plan-20261020.md) 전체의 완료를 뜻하지 않는다.

## 정상 HTTP 비교

A100-SXM4-40GB SM80, Qwen3-4B BF16, `aws-a100` / `ldccai-a100-02`. C16/GPU6/CPU24~31, C32/GPU7/CPU32~39에서 v10·E·stock을 각각5회 새 프로세스로 실행했다. 홀수 반복은 엔진 순서를 뒤집었다.

- Budget576, max sequence1024, KV block16/blocks=C×64, prefix cache off, own+holder22528MiB 상한.
- 기존 fixed12 prompts(125~130 tokens), cap8/16/24/32, greedy·원래 EOS. 각 arm64warm+192measured, 초기 측정 cohort 포함.
- v10 Graph off, E `--owned-decode-graph segments-v1`, stock vLLM0.27.1+cu129/Torch2.13.0+cu129 기본 compile/Graph 유지.
- 지표별5회 중앙값이며 pooled percentile이 아니다. TPOT는 응답 평균이다. 새 프로세스마다 OS 파일/compile cache까지 비우는 완전 cold-start 실험은 아니다.

| 동시성 | 엔진 | output tok/s | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | E2E p50/p95/p99 ms |
|---|---|---:|---|---|---|
| C16 | v10 | 590.297 | 68.739/95.832/196.304 | 23.963/25.475/27.090 | 476.207/835.722/879.175 |
| C16 | v11 E | 696.113 | 61.119/93.287/190.782 | 20.253/21.553/23.428 | 398.864/712.651/751.292 |
| C16 | stock vLLM | 1141.416 | 45.209/82.798/110.534 | 11.644/12.501/13.136 | 259.143/422.114/454.075 |
| C32 | v10 | 755.605 | 81.108/300.618/414.932 | 38.408/40.902/48.578 | 709.642/1305.163/1486.622 |
| C32 | v11 E | 888.536 | 70.483/294.832/396.571 | 32.665/34.980/45.361 | 589.682/1109.780/1326.938 |
| C32 | stock vLLM | 1685.504 | 52.946/165.842/219.417 | 15.575/17.561/21.569 | 338.278/554.891/668.075 |

| 동시성·반복 | v10→E tok/s | v10→E TTFT p95 ms | stock tok/s |
|---|---:|---:|---:|
| C16 · 0 | 592.061→696.113 | 95.512→92.262 | 1137.065 |
| C16 · 1 | 588.928→712.414 | 101.205→91.752 | 1142.250 |
| C16 · 2 | 593.515→687.430 | 96.342→93.352 | 1136.990 |
| C16 · 3 | 590.297→690.422 | 95.832→93.287 | 1141.416 |
| C16 · 4 | 586.609→700.460 | 94.324→93.600 | 1172.798 |
| C32 · 0 | 758.092→887.331 | 299.521→299.688 | 1678.783 |
| C32 · 1 | 755.513→888.536 | 306.334→293.969 | 1691.546 |
| C32 · 2 | 755.605→886.774 | 309.681→294.832 | 1685.504 |
| C32 · 3 | 762.470→897.574 | 299.467→294.853 | 1633.024 |
| C32 · 4 | 754.210→893.781 | 300.618→293.414 | 1694.513 |

처리량은10개 paired 반복 모두 개선됐다. TTFT/TPOT p95·p99의2% 초과 악화 flag: 없음. 모든 반복을 포함했고 초기 cohort나 나쁜 반복을 제외하지 않았다. 유한5회 결과로 임의 부하의 안정성이나 작은 tail 차이의 통계적 유의성을 보장하지 않는다.

### 서버 시작과 Graph 준비

C16 launch→ready 중앙값은 v10 52.36s→E **53.62s**, stock 43.09s였다. C32 launch→ready 중앙값은 v10 51.61s→E **56.62s**, stock 43.34s였다. 후보10개 서버 모두 R2..C를 listen 전에 준비했고 listen 이후 capture는0이었다. 준비 비용과 정상 TTFT를 구분한다. Stock 첫 compile의 긴 준비 시간도 원자료에 보존했다.

## 구현 범위

최종 E는 immutable v10 위의 7파일 layer다. `native-prefill-sm80-v5`의 기존 Qwen SM80 geometry에 한정하며 정확도 허용오차·정밀도·EOS 정책을 변경하지 않았다.

1. **O·Down의 실제 배치 projection:** 기존 native cuBLASLt 알고리즘을 고정한 채 R2..32 layout을 준비하고 각 행 수에 `AlgoCheck`를 수행한다. 작은 chunk를 반복하던 O와 기존 custom Down 대신 한 번의 배치 호출을 선택한다. 기존 bounded workspace와 주소는 유지하고 별도 cold workspace를 plan당 최대4MiB로 제한한다. 완료 fence·Graph 수명까지 use ledger를 유지한다.
2. **K/V와 Gate/Up:** K/V는 K256 shared tile·cp.async double buffer로 입력/weight를 재사용하고 원래80개 FFMA와32-lane tree를 유지한다. Gate/Up은 입력 tile을 공유하며 원래 두 FP32 누적 chain과 projection→BF16·SiLU→BF16·multiply→BF16 경계를 유지한다. Prefill이 섞인 iteration에서도 연속 singleton decode group을 묶고, prefill은 기존 merge/native recipe를 사용한다.
3. **Attention:** QK·softmax·PV의3개 kernel로 GPU token/depth CTA 병렬성을 유지한다. 같은 KV를 사용하는 query4개의 실제 WMMA 열을 채우고 scalar PV는2개 head를 공유한다. GPU의 현재 positions에서 길이와 family를 결정하며 요청별 score는 고정 stride512를 사용한다. CSR capacity·alias 검증은 capture 범위를 포함한다. 지원T24..216 밖은 기존 eager 경로다.
4. **LM head:** R17..32를 native K32 두-chain WMMA 한 번으로 계산한다. R<=16은 기존 경로를 유지한다. 실제 tied embedding weight로 수치를 검증했다.
5. **Qwen decode 본체 Graph:** decoder layers·final norm·LM head를 포함한다. Embedding·metadata·sampling·결과 전달은 밖에 있으므로 전체 HTTP step Graph라고 부르지 않는다. 실제 행 수별 고정주소 key와 Greedy/logits의 별도 key를 사용한다. 서버 listen 전 private KVpages에서 준비하고 scheduler 요청·예약을 만들지 않는다. Shape history/output readiness를 복구하며 첫 외부 요청이 prefix를 정상 덮어쓴다. R1·지원 범위 밖 context는 eager다. 기존 일반 step-Graph의 owned-HF 차단은 해제하지 않았다.

전체 attention을 한 CTA에 합친 최초 후보는 커널 수가 줄어도 실제 C16 attention GPU 합이217.237→253.862ms로 악화하여 제외했다. 최종3kernel 구조는 이 직렬화 문제를 피한다. O의 FP32/BF16 부분합을 수작업으로 재구성한 일부 후보도 원래 출력과 달라 제외했다. 실패 기록은 EFS 연구 루트에 보존했다.

## 정확도 및 적용 제한

- CPU519개 통과·40개 ignored. 기존 C02 fixture4개만 승인된 private `/tmp`에서 재검사해 통과했다. 합계523개 통과이며 임시 트리는 삭제했다. 보안 검증은 완화하지 않았다.
- C1 eager·행 수가 변하는 Graph는 각각104 full-vocabulary 행/208 BF16파일을 원래 native·HF와 바이트 비교했다.
- 별도 natural128 teacher는 R1..32 각각32step,1024행/2048 BF16파일을 비교했다. Greedy startup 후 logits 실행으로 key 분리를 함께 검사했다. Graph62capture/62first replay, 실제 row key2..32; R1은 eager다. 전체104행×32의 Cartesian 검사는 아니다.
- 총2464개 BF16파일은 literal byte audit를 통과했다. 유한 corpus 검증이며 임의 입력의 수학적 동치 증명은 아니다.
- Standalone attention9264비교/483803136BF16, O·Down 각각192비교/8110080BF16, LM head192비교/481333248BF16에서 차이0. O·Down은 실제 layer0 가중치, LM head는 실제 tied embedding weight를 사용했다. Projection은6개 유한 상태·모든R1..32, attention은6개 상태·R1/2/16/32·모든T24..216·혼합 길이·페이지 순열을 포함한다. 전체 모델 검증을 대신하지 않는다.

Mode10 전용 선택은 공통 native mode 정규화 이전의 값을 보존한다. 새 분기와 workspace use 등록 모두 같은 보존 값을 사용한다. 중간 D는 정규화 후10을 검사해 네 경로가 실행되지 않았고, 최종 E에서는 이 오류를 수정한 뒤 전체 정확도와 실제 선택 검사를 다시 수행했다. D의 약7~8% 결과를 새 경로의 성과로 사용하지 않는다.

## 실제 GPU 경로와 남은 병목

별도64warm+64measured Nsight 진단이며 정상 처리량과 합치지 않는다. CPU API wait와 GPU kernel 시간을 중복 합산하지 않는다.

| 범위 | kernel 수 v10→E | kernel 시간 합 ms v10→E |
|---|---:|---:|
| C16 | 128033→109062 | 2040.357→1771.803 |
| C32 | 100991→85048 | 1681.199→1442.777 |

Stock C32는 32321 kernels/843.515ms였다. 실제 CUDA Graph launch와 FlashAttention split-KV 실행을 확인했다. 익명 stock GEMM을 Gate/Up·Down으로 추측해 배정하지 않았다.

C16에서 새 K/V tile 6480회, 새 LM head 0회, Graph launch 69회를 확인했다. C32에서 새 K/V tile 3888회, 새 LM head 38회, Graph launch 34회를 확인했다. C16의 새 LM head0회는 R17 이상에서만 선택하기 때문이며 기존 작은 행 경로를 유지한다. O/Down은 NVTX 연산 geometry에서 기존8/4행 한도를 넘는 실제 호출을 확인했고, 기존 custom v9 Down은 두 trace 모두0회였다.

C32 E에서 직접 식별된 최소 시간은 다음과 같다.

| 영역 | kernel 합 ms | 전체 합 비율 |
|---|---:|---:|
| Gate/Up | 376.704 | 26.11% |
| K/V | 183.961 | 12.75% |
| QK/PV + softmax | 162.974 | 11.30% |

Captured native GEMM 일부는 수집 구간에서 NVTX correlation이 없어 미분류로 남겼다(C32 209.246ms). 따라서 부분적으로 식별된 O·Down·Q·LM head를 완전한 영역 합처럼 비교하지 않는다. DRAM/L2 counter 없이 대역폭 포화를 단정하지 않는다.

직접 식별한 C32 Gate/Up 중 prefill(M>32) native 호출은166.329ms, 새 decode fusion은200.960ms다. 두 단계 모두 남아 있는 개선 대상이다. 새 full-row O/Down은 decode R2..32에 적용되고 큰 prefill은 기존 수치 recipe를 유지한다. 본체 Graph도 순수 decode에 적용되므로 prefill/혼합 iteration의 개별 제출, embedding 검증·metadata·sampling·출력 전달과 동기화가 남는다. 커널 시간 자체도 stock보다 커서 Graph 범위 확대만으로 남은 격차가 해소된다고 볼 근거는 없다.

후속 우선순위는 큰 prefill Gate/Up과 decode fusion의 실제 tile/pipeline 개선, 혼합 배치 attention/Graph 범위 확대, 검증과 실패 계약을 유지하는 upload·embedding 검증·결과 전달의 동기화 통합이다. 이 문서에서 실행하지 않은 후속 과제를 성과로 포함하지 않는다.

## 감사·재현 자료·GPU 홀드

- 정상30개 arm의 SSE 6042개 값을 재구성했다. Riley 4032개(최종 E 2016개)는 token IDs·text·prompt IDs·usage·finish reason을 포함한 원래 native 값과 일치했다. Stock 2010개는 자체 SSE/동일 prompt token 수/보고된 agreement flag를 검증한 별도 범위이며 native와 전부 같다는 뜻은 아니다.
- 실제 argv·binary/runner SHA·처리량·quantile·원래 EOS·자원 회수를 독립 감사했다. 정상 peak own+holder는 14468MiB로22528MiB 이내였고 foreign GPU PID는 없었다.
- 설치 stock 핵심소스4개 SHA와 버전이 sealed v10 증거와 동일하다. Native/HF oracle과 stock 소스는 덮어쓰지 않았다.
- 모델·빌드·스크립트·벤치마크·scratch는 EFS에 유지했다. 승인된4개 C02 fixture만 private `/tmp`에서 재검사하고 삭제했다.
- 최종 finite 작업 종료 후3회 새 표본에서 원래 holder PID2027914/2027915, start ticks6083215119/6083215121, 각 GPU UUID·idle20GiB·다른 GPU 프로세스 없음이 유지됐다. CUDA context를 중단·재시작하거나 점유를 해제하지 않았다. GPU0~5의 작업은 변경하지 않았다.
- 최종 E는 빈 target에서 재빌드했다. 초기 복사/build cache 실패와 중간 D의 미실행 경로 기록은 별도 보존했다.
- 소스935개 로컬/qualified manifest 일치, immutable v10 위7파일 patch reverse-apply 및 whitespace 검사를 통과했다. 기존 dirty/untracked 작업을 보존했고 commit/push하지 않았다.

후보 binary SHA256: `2fb223ba679556fc3797fdc745ea5597744352c9de48f71396cf9a324c4e44a4`. 소스 manifest SHA256: `c9adb87d166ccb67fd3a5366ae96ea95a080806938a304562caa1b52a9a28d27`.

- [Candidate patch·선택 조건·rollback](../benchmarks/candidates/sm80-batched-v11/README.md)
- [정상/프로파일/정확도/홀드 증거](../benchmarks/evidence/20261020-batched-v11/README.md)
- [체크섬을 포함한 compact ZIP](../benchmarks/evidence/20261020-batched-v11.zip)

최종 EFS: `/mnt/efs/nlp/psyche/riley-batched-v11e-20261020`. 원시 BF16·바이너리·Nsight trace는 EFS에 남기고 compact 자료의 heavy index에 경로·SHA를 기록했다. A~D 연구 원본과 제외 후보도 보존했다.
