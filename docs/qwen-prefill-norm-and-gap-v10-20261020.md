# Qwen3-4B 종합 개선 계획 실행 v10

작성일: 2026-10-20. 상태: 구현·기존 정확도·C16/C32 fresh5회·C1/C8 보조·C1 추가3회·별도 원시 응답 감사 완료. 종합 계획 전체를 완료했다는 뜻은 아니며, 이번 채택 layer는 prefill RMSNorm의 제출 범위와 CUDA 실행 구조 개선이다.

## 결과와 범위

[종합 개선 계획](inference-gap-comprehensive-plan-20261020.md)을 immutable v9 위에서 실행했다. C16 초기 TTFT p95는5회 모두 개선됐으며 중앙값이11.5% 감소했다. C32 TTFT p95는9.8% 감소했다. 처리량은각각2.0%·3.8% 증가해 계획의 첫 묶음 +10% 처리량 목표에는 미달했다. 큰 Gate/Up·K/V·Down·QK/PV 병목은 남아 있다.

Original native/HF full-vocabulary BF16 바이트, token IDs·text·prompt IDs·usage·finish reason·원래 EOS 기준을 유지했다. 정확도 허용오차·양자화를 추가하지 않았다. GPU6/7의 원래 holder PID/start ticks와 CUDA context를 중단·재시작하지 않았다.

## 구현한 두 파일

- `crates/riley-cuda/src/primitives.rs`: 인접한16행 이상 wide-HF prefill 요청의 RMSNorm을 하나로 제출한다. 이 범위는 기존에도 동일한32-thread row reduction이다. Singleton·16행 미만·Smol 경로의 분할과 reduction topology는 유지한다. Cached/uncached 진입점은 같은 helper를 사용한다. Parent span/context/use 검증을 유지하고 hot allocation을 추가하지 않는다.
- `kernels/src/primitives.cu`: 32-thread wide-HF topology의 독립된4행을 한 CTA의4개 warp에 배치한다. 각 lane의4개 FP32 accumulator, descending shuffle tree, epsilon/rsqrt, normalize→BF16→weight multiply→BF16 경계를 유지한다. Shared partial/inverse 저장과 block barrier를 없애고 warp 내 전달을 사용한다. Singleton의512-thread reduction과 다른 thread topology, 기존 resource ledger·completion·FFI는 유지한다.

이 shared wide-HF primitive는 width2048/2560/4096에 적용된다. Standalone 수치 검사는 세 폭을 포함하지만 전체 모델·서빙 성능 검증은 Qwen3-4B/A100 범위다. 다른 모델·GPU의 전체 모델 자격 검증을 대신하지 않는다. Gate/Up·Down·Attention 산술을 이번 layer에서 바꾸지 않았다.

외부 구현은 [원래 조사와 공식 참고 자료](inference-gap-comprehensive-plan-20261020.md#3-외부-구현에서-가져올-설계)를 참고했다. vLLM의 merged linear는 별도 component로 검토했고, packed 작업 제출과 GPU 실행 단위 개선을 기존 strict backend에 적용했다. 외부 kernel을 그대로 채택하거나 HF 수치 경계를 완화하지 않았다.

## 같은 조건의 정상 HTTP 비교

A100-SXM4-40GB SM80, Qwen3-4B BF16, C16/GPU6·C32/GPU7에서 v9/후보/설치 stock vLLM을 각5회 fresh process로 비교했다. 같은 동시성 안에서 GPU·CPU affinity·입력·sampling·메모리 예산을 맞췄다. Riley 양쪽은 동일한 mode10·staged uploads·fused SiLU 설정이다.

- Budget576, max sequence1024, KV block16/blocks=C×64, prefix cache off.
- 측정은 fixed12 prompts(125~130 tokens), cap8/16/24/32, greedy·원래 EOS. 별도 정확도 검사에는 natural EOS 사례를 포함한15 reference items를 사용한다.
- 각 arm은64 warm+192 measured. Riley Graph off, stock 기본 compile/Graph 유지. 초기 측정 cohort를 제외하지 않았다. 완전 cold start 지연을 측정한 것은 아니다.
- 아래는 지표별5회 중앙값이며 pooled percentile이 아니다. TPOT는 응답 평균이며 token ITL 분포가 아니다.5회의 유한 실험으로 모든 환경의 안정성이나 작은 차이의 통계적 유의성을 증명하지 않는다.

| 동시성 | 엔진 | output tok/s | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | E2E p50/p95/p99 ms |
|---|---|---:|---|---|---|
| C16 | v9 | 576.041 | 70.426/107.374/219.982 | 24.618/26.191/27.798 | 481.898/859.602/911.155 |
| C16 | v10 | 587.621 | 67.534/95.024/196.177 | 24.269/25.525/26.658 | 466.144/841.440/875.799 |
| C16 | stock vLLM | 1159.399 | 43.462/85.819/112.682 | 11.527/12.552/13.297 | 262.080/414.652/448.482 |
| C32 | v9 | 734.632 | 83.508/339.104/463.039 | 39.494/42.917/51.838 | 730.383/1338.355/1590.533 |
| C32 | v10 | 762.190 | 80.727/305.716/415.589 | 38.098/40.537/48.473 | 706.180/1293.282/1479.297 |
| C32 | stock vLLM | 1677.338 | 53.012/164.833/217.856 | 15.732/18.064/21.552 | 339.804/564.323/678.686 |

- C16: 처리량 **+2.01%**, TTFT p95 **-11.50%**, p99 -10.82%. Stock 대비 처리량은 49.68%→**50.68%**.
- C32: 처리량 **+3.75%**, TTFT p95 **-9.85%**, p99 -10.25%. Stock 대비 처리량은 43.80%→**45.44%**.

두 동시성 모두5회 각각에서 처리량과 TTFT p95가 개선됐다. 어느 반복에서도 TTFT p95/p99·TPOT p95의2% 초과 악화 flag가 없었다. 처리량의 vLLM 격차는 여전히 크며, 이 결과를 성능 동등성으로 표현하지 않는다.

| 동시성·반복 | v9→v10 tok/s | v9→v10 TTFT p95 ms | stock tok/s |
|---|---:|---:|---:|
| C16 · 0 | 577.866→597.763 | 107.374→101.509 | 1112.371 |
| C16 · 1 | 573.996→586.703 | 104.898→98.509 | 1162.637 |
| C16 · 2 | 576.041→587.621 | 109.355→92.717 | 1159.399 |
| C16 · 3 | 570.305→585.014 | 105.141→94.785 | 1139.488 |
| C16 · 4 | 580.457→601.527 | 108.077→95.024 | 1161.741 |
| C32 · 0 | 733.950→765.021 | 339.104→303.532 | 1672.009 |
| C32 · 1 | 734.632→762.190 | 342.634→305.716 | 1679.197 |
| C32 · 2 | 730.475→756.611 | 355.041→313.979 | 1677.338 |
| C32 · 3 | 737.594→761.488 | 333.693→303.741 | 1677.481 |
| C32 · 4 | 735.239→767.458 | 338.984→307.116 | 1667.138 |

### 초기 cohort 해석

C16 before의 p95 interpolation 양쪽 요청은5회 모두 첫 cohort(측정 시작 후0.263~1.513ms)였다. After는4회에서 양쪽 모두 첫 cohort였고, 반복4에서는 첫 cohort 요청 하나와 측정 시작 약896.9ms 뒤의 요청 하나가 경계에 있었다. 초기 tail이 줄었지만 이후 요청도 p95에 걸릴 수 있다. 모든 요청을 포함했으며 초기 요청이나 나쁜 반복을 제외하지 않았다. Request 단계별 queue/admission/GPU/SSE를 완전히 분해한 계측은 후속 과제로 남는다.

## 실제 GPU 경로와 남은 병목

별도64warm+64measured profile이며 HTTP 지연 기여도와 구분한다. Kernel-duration sum·GPU span·CPU API wait는 중복 합산하지 않는다.

| 진단 지표 | C16 v9→v10 | C32 v9→v10 |
|---|---:|---:|
| GPU kernel 수 | 130875→127998 | 104273→101573 |
| GPU kernel 합 ms | 2131.595→2064.462 | 1759.591→1678.713 |
| Norm kernel 수 | 12045→9636 | 9417→6789 |
| Norm kernel 합 ms | 177.982→110.767 | 161.684→91.230 |

새 `hugging_face_qwen_prefill_warp_rms_norm_kernel`은 after에만 등장했다(C16 2263회, C32 2044회). 기존 decode/짧은 요청 norm은 유지됐다. C32의 남은 legacy norm은22.104ms로 전체 kernel 합의 약1.3%다. 작은 norm 구간만 더 조정해 큰 격차를 줄이기는 어렵다.

| C32 v10 남은 영역 | kernel 합 ms |
|---|---:|
| Gate/Up | 384.621 |
| K/V projection | 228.748 |
| Down | 213.247 |
| QK/PV | 205.823 |
| O projection | 173.089 |
| Q projection | 98.226 |
| RMSNorm | 91.230 |
| KV gather | 8.893 |

Stock C32 trace는 32253 kernels/847.395ms였다. 실제 Graph launch와 FlashAttention split-KV는 별도 selection audit에서 확인했다. 익명 stock GEMM을 Gate/Up·Down에 추측으로 배정하지 않았다. DRAM/L2 counter 없이 대역폭 포화를 단정하지 않는다.

## 제외한 연구 후보

1. **Token budget576→1024:** 동일 v9 바이너리로 fresh3회 검사했다. C16은582.255→571.001tok/s, p95106.892→108.669ms로 악화했다. C32는732.155→728.006tok/s이며 p95338.744→296.502ms는 개선됐지만 p5083.031→112.893ms가 악화했다. 채택하지 않았다. 같은 자원 조건의 코드 개선과 섞지 않는다.
2. **Gate/Up K128/K256 staging 확대:** native K32 누적 순서를 보존하는12설정을 준비했다. 완료한9설정의 비교는 차이0이지만 대체로 느렸고, 일부 R24/32의 작은 이득으로 채택하지 않았다. 후반 큰 static shared-memory 설정에서 launch가 실패해 전체 연구는 실패 종료했다. 완료 부분과 미실행 설정을 구분해 원시 기록을 보존했다.
3. **일반 merged Gate/Up GEMM:** 실제 L0 Gate/Up 가중치를 합친 N19456 레시피는 R1/state0에서19456개 중4991개 BF16 값이 두 개의 원래 native N9728 M1 호출과 달랐다. 원래 ldc9728을 맞춘 재검사에서도 같았다. 이 레시피를 제외했으며 다른 병합 구현까지 불가능하다는 결론은 아니다. Owned-HF의 미검증 merged-MLP 차단을 해제하지 않았다.
4. **Singleton norm virtual-thread 분해:** 원래512-thread partial/reduction tree를 보존하는 physical32/64/128/256-thread 후보를2304회 검사해 수치는 일치했다. Host launch gap을 피한 CUDA Graph microbench에서 width2560의 aggregate 시간 비율은 각각6.25/3.43/1.93/1.27배로 느렸다. 채택하지 않았다. CUDA Graph microbench를 정상 serving 성과로 사용하지 않는다.

## C1·C8 보조 검사와 C1 확대 확인

초기 보조 검사는각2회,32warm+96measured/arm이었다.

| 동시성 | v9→v10 tok/s | TTFT p95 ms | TTFT p99 ms | TPOT p95 ms |
|---|---:|---:|---:|---:|
| C1 | 91.917→89.847 | 22.393→22.410 | 22.544→22.735 | 10.521→10.801 |
| C8 | 401.282→404.882 | 79.080→75.145 | 108.034→95.401 | 18.458→18.369 |

C1의 두 번째 after가88.146tok/s로 낮아, 같은 GPU6에서각3회 fresh·64warm+192measured로 확대 확인했다. 초기 나쁜 반복은 보존하고 별도 결과로 제시한다. 서로 다른 warm/keep 설정의 두 묶음을 pooled 결과로 합치지 않는다.

| C1 확대 확인 | v9 | v10 |
|---|---:|---:|
| output tok/s | 91.378 | 92.096 |
| TTFT p95 ms | 22.390 | 22.127 |
| TTFT p99 ms | 22.672 | 22.221 |
| TPOT p95 ms | 10.556 | 10.494 |

확대3회에서 처리량 변화는 각각+1.25%,+0.79%,-0.19%로 초기 큰 하락은 재현되지 않았다. TPOT p95 변화는-1.65%,-0.53%,+0.68%였다. 초기 한 번의 악화 원인을 확정하거나 임의 환경의 안정성을 보장한 것은 아니다. 새 kernel은 prefill에 적용되고 singleton decode의512-thread 경로는 유지된다.

## 정확도·재현성·GPU 홀드

- CPU518개 통과·40개 ignored. EFS 상위 경로 UID 때문에 실패한 기존 C02 fixture4개만 승인된 private `/tmp`에서 재검사해 통과했다. 합계522개 통과이며 임시 트리 제거를 확인했다. 보안 검증 코드는 완화하지 않았다.
- Norm component: width2048/2560/4096,6개 유한 입력 상태,15개 행 수,270비교/589138944 BF16 값의 차이0. Reference는 기존32-thread CUDA kernel이며 임의 모델/입력의 수학적 동치 증명은 아니다.
- 전체 모델 C1 eager와 실제 R1..32 Graph에서 각각208 BF16파일·104 HF full-vocabulary 행이 original native/HF와 바이트 일치했다. Graph capture/first replay 각각2304회와 allocation 회수를 확인했다. 정상 성능은 Graph off다.
- 별도 감사가 accepted61arm의 전체 SSE 응답값 **11183개**를 재구성하고 실제 argv·binary/runner hash·quantile·처리량·원래 EOS·자원 회수를 검증했다. Riley9064개 값은 original native oracle과 같고, 이 중 최종 v10 바이너리 응답은3284개다. Stock2119개는 자체 SSE 재구성·동일 prompt token 수·보고된 agreement flag를 검증한 별도 범위이며 native와 모두 같다는 뜻이 아니다.
- 설치 stock vLLM0.27.1+cu129/Torch2.13.0+cu129의 버전과 핵심소스4개 해시가 이전 sealed 검증과 같은지 확인했다. 이 작업에서 stock 소스를 수정하지 않았다.
- 최종 소스935개는 qualified manifest 및 현재 로컬 파일과 일치한다. 이 layer는 immutable v9 위의2파일이며 clean HEAD 전체 diff가 아니다. Reverse apply·whitespace 검사를 통과했다. 기존 dirty/untracked 변경을 보존했다.
- GPU6 PID2027914/start ticks6083215119, GPU7 PID2027915/start ticks6083215121을 유지했다. 대기20GiB·실행64MiB로 크기만 조정했고 holder context를 중단·재시작하지 않았다. 모든 유한 GPU 작업 종료 뒤3회 fresh 확인에서 두 holder 모두20GiB idle·추가 GPU PID0개였다. 감사 구간 own+holder peak14316MiB로22528MiB 한도 안이었다. 외부 작업과 GPU0~5를 변경하지 않았다. 메모리 홀드는 물리 GPU의 배타 잠금이 아니다.

모델·스크립트·빌드·벤치·scratch는 EFS에 유지했다. `/tmp` 예외는 승인된 C02 fixture4개뿐이며 제거했다. 작업트리 반영과 별도 artifact를 제공하며 commit/push하지 않았다.

### 실행 중 발견한 검증 도구 문제

초기 stock 시작은 새 실행 root의 IPC 디렉터리 누락으로 실패했다. 완료한 Riley r0 arm은 보존하고 실패 stock만 재개했다. 자식 벤치마크 실패 상태를 기록하면서도 상위 종료 코드를0으로 반환하던 helper도 수정하고 CPU exit-code3 전달 검사를 추가했다.

재개 옵션의 `repeats`가 count가 아닌 exclusive stop index여서 coverage 검사가 누락된 r4를 검출했다. 완료한 r0..3을 재측정하거나 제외하지 않고 r4만 추가했다. 최종 matrix는 C16/C32 각각 r0..4×3엔진의30개 고유 arm을 확인했다. 원래 실패한 campaign root의 성공 arm을 개별 완전 감사하므로 감사 파일의 partial 표시와 primary matrix 완료 여부를 구분한다. 실패 시도·후속 감사 도구의 반환 코드 해석 오류도 원시 로그에 남겼다.

## 계획의 남은 범위와 다음 우선순위

| 계획 영역 | 이번 상태 | 다음 작업 |
|---|---|---|
| P0 기준선·분해 | 실제 경로·행 분포·시간 합·raw SSE/자원·source/binary 고정 | queue/admission/GPU/SSE 연결 계측과 대표 hot kernel DRAM/L2/Tensor Core counter |
| P1 초기 prefill | Norm 제출/실행 구조 개선, C16 p95 finite5회 개선 | 큰 Gate/Up·prefill Down의 native-equivalent batch/recipe 범위 확대 |
| P2 decode Gate/Up | K staging·일반 merged 레시피는 채택 실패 | native 누적 구조를 보존한 새 타일/가중치 재사용 가설을 counter로 제한해 검증 |
| P3 융합·중간 저장 | Norm의 shared/barrier 제거, singleton 분해는 느려 제외 | Down epilogue·residual 등은 비중과 BF16 경계를 확인해 독립 후보로 평가 |
| P4 K/V·QK/PV | 기존 경로 유지 | 요청/head 축 배치·paged 접근을 기존 scale/mask/softmax/reduction 순서로 검증 |
| P5 Graph | 기존 정상 Graph off 유지 | 단순 segment Graph 재활성화 대신 적격 backend와 넓은 capture 범위·고정 metadata 설계 |
| P6 overlap·일반화 | 신규 구현하지 않음 | CPU 기인 GPU idle 증거 후 reservation/EOS/취소 의존성을 보존하는 준비 중첩; 다른 모델/긴·혼합 부하 검증 |

이번 결과는 초기 tail 개선이며 처리량 격차의 근본 해소가 아니다. 앞으로도 큰 투영·Attention 구간을 우선하고, 정확도 통과만으로 느린 후보를 남기거나 수치 허용오차를 늘리지 않는다.

## 검토·재현 자료

- [2파일 patch](../benchmarks/candidates/sm80-prefill-norm-v10/candidate.patch)
- [적용 범위](../benchmarks/candidates/sm80-prefill-norm-v10/README.md)
- [layer manifest](../benchmarks/candidates/sm80-prefill-norm-v10/source-layer-manifest.json)
- [전체 primary matrix](../benchmarks/evidence/20261020-comprehensive-v10/primary-matrix-audit.json)
- [최종 지표·반복별 변화](../benchmarks/evidence/20261020-comprehensive-v10/final-statistics.json)
- [초기 p95 인접 요청](../benchmarks/evidence/20261020-comprehensive-v10/initial-cohort-tail-audit.json)
- [전체 SSE·자원 감사](../benchmarks/evidence/20261020-comprehensive-v10/independent-audit.json)
- [전체 vocabulary 검증](../benchmarks/evidence/20261020-comprehensive-v10/final-teacher-audit.json)
- [C1 확대 확인](../benchmarks/evidence/20261020-comprehensive-v10/c1-confirmation-audit.json)
- [Norm 실제 kernel 선택](../benchmarks/evidence/20261020-comprehensive-v10/norm-profile-selection-audit.json)
- [Stock 실제 Graph/Attention 선택](../benchmarks/evidence/20261020-comprehensive-v10/stock-profile-selection-audit.json)
- [최종 GPU 홀드](../benchmarks/evidence/20261020-comprehensive-v10/terminal-hold-proof.json)
- [자격 검증 종료](../benchmarks/evidence/20261020-comprehensive-v10/qualification-completion.json)
- [소스·바이너리 연결](../benchmarks/evidence/20261020-comprehensive-v10/source-binary-binding.json)

EFS 실행 root: `/mnt/efs/nlp/psyche/riley-comprehensive-v10-20261020`. 최종 바이너리: `/mnt/efs/nlp/psyche/riley-comprehensive-v10-20261020/riley-comprehensive-v10-final`. SHA256: `1dc816aadc7355ec84b061f282d3e9a85a9d575d4b65b5852e56001b160c4639`. 무거운 BF16/trace/binary는 EFS의 raw-heavy-index와 함께 보존하며 압축 증거의 CHECKSUMS로 내려받은 자료를 확인할 수 있다.
