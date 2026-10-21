# Coalesced Gate/Up + native-order QK/PV v16G

2026-10-21 KST. **미채택. 기존 runtime은 v15C를 유지한다.** 정상30arms를 모두 보존했다. 아래는 동일GPU7에서 새로 비교한 v15C·후보·stock의5반복 중앙값이다. 다른campaign 수치와 빼서 개선율을 만들지 않는다.

## C16/C32 정상 비교

| C | v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate/stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 735.892 | 747.695 | +1.60% | 1143.677 | 65.38% |
| 32 | 979.990 | 998.974 | +1.94% | 1690.953 | 59.08% |

지연 단위ms, 기존→후보 / stock. TPOT은 요청별 평균 output-token 지연의 percentile이며 token-level ITL이 아니다.

| C | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 81.424→80.836 / 83.103 | 162.410→162.926 / 110.291 | 20.251→19.999 / 12.532 | 21.790→21.329 / 13.093 | 673.583→661.915 / 421.597 |
| 32 | 246.809→243.580 / 163.177 | 337.540→329.515 / 215.892 | 31.764→30.835 / 17.282 | 38.462→38.151 / 21.554 | 1010.106→983.350 / 549.271 |

기존 채택 기준은 두 concurrency의 처리량 증가 및 paired TTFT/TPOT p95/p99의2% 초과 악화 없음이다. 전체 경고:

- C16 r0: ttft_ms_p95 +2.42%.
- C32 r3: ttft_ms_p95 +5.66%.
- C32 r3: ttft_ms_p99 +5.64%.
- C32 r3: tpot_ms_p99 +8.74%.
- C32 r4: tpot_ms_p99 +12.37%.

불리한 표본도 포함한 [전체5회 수치](../benchmarks/evidence/20261021-mlp-batch-v16g/final-statistics.json)를 보존했다. 구성요소 이득이나 유리한 중앙값으로 tail 기준을 면제하지 않는다. 큰 처리량 향상 목표는 달성하지 못했다.

## 구조 변경

별도 native M576 prefill plan으로 suffix의 대상 prefill 그룹들을 N19456/K2560 Gate/Up 한 GEMM으로 합친다. 원래 각 그룹은65..130행, 전체는65..576행이어야 한다. fused SiLU·prefix MLP Graph·별도 decode31 plan과 대상 밖 입력의 기존 경로를 유지한다. pinned prefill family는21/20/11/0/0/0/0이다. raw scratch는 weights upload후 plan/Graph 준비 전에 cold 확보한다(최대576×19456×2B). byte report를 갱신하며 hot resize는 추가하지 않는다.

독립 재빌드한 G는 F의 empty-slot guard 오류를 고쳤다. 실제 graph handle·warmed/replayed slot·captured step은 거부하고, 아직 사용하지 않은 미리 확보된 slot은 허용한다. B의 native-order QK shuffle과 four-warp PV CTA도 포함한다. BF16 경계·native 누적 순서·softmax·paged layout은 유지한다. 결합 실험이므로 각 변경의 서버 기여도를 분리해서 주장하지 않는다.

[정확한6파일 패치](../benchmarks/candidates/native-coalesced-v16g/candidate.patch)와937-source lineage를 보존했다. 원래 Down은 이미161..576행을 합쳐 Lt6으로 실행하므로 이번 변경을 새 Down batching 이득으로 표현하지 않는다. 새 Down pipeline/Lt 탐색, KV layout/FP32 expansion, small mixed Gate/Up은 A/C/E에서 검사 후 탈락했다. TF32 KV는 literal bytes가 달랐다. 실패 결과도 [A Gate/Up·Down·KV](qwen-mlp-decode-v16a-20261021.md), [C Down 탐색·서버 비교](qwen-mlp-prefill-v16c-20261021.md), [E small mixed](qwen-mlp-small-v16e-20261021.md)에 남겼다.

## 정확도·계약 검사

원래 native/HF full-vocabulary BF16 4,480파일 + serving-capacity576 추가2,048파일, 총 **6,528 literal equality**를 통과했다. reference/tolerance는 바꾸지 않았다. 추가 검사는 같은 natural128의 R1..32/chunks4로 merged256/384/512 및 serving cold resize를 실행한다. 모든 prompt×R Cartesian coverage라는 주장은 아니다.

CPU 524 pass / 40 ignored, 실제 GPU 계약6개를 통과했다. 기존120 alignment/view 사례와 새 raw merged66사례(11shapes×3inputs×per-op/batch)를 포함한다. 원래4개 C02 fixture tree만 승인된 private/tmp에서 검사 후 삭제했다. 바이너리·모델·테스트·벤치마크·나머지 scratch는 EFS다. source937/빈 target 독립 빌드이며, 검증과 측정 후 source/binary/runner/normal argv hash를 재검사했다.

## 프로파일 진단·실제 dispatch

정확도 검사와64warm 이후 별도64keep만 Nsight로 포착했다. 정상192keep 시간에 profiler를 넣지 않았다. kernel-duration 합계와 CPU CUDA API wait는 겹치므로 host overhead로 더하지 않는다. 같은 HTTP 요청이어도 batch 구성·호출수가 달라질 수 있어 aggregate 차이를 고정 작업량 kernel speedup으로 계산하지 않는다. DRAM/L2/SM counter는 확보하지 못했으므로 bandwidth 포화로 확정하지 않는다.

| C/mode | Kernel sum ms | Gate/Up ms | Down ms | KV projection ms | QK/PV ms | Gate/Up+Down+KV share | QK/PV share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 before | 1674.345 | 491.874 | 256.750 | 190.354 | 160.306 | 56.08% | 9.57% |
| 16 after | 1642.171 | 467.225 | 255.415 | 190.029 | 153.490 | 55.58% | 9.35% |
| 32 before | 1330.133 | 362.219 | 197.836 | 171.886 | 139.210 | 55.03% | 10.47% |
| 32 after | 1308.997 | 349.789 | 198.420 | 173.412 | 127.700 | 55.13% | 9.76% |

NVTX로 직접 귀속된 prefill vendor Gate/Up GEMM 시간·호출수(전체 Gate/Up·SiLU와 구분):

- C16 before: 176.847ms / 2376calls.
- C16 after: 152.170ms / 1188calls.
- C32 before: 162.908ms / 2088calls.
- C32 after: 153.111ms / 1116calls.

C32 후보 trace의 prefill Gate/Up vendor GEMM은 전체 kernel 합계의11.70%(153.111/1308.997ms)다. Gate/Up·Down·KV projection 전체는55.13%다. 이번 merged prefill 경로의 영향 범위가 전체 projection 비용보다 작아 서버 이득도 제한됐다는 해석을 뒷받침한다. 이 비중을 정상 HTTP 처리량의 인과적 기여도로 확정하거나, 서로 다른 배치의 aggregate 차이를 kernel 개선율로 사용하지 않는다. 남은 decode Gate/Up·Down·KV projection 병목은 이번 실험으로 해결하지 못했다.

Nsight는 EFS scratch에 원격 filesystem 성능 경고를 남겼다. 승인된/tmp 예외는4개 C02 fixture에만 적용하므로 profile scratch는 EFS에 유지했다. profile의 span/idle 차이를 정상 host speedup으로 해석하지 않는다. 별도64keep trace는 정상192keep r3/r4 tail 악화의 원인 증명도 아니다.

실제로 관측한 merged 행수: C16: [115, 120, 125, 128, 130, 241, 245, 246, 250, 253, 254, 256, 258, 374, 493, 497, 499, 506, 512]; C32: [115, 120, 125, 126, 128, 130, 245, 246, 253, 254, 258, 373, 386, 494, 497, 499, 503, 505, 507, 509, 514]. compile된 소스만 보고 추정하지 않고 실제 경로 실행을 확인했다. [PV 실제 geometry](../benchmarks/evidence/20261021-mlp-batch-v16g/actual-pv-dispatch-proof.json)에서도 before gridX8/block32와 after gridX2/block128을 두 concurrency에서 확인했다. standalone gather 시간은 fused paged KV 읽기를 제외하므로 전체 KV attention 비용이 아니다.

## 비교 조건·홀드·보존

Qwen3-4B BF16 greedy, token budget576, max sequence1024, KV blocks=C×64, prefix cache off.64warm+192keep,12fixed prompts115..130tokens 및 균형 output8/16/24/32, 정상arm당3840output tokens다. Riley는 명시적 owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1이다. stock vLLM0.27.1+cu129/Torch2.13+cu129 FA2/FULL_AND_PIECEWISE의10startup 로그를 확인했다. stock은 Riley 정확도 기준이 아니며, 모든 모델/default 설정으로 확대하지 않는다. 노드 전체 CPU/I/O 격리는 아니다.

모든 유한 GPU 작업은7에서 직렬 실행했다. GPU6 외부 작업은 건드리지 않았다. original holder PID2027914/2027915, ticks6083215119/6083215121을 해제·재시작하지 않았다. 마지막3개 fresh idle20GiB 표본에서 둘을 검증했고 GPU7에는 holder만 남았다. 각 유한 GPU7 작업의 실제 전체 device resource 표본도22,528MiB 이하로 확인했다.

[Archive](../benchmarks/evidence/20261021-mlp-batch-v16g.zip): 5,996,613B/647entries, SHA256 `90ce769950a5ea8361ce83e16b152ab08e376883b950febbe37f850fd6aa082e`. Independent SSE audit verifies34 arms and6490stored values, with exactly30 normal and4 separate profile arms. Candidate binary SHA256 `3eacb06825c4cdd0effc693bc3e45c546672cc0a88e2c2725e979947db75fd86`. Large BF16/ELF/SQLite/Nsight artifacts are referenced with SHA/size by the archived EFS heavy index. [Evidence index](../benchmarks/evidence/20261021-mlp-batch-v16g/README.md) includes source reconstruction and local CRC/SHA validation.
