# Decode projection PTX v17A: serving Gate/Up and current-route Down screen

2026-10-21 KST. **미채택**. 두 독립 실험은 같은 v15C에서 출발한다. 아래는 자체 fresh controls만 비교하며 이전 campaign 중앙값을 빼서 개선율을 만들지 않는다.

## C16/C32 정상 비교

| C | v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate/stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 735.627 | 743.473 | +1.07% | 1155.362 | 64.35% |
| 32 | 978.131 | 981.322 | +0.33% | 1691.051 | 58.03% |

지연ms, 기존→후보 / stock. TPOT은 요청별 평균 output-token 지연의 percentile이다.

| C | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 81.160→82.186 / 82.181 | 164.155→163.789 / 110.177 | 20.362→20.219 / 12.416 | 21.486→21.535 / 12.985 | 672.482→661.668 / 416.450 |
| 32 | 254.573→249.948 / 162.825 | 339.764→336.174 / 215.719 | 31.777→31.565 / 17.067 | 36.868→38.600 / 21.625 | 1005.734→1003.083 / 548.593 |

C16 paired throughput: r0 +1.06%, r1 +0.11%, r2 -0.38%, r3 +1.75%, r4 +1.88%

C32 paired throughput: r0 -0.49%, r1 +0.00%, r2 +0.33%, r3 +1.05%, r4 +1.06%

기존 기준은 두 concurrency 처리량 증가와 paired TTFT/TPOT p95/p99의2% 초과 악화 없음이다. 모든 경고:

- C16 r1: tpot_ms_p99 +5.39%.
- C16 r4: ttft_ms_p95 +5.77%.
- C32 r0: ttft_ms_p95 +3.07%.
- C32 r0: ttft_ms_p99 +2.89%.
- C32 r0: tpot_ms_p99 +7.75%.
- C32 r1: tpot_ms_p99 +3.74%.
- C32 r3: tpot_ms_p99 +4.62%.

정상30arms를 전부 보존했다. 이 후보는 tail 기준에서 탈락했으므로 B4와 결합하지 않는다. B4의 단독 효과는 별도로 판단한다. 큰 처리량 개선 목표는 달성하지 못했다.

## 구조 변경과 범위

독립 baseline v15C에서 C++ 한 파일만 바꾼다. WMMA보다 작은 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` fragment로 실제 Gate/Up R2..8을 교체한다. custom Down R5..16 분기도 수정했지만, 현재 mode10의 full-projection Lt 분기가 앞서 선택하므로 서빙 Down 교체는 적용되지 않았다. 초기 분기 판단을 바로잡으며 이 정상 비교를 Gate/Up 효과로만 보고한다. Gate/Up의 두 FP32 chain·K16 누적 순서·BF16 경계를 유지한다. 새 Down component도 기존4개 FP32 partial과 BF16 reduction 경계를 재현한다. R9..16 Gate/Up과 R17+ Lt31, KV projection은 원래 경로다. 전체 projection 비용을 모두 교체했다는 주장이 아니다.

13개 R×6개 finite 상태×2연산의156 component cases가 literal equality를 통과했다. component의 R17+ 비교 대상은 구 custom WMMA이며 실제 현재 Lt31이 아니므로 서버 채택 근거로 사용하지 않았다. 추가 A2 `ldmatrix.x4/x2` operand staging screen도156개를 통과했지만 유의한 개선을 보이지 않아 서버 후보에 넣지 않았다. 원본 소스·실행 로그·finite observer 영수증은 archive의 research 및 operand-staging-screen에 보존했다.

NVIDIA의 [PTX fragment 정의](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/ptx_isa_8.4.pdf)와 [CUTLASS tile/warp 구조](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)를 참고했다. 이 후보는 CUTLASS backend를 통째로 도입한 것이 아니다. 실제 Nsight의 v17_gateup/v17_down별 실행수는 projection-route-audit.json에 기록한다. Down 시간의 aggregate 차이를 새 Down의 이득으로 주장하지 않는다. 추가 current-route-screen.json은 현재 Down Lt31/15/18/2/4/1/0 및 R17+ Gate/Up merged Lt31/15/17/0/0/0/0+원래 BF16 SiLU를 정확히 복제해 비교한 별도 component screen이다. 원래 source/binary/정상 표본을 바꿔 결과를 만든 것이 아니다.

현재 실제 recipe 대조 component: 156/156개 finite 사례가 일치했다. 불일치 사례와 unsupported status도 원자료에 남긴다. 아래는 state0에서 바이트가 일치한 경우만의µs다. 셀을 통과한 것으로 전체 모델을 통과했다고 보지 않는다.

| Op | R | Current recipe µs | PTX candidate µs |
|---|---:|---:|---:|
| Down | 2 | 42.409 | 45.100 |
| Down | 8 | 43.028 | 46.511 |
| Down | 16 | 43.478 | 48.152 |
| Gate/Up | 17 | 82.266 | 86.771 |
| Gate/Up | 24 | 83.581 | 87.139 |
| Down | 24 | 44.146 | 53.643 |
| Gate/Up | 32 | 83.270 | 87.627 |
| Down | 32 | 44.444 | 54.379 |

실제 server trace의 새 Gate/Up 호출수는 C16 684회, C32 468회다. 새 Down은 둘 다0회다. 새 Gate/Up 시간은 각각56.788ms/1662.982ms(**3.41%**),39.087ms/1308.160ms(**2.99%**)로, 별도 profile의 전체 kernel 합에서 차지하는 범위가 작다. 전체 projection 병목을 교체했다는 초기 가정은 맞지 않았다. 이 범위 제한은 큰 이득이 없었던 이유를 설명하는 근거이며, 별도64keep trace의 비중을 정상192keep 처리량 기여율로 환산하지 않는다.

더 큰 backend 구조로는 K64의 두 K32 native chain을 각 K1280으로 cold pack하고, 두 BF16 GEMM의 FP32 output을 원래 순서로 합친 뒤 기존 BF16 Gate/Up·SiLU 경계를 적용했다. 입력 packing·GEMM·reduction을 시간에 포함하고 cold weight packing만 제외했다. 17 cuBLAS algorithm selector × R2..16 ×6상태의 1530개 결과 중 바이트 일치 1530개, 17개 selector 모두 각90개 사례를 통과했다. Selector들이 같은 Ampere 구현으로 귀결될 수 있으므로17개 서로 다른 kernel이라는 주장은 아니다. 이 결과는 별도 component screen이며 server source에 넣은 결과가 아니다.

기본 selector99의 state0 old→packed 시간은 R2 79.795→81.142µs, R8 81.163→84.304µs, R16 82.010→86.854µs다. R16에서 **5.91% 느려** 서버에 통합하지 않는다. 새 Down도 현재 Lt 대조보다 R16에서43.478→48.152µs로 느리다. 이 component 속도 비율은 HTTP 처리량 변화가 아니다. 더 큰 구조여도 실제 현재 경로보다 느리면 적용하지 않는다.

## 정확도·계약

원래 native/HF full-vocabulary BF164480파일과 같은 원본 reference를 쓰는 serving-capacity576 추가2048파일을 통과했다. 총 **6528 literal BF16 equality**, CPU **524 pass /40ignored**, 실제 GPU 계약5개와 기존120 alignment/view 사례다. reference/tolerance는 그대로다. 모든 prompt×R Cartesian 검사는 아니다. capacity 추가검사는 natural128 R1..32/chunks4로 M256/384/512를 실행한다. inherited bounded audit의 cold raw scratch-resize 문구는 이 후보에 해당하지 않으며 extra-coverage-scope.json에 바로잡았다.

원래4개 C02 fixture tree만 승인된 private/tmp에서 재검사 후 삭제했다. 모델·바이너리·벤치마크·나머지 scratch는 EFS다. 빈 target으로937-source 독립 빌드했고 검증/측정 후 source·binary·runner·정상 argv hash를 재검사했다.

## 별도 프로파일

| C/mode | Kernel sum ms | Gate/Up ms | Down ms | KV projection ms | QK/PV ms |
|---|---:|---:|---:|---:|---:|
| 16 before | 1669.164 | 488.268 | 259.110 | 192.104 | 160.351 |
| 16 after | 1662.982 | 487.244 | 256.278 | 190.945 | 160.467 |
| 32 before | 1320.166 | 359.541 | 194.787 | 170.683 | 139.528 |
| 32 after | 1308.160 | 350.621 | 192.487 | 172.772 | 139.545 |

CUDA API 호출수와 GPU kernel busy 비율(별도 profile 구간):

| C/mode | Kernel launch API calls | Graph launch API calls | Stream synchronize calls | Kernel busy/span |
|---|---:|---:|---:|---:|
| 16 before | 44847 | 1221 | 204 | 75.58% |
| 16 after | 44487 | 1221 | 204 | 76.42% |
| 32 before | 42930 | 1149 | 130 | 73.35% |
| 32 after | 42493 | 1114 | 130 | 78.88% |

64warm 이후 별도64keep만 Nsight로 포착했다. 정상192keep에는 profiler를 넣지 않았다. CPU CUDA wait는 GPU 실행과 겹치므로 kernel 시간에 더하지 않는다. batch 구성/호출수가 달라 aggregate 차이를 고정 작업량 kernel speedup으로 계산하지 않는다. DRAM/L2/SM counter가 없어 bandwidth 포화로 확정하지 않는다. mixed body의 opaque vendor GEMM은 pure-decode 단계 순서로 억지 귀속하지 않는다. EFS scratch에 대한 Nsight 경고도 보존했다. 새 v17_gateup도 MLP/Down 전환 anchor로 인식하도록 진단 분류기를 분석 전에 수정했고 원본·수정 SHA를 profile-attribution-correction.json에 남겼다. 원래 분류기를 그대로 쓰면 unchanged Lt Down을 Gate/Up에 잘못 귀속할 수 있다. 정상 수치·바이너리·정확도 reference는 바꾸지 않았다. 이 별도 trace로 정상 r1/r4 tail 악화의 인과 원인을 확정하지 않는다.

actual-candidate-dispatch.json과 projection-route-audit.json에서 실제 Gate/Up 실행수와 Down 실행 유무를 검증한다. Down은 현재 Lt 경로가 우선하므로 이 서버 비교의 Down 개선 주장에 쓰지 않는다.

## 비교 조건·보존·홀드

같은 physical A100 GPU7, Qwen3-4B BF16 greedy, budget576, max sequence1024, KV=C×64, prefix cacheoff. 각 engine/C마다 fresh process5회,64warm+192keep,12fixed prompts115..130 및 균형 output8/16/24/32, 정상 arm당3840output tokens다. Riley recipe는 owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1이다. stock vLLM0.27.1+cu129/Torch2.13+cu129, FA2/FULL_AND_PIECEWISE/version2 startup10개를 확인했다. stock은 Riley 정확도의 reference가 아니다.

후보의 원래 유한 검증·정상·프로파일은 GPU7에서 직렬 실행했다. B4의 추가 canonical 검사와 이전 B3 revision의 numerical/canonical은 GPU6에서 실행했고 정상 timing 전 모두 끝냈다. 외부 작업은 건드리지 않았다. original holder6/7 PID2027914/2027915, ticks6083215119/6083215121을 해제·재시작하지 않았다. 마지막3개 fresh 표본에서 idle20GiB와 동일 identity를 확인했다. finite whole-device resource는22528MiB 이하로 검증했다. 서버 전체 CPU/I/O 독점은 아니다.

[Archive](../benchmarks/evidence/20261021-projection-v17a.zip): 6,058,944B / 701entries, SHA256 `02aca9c1fdd3d9e3f76b1b2e656f68b4856d0161dd874e8717c5b6f65b120832`. Independent audit는34arms와6490stored SSE values를 검증했다. [전체5회 통계](../benchmarks/evidence/20261021-projection-v17a/final-statistics.json), [정확한 패치](../benchmarks/candidates/projection-v17a/candidate.patch), [근거 및 local CRC/SHA/source reconstruction](../benchmarks/evidence/20261021-projection-v17a/README.md). Candidate binary SHA256 `2f3be741db508fea7feb835a1d87b45a7f4db0c587e1ff0ee35db2ef1eef6efd`. 대형 BF16/ELF/SQLite/Nsight는 archive의 EFS SHA/size index로 보존한다.
