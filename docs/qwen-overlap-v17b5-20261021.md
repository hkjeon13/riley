# Mixed-batch fixed-metadata Graph and committed-event overlap v17B5

2026-10-21 KST. **미채택**. 두 독립 실험은 같은 v15C에서 출발한다. 아래는 자체 fresh controls만 비교하며 이전 campaign 중앙값을 빼서 개선율을 만들지 않는다.

## C16/C32 정상 비교

| C | v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate/stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 738.330 | 744.243 | +0.80% | 1148.787 | 64.79% |
| 32 | 980.503 | 986.012 | +0.56% | 1680.887 | 58.66% |

지연ms, 기존→후보 / stock. TPOT은 요청별 평균 output-token 지연의 percentile이다.

| C | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 80.525→81.552 / 82.736 | 163.370→162.251 / 110.195 | 20.192→20.053 / 12.332 | 21.511→21.183 / 13.103 | 671.296→666.363 / 418.164 |
| 32 | 248.491→247.848 / 162.590 | 340.386→333.208 / 215.566 | 31.986→31.918 / 17.275 | 37.892→37.716 / 21.590 | 1005.121→1005.090 / 555.532 |

C16 paired throughput: r0 -1.15%, r1 +2.40%, r2 -0.42%, r3 -1.53%, r4 +0.98%

C32 paired throughput: r0 +0.34%, r1 +1.40%, r2 -0.90%, r3 +0.99%, r4 +1.03%

기존 기준은 두 concurrency 처리량 증가와 paired TTFT/TPOT p95/p99의2% 초과 악화 없음이다. 모든 경고:

- C16 r0: ttft_ms_p95 +2.31%.
- C16 r3: ttft_ms_p95 +3.75%.
- C32 r4: tpot_ms_p99 +8.95%.

정상30arms는 전부 보존했다. A1과 B5가 모두 기존 paired tail 기준을 넘으므로 채택·결합하지 않았다. main 실행 경로는 v15C를 유지하며 큰 처리량 개선 목표는 미달이다. B5의 실제 mixed Graph 실행은 확인했지만 위의 정상 비교를 뒤집을 채택 근거는 없다.

## 구조 변경과 범위

독립 baseline v15C에서 Rust8파일을 바꾼다. B4의 decode-first input packing·최대24개 startup mixed full-body Graph·committed-event worker에 고정 주소 metadata를 추가했다. GPU36B/pinned36B의 prefill page-id buffer를 cold allocate하고, 매 step 검증된 fresh request page ID를 기존 command batch 안에서 Graph 실행 전에 갱신한다. block-table CPU offset을 키에서 제거하되 dense rows/R/L/output mode와 stream lifetime identity를 유지한다. output index는 GPU가 읽는 동적 데이터이므로 값의 순서를 cache key 조건으로 삼지 않는다. public output-slot 의미는 그대로다. 메타데이터 범위 검증·자원 수명·명시적 종료와 device byte accounting을 유지하고 추가 동기화나 hot allocation을 넣지 않았다. ragged 예약으로 offset이 달라져도 같은 fresh page를 얻는 경우와 잘못된 table을 거절하는 두 unit checks를 추가했다.

대상은 단일 fresh prefill65..130 + decode R2..31, decode context24..216이다. 시작 예열은 R7/15/23/31 × L115/120/125/126/128/130으로 capacity에 따라 줄어든다. embedding·36layers·LM head를 Graph에 포함하고 argmax/report/D2H·완료 처리는 밖에 둔다. 미리 준비하지 않은 shape 및 multi-prefill은 eager로 실행한다. 연산별 BF16/FP32 산술과 C++ backend는 v15C 그대로이며 부모 capture 안의 MLP는 원래 연산을 기록하고 중첩 replay 제한을 유지한다.

별도 bounded2-job worker는 이미 commit된 event/channel/atomic cancellation만 소유한다. 첫 토큰과 종료 이벤트는 즉시 전달하고 이후 committed token은 다음 GPU 제출 직후 넘긴다. 종료 전에 drain/join한다. 이는 완료된 토큰 전달과 GPU 실행을 겹치는 범위다. 다음 iteration의 scheduler·metadata 준비 전체를 한 단계 앞서 수행하는 구조까지 구현한 것은 아니다.

[vLLM Graph 정책](https://docs.vllm.ai/en/latest/design/cuda_graphs/)과 [SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [FlashInfer v0.5.0 고정 GPU metadata buffer 구현](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.0/flashinfer/decode.py)을 참고했다. FlashInfer kernel을 직접 도입한 결과는 아니다.

선행 B4는 numerical7552개를 통과했으나 정상 처리량 C16+0.26%/C32+1.05%, paired tail 기준 실패로 미채택했다. 시작 mixed Graph12/24개는 있었으나 측정 중 실행0이었다. 본 후보는 해당 cache 제약을 고친 별도 source/binary이며 B4 표본을 섞지 않는다. A1도 tail 기준 실패했으므로 projection 후보와 결합하지 않았다. B1 preflight와 B2/B3의 검사기 실패/수정·재개 기록을 보존하며, 추가 mixed warmup capture를 원래62개 pure-body 검사에서 분리한 것은 numerical equality 예외가 아니다.

별도64keep trace의 실제 mixed Graph 실행(단위ms, startup 제외):

| C | Mixed Graphs | Complete launches | Mixed kernel sum | All-kernel share |
|---:|---:|---:|---:|---:|
| 16 | 4 | 4 | 93.375 | 5.55% |
| 32 | 1 | 1 | 24.561 | 1.84% |

이 비율은 profiler64keep의 kernel 합이며 정상192keep의 인과적 speedup이나 eligible batch hit-rate가 아니다. 아래 Gate/Up·Down·KV 표에는 mixed Graph 안의 opaque vendor GEMM을 순서만으로 분류하지 않았다. 따라서 분류량 감소를 projection kernel 가속으로 해석하지 않는다.

## 정확도·계약

원래 native/HF full-vocabulary BF164480파일과 같은 원본 reference를 쓰는 serving-capacity576 추가2048파일 및 canonical mixed 출력순서 추가1024파일을 통과했다. 총 **7552 literal BF16 equality**, CPU **529 pass /40ignored**, 실제 GPU 계약5개와 기존120 alignment/view 사례다. reference/tolerance는 그대로다. 모든 prompt×R Cartesian 검사는 아니다. capacity 추가검사는 natural128 R1..32/chunks4로 M256/384/512를 실행한다. inherited bounded audit의 cold raw scratch-resize 문구는 이 후보에 해당하지 않으며 extra-coverage-scope.json에 바로잡았다.

원래4개 C02 fixture tree만 승인된 private/tmp에서 재검사 후 삭제했다. 모델·바이너리·벤치마크·나머지 scratch는 EFS다. 빈 target으로937-source 독립 빌드했고 검증/측정 후 source·binary·runner·정상 argv hash를 재검사했다.

## 별도 프로파일

| C/mode | Kernel sum ms | Gate/Up ms | Down ms | KV projection ms | QK/PV ms |
|---|---:|---:|---:|---:|---:|
| 16 before | 1669.987 | 489.049 | 257.965 | 191.597 | 159.755 |
| 16 after | 1683.848 | 478.494 | 245.086 | 188.899 | 156.921 |
| 32 before | 1321.924 | 358.926 | 194.778 | 171.782 | 140.028 |
| 32 after | 1334.405 | 356.637 | 194.218 | 171.863 | 139.268 |

CUDA API 호출수와 GPU kernel busy 비율(별도 profile 구간):

| C/mode | Kernel launch API calls | Graph launch API calls | Stream synchronize calls | Kernel busy/span |
|---|---:|---:|---:|---:|
| 16 before | 44703 | 1221 | 204 | 81.28% |
| 16 after | 42739 | 1186 | 204 | 79.98% |
| 32 before | 43362 | 1149 | 130 | 84.06% |
| 32 after | 44226 | 1184 | 130 | 73.13% |

64warm 이후 별도64keep만 Nsight로 포착했다. 정상192keep에는 profiler를 넣지 않았다. CPU CUDA wait는 GPU 실행과 겹치므로 kernel 시간에 더하지 않는다. batch 구성/호출수가 달라 aggregate 차이를 고정 작업량 kernel speedup으로 계산하지 않는다. DRAM/L2/SM counter가 없어 bandwidth 포화로 확정하지 않는다. mixed body의 opaque vendor GEMM은 pure-decode 단계 순서로 억지 귀속하지 않는다. EFS scratch에 대한 Nsight 경고도 보존했다.

actual-candidate-dispatch.json의 mixed_full_body_graphs는 residual72·prefill KV gather 또는 compact-prefill staging·QK 노드를 가진 실제 실행 Graph다. publication-worker-proof는 whole-process after-submit 전달과 drain을 검증하며 correctness/warmup도 포함하므로 측정 구간의 인과적 기여율은 아니다.

## 비교 조건·보존·홀드

같은 physical A100 GPU7, Qwen3-4B BF16 greedy, budget576, max sequence1024, KV=C×64, prefix cacheoff. 각 engine/C마다 fresh process5회,64warm+192keep,12fixed prompts115..130 및 균형 output8/16/24/32, 정상 arm당3840output tokens다. Riley recipe는 owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1이다. stock vLLM0.27.1+cu129/Torch2.13+cu129, FA2/FULL_AND_PIECEWISE/version2 startup10개를 확인했다. stock은 Riley 정확도의 reference가 아니다.

후보의 원래 유한 검증·정상·프로파일은 GPU7에서 직렬 실행했다. B 계열의 추가 canonical 검사와 이전 B3 revision의 numerical/canonical은 GPU6에서 실행했고 정상 timing 전 모두 끝냈다. 외부 작업은 건드리지 않았다. original holder6/7 PID2027914/2027915, ticks6083215119/6083215121을 해제·재시작하지 않았다. 마지막3개 fresh 표본에서 idle20GiB와 동일 identity를 확인했다. finite whole-device resource는22528MiB 이하로 검증했다. 서버 전체 CPU/I/O 독점은 아니다.

[Archive](../benchmarks/evidence/20261021-overlap-v17b5.zip): 6,716,696B / 976entries, SHA256 `f423004646d54fa828c462b7e7751b5eb4509f8748e668d2d9ecb970118104b0`. Independent audit는34arms와6490stored SSE values를 검증했다. [전체5회 통계](../benchmarks/evidence/20261021-overlap-v17b5/final-statistics.json), [정확한 패치](../benchmarks/candidates/overlap-v17b5/candidate.patch), [근거 및 local CRC/SHA/source reconstruction](../benchmarks/evidence/20261021-overlap-v17b5/README.md). Candidate binary SHA256 `2f5ce7d5bf0c535a80c049be9cc23e2f581bb6ca77b0cf2a1ee10ea6e3fd0fd2`. 대형 BF16/ELF/SQLite/Nsight는 archive의 EFS SHA/size index로 보존한다.
