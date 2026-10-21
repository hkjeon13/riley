# Mixed-batch executor and committed-event overlap v17B4

2026-10-21 KST. **미채택**. 두 독립 실험은 같은 v15C에서 출발한다. 아래는 자체 fresh controls만 비교하며 이전 campaign 중앙값을 빼서 개선율을 만들지 않는다.

## C16/C32 정상 비교

| C | v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate/stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 746.287 | 748.243 | +0.26% | 1145.795 | 65.30% |
| 32 | 975.412 | 985.618 | +1.05% | 1679.361 | 58.69% |

지연ms, 기존→후보 / stock. TPOT은 요청별 평균 output-token 지연의 percentile이다.

| C | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 79.955→79.700 / 82.948 | 163.006→162.569 / 110.049 | 20.152→20.201 / 12.371 | 21.431→21.655 / 13.166 | 662.036→659.708 / 417.064 |
| 32 | 257.114→247.202 / 163.163 | 341.182→334.036 / 216.063 | 31.645→31.509 / 18.285 | 38.031→37.999 / 21.647 | 1015.186→1001.198 / 550.236 |

C16 paired throughput: r0 +1.47%, r1 +0.45%, r2 +1.62%, r3 +0.52%, r4 -1.86%

C32 paired throughput: r0 +0.45%, r1 +0.86%, r2 +2.13%, r3 +0.96%, r4 +1.38%

기존 기준은 두 concurrency 처리량 증가와 paired TTFT/TPOT p95/p99의2% 초과 악화 없음이다. 모든 경고:

- C16 r4: ttft_ms_p95 +4.84%.
- C16 r4: tpot_ms_p99 +5.88%.
- C32 r0: tpot_ms_p99 +3.01%.
- C32 r3: tpot_ms_p99 +2.63%.

정상30arms는 전부 보존했다. A1과 B4 모두 기존 tail 기준을 넘으므로 채택·결합하지 않았고 main 실행 경로는 v15C를 유지한다.

## 구조 변경과 범위

독립 baseline v15C에서 Rust7파일을 바꾼다. decode prefix + fresh prefill suffix를 canonical 순서로 pack하고 원래 output-slot 의미를 유지한다. 시작 단계에만 최대24개의 exact mixed full-body Graph를 만든다. embedding·36layers·LM head를 포함하며 report/argmax/D2H·권위 있는 완료 처리는 밖에 둔다. 대상은 decode R2..31 + fresh prefill65..130, decode context24..216, canonical slots/output order다. 시작 예열은 R7/15/23/31 × L115/120/125/126/128/130이고 실제 capacity에 따라 줄어든다. 미리 만든 키가 없으면 eager다. hot capture는 추가하지 않았다.

키에 dense rows/R/L/output mode와 **prefill block-table subspan offset**을 포함하고 stream의 고유 lifetime identity를 보유한다. stream 변경 시 Graph를 먼저 폐기하고 handle을 재결합한다. 부모 capture 안의 MLP는 원래 연산별 산술 경로를 쓰며 중첩 Graph 금지 조건을 제거하지 않는다. 따라서 작은 exact-key cache의 miss는 여전히 launch 비용을 지불한다.

별도 bounded2-job worker가 이미 commit된 event/channel/atomic cancellation만 소유한다. scheduler·GPU 소유권·rollback은 engine thread에 남긴다. 첫 토큰과 종료 이벤트는 즉시 enqueue하고 이후 committed token은 다음 GPU 제출 직후 worker에 넘긴다. GPU 없는 step도 즉시 보낸다. 종료 전에 drain/join한다. 이는 **완료된 토큰의 전달과 GPU 실행을 겹치는 범위**이며 다음 iteration의 scheduler·metadata 준비 전체를 GPU와 한 단계 앞서 수행하는 구조는 아니다. 순서·취소·첫 토큰 즉시 전달의3개 unit checks를 추가했다.

[vLLM Graph 정책](https://docs.vllm.ai/en/latest/design/cuda_graphs/)과 [SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)를 참고했다. 두 라이브러리의 전체 backend나 one-batch-ahead scheduler와 동등한 구현/성능을 주장하지 않는다.

B1은 static key의 block-table offset 누락을 GPU 실행 전 발견해 차단했다. B2는 이를 수정하고 즉시 비동기 전달을 넣었으며, B3는 이후 토큰의 after-submit 전달을 추가했다. B4는 first token도 지연될 수 있는 B3 설계를 보완했다. B2/B3는 원래4480+capacity2048을 통과했고 B3 canonical1024도 통과했다. B1 preflight와 B2/B3의 패치·로그·검사기 실패/수정·재개 영수증은 archive에 그대로 보존한다. 이 revision들은 정상 비교 표본에 섞지 않았다.

시작 mixed capture24개가 추가되어 기존 검사기의 총62개 가정이 실패했다. 수정 검사기는 mixed warmup 바로 앞24개 capture/replay 쌍을 정확히 검증·분리하고 원래 pure-body62쌍·R2..32 키·모든 BF16 바이트 검사를 그대로 요구한다. numerical mismatch를 예외 처리한 것이 아니다. 원본 검사기와 실패 영수증도 보존했다.

## 정확도·계약

원래 native/HF full-vocabulary BF164480파일과 같은 원본 reference를 쓰는 serving-capacity576 추가2048파일 및 canonical mixed 출력순서 추가1024파일을 통과했다. 총 **7552 literal BF16 equality**, CPU **527 pass /40ignored**, 실제 GPU 계약5개와 기존120 alignment/view 사례다. reference/tolerance는 그대로다. 모든 prompt×R Cartesian 검사는 아니다. capacity 추가검사는 natural128 R1..32/chunks4로 M256/384/512를 실행한다. inherited bounded audit의 cold raw scratch-resize 문구는 이 후보에 해당하지 않으며 extra-coverage-scope.json에 바로잡았다.

원래4개 C02 fixture tree만 승인된 private/tmp에서 재검사 후 삭제했다. 모델·바이너리·벤치마크·나머지 scratch는 EFS다. 빈 target으로937-source 독립 빌드했고 검증/측정 후 source·binary·runner·정상 argv hash를 재검사했다.

## 별도 프로파일

| C/mode | Kernel sum ms | Gate/Up ms | Down ms | KV projection ms | QK/PV ms |
|---|---:|---:|---:|---:|---:|
| 16 before | 1667.272 | 488.260 | 254.826 | 191.713 | 160.543 |
| 16 after | 1652.365 | 485.281 | 254.365 | 188.185 | 160.673 |
| 32 before | 1333.759 | 363.525 | 196.641 | 174.161 | 139.359 |
| 32 after | 1319.693 | 361.483 | 195.062 | 169.852 | 139.506 |

CUDA API 호출수와 GPU kernel busy 비율(별도 profile 구간):

| C/mode | Kernel launch API calls | Graph launch API calls | Stream synchronize calls | Kernel busy/span |
|---|---:|---:|---:|---:|
| 16 before | 45571 | 1256 | 204 | 79.79% |
| 16 after | 42755 | 1186 | 204 | 82.80% |
| 32 before | 44842 | 1184 | 130 | 74.85% |
| 32 after | 41882 | 1114 | 130 | 68.28% |

64warm 이후 별도64keep만 Nsight로 포착했다. 정상192keep에는 profiler를 넣지 않았다. CPU CUDA wait는 GPU 실행과 겹치므로 kernel 시간에 더하지 않는다. batch 구성/호출수가 달라 aggregate 차이를 고정 작업량 kernel speedup으로 계산하지 않는다. DRAM/L2/SM counter가 없어 bandwidth 포화로 확정하지 않는다. mixed body의 opaque vendor GEMM은 pure-decode 단계 순서로 억지 귀속하지 않는다. EFS scratch에 대한 Nsight 경고도 보존했다.

시작 단계 mixed Graph는 C16에서12개, C32에서24개 생성됐지만, 별도 측정 구간에서 실제 mixed full-body Graph 실행은 **양쪽 모두0개**였다. 원래 residual72·KV gather·QK 분류에 compact-prefill staging을 추가한 별도 재검사도0개였다. staging 커널2376개씩은 Graph 밖에서 실행됐으며 generic decode fallback도 포함하므로 전부 prefill로 해석하지 않는다. 추가 검사 expanded-mixed-marker-audit.json은 archive 생성 후 저장됐고 해당 trace SHA를 기록한다. 따라서 이 결과로 mixed Graph 가속을 입증할 수 없다. block-table CPU offset을 키에 포함하고 출력순서도 제한하는 구조가 있으나 step별 miss 사유 counter가 없어 각 조건의 기여율은 알 수 없다. publication-worker-proof는 whole-process after-submit 전달과 drain을 검증하며 correctness/warmup도 포함하므로 측정 구간의 인과적 기여율은 아니다.

## 비교 조건·보존·홀드

같은 physical A100 GPU7, Qwen3-4B BF16 greedy, budget576, max sequence1024, KV=C×64, prefix cacheoff. 각 engine/C마다 fresh process5회,64warm+192keep,12fixed prompts115..130 및 균형 output8/16/24/32, 정상 arm당3840output tokens다. Riley recipe는 owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1이다. stock vLLM0.27.1+cu129/Torch2.13+cu129, FA2/FULL_AND_PIECEWISE/version2 startup10개를 확인했다. stock은 Riley 정확도의 reference가 아니다.

후보의 원래 유한 검증·정상·프로파일은 GPU7에서 직렬 실행했다. B4의 추가 canonical 검사와 이전 B3 revision의 numerical/canonical은 GPU6에서 실행했고 정상 timing 전 모두 끝냈다. 외부 작업은 건드리지 않았다. original holder6/7 PID2027914/2027915, ticks6083215119/6083215121을 해제·재시작하지 않았다. 마지막3개 fresh 표본에서 idle20GiB와 동일 identity를 확인했다. finite whole-device resource는22528MiB 이하로 검증했다. 서버 전체 CPU/I/O 독점은 아니다.

[Archive](../benchmarks/evidence/20261021-overlap-v17b4.zip): 6,712,910B / 979entries, SHA256 `ff68b26a35add42143904c65f75e0a745e88ebbce21bc9543f8f47ce988fd87a`. Independent audit는34arms와6490stored SSE values를 검증했다. [전체5회 통계](../benchmarks/evidence/20261021-overlap-v17b4/final-statistics.json), [정확한 패치](../benchmarks/candidates/overlap-v17b4/candidate.patch), [근거 및 local CRC/SHA/source reconstruction](../benchmarks/evidence/20261021-overlap-v17b4/README.md). Candidate binary SHA256 `9459cfb3940901c1fcfa6df09fcc453b3b59eabccbcaf094cb4dc787df61d008`. 대형 BF16/ELF/SQLite/Nsight는 archive의 EFS SHA/size index로 보존한다.
