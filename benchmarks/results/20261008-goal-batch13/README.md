# Riley 목표 진행 기록 — 2026-10-08 (Batch13, 안정성, Qwen HTTP 정확성)

`20261008-main-checkpoint` 이후, 2026-10-08에 설정된 전체 목표(vLLM 대비 8개 핵심 조건 최소 기준 + 모델별 정확성 + 안정성, 이후 도전 목표)에 따라 진행한 작업의 기록입니다. 원자료는 `ssh ai-assistant:/data/riley-serving-261007/`에 있고, 여기에는 보고서·계획·receipt·스크립트만 복사했습니다.

## 1. 기준 복구와 재현성

- Batch12 후보(`da5ec52a`, binary `cc23b7a4…`)와 V52 기준(binary `6f424278…`), 재구성 vLLM 0.27.1을 그대로 고정해 사용했습니다. 기존 실패 시도(Batch12 attempt 1·2)는 보존되어 있습니다.
- **Batch12 attempt 3 (원래 gate, 원래 계획)**: 3/96 lane 후 `c1-fixed-r1-vllm-start` host gate timeout으로 중단. 원인은 공유 호스트의 무관한 서비스(postgres 등)가 만든 system-wide IO pressure입니다. 증거: `batch12-attempt03/`. 성능 결론 없음, 보존만 함.
- **gate v2 (별도 비교 조건으로 명시)**: CPU·memory pressure 기준은 그대로 중단 조건으로 두고, IO full pressure는 60초 대기 후 lane별로 기록(`io_elevated`)만 합니다. 근거: Batch09–11의 288 lane에서 retained 구간 IO pressure(0–10)와 처리량 편차의 상관 r=0.056. gate v1 결과와 섞지 않고, V52/Batch12/Batch13/vLLM을 한 attempt 안에서 모두 다시 측정했습니다. 계획은 측정 전에 고정(`serving/batch-plan.json`).
- 원래 gate(v1) 조건의 Batch12 전체 matrix는 **미실행**(외부 제약: 호스트 IO pressure). 재개 조건: `/proc/pressure/io` full avg10 ≤ 5가 lane 시작마다 유지되는 시간대.

## 2. Batch13 후보

실측 병목(Batch11 nsys 프로파일): C32에서 graph replay 사이의 host 구간이 wall time의 16–17%(natural 332µs, fixed 121µs/decode 반복, KV 블록 수에 비례), 그중 약 40%가 admit 안의 prompt 토큰화. C1에서는 native 패킷 검증의 byte 단위 스캔.

변경(모두 host 경로, device kernel 소스 변경 없음 — `source/`의 patch 4개):

1. wire validate / block table / execution authority의 블록 소유·중복 검증을 B-tree·HashSet·이차 스캔에서 pool 범위 bitmap·prefix 스캔으로 교체(검사 내용 동일).
2. 엔진에 활성 요청이 있을 때 prompt 토큰화를 제출 스레드에서 수행(실패 시 admit이 기존 순서대로 재현).
3. native shape packet 검증의 padding 스캔을 word 단위로.

정확성: native 3161 case BF16/FP32 exact(로그 SHA가 Batch12와 동일), HTTP60 exact screen 통과, 그리고 전체 측정·안정성의 모든 Riley 요청이 고정 token/text/finish reference와 일치.

## 3. 성능 (실제 HTTP, 128 lane, 제외 없음)

전체 표: [`serving/report.md`](serving/report.md), 수치·lane 기록: `serving/analysis.json`. 요청 행에서 독립 재계산했고 controller 요약과 일치합니다.

| 조건 | 처리량 후보/vLLM | TTFT p50 | TPOT p50 | 최소 기준 | 도전 목표 |
|---|---:|---:|---:|:-:|:-:|
| c1-fixed | 1.260 | 0.406 | 0.991 | 통과 | 미달 (TPOT) |
| c1-natural | 1.148 | 0.404 | 0.871 | 통과 | 미달 (처리량 1.148 < 1.15) |
| c8-fixed | 1.659 | 0.354 | 0.722 | 통과 | 통과 |
| c8-natural | 1.386 | 0.554 | 0.778 | 통과 | 통과 |
| c16-fixed | 1.634 | 0.297 | 0.790 | 통과 | 통과 |
| c16-natural | 1.275 | 0.525 | 0.834 | 통과 | 통과 |
| c32-fixed | 1.501 | 0.197 | 0.935 | 통과 | 미달 (TPOT) |
| c32-natural | 1.197 | 0.350 | 0.930 | 통과 | 미달 (TPOT) |

주의할 점:

- 최소 기준은 4회 반복 평균 기준으로 8개 조건 모두 통과(P95/P99 포함). "후보의 최악 반복 vs vLLM의 최고 반복"으로 보면 c32-fixed TPOT 한 항목만 뒤집힙니다(vLLM 반복 편차가 큼: 2.997 ± 0.345 ms).
- 이번 실행의 vLLM은 Batch11 실행(gate v1) 때보다 일부 조건에서 느렸습니다(예: c32-natural 10,668 vs 11,450 tok/s). Batch11의 vLLM 수치와 비교해도 후보는 모든 조건에서 기준을 넘지만, 여유가 얇은 곳은 c1-fixed TPOT(0.882 vs 0.891)와 c32-natural TPOT(2.310 vs 2.357)입니다. 서로 다른 gate 조건의 교차 비교이므로 참고용입니다.
- IO pressure가 높은 상태로 입장한 lane 6개(목록은 `analysis.json`의 `io_elevated_lanes`). c1-natural r2/r3 무렵 운영자 파일 전송으로 인한 디스크 쓰기가 있었습니다(`serving/operator-disturbance-note.json`). 어느 lane도 제외하지 않았습니다.

## 4. 안정성

[`stability/report.md`](stability/report.md). 후보 8개 조건 × (300초 지속 부하 + idle 취소 40회 + 매회 exact 재요청 + 부하 중 취소 40회 + 종료 회수), vLLM은 C32 두 조건을 참고로 실행. 10 lane 모두 통과: 요청 오류 0, Riley 전 요청 exact, 매 취소·재요청 후 active/waiting/KV 블록 0으로 drain, 종료 시 exit 0·GPU 프로세스 없음·port 닫힘·native device/pinned 할당 0.

한계: early 취소는 요청 직후 연결을 끊는 방식이라 서버 쪽 단계(대기/prefill)를 클라이언트에서 특정할 수 없습니다. 메모리는 1초 간격 표본입니다. 300초 구간에서 RSS 기울기는 0.05–0.25 MiB/min이었고, 이를 확인하기 위한 C32 natural 30분 soak 결과는 `soak/`에 있습니다.

## 5. Qwen cache-on 정확성 (HTTP)

- 기존 scheduler 경로는 Qwen에서 HF-exact가 아니며(prefill row 0부터 logits 상이, 생성 index 3에서 token 불일치) 이번에도 **그대로**입니다. exact 산술은 native probe owner에만 있었습니다.
- 추가: `--numerics-profile hf-eager-qwen-p2048-o128-probe-v1`(`hf-exact-probe` 빌드). 실제 HTTP 서비스·엔진 worker가 그 exact owner를 구동합니다(단일 2048-token prefill, contiguous KV, singleton decode). **범위는 prompt 2048 token, greedy, 출력 ≤128, 동시 1건**이며 그 밖의 요청은 400으로 거절합니다. runtime/FFI/kernel 소스는 변경 없음. 브랜치 `codex/qwen-http-hf-exact`.
- HTTP gate(`qwen-http-exact/qwen3b-http-hf-exact-c1-result.json`): streaming 요청과 취소 2회 뒤 재요청 모두 128 token이 HF teacher와 일치, logit 128행 BF16 byte-exact, 대기열 2건 exact, 활성 요청 중 shutdown 503 및 할당 0.
- 실제 `riley serve` 프로세스(`qwen-http-exact/binary-harness-result.json`): token 일치, logit 128/128행 exact, 정상 종료.
- attempt 1은 테스트 전제(워크로드 파일의 기록 출력 = HF teacher) 오류로 실패했습니다. 워크로드 파일의 `output_token_ids`는 HF 기준과 다른 생산자의 기록이었고, 기준은 기존 native gate와 같은 pinned teacher oracle입니다(tolerance·출력 길이·EOS 변경 없음).
- 미실행: Qwen의 serving 성능, 이 profile의 지속 부하 안정성, 2048-token 이외 geometry.

## 6. 남은 격차와 다음 실험

도전 목표 미달 4개 조건: c1-fixed TPOT(0.991 → 0.90 필요), c1-natural 처리량(1.148 → 1.15), c32-fixed TPOT(0.935), c32-natural TPOT(0.930).

Batch13 binary에 진단 계측만 추가한 빌드로 프로파일러 없이 잰 반복당 시간(`diagnostics/unprofiled-step-and-ffi-timing.txt`, 진단용이며 성능 근거 아님):

| 조건 | graph launch | stream sync(GPU 실행) | step 내 host 합계 | 그중 publish |
|---|---:|---:|---:|---:|
| C1 fixed | 4 µs | 945 µs (prefill 반복 포함 평균) | 약 8 µs | 0.8 µs |
| C32 natural | 10–14 µs | 2030–2270 µs | 약 130 µs | 37–42 µs |

- nsys에서 보이던 반복당 116 µs의 "launch" 구간은 프로파일러 오버헤드였고 실제 `cudaGraphLaunch`는 4–14 µs입니다. 따라서 다음 반복을 미리 launch하는 pipelining의 이득은 작습니다(기각).
- **C1은 host 여유가 없습니다.** TPOT은 사실상 GPU 실행 시간(1행 decode graph의 kernel 약 276개)입니다. c1-fixed를 도전 목표까지 내리려면 kernel 경로에서 약 9%가 필요합니다. 다음 실험: 1행 decode의 layer당 kernel 수 축소(fusion)와 FFN/attention value kernel의 1행 경로 — BF16/FP32 exact 유지가 전제.
- C32는 host가 약 6%입니다. 가장 큰 항목은 engine thread의 행별 event 전송(publish 37–42 µs), 그다음 plan(17–19), authorize(12), sample(12)입니다. 다음 실험(Batch14 후보): event 배치를 전용 publisher thread로 넘기기, plan/authorize의 행별 table 복사 제거. 예상 이득은 C32에서 2–3%로 그것만으로는 0.90에 못 미칠 수 있어 kernel 작업과 함께 진행해야 합니다.

## 7. 30분 soak

`soak/soak-summary.json`: C32 natural 1,802초, 306,048 요청 전부 exact, 오류 0, GPU 1,938 MiB 고정, round 사이 KV 0. RSS는 760.9 → 762.1 MiB(+1.2 MiB)이고 기울기는 전체 0.045, 후반 0.021 MiB/min으로 줄어듭니다. 포화하는 형태이나 0은 아니므로, 수 시간 단위 누수 여부는 이 측정으로 확정되지 않습니다.

## 8. 추가 기록 (같은 날 후속)

- **원래 gate(v1) 128-lane 재시도** (`gatev1-user-stopped/`): IO가 조용한 시점에 같은 계획으로 시작해 28 lane까지 원래 gate를 통과하며 진행했으나, 사용자 지시로 중단했습니다(Qwen 측정 우선). 미완료 matrix이므로 결론 없음, gate v2 결과와 병합하지 않음.
- **Qwen HF-exact 경로를 일반 scheduler 위로 이전**: 입장·대기·iteration 계획·KV 블록 예약/commit·취소 정산·종료 회수를 `riley_scheduler`가 수행하고 exact owner는 계획된 iteration만 실행합니다. HTTP gate 재통과(`qwen-http-exact/qwen3b-http-hf-exact-c1-result.json`은 이 버전, attempt 3). 범위 제한(P2048·greedy·≤128·동시 1건)과 기존 batch-executor 경로의 HF 불일치는 그대로입니다.
- **Qwen HF-exact 안정성** (`qwen-http-exact/stability-result.json`, 실제 `riley serve`): 310초 지속 부하 46건 전부 HF token 일치·오류 0, 취소 20회(직후/1/5/40 token 뒤)마다 drain 후 exact 재요청, 활성 중 대기 요청 취소, 종료 시 KV·할당 0. GPU 8,079 → 8,078 MiB, RSS 6,409.0 → 6,409.3 MiB.
- **Qwen 간단 성능 비교** (`qwen-simple-perf/`, P2048/O128 greedy, 동시 1건, 10건 × 2회, 서술용): vLLM 122.5 tok/s·TTFT 88 ms·TPOT 7.5 ms / Riley 기존 scheduler 경로 36.5 tok/s·TTFT 1,320 ms·TPOT 17.2 ms(HF token 불일치) / Riley HF-exact 19.0 tok/s·TTFT 123 ms·TPOT 52 ms(HF 일치). Riley의 Qwen 경로는 성능 최적화가 되어 있지 않으며 vLLM 대비 처리량 약 1/3.4, 1/6.4입니다.

## 9. Qwen 성능 1차 개선 (HF-exact 경로, 산술 불변)

프로파일(`ai-assistant:/data/riley-serving-261007/qwen-profile-diagnostic-*`): HF-exact decode step 53 ms 중 GPU kernel은 16 ms뿐이고, attention QK·AV가 호출마다 `cublasCreate`/`cublasDestroy`를 해서(step당 72회) 내부 device 할당·해제에 약 33 ms를 쓰고 있었습니다.

| 변경 | TPOT (HTTP) | logit 128행 byte-exact | 채택 |
|---|---:|:-:|:-:|
| 변경 전 | 52.0 ms | 예 | — |
| stream당 cuBLAS handle 1개 재사용 | 18.5 ms | 예 | 채택 |
| reference AV kernel의 load를 4 token 앞서 읽기(누산 순서 동일) | 16.2 ms | 예 | 채택 |
| 같은 load를 16 token 앞서 읽기 | 17.3 ms | 예 | 미채택(되돌림) |

재측정(`qwen-simple-perf/attempt02-after-exact-route-fixes/`, P2048/O128 greedy, 동시 1건, 10건 × 2회): vLLM 122.6 tok/s·TTFT 88 ms·TPOT 7.5 ms / Riley HF-exact 58.5 tok/s·TTFT 123 ms·TPOT 16.2 ms(HF 일치) / Riley 기존 scheduler 경로 36.4 tok/s·TTFT 1,321 ms·TPOT 17.2 ms(HF 불일치).

남은 step 구성(프로파일, 약 13 ms kernel + 3–4 ms host): cuBLAS GEMM 7.5 ms(HF와 동일 호출이라 그대로 둬야 함), AV 3.2 ms, K/V 반복 복사 1.8 ms, 연산마다 stream 동기화 약 709회. 다음 후보: step당 동기화 축소, K/V 반복 버퍼의 증분 갱신, AV의 순서 보존 가속.

## 10. Qwen 8개 조건 기준선과 병목 (서술용, host gate 없음)

`qwen-matrix/`: SmolLM2 benchmark의 fixture prompt를 Qwen2.5-3B에 그대로 사용(fixed: prompt 약 126 token·출력 평균 20, natural: prompt 약 180·출력 평균 75), 조건당 warmup 12 + 48 요청, 순서를 뒤집어 2회. 이 prompt들에 대한 HF reference는 없으므로 정확성은 "반복 간 출력 동일"만 확인했습니다(Riley는 전 조건 동일, vLLM은 C8 이상에서 반복 간 출력이 달라짐).

| 조건 | 처리량 tok/s Riley 기존 / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| c1-fixed | 69 / 130 | 0.53 | 49 / 13 | 12.5 / 7.4 |
| c1-natural | 74 / 132 | 0.56 | 49 / 13 | 12.5 / 7.4 |
| c8-fixed | 271 / 697 | 0.39 | 261 / 32 | 14.6 / 8.9 |
| c8-natural | 374 / 763 | 0.49 | 142 / 38 | 16.8 / 8.9 |
| c16-fixed | 271 / 1060 | 0.26 | 750 / 42 | 14.7 / 9.9 |
| c16-natural | 376 / 1266 | 0.30 | 1524 / 60 | 16.6 / 9.4 |
| c32-fixed | 266 / 1440 | 0.18 | 1702 / 128 | 14.8 / 11.3 |
| c32-natural | 376 / 1792 | 0.21 | 3573 / 133 | 16.6 / 9.9 |

(C32는 attempt 1에서 vLLM이 checkpoint 로딩 중 600초 안에 준비되지 않아 중단됐고, attempt 2에서 C32만 다시 측정했습니다.)

찾은 병목 두 가지와 진단 결과(`qwen-matrix-wide-256*.log`, 진단 빌드 — 채택·검증 전):

1. **HTTP worker 8개 고정**: streaming 응답 하나가 worker 하나를 끝까지 점유하는데 pool이 8로 고정돼, `--max-active-sequences`와 무관하게 동시에 8건만 처리됐습니다. `max_active_sequences` 이상으로 잡도록 수정(브랜치 커밋, 정식 검증 전).
2. **반복당 32 token 제한**: D128 attention backend가 batch token budget을 32로 제한해 prefill과 decode가 32 token을 나눠 씁니다. CLI·runtime 상수를 512로 올린 진단 빌드는 정상 동작했습니다(이 상수는 검증된 범위 표시이므로 정식 채택에는 kernel 검증이 필요).

두 가지를 함께 적용(budget 256, power-of-two shape)한 진단 수치: c16 502–525 tok/s(TTFT 55–85 ms), c32 612–669 tok/s(TTFT 150–190 ms). 기준선 대비 1.4–2.5배지만 vLLM의 0.37–0.47배입니다. decode step 시간이 행 수에 비례해 늘어나는 것(1행 15 ms → 8행 20 → 16행 25 → 32행 31–35 ms)이 남은 주원인이고, C1에서는 범용 GEMM kernel이 step의 61%입니다.

부수 관찰: prefill chunk 크기에 따라 기존 경로의 출력 token이 달라집니다(P2048 요청에서 chunk 256일 때만 128 token이 HF와 일치). 즉 이 경로의 산술은 chunk 불변이 아닙니다.

## 11. Qwen 다중 동시성: CPU sampling 병목 (진단 + 브랜치 변경)

C32 프로파일(진단 빌드): decode step 27 ms 중 GPU kernel 17 ms, 나머지 약 10 ms는 GPU 밖. 원인은 출력 행마다 CPU에서 어휘 크기(151,936) f64 배열 여러 개를 채우고 훑는 sampling이었습니다. Qwen은 addressable vocabulary가 logit 행보다 작아 GPU greedy 대상에서 제외되므로 항상 이 경로를 탑니다(`--sampling-backend gpu-greedy`를 줘도 수치 변화 없음을 확인).

변경(브랜치 `codex/qwen-http-hf-exact`): temperature 0·repetition penalty 없음인 행은 한 번의 검증 pass로 token을 고릅니다. 전체 pipeline과 같은 token·확률·오류를 내는지 단위 테스트(동점, 음의 0, mask 포함 800 비교)로 확인했고, HF-exact HTTP gate도 다시 통과했습니다(attempt 9, logit 128행 byte-exact, TPOT 16.2 → 15.8 ms).

진단 빌드(worker 수 수정 + 32 token 제한을 512로 올림 + budget 256 + greedy 빠른 경로) 수치 — 32 token 제한 변경은 아직 채택·검증 전이므로 서술용입니다:

| 조건 | 기준선 tok/s | 진단 tok/s | vLLM tok/s | 진단/vLLM | TTFT p50 ms 진단 / vLLM | TPOT p50 ms 진단 / vLLM |
|---|---:|---:|---:|---:|---:|---:|
| c1-fixed | 69 | 68 | 130 | 0.52 | 18 / 13 | 14.6 / 7.4 |
| c1-natural | 74 | 67 | 132 | 0.50 | 18 / 13 | 14.7 / 7.4 |
| c8-fixed | 271 | 387 | 697 | 0.56 | 39 / 32 | 17.2 / 8.9 |
| c8-natural | 374 | 401 | 763 | 0.53 | 60 / 38 | 17.2 / 8.9 |
| c16-fixed | 271 | 584 | 1060 | 0.55 | 48 / 42 | 20.4 / 9.9 |
| c16-natural | 376 | 623 | 1266 | 0.49 | 75 / 60 | 20.3 / 9.4 |
| c32-fixed | 266 | 741 | 1440 | 0.51 | 149 / 128 | 24.4 / 11.3 |
| c32-natural | 376 | 829 | 1792 | 0.46 | 169 / 133 | 25.4 / 9.9 |

남은 것: C1 TPOT이 진단 설정(power-of-two shape)에서 14.6 ms로 기준선 12.5 ms(fixed-max)보다 느림 — 원인 미확인. 행당 약 0.33 ms씩 늘어나는 step 시간(logit 행 전체 D2H + CPU 스캔, GEMM·attention의 행 비례분). GEMM kernel 자체(C1 step의 61%).

## 12. Qwen GEMM 진단 (`qwen-gemm-diagnostics/`, 진단용)

- cache 영향을 뺀 microbench(가중치 사본 14개를 돌려가며 호출): 45 MB 행렬(2048↔11008) GEMM은 좋은 알고리즘에서 행 수 1–32와 무관하게 약 50 µs로, 메모리 대역폭 한계입니다. 즉 layer당 약 175 µs, step당 약 6.3 ms가 누구에게나 바닥이고, vLLM의 C1 TPOT 7.4 ms는 이 바닥에 가깝습니다. (같은 가중치를 반복 호출하면 GPU cache 덕에 17–22 µs로 보이므로 그 수치는 쓰면 안 됩니다.)
- C32 프로파일에서 decode만 하는 step(32행, 주기 24.4 ms)을 골라 보면 kernel 16.4 ms, GPU 밖 8.0 ms(이 프로파일은 greedy 빠른 경로 적용 전)입니다. kernel 중 down projection GEMM이 156 µs × 37, gate/up이 62 µs × 72로, 바닥(약 50 µs)의 3배·1.2배입니다.
- 원인: Riley의 표준 GEMM plan은 cuBLASLt heuristic의 첫 후보를 받아 split-K를 1, reduction을 none으로 강제합니다. 다행 down projection 형상에서는 후보가 모두 split-K 알고리즘이라, 강제 후에는 느린 구성이 됩니다(microbench에서도 52.7 → 65.6 µs, 다른 후보는 최대 135 µs).
- 다음 실험: 강제 후에도 한 번만 읽는 구성을 고르는 결정적 선택 규칙(또는 down projection에 gemmEx 형태 사용). 기대 효과는 C32 step에서 약 4 ms(18%)입니다. 출력 수치가 달라지므로 기존 경로의 반복 간 동일성 검증이 필요합니다.

## 13. Qwen GEMM 정책 진단 결과

`crates/riley-runtime/src/llama/gemm_policy.rs`: SmolLM2(검토된 Llama geometry)는 hidden/key-value/down projection에 cuBLASLt의 첫 heuristic split-K 알고리즘을 그대로 쓰는 정책이지만, Qwen2는 전부 no-split 강제(STRICT)입니다. 진단 빌드에서 Qwen2의 모든 projection에 split-K 허용 정책을 적용해 다시 측정했습니다(`qwen-matrix/qwen-matrix-wide-256-splitk.log`; 진단 빌드 = worker 수 수정 + 32→512 token 제한 + budget 256 + greedy 빠른 경로 + 이 정책. 정책 변경과 token 제한 변경은 채택·검증 전).

| 조건 | 진단 tok/s | vLLM tok/s | 비율 | TTFT p50 ms 진단 / vLLM | TPOT p50 ms 진단 / vLLM |
|---|---:|---:|---:|---:|---:|
| c1-fixed | 86 | 130 | 0.66 | 14 / 13 | 11.4 / 7.4 |
| c1-natural | 85 | 132 | 0.64 | 15 / 13 | 11.5 / 7.4 |
| c8-fixed | 473 | 697 | 0.68 | 33 / 32 | 13.9 / 8.9 |
| c8-natural | 491 | 763 | 0.64 | 46 / 38 | 14.1 / 8.9 |
| c16-fixed | 693 | 1060 | 0.65 | 42 / 42 | 17.5 / 9.9 |
| c16-natural | 741 | 1266 | 0.59 | 69 / 60 | 17.2 / 9.4 |
| c32-fixed | 878 | 1440 | 0.61 | 141 / 128 | 21.1 / 11.3 |
| c32-natural | 969 | 1792 | 0.54 | 168 / 133 | 22.0 / 9.9 |

오류 0. 기준선(0.18–0.56배) 대비 전 조건이 0.54–0.68배가 됐고 TTFT는 vLLM과 비슷한 수준입니다. 남은 TPOT 격차(C1 4 ms, C32 12 ms)의 후보: attention kernel 3개(layer당 약 90 µs), logit 행 전체 D2H와 CPU 스캔(행당), 연산별 완료 대기, GEMM의 남은 비효율.

정책 변경은 Qwen 기존 경로의 출력 수치를 바꿉니다(이 경로는 원래 HF-exact가 아님). 채택하려면 결정성(반복 간 동일)과 SmolLM2 경로 무영향을 정식으로 확인해야 합니다.

## 14. Qwen 채택 빌드 측정 (브랜치 `codex/qwen-http-hf-exact` c0fa7ee5)

브랜치에 채택한 것: HTTP worker pool을 active sequence 수에 맞춤, greedy 한 번 훑기 선택, D128 batch 상한 32 → 256(CLI 계약 테스트 갱신), 선택 플래그 `--gemm-reduction-policy first-heuristic-split-k-v1`(기본값은 기존 정책 그대로). 256 상한의 근거는 서빙 수준 증거입니다: budget 256 진단 측정 두 번과 기준선에서, 모든 prompt의 출력(token+text 해시)이 C1/C8/C16/C32에서 동일했습니다. native per-row 테스트는 여전히 8행까지만 다룹니다.

HF-exact HTTP gate는 이 빌드에서도 통과(attempt 10, logit 128행 byte-exact, 결과 해시 동일).

측정(`qwen-matrix/table.md`, plan은 `qwen-matrix/plan.json`; Riley: budget 256·power-of-two shape·first-heuristic GEMM 정책; 조건당 warmup 12 + 48 요청, 순서 뒤집어 2회, host gate 없음):

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms | E2E p50 ms |
|---|---:|---:|---:|---:|---:|
| c1-fixed | 86 / 129 | 0.67 | 14 / 13 | 11.4 / 7.4 | 233 / 155 |
| c1-natural | 85 / 132 | 0.64 | 14 / 14 | 11.5 / 7.4 | 738 / 482 |
| c8-fixed | 470 / 690 | 0.68 | 33 / 34 | 14.1 / 9.1 | 304 / 207 |
| c8-natural | 493 / 764 | 0.65 | 48 / 38 | 14.0 / 8.9 | 927 / 595 |
| c16-fixed | 692 / 1066 | 0.65 | 41 / 41 | 17.8 / 9.9 | 375 / 240 |
| c16-natural | 741 / 1265 | 0.59 | 69 / 59 | 17.1 / 9.4 | 1131 / 655 |
| c32-fixed | 890 / 1444 | 0.62 | 137 / 108 | 20.7 / 11.1 | 532 / 338 |
| c32-natural | 969 / 1802 | 0.54 | 155 / 125 | 22.3 / 9.9 | 1662 / 849 |

오류 0/0. Riley는 전 조건에서 반복 간 출력 동일, vLLM은 C8 이상에서 반복 간 출력이 달라짐. GPU peak는 Riley 11.1–12.2 GiB, vLLM 15.2–15.4 GiB(vLLM은 사전 할당 포함). 최소 기준(vLLM 이상)은 Qwen 8개 조건 모두 미달입니다.

## 15. Qwen 채택 빌드 안정성과 greedy 스캔 2차 개선

**안정성** (`qwen-scheduler-stability/`, 브랜치 c0fa7ee5 빌드, 실제 `riley serve`, budget 256·first-heuristic 정책): 이 prompt들에는 HF reference가 없어 정확성은 "같은 prompt의 모든 완료가 처음 완료와 token·text 동일"로 확인했습니다.

| 조건 | 지속 부하 | 요청 | 오류 / 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(idle/부하 중) | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| c32-natural | 302 s | 4,512 | 0 / 0 | 1,106 / 1,122 | 12,232 / 12,232 | 6,415.2 / 6,416.0 | 20 / 40 | 통과 |
| c16-fixed | 300 s | 11,328 | 0 / 0 | 751 / 755 | 11,220 / 11,220 | 6,405.8 / 6,406.5 | 20 / 40 | 통과 |
| c1-natural | 170 s | 192 | 0 / 0 | 84.6 / 84.5 | 11,106 / 11,106 | 6,398.8 / 6,399.1 | 20 / 40 | 통과 |

매 취소·재요청 뒤 active/waiting/KV 블록 0, 종료 시 exit 0·native 할당 0.

**greedy 스캔 2차 개선** (커밋 "Scan greedy rows with branch-free integer passes"): 채택 빌드의 C32 프로파일에서 decode step 21.2 ms 중 kernel 14.1 ms, D2H 0.55 ms(9.6 MB), 그 밖 6.5 ms였습니다. BF16 logit을 순서 보존 정수 key로 비교하는 분기 없는 pass로 바꿨고, 동일성 테스트에 비유한값·전부 mask된 행을 추가했습니다. HF-exact gate attempt 11 통과(결과 해시 동일). Riley만 다시 측정(`qwen-matrix/rileyonly-attempt01-result.json`; vLLM 수치는 14절 측정값):

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| c1-fixed | 87 / 129 | 0.67 | 14 / 13 | 11.3 / 7.4 |
| c1-natural | 85 / 132 | 0.65 | 14 / 14 | 11.4 / 7.4 |
| c8-fixed | 491 / 690 | 0.71 | 32 / 34 | 13.5 / 9.1 |
| c8-natural | 515 / 764 | 0.67 | 47 / 38 | 13.4 / 8.9 |
| c16-fixed | 741 / 1066 | 0.70 | 39 / 41 | 16.2 / 9.9 |
| c16-natural | 788 / 1265 | 0.62 | 63 / 59 | 15.9 / 9.4 |
| c32-fixed | 935 / 1444 | 0.65 | 143 / 108 | 19.0 / 11.1 |
| c32-natural | 1054 / 1802 | 0.58 | 144 / 125 | 19.6 / 9.9 |

오류 0, 반복 간 출력 동일, 그리고 모든 조건의 출력이 14절 측정(스캔 변경 전 빌드)과 동일합니다. vLLM은 같은 실행에서 다시 재지 않았으므로 비율은 참고용입니다.

남은 C32 decode step(프로파일): gate/up GEMM 63 µs × 72, down GEMM 90 µs × 37(바닥 약 50 µs), attention partial-state kernel 90 µs × 36, host 약 3 ms.

## 16. Qwen C1의 한계와 GEMM anchor 진단

**C1 프로파일(채택 빌드, 1행 decode step 11.9 ms)**: kernel 11.0 ms(GEMM 8.85, attention 1.26, norm 0.3 등), host 0.9 ms.

**Riley의 GEMM 선택 방식** (`qwen-gemm-diagnostics/riley-selected-algorithms.txt`): projection 형상마다 plan을 하나만(최대 행 수 M=256 기준) 만들고, 그 알고리즘을 모든 행 수 bucket에 그대로 씁니다. 같은 요청의 출력이 동시성에 따라 달라지지 않게 하려는 설계이고(실제로 Riley는 전 조건에서 출력이 동일, vLLM은 C8 이상에서 달라짐), 대신 1–32행에서는 256행용 알고리즘이 최적이 아닙니다: gate/up 57 µs, down 72 µs, q/o 16 µs.

**anchor microbench** (`gemm_bench4.log`, cold weights): 모든 행 수에서 쓸 수 있으면서 더 빠른 구성이 있습니다(예: down은 54.6 µs @≤32행 / 81 µs @256행, 현재 72 / 86). 형상별로 고정하면 decode step당 약 1.1 ms(C1의 약 9%) 절감이 예상됩니다. 다만 형상별 후보 번호가 달라 "고정 알고리즘 표"가 필요합니다(미구현).

**C1에서 vLLM을 넘을 수 있는가**: Qwen2.5-3B는 token 하나에 projection 가중치 약 6.2 GB를 읽습니다. cold 측정에서 45 MB 행렬 하나가 약 50 µs(≈900 GB/s)이므로 GEMM만의 바닥이 step당 약 6.85 ms입니다. vLLM의 C1 TPOT 7.4 ms는 이 바닥의 약 93% 지점입니다. 따라서 이 GPU에서 BF16 가중치로는 어떤 엔진도 C1에서 vLLM을 의미 있게 앞설 수 없고, Riley가 낼 수 있는 최선은 동률 부근입니다(현재 11.3 ms; GEMM anchor·attention·host를 모두 줄이면 8–9 ms 추정). "처리량 vLLM 이상" 기준은 Qwen C1 두 조건에서 하드웨어 한계에 묶여 있습니다. 동시성 8 이상에서는 vLLM도 9–11 ms로 같은 바닥 위에 있어, 격차는 Riley 쪽 비효율(행 수 비례 비용, anchor 알고리즘, attention)입니다.

## 17. SmolLM2 원래 gate(v1) 전체 matrix 완료

`serving-gatev1/` (`kernel-batch13-gatev1-quiet-attempt02`): gate v2 측정과 같은 고정 계획·controller·binary·순서·반복 수에, host gate만 원래 조건(cpu some ≤ 5, io full ≤ 5, memory full ≤ 0.5, 3회 연속, 300초 내 미충족 시 전체 중단)으로 실행했습니다. 128 lane 전부 gate를 통과해 완료, 오류 0, Riley exact reference 전부 일치, 제외 없음. attempt 1(사용자 지시로 28 lane에서 중단)은 보존되어 있고 병합하지 않았습니다.

| 조건 | 처리량 후보/vLLM | TTFT p50 | TPOT p50 | 최소 | 도전 |
|---|---:|---:|---:|:-:|:-:|
| c1-fixed | 1.337 | 0.382 | 0.991 | 통과 | 미달 |
| c1-natural | 1.178 | 0.387 | 0.859 | 통과 | 통과 |
| c8-fixed | 1.639 | 0.364 | 0.712 | 통과 | 통과 |
| c8-natural | 1.310 | 0.634 | 0.791 | 통과 | 통과 |
| c16-fixed | 1.633 | 0.299 | 0.761 | 통과 | 통과 |
| c16-natural | 1.249 | 0.522 | 0.852 | 통과 | 통과 |
| c32-fixed | 1.482 | 0.212 | 0.936 | 통과 | 미달 |
| c32-natural | 1.202 | 0.326 | 0.930 | 통과 | 미달 |

P95/P99(TTFT·TPOT·E2E)도 전 조건 vLLM 이하. gate v2 결과와 같은 결론입니다(이번에는 c1-natural도 도전 목표 통과, 미달은 c1-fixed·c32-fixed·c32-natural의 TPOT). "후보 최악 반복 vs vLLM 최고 반복"으로는 c32-fixed와 c32-natural의 TPOT이 뒤집힙니다(평균은 통과; vLLM 반복 편차 ±0.14–0.20 ms).

## 18. GEMM anchor 고정 진단 (채택 전)

`qwen-gemm-diagnostics/pinned-anchor-diagnostic.patch`(진단 전용, 브랜치 미반영): Qwen projection 세 형상에 대해, 최대 행 수의 첫 heuristic 대신 `gemm_bench4.log`에서 고른 "모든 행 수에서 쓸 수 있는" 알고리즘을 고정했습니다(gate/up: 256행 heuristic의 후보 2, down·q/o: 32행 heuristic의 후보 2; 세 형상 모두 plan 형상에 대한 AlgoCheck 통과). Riley만 측정(`qwen-matrix/qwen-matrix-pinned.log`, vLLM 수치는 14절):

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| c1-fixed | 95 / 129 | 0.74 | 14 / 13 | 10.3 / 7.4 |
| c1-natural | 94 / 132 | 0.71 | 14 / 14 | 10.3 / 7.4 |
| c8-fixed | 528 / 690 | 0.77 | 32 / 34 | 12.4 / 9.1 |
| c8-natural | 557 / 764 | 0.73 | 48 / 38 | 12.4 / 8.9 |
| c16-fixed | 769 / 1066 | 0.72 | 39 / 41 | 15.5 / 9.9 |
| c16-natural | 845 / 1265 | 0.67 | 58 / 59 | 14.9 / 9.4 |
| c32-fixed | 989 / 1444 | 0.69 | 127 / 108 | 18.9 / 11.1 |
| c32-natural | 1115 / 1802 | 0.62 | 163 / 125 | 18.9 / 9.9 |

오류 0, 반복 간 출력 동일, 15개 prompt 모두 동시성에 따라 출력이 달라지지 않음. step당 약 1 ms 이득으로 microbench 예측과 맞습니다. 채택하려면 형상별 고정을 설정(코드 내 하드코딩이 아닌)으로 전달하는 배관과, 다른 GPU·cuBLAS 버전에서의 동작 정의가 필요합니다.

attention partial-state kernel 진단: page(16 token) × head × 행마다 warp 하나가 16 token을 직렬로 처리(1행일 때도 약 22 µs = 지연 시간 한계). 32행에서는 warp 수가 GPU 동시 수용량을 넘어 약 107 µs가 됩니다. 줄이려면 kernel 내부 누산 구조를 바꿔야 하고, 그 경우 기존 per-row native 동일성 테스트를 다시 통과시켜야 합니다(미착수).

## 19. 128행 anchor 규칙 채택 (브랜치)

anchor 행 수 스캔(`qwen-gemm-diagnostics/gemm_bench5.log`): 128행에서 첫 heuristic을 고르면 세 projection 모두 1–32행에서 바닥 근처(54/56/15 µs)이고 256행에서도 기존보다 느리지 않습니다(83/76/22 µs; 기존 256행 anchor는 1–32행 57/72/16, 256행 83/86/17.5). 형상별 표 없이 규칙 하나로 충분해서, 선택형 정책(`--gemm-reduction-policy first-heuristic-split-k-v1`)에 "128행에서 선택 후 실제 형상에 검증 결합(실패 시 중단)"을 넣었습니다. 기본 정책과 SmolLM2 경로는 변경 없음.

HF-exact HTTP gate attempt 12 통과(결과 해시 동일). Riley만 측정(`qwen-matrix/rileyonly-anchor128-attempt01-result.json`; vLLM 수치는 14절):

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| c1-fixed | 95 / 129 | 0.73 | 14 / 13 | 10.4 / 7.4 |
| c1-natural | 93 / 132 | 0.70 | 14 / 14 | 10.4 / 7.4 |
| c8-fixed | 522 / 690 | 0.76 | 31 / 34 | 12.5 / 9.1 |
| c8-natural | 554 / 764 | 0.73 | 44 / 38 | 12.4 / 8.9 |
| c16-fixed | 769 / 1066 | 0.72 | 38 / 41 | 15.7 / 9.9 |
| c16-natural | 847 / 1265 | 0.67 | 63 / 59 | 14.9 / 9.4 |
| c32-fixed | 995 / 1444 | 0.69 | 142 / 108 | 18.4 / 11.1 |
| c32-natural | 1109 / 1802 | 0.62 | 148 / 125 | 18.8 / 9.9 |

오류 0, 반복 간 출력 동일, 15개 prompt 모두 동시성에 따라 출력 불변. 18절의 형상별 고정 진단과 같은 수준입니다(그 진단 patch는 이 규칙으로 대체, 미채택으로 보존).

## 20. 최신 브랜치 빌드(5a12df19) 안정성과 C32 step 구성

안정성 재검증(`qwen-scheduler-stability/*-attempt02-result.json`): c32-natural 300초 5,184건 오류 0·출력 불일치 0, 처리량 1,285 → 1,275 tok/s, RSS 6,417.7 → 6,418.6 MiB, 취소 20 + 부하 중 40, 종료 회수 통과. c8-fixed 200초도 같은 항목으로 실행(결과 파일 참조).

C32 decode step(프로파일, `qwen-matrix/c32-step-breakdown-anchor128.txt`, 주기 18.2 ms): kernel 13.1 ms(gate/up·q/o GEMM 5.2, down 2.0, LM head 0.7, attention partial-state 3.2, 기타 2.0), D2H 0.55 ms(logit 9.6 MB), 그 밖 4.5 ms. 그 밖의 구성: kernel launch 호출 약 800회 ≈ 1.2 ms, CPU greedy 스캔 27 µs × 32행 ≈ 0.9 ms(호스트에서 실측), 나머지 약 2.4 ms는 scheduler·metadata·event 처리.

vLLM의 같은 조건 TPOT은 9.9–11.1 ms입니다. 남은 8 ms 격차는 한 가지 원인이 아니라 attention kernel(3 ms), eager 실행의 launch·host 비용(약 4.5 ms), logit 내려받기·스캔(1.4 ms), GEMM의 바닥 대비 초과분으로 나뉘어 있어, SmolLM2 계열에서 했던 것(CUDA graph 실행, kernel 융합, GPU greedy, host 경로 경량화)을 Qwen 경로에 옮기는 여러 batch가 필요합니다.

## 21. Qwen GPU greedy (addressable 상한) 채택

Qwen의 logit 행(151,936)은 tokenizer가 주소 지정하는 범위(151,665)보다 넓어서, 지금까지 모든 Qwen 요청이 GPU greedy 대상에서 제외되고 행마다 logit 전체를 host로 내려받아(32행 step당 9.6 MB) CPU에서 훑었습니다. argmax primitive에 후보 상한을 추가했습니다(ABI의 reserved word 사용; 모든 열의 유한성 검사는 유지, 상한 미만 ID만 선택 가능). 상한이 없는 기존 호출은 그대로입니다. 엔진은 `--sampling-backend gpu-greedy`이고 tokenizer 범위가 행보다 좁을 때만 상한을 설정합니다.

HF-exact HTTP gate attempt 14 통과(결과 해시 동일). Riley만 측정(`qwen-matrix/rileyonly-gpugreedy-attempt01-result.json`; `--sampling-backend gpu-greedy` 추가, vLLM 수치는 14절):

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| c1-fixed | 94 / 129 | 0.73 | 14 / 13 | 10.4 / 7.4 |
| c1-natural | 93 / 132 | 0.70 | 14 / 14 | 10.4 / 7.4 |
| c8-fixed | 543 / 690 | 0.79 | 30 / 34 | 12.0 / 9.1 |
| c8-natural | 575 / 764 | 0.75 | 45 / 38 | 11.9 / 8.9 |
| c16-fixed | 832 / 1066 | 0.78 | 36 / 41 | 14.2 / 9.9 |
| c16-natural | 902 / 1265 | 0.71 | 59 / 59 | 13.7 / 9.4 |
| c32-fixed | 1094 / 1444 | 0.76 | 120 / 108 | 16.1 / 11.1 |
| c32-natural | 1243 / 1802 | 0.69 | 131 / 125 | 16.0 / 9.9 |

오류 0. 모든 조건·모든 prompt의 출력이 CPU greedy를 쓰던 직전 빌드(19절)와 동일합니다. 첫 빌드 시도(attempt 13)는 cuda 전용 생성자에 새 필드를 빠뜨려 컴파일 실패했고 바로 수정했습니다(실패 로그 보존). native GPU 테스트(`bf16_argmax_gpu.rs`)에 상한 경로 케이스는 아직 추가하지 않았습니다.

## 22. GPU greedy 빌드 검증 보강과 D128 attention shared-KV producer 채택

**21절 빌드의 남은 검증** (`qwen-scheduler-stability/gpu-greedy-build/`, binary sha `8a72f2f5…`):
- 상한 argmax native GPU 테스트 추가(`bf16_argmax_gpu.rs`의 `bounded_argmax_matches_cpu_oracle_and_ignores_columns_past_the_bound`, commit 972394aa): 상한 밖 최대값 무시, 상한 마지막 후보, 동률·부호 있는 0, 상한 안/밖 non-finite, 상한 0·vocab 초과 거부, 상한 = vocab일 때 기존 경로와 동일. RTX 4090에서 4/4 통과.
- `--sampling-backend gpu-greedy`로 안정성 재실행: c32-natural 300초 5,856건, c8-fixed 302초 8,640건, c1-natural 310초 384건 — 모두 오류 0·출력 불일치 0, GPU 메모리 불변, RSS 6,399–6,407 MiB 평탄, 취소 20 + 부하 중 40, KV 0 복귀, 종료 회수 통과.

**shared-KV producer** (commit 2b9e1601 + 테스트 buffer 해제 수정): D128 partial-state producer는 (page, query head, 행)마다 warp 하나가 같은 K/V page를 global memory에서 읽었습니다. Qwen은 query head 16 / KV head 2라 같은 page를 8번 읽습니다. 새 launch 배치는 (page, KV head, 행)마다 CTA 하나(warp 8개)가 K/V tile을 shared memory에 BF16 그대로 한 번 올리고, 각 warp는 기존과 같은 연산을 같은 순서로 수행합니다. 기존 per-head 배치는 그대로 남아 있습니다.

- native GPU 테스트(`native_bf16_ragged_paged_d128_two_stage_v3_c2048_b8_matches_v1_prefix_reducer_per_row`)에서 두 배치의 partial state·transition·normalizer·출력 전체 byte 동일 확인. 첫 실행은 테스트가 비교용 buffer 4개를 닫지 않아 allocation 검사에서 실패(로그 `…attempt01-failed-unreleased-buffers.log` 보존), 수정 후 통과.
- 진단 A/B(같은 binary, 환경 변수로 배치만 전환, `qwen-matrix/shared-kv/qwen-matrix-sharedkv-diagnostic-*`): 16개 lane 모두 개선, 출력 동일.
- HF-exact HTTP gate attempt 15 통과(결과 해시 `5b025fdb…` 동일, 이 경로는 다른 kernel이라 TPOT 15.75 ms 그대로).
- 채택 빌드(sha `8f916a9e…`) Riley만 측정(`qwen-matrix/shared-kv/qwen-matrix-rileyonly-sharedkv-attempt01/result.json`, vLLM 수치는 14절). 오류 0, 모든 prompt 출력이 직전 빌드와 동일:

| 조건 | 처리량 tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms | Riley TPOT p95 | 오류 |
|---|---:|---:|---:|---:|---:|---:|
| c1-fixed | 99 / 129 | 0.77 | 13 / 13 | 9.9 / 7.4 | 9.9 | 0 |
| c1-natural | 98 / 132 | 0.74 | 13 / 14 | 9.9 / 7.4 | 10.2 | 0 |
| c8-fixed | 576 / 690 | 0.84 | 27 / 34 | 11.4 / 9.1 | 12.1 | 0 |
| c8-natural | 604 / 764 | 0.79 | 39 / 38 | 11.3 / 8.9 | 11.5 | 0 |
| c16-fixed | 874 / 1066 | 0.82 | 33 / 41 | 13.5 / 9.9 | 14.5 | 0 |
| c16-natural | 988 / 1265 | 0.78 | 50 / 59 | 12.4 / 9.4 | 13.3 | 0 |
| c32-fixed | 1189 / 1444 | 0.82 | 115 / 108 | 14.3 / 11.1 | 16.8 | 0 |
| c32-natural | 1391 / 1802 | 0.77 | 125 / 125 | 13.7 / 9.9 | 17.2 | 0 |

- 채택 빌드 안정성(`qwen-scheduler-stability/shared-kv-build/`): c32-natural 304초 6,816건(1,687 → 1,669 tok/s), c16-fixed 301초 14,400건, c1-natural 366초 480건 — 오류 0·불일치 0, RSS 6,400–6,408 MiB 평탄, 취소 60회씩, 종료 회수 통과.

한계: Qwen은 여전히 8개 조건 모두 vLLM 미달(0.74–0.84×). vLLM 수치는 14절의 것을 재사용했고 같은 실행에서 다시 재지 않았습니다. scheduler 경로의 정확성 근거는 HF 일치가 아니라 반복·동시성 불변과 빌드 간 출력 동일입니다.

## 23. shared-KV 빌드 프로파일, GEMM 바닥 재측정, 실패한 후보

증거: `qwen-profile-sharedkv/` (nsys kernel 분해, host API·gap 분석, `gemm_bench6` 소스·로그, graph 정책 진단 로그).

**decode step 구성 (프로파일 중 측정)**

| | C1 (주기 10.3 ms) | C32 (주기 12.8 ms) |
|---|---:|---:|
| GEMM 전체(cuBLASLt + splitKreduce) | 7.95 | 8.4 |
| attention 3개 kernel | 0.79 | 1.82 |
| 나머지 kernel(RMSNorm, bias, RoPE, residual, KV write, SiLU, argmax 등) | 0.86 | 약 1.0 |
| GPU 유휴(kernel 경계 0.6 µs × 980개 + step 전환) | 0.69 | 1.6 |

step당 kernel 980개, `cudaStreamSynchronize` 11회. host의 launch 호출 자체(약 2.4 ms)는 GPU 실행과 겹치므로 병목이 아닙니다. C32에서는 step 종료 후 다음 step 첫 launch까지 host 구간이 약 0.27 ms입니다.

**GEMM 바닥 재측정 (`gemm_bench6`, 가중치 복사본을 돌려 cold 상태)**: 45 MB projection을 단순히 읽기만 해도 49.0 µs(약 920 GB/s), LM head 622 MB는 651 µs. `cublasGemmEx`와 cuBLASLt(M에 맞춘 heuristic)는 M=1에서 49.8 / 51.3 / 11.7 / 653.6 µs로 이미 읽기 속도와 같습니다. 직접 작성한 warp-per-output GEMV kernel은 M=1에서 50.2–54 µs(동률 이하), M=8에서 88–110 µs(cuBLAS 56 µs보다 느림) — **미채택**.

여기서 나오는 결론: Qwen 한 step의 가중치 읽기 바닥은 36 × (49 × 3 + 11.2 × 2 + K/V) + 651 ≈ 6.9 ms입니다. vLLM의 C1 TPOT 7.4 ms는 이 바닥에서 0.5 ms 이내이고, Riley는 GEMM만 7.95 ms입니다(128행 anchor 알고리즘을 모든 행 수에 재사용하는 비용이 약 0.65 ms, 이 재사용이 동시성 불변 출력의 근거). attention·기타 kernel·유휴를 전부 0으로 만들어도 C1은 vLLM보다 느립니다. **C1에서 vLLM 이상은 현재 구조(동시성 불변 GEMM 선택)에서는 도달 불가**로 판단합니다.

**실패·미채택 후보**
- `--execution-graph-policy auto`: 시작 거부. `native-bf16-paged-split-gqa-d128-two-stage is eager-only and requires --execution-graph-policy disabled`. D128 backend를 graph capture 대상으로 만드는 작업이 선행되어야 합니다(미착수). 기대 이득 상한은 GPU 유휴 0.7–1.6 ms.
- 직접 작성 GEMV kernel: 위 표대로 이득 없음.
- 기존 플래그 3종을 shared-KV 빌드에서 16 lane씩 측정(`qwen-matrix/flag-variants/`):
  - `--residual-rmsnorm fused`: 시작 거부(`fused residual RMSNorm may only be used with per-operation completion`). 미실행.
  - `--projection-bias-backend cublaslt-bias-epilogue-experimental-v1`: 출력이 달라지고(모든 조건), C1은 +5%(TPOT 9.86 → 9.39)지만 C8–C32는 1–3% 느려짐(c32-natural 1,391 → 1,351 tok/s). **미채택**.
  - `--metadata-transport packed-async`: 출력 동일, 변화 ±1% 이내(잡음 수준). **미채택**.

## 24. Qwen scheduler 경로의 HF token 단위 비교 (일반 HTTP 경로, 벤치마크 prompt 15개)

증거: `qwen-hf-token-reference/`. 사전 고정 조건: reference = HF transformers 5.15.1 / torch 2.13 eager(attention eager, cache on), BF16, greedy, batch 1, 기본 EOS, `max_new_tokens` = fixture 값. Riley = 채택 플래그 그대로 HTTP, C1(순차 15건)과 C32(60건 동시). 제외 없음. 이것은 pinned 이미지 기준의 logit byte 일치 계약(HF-exact 경로, P2048)과는 별개의 더 약한 검사입니다.

- 15개 중 10개 prompt는 전체 token 열이 HF와 동일. 5개는 갈라짐: editing(28번째/32), sql(4/16), natural-16(2/32), natural-128(58/64), natural-398(69/128).
- C1과 C32의 결과가 완전히 같음(C32: 60건 중 40건 일치, 같은 5개 prompt가 같은 위치에서 갈라짐).
- 갈라진 5개 위치 모두 HF의 top-1과 top-2 logit 차이가 0–0.25(그 크기에서 BF16 한두 칸)이고, Riley가 고른 token은 HF 순위 0 또는 1입니다. 그중 3개(sql, natural-128, natural-398)는 같은 HF 모델을 cache 없이 전체 prefix로 다시 돌리면 Riley의 token이 1위입니다. 즉 HF 자신의 cache-on 생성과 cache-off forward가 그 위치에서 서로 다릅니다.

해석: scheduler 경로는 HF와 byte 단위로 같지 않지만, 관측된 차이는 전부 BF16 반올림 한두 칸 안의 근소차 위치에서만 발생했습니다. byte 일치가 필요한 계약은 여전히 HF-exact 경로(P2048/C1/greedy)만 만족합니다.

## 25. Qwen 정상상태 8조건 비교 (shared-KV 빌드 vs vLLM 0.27.1)

증거: `qwen-matrix/steady/` (`plan.json`은 측정 전 기록). 이전 Qwen 비교(14·21·22절)는 lane당 48건이라 C32에서 동시성이 1.5 wave만 유지됐습니다. 여기서는 warmup 32 / 측정 192건, 반복 2회, 순서 교대(R→V, V→R), 같은 harness, host gate 없음, 제외 없음. Riley binary sha `8f916a9e…`. 32 lane 완료, 오류 0.

값은 Riley / vLLM, 2회 평균.

| 조건 | 처리량 tok/s | 비율 | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 | E2E p50 | E2E p95 | E2E p99 | GPU 최대 MiB | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 99 / 129 | 0.77 | 13 / 13 | 17 / 19 | 17 / 21 | 9.9 / 7.4 | 9.9 / 7.4 | 9.9 / 7.4 | 203 / 157 | 319 / 243 | 320 / 248 | 11086 / 15256 | 0.0 / 0.5 |
| c1-natural | 98 / 132 | 0.74 | 13 / 13 | 39 / 25 | 39 / 26 | 9.9 / 7.4 | 10.2 / 7.5 | 10.3 / 7.5 | 636 / 482 | 1339 / 971 | 1341 / 971 | 11110 / 15222 | 0.0 / 0.0 |
| c8-fixed | 622 / 754 | 0.83 | 27 / 33 | 43 / 38 | 69 / 51 | 11.5 / 9.1 | 12.1 / 9.4 | 12.3 / 9.5 | 247 / 203 | 400 / 320 | 410 / 325 | 11148 / 15222 | 0.2 / 0.2 |
| c8-natural | 664 / 835 | 0.80 | 30 / 37 | 58 / 44 | 76 / 68 | 11.3 / 9.0 | 11.4 / 9.1 | 11.5 / 9.1 | 742 / 593 | 1503 / 1180 | 1506 / 1185 | 11364 / 15266 | 0.2 / 0.0 |
| c16-fixed | 1003 / 1285 | 0.78 | 32 / 37 | 75 / 93 | 139 / 94 | 14.2 / 10.3 | 14.8 / 10.9 | 15.2 / 11.4 | 301 / 237 | 484 / 367 | 512 / 409 | 11222 / 15358 | 1.8 / 1.3 |
| c16-natural | 1162 / 1496 | 0.78 | 41 / 41 | 96 / 92 | 203 / 125 | 12.5 / 9.6 | 12.8 / 9.8 | 13.4 / 9.9 | 821 / 644 | 1665 / 1275 | 1757 / 1318 | 11654 / 15367 | 0.5 / 0.3 |
| c32-fixed | 1552 / 1887 | 0.82 | 51 / 53 | 289 / 160 | 308 / 170 | 16.7 / 13.4 | 17.5 / 15.1 | 17.7 / 20.0 | 388 / 319 | 632 / 503 | 800 / 555 | 11370 / 15468 | 0.9 / 0.5 |
| c32-natural | 1874 / 2493 | 0.75 | 51 / 62 | 322 / 202 | 446 / 245 | 14.7 / 10.8 | 16.9 / 11.6 | 17.3 / 12.6 | 966 / 741 | 2054 / 1505 | 2276 / 1556 | 12234 / 15396 | 0.5 / 0.1 |

시간 단위 ms. 반복 편차 = 처리량 (최대 − 최소) / 평균.

- 판정: 8개 조건 모두 처리량·TPOT 미달(0.74–0.83×). C1은 사용자 결정(2026-10-09)에 따라 동률 범위로 취급하지만 수치는 0.74–0.77×입니다. TTFT p50은 C8 이상 6개 조건 모두 vLLM 이하(동률 포함)이나, TTFT p95/p99는 c16-fixed p95를 제외한 C8 이상 조건에서 vLLM보다 나쁩니다(c32: 289–322 vs 160–202 ms).
- 정상상태에서는 두 엔진 모두 이전 48건 측정보다 처리량이 높게 나옵니다(c32-natural: Riley 1,391 → 1,874, vLLM 1,802 → 2,493). 비율은 비슷하거나 약간 낮습니다. 이후 Qwen 비교는 이 조건을 기준으로 합니다.
- 출력 결정성: Riley는 모든 lane·모든 동시성에서 prompt당 출력이 1종(15개 prompt 전부). vLLM은 같은 lane 안에서 같은 prompt의 출력이 2종 이상인 경우가 45건.
- GPU 메모리: Riley 11.1–12.2 GiB, vLLM 15.2–15.5 GiB(`--gpu-memory-utilization 0.6`).

## 26. attention replay의 load 선행 채택 (commit 7efb16c7)

ragged D128 transition·reduce kernel은 partition을 순서대로 누적하는데, 매 partition의 입력을 그때그때 읽어 메모리 지연이 직렬로 쌓였습니다. 다음 4개 partition의 입력을 먼저 읽게 하고 누적 순서와 연산은 그대로 두었습니다.

- native GPU 테스트(per-row V1 reducer 및 두 producer 배치와 byte 동일) 통과, HF-exact HTTP gate attempt 16 통과(해시 `5b025fdb…` 동일). binary sha `18f7f845…`.
- 정상상태 조건(25절과 동일, Riley만 2회, `qwen-matrix/lookahead/`), 직전 빌드 대비: c1 +0.5 / +1.5%, c8 +0.8 / +2.3%, c16 +0.2 / +2.0%, c32 +0.1 / +1.5% (fixed / natural). 오류 0, 모든 prompt 출력 동일. context가 긴 natural에서만 의미 있는 이득입니다.
- 이 빌드(sha `18f7f845…`) 안정성 재실행(`qwen-scheduler-stability/lookahead-build/`): c32-natural 303초 6,912건(1,705 → 1,700 tok/s), c16-fixed 300초 14,400건, c8-natural 305초 2,688건 — 모두 오류 0·출력 불일치 0, GPU 메모리 불변, RSS 6,400–6,408 MiB 평탄, 취소 20 + 부하 중 40, KV 0 복귀, 종료 회수 통과.

## 27. step당 token 예산 확대 진단 — 미채택

가설: C32의 꼬리 TTFT(p95 274–332 ms, vLLM 160–202 ms)는 요청이 몰릴 때 step당 예산 256 token에 묶여 prefill이 여러 step으로 밀리기 때문이므로, 예산을 올리면 줄어든다. 진단 빌드(D128 행 상한 256 → 1024, 그 외 채택 빌드와 동일), 정상상태 조건, C8/C16/C32, Riley만 2회. 증거 `qwen-matrix/budget-diagnostic/`.

- attempt 1: 스크립트가 prefill chunk도 함께 올려 fixed의 sequence 상한(256)을 넘겨 서버 시작 실패(`max_prefill_chunk_tokens: must not exceed max_sequence_tokens`). 측정 없음, 보존.
- attempt 2 (chunk 256 고정), 예산 512: 6개 조건 모두 처리량 하락(c8 −2~3%, c16 −3%, c32-fixed 1,553 → 1,319, c32-natural 1,902 → 1,799), TPOT p50·p99 악화(c32-fixed 16.8 → 21.1 ms), TTFT p50 악화(c8 27 → 40 ms). C32의 TTFT p95/p99만 15–20% 개선(274 → 221, 332 → 304)되었으나 여전히 vLLM보다 나쁨. 출력은 전부 동일.
- 예산 1024: c8-fixed에서 처리량 627 → 524로 더 나빠졌고, c8-natural에서 attention workspace 할당 중 GPU OOM으로 서버 시작 실패, 중단(로그 보존).

결론: prefill을 한 step에 더 많이 넣는 것은 해법이 아닙니다. prefill token당 비용 자체가 큽니다(C32 혼합 step에서 약 37 µs/token). 다음 대상은 그 비용의 가장 큰 덩어리인 attention producer(혼합 step에서 layer당 92 µs)입니다.

## 28. packed-heads attention producer — 미채택 (되돌림)

가설: producer는 state마다 thread 32개(shared-KV는 CTA 256 thread)를 쓰고 C32에서는 한 번에 상주 가능한 양의 4–5배라, query head당 4 lane(warp 하나가 KV head의 8개 head 담당)으로 thread 수를 1/8로 줄이면 빨라진다. 각 lane이 원래 lane 8개의 부분합을 따로 유지하고 같은 tree 순서로 접어 연산은 동일하게 유지했습니다.

- native GPU 테스트: per-head·shared-KV 배치와 workspace·출력 전체 byte 동일 — 통과.
- 정상상태 진단(같은 binary에서 환경 변수로 전환, 8조건 × 2회, `qwen-matrix/packed-heads-diagnostic/`): 출력 동일, 오류 0, **모든 조건에서 느려짐**. 처리량 c1 100 → 85, c8 627/680 → 548/592, c16 1,005/1,186 → 909/1,053, c32 1,553/1,902 → 1,416/1,706. TPOT p50은 조건마다 +1.4~1.8 ms(layer당 약 +50 µs).
- 원인 판단: lane당 token마다 K/V 64개를 global memory에서 직접 읽는 직렬 사슬이 길어져(warp당 약 57 µs, 기존 약 6–7 µs), thread 수 감소 이득을 넘었습니다. producer의 한계는 thread 수가 아니라 warp당 지연이었습니다.
- 조치: commit 30c02713을 revert(patch는 `packed-heads-producer-not-adopted.patch`로 보존). 런타임은 계속 shared-KV 배치를 씁니다.

## 29. vLLM decode step 실측 — 남은 격차의 지도

지금까지 vLLM의 step 구성은 추정이었습니다. vLLM 0.27.1을 C32-natural에서 nsys(CUDA graph node 단위 추적)로 프로파일했습니다. 증거 `qwen-profile-vllm/` (원 trace는 호스트 `qwen-profile-diagnostic-vllm-c32-attempt0{1,2}`). 프로파일 중 vLLM 처리량 2,462 tok/s(비프로파일 2,493)로 왜곡은 작습니다.

- vLLM은 decode step을 **CUDA graph**(layer당 kernel 16개, 총 577개)로 실행하고 `torch.compile`로 fusion된 kernel을 씁니다. graph 구간 8.0 ms(GPU busy 7.86, 유휴 0.15), step 간격 8.8 ms. kernel 경계 간격 중앙값 64 ns(Riley eager는 약 600 ns).
- 앞 절들에서 "vLLM도 GEMM 7.5 ms + 그 외 1.5 ms"라고 본 추정은 틀렸습니다. 실제 decode step 격차는 Riley 11.9 ms 대 vLLM 8.8 ms, **약 3.1 ms**입니다.

layer당 시간(µs, C32 pure decode):

| 구성 | vLLM | Riley | 차이 |
|---|---:|---:|---:|
| Q·K·V projection (+split-K reduce, bias) | 15.0 (병합 GEMM 1개) | 30.1 (GEMM 3 + reduce 3 + bias 3) | 15.1 |
| attention (KV 기록 포함) | 26.3 (flash 18.2 + combine 3.1 + 기타) | 약 43–47 (producer 27.5 + transition + reduce + KV write) | 17–21 |
| O projection | 13.9 | 16.1 | 2.2 |
| gate·up + SiLU·곱 | 106.7 (병합 GEMM + fused) | 109.3 | 2.6 |
| down projection | 51.3 | 55.8 | 4.5 |
| residual + RMSNorm ×2 | 3.6 (fused) | 10.4 | 6.8 |
| RoPE | 1.3 | 2.8 | 1.5 |
| kernel 경계 유휴 | 약 4 (16개) | 약 16 (27개) | 12 |
| 합 | 약 222 | 약 286 | 약 63 (× 36 = 2.3 ms) |

나머지 0.8 ms는 step 단위(LM head, argmax, step 전환 host 구간)입니다.

읽는 법: 격차는 한 곳이 아니라 네 묶음입니다 — attention(약 0.7 ms), QKV 병합(약 0.55 ms), graph 실행에 따른 kernel 경계(약 0.45 ms), 작은 kernel fusion(약 0.3 ms), 그리고 GEMM 알고리즘 선택(약 0.3 ms). 네 묶음을 모두 해야 C8 이상에서 동률 근처가 됩니다. 순서: QKV 병합(+bias 분리 kernel) → attention → fusion·graph.

## 30. 병합 projection GEMM microbenchmark (`qwen-gemm-diagnostics/gemm_bench7.*`)

29절 지도의 "QKV 병합" 항목을 Riley의 알고리즘 선택 규칙(128행 anchor, 모든 행 수에 재사용)에서 확인했습니다. cold 가중치, µs, M=1 / 8 / 32 / 256.

| 형상 | 128행 anchor | 비고 |
|---|---|---|
| Q+K+V 병합 (2048 → 2560) | 17.7 / 17.7 / 18.0 / 27.2 | 분리 실행 합(Q 14.7 + K 5.0 + V 5.0 = 24.7–25.2)보다 약 7 µs 적음 |
| gate+up 병합 (2048 → 22016) | 104.2 / 105.7 / 109.1 / 156.4 | 분리 실행 합(2 × 53.7 = 107.4)과 같음 — 이득 없음 |

- QKV 병합은 bias kernel 3개가 분리·bias kernel 1개로 줄어드는 것까지 합쳐 layer당 약 10 µs, step당 약 0.38 ms(C32에서 약 3%, C1에서 약 4%)가 기대치입니다. vLLM은 M에 맞춘 알고리즘이라 13.3 µs입니다. 구현하려면 병합 가중치, 새 분리·bias kernel, 실행 계획 변경이 필요하고 GEMM 누산 순서가 달라져 scheduler 경로 출력이 바뀔 수 있습니다. **미착수**.
- gate+up 병합은 하지 않습니다.
- anchor를 64행으로 내리면 decode는 layer당 약 2.8 µs 빨라지지만 256행(prefill 혼합 step)이 layer당 약 29 µs 느려져 순손해 — 128행 유지.

## 31. fused residual+RMSNorm 진단 (guard 해제 빌드)

기존 `residual_rms_norm` kernel은 residual을 BF16으로 저장한 값으로 norm을 계산해 분리 실행과 같은 경계에서 반올림하도록 작성되어 있지만, 서버가 `--execution-completion iteration-batch`와의 조합을 막고 있습니다(11절 flag 진단에서 시작 거부). 진단 빌드에서 그 guard만 해제하고 정상상태 조건(Riley만 2회, `qwen-matrix/fused-norm-diagnostic/`)으로 측정했습니다.

- 8개 조건·15개 prompt 출력 전부 동일, 오류 0.
- 처리량 +0.1~1.2%(c16-fixed 1,005 → 1,017이 최대), TPOT p50 −0.04~−0.23 ms. 나빠진 조건은 없으나 크기는 잡음 수준에 가깝습니다.
- 상태: **보류**. step당 kernel 72개가 줄어드는 변경이라 QKV 병합과 함께 묶어 채택 여부를 정합니다(guard 해제는 별도 commit으로).

## 32. QKV 병합 projection (commit 96ca5b43) — 정상상태 A/B

`--projection-bias-backend merged-qkv-strict-staged-v1`: layer마다 Q·K·V 가중치를 이어 붙인 buffer(+약 280–380 MB GPU)로 GEMM 한 번을 실행하고, 새 `qkv_split_bias` kernel이 Q/K/V로 나누면서 기존과 같은 F32 덧셈 → BF16 반올림으로 bias를 더합니다. 기본값은 그대로이고 opt-in입니다.

- native GPU 테스트 `merged_qkv_split_bias_matches_separate_row_bias_adds`: 병합 분리 결과가 "host에서 자른 뒤 기존 row-bias add" 결과와 byte 동일(3개 형상, 부호 있는 0·무한대·NaN 포함), 폭 0 거부. 통과.
- HF-exact HTTP gate attempt 17 통과(해시 `5b025fdb…` 동일; 이 경로는 병합을 쓰지 않음). binary sha `fdba84d3…`.
- 정상상태 조건(25절과 동일), 같은 binary에서 플래그만 바꿔 각 16 lane. 증거 `qwen-matrix/merged-qkv/`. 값은 기존 / 병합 / vLLM(25절), 시간 ms.

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | 병합÷vLLM | 병합 GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 100 / 105 / 129 | 9.8 / 9.3 / 7.4 | 9.9 / 9.4 / 7.4 | 13 / 13 / 13 | 16 / 16 / 19 | 17 / 16 / 21 | 202 / 192 / 157 | 318 / 303 / 248 | 0.81 | 11360 |
| c1-natural | 99 / 104 / 132 | 9.8 / 9.4 / 7.4 | 10.0 / 9.6 / 7.5 | 13 / 13 / 13 | 38 / 37 / 25 | 39 / 37 / 26 | 632 / 603 / 482 | 1310 / 1251 / 971 | 0.79 | 11384 |
| c8-fixed | 628 / 653 / 754 | 11.4 / 11.0 / 9.1 | 12.1 / 11.8 / 9.5 | 27 / 26 / 33 | 42 / 42 / 38 | 69 / 67 / 51 | 244 / 235 / 203 | 404 / 394 / 325 | 0.87 | 11422 |
| c8-natural | 679 / 709 / 835 | 11.1 / 10.6 / 9.0 | 11.2 / 10.7 / 9.1 | 29 / 28 / 37 | 59 / 56 / 44 | 94 / 87 / 68 | 726 / 695 / 593 | 1477 / 1415 / 1185 | 0.85 | 11638 |
| c16-fixed | 1008 / 1034 / 1285 | 14.0 / 14.0 / 10.3 | 15.2 / 14.9 / 11.4 | 32 / 31 / 37 | 84 / 74 / 93 | 139 / 135 / 94 | 303 / 285 / 237 | 521 / 512 / 409 | 0.80 | 11496 |
| c16-natural | 1185 / 1232 / 1496 | 12.3 / 11.8 / 9.6 | 13.2 / 12.6 / 9.9 | 39 / 38 / 41 | 109 / 93 / 92 | 216 / 196 / 125 | 805 / 775 / 644 | 1742 / 1668 / 1318 | 0.82 | 11928 |
| c32-fixed | 1548 / 1604 / 1887 | 16.7 / 16.1 / 13.4 | 17.6 / 16.9 / 20.0 | 51 / 48 / 53 | 291 / 269 / 160 | 309 / 293 / 170 | 381 / 373 / 319 | 776 / 760 / 555 | 0.85 | 11644 |
| c32-natural | 1901 / 1978 / 2493 | 14.5 / 13.9 / 10.8 | 17.0 / 16.9 / 12.6 | 47 / 48 / 62 | 315 / 305 / 202 | 448 / 412 / 245 | 955 / 917 / 741 | 2243 / 2151 / 1556 | 0.79 | 12508 |

- 처리량 +2.5~4.9%(전 조건), TPOT p50 −0.1~−0.6 ms, 나빠진 지표 없음, 오류 0.
- 출력: 병합 모드에서도 15개 prompt 모두 조건·동시성에 관계없이 출력 1종(동시성 불변 유지). 기존 방식과 비교하면 15개 중 11개 prompt는 출력 동일, 4개는 달라짐(GEMM 누산 순서 차이). HF token 비교는 33절에서 병합 모드로 다시 실행.

## 33. QKV 병합 + fused RMSNorm 채택 (commits 96ca5b43, 3f8ba76c)

사용자 결정(2026-10-09): QKV 병합을 진행하고, 결과가 좋으면 fused RMSNorm도 함께 채택. 채택 구성 = 22·26절 구성 + `--projection-bias-backend merged-qkv-strict-staged-v1 --residual-rmsnorm fused`. fused RMSNorm은 서버가 막던 iteration-batch 조합을 native D128 경로에 한해 허용하도록 바꿨습니다(3f8ba76c). binary sha `58f13970…`.

**정확성**
- HF-exact HTTP gate attempt 18 통과(token·logit byte 일치, 해시 `5b025fdb…` 동일).
- 병합만 켠 구성과 병합 + fused 구성의 출력은 15개 prompt 전부 동일(fused RMSNorm은 출력을 바꾸지 않음).
- 채택 구성에서 15개 prompt 모두 8개 조건·모든 동시성에서 출력 1종(동시성 불변).
- HF token 비교(24절과 같은 사전 조건, `qwen-hf-token-reference/merged-qkv-fused-norm/`): 15개 중 **11개** prompt가 HF greedy와 전체 일치(이전 구성 10개). 갈라지는 4개: arithmetic(14번째/24), sql(4/16), natural-16(29/32), natural-398(69/128). C1(15건)과 C32(60건 중 44건 일치)의 결과가 같음. 갈라진 4개 위치의 HF logit margin 분석(`hf-margins.json`): 모두 HF top-1과 top-2 차이가 0–0.25(BF16 한두 칸)이고 Riley의 token은 HF 순위 0 또는 1. 그중 3개(sql, natural-16, natural-398)는 같은 HF 모델을 cache 없이 다시 돌리면 Riley의 token이 1위이고, arithmetic은 2위(차이 0.125)입니다.

**성능** (정상상태 조건, Riley 2회 평균, `qwen-matrix/merged-qkv-fused-norm/`; 값은 직전 채택 구성 / 새 채택 구성 / vLLM(25절), 시간 ms)

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p95 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | 새÷vLLM | GPU MiB 새 / vLLM | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 100 / 105 / 129 | 9.8 / 9.3 / 7.4 | 9.8 / 9.3 / 7.4 | 9.9 / 9.4 / 7.4 | 13 / 13 / 13 | 16 / 16 / 19 | 17 / 16 / 21 | 202 / 191 / 157 | 318 / 302 / 248 | 0.82 | 11360 / 15256 | 0.0 |
| c1-natural | 99 / 105 / 132 | 9.8 / 9.3 / 7.4 | 10.0 / 9.5 / 7.5 | 10.0 / 9.5 / 7.5 | 13 / 13 / 13 | 38 / 37 / 25 | 39 / 37 / 26 | 632 / 600 / 482 | 1310 / 1246 / 971 | 0.79 | 11384 / 15222 | 0.0 |
| c8-fixed | 628 / 659 / 754 | 11.4 / 10.8 / 9.1 | 12.0 / 11.4 / 9.4 | 12.1 / 11.6 / 9.5 | 27 / 26 / 33 | 42 / 43 / 38 | 69 / 66 / 51 | 244 / 231 / 203 | 404 / 390 / 325 | 0.87 | 11422 / 15222 | 0.4 |
| c8-natural | 679 / 714 / 835 | 11.1 / 10.5 / 9.0 | 11.2 / 10.7 / 9.1 | 11.2 / 10.8 / 9.1 | 29 / 28 / 37 | 59 / 55 / 44 | 94 / 77 / 68 | 726 / 690 / 593 | 1477 / 1408 / 1185 | 0.85 | 11638 / 15266 | 0.1 |
| c16-fixed | 1008 / 1055 / 1285 | 14.0 / 13.5 / 10.3 | 14.7 / 14.2 / 10.9 | 15.2 / 14.5 / 11.4 | 32 / 31 / 37 | 84 / 76 / 93 | 139 / 133 / 94 | 303 / 285 / 237 | 521 / 517 / 409 | 0.82 | 11496 / 15358 | 2.3 |
| c16-natural | 1185 / 1236 / 1496 | 12.3 / 11.8 / 9.6 | 12.6 / 12.1 / 9.8 | 13.2 / 12.6 / 9.9 | 39 / 37 / 41 | 109 / 96 / 92 | 216 / 204 / 125 | 805 / 771 / 644 | 1742 / 1672 / 1318 | 0.83 | 11928 / 15367 | 0.1 |
| c32-fixed | 1548 / 1623 / 1887 | 16.7 / 16.1 / 13.4 | 17.4 / 16.7 / 15.1 | 17.6 / 16.9 / 20.0 | 51 / 48 / 53 | 291 / 263 / 160 | 309 / 300 / 170 | 381 / 360 / 319 | 776 / 748 / 555 | 0.86 | 11644 / 15468 | 0.2 |
| c32-natural | 1901 / 1979 / 2493 | 14.5 / 13.9 / 10.8 | 16.4 / 16.6 / 11.6 | 17.0 / 17.1 / 12.6 | 47 / 45 / 62 | 315 / 324 / 202 | 448 / 433 / 245 | 955 / 917 / 741 | 2243 / 2166 / 1556 | 0.79 | 12508 / 15396 | 0.7 |

오류 0. 처리량 +4.1~5.4%(전 조건). 직전 구성 대비 나빠진 것은 c32-natural의 TPOT p95(16.4 → 16.6), TPOT p99(17.0 → 17.1), TTFT p95(315 → 324)와 c8-fixed TTFT p95(42 → 43)로, 모두 3% 이내입니다.

**안정성** (채택 구성, `qwen-scheduler-stability/merged-qkv-fused-norm-build/`)

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-natural | 303초 | 7,200 | 0 | 0 | 1,783 / 1,770 | 12,508 / 12,508 | 6,407 / 6,408 | 20 / 40 | 0 | 통과 |
| c16-fixed | 302초 | 15,168 | 0 | 0 | 1,015 / 994 | 11,496 / 11,496 | 6,403 / 6,403 | 20 / 40 | 0 | 통과 |
| c8-natural | 301초 | 2,784 | 0 | 0 | 691 / 690 | 11,638 / 11,638 | 6,401 / 6,401 | 20 / 40 | 0 | 통과 |
| c1-fixed | 310초 | 1,632 | 0 | 0 | 105 / 105 | 11,360 / 11,360 | 6,400 / 6,400 | 20 / 40 | 0 | 통과 |

**판정**: 채택. Qwen은 여전히 8개 조건 모두 vLLM 미달(0.79–0.87×)이며, vLLM 수치는 25절 측정을 재사용했습니다(같은 실행에서 다시 재지 않음). 29절 지도에서 남은 묶음은 attention(약 0.7 ms), graph 실행(약 0.45 ms), GEMM 알고리즘 선택·기타 kernel입니다.

## 34. batch 형상 bucket 세분화

배경: power-of-two 정책은 한 step의 행 수(decode 행 + prefill token)를 1, 2, 4, …, 256으로 올림합니다. 예를 들어 decode 32행 + prefill 100 token = 132행은 256행으로 계산됩니다. GEMM 비용은 가중치 읽기 바닥을 넘는 부분이 행 수에 비례하므로 prefill이 섞인 step에서 낭비가 큽니다. 알고리즘은 128행 anchor로 고정되어 있어 bucket을 바꿔도 행별 결과는 같아야 합니다.

**10개 bucket** (`--batch-shape-buckets 1,8,16,32,64,96,128,160,208,256`, 코드 변경 없음, 채택 구성 위에서 정상상태 2회, `qwen-matrix/shape-buckets/`):
- 출력 15개 prompt 전부 동일, 오류 0.
- 처리량: c1 +0.2 / +0.1%, c8 +2.0 / +0.4%, c16 +5.9 / +1.2%, c32 +0.9 / +0.7% (fixed / natural). TPOT p50은 같거나 감소(c16-fixed 13.49 → 12.66 ms).
- 나빠진 지표: c8-natural TTFT p99 77 → 92 ms, c16-natural TTFT p95 96 → 105 ms, c32-fixed TTFT p95 263 → 268 ms, c8-fixed·c32-fixed E2E p50 +1~2%.
- 처음 시도한 15개 목록은 서버 한도(10개)에 걸려 실행 전에 교체했습니다(측정 없음).

**21개 bucket** (commit 12cc92a4로 한도 10 → 24, `--batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256`): HF-exact HTTP gate attempt 19 통과(해시 `5b025fdb…` 동일), binary sha `d89bef71…`. 정상상태 2회, 값은 33절 구성(power-of-two) / 21개 bucket / vLLM(25절), 시간 ms.

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p95 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | 새÷vLLM | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 105 / 105 / 129 | 9.3 / 9.3 / 7.4 | 9.3 / 9.3 / 7.4 | 9.4 / 9.3 / 7.4 | 13 / 13 / 13 | 16 / 14 / 19 | 16 / 14 / 21 | 191 / 193 / 157 | 302 / 301 / 248 | 0.82 | 0.0 |
| c1-natural | 105 / 105 / 132 | 9.3 / 9.3 / 7.4 | 9.5 / 9.5 / 7.5 | 9.5 / 9.5 / 7.5 | 13 / 13 / 13 | 37 / 35 / 25 | 37 / 35 / 26 | 600 / 600 / 482 | 1246 / 1244 / 971 | 0.79 | 0.1 |
| c8-fixed | 659 / 673 / 754 | 10.8 / 10.6 / 9.1 | 11.4 / 11.1 / 9.4 | 11.6 / 11.2 / 9.5 | 26 / 26 / 33 | 43 / 42 / 38 | 66 / 66 / 51 | 231 / 230 / 203 | 390 / 380 / 325 | 0.89 | 0.2 |
| c8-natural | 714 / 717 / 835 | 10.5 / 10.5 / 9.0 | 10.7 / 10.6 / 9.1 | 10.8 / 10.6 / 9.1 | 28 / 28 / 37 | 55 / 58 / 44 | 77 / 89 / 68 | 690 / 686 / 593 | 1408 / 1400 / 1185 | 0.86 | 0.1 |
| c16-fixed | 1055 / 1116 / 1285 | 13.5 / 12.7 / 10.3 | 14.2 / 13.0 / 10.9 | 14.5 / 13.4 / 11.4 | 31 / 27 / 37 | 76 / 77 / 93 | 133 / 127 / 94 | 285 / 270 / 237 | 517 / 473 / 409 | 0.87 | 0.4 |
| c16-natural | 1236 / 1255 / 1496 | 11.8 / 11.6 / 9.6 | 12.1 / 12.0 / 9.8 | 12.6 / 12.6 / 9.9 | 37 / 36 / 41 | 96 / 96 / 92 | 204 / 200 / 125 | 771 / 757 / 644 | 1672 / 1651 / 1318 | 0.84 | 0.0 |
| c32-fixed | 1623 / 1651 / 1887 | 16.1 / 15.7 / 13.4 | 16.7 / 16.2 / 15.1 | 16.9 / 16.4 / 20.0 | 48 / 48 / 53 | 263 / 256 / 160 | 300 / 290 / 170 | 360 / 360 / 319 | 748 / 738 / 555 | 0.87 | 0.7 |
| c32-natural | 1979 / 2007 / 2493 | 13.9 / 13.8 / 10.8 | 16.6 / 15.8 / 11.6 | 17.1 / 16.2 / 12.6 | 45 / 47 / 62 | 324 / 307 / 202 | 433 / 427 / 245 | 917 / 901 / 741 | 2166 / 2131 / 1556 | 0.80 | 0.1 |

- 출력 15개 prompt 전부 동일(33절 구성 및 10개 bucket과), 오류 0. 처리량 +0.1~5.8%, 10개 bucket 대비로는 C32에서 +0.7% 추가.
- 나빠진 지표: c8-natural TTFT p95 55 → 58 ms, p99 77 → 89 ms(10개 bucket에서도 92 ms로 같은 방향), c32-natural TTFT p50 45 → 47 ms, c1-fixed E2E p50 191 → 193 ms, c16-fixed TTFT p95 76 → 77 ms.
- 판정: **채택**(21개 목록). c8-natural의 TTFT 꼬리 악화는 두 번의 bucket 측정에서 같은 방향이라 실제 효과일 수 있어 한계로 남깁니다.

**21개 bucket 구성 안정성** (`qwen-scheduler-stability/shape-buckets21-build/`, binary sha `d89bef71…`)

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-natural | 300초 | 7,200 | 0 | 0 | 1,788 / 1,788 | 12,508 / 12,508 | 6,408 / 6,409 | 20 / 40 | 0 | 통과 |
| c16-fixed | 301초 | 15,840 | 0 | 0 | 1,060 / 1,049 | 11,496 / 11,496 | 6,403 / 6,404 | 20 / 40 | 0 | 통과 |
| c8-fixed | 301초 | 9,792 | 0 | 0 | 654 / 651 | 11,422 / 11,422 | 6,401 / 6,401 | 20 / 40 | 0 | 통과 |
| c1-natural | 342초 | 480 | 0 | 0 | 105 / 105 | 11,384 / 11,384 | 6,400 / 6,400 | 20 / 40 | 0 | 통과 |

현재 채택 구성(브랜치 `codex/qwen-http-hf-exact` head 12cc92a4)의 플래그: 22절 구성 + `--sampling-backend gpu-greedy --projection-bias-backend merged-qkv-strict-staged-v1 --residual-rmsnorm fused --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256`.

## 35. 촘촘한 bucket + 더 큰 token 예산 진단 — 미채택

가설: 27절에서 예산 확대가 실패한 것은 257행 이상이 512행으로 올림됐기 때문이고, 촘촘한 bucket이 있으면 예산을 키워 C32 꼬리 TTFT를 줄이면서 처리량을 유지할 수 있다. 진단 빌드(D128 행 상한 256 → 512, 그 외 채택 구성과 동일), 예산 384와 512에 각각 24개 bucket, 정상상태 8조건 × 2회. 증거 `qwen-matrix/budget-fine-buckets-diagnostic/`. 비교 기준은 34절 채택 구성(예산 256, 21개 bucket).

| | 예산 384 | 예산 512 |
|---|---|---|
| 처리량 | c1 ±0, c8 −1.3 / −0.7%, c16 −2.3 / −1.3%, c32 −4.8 / −2.4% | c8 −2.3 / −1.4%, c16 −4.1 / −2.3%, c32 −6.3 / −3.7% |
| TPOT p50 (c32-fixed) | 15.7 → 17.2 ms | 15.7 → 17.7 ms |
| TPOT p99 (c32-fixed) | 16.4 → 22.6 ms | 16.4 → 26.4 ms |
| TTFT p95 / p99 (c32-fixed) | 256 → 223 / 290 → 261 ms | 256 → 215 / 290 → 255 ms |
| TTFT p50 (c8·c16) | 26–36 → 28–43 ms | 26–36 → 33–43 ms |
| GPU 최대 | 14.7 GiB | 16.8 GiB |

출력은 전부 동일, 오류 0. 꼬리 TTFT는 10–15% 줄지만(여전히 vLLM 160–202 ms보다 큼) 처리량·TPOT·TTFT p50·메모리가 모두 나빠집니다. 가설 기각: 낭비의 원인은 올림만이 아니고, 큰 step에서 prefill token당 비용 자체가 늘어납니다. **미채택**, 예산 256 유지.

## 36. CUDA graph 실행 조사 — 구조적 제약 (미착수)

29절 지도의 "graph 실행" 묶음(step당 약 0.45 ms)을 기존 graph 구조로 얻을 수 있는지 코드를 조사했습니다(읽기 전용, 빌드·실행 없음).

- 현재 실제로 capture·replay되는 decode graph는 **한 sequence, 한 token(M=1), head 크기 64, bias 없는 모델** 전용입니다(`graph_decode_full.rs`의 admission, `kernels/src/graph_resources.cu`의 recorder). `PURE_DECODE_GRAPH_BUCKETS`와 `graph_decode_*` 파일들은 계획·감사용이고 서버의 replay 경로에 연결되어 있지 않습니다.
- Qwen이 걸리는 지점: head 크기 64 고정, Q/K/V bias 거부, 1행 고정, grouped-heads attention만 허용, fused residual norm 불가, split-K GEMM 계획 거부, 병합 QKV 계획 슬롯 없음, physical block 4096 상한, 1행 argmax(후보 상한 없음).
- D128 attention이 eager 전용인 실제 이유: producer·transition·reduce의 launch grid와 값 인자(`launch_partial_state_count`, `active_row_count`)가 step마다 host에서 계산되어 capture 시점에 고정됩니다. kernel 자체는 범위를 벗어난 행·page에서 조기 종료하므로 고정 grid로도 결과는 맞지만, 최대 page 수로 고정하면 빈 CTA가 대량으로 launch됩니다.
- 가장 큰 구조 문제: graph는 수명 동안 모든 buffer·GEMM 계획을 배타적으로 점유합니다. Qwen은 같은 KV pool에서 prefill·혼합 step을 eager로 계속 돌려야 하므로 점유를 step 단위로 넘겨주는 구조가 새로 필요합니다.

필요한 작업(요약): D128·N행용 native recorder와 enqueue 함수군, 여러 행의 metadata slab과 replay 검증, padding 행 처리 규칙, bucket별 graph 여러 개를 한 소유권 장부에서 관리, eager와 graph의 자원 공유, 서버 engine이 pure-decode step만 graph로 보내는 분기, 그리고 bucket마다 eager와 byte 동일성 검증. 새 하위 시스템 규모이며 기대 이득은 step당 약 0.45 ms(약 4%)입니다.

판정: 지금 구조에서는 "설정으로 켤 수 있는" 항목이 아닙니다. **미착수**. 재개 조건: graph와 eager의 자원 공유 방식 결정, 그리고 split-K GEMM 계획이 capture에서 동일 결과를 내는지 확인.

**attention 묶음에 대한 판단**: shared-KV producer는 thread-시간 기준으로 이미 효율적이고(28절의 packed-heads 실패, token 병렬 배치의 thread 수 증가), vLLM의 flash attention과의 차이는 연산 구조 자체에서 옵니다. page 순서 online softmax 계약을 유지한 채로는 더 줄일 여지가 작습니다. 줄이려면 새 수치 계약의 attention backend가 필요합니다(미착수).

## 37. 채택 구성 vs vLLM — 같은 실행에서 다시 측정

34절까지의 vLLM 수치는 25절 측정을 재사용한 것이었습니다. 채택 구성(브랜치 head 12cc92a4, binary sha `d89bef71…`)과 vLLM 0.27.1을 같은 실행에서 교대 순서로 다시 쟀습니다. 정상상태 조건(warmup 32 / 192건, 2회, host gate 없음, 제외 없음). 증거 `qwen-matrix/steady-adopted/`. 32 lane 완료, 오류 0. 값은 Riley / vLLM, 시간 ms.

| 조건 | 처리량 tok/s | 비율 | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 | E2E p50 | E2E p95 | E2E p99 | GPU MiB | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 105 / 130 | 0.81 | 13 / 13 | 14 / 16 | 14 / 20 | 9.3 / 7.4 | 9.3 / 7.4 | 9.3 / 7.4 | 190 / 156 | 301 / 243 | 302 / 246 | 11360 / 15170 | 0.0 / 0.1 |
| c1-natural | 105 / 132 | 0.79 | 13 / 13 | 35 / 25 | 36 / 25 | 9.3 / 7.4 | 9.5 / 7.4 | 9.5 / 7.5 | 600 / 482 | 1243 / 970 | 1244 / 971 | 11384 / 15136 | 0.0 / 0.0 |
| c8-fixed | 671 / 757 | 0.89 | 26 / 32 | 41 / 38 | 66 / 51 | 10.7 / 9.1 | 11.1 / 9.4 | 11.2 / 9.6 | 227 / 203 | 368 / 318 | 378 / 323 | 11422 / 15136 | 0.6 / 1.0 |
| c8-natural | 717 / 838 | 0.86 | 27 / 37 | 54 / 42 | 97 / 63 | 10.5 / 8.9 | 10.6 / 9.0 | 10.8 / 9.1 | 687 / 592 | 1390 / 1178 | 1403 / 1181 | 11638 / 15155 | 0.2 / 0.1 |
| c16-fixed | 1117 / 1279 | 0.87 | 27 / 37 | 68 / 94 | 135 / 94 | 12.7 / 10.3 | 13.0 / 10.9 | 13.9 / 11.2 | 279 / 237 | 429 / 368 | 485 / 411 | 11496 / 15272 | 0.4 / 0.0 |
| c16-natural | 1242 / 1500 | 0.83 | 36 / 40 | 85 / 95 | 200 / 125 | 11.6 / 9.6 | 11.9 / 9.8 | 12.3 / 9.9 | 757 / 645 | 1537 / 1270 | 1633 / 1314 | 11928 / 15280 | 2.2 / 0.0 |
| c32-fixed | 1660 / 1877 | 0.88 | 48 / 53 | 275 / 170 | 291 / 171 | 15.6 / 13.5 | 16.1 / 15.0 | 16.4 / 19.0 | 357 / 323 | 580 / 502 | 714 / 560 | 11644 / 15336 | 1.4 / 0.1 |
| c32-natural | 2001 / 2493 | 0.80 | 43 / 63 | 319 / 202 | 432 / 240 | 13.8 / 10.8 | 15.5 / 11.5 | 16.7 / 13.1 | 901 / 743 | 1931 / 1502 | 2139 / 1552 | 12508 / 15288 | 0.1 / 0.1 |

- vLLM 수치는 25절과 1% 이내로 재현됩니다. Riley는 8개 조건 모두 처리량·TPOT 미달(0.79–0.89×).
- Riley가 vLLM 이하인 지표: TTFT p50(전 조건, C1은 동률), c1-fixed의 TTFT p95/p99, c16의 TTFT p95, c32-fixed의 TPOT p99(16.4 vs 19.0).
- 출력: Riley는 15개 prompt 모두 조건·동시성과 무관하게 1종. vLLM은 같은 lane 안에서 같은 prompt의 출력이 갈린 경우 43건.

## 38. 새 attention(2번) — microbenchmark로 설계 검증

사용자 결정(2026-10-09): 구조적 선택지 중 "새 수치 계약의 attention"을 먼저 진행. 본 코드에 넣기 전에 독립 microbenchmark(`qwen-attention-v2-diagnostics/attn_bench*.cu`, 같은 page 배치·데이터)로 속도 가정을 확인했습니다.

**기각된 설계 두 가지**
- two-pass softmax, token 32개 단위 CTA(`attn_bench`): 32행·1000 token에서 239 µs. thread 하나가 128차원 내적을 직렬로 도는 구조가 GPU에서 매우 느립니다(1행에서도 250 µs).
- page 병렬 two-pass(점수 → 통계 → 가중합 → page 합, `attn_bench2/3`): 32행에서 99.5 / 54.7 / 21.5 µs(1000 / 500 / 150 token). 같은 harness의 현재 방식은 92.1 / 49.9 / 20.0 µs. **빠르지 않습니다.**
- 원인: 32행·1000 token이면 K와 V를 합쳐 64 MB를 읽어야 하고, 현재 producer(64 µs)는 이미 그 읽기 속도(약 1 GB/ms)에 붙어 있습니다. attention의 page 처리 단계는 C8 이상에서 메모리 대역폭에 묶여 있어, 연산 구조를 바꿔도 줄지 않습니다. 29절 지도에서 attention 격차를 producer 탓으로 본 것은 틀렸고, 격차는 page 결과를 합치는 두 kernel에 있습니다.

**채택 후보: global-max page merge** (`attn_bench4`)
- producer는 그대로 두고, 순서대로 합치는 transition + replay 두 kernel을 kernel 하나로 대체합니다. page 최댓값 중 최댓값 M을 구하고, 각 page를 exp(m − M)으로 가중해 page 오름차순으로 합칩니다. 수학적으로 같은 softmax 합성이고 반올림만 다릅니다.
- microbenchmark: transition + replay 7–28 µs → merge 3–7 µs(행 256에서는 14–25 → 9–15). 현재 방식과 출력 원소의 99.99% 이상이 bit 동일, 나머지는 BF16 한 칸 차이.
- 본 코드 반영: commit 52717649, `--decode-attention-merge global-max-v1`(기본값 ordered-v1 유지, 행당 128 page = 2048 token 이하). 검증 결과는 39절.

## 39. global-max page merge 검증 (commits 52717649, 56f463d5)

**첫 시도 실패(보존)**: 52717649는 kernel을 단순화하려고 "행당 128 page 이하"를 요구했는데, 실제 workspace 용량은 그보다 큽니다. native GPU 테스트가 그 검사에서 거부됐고(`qwen-globalmerge-native-gpu-test-attempt01-failed-capacity-limit.log`), 서버 측정(attempt 01)에서는 모든 요청이 token 없이 실패했습니다. microbenchmark는 이 조건을 드러내지 못했습니다. 56f463d5에서 page 수 제한을 없앴습니다(thread가 128 간격으로 page를 접고, 128번째 이후 page의 가중치는 같은 값을 다시 계산).

**수정 후** (binary sha `fa539dbd…`, 증거 `qwen-matrix/global-merge/`, `qwen-hf-token-reference/global-merge/`)
- native GPU 테스트 통과: ordered merge와 원소의 99.9% 이상 bit 동일, 나머지는 BF16 한 칸 이내, NaN 위치 동일, 비활성 행 0 동일.
- HF-exact HTTP gate attempt 21 통과(해시 `5b025fdb…` 동일).
- 정상상태, 같은 계열 binary에서 끈 것 / 켠 것 / vLLM(37절, 같은 날 재측정). 시간 ms, 2회 평균.

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p95 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | 켠 것÷vLLM | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 105 / 107 / 130 | 9.3 / 9.2 / 7.4 | 9.3 / 9.2 / 7.4 | 9.3 / 9.2 / 7.4 | 13 / 13 / 13 | 14 / 14 / 16 | 14 / 14 / 20 | 190 / 188 / 156 | 302 / 299 / 246 | 0.82 | 0.0 |
| c1-natural | 105 / 107 / 132 | 9.3 / 9.2 / 7.4 | 9.5 / 9.2 / 7.4 | 9.5 / 9.3 / 7.5 | 13 / 13 / 13 | 35 / 35 / 25 | 35 / 35 / 25 | 600 / 592 / 482 | 1243 / 1210 / 971 | 0.81 | 0.0 |
| c8-fixed | 672 / 680 / 757 | 10.6 / 10.5 / 9.1 | 11.1 / 11.0 / 9.4 | 11.2 / 11.1 / 9.6 | 26 / 26 / 32 | 41 / 39 / 38 | 67 / 66 / 51 | 232 / 231 / 203 | 376 / 371 / 323 | 0.90 | 0.5 |
| c8-natural | 717 / 737 / 838 | 10.5 / 10.2 / 8.9 | 10.6 / 10.3 / 9.0 | 10.8 / 10.4 / 9.1 | 26 / 27 / 37 | 54 / 53 / 42 | 90 / 71 / 63 | 686 / 667 / 592 | 1402 / 1360 / 1181 | 0.88 | 0.1 |
| c16-fixed | 1120 / 1126 / 1279 | 12.7 / 12.5 / 10.3 | 13.0 / 12.9 / 10.9 | 13.3 / 13.6 / 11.2 | 27 / 27 / 37 | 72 / 77 / 94 | 132 / 133 / 94 | 269 / 282 / 237 | 458 / 477 / 411 | 0.88 | 0.1 |
| c16-natural | 1257 / 1287 / 1500 | 11.6 / 11.3 / 9.6 | 11.9 / 11.6 / 9.8 | 12.4 / 12.2 / 9.9 | 36 / 36 / 40 | 99 / 96 / 95 | 204 / 187 / 125 | 756 / 739 / 645 | 1651 / 1609 / 1314 | 0.86 | 0.1 |
| c32-fixed | 1650 / 1663 / 1877 | 15.7 / 15.6 / 13.5 | 16.3 / 16.1 / 15.0 | 16.6 / 16.6 / 19.0 | 46 / 46 / 53 | 266 / 271 / 170 | 294 / 283 / 171 | 359 / 362 / 323 | 738 / 740 / 560 | 0.89 | 0.0 |
| c32-natural | 2004 / 2044 / 2493 | 13.7 / 13.5 / 10.8 | 15.9 / 15.5 / 11.5 | 17.1 / 16.4 / 13.1 | 48 / 39 / 63 | 323 / 315 / 202 | 414 / 421 / 240 | 902 / 885 / 743 | 2120 / 2093 / 1552 | 0.82 | 0.0 |

- 처리량 +0.5~2.7%(fixed +0.5~1.2%, natural +2.0~2.7%), 오류 0. microbenchmark로 기대한 step당 0.3 ms보다 작습니다(실제 0.1~0.27 ms).
- 나빠진 지표: c16-fixed의 E2E p50(269 → 282)·p99(458 → 477), TPOT p99(13.3 → 13.6), TTFT p95(72 → 77); c32-fixed TTFT p95(266 → 271); c32-natural TTFT p99(414 → 421).
- 출력: 15개 prompt 모두 조건·동시성에 무관하게 1종(동시성 불변 유지). 끈 구성과 비교해 13개 prompt 출력 동일, 2개 변경.
- HF token 비교: 15개 중 11개 일치(끈 구성도 11개). 갈라지는 prompt는 arithmetic(14), planning(25), sql(4), natural-398(69) — natural-16이 일치로 바뀌고 planning이 새로 갈라짐. C1과 C32 결과 동일.

**HF margin** (`qwen-hf-token-reference/global-merge/…/hf-margins.json`): 갈라진 4개 위치 모두 HF top-1과 top-2 차이 0–0.25(BF16 한두 칸). planning(25)·sql(4)·natural-398(69)은 cache 없이 다시 돌린 HF에서 Riley의 token이 1위, arithmetic(14)은 2위(차이 0.125).

**안정성** (global merge 켠 구성, `qwen-scheduler-stability/global-merge-build/`)

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-natural | 302초 | 7,392 | 0 | 0 | 1,833 / 1,831 | 12,508 / 12,508 | 6,408 / 6,409 | 20 / 40 | 0 | 통과 |
| c16-fixed | 301초 | 15,936 | 0 | 0 | 1,066 / 1,050 | 11,496 / 11,496 | 6,403 / 6,404 | 20 / 40 | 0 | 통과 |
| c8-natural | 302초 | 2,880 | 0 | 0 | 714 / 711 | 11,638 / 11,638 | 6,401 / 6,401 | 20 / 40 | 0 | 통과 |
| c1-fixed | 307초 | 1,632 | 0 | 0 | 107 / 106 | 11,360 / 11,360 | 6,400 / 6,401 | 20 / 40 | 0 | 통과 |

**판정**: 채택(opt-in 플래그 `--decode-attention-merge global-max-v1`를 채택 구성에 추가). c16-fixed의 지연 지표 악화는 2회 측정에서만 본 것이고, 같은 조건의 300초 지속 부하 처리량은 직전 구성(1,060 → 1,049)과 같은 수준(1,066 → 1,050)이라 한계로만 남깁니다.

현재 채택 구성(브랜치 head 56f463d5): `--decode-attention-backend native-bf16-paged-split-gqa-d128-two-stage --decode-attention-merge global-max-v1 --gemm-reduction-policy first-heuristic-split-k-v1 --sampling-backend gpu-greedy --projection-bias-backend merged-qkv-strict-staged-v1 --residual-rmsnorm fused --batch-shape-policy power-of-two --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --batch-token-budget 256 --prefill-chunk-tokens 256 --execution-completion iteration-batch --execution-graph-policy disabled`. vLLM 대비 0.81–0.90×.

## 40. 채택 구성 안정성 8개 조건 전체, 그리고 prefill attention 가설 두 가지 기각

**안정성 나머지 4개 조건** (39절과 같은 구성·binary, `qwen-scheduler-stability/global-merge-build/`). 39절의 4개와 합쳐 8개 핵심 조건 전부를 덮습니다.

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-fixed | 301초 | 22,272 | 0 | 0 | 1,510 / 1,469 | 11,644 / 11,644 | 6,407 / 6,409 | 20 / 40 | 0 | 통과 |
| c16-natural | 302초 | 4,896 | 0 | 0 | 1,213 / 1,208 | 11,928 / 11,928 | 6,403 / 6,404 | 20 / 40 | 0 | 통과 |
| c8-fixed | 301초 | 9,888 | 0 | 0 | 658 / 658 | 11,422 / 11,422 | 6,401 / 6,401 | 20 / 40 | 0 | 통과 |
| c1-natural | 335초 | 480 | 0 | 0 | 107 / 107 | 11,384 / 11,384 | 6,400 / 6,401 | 20 / 40 | 0 | 통과 |

c32-fixed의 처리량은 첫 round 대비 끝 round가 2.7% 낮습니다(RSS·GPU 메모리는 평탄).

**prefill이 섞인 step의 attention(layer당 약 92 µs)을 줄이려는 가설 두 가지** — 둘 다 microbenchmark에서 기각, 본 코드 미반영(`qwen-attention-v2-diagnostics/attn_bench5/6.*`).
- 가설 A: 긴 decode 행의 page 수에 맞춘 launch grid 때문에 짧은 prefill 행이 빈 CTA를 대량으로 만든다. 측정: 32행×400 token + 224행×130 token을 한 번에 launch하면 84.5 µs, 따로 launch한 합은 27.7 + 59.2 = 86.9 µs. 빈 CTA 비용은 없습니다. **기각.**
- 가설 B: 같은 sequence의 prefill 행 4개가 K/V tile 하나를 공유하면 tile 적재가 줄어 빨라진다. 측정(state 전부 bit 동일): 224행 36.7 → 44.9 µs, 58.6 → 70.5 µs, 128행 67.7 → 82.1 µs. 모두 **더 느림. 기각.** tile 적재가 아니라 (page, head, 행)마다의 warp 연산이 비용입니다.

## 41. 채택 구성 step 재프로파일과 host 단계 타이머

증거 `qwen-profile-adopted/` (`…attempt06/step.txt`, `phase-timing.txt`, 진단 patch `phase_patch.py`는 커밋하지 않음).

**C32-natural step 구성** (채택 구성, 프로파일 중 처리량 2,017 tok/s):

| | pure decode (step의 67%, 시간의 56%) | prefill 혼합 (step의 33%, 시간의 44%) |
|---|---:|---:|
| 평균 주기 | 10.78 ms | 17.10 ms |
| GEMM (cuBLASLt + split-K reduce + LM head) | 8.06 | 11.09 |
| attention producer | 0.99 (27.5 µs × 36) | 3.17 (88 µs × 36) |
| page merge | 0.13 | 0.33 |
| norm·RoPE·SiLU·bias·KV write·argmax 등 | 약 0.85 | 약 1.4 |
| GPU 유휴 | 0.76 | 1.09 |

step당 kernel 수 692(이전 980). pure decode는 11.9 → 10.8 ms로 줄었습니다. vLLM의 decode step 간격은 8.8 ms(29절)입니다. 남은 약 2 ms: GEMM 약 0.84(128행 anchor 알고리즘 재사용의 비용, vLLM은 행 수별 선택), 유휴·step 사이 약 0.5, attention 약 0.25, norm 등 약 0.4.

**엔진 step 내부 host 단계** (진단 타이머, µs, step 평균):

| 조건 | 계획 | sampling | commit | 이벤트 생성 | step 사이 |
|---|---:|---:|---:|---:|---:|
| c1-natural | 1.5–2.2 | 0.9–1.3 | 0.6–0.9 | 0.2–0.3 | 15–19 |
| c8-fixed | 5.4 | 5.4 | 2.2 | 1.4–1.5 | 148–177 |
| c8-natural | 5.7–6.6 | 4.3–5.1 | 1.4–1.8 | 0.9–1.1 | 66–73 |
| c16-fixed | 11.7 | 11.5 | 5.2 | 3.1 | 332 |
| c16-natural | 14.0–16.5 | 9.0–10.3 | 3.4–4.0 | 2.2–2.5 | 100–146 |
| c32-natural | 21.6–29.9 | 12.4–19.6 | 5.5–8.4 | 2.9–4.5 | 141–287 |

step 내부 host 단계의 합은 C32에서도 약 45–60 µs입니다. 큰 것은 **step 사이 구간**(C8 이상 66–332 µs, 요청 회전이 잦은 fixed에서 더 큼)이고, 이 구간에서 엔진 thread가 새 요청의 prompt를 tokenize합니다(`CudaBackend::admit`). SmolLM2 계열(Batch13)에는 submit하는 thread에서 미리 encode하는 변경이 있었지만 이 계열에는 없었습니다. commit e7a3348b로 같은 변경을 옮겼습니다(엔진에 활성 요청이 있을 때만, 검증 순서·오류 불변). 검증 결과는 42절.

## 42. submit thread에서 prompt encode (commit e7a3348b) — 검증

HF-exact HTTP gate attempt 22 통과(해시 `5b025fdb…` 동일; 이 gate는 prefill 중 취소·token 3개 뒤 취소·재요청·대기열·종료 회수를 포함). binary sha `c94b7831…`. 정상상태, 직전 채택 구성(39절의 켠 구성) / 새 빌드 / vLLM(37절). 시간 ms, 2회 평균. 증거 `qwen-matrix/pretokenize/`.

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p95 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | 새÷vLLM | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 107 / 107 / 130 | 9.2 / 9.2 / 7.4 | 9.2 / 9.2 / 7.4 | 9.2 / 9.2 / 7.4 | 13 / 13 / 13 | 14 / 14 / 16 | 14 / 14 / 20 | 188 / 188 / 156 | 299 / 299 / 246 | 0.82 | 0.0 |
| c1-natural | 107 / 107 / 132 | 9.2 / 9.2 / 7.4 | 9.2 / 9.2 / 7.4 | 9.3 / 9.3 / 7.5 | 13 / 13 / 13 | 35 / 35 / 25 | 35 / 35 / 25 | 592 / 592 / 482 | 1210 / 1210 / 971 | 0.81 | 0.0 |
| c8-fixed | 680 / 686 / 757 | 10.5 / 10.4 / 9.1 | 11.0 / 10.9 / 9.4 | 11.1 / 11.0 / 9.6 | 26 / 25 / 32 | 39 / 39 / 38 | 66 / 66 / 51 | 231 / 229 / 203 | 371 / 367 / 323 | 0.91 | 0.2 |
| c8-natural | 737 / 740 / 838 | 10.2 / 10.2 / 8.9 | 10.3 / 10.3 / 9.0 | 10.4 / 10.4 / 9.1 | 27 / 24 / 37 | 53 / 50 / 42 | 71 / 76 / 63 | 667 / 665 / 592 | 1360 / 1358 / 1181 | 0.88 | 0.1 |
| c16-fixed | 1126 / 1150 / 1279 | 12.5 / 12.2 / 10.3 | 12.9 / 12.7 / 10.9 | 13.6 / 13.9 / 11.2 | 27 / 26 / 37 | 77 / 71 / 94 | 133 / 132 / 94 | 282 / 271 / 237 | 477 / 469 / 411 | 0.90 | 0.1 |
| c16-natural | 1287 / 1300 / 1500 | 11.3 / 11.2 / 9.6 | 11.6 / 11.7 / 9.8 | 12.2 / 12.1 / 9.9 | 36 / 34 / 40 | 96 / 79 / 95 | 187 / 191 / 125 | 739 / 732 / 645 | 1609 / 1596 / 1314 | 0.87 | 0.1 |
| c32-fixed | 1663 / 1726 / 1877 | 15.6 / 15.1 / 13.5 | 16.1 / 15.7 / 15.0 | 16.6 / 16.0 / 19.0 | 46 / 44 / 53 | 271 / 256 / 170 | 283 / 296 / 171 | 362 / 349 / 323 | 740 / 698 / 560 | 0.92 | 0.0 |
| c32-natural | 2044 / 2072 / 2493 | 13.5 / 13.3 / 10.8 | 15.5 / 16.0 / 11.5 | 16.4 / 16.5 / 13.1 | 39 / 42 / 63 | 315 / 321 / 202 | 421 / 408 / 240 | 885 / 869 / 743 | 2093 / 2061 / 1552 | 0.83 | 0.2 |

- 처리량: C1 변화 없음(의도대로, 엔진이 한가할 때는 기존 경로), c8 +0.9 / +0.4%, c16 +2.2 / +1.0%, c32 +3.7 / +1.4% (fixed / natural). 오류 0, 15개 prompt 출력 전부 동일.
- 나빠진 지표: c32-natural TTFT p50(39 → 42)·p95(315 → 321)·TPOT p95(15.5 → 16.0), c32-fixed TTFT p99(283 → 296), c8-natural TTFT p99(71 → 76), c16-fixed TPOT p99(13.6 → 13.9), c16-natural TTFT p99(187 → 191).
- 판정: 채택(지속 부하·취소 안정성은 43절).

## 43. pre-tokenize 빌드 안정성과 c32-fixed 처리량 하락 관찰

**안정성** (채택 구성 + e7a3348b, binary sha `c94b7831…`, `qwen-scheduler-stability/pretokenize-build/`)

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-fixed | 301초 | 22,944 | 0 | 0 | 1,554 / 1,506 | 11,644 / 11,644 | 6,408 / 6,409 | 20 / 40 | 0 | 통과 |
| c32-natural | 302초 | 7,488 | 0 | 0 | 1,864 / 1,846 | 12,508 / 12,508 | 6,408 / 6,409 | 20 / 40 | 0 | 통과 |
| c16-fixed | 301초 | 16,320 | 0 | 0 | 1,090 / 1,081 | 11,496 / 11,496 | 6,403 / 6,404 | 20 / 40 | 0 | 통과 |
| c8-natural | 300초 | 2,880 | 0 | 0 | 718 / 716 | 11,638 / 11,638 | 6,401 / 6,402 | 20 / 40 | 0 | 통과 |
| c1-natural | 335초 | 480 | 0 | 0 | 107 / 107 | 11,384 / 11,384 | 6,400 / 6,401 | 20 / 40 | 0 | 통과 |

판정: e7a3348b 채택. 현재 브랜치 head ae073b03.

**관찰**: c32-fixed에서 300초 동안 처리량이 첫 round 대비 끝 round에서 약 3% 낮습니다(이 빌드 1,554 → 1,506, 직전 빌드 1,510 → 1,469). 메모리는 평탄합니다. 원인(온도에 따른 GPU clock, 다른 무엇)을 가리기 위해 round별 처리량과 GPU 온도·clock을 남기는 900초 실행을 추가로 돌립니다(44절).

## 44. c32-fixed 900초 추세 — 처리량 하락은 예열 구간의 온도 효과

43절의 관찰을 가리기 위해 round별 처리량·TPOT와 GPU 온도·clock·RSS를 남기는 900초 실행(`qwen-scheduler-stability/c32-fixed-trend-900s/`, 같은 구성·binary). 714 round, 68,544건, 오류 0, 출력 불일치 0, 취소 20 + 부하 중 40, 종료 회수 통과.

실행 시간을 10등분한 구간 평균:

| 구간 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 처리량 tok/s | 1,533 | 1,525 | 1,522 | 1,522 | 1,520 | 1,520 | 1,520 | 1,522 | 1,521 | 1,522 |
| TPOT p50 ms | 14.61 | 14.71 | 14.73 | 14.73 | 14.76 | 14.75 | 14.73 | 14.76 | 14.74 | 14.74 |
| GPU 온도 °C | 50 | 70 | 72 | 73 | 74 | 74 | 73 | 73 | 73 | 72 |
| RSS MiB | 5,414 | 6,408 | 6,409 | 6,409 | 6,409 | 6,409 | 6,409 | 6,410 | 6,410 | 6,410 |

(구간 1의 온도·RSS는 서버 시작과 예열을 포함합니다.) 처리량은 GPU가 50 → 72 °C로 데워지는 첫 구간에서 약 0.7% 내려간 뒤 13분 동안 1,520–1,522로 평탄합니다. GPU clock과 메모리도 평탄합니다. 300초 실행에서 본 "첫 round 대비 끝 round 3%"는 예열 직후의 첫 round가 높게 나온 것이고, 지속적인 하락이나 누수는 아닙니다.

## 45. GEMM: 동시성 불변 조건 아래에서의 전수 탐색

41절에서 남은 차이의 가장 큰 덩어리가 GEMM(약 0.8 ms)이었습니다. Riley는 한 알고리즘을 모든 행 수에 재사용해서 "행의 결과가 batch 크기에 따라 달라지지 않는다"를 보장합니다. 그 제약 안에서 더 좋은 알고리즘이 있는지 cuBLASLt의 알고리즘 id × tile × stages × split-K × reduction 구성 전부를 열거해, M=1·32·256 모두에서 유효한 구성(형상당 1,024–1,489개)을 cold 가중치로 측정했습니다(`qwen-gemm-diagnostics/gemm_search*`). 가중 비용 = 0.67 × M=32 + 0.33 × M=256(41절의 step 구성 비율).

| 형상 | 현재(128행 first heuristic) M=32 / M=256 / 가중 | 탐색 최선 M=32 / M=256 / 가중 | 개선 |
|---|---|---|---:|
| Q+K+V 병합 2048 → 2560 | 17.9 / 27.0 / 20.9 | 16.1 / 25.9 / 19.4 | 1.5 µs |
| O projection 2048 → 2048 | 15.2 / 22.6 / 17.6 | 14.3 / 19.5 / 16.0 | 1.6 µs |
| gate·up 2048 → 11008 | 53.6 / 83.2 / 63.4 | 54.6 / 81.3 / 63.4 | 0 |
| down 11008 → 2048 | 56.2 / 76.3 / 62.8 | 55.6 / 75.5 / 62.2 | 0.6 µs |

모든 구성 중 최선을 골라도 layer당 약 3.7 µs, step당 약 0.13 ms(약 1%)입니다. 즉 **Riley의 GEMM은 동시성 불변 제약 아래에서 이미 최적에 가깝습니다.** vLLM이 M=32에서 더 빠른 것(예: QKV 13.3 µs, O 12.6 µs)은 그 행 수에서만 유효하거나 유리한 알고리즘을 행 수마다 고르기 때문이고, 그 대가로 출력이 batch 구성에 따라 달라집니다(37절: 같은 prompt의 출력이 갈린 경우 43건).

불변성을 포기했을 때 실제로 얼마나 얻는지는 46절의 진단(행 수 bucket마다 자체 heuristic)으로 측정합니다.

## 46. 동시성 불변의 가격 — 행 수별 GEMM 진단 (미채택)

진단 빌드: 행 수 bucket마다 cuBLASLt first heuristic을 따로 고르게 함(128행 anchor 재사용 해제, 그 외 채택 구성과 동일). 정상상태 2회, 비교 기준은 42절 빌드. 증거 `qwen-matrix/per-row-count-gemm-diagnostic/`.

| 조건 | 처리량 tok/s (불변 → 행 수별) | 변화 | TPOT p50 ms |
|---|---:|---:|---:|
| c1-fixed | 107 → 118 | +11.2% | 9.19 → 8.21 |
| c1-natural | 107 → 119 | +11.7% | 9.19 → 8.22 |
| c8-fixed | 686 → 696 | +1.4% | 10.43 → 10.43 |
| c8-natural | 740 → 725 | −2.0% | 10.21 → 10.43 |
| c16-fixed | 1,150 → 1,224 | +6.4% | 12.24 → 11.43 |
| c16-natural | 1,300 → 1,308 | +0.6% | 11.21 → 11.11 |
| c32-fixed | 1,726 → 1,721 | −0.3% | 15.08 → 14.99 |
| c32-natural | 2,072 → 2,076 | +0.2% | 13.28 → 13.23 |

- 오류 0. **출력의 동시성 불변이 깨집니다**: 조건·동시성에 무관하게 출력이 1종인 prompt가 15개 중 7개로 줄었습니다(불변 구성은 15개 전부).
- 이득은 C1(+11%)과 c16-fixed(+6%)에만 있고 C8·C32에서는 ±2% 이내입니다. C1에서 TPOT 8.2 ms가 되어도 vLLM 7.4 ms에는 못 미칩니다.
- 판정: **미채택**. C8 이상의 목표에는 도움이 되지 않고 불변성을 잃습니다.

**vLLM의 projection kernel이 더 빨라 보이던 이유** (`qwen-gemm-diagnostics/gemm_bench8*`, `torch_linear_bench.*`): cuBLAS 12.8과 13.0, `cublasGemmEx`의 compute type·인자 순서·가중치 방향을 바꿔도 M=32의 Q+K+V 병합은 16.3–16.6 µs로 같았고, vLLM 환경에서 `torch.nn.functional.linear`를 직접 재도 16.6 / 14.0 / 102.0 / 52.7 µs(QKV / O / gate+up / down)였습니다. vLLM 프로파일의 13.3 µs는 split-K reduce kernel과 kernel 경계를 뺀 주 kernel만의 시간입니다. 같은 기준으로 맞추면 layer당 GEMM은 vLLM 약 185 µs, Riley 약 197 µs(단독 측정) / 약 204 µs(서버 안)이고, 차이는 128행 anchor 재사용(약 11 µs)과 서버 안에서의 추가 지연(약 7 µs)입니다.

## 47. C32 꼬리 TTFT의 정체

42절 결과의 요청별 원자료로 TTFT 상위 10%를 분석했습니다(같은 harness의 vLLM 37절 원자료와 비교).

| | 하위 50%: TTFT ms / 동시 도착 수 / prompt token | 상위 10%: TTFT ms / 동시 도착 수 / prompt token |
|---|---|---|
| Riley c32-fixed | 31 / 2.8 / 125 | 237 / 32 / 126 |
| vLLM c32-fixed | 43 / 2.9 / 126 | 159 / 32 / 126 |
| Riley c32-natural | 27 / 3.5 / 74 | 275 / 32 / 240 |
| vLLM c32-natural | 52 / 2.5 / 180 | 206 / 32 / 175 |
| Riley c16-fixed | 24 / 1.5 / 126 | 84 / 11.2 / 125 |
| vLLM c16-fixed | 33 / 1.8 / 126 | 82 / 12.6 / 125 |

- TTFT p95/p99를 만드는 것은 두 엔진 모두 **측정 구간 시작 때 32개 요청이 동시에 들어오는 burst**(192건 중 32건 = 16.7%)입니다. 그 밖의 정상 흐름에서는 Riley의 TTFT가 vLLM보다 짧습니다(하위 50%: 31 vs 43, 27 vs 52 ms).
- burst의 prefill은 32 × 126 = 4,032 token입니다. Riley는 step당 256 token이라 16 step에 걸쳐 처리하고, step마다 가중치 전체를 다시 읽는 고정 비용(약 12 ms)을 냅니다. prefill 자체의 연산은 token당 약 41 µs(256행 GEMM이 4090의 BF16 연산 한계 근처)로 약 165 ms이며, vLLM의 159 ms는 사실상 이 연산 시간입니다.
- 예산을 키워 step 수를 줄이는 시도(27·35절)는 꼬리를 10–15% 줄였지만, 128행 anchor 알고리즘이 큰 행 수에서 비효율적이라 처리량·TPOT가 나빠졌습니다. 즉 이 꼬리도 "모든 행 수에 같은 GEMM 알고리즘"이라는 불변 조건의 비용입니다.

판정: 현재 구조에서 burst TTFT를 vLLM 수준으로 내릴 수단은 찾지 못했습니다. 정상 흐름의 TTFT는 이미 vLLM 이하입니다.

## 48. attention workspace를 serving sequence 상한으로 (commit 8b45c5ff)

native D128 partial-state workspace는 행당 page 용량을 checkpoint의 최대 context(32,768 token = 2,048 page)로 잡고 있었습니다: 256행 × 2,048 page × 16 head × 130 × 4 B ≈ 4.4 GiB. scheduler는 접수 때 `--max-sequence-tokens`(벤치마크에서는 256 또는 1,024)를 강제하므로, 서버가 그 상한을 executor에 알리고 workspace를 그 크기로 잡게 했습니다. 상한을 넘는 행은 기존처럼 검증 단계와 kernel에서 실패 처리됩니다.

- HF-exact HTTP gate attempt 23 통과(해시 `5b025fdb…` 동일). binary sha `2c6349c4…`.
- 정상상태 8조건 × 2회(`qwen-matrix/workspace-bound/`), 42절 빌드 대비: 처리량 변화 −0.0~+0.6%, 15개 prompt 출력 전부 동일, 오류 0.
- GPU 최대 사용량(MiB): c1 11,360 → 7,170 / 11,384 → 7,292, c8 11,422 → 7,234 / 11,638 → 7,546, c16 11,496 → 7,308 / 11,928 → 7,836, c32 11,644 → 7,456 / 12,508 → 8,416 (fixed / natural). 약 4.1 GiB 감소. vLLM은 15.1–15.3 GiB.
- 안정성은 49절.

## 49. workspace 상한 빌드 안정성 — 채택 확정

채택 구성 + 8b45c5ff(binary sha `2c6349c4…`), `qwen-scheduler-stability/workspace-bound-build/`.

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-natural | 302초 | 7,488 | 0 | 0 | 1,860 / 1,851 | 8,416 / 8,416 | 6,408 / 6,409 | 20 / 40 | 0 | 통과 |
| c32-fixed | 301초 | 22,944 | 0 | 0 | 1,555 / 1,529 | 7,456 / 7,456 | 6,407 / 6,409 | 20 / 40 | 0 | 통과 |
| c16-natural | 305초 | 4,992 | 0 | 0 | 1,226 / 1,223 | 7,836 / 7,836 | 6,403 / 6,404 | 20 / 40 | 0 | 통과 |
| c8-fixed | 301초 | 9,984 | 0 | 0 | 668 / 665 | 7,234 / 7,234 | 6,401 / 6,402 | 20 / 40 | 0 | 통과 |
| c1-natural | 335초 | 480 | 0 | 0 | 107 / 107 | 7,292 / 7,292 | 6,400 / 6,401 | 20 / 40 | 0 | 통과 |

판정: 8b45c5ff 채택. c32-natural은 sequence가 설정 상한(1,024 token)에 가장 가까이 가는 조건이며 오류·불일치 없이 통과했습니다. 현재 브랜치 `codex/qwen-http-hf-exact` head는 8b45c5ff입니다.

## 50. 최종 같은-실행 비교 — 현재 채택 빌드 vs vLLM 0.27.1

37절 이후 채택된 변경(global merge, submit thread tokenize, workspace 상한)을 모두 포함한 빌드(브랜치 head 8b45c5ff, binary sha `2c6349c4…`)와 vLLM을 같은 실행에서 교대 순서(R→V, V→R)로 측정. 정상상태 조건(warmup 32 / 192건, 2회), host gate 없음, 제외 없음. `plan.json`은 측정 전 기록. 증거 `qwen-matrix/steady-final/`. 32 lane 완료, 오류 0. 값은 Riley / vLLM, 시간 ms.

| 조건 | 처리량 tok/s | 비율 | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 | E2E p50 | E2E p95 | E2E p99 | GPU MiB | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 107 / 130 | 0.82 | 13 / 13 | 14 / 17 | 14 / 20 | 9.2 / 7.4 | 9.2 / 7.4 | 9.3 / 7.4 | 188 / 156 | 298 / 243 | 298 / 246 | 7170 / 15170 | 0.0 / 0.3 |
| c1-natural | 107 / 132 | 0.81 | 13 / 13 | 35 / 25 | 35 / 26 | 9.2 / 7.4 | 9.2 / 7.5 | 9.3 / 7.5 | 592 / 482 | 1209 / 971 | 1211 / 971 | 7292 / 15136 | 0.1 / 0.0 |
| c8-fixed | 687 / 765 | 0.90 | 25 / 31 | 38 / 38 | 66 / 51 | 10.4 / 9.0 | 10.9 / 9.3 | 11.0 / 9.4 | 229 / 202 | 359 / 317 | 368 / 321 | 7234 / 15136 | 0.1 / 0.5 |
| c8-natural | 740 / 839 | 0.88 | 26 / 37 | 55 / 42 | 90 / 63 | 10.2 / 8.9 | 10.2 / 9.0 | 10.3 / 9.1 | 665 / 590 | 1344 / 1175 | 1359 / 1178 | 7546 / 15177 | 0.0 / 0.2 |
| c16-fixed | 1158 / 1287 | 0.90 | 26 / 37 | 68 / 92 | 129 / 92 | 12.1 / 10.3 | 12.5 / 10.9 | 13.3 / 11.1 | 271 / 235 | 412 / 368 | 456 / 407 | 7308 / 15272 | 0.7 / 1.3 |
| c16-natural | 1300 / 1498 | 0.87 | 34 / 41 | 78 / 100 | 191 / 127 | 11.2 / 9.6 | 11.7 / 9.8 | 12.3 / 10.0 | 737 / 645 | 1482 / 1276 | 1598 / 1316 | 7836 / 15280 | 0.1 / 0.2 |
| c32-fixed | 1724 / 1886 | 0.91 | 38 / 53 | 258 / 168 | 289 / 177 | 15.1 / 13.4 | 15.7 / 15.2 | 16.0 / 19.1 | 344 / 324 | 566 / 496 | 716 / 556 | 7456 / 15315 | 0.1 / 0.7 |
| c32-natural | 2072 / 2491 | 0.83 | 41 / 64 | 294 / 207 | 413 / 243 | 13.3 / 10.7 | 15.5 / 11.6 | 16.3 / 13.0 | 870 / 741 | 1859 / 1502 | 2067 / 1557 | 8416 / 15317 | 0.0 / 0.2 |

**판정 (조건별, 최소 기준 = 처리량 ≥ vLLM, TTFT/TPOT ≤ vLLM, P95/P99 회귀 없음)**
- 처리량·TPOT p50/p95: 8개 조건 모두 미달(처리량 0.81–0.91×).
- TTFT p50: 8개 조건 모두 vLLM 이하(C1은 동률).
- TTFT p95: c1-fixed, c8-fixed(동률), c16-fixed, c16-natural은 vLLM 이하. c1-natural, c8-natural, c32 둘은 미달.
- TTFT p99: c1-fixed만 vLLM 이하. TPOT p99는 c32-fixed만 vLLM 이하(16.0 vs 19.1).
- 출력: Riley 15개 prompt 모두 조건·동시성과 무관하게 1종. vLLM은 같은 lane 안에서 같은 prompt의 출력이 갈린 경우 42건.
- GPU 메모리: Riley 7.2–8.4 GiB, vLLM 15.1–15.3 GiB.

**최소 완료 여부: 미달.** Qwen2.5-3B는 8개 핵심 조건 어디에서도 처리량·TPOT 기준을 충족하지 못했습니다. C1은 사용자 결정(2026-10-09)으로 동률 범위로 취급하되 실제 수치는 0.81–0.82×입니다.

**하루 동안의 변화 (Qwen, 정상상태 기준 25절 → 50절)**: c1 0.77 / 0.74 → 0.82 / 0.81, c8 0.83 / 0.80 → 0.90 / 0.88, c16 0.78 / 0.78 → 0.90 / 0.87, c32 0.82 / 0.75 → 0.91 / 0.83 (fixed / natural). GPU 메모리 11.1–12.2 → 7.2–8.4 GiB.

**남은 격차의 원인과 재개 조건** (45–47절의 측정 근거)
- GEMM: 동시성 불변 조건 안에서 최적의 1% 이내. 더 줄이려면 불변성을 포기해야 하고, 그래도 C8·C32에서는 이득이 없습니다(46절).
- attention: page 처리 단계는 메모리 대역폭에 묶여 있습니다(38·40절).
- kernel 경계·step 사이: CUDA graph 실행으로 약 2–4% 추정. Qwen D128용 graph 하위 시스템을 새로 만들어야 합니다(36절). 착수 여부는 사용자 결정 대기.
- C32 꼬리 TTFT: 32개 동시 도착의 prefill 처리. 예산 확대는 처리량을 해칩니다(27·35·47절).

## 51. CUDA graph replay의 실제 이득 측정 (microbenchmark)

36절에서 graph 실행의 이득을 vLLM의 kernel 경계 간격에서 추정(약 0.45 ms)만 했습니다. Qwen 형상의 decode step을 흉내 낸 독립 측정으로 확인했습니다(`qwen-gemm-diagnostics/graph_bench.*`): 36 layer × (cuBLASLt projection 5개 + 작은 kernel 11개) + 최종 norm + LM head + argmax 크기 kernel, 가중치는 layer마다 별도(cold, 약 6 GB). 같은 launch 열을 eager로 실행한 것과 `cudaStreamBeginCapture`로 한 번 capture해 `cudaGraphLaunch`로 재실행한 것을 비교.

| 행 수 | launch / graph node | eager | graph | 절감 | launch당 |
|---:|---:|---:|---:|---:|---:|
| 1 | 579 / 687 | 8,353 µs | 7,902 µs | 437 µs (5.2%) | 0.64 µs |
| 8 | 579 / 687 | 8,449 µs | 8,044 µs | 409 µs (4.8%) | 0.60 µs |
| 32 | 579 / 687 | 8,703 µs | 8,312 µs | 391 µs (4.5%) | 0.57 µs |

- graph replay는 step당 약 0.39–0.44 ms를 줄입니다. Riley의 실제 decode step(C1 9.2 ms, C32 10.8 ms) 대비 약 3.6–4.7%이고, 36절의 추정과 맞습니다.
- cuBLASLt projection은 capture 안에서 문제없이 기록됩니다(node 687 = launch 579 + split-K reduce).
- 참고: 이 합성 step의 eager 시간 8.4–8.7 ms는 "GEMM + 최소한의 작은 kernel"만의 비용입니다. Riley의 실제 C32 decode step 10.8 ms와의 차이 약 2.1 ms가 attention(약 1.2), 실제 작은 kernel 연산, step 전환입니다.

결론: graph 실행의 이득은 실재하고 크기는 약 0.4 ms/step입니다. 얻으려면 D128 경로의 step을 capture 가능한 형태로 만들어야 합니다(다음 절에서 범위 조사).

## 52. pure-decode step의 CUDA graph 기록·재실행 (commit 43ccbcf8) — 진단 단계

사용자 지시의 순서(2번 다음 1번)에 따라 graph 실행에 착수했습니다. 36절의 "새 하위 시스템" 대신 더 작은 방식을 택했습니다: iteration의 command batch 안에서 eager 진입점들을 그대로 한 번 `cudaStreamBeginCapture`로 기록해 두고, 같은 형태의 다음 step은 `cudaGraphLaunch`로 재실행합니다. 코드 조사(읽기 전용 에이전트)로 확인한 근거:
- capture를 막던 검사는 대부분 이 코드의 정책이고 CUDA 제약은 두 곳(embedding의 host 오류 보고 readback + 동기화, batch 종료 동기화)뿐입니다. embedding은 매 step eager로 두고 그 뒤부터 기록합니다.
- step마다 달라지던 launch 인자는 둘입니다: D128 attention의 page 범위(grid X)와 `DeviceBatch.block_count`. 기록 중에는 행당 용량 전체와 넓은 상한을 씁니다(범위 밖 CTA는 즉시 종료하며 40절에서 비용이 없음을 확인, 48절로 용량이 serving 상한으로 줄어 있음).
- step마다 바뀌는 데이터(token, 위치, block table)는 기록 범위 밖의 H2D 업로드로 고정 device buffer에 들어갑니다.

구현: native `riley_cuda_stream_command_batch_step_capture_{begin,end,abort}` / `…_step_graph_launch` / `riley_cuda_step_graph_destroy`, Rust `CudaStepGraph`와 자유 함수 4개, executor의 dense-row shape마다 `(활성 행 수 → 기록된 graph)` 목록, 서버 플래그 `--decode-step-graph capture-v1`(기본 disabled, native D128 + iteration-batch + synchronous transport 요구). 대상은 "모든 활성 행이 decode 한 token이고 출력이 GPU greedy"인 step이며, prefill이 섞인 step은 eager입니다. 재실행 step은 건너뛴 wrapper들의 host 입력 검증(RoPE 위치, gather index, D128 page 범위)을 `validate_replayed_decode_step`으로 다시 수행합니다.

**진단 측정** (진단 디렉터리 빌드, 정상상태 2회, 기준 = 48절 빌드, `qwen-matrix/step-graph-diagnostic/`):

| 조건 | attempt 1: bucket이 꽉 찬 step만 | attempt 2: 활성 행 수별 |
|---|---:|---:|
| c1-fixed / natural | +4.6% / +4.8% | +4.6% / +4.8% |
| c8-fixed / natural | +2.0% / +2.3% | +2.8% / +2.4% |
| c16-fixed / natural | +0.3% / +2.2% | +0.7% / +2.9% |
| c32-fixed / natural | −0.4% / +1.0% | −1.3% / +1.6% |

- 두 시도 모두 15개 prompt 출력이 기준과 동일, 오류 0. C1의 TPOT p50 9.18 → 8.75 ms(51절 microbenchmark의 0.43 ms와 일치).
- attempt 2에서 나빠진 것: c32-fixed 처리량 −1.3%, c16-fixed TPOT p99 12.9 → 14.6 ms. 활성 행 수마다 첫 기록 비용이 한 번씩 들고, c32-fixed의 측정 구간이 2–3초로 짧아 섞인 것으로 추정.
- 300초 지속 부하(attempt 2 빌드, 구간별 처리량): c32-fixed 1,552 → 1,541(기준 1,555 → 1,529), c32-natural 1,899 → 1,895(기준 1,860 → 1,851, +2.2%), c16-fixed 1,104 → 1,100(42절 빌드 1,090 → 1,081, +1.5%), c8-natural 736 → 735(718 → 716, +2.6%). 네 조건 모두 오류 0·출력 불일치 0·종료 회수 통과. GPU 메모리 +약 200 MiB.

해석: 이득은 pure-decode step의 비중에 비례합니다. prefill이 잦은 c32-fixed에서는 거의 0이고, C1에서 가장 큽니다. 정식 검증(native 테스트, exact gate, 같은 binary A/B, 안정성)은 53절.

알려진 한계: 재실행 step은 buffer를 command-batch 사용 장부에 등록하지 않습니다(buffer는 executor 소유라 step 중 해제되지 않지만 기존 fail-closed 보장보다 약함).

## 53. step graph 정식 검증 (commit 43ccbcf8) — 채택

- native GPU 테스트 `step_capture_gpu` 통과: 기록만으로는 실행되지 않음, 입력을 바꾼 3회 재실행이 eager 결과와 byte 동일, 중단한 기록 뒤에도 batch가 정상, command batch 밖의 기록 시도는 거부, 자원 전부 회수.
- ragged D128 native 테스트 통과(기록 모드용 native 변경이 eager 경로를 바꾸지 않음).
- HF-exact HTTP gate attempt 24 통과(해시 `5b025fdb…` 동일). binary sha `3c2bc689…`.
- 같은 binary에서 플래그 끔 / 켬 / vLLM(50절), 정상상태 2회, 시간 ms. 증거 `qwen-matrix/step-graph/`.

| 조건 | 처리량 tok/s | TPOT p50 | TPOT p95 | TPOT p99 | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p99 | GPU MiB | 켬÷vLLM | 반복 편차 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | 107 / 111 / 130 | 9.2 / 8.8 / 7.4 | 9.2 / 8.8 / 7.4 | 9.2 / 8.8 / 7.4 | 13 / 13 / 13 | 14 / 14 / 17 | 14 / 14 / 20 | 189 / 180 / 156 | 298 / 285 / 246 | 7170 / 7176 / 15170 | 0.86 | 0.0 |
| c1-natural | 107 / 112 / 132 | 9.2 / 8.8 / 7.4 | 9.2 / 8.8 / 7.5 | 9.3 / 8.8 / 7.5 | 13 / 13 / 13 | 35 / 35 / 25 | 35 / 35 / 26 | 592 / 565 / 482 | 1210 / 1154 / 971 | 7292 / 7298 / 15136 | 0.85 | 0.1 |
| c8-fixed | 687 / 704 / 765 | 10.4 / 10.2 / 9.0 | 10.9 / 10.6 / 9.3 | 11.1 / 10.8 / 9.4 | 25 / 25 / 31 | 38 / 38 / 38 | 66 / 65 / 51 | 230 / 220 / 202 | 372 / 364 / 321 | 7234 / 7279 / 15136 | 0.92 | 0.4 |
| c8-natural | 740 / 758 / 839 | 10.2 / 10.0 / 8.9 | 10.2 / 10.0 / 9.0 | 10.4 / 10.2 / 9.1 | 24 / 25 / 37 | 49 / 47 / 42 | 77 / 76 / 63 | 665 / 651 / 590 | 1359 / 1326 / 1178 | 7546 / 7594 / 15177 | 0.90 | 0.0 |
| c16-fixed | 1158 / 1167 / 1287 | 12.3 / 12.1 / 10.3 | 12.6 / 12.5 / 10.9 | 13.2 / 13.0 / 11.1 | 26 / 26 / 37 | 60 / 65 / 92 | 129 / 130 / 92 | 267 / 264 / 235 | 479 / 445 / 407 | 7308 / 7391 / 15272 | 0.91 | 0.1 |
| c16-natural | 1301 / 1337 / 1498 | 11.2 / 10.9 / 9.6 | 11.6 / 11.3 / 9.8 | 12.3 / 11.9 / 10.0 | 31 / 34 / 41 | 84 / 100 / 100 | 190 / 194 / 127 | 734 / 714 / 645 | 1597 / 1558 / 1316 | 7836 / 7919 / 15280 | 0.89 | 0.1 |
| c32-fixed | 1728 / 1714 / 1886 | 15.1 / 15.2 / 13.4 | 15.6 / 15.8 / 15.2 | 16.0 / 16.1 / 19.1 | 44 / 44 / 53 | 249 / 252 / 168 | 276 / 279 / 177 | 350 / 353 / 324 | 693 / 697 / 556 | 7456 / 7589 / 15315 | 0.91 | 0.1 |
| c32-natural | 2072 / 2106 / 2491 | 13.3 / 13.0 / 10.7 | 16.7 / 16.4 / 11.6 | 17.3 / 16.7 / 13.0 | 45 / 43 / 64 | 296 / 313 / 207 | 416 / 416 / 243 | 869 / 855 / 741 | 2055 / 2033 / 1557 | 8416 / 8599 / 15317 | 0.85 | 0.2 |

- 처리량: c1 +4.5 / +4.8%, c8 +2.4 / +2.5%, c16 +0.7 / +2.8%, c32 −0.8 / +1.7% (fixed / natural). 오류 0. 15개 prompt 출력이 끈 구성과 전부 동일하고 조건·동시성에 무관하게 1종.
- 나빠진 지표: c32-fixed 처리량 −0.8%·TPOT p50 15.1 → 15.2, c16-natural TTFT p50 31 → 34·p95 84 → 100, c16-fixed TTFT p95 60 → 65, c32-natural TTFT p95 296 → 313. GPU 메모리 +6~183 MiB.
- c32-fixed의 −0.8%는 2–3초짜리 측정 구간에 활성 행 수별 첫 기록 비용이 섞인 것으로 보이며, 아래 300초 지속 부하에서는 채택 빌드(49절 1,555 → 1,529)와 같은 수준(1,544 → 1,539)입니다.

**안정성** (플래그 켬, `qwen-scheduler-stability/step-graph-build/`)

| 조건 | 지속 | 요청 | 오류 | 출력 불일치 | 처리량 첫/끝 tok/s | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | 취소(대기/부하 중) | KV 복귀 | 종료 회수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c32-natural | 302초 | 7,680 | 0 | 0 | 1,879 / 1,896 | 8,606 / 8,614 | 6,573 / 6,582 | 20 / 40 | 0 | 통과 |
| c32-fixed | 301초 | 23,232 | 0 | 0 | 1,544 / 1,539 | 7,652 / 7,654 | 6,578 / 6,581 | 20 / 40 | 0 | 통과 |
| c16-natural | 302초 | 5,088 | 0 | 0 | 1,256 / 1,261 | 7,931 / 7,934 | 6,487 / 6,491 | 20 / 40 | 0 | 통과 |
| c8-fixed | 301초 | 10,272 | 0 | 0 | 678 / 684 | 7,282 / 7,282 | 6,444 / 6,444 | 20 / 40 | 0 | 통과 |
| c1-natural | 320초 | 480 | 0 | 0 | 112 / 112 | 7,298 / 7,298 | 6,407 / 6,407 | 20 / 40 | 0 | 통과 |

- RSS가 C8 이상에서 약 40–175 MiB 늘고(기록된 graph의 host 쪽 구조), 300초 동안 첫¼ 대비 끝¼에서 3–9 MiB 더 증가합니다. 새 활성 행 수가 처음 나올 때마다 graph가 하나씩 추가되기 때문이며, 조합 수는 (dense-row shape × 활성 행 수 ≤ 최대 동시 요청 수)로 유한합니다. 장시간 추세는 아직 측정하지 않았습니다.

**판정**: 채택(opt-in 플래그 `--decode-step-graph capture-v1`를 채택 구성에 추가). 현재 브랜치 head 43ccbcf8. vLLM 대비 0.83–0.92×(c1 0.86 / 0.85, c8 0.92 / 0.90, c16 0.91 / 0.89, c32 0.91 / 0.85 — vLLM 수치는 50절 재사용).

남은 한계: 재실행 step은 buffer를 command-batch 사용 장부에 등록하지 않습니다. prefill이 섞인 step은 여전히 eager입니다.

## 54. step graph 30분 soak (c32-natural, attempt 14, binary 3c2bc689…)

raw: `qwen-scheduler-stability/step-graph-soak-c32-natural-attempt14/` (`result.json`, `host-samples.json`, `native-close.json`, `server.log`)

| 항목 | 값 |
|---|---|
| 지속 시간 / round / 요청 | 1803 s / 476 / 45,696 |
| 오류 / 출력 불일치 | 0 / 0 |
| 처리량 구간 평균(10분위, tok/s) | 1898.8 → 1893.2 → 1892.6 → 1893.0 → 1892.3 → 1892.5 → 1891.5 → 1892.5 → 1891.4 → 1891.2 |
| round 처리량 min / max | 1877.2 / 1907.2 |
| TPOT p50 구간 평균(ms) | 12.80 → 12.86 → 12.87 → 12.85 → 12.85 → 12.86 → 12.87 → 12.87 → 12.86 → 12.87 |
| GPU 메모리 첫¼ / 끝¼ | 8612.9 / 8614.0 MiB |
| host RSS 첫¼ / 끝¼ / 최대 | 6579.5 / 6582.1 / 6582.1 MiB |
| GPU 온도 최대 | 77 °C |
| 취소 (idle / 부하 중) | 20 / 40, 서버 카운터 cancellations 60 |
| round 후 KV 잔여 / 종료 후 회수 | 0 / reclaimed, exit 0 |

- RSS는 초반 graph 기록 구간 이후 30분 동안 +2.6 MiB(첫¼→끝¼)로, 증가분이 구간마다 줄어 수렴합니다(10분위 표본 6580.3 → 6582.1). 300초 실행에서 보인 3–9 MiB 증가는 기록 단계의 것이며 계속 누적되지 않습니다.
- 처리량은 첫 구간 대비 −0.4%이고 온도 상승 시점과 겹칩니다. 2번째 구간 이후로는 ±0.1% 안에서 평탄합니다.

**판정**: 통과. step graph의 host 메모리 추세 우려는 해소.

## 55. 채택 구성(step graph 포함) vs vLLM 같은-실행 정상상태 비교 (final2, head 43ccbcf8)

raw: `qwen-matrix/steady-final2/` (`result.json`, 실행 로그). warmup 32 / retained 192, 2회 반복, 순서 교대, 제외 없음, 오류 0(양쪽 전 조건). 백분위는 2회 반복 384행 합산.

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | TTFT p50/p95/p99 R | TTFT p50/p95/p99 V | TPOT p50/p95/p99 R | TPOT p50/p95/p99 V | E2E p50/p95/p99 R | E2E p50/p95/p99 V | GPU MiB R / V |
|---|---|---|---|---|---|---|---|---|---|---|
| c1-fixed | 111.5 (0.0%) | 129.8 (0.1%) | 0.859 | 12.7/14.1/14.2 | 12.9/17.0/20.4 | 8.75/8.77/8.82 | 7.40/7.41/7.44 | 214/284/285 | 182/243/247 | 7176 / 15170 |
| c1-natural | 112.0 (0.0%) | 132.1 (0.0%) | 0.848 | 12.7/34.8/35.3 | 13.1/24.9/27.3 | 8.77/8.81/8.83 | 7.44/7.45/7.46 | 565/1154/1155 | 482/971/971 | 7298 / 15136 |
| c8-fixed | 705.2 (0.0%) | 758.6 (1.5%) | 0.930 | 24.6/37.5/64.0 | 32.8/38.1/51.9 | 10.12/10.67/10.76 | 9.04/9.41/9.51 | 231/352/357 | 222/318/325 | 7282 / 15136 |
| c8-natural | 758.9 (0.0%) | 837.1 (0.3%) | 0.907 | 22.9/47.1/77.4 | 37.4/42.3/63.3 | 9.96/10.00/10.40 | 8.94/9.04/9.24 | 649/1313/1329 | 592/1178/1188 | 7594 / 15224 |
| c16-fixed | 1165.6 (0.3%) | 1277.9 (0.1%) | 0.912 | 26.2/64.7/128.0 | 37.1/90.9/93.9 | 12.14/12.49/12.79 | 10.33/10.94/11.33 | 266/410/463 | 237/368/406 | 7400 / 15272 |
| c16-natural | 1338.9 (0.2%) | 1497.3 (0.3%) | 0.894 | 34.0/85.4/183.5 | 40.7/107.6/126.5 | 10.89/11.35/12.21 | 9.61/9.74/9.88 | 711/1441/1552 | 644/1274/1323 | 7934 / 15280 |
| c32-fixed | 1704.8 (1.9%) | 1876.7 (1.1%) | 0.908 | 44.8/257.1/288.4 | 53.2/169.6/192.2 | 15.07/15.65/15.99 | 13.44/15.21/19.16 | 358/563/716 | 323/499/563 | 7586 / 15306 |
| c32-natural | 2104.3 (0.5%) | 2488.8 (0.3%) | 0.845 | 43.4/340.5/410.9 | 63.8/181.9/238.6 | 13.05/15.52/16.74 | 10.76/11.66/12.52 | 856/1911/2024 | 744/1534/1551 | 8590 / 15322 |

**판정**: 최소 기준 미달 — 8조건 모두 처리량·TPOT p50에서 vLLM 미만(0.845–0.930×). TTFT p50은 8조건 모두 Riley ≤ vLLM. TTFT p99는 c1-fixed만 Riley 우위, 나머지는 열세. c32-fixed TPOT p99만 Riley 우위(15.99 vs 19.16). GPU 메모리는 vLLM의 47–56%.

50절(step graph 이전 head 8b45c5ff, 같은 방식의 같은-실행 측정)과 비교하면 Riley는 c1 107→111.5 / 107→112.0(+4.2% / +4.7%), c8 687→705 / 740→759(+2.6%), c16 1158→1166 / 1300→1339(+0.7% / +3.0%), c32 1724→1705 / 2072→2104(−1.1% / +1.5%)입니다. 53절에서 본 step graph 이득(Riley 단독 측정)과 같은 크기이며, vLLM 수치는 두 실행에서 ±1% 안으로 같습니다. c32-fixed의 −1.1%는 이 조건의 반복 편차(1.9%) 안입니다.

출력 불변성(같은 실행): Riley는 15개 prompt 모두 전 조건·반복에서 출력 1종(15/15). vLLM은 7/15만 1종이고 나머지는 동시성에 따라 2–25종.

## 56. step graph 구성의 C32-natural step 재프로파일 — 남은 격차 재분해

증거 `qwen-profile-step-graph/step.txt` (host trace `qwen-profile-diagnostic-c32-attempt07`, 비교 대상은 41절의 attempt06). 프로파일 중 처리량 2,080 tok/s. graph로 재실행된 step의 kernel은 CUPTI에 대부분 잡히지 않아(step당 34개만 기록) kernel별 시간은 attempt06 값을 씁니다.

| | graph 없음(attempt06) | graph 있음(attempt07) |
|---|---:|---:|
| pure decode 평균 주기 | 10.78 ms | 10.30 ms |
| prefill 혼합 평균 주기 | 17.10 ms | 16.97 ms |
| 전체 평균 주기 | 12.89 ms | 12.51 ms |
| 혼합 step의 시간 비중 | 0.44 | 0.45 |

- pure decode가 0.48 ms 줄었습니다(51절 microbenchmark의 0.4 ms와 일치). 혼합 step은 eager로 남아 거의 그대로이므로, 혼합 step 비중이 큰 C32에서는 HTTP 이득이 작고(−1.1% / +1.5%) C1·C8에서 큽니다(+2.6–4.7%, 55절).
- vLLM과의 c32-natural 격차(TPOT p50 13.05 vs 10.76 ms, 2.3 ms): pure decode step 10.3 vs 8.8 ms(약 1.5 ms)와, prefill 혼합 step의 초과분(Riley 평균 +2.75 ms/step, vLLM 평균 약 +2 ms/step, 차이 약 0.75 ms)으로 나뉩니다.
- 혼합 step에서 attention producer는 3.17 ms(pure decode 0.99 ms)로, prefill 행 때문에 step당 +2.2 ms입니다. 전체 step 평균으로는 약 0.7 ms. 40절에서 launch 형태를 바꾸는 가설 두 가지는 기각했고, 비용은 (page, head, 행)마다의 warp 연산입니다. 줄이려면 prefill 행 전용의 다른 연산 구조(수치 계약 변경)가 필요합니다.

남은 항목별 상한(c32-natural 기준, 서로 독립이라고 가정): GEMM 행 수별 알고리즘 약 0.84 ms(동시성 불변을 포기해야 함, 46절), prefill attention 약 0.7 ms, 작은 kernel fusion 약 0.4 ms. 셋을 모두 해야 C8 이상에서 동률 근처이고, C1(0.85×)은 GEMM 항목 없이는 줄지 않습니다.

## 57. 작은 kernel fusion 두 가지 (commits 2bad98dc, 6e3187ae) — 정상상태 A/B

56절의 "작은 kernel fusion" 항목. 둘 다 opt-in 플래그이고 기존 kernel과 bit 동일하도록 중간 BF16 반올림을 유지합니다.

- `--mlp-activation fused-silu-multiply-v1`: SiLU와 gate 곱셈을 kernel 하나로(layer당 −1).
- `--query-key-rope fused-split-v1`: 병합 QKV 분리 + bias + query·key RoPE를 kernel 하나로(layer당 −2). 활성 행 뒤의 padding 행은 value만 기록.

정확성: native 테스트 `silu_gated_multiply_gpu`(gate BF16 65,536값 × up 6종, 차이 0), `qkv_split_rope_gpu`(4개 형상, query·key·value 차이 0), 기존 `row_bias_gpu` 3개 통과. HF-exact HTTP gate attempt 25, 26 모두 128 token 일치. raw: `qwen-matrix/kernel-fusion/`.

정상상태 A/B(같은 날, Riley 단독 2회 반복, warmup 32 / retained 192, 오류 0, 제외 없음). 끔 = attempt30(binary 53ccdab4…, 플래그 끔), SiLU만 = attempt31(같은 binary), 둘 다 = attempt32(6e3187ae 빌드). vLLM 수치는 같은 날 오전 55절의 것입니다(같은 실행 아님).

| 조건 | 끔 | SiLU만 | 둘 다 (편차) | 끔 대비 | vLLM(55절) | 비율 |
|---|---:|---:|---:|---:|---:|---:|
| c1-fixed | 111.5 | 112.0 | 113.0 (0.1%) | +1.4% | 129.8 | 0.871 |
| c1-natural | 112.1 | 112.4 | 113.5 (0.0%) | +1.3% | 132.1 | 0.860 |
| c8-fixed | 705.5 | 712.2 | 718.8 (0.1%) | +1.9% | 758.6 | 0.948 |
| c8-natural | 759.2 | 764.3 | 771.2 (0.0%) | +1.6% | 837.1 | 0.921 |
| c16-fixed | 1167.8 | 1183.9 | 1196.2 (0.1%) | +2.4% | 1277.9 | 0.936 |
| c16-natural | 1338.2 | 1349.0 | 1362.0 (0.1%) | +1.8% | 1497.3 | 0.910 |
| c32-fixed | 1706.1 | 1736.3 | 1752.1 (0.6%) | +2.7% | 1876.7 | 0.934 |
| c32-natural | 2110.6 | 2128.1 | 2143.2 (0.2%) | +1.5% | 2488.8 | 0.861 |

둘 다 켠 구성의 지연(ms, p50 / p95 / p99)과 메모리:

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V | 오류 | GPU MiB R / V |
|---|---|---|---|---|---|---|---:|---|
| c1-fixed | 12.3 / 13.7 / 13.9 | 12.9 / 17.0 / 20.4 | 8.64 / 8.65 / 8.68 | 7.40 / 7.41 / 7.44 | 211 / 281 / 281 | 182 / 243 / 247 | 0 | 7176 / 15170 |
| c1-natural | 12.4 / 34.0 / 34.3 | 13.1 / 24.9 / 27.3 | 8.65 / 8.70 / 8.71 | 7.44 / 7.45 / 7.46 | 558 / 1139 / 1139 | 482 / 971 / 971 | 0 | 7298 / 15136 |
| c8-fixed | 24.1 / 38.5 / 64.2 | 32.8 / 38.1 / 51.9 | 9.92 / 10.38 / 10.51 | 9.04 / 9.41 / 9.51 | 232 / 343 / 350 | 222 / 318 / 325 | 0 | 7260 / 15136 |
| c8-natural | 22.8 / 45.9 / 88.1 | 37.4 / 42.3 / 63.3 | 9.80 / 9.84 / 10.17 | 8.94 / 9.04 / 9.24 | 639 / 1292 / 1306 | 592 / 1178 / 1188 | 0 | 7580 / 15224 |
| c16-fixed | 25.5 / 63.2 / 125.6 | 37.1 / 90.9 / 93.9 | 11.83 / 12.26 / 12.54 | 10.33 / 10.94 / 11.33 | 252 / 401 / 465 | 237 / 368 / 406 | 0 | 7374 / 15272 |
| c16-natural | 33.5 / 80.6 / 194.3 | 40.7 / 107.6 / 126.5 | 10.71 / 11.20 / 11.71 | 9.61 / 9.74 / 9.88 | 703 / 1419 / 1525 | 644 / 1274 / 1323 | 0 | 7902 / 15280 |
| c32-fixed | 43.6 / 249.7 / 265.1 | 53.2 / 169.6 / 192.2 | 14.74 / 15.20 / 15.63 | 13.44 / 15.21 / 19.16 | 342 / 524 / 702 | 323 / 499 / 563 | 0 | 7558 / 15306 |
| c32-natural | 45.4 / 325.6 / 406.5 | 63.8 / 181.9 / 238.6 | 12.81 / 14.87 / 16.23 | 10.76 / 11.66 / 12.52 | 841 / 1870 / 1991 | 744 / 1534 / 1551 | 0 | 8550 / 15322 |

출력: 15개 prompt 모두 끔과 동일, 전 조건에서 prompt당 1종.

**판정**: 두 플래그 모두 채택 후보(전 조건 +1.3–2.7%, 회귀 없음, 출력 불변). 안정성 실행 후 채택 확정. 최소 기준은 여전히 미달(0.86–0.95×).

## 58. kernel fusion 빌드 안정성 (attempt 15, head 6e3187ae, 두 fusion 플래그 켬) — 채택 확정

raw: `qwen-scheduler-stability/kernel-fusion-*-attempt15/`, 요약 `kernel-fusion-summary.txt`. 각 조건 300초 지속 부하 + 취소(idle 20, 부하 중 40) + 종료 후 회수.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 7,776 | 0 | 0 | 1921 / 1925 / 1917 | 6550.1 / 6557.4 | 8552.5 / 8558.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 301 s | 16,896 | 0 | 0 | 1119 / 1114 / 1111 | 6477.7 / 6478.7 | 7377.7 / 7378.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 10,464 | 0 | 0 | 697 / 695 / 692 | 6437.7 / 6438.3 | 7267.8 / 7268.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 316 s | 480 | 0 | 0 | 113.6 / 113.5 / 113.5 | 6406.0 / 6406.3 | 7298.0 / 7298.0 | 20 / 40 | 0 | 완료 | 통과 |

c32-natural의 RSS +7.4 MiB는 step graph 기록 구간의 증가로, 54절 30분 soak에서 수렴을 확인한 것과 같은 크기입니다.

**판정**: `--mlp-activation fused-silu-multiply-v1`, `--query-key-rope fused-split-v1` 채택. 채택 구성은 53절 플래그 + 이 둘.

### 58-b. 나머지 네 조건 (같은 빌드·플래그, attempt 15) — 8조건 전체 통과

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-fixed | 301 s | 23,616 | 0 | 0 | 1559 / 1558 / 1547 | 6553.8 / 6557.2 | 7596.3 / 7598.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 303 s | 5,184 | 0 | 0 | 1277 / 1280 / 1275 | 6474.0 / 6478.9 | 7902.5 / 7906.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 307 s | 3,072 | 0 | 0 | 748 / 746 / 745 | 6437.7 / 6438.3 | 7579.9 / 7580.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 306 s | 1,728 | 0 | 0 | 113.1 / 113.0 / 112.9 | 6406.2 / 6406.6 | 7176.0 / 7176.0 | 20 / 40 | 0 | 완료 | 통과 |

## 59. MLP residual 덧셈 + 다음 norm fusion (commit 44a916a9) — 미채택

`--mlp-residual-norm fused-next-norm-v1`: 각 layer의 MLP residual 덧셈을 뒤따르는 norm(다음 layer 입력 norm 또는 final norm)과 함께 기존 fused residual+RMSNorm kernel로 실행(layer당 kernel −1). HF-exact HTTP gate attempt 27 통과, 출력 15/15 불변. raw: `qwen-matrix/kernel-fusion/fuse3-*`, `ab-mlp-residual-norm.txt`.

같은 binary, 플래그 끔(attempt33) / 켬(attempt34), Riley 단독 2회 반복, 오류 0, 제외 없음. vLLM은 55절 값.

| 조건 | 끔 tok/s (편차) | 켬 tok/s (편차) | 변화 | vLLM | 켬 / vLLM |
|---|---:|---:|---:|---:|---:|
| c1-fixed | 113.1 (0.0%) | 112.9 (0.0%) | −0.1% | 129.8 | 0.870 |
| c1-natural | 113.5 (0.1%) | 113.5 (0.0%) | −0.1% | 132.1 | 0.859 |
| c8-fixed | 717.3 (0.1%) | 718.9 (0.1%) | +0.2% | 758.6 | 0.948 |
| c8-natural | 769.9 (0.0%) | 771.3 (0.1%) | +0.2% | 837.1 | 0.921 |
| c16-fixed | 1190.8 (0.4%) | 1197.2 (0.3%) | +0.5% | 1277.9 | 0.937 |
| c16-natural | 1360.7 (0.2%) | 1363.2 (0.0%) | +0.2% | 1497.3 | 0.910 |
| c32-fixed | 1750.2 (0.4%) | 1756.2 (1.3%) | +0.3% | 1876.7 | 0.936 |
| c32-natural | 2142.8 (0.2%) | 2148.4 (0.0%) | +0.3% | 2488.8 | 0.863 |

TPOT p50(ms) 끔 → 켬: c1 8.64 → 8.65, 8.65 → 8.66; c8 9.93 → 9.97, 9.81 → 9.80; c16 11.90 → 11.83, 10.71 → 10.69; c32 14.81 → 14.68, 12.85 → 12.81. TTFT·E2E·메모리는 편차 범위 안에서 같음.

**판정**: 미채택. C1에서 −0.1%(편차 0.0%), C8 이상 +0.2–0.5%로 반복 편차와 구분되지 않습니다. kernel 수는 줄지만 fused residual+norm kernel이 행 단위 reduction 구조라, 원소 단위 덧셈 kernel을 없앤 이득이 상쇄되는 것으로 보입니다(추정). 코드는 기본값 끔으로 남기고 채택 구성에는 넣지 않습니다.

작은 kernel fusion의 합계: 채택된 두 가지로 +1.3–2.7%. 남은 후보(KV 기록 통합)는 같은 크기 이하로 예상되어 보류.

## 60. prefill 행 attention을 batched GEMM으로 — microbenchmark (설계 검증 단계)

56절에서 prefill이 섞인 step의 attention producer가 step당 +2.2 ms였습니다. 40절에서 launch 형태만 바꾸는 방법은 기각했으므로, 연산 구조 자체를 바꾸는 후보를 독립 microbenchmark로 먼저 쟀습니다(`qwen-attention-v2-diagnostics/attn_bench7.*`, `attn_bench8.*`, RTX 4090, cuBLAS 12.8).

구조: (sequence, KV head)마다 Q·K·V를 연속 행렬로 모으고 → scores = KᵀQ (BF16 `cublasGemmStridedBatchedEx`) → causal softmax kernel(기존 계약처럼 dot을 BF16으로 반올림한 뒤 scale) → output = V·P (BF16 batched GEMM) → 원래 배치로 되돌림. 최적화하지 않은 첫 구현입니다.

| prefill 형상 | 모으기 | QK | softmax | PV | 되돌리기 | 합 | 현재 paged producer(attn_bench6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 224행 / 2 sequence, key 130 | 4.8 | 5.5 | 12.6 | 4.3 | 2.8 | 30.1 µs | 36.7 µs |
| 224행 / 4 sequence, key 156 | 5.1 | 5.5 | 12.7 | 4.4 | 2.8 | 30.6 µs | 58.6 µs |
| 128행 / 1 sequence, key 328 | 4.3 | 5.0 | 8.8 | 5.1 | 2.4 | 25.8 µs | 67.7 µs |
| 224행 / 8 sequence, key 32 | 4.8 | 2.9 | 3.9 | 2.9 | 2.8 | 17.2 µs | 14.4 µs |
| 64행 / 2 sequence, key 332 | 4.1 | 4.4 | 5.8 | 5.4 | 1.8 | 21.5 µs | 42.4 µs |
| 256행 / 1 sequence, key 398 | 5.2 | 10.0 | 15.6 | 6.8 | 3.1 | 40.8 µs | (미측정) |

현재 방식 쪽 숫자는 producer만이고 page merge(약 8–9 µs)는 빠져 있습니다. 긴 prompt일수록 이득이 크고(2.0–2.6배), 짧은 prompt 여러 개는 비슷합니다.

**chunk 분할 불변성** (`attn_bench8`, 같은 sequence의 position 200..327): 128행 한 번에 계산한 결과와 64행 두 번, 32행, 1행으로 나눠 계산한 결과가 모든 원소에서 bit 동일(차이 0 / 131,072, 0 / 65,536, 0 / 2,048). key 수와 행 수가 달라져도 행의 결과가 바뀌지 않았습니다. 즉 부하에 따라 prefill chunk가 달라져도 출력이 바뀌지 않을 가능성이 높습니다(이 형상들에 한해 확인, cuBLAS 기본 알고리즘 선택).

추정 이득: c32-natural의 혼합 step에서 layer당 약 34 µs → step당 약 1.2 ms, 전체 step 평균 약 0.4 ms(처리량 약 3%). fixed 계열은 중립. prefill position의 수치 계약이 바뀌므로(현재 warp 순서 reduction → GEMM), 본 코드 반영 시 native 비교 테스트·HF token 비교·안정성을 다시 거쳐야 합니다.

**판정**: 설계 유효, 본 코드 반영 진행 대상. 아직 미구현.

## 61. prefill GEMM attention의 launch 형상 불변성 sweep — 기본 cuBLAS로는 엄밀 불변이 아님

60절의 불변성 확인은 형상 5개뿐이었으므로 본 코드 설계 전에 넓게 쟀습니다(`qwen-attention-v2-diagnostics/attn_bench9.*`). 1024-token sequence 하나에서, position별 기준값(1행·정확한 key 수·batch 2)을 만들고, chunk 시작 6종 × 행 수 6종 × key padding 5종 × 행 padding 3종 × batch 복제 3종으로 같은 position을 다시 계산해 bit 비교했습니다.

- 846개 조합 중 842개는 전 원소 bit 동일. **4개 조합에서 각 1개 원소(4,096개 중)가 다름**: 모두 chunk 500..515, key padding 528이고 batch 32(행 padding 16 / 64 / 256) 또는 행 padding 256 · batch 8.
- 해석: cuBLAS가 전체 크기(행 × batch)에 따라 다른 GEMM 알고리즘을 고르고, 합산 순서 차이가 BF16 반올림을 넘는 드문 경우에만 드러납니다. 함께 처리되는 sequence 수(=부하)에 따라 batch가 달라지므로, 여러 sequence를 padding해 한 번에 계산하는 설계는 **출력의 동시성 불변을 확률적으로 깹니다**(이 sweep에서 원소당 약 6×10⁻⁷, 요청 하나가 수백만 원소를 거치므로 무시할 수 없음).
- 불변을 지키려면 launch 형상을 부하와 무관한 값(prompt 길이)으로만 정해야 합니다: sequence마다 따로(batch 2), 행은 고정 padding. 그러면 sequence당 약 25–40 µs가 들어 natural 계열 이득이 약 3% → 2% 이하로 줄고, 짧은 prompt가 여러 개 들어오는 fixed 계열은 현재(약 15 µs)보다 느려집니다.

**판정**: 기본 cuBLAS batched GEMM 기반 설계는 미채택(본 코드 미반영). 남은 변형은 cuBLASLt로 알고리즘을 고정한 batched GEMM인데(프로젝션 GEMM에서 쓰는 방식), 형상별 plan 관리가 필요하고 이득 상한은 natural 계열 약 3%입니다.

## 62. bit 동일 GEMM 구성의 행 수별 선택 (commits 5ac35c99, d586aff0, b72bc071)

45절의 전수 탐색은 "모든 행 수에 구성 하나"만 찾았습니다. 이번에는 구성마다 출력 bit를 anchor(128행 first heuristic)와 비교해, **결과가 bit 동일한 구성의 집합**을 구했습니다(`qwen-gemm-diagnostics/gemm_equiv.*`, `gemm_equiv2.*`). projection 형상마다 유효 구성 약 2,300개 중 84개가 anchor와 전 원소 bit 동일이고, 그중 최선이 행 수 구간마다 다릅니다.

| projection | anchor M≤32 | bit 동일 최선 M≤32 | M=128 anchor → 최선 | M=160 anchor → 최선 |
|---|---:|---:|---|---|
| Q+K+V 2048→2560 | 17.4 µs | 15.0 µs | 18.5 → 17.4 | 25.3 → 20.5 |
| O 2048→2048 | 14.7 | 12.7 | 15.4 → 14.9 | 21.2 → 18.9 |
| gate·up 2048→11008 | 53.7 | 50.2 | 63.8 → 54.7 | (anchor 유지) |
| down 11008→2048 | 56.1 | 52.1 | 57.0 → 55.5 | 75.7 → 65.9 |

구현: `--gemm-algorithm-selection bit-equivalent-table-v1`(기본 anchor-v1). anchored plan을 준비할 때 표의 구성이 (1) AlgoCheck, (2) 결정적 기록 계약, (3) **그 형상 그대로의 probe GEMM에서 anchor 알고리즘과 출력 bit 동일**을 모두 통과하면 대신 쓰고, 아니면 anchor 알고리즘을 유지합니다. 표는 21개 bucket 측정으로 구간을 정했고, anchor가 더 빠른 구간(gate·up 129행 이상, down 193행 이상)은 항목이 없습니다.

정확성
- native 테스트 `bit_equivalent_selection_preserves_qwen_projection_bytes_for_every_bucket`: 4개 projection × 12개 행 수 × anchor 2종 = 96 비교 모두 차이 0. 128행 anchor에서 43개가 교체됨, 256행 anchor(다른 연산 순서)에서는 probe가 전부 거부해 교체 0. raw `qwen-matrix/gemm-equivalent/qwen-gequiv-native-test2.log`.
- 서버 시작 시 probe: 정식 A/B의 16개 서버 실행에서 1,152건 모두 통과(같음·기록됨).
- HF-exact HTTP gate attempt 29(진단), 30(정식) 128 token 일치.
- 출력: 15개 prompt 모두 끔과 동일, 전 조건 prompt당 1종.

정상상태 A/B(같은 binary, 플래그 끔 attempt37 / 켬 attempt38, Riley 단독 2회, 오류 0, 제외 없음; vLLM은 55절 값, 같은 실행 아님). raw `qwen-matrix/gemm-equivalent/`.

| 조건 | 끔 tok/s (편차) | 켬 tok/s (편차) | 변화 | vLLM | 켬 / vLLM | TPOT 끔 p50/p95/p99 | TPOT 켬 | TPOT vLLM | TTFT 켬 | TTFT vLLM | E2E 켬 | GPU MiB |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| c1-fixed | 113.0 (0.1%) | 121.0 (0.1%) | +7.0% | 129.8 | 0.932 | 8.64 / 8.66 / 8.70 | 8.07 / 8.09 / 8.13 | 7.40 / 7.41 / 7.44 | 11.6 / 13.0 / 13.3 | 12.9 / 17.0 / 20.4 | 197 / 262 / 262 | 7176 |
| c1-natural | 113.5 (0.0%) | 121.5 (0.0%) | +7.0% | 132.1 | 0.920 | 8.65 / 8.69 / 8.70 | 8.08 / 8.12 / 8.14 | 7.44 / 7.45 / 7.46 | 11.6 / 33.1 / 33.5 | 13.1 / 24.9 / 27.3 | 521 / 1065 / 1067 | 7298 |
| c8-fixed | 718.6 (0.0%) | 764.0 (0.1%) | +6.3% | 758.6 | 1.007 | 9.92 / 10.47 / 10.60 | 9.35 / 9.83 / 9.96 | 9.04 / 9.41 / 9.51 | 23.1 / 35.6 / 62.6 | 32.8 / 38.1 / 51.9 | 213 / 324 / 331 | 7268 |
| c8-natural | 771.3 (0.0%) | 816.4 (0.0%) | +5.8% | 837.1 | 0.975 | 9.80 / 9.84 / 10.16 | 9.25 / 9.32 / 9.53 | 8.94 / 9.04 / 9.24 | 21.2 / 44.7 / 86.5 | 37.4 / 42.3 / 63.3 | 603 / 1220 / 1240 | 7580 |
| c16-fixed | 1194.9 (0.1%) | 1250.8 (0.1%) | +4.7% | 1277.9 | 0.979 | 11.88 / 12.19 / 13.23 | 11.27 / 11.64 / 12.70 | 10.33 / 10.94 / 11.33 | 24.3 / 76.7 / 125.3 | 37.1 / 90.9 / 93.9 | 256 / 383 / 446 | 7378 |
| c16-natural | 1360.3 (0.3%) | 1429.2 (0.1%) | +5.1% | 1497.3 | 0.955 | 10.71 / 11.19 / 11.73 | 10.18 / 10.54 / 10.99 | 9.61 / 9.74 / 9.88 | 33.0 / 102.9 / 196.8 | 40.7 / 107.6 / 126.5 | 669 / 1352 / 1455 | 7902 |
| c32-fixed | 1751.5 (0.1%) | 1800.7 (0.1%) | +2.8% | 1876.7 | 0.959 | 14.81 / 15.46 / 15.68 | 14.45 / 15.05 / 15.35 | 13.44 / 15.21 / 19.16 | 41.6 / 246.4 / 275.3 | 53.2 / 169.6 / 192.2 | 335 / 556 / 682 | 7546 |
| c32-natural | 2145.0 (0.2%) | 2221.9 (0.2%) | +3.6% | 2488.8 | 0.893 | 12.82 / 15.64 / 16.57 | 12.41 / 15.27 / 16.50 | 10.76 / 11.66 / 12.52 | 35.9 / 279.8 / 402.0 | 63.8 / 181.9 / 238.6 | 813 / 1820 / 1939 | 8528 |

진단 단계(환경 변수, 거친 구간 표; `qwen-matrix/gemm-equivalent-diagnostic/`)에서는 c16-fixed −0.6%, c8-fixed +1.3%였고, bucket별 표로 바꾼 뒤 각각 +4.7%, +6.3%가 됐습니다. 첫 진단 빌드는 컴파일 오류(예외 처리 비활성)로 실패했고 그때 잘못 시작된 측정은 중단·폐기했습니다(rebuild log 28 보존).

**판정**: 채택 후보(전 조건 +2.8–7.0%, 출력 불변). 안정성 8조건(attempt 16)과 vLLM 같은-실행 비교(final3) 진행 중. 최소 기준은 아직 미달(c8-fixed만 처리량 동률, 나머지 0.89–0.98×).

## 63. bit 동일 GEMM 선택 빌드 안정성 (attempt 16, head b72bc071) — 8조건 통과, 채택

채택 구성 전체 플래그(58절 구성 + `--gemm-algorithm-selection bit-equivalent-table-v1`), binary sha `ab725a7d…`. 각 조건 300초 지속 부하 + 취소 + 종료 후 회수. raw `qwen-scheduler-stability/gemm-equivalent-*-attempt16/`, 요약 `gemm-equivalent-summary.txt`.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,064 | 0 | 0 | 1987 / 1997 / 1987 | 6579.7 / 6585.3 | 8554.1 / 8558.0 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 301 s | 24,288 | 0 | 0 | 1594 / 1611 / 1586 | 6581.2 / 6585.0 | 7595.9 / 7598.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 303 s | 5,472 | 0 | 0 | 1350 / 1348 / 1342 | 6503.0 / 6506.9 | 7903.2 / 7906.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 300 s | 17,664 | 0 | 0 | 1184 / 1174 / 1163 | 6506.1 / 6506.8 | 7378.0 / 7378.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 308 s | 3,264 | 0 | 0 | 792 / 791 / 789 | 6466.2 / 6466.7 | 7580.0 / 7580.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 302 s | 11,136 | 0 | 0 | 740 / 735 / 734 | 6466.1 / 6466.7 | 7267.8 / 7268.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 354 s | 576 | 0 | 0 | 121.5 / 121.4 / 121.4 | 6434.6 / 6434.9 | 7298.0 / 7298.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 302 s | 1,824 | 0 | 0 | 121.1 / 120.9 / 120.9 | 6434.8 / 6435.2 | 7176.0 / 7176.0 | 20 / 40 | 0 | 완료 | 통과 |

host RSS는 이전 빌드보다 약 28 MiB 높습니다(조건별 plan이 더 많은 알고리즘 종류를 쓰는 데 따른 cuBLASLt 쪽 상주분으로 추정, 추세는 평탄: 첫¼→끝¼ 최대 +5.6 MiB).

**판정**: `--gemm-algorithm-selection bit-equivalent-table-v1` 채택.

## 64. 현재 채택 구성 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final3, head b72bc071)

raw `qwen-matrix/steady-final3/`. 측정 전 고정: warmup 32 / retained 192, 2회 반복, 순서 교대(Riley→vLLM, vLLM→Riley), 제외 없음. Riley binary sha `ab725a7d…`, 채택 플래그 = 53절 + fused SiLU multiply + RoPE in split + bit-equivalent GEMM selection. 조건당 384행. 오류 양쪽 0.

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | GPU MiB R / V |
|---|---:|---:|---:|---|
| c1-fixed | 121.0 (0.0%) | 129.2 (0.3%) | 0.937 | 7176 / 15170 |
| c1-natural | 121.5 (0.0%) | 132.0 (0.0%) | 0.920 | 7298 / 15136 |
| c8-fixed | 762.6 (0.1%) | 761.5 (0.2%) | 1.001 | 7268 / 15136 |
| c8-natural | 816.3 (0.2%) | 838.5 (0.2%) | 0.973 | 7580 / 15180 |
| c16-fixed | 1253.4 (0.3%) | 1279.0 (0.4%) | 0.980 | 7378 / 15272 |
| c16-natural | 1431.3 (0.2%) | 1498.8 (0.1%) | 0.955 | 7906 / 15282 |
| c32-fixed | 1789.9 (0.1%) | 1868.9 (0.2%) | 0.958 | 7558 / 15284 |
| c32-natural | 2218.3 (0.2%) | 2493.7 (0.2%) | 0.890 | 8546 / 15288 |

지연(ms, p50 / p95 / p99):

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V |
|---|---|---|---|---|---|---|
| c1-fixed | 11.6 / 13.0 / 13.0 | 13.5 / 19.1 / 21.3 | 8.07 / 8.08 / 8.11 | 7.40 / 7.42 / 7.43 | 197 / 262 / 262 | 183 / 244 / 249 |
| c1-natural | 11.6 / 33.1 / 33.2 | 13.3 / 25.1 / 25.8 | 8.08 / 8.12 / 8.14 | 7.44 / 7.45 / 7.46 | 521 / 1065 / 1065 | 482 / 971 / 971 |
| c8-fixed | 22.8 / 35.3 / 64.9 | 31.0 / 38.4 / 51.7 | 9.35 / 9.89 / 9.96 | 9.01 / 9.38 / 9.64 | 212 / 325 / 332 | 222 / 317 / 325 |
| c8-natural | 24.4 / 44.8 / 81.1 | 37.4 / 42.1 / 62.8 | 9.24 / 9.30 / 9.54 | 8.94 / 8.98 / 9.06 | 603 / 1221 / 1238 | 591 / 1176 / 1178 |
| c16-fixed | 24.3 / 77.0 / 124.9 | 37.3 / 90.0 / 95.6 | 11.26 / 11.70 / 12.14 | 10.33 / 10.94 / 11.29 | 254 / 381 / 447 | 238 / 367 / 407 |
| c16-natural | 32.8 / 76.7 / 193.3 | 40.7 / 106.1 / 133.3 | 10.17 / 10.72 / 11.25 | 9.60 / 9.78 / 9.89 | 665 / 1353 / 1462 | 643 / 1272 / 1322 |
| c32-fixed | 43.0 / 249.3 / 276.9 | 54.0 / 179.9 / 184.1 | 14.48 / 15.04 / 15.55 | 13.56 / 15.12 / 18.08 | 341 / 532 / 679 | 331 / 496 / 566 |
| c32-natural | 36.5 / 282.4 / 402.3 | 62.4 / 223.7 / 237.7 | 12.46 / 15.40 / 15.75 | 10.73 / 11.52 / 12.23 | 815 / 1801 / 1935 | 740 / 1535 / 1554 |

출력 불변성(같은 실행): Riley 15/15 prompt가 전 조건·반복에서 1종, vLLM 7/15.

**판정**: 최소 기준 미달. 처리량은 c8-fixed만 vLLM 이상(1.001), 나머지 0.890–0.980. TPOT p50은 8조건 모두 vLLM보다 높음(차이 0.30–1.73 ms). TTFT p50은 8조건 모두 Riley 우위. TTFT p95는 c1-fixed·c8-fixed·c16 두 조건에서 우위, p99는 c1-fixed만 우위. c32-fixed TPOT p95/p99는 Riley 우위. 55절(오전, 0.845–0.930) 대비 전 조건 +5–8%p 개선.

남은 격차(TPOT p50): c1 0.64–0.67 ms, c8 0.30–0.34, c16 0.57–0.93, c32 0.92–1.73.

## 65. C1 재프로파일 → wide-block RMSNorm, 1024-thread argmax (commit 35c7d268)

62절 이후 GEMM은 bit 동일 집합 안의 최선이 제약 없는 최선과 1 µs 이내라 더 줄일 곳이 없습니다. C1-natural을 graph 없이 프로파일(host trace `qwen-profile-diagnostic-c1-attempt08`)해 GEMM 외 항목을 쟀습니다: pure decode 8.62 ms 중 GEMM 6.98(LM head 0.70 포함), attention producer 0.276 + merge 0.098, RMSNorm 0.167 + fused residual RMSNorm 0.166(73회, 회당 4.5 µs), split-K reduce 0.126, argmax 0.105, KV write 0.077, split+RoPE 0.058, SiLU·곱 0.044, residual add 0.041, GPU 유휴 0.47.

RMSNorm 진단(`qwen-norm-diagnostics/norm_bench*.{cu,log}`, GPU 쪽 시간은 nsys로 측정):
- barrier 9회의 비용은 0.25 µs뿐이고, dynamic shared memory·64-bit index도 원인이 아님.
- register에 값을 담아 두 번째 load를 없애는 변형, 마지막 tree 단계를 warp shuffle로 바꾸는 변형: 3.74 → 3.84–3.90 µs로 **이득 없음(미채택)**.
- 원인은 thread당 순차 global load 8회. 1024 thread가 2개씩 읽고 shared memory에 둔 뒤 앞 256 thread가 원래의 누적·tree를 수행하는 **wide-block 변형**: 3.74 → 2.27 µs(1·32행), 256행 4.16 → 4.06, 전 원소 bit 동일.

반영: BF16·hidden 2048·256행 이하의 `rms_norm`·`residual_rms_norm`은 wide-block kernel 사용(그 외는 기존 kernel). `bf16_argmax`는 block당 1024 thread(선택 규칙이 열 분할과 무관). 플래그 없음(출력 bit 불변).

정확성: native 테스트 `wide_bf16_rms_norm_kernels_match_the_256_thread_kernels`(300행을 기존 경로 1회 vs wide 경로 2회로 계산, norm·residual·fused 모두 차이 0), 기존 norm 테스트·argmax 테스트 4개 통과, HF-exact HTTP gate attempt 31 통과. raw `qwen-matrix/wide-norm-argmax/`.

정상상태(Riley 단독 2회, 채택 플래그 동일). 기준 = 직전 채택 binary의 attempt38(같은 날, 다른 binary — 같은-실행 A/B 아님; 직전 binary는 host `qwen-riley-ab725a7d-adopted`로 보존). vLLM은 64절 값.

| 조건 | 직전 tok/s | 새 tok/s | 변화 | TPOT p50 ms | vLLM tok/s (64절) | 새 / vLLM |
|---|---:|---:|---:|---|---:|---:|
| c1-fixed | 121 | 124 | +2.8% | 8.07 → 7.84 | 129.2 | 0.96 |
| c1-natural | 121 | 125 | +2.9% | 8.08 → 7.85 | 132.0 | 0.95 |
| c8-fixed | 764 | 780 | +2.1% | 9.36 → 9.13 | 761.5 | 1.02 |
| c8-natural | 816 | 836 | +2.4% | 9.25 → 9.04 | 838.5 | 1.00 |
| c16-fixed | 1251 | 1275 | +1.9% | 11.26 → 11.11 | 1279.0 | 1.00 |
| c16-natural | 1429 | 1462 | +2.3% | 10.18 → 9.97 | 1498.8 | 0.98 |
| c32-fixed | 1801 | 1806 | +0.3% | 14.46 → 14.29 | 1868.9 | 0.97 |
| c32-natural | 2222 | 2260 | +1.7% | 12.41 → 12.22 | 2493.7 | 0.91 |

출력 15/15 불변, 오류 0. LM head(2048→151936)의 bit 동일 집합도 탐색(`qwen-gemm-diagnostics/gemm_equiv3_2048_151936.log`): 8개 구성, M≤32에서 anchor 695–724 → 660–688 µs(step당 약 0.035 ms). 아직 표에 미반영.

**판정**: 채택 후보(전 조건 개선, 출력 불변). 안정성·vLLM 같은-실행 비교는 다음 batch 끝에 함께 수행.

## 66. shared-KV producer의 token당 연산 수 축소 + LM head 표 추가 (commit de9443e7)

56절·65절 프로파일에서 attention producer는 prefill 혼합 step에서 3.2 ms, C1에서 0.28 ms였습니다. 연산 구조(수치)는 그대로 두고 warp 명령 수만 줄이는 변형을 독립 측정했습니다(`qwen-attention-v2-diagnostics/attn_bench10.*`).

- float tile: K/V tile을 shared memory에 FP32로 올림(변환은 정확). 단독으로는 0–4%.
- butterfly: 점수 합산을 xor butterfly로(각 단계에서 shfl_down tree의 lane 0과 같은 두 부분합을 더하므로 모든 lane이 lane 0과 같은 bit) → 점수·alpha·beta broadcast 3회 제거, numerator 분기를 token당 1회로.

| 형상 | 기존 | float tile | + butterfly | 상태값 차이 |
|---|---:|---:|---:|---:|
| 1행 × 300 token | 7.6 µs | 7.3 | 6.1 | 0 |
| 8행 × 300 | 10.3 | 9.9 | 8.6 | 0 |
| 32행 × 400 | 32.4 | 31.6 | 26.9 | 0 |
| 32행 × 150 | 15.1 | 14.5 | 12.6 | 0 |
| 256행 (32×400 + 224×130) | 98.0 | 96.0 | 80.5 | 0 |
| 160행 (32×400 + 128×200) | 84.9 | 85.4 | 72.5 | 0 |
| 96행 × 330 | 70.2 | 70.0 | 58.7 | 0 |
| 256행 × 130 | 73.6 | 73.5 | 62.4 | 0 |

(입력에 ±inf를 섞어 online update의 비유한 분기도 포함.)

반영: shared-KV producer kernel을 butterfly 형태로 교체(플래그 없음, 출력 bit 불변). LM head(2048→151936)를 bit 동일 GEMM 표에 추가(서버 probe 모두 same=1).

정확성: native `native_bf16_paged_split_gqa_gpu` 3개(reference·CPU 비교, V1 prefix reducer와 행별 비교 포함), `step_capture_gpu`, GEMM 동일성 테스트 통과. HF-exact HTTP gate attempt 32 통과. raw `qwen-matrix/lean-producer/`.

정상상태(Riley 단독 2회). 기준 = 65절 빌드의 attempt40(같은 날, 직전 binary; host `qwen-riley-35c7d268-wide` 보존). vLLM은 64절 값.

| 조건 | 직전 tok/s | 새 tok/s | 변화 | TPOT p50 ms | vLLM tok/s (64절) | 새 / vLLM |
|---|---:|---:|---:|---|---:|---:|
| c1-fixed | 124 | 126 | +1.4% | 7.84 → 7.74 | 129.2 | 0.98 |
| c1-natural | 125 | 127 | +1.4% | 7.85 → 7.74 | 132.0 | 0.96 |
| c8-fixed | 780 | 791 | +1.4% | 9.13 → 9.02 | 761.5 | 1.04 |
| c8-natural | 836 | 859 | +2.7% | 9.04 → 8.79 | 838.5 | 1.02 |
| c16-fixed | 1275 | 1297 | +1.7% | 11.11 → 10.91 | 1279.0 | 1.01 |
| c16-natural | 1462 | 1502 | +2.7% | 9.97 → 9.69 | 1498.8 | 1.00 |
| c32-fixed | 1806 | 1842 | +2.0% | 14.29 → 14.17 | 1868.9 | 0.99 |
| c32-natural | 2260 | 2326 | +2.9% | 12.22 → 11.84 | 2493.7 | 0.93 |

출력 15/15 불변, 오류 0. 안정성(attempt 17)과 vLLM 같은-실행 비교(final4)는 이 빌드로 진행 중.

## 67. 현재 빌드(head de9443e7) 안정성 — 8조건 통과 (attempt 17)

65절(wide RMSNorm·argmax)과 66절(producer·LM head)을 포함한 빌드, 채택 플래그 전체. 각 조건 300초 지속 부하 + 취소 + 종료 후 회수. raw `qwen-scheduler-stability/lean-producer-*-attempt17/`, 요약 `lean-producer-summary.txt`.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,448 | 0 | 0 | 2077 / 2097 / 2077 | 6580.6 / 6587.5 | 8552.9 / 8558.0 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 301 s | 25,056 | 0 | 0 | 1660 / 1665 / 1645 | 6584.2 / 6587.2 | 7596.6 / 7598.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 305 s | 5,760 | 0 | 0 | 1412 / 1409 / 1408 | 6506.3 / 6509.3 | 7904.1 / 7906.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 302 s | 18,336 | 0 | 0 | 1217 / 1212 / 1204 | 6508.1 / 6509.2 | 7377.7 / 7378.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 302 s | 3,360 | 0 | 0 | 833 / 833 / 830 | 6467.9 / 6468.9 | 7579.5 / 7580.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 11,520 | 0 | 0 | 766 / 764 / 761 | 6468.6 / 6469.2 | 7267.9 / 7268.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 340 s | 576 | 0 | 0 | 126.8 / 126.6 / 126.6 | 6437.0 / 6437.3 | 7298.0 / 7298.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 305 s | 1,920 | 0 | 0 | 126.2 / 126.0 / 126.0 | 6436.5 / 6436.8 | 7176.0 / 7176.0 | 20 / 40 | 0 | 완료 | 통과 |

**판정**: 65절·66절 변경 채택. 메모리는 63절과 같은 수준(GPU 동일, RSS 첫¼→끝¼ 최대 +6.9 MiB).

## 68. 현재 채택 구성 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final4, head de9443e7)

raw `qwen-matrix/steady-final4/`. 측정 전 고정: warmup 32 / retained 192, 2회 반복, 순서 교대, 제외 없음. Riley = 64절 플래그 그대로 + 65·66절 변경(플래그 없음). 조건당 384행, 오류 양쪽 0.

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | GPU peak MiB R / V |
|---|---:|---:|---:|---|
| c1-fixed | 126.1 (0.0%) | 129.8 (0.3%) | 0.972 | 7722 / 15170 |
| c1-natural | 126.7 (0.0%) | 132.1 (0.0%) | 0.959 | 7740 / 15136 |
| c8-fixed | 791.9 (0.0%) | 766.1 (1.1%) | 1.034 | 7730 / 15136 |
| c8-natural | 858.9 (0.1%) | 837.8 (0.1%) | 1.025 | 7736 / 15180 |
| c16-fixed | 1292.8 (0.3%) | 1279.8 (0.0%) | 1.010 | 7710 / 15272 |
| c16-natural | 1503.6 (0.2%) | 1496.2 (0.0%) | 1.005 | 7896 / 15280 |
| c32-fixed | 1847.0 (0.9%) | 1878.1 (0.4%) | 0.983 | 7740 / 15330 |
| c32-natural | 2324.4 (0.2%) | 2488.7 (0.2%) | 0.934 | 8528 / 15288 |

지연(ms, p50 / p95 / p99):

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V |
|---|---|---|---|---|---|---|
| c1-fixed | 11.1 / 12.6 / 12.7 | 13.0 / 17.0 / 20.1 | 7.74 / 7.76 / 7.79 | 7.40 / 7.41 / 7.42 | 189 / 251 / 252 | 183 / 243 / 245 |
| c1-natural | 11.2 / 31.5 / 31.9 | 13.1 / 25.0 / 25.7 | 7.75 / 7.79 / 7.81 | 7.44 / 7.45 / 7.46 | 499 / 1021 / 1023 | 482 / 971 / 971 |
| c8-fixed | 22.3 / 34.7 / 60.8 | 29.5 / 41.0 / 52.0 | 9.00 / 9.47 / 9.57 | 8.97 / 9.46 / 9.74 | 207 / 312 / 320 | 221 / 315 / 327 |
| c8-natural | 20.6 / 43.1 / 72.4 | 37.5 / 42.1 / 64.2 | 8.79 / 8.83 / 9.02 | 8.94 / 9.01 / 9.06 | 573 / 1160 / 1175 | 591 / 1177 / 1179 |
| c16-fixed | 23.6 / 59.2 / 122.2 | 36.7 / 90.6 / 95.6 | 10.99 / 11.43 / 11.95 | 10.31 / 10.91 / 11.31 | 241 / 369 / 420 | 239 / 367 / 405 |
| c16-natural | 31.1 / 75.6 / 184.2 | 40.9 / 96.1 / 126.1 | 9.69 / 10.20 / 10.84 | 9.61 / 9.82 / 9.88 | 632 / 1288 / 1390 | 645 / 1276 / 1322 |
| c32-fixed | 41.2 / 241.2 / 269.1 | 53.1 / 164.0 / 175.7 | 14.05 / 14.95 / 15.23 | 13.54 / 15.17 / 19.55 | 324 / 531 / 665 | 321 / 498 / 561 |
| c32-natural | 42.0 / 295.8 / 383.1 | 63.2 / 211.8 / 248.0 | 11.86 / 13.94 / 15.64 | 10.77 / 11.72 / 12.95 | 777 / 1740 / 1851 | 741 / 1539 / 1556 |

출력 불변성(같은 실행): Riley 15/15, vLLM 7/15.

GPU peak 주의: C1·C8·c16-fixed의 Riley peak가 64절(7176–7378)보다 약 350–550 MiB 높게 찍혔습니다. 67절 안정성 실행의 정상상태 값은 그대로(c1 7176, c8-fixed 7268, c16-fixed 7378)이므로, 이번에 LM head를 bit 동일 표에 넣으면서 **서버 시작 시 probe가 LM head 크기의 임시 buffer(약 620 MiB)를 잠깐 잡는 것**이 peak에 잡힌 것으로 판단합니다(직접 분리 측정은 안 함). 정상상태 사용량은 변하지 않았습니다.

**판정**: 최소 기준 미달이지만 절반이 넘어감.
- 처리량 ≥ vLLM: **c8-fixed, c8-natural, c16-fixed, c16-natural 충족**(1.005–1.034). c1-fixed 0.972, c1-natural 0.959, c32-fixed 0.983, c32-natural 0.934 미충족.
- TPOT p50 ≤ vLLM: c8-natural만 충족(8.79 vs 8.94). c8-fixed 9.00 vs 8.97, c16-natural 9.69 vs 9.61로 근접. 나머지 차이 0.31–1.09 ms.
- TTFT p50: 8조건 모두 충족. TTFT p95: c1-fixed·c8-fixed·c16 두 조건 충족. TTFT p99: c1-fixed만.
- TPOT p95/p99: c8-natural 충족, c32-fixed p95/p99 충족, c8-fixed p99 충족.
- E2E p50: c8 두 조건, c16-natural 충족.

남은 격차(TPOT p50): c1 0.31–0.34 ms, c16-fixed 0.68, c32-fixed 0.51, c32-natural 1.09.

## 69. MLP residual + 다음 norm fusion 재측정 (현재 빌드 de9443e7)

59절에서 −0.1 ~ +0.5%로 미채택했던 `--mlp-residual-norm fused-next-norm-v1`을, fused residual+RMSNorm kernel이 wide-block으로 빨라진(65절) 현재 빌드에서 다시 쟀습니다. 같은 binary, 플래그 끔(attempt41) / 켬(attempt42), Riley 단독 2회, 오류 0, 제외 없음. vLLM은 68절 값. raw `qwen-matrix/mlp-residual-norm-remeasure/`.

| 조건 | 끔 tok/s (편차) | 켬 tok/s (편차) | 변화 | vLLM | 켬 / vLLM | TPOT p50 끔 → 켬 (vLLM) |
|---|---:|---:|---:|---:|---:|---|
| c1-fixed | 126.1 (0.0%) | 126.6 (0.0%) | +0.4% | 129.8 | 0.976 | 7.74 → 7.71 (7.40) |
| c1-natural | 126.7 (0.0%) | 127.2 (0.0%) | +0.4% | 132.1 | 0.963 | 7.74 → 7.72 (7.44) |
| c8-fixed | 790.8 (0.2%) | 795.4 (0.1%) | +0.6% | 766.1 | 1.038 | 9.04 → 8.97 (8.97) |
| c8-natural | 858.8 (0.2%) | 862.4 (0.1%) | +0.4% | 837.8 | 1.029 | 8.79 → 8.75 (8.94) |
| c16-fixed | 1296.6 (0.3%) | 1301.1 (0.3%) | +0.3% | 1279.8 | 1.017 | 10.90 → 10.90 (10.31) |
| c16-natural | 1501.9 (0.1%) | 1507.1 (0.2%) | +0.3% | 1496.2 | 1.007 | 9.69 → 9.66 (9.61) |
| c32-fixed | 1842.5 (0.7%) | 1847.7 (0.1%) | +0.3% | 1878.1 | 0.984 | 14.18 → 13.99 (13.54) |
| c32-natural | 2325.5 (0.1%) | 2340.0 (0.1%) | +0.6% | 2488.7 | 0.940 | 11.84 → 11.79 (10.77) |

출력 15/15 불변. 이번에는 8조건 모두 양(+0.3–0.6%)이고 편차 0.0%인 C1에서도 +0.4%입니다.

**판정**: 채택 후보로 전환(59절의 미채택은 당시 kernel 기준). 다음 batch의 안정성 실행에 포함.

## 70. 채택 구성(MLP residual+norm fusion 포함, LM head 표 제외; head 1f24df2b) 안정성 — attempt 18

69절의 fusion 플래그를 켜고 68절에서 지적한 LM head probe를 뺀 빌드. HF-exact HTTP gate attempt 33 통과. 각 조건 300초 지속 부하 + 취소 + 종료 후 회수. raw `qwen-scheduler-stability/adopt5-*-attempt18/`.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,448 | 0 | 0 | 2081 / 2095 / 2081 | 6571.2 / 6577.4 | 8545.7 / 8550.0 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 300 s | 25,056 | 0 | 0 | 1654 / 1679 / 1643 | 6573.8 / 6576.9 | 7588.4 / 7590.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 304 s | 5,760 | 0 | 0 | 1411 / 1412 / 1408 | 6499.8 / 6503.1 | 7899.7 / 7902.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 301 s | 18,336 | 0 | 0 | 1218 / 1218 / 1203 | 6501.7 / 6502.8 | 7373.6 / 7374.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 302 s | 3,360 | 0 | 0 | 834 / 832 / 830 | 6464.3 / 6464.8 | 7578.0 / 7578.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 11,520 | 0 | 0 | 767 / 763 / 763 | 6464.2 / 6464.8 | 7265.9 / 7266.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 340 s | 576 | 0 | 0 | 126.6 / 126.5 / 126.5 | 6434.4 / 6434.7 | 7296.0 / 7296.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 305 s | 1,920 | 0 | 0 | 126.1 / 125.9 / 125.7 | 6434.7 / 6435.1 | 7174.0 / 7174.0 | 20 / 40 | 0 | 완료 | 통과 |

**판정**: `--mlp-residual-norm fused-next-norm-v1` 채택, LM head 표 제외 확정. 채택 플래그 = 64절 + `--mlp-residual-norm fused-next-norm-v1`.

## 71. 현재 채택 구성 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final5, head 1f24df2b)

raw `qwen-matrix/steady-final5/`. warmup 32 / retained 192, 2회 반복, 순서 교대, 제외 없음, 오류 양쪽 0, 조건당 384행.

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | GPU peak MiB R / V |
|---|---:|---:|---:|---|
| c1-fixed | 126.0 (0.1%) | 129.7 (0.0%) | 0.972 | 7174 / 15170 |
| c1-natural | 126.6 (0.0%) | 132.1 (0.1%) | 0.958 | 7296 / 15136 |
| c8-fixed | 793.2 (0.1%) | 762.7 (0.1%) | 1.040 | 7266 / 15136 |
| c8-natural | 858.6 (0.2%) | 835.2 (0.2%) | 1.028 | 7578 / 15224 |
| c16-fixed | 1298.0 (0.5%) | 1268.5 (3.4%) | 1.023 | 7374 / 15272 |
| c16-natural | 1503.6 (0.2%) | 1496.6 (0.3%) | 1.005 | 7892 / 15280 |
| c32-fixed | 1853.9 (0.2%) | 1877.1 (1.0%) | 0.988 | 7580 / 15306 |
| c32-natural | 2333.4 (0.4%) | 2489.1 (0.1%) | 0.937 | 8550 / 15354 |

지연(ms, p50 / p95 / p99):

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V |
|---|---|---|---|---|---|---|
| c1-fixed | 11.1 / 12.5 / 12.8 | 13.1 / 16.7 / 20.4 | 7.75 / 7.77 / 7.79 | 7.40 / 7.42 / 7.43 | 189 / 252 / 252 | 183 / 243 / 244 |
| c1-natural | 11.1 / 31.4 / 31.8 | 13.1 / 24.8 / 25.8 | 7.76 / 7.80 / 7.81 | 7.44 / 7.45 / 7.45 | 500 / 1022 / 1024 | 482 / 970 / 971 |
| c8-fixed | 22.2 / 35.9 / 60.9 | 30.9 / 38.0 / 51.7 | 8.98 / 9.47 / 9.54 | 9.00 / 9.36 / 9.44 | 205 / 312 / 319 | 221 / 317 / 319 |
| c8-natural | 20.6 / 48.0 / 65.9 | 37.5 / 42.7 / 62.8 | 8.79 / 8.83 / 8.97 | 8.94 / 9.04 / 9.59 | 573 / 1161 / 1171 | 592 / 1178 / 1189 |
| c16-fixed | 23.5 / 76.1 / 121.5 | 36.6 / 91.0 / 96.0 | 10.89 / 11.27 / 11.83 | 10.29 / 11.00 / 12.12 | 249 / 368 / 425 | 240 / 368 / 407 |
| c16-natural | 30.9 / 66.8 / 189.1 | 40.5 / 98.1 / 122.5 | 9.69 / 10.17 / 11.05 | 9.62 / 9.77 / 9.90 | 632 / 1286 / 1399 | 645 / 1273 / 1321 |
| c32-fixed | 41.3 / 240.6 / 270.1 | 53.7 / 166.5 / 180.0 | 14.00 / 14.56 / 15.02 | 13.50 / 15.06 / 18.96 | 330 / 510 / 662 | 320 / 500 / 559 |
| c32-natural | 37.5 / 275.6 / 385.8 | 63.4 / 222.7 / 241.2 | 11.80 / 14.59 / 15.21 | 10.77 / 11.49 / 12.97 | 774 / 1731 / 1860 | 741 / 1541 / 1558 |

출력 불변성(같은 실행): Riley 15/15, vLLM 6/15. GPU peak는 64절 수준으로 복귀(c1 7174, c8-fixed 7266) — 68절의 높은 peak가 LM head probe였음을 확인.

**판정**: 최소 기준 미달(68절과 같은 구도).
- 처리량 ≥ vLLM: c8-fixed 1.040, c8-natural 1.028, c16-fixed 1.023, c16-natural 1.005 충족. c1 0.972 / 0.958, c32-fixed 0.988, c32-natural 0.937 미충족. (c16-fixed의 vLLM 반복 편차가 3.4%로 큼.)
- TPOT p50 ≤ vLLM: c8-fixed(8.98 vs 9.00), c8-natural(8.79 vs 8.94) 충족. c16-natural +0.07 ms, c1 +0.32–0.35, c32-fixed +0.50, c16-fixed +0.60, c32-natural +1.03.
- TTFT p50: 8조건 모두 충족. TPOT p95/p99: c8-natural, c32-fixed 충족, c16-fixed p99 충족.

## 72. 닫힌 조사 두 가지 (미채택, 본 코드 미반영)

**step당 token 예산 확대의 재검토** (`qwen-gemm-diagnostics/gemm_equiv4.*`): bit 동일 선택이 생겼으므로 큰 행 수에서의 GEMM을 다시 쟀습니다. bit 동일 집합의 최선 기준 layer당 합: M=32 182 µs, M=256 295, M=512 543, M=1024 1058. prefill 행 하나의 추가 비용은 256행에서 0.50 µs, 512행 0.75, 1024행 0.88로 **행 수가 커질수록 비싸집니다**. 예산을 키우면 step 수는 줄지만 총 prefill 비용이 늘어 처리량이 나빠지므로 기각. (C32 burst TTFT 꼬리는 4,032 token의 연산 시간 자체가 약 150–170 ms라 vLLM의 164–180 ms p95와 같은 수준이 한계.)

**prefill 행의 tile 공유 재측정** (`qwen-attention-v2-diagnostics/attn_bench11.*`): 66절의 butterfly producer로 같은 sequence의 2행·4행이 tile 하나를 공유하도록 묶음. 상태값은 모두 bit 동일하지만 시간은 per-row CTA 35.3 / 56.2 / 65.0 / 118.1 µs 대비 2행 36.4 / 57.1 / 65.4 / 126.9, 4행 43.0 / 68.3 / 79.9 / 143.8로 **빨라지지 않음**. prefill producer는 대역폭이 아니라 warp 연산량에 묶여 있습니다. 기각(40절 결론 재확인).

## 73. 혼합 step의 LM head를 출력 행에만 (commit 5d1a1c1b)

c32-natural 재프로파일(host trace `qwen-profile-diagnostic-c32-attempt09`, 현재 빌드·graph 없음): pure decode 9.48 ms(GEMM 7.53, producer 0.79, merge 0.13, norm 0.18, 유휴 0.57), prefill 혼합 15.79 ms(GEMM 약 11.0, producer 2.75, merge 0.36, norm 0.34, SiLU·곱 0.34, 유휴 0.65), 혼합 step이 시간의 42%.

발견: LM head(2048→151936)가 dense 행 전체에 대해 계산됩니다. 혼합 step에서는 256행 중 출력 행이 32개 안팎인데 LM head kernel이 1.02 ms(pure decode 0.69 ms)이고 78 MB logits buffer를 채웁니다.

변경: `--lm-head-rows output-rows-v1`(기본 dense-v1). 출력 행 수가 dense 행 수보다 적은 step에서, 정규화된 출력 행을 먼저 gather(`hidden_rotary`를 scratch로 사용)하고, 그 행 수를 담는 가장 작은 준비된 bucket의 anchored LM head plan으로 계산해 gathered-logits buffer에 직접 씁니다. anchored plan은 행 수와 무관하게 행 결과가 같으므로 logits bit 불변. pure decode step(graph 재실행 포함)은 그대로.

같은 binary, 플래그 끔(attempt43) / 켬(attempt44), Riley 단독 2회, 오류 0, 제외 없음. HF-exact HTTP gate attempt 34 통과. vLLM은 71절 값. raw `qwen-matrix/lm-head-output-rows/`.

| 조건 | 끔 tok/s (편차) | 켬 tok/s (편차) | 변화 | vLLM (71절) | 켬 / vLLM |
|---|---:|---:|---:|---:|---:|
| c1-fixed | 126.0 (0.0%) | 126.1 (0.0%) | +0.0% | 129.7 | 0.972 |
| c1-natural | 126.6 (0.0%) | 126.6 (0.0%) | -0.0% | 132.1 | 0.958 |
| c8-fixed | 793.2 (0.1%) | 798.7 (0.2%) | +0.7% | 762.7 | 1.047 |
| c8-natural | 859.2 (0.0%) | 860.7 (0.0%) | +0.2% | 835.2 | 1.031 |
| c16-fixed | 1298.3 (0.4%) | 1321.4 (0.0%) | +1.8% | 1268.5 | 1.042 |
| c16-natural | 1502.1 (0.1%) | 1511.0 (0.1%) | +0.6% | 1496.6 | 1.010 |
| c32-fixed | 1861.8 (0.3%) | 1903.8 (0.1%) | +2.3% | 1877.1 | 1.014 |
| c32-natural | 2326.5 (0.1%) | 2358.8 (0.0%) | +1.4% | 2489.1 | 0.948 |

지연(ms, p50 / p95 / p99):

| 조건 | TPOT 끔 | TPOT 켬 | TPOT vLLM | TTFT 끔 | TTFT 켬 | TTFT vLLM |
|---|---|---|---|---|---|---|
| c1-fixed | 7.75 / 7.76 / 7.77 | 7.75 / 7.76 / 7.77 | 7.40 / 7.42 / 7.43 | 11.1 / 12.5 / 12.6 | 11.0 / 12.3 / 12.4 | 13.1 / 16.7 / 20.4 |
| c1-natural | 7.75 / 7.80 / 7.81 | 7.76 / 7.80 / 7.81 | 7.44 / 7.45 / 7.45 | 11.1 / 31.3 / 31.4 | 11.1 / 31.1 / 31.7 | 13.1 / 24.8 / 25.8 |
| c8-fixed | 8.99 / 9.45 / 9.56 | 8.93 / 9.42 / 9.54 | 9.00 / 9.36 / 9.44 | 22.3 / 34.1 / 62.0 | 21.5 / 34.4 / 61.2 | 30.9 / 38.0 / 51.7 |
| c8-natural | 8.79 / 8.84 / 9.00 | 8.77 / 8.81 / 8.94 | 8.94 / 9.04 / 9.59 | 20.7 / 42.9 / 64.2 | 22.0 / 48.0 / 82.4 | 37.5 / 42.7 / 62.8 |
| c16-fixed | 10.90 / 11.30 / 12.37 | 10.70 / 11.09 / 11.64 | 10.29 / 11.00 / 12.12 | 23.4 / 60.7 / 120.5 | 22.9 / 57.7 / 118.1 | 36.6 / 91.0 / 96.0 |
| c16-natural | 9.69 / 10.18 / 11.06 | 9.64 / 10.11 / 10.97 | 9.62 / 9.77 / 9.90 | 30.9 / 66.8 / 188.3 | 30.3 / 65.6 / 186.1 | 40.5 / 98.1 / 122.5 |
| c32-fixed | 14.00 / 14.46 / 14.97 | 13.63 / 14.25 / 14.66 | 13.50 / 15.06 / 18.96 | 41.2 / 239.0 / 268.7 | 40.0 / 233.5 / 262.4 | 53.7 / 166.5 / 180.0 |
| c32-natural | 11.82 / 14.31 / 15.31 | 11.68 / 14.67 / 15.48 | 10.77 / 11.49 / 12.97 | 34.1 / 295.6 / 389.9 | 33.0 / 272.1 / 365.7 | 63.4 / 222.7 / 241.2 |

| 조건 | E2E 끔 | E2E 켬 | E2E vLLM | 오류 끔 / 켬 | GPU peak MiB 끔 / 켬 |
|---|---|---|---|---|---|
| c1-fixed | 189 / 252 / 252 | 189 / 252 / 252 | 183 / 243 / 244 | 0 / 0 | 7174 / 7174 |
| c1-natural | 500 / 1022 / 1023 | 500 / 1022 / 1023 | 482 / 970 / 971 | 0 / 0 | 7296 / 7296 |
| c8-fixed | 205 / 312 / 319 | 204 / 309 / 317 | 221 / 317 / 319 | 0 / 0 | 7266 / 7266 |
| c8-natural | 573 / 1161 / 1169 | 571 / 1157 / 1173 | 592 / 1178 / 1189 | 0 / 0 | 7578 / 7578 |
| c16-fixed | 245 / 366 / 420 | 242 / 363 / 414 | 240 / 368 / 407 | 0 / 0 | 7374 / 7368 |
| c16-natural | 632 / 1286 / 1398 | 629 / 1278 / 1393 | 645 / 1273 / 1321 | 0 / 0 | 7902 / 7902 |
| c32-fixed | 327 / 505 / 659 | 317 / 512 / 647 | 320 / 500 / 559 | 0 / 0 | 7544 / 7552 |
| c32-natural | 774 / 1742 / 1846 | 765 / 1732 / 1809 | 741 / 1541 / 1558 | 0 / 0 | 8546 / 8524 |
출력 15/15 불변, 전 조건 prompt당 1종. 이득은 혼합 step 비중에 비례: C1 0, c8 +0.2–0.7%, c16 +0.6–1.8%, c32 +1.4–2.3%. c32-fixed가 1903.8 tok/s로 71절 vLLM(1877.1)을 넘고 TPOT p50 13.63(vLLM 13.50)까지 접근.

**판정**: 채택 후보. 안정성(attempt 19)·vLLM 같은-실행 비교(final6) 진행.

## 74. output-row LM head 포함 구성 안정성 — attempt 19 (head 5d1a1c1b)

71절 채택 플래그 + `--lm-head-rows output-rows-v1`. 각 조건 300초 지속 부하 + 취소 + 종료 후 회수. raw `qwen-scheduler-stability/adopt6-*-attempt19/`.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,544 | 0 | 0 | 2094 / 2119 / 2094 | 6571.5 / 6577.0 | 8546.2 / 8550.0 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 300 s | 25,632 | 0 | 0 | 1680 / 1704 / 1680 | 6573.7 / 6576.6 | 7588.5 / 7590.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 303 s | 5,760 | 0 | 0 | 1422 / 1421 / 1418 | 6498.5 / 6502.8 | 7898.8 / 7902.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 300 s | 18,624 | 0 | 0 | 1250 / 1233 / 1229 | 6501.9 / 6502.8 | 7373.9 / 7374.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 301 s | 3,360 | 0 | 0 | 835 / 834 / 833 | 6464.0 / 6464.4 | 7578.0 / 7578.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 11,616 | 0 | 0 | 775 / 772 / 768 | 6464.0 / 6464.4 | 7266.0 / 7266.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 340 s | 576 | 0 | 0 | 126.7 / 126.6 / 126.6 | 6434.2 / 6434.5 | 7296.0 / 7296.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 305 s | 1,920 | 0 | 0 | 126.1 / 126.0 / 125.9 | 6434.6 / 6435.0 | 7174.0 / 7174.0 | 20 / 40 | 0 | 완료 | 통과 |

**판정**: 안정성 8조건 통과 → `--lm-head-rows output-rows-v1` 채택.

## 75. 현재 채택 구성 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final6, head 5d1a1c1b)

raw `qwen-matrix/steady-final6/`. warmup 32 / retained 192, 2회 반복, 순서 교대, 제외 없음, 오류 양쪽 0, 조건당 384행. Riley 플래그 = 71절 + `--lm-head-rows output-rows-v1`.

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | GPU peak MiB R / V |
|---|---:|---:|---:|---|
| c1-fixed | 126.1 (0.1%) | 129.9 (0.0%) | 0.970 | 7174 / 15170 |
| c1-natural | 126.6 (0.1%) | 132.0 (0.1%) | 0.959 | 7296 / 15136 |
| c8-fixed | 799.6 (0.2%) | 753.0 (0.1%) | 1.062 | 7266 / 15136 |
| c8-natural | 862.1 (0.1%) | 837.4 (0.2%) | 1.029 | 7578 / 15180 |
| c16-fixed | 1321.3 (0.7%) | 1283.1 (0.7%) | 1.030 | 7374 / 15272 |
| c16-natural | 1513.5 (0.1%) | 1495.1 (0.3%) | 1.012 | 7902 / 15280 |
| c32-fixed | 1892.7 (0.2%) | 1887.3 (0.1%) | 1.003 | 7548 / 15338 |
| c32-natural | 2354.2 (0.0%) | 2492.9 (0.2%) | 0.944 | 8540 / 15288 |

지연(ms, p50 / p95 / p99):

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V |
|---|---|---|---|---|---|---|
| c1-fixed | 11.0 / 12.3 / 12.4 | 13.0 / 14.3 / 19.3 | 7.75 / 7.77 / 7.79 | 7.40 / 7.42 / 7.43 | 189 / 252 / 252 | 183 / 243 / 244 |
| c1-natural | 11.1 / 31.2 / 31.5 | 13.3 / 25.1 / 26.4 | 7.76 / 7.80 / 7.81 | 7.45 / 7.45 / 7.46 | 500 / 1021 / 1023 | 482 / 971 / 972 |
| c8-fixed | 21.7 / 33.8 / 60.1 | 33.0 / 38.4 / 53.9 | 8.93 / 9.42 / 9.48 | 9.10 / 9.42 / 9.51 | 204 / 309 / 314 | 227 / 320 / 325 |
| c8-natural | 20.1 / 41.5 / 69.4 | 37.5 / 42.4 / 64.0 | 8.76 / 8.80 / 9.13 | 8.93 / 9.04 / 9.09 | 571 / 1155 / 1171 | 593 / 1177 / 1182 |
| c16-fixed | 22.9 / 60.3 / 117.9 | 36.5 / 91.9 / 101.5 | 10.67 / 11.06 / 11.40 | 10.27 / 10.94 / 11.35 | 234 / 363 / 397 | 237 / 369 / 410 |
| c16-natural | 30.4 / 64.9 / 178.0 | 40.8 / 101.3 / 122.6 | 9.63 / 10.08 / 10.88 | 9.62 / 9.81 / 9.93 | 630 / 1276 / 1385 | 645 / 1278 / 1319 |
| c32-fixed | 40.6 / 220.9 / 263.0 | 53.1 / 171.0 / 172.9 | 13.72 / 14.31 / 14.65 | 13.44 / 15.38 / 18.70 | 316 / 520 / 646 | 323 / 498 / 555 |
| c32-natural | 36.9 / 286.5 / 374.6 | 62.7 / 180.3 / 240.1 | 11.69 / 14.34 / 15.18 | 10.75 / 11.62 / 12.62 | 766 / 1718 / 1825 | 743 / 1537 / 1555 |

출력 불변성(같은 실행): Riley 15/15, vLLM 6/15.

**판정**: 최소 기준 미달이지만 처리량은 8조건 중 5조건 충족.
- 처리량 ≥ vLLM: c8-fixed 1.062, c8-natural 1.029, c16-fixed 1.030, c16-natural 1.012, c32-fixed 1.003 충족. c1-fixed 0.970, c1-natural 0.959, c32-natural 0.944 미충족.
- TPOT p50 ≤ vLLM: c8-fixed(8.93 vs 9.10), c8-natural(8.76 vs 8.93) 충족. c16-natural +0.01 ms(9.63 vs 9.62), c32-fixed +0.28, C1 +0.31–0.35, c16-fixed +0.40, c32-natural +0.94.
- TPOT p95/p99: c32-fixed 둘 다 충족, c8-fixed p99, c8-natural p95 충족.
- TTFT p50: 8조건 모두 충족. TTFT p95: C1-fixed, C8 둘, C16 둘 충족. TTFT p99: c1-fixed만.
- E2E p50/p95/p99: c8 둘·c16-fixed 전부 충족, c16-natural p50/p95, c32-fixed p50 충족.
- 73절에서 올랐던 c8-natural TTFT p95/p99는 41.5 / 69.4 ms로 이전 범위(vLLM 42.4 / 64.0)로 돌아옴 — 변동이었음.

## 76. step당 token 예산 128 진단 — 미채택

근거: bit 동일 선택 후의 GEMM 표에서 layer당 합이 32행 182 µs, 128행 197 µs(추가 행당 0.15 µs), 144행 237 µs, 256행 295 µs(추가 행당 0.45–0.5 µs)로 128행을 넘을 때 급증합니다. 예산을 128로 줄이면 prefill 행이 싼 구간에 얹힐 것으로 기대했습니다.

측정: 채택 플래그에서 `--batch-token-budget 128 --prefill-chunk-tokens 128`, bucket 목록은 128까지(첫 시도 attempt45는 "bucket 목록이 예산에서 끝나야 한다"는 인자 검증으로 서버 시작 실패, 디렉터리 보존). 비교 기준은 같은 binary의 예산 256(73절 attempt44). raw `qwen-matrix/budget128-diagnostic/`.

| 조건 | 예산 256 tok/s | 예산 128 tok/s | 변화 | TPOT p50 ms | TTFT p50 / p95 / p99 ms (256 → 128) |
|---|---:|---:|---:|---|---|
| c1-fixed | 126 | 125 | −1.2% | 7.75 → 7.74 | 11 / 12 / 12 → 12 / 20 / 20 |
| c1-natural | 127 | 125 | −0.9% | 7.76 → 7.76 | 11 / 31 / 32 → 12 / 47 / 47 |
| c8-fixed | 799 | 762 | −4.6% | 8.93 → 8.95 | 21 / 34 / 61 → 30 / 45 / 90 |
| c8-natural | 861 | 852 | −1.0% | 8.77 → 8.75 | 22 / 48 / 74 → 28 / 57 / 83 |
| c16-fixed | 1321 | 1251 | −5.3% | 10.70 → 10.41 | 23 / 55 / 117 → 37 / 135 / 187 |
| c16-natural | 1511 | 1483 | −1.8% | 9.64 → 9.62 | 29 / 68 / 186 → 36 / 106 / 260 |
| c32-fixed | 1904 | 1451 | −23.8% | 13.63 → 11.09 | 40 / 228 / 257 → 187 / 352 / 408 |
| c32-natural | 2359 | 2250 | −4.6% | 11.68 → 11.49 | 33 / 276 / 369 → 67 / 407 / 556 |

출력 15/15 불변(예산이 달라도 출력은 같음), 오류 0.

해석: TPOT는 예상대로 내려갔지만(c32-fixed 13.63 → 11.09, vLLM 13.44보다 낮음) step당 처리할 수 있는 prefill token이 도착 속도보다 적어져 요청이 prefill에서 밀리고, TTFT와 처리량이 크게 나빠집니다. 싼 구간에 얹어도 "행 수 × 시간" 총량은 그대로입니다. **미채택.**

**c32-natural 격차의 재분해** (vLLM C32 trace `qwen-profile-diagnostic-vllm-c32-attempt02` 재분석 + 73절 Riley 프로파일):
- pure decode step: vLLM graph 7.86 ms + graph 밖 LM head 등 → step 간격 8.82 ms. Riley는 graph 재실행 기준 약 9.0–9.1 ms(프로파일 9.48 − eager 유휴). 차이 약 0.2 ms.
- prefill이 섞인 step: vLLM은 501 step 중 83개(17%), 평균 약 20.5 ms(TPOT 10.75에서 역산). Riley는 520 step 중 159개(31%), 평균 약 17.7 ms. pure 대비 초과 시간의 합은 Riley 약 1,375 ms, vLLM 약 969 ms로 **Riley가 prefill에 약 1.4배를 씁니다.**
- 그 차이(약 400 ms)의 대부분은 prefill 행의 attention입니다: Riley는 혼합 step마다 producer 2.75 ms(pure 0.79)와 merge 0.36 ms(pure 0.13) → 초과 약 2.2 ms × 159 ≈ 350 ms.

즉 c32-natural의 남은 격차는 prefill attention의 연산 구조에 있고(40·61·72절에서 launch 형태·tile 공유·기본 cuBLAS batched GEMM은 각각 기각), 줄이려면 prompt 구간 position의 수치 계약을 바꾸는 전용 kernel이 필요합니다.

## 77. prefill attention 전용 구조 재검토 — two-pass(손으로 쓴 kernel)는 기각, tensor-core 경로의 손익 계산

76절의 결론(c32-natural 격차 = prefill attention)에 따라 prefill 형상에서 다시 쟀습니다.

**page 병렬 two-pass softmax** (`qwen-attention-v2-diagnostics/attn_bench12.*`; 38절 설계를 66절의 butterfly producer + global merge와 prefill 형상에서 비교, µs):

| 형상 | 현재(producer + merge) | two-pass 합 (점수 / 통계 / 값 / 합산) |
|---|---:|---|
| 32행 × 400 | 31.4 | 46.4 (17.5 / 15.3 / 10.1 / 3.5) |
| 256행 (32×400 + 224×130) | 93.2 | 127.4 (53.2 / 34.4 / 31.2 / 7.2) |
| 160행 (32×400 + 128×200) | 80.8 | 117.4 |
| 96행 × 330 | 66.8 | 95.4 |
| 256행 × 130 | 71.0 | 91.7 |
| 224행 (position 174..397) | 152.8 | 206.3 |
| 128행 (position 200..327) | 70.0 | 104.7 |

모든 형상에서 **더 느립니다. 기각.** 점수 계산 kernel만으로 현재 producer의 약 65%가 들고, online softmax를 없애도 K/V를 두 번 읽는 비용이 더 큽니다. warp 단위 내적으로는 현재 producer가 사실상 한계입니다.

**tensor-core(행렬 곱) 경로의 손익** (60·61절 수치로 계산, 미구현):
- 긴 prompt chunk에서는 이득이 큽니다: 128행·key 328개 70.0 → 약 26 µs, 224행·position 174..397 152.8 → 약 40 µs.
- 그러나 출력 불변을 지키려면 prompt 구간은 **chunk 크기와 무관하게 항상** 이 경로를 써야 하고(작은 chunk만 기존 경로로 보내면 부하에 따라 수치가 갈림), sequence·layer당 고정 비용(모으기·GEMM 2회·softmax·되돌리기 약 20–26 µs)이 듭니다. 110-token 안팎의 chunk에서는 기존 경로(약 17 µs)보다 느립니다.
- 추정: c32-natural 약 +3%(0.944 → 약 0.97), c16-natural 약 +1.5%, fixed 계열은 중립 또는 약 −1%(여러 sequence를 padding해 한 번에 계산하려면 61절의 불변성 문제 때문에 cuBLASLt 고정 알고리즘이 필요).

판정: c32-natural을 vLLM 이상으로 올리지 못하면서 prompt 구간의 수치 계약을 바꾸고 fixed 계열을 위험에 두므로, **사용자 결정 없이 착수하지 않음.**

## 78. prefill 혼합 step의 graph 기록·재실행 (commits 500a0bc5 … 626a1cb6) — 진행 중

53절의 step graph는 pure decode step만 기록했습니다(행 수가 step마다 다른 혼합 step은 기록한 graph를 재사용할 수 없었음). 이번에 **행 수를 device에서 읽는 dynamic capture**를 넣었습니다: split+RoPE, KV write, D128 producer·merge, row gather, argmax kernel이 활성 행 수와 출력 행 수를 device row-position buffer의 끝 두 word에서 읽고, grid는 dense 행·출력 행 범위로 launch합니다. graph 하나가 (dense 형상, LM head 출력 형상)마다 decode-only step과 혼합 step 모두에 쓰입니다. 계산은 그대로라 출력 bit 불변.

경과
- 500a0bc5: 첫 구현. 첫 빌드는 header 선언의 `noexcept` 누락으로 컴파일 실패(rebuild log 35 보존) → 수정 후 gate 36 통과. native `native_bf16_paged_split_gqa_gpu`의 한 테스트가 첫 실행에서 "capture/제어 작업이 복사를 막는다"는 오류로 실패, 같은 binary로 병렬 3회·단일 thread 1회 재실행은 모두 통과(간헐적; 원인 미확정, `qwen-matrix/mixed-step-graph/qwen-dyn-native-test.log` 보존).
- b7cc6ab3: 최대 dense 형상(256행)의 step도 기록(이전에는 보관 자리가 없어 항상 eager). gate 37 통과.
- c5abdc81: step마다 행 수 2 word를 올리는 복사를 값이 바뀔 때만 수행(c1-natural −0.2%의 원인). gate 38 통과.
- 626a1cb6: 기록 graph 수 상한 32(아래 메모리 관찰 때문).

정상상태(Riley 단독 2회, 채택 플래그 그대로; 기준 = 직전 채택 빌드 attempt44). raw `qwen-matrix/mixed-step-graph/`.

| 조건 | 직전 tok/s | 첫 구현(attempt47) | 최대 형상 포함(attempt48, 편차) | 직전 대비 |
|---|---:|---:|---:|---:|
| c1-fixed | 126.1 | 126.1 (+0.0%) | 126.1 (0.1%) | +0.0% |
| c1-natural | 126.6 | 126.3 (−0.2%) | 126.3 (0.0%) | −0.2% |
| c8-fixed | 798.7 | 807.3 (+1.1%) | 809.1 (0.1%) | +1.3% |
| c8-natural | 860.7 | 861.8 (+0.1%) | 862.7 (0.0%) | +0.2% |
| c16-fixed | 1321.4 | 1339.6 (+1.4%) | 1333.4 (0.1%) | +0.9% |
| c16-natural | 1511.0 | 1511.6 (+0.0%) | 1512.5 (0.0%) | +0.1% |
| c32-fixed | 1903.8 | 1900.8 (−0.2%) | 1925.9 (0.5%) | +1.2% |
| c32-natural | 2358.8 | 2357.2 (−0.1%) | 2362.7 (0.4%) | +0.2% |

출력 15/15 불변, 오류 0. 이득은 예상(혼합 step당 0.4–0.5 ms)보다 작고 fixed 계열 약 1%. c32-fixed TPOT p50 13.63 → 13.52 ms.

안정성(attempt 21, c5abdc81 빌드, 상한 없음): 8조건 통과. raw `qwen-scheduler-stability/adopt7-*-attempt21/`. RSS 열은 첫¼ / 끝¼ / 최대.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB 첫¼ / 끝¼ / 최대 | GPU MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,544 | 0 | 0 | 2106 / 2120 / 2106 | 6555.8 / 6582.2 / 6586.3 | 8530.9 / 8554.3 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 300 s | 25,920 | 0 | 0 | 1700 / 1723 / 1699 | 6557.3 / 6566.5 / 6566.6 | 7572.8 / 7580.0 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 302 s | 5,760 | 0 | 0 | 1417 / 1421 / 1417 | 6539.2 / 6550.4 / 6553.6 | 7939.3 / 7949.1 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 300 s | 18,816 | 0 | 0 | 1264 / 1257 / 1240 | 6503.0 / 6507.8 / 6511.7 | 7375.9 / 7380.4 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 301 s | 3,360 | 0 | 0 | 831 / 834 / 831 | 6510.9 / 6521.6 / 6521.6 | 7624.0 / 7634.0 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 11,712 | 0 | 0 | 780 / 779 / 774 | 6475.1 / 6428.8 / 6480.8 | 7276.8 / 7285.9 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 340 s | 576 | 0 | 0 | 126.4 / 126.3 / 126.3 | 6450.5 / 6450.8 / 6450.8 | 7312.0 / 7312.0 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 305 s | 1,920 | 0 | 0 | 126.1 / 126.0 / 125.9 | 6442.4 / 6424.4 / 6442.8 | 7182.0 / 7182.0 | 20 / 40 | 0 | 완료 | 통과 |

**메모리 관찰**: 새 (형상, 출력 형상) 조합이 처음 나올 때마다 graph가 하나 기록되고 host RSS와 GPU 메모리가 각각 약 4–12 MiB씩 계단식으로 늡니다(c32-natural 10분위 GPU 8532 → 8536 → 8546 → 8558 → 취소 구간 8616; c16-natural 7944 → 7948 → 7974 → 7998). 조합 수는 유한하지만(형상 21 × 출력 형상 약 9) 그대로 두면 장시간에 수백 MiB–1 GiB대가 될 수 있어, 기록 수 상한 32를 넣었습니다(626a1cb6). 상한 빌드의 안정성(attempt 22)·30분 soak(attempt 23)·vLLM 같은-실행 비교(final7)는 진행 중. 상한 전 빌드로 시작했던 final7 실행과 그 전의 attempt 20 안정성 실행은 중단했고 디렉터리를 `-aborted`로 보존.

안정성(attempt 22, 상한 32 빌드 626a1cb6, HF-exact gate attempt 39 통과): 8조건 통과. raw `qwen-scheduler-stability/adopt8-*-attempt22/`. RSS는 첫¼ / 끝¼ / 최대, GPU는 첫¼ / 끝¼ / 최대.

| 조건 | 시간 | 요청 | 오류 | 출력 불일치 | tok/s 첫 / 끝 / 최저 round | RSS MiB | GPU MiB | 취소 idle / 부하 | round 후 KV | 회수 | 판정 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| c32-natural | 301 s | 8,544 | 0 | 0 | 2100 / 2119 / 2024 | 6560.4 / 6522.4 / 6576.5 | 8535.6 / 8550.0 / 8550 | 20 / 40 | 0 | 완료 | 통과 |
| c32-fixed | 301 s | 26,016 | 0 | 0 | 1730 / 1721 / 1703 | 6555.5 / 6562.3 / 6562.4 | 7570.9 / 7576.0 / 7590 | 20 / 40 | 0 | 완료 | 통과 |
| c16-natural | 303 s | 5,760 | 0 | 0 | 1417 / 1421 / 1417 | 6543.7 / 6560.8 / 6561.8 | 7943.6 / 7959.2 / 7970 | 20 / 40 | 0 | 완료 | 통과 |
| c16-fixed | 300 s | 18,816 | 0 | 0 | 1261 / 1250 / 1238 | 6505.5 / 6437.7 / 6511.0 | 7378.7 / 7384.0 / 7408 | 20 / 40 | 0 | 완료 | 통과 |
| c8-natural | 301 s | 3,360 | 0 | 0 | 832 / 833 / 832 | 6510.0 / 6519.4 / 6521.0 | 7624.0 / 7632.5 / 7676 | 20 / 40 | 0 | 완료 | 통과 |
| c8-fixed | 301 s | 11,712 | 0 | 0 | 782 / 780 / 774 | 6476.0 / 6477.1 / 6477.1 | 7277.4 / 7278.0 / 7310 | 20 / 40 | 0 | 완료 | 통과 |
| c1-natural | 341 s | 576 | 0 | 0 | 126.4 / 126.3 / 126.3 | 6450.8 / 6451.1 / 6451.2 | 7312.0 / 7312.0 / 7312 | 20 / 40 | 0 | 완료 | 통과 |
| c1-fixed | 305 s | 1,920 | 0 | 0 | 126.0 / 125.9 / 125.9 | 6442.1 / 6442.5 / 6442.5 | 7182.0 / 7182.0 / 7182 | 20 / 40 | 0 | 완료 | 통과 |

상한 적용 후 GPU 최대는 c32-natural 8550(상한 전 8616), c16-natural 7970(7998)으로 줄었습니다. c32-natural의 최저 round 2024 tok/s는 첫 round 대비 −3.6%인 한 round입니다(나머지는 2100–2119).

## 79. 상한 빌드 30분 soak (c32-natural, attempt 23, head 626a1cb6)

raw `qwen-scheduler-stability/adopt8-soak-c32-natural-attempt23/`.

| 항목 | 값 |
|---|---|
| 지속 시간 / round / 요청 | 1801 s / 531 / 50,976 |
| 오류 / 출력 불일치 | 0 / 0 |
| 처리량 구간 평균(10분위, tok/s) | 2116.8 → 2112.7 → 2112.8 → 2112.4 → 2112.5 → 2113.8 → 2114.1 → 2113.6 → 2112.9 → 2113.5 |
| round 처리량 min / max | 2099.1 / 2124.9 |
| TPOT p50 구간 평균(ms) | 11.49 → 11.53 → 11.52 → 11.53 → 11.52 → 11.52 → 11.52 → 11.53 → 11.53 → 11.51 |
| host RSS 첫¼ / 끝¼ / 최대 | 6575.7 / 6579.0 / 6579.1 MiB |
| GPU 메모리 첫¼ / 끝¼ | 8548.6 / 8550.0 MiB (10분위 표본 모두 8550) |
| GPU 온도 최대 | 78 °C |
| 취소 (idle / 부하 중) | 20 / 40 |
| round 후 KV 잔여 / 종료 후 회수 | 0 / 완료 |

30분 동안 처리량 ±0.2%, RSS +3.3 MiB, GPU 메모리 평탄. 22번 실행에서 본 한 round의 −3.6%는 재현되지 않음(531 round 최저 2099.1 = −0.6%). 54절 soak(어제 오전, 같은 조건) 대비 처리량 1,893 → 2,113 tok/s.

**판정**: 통과. 혼합 step graph(상한 32) 채택 후보 유지, vLLM 같은-실행 비교(final7) 대기.

## 80. 현재 채택 구성 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final7, head 626a1cb6)

raw `qwen-matrix/steady-final7/`. warmup 32 / retained 192, 2회 반복, 순서 교대, 제외 없음, 오류 양쪽 0, 조건당 384행. Riley 플래그 = 75절과 동일(빌드만 78절 변경 포함: 혼합 step graph, 최대 형상, 상한 32).

| 조건 | Riley tok/s (편차) | vLLM tok/s (편차) | 비율 | GPU peak MiB R / V |
|---|---:|---:|---:|---|
| c1-fixed | 126.1 (0.0%) | 127.9 (2.0%) | 0.986 | 7182 / 15170 |
| c1-natural | 126.3 (0.0%) | 131.9 (0.1%) | 0.958 | 7312 / 15136 |
| c8-fixed | 807.0 (0.0%) | 756.5 (0.9%) | 1.067 | 7274 / 15136 |
| c8-natural | 861.2 (0.1%) | 837.9 (0.1%) | 1.028 | 7612 / 15224 |
| c16-fixed | 1335.4 (0.0%) | 1289.6 (2.0%) | 1.036 | 7368 / 15272 |
| c16-natural | 1512.3 (0.3%) | 1499.3 (0.1%) | 1.009 | 7920 / 15280 |
| c32-fixed | 1906.1 (1.0%) | 1867.2 (0.2%) | 1.021 | 7556 / 15284 |
| c32-natural | 2355.0 (0.0%) | 2494.7 (0.2%) | 0.944 | 8504 / 15364 |

지연(ms, p50 / p95 / p99):

| 조건 | TTFT R | TTFT V | TPOT R | TPOT V | E2E R | E2E V |
|---|---|---|---|---|---|---|
| c1-fixed | 10.7 / 12.0 / 12.3 | 13.2 / 19.9 / 21.6 | 7.76 / 7.78 / 7.81 | 7.40 / 7.43 / 7.45 | 189 / 252 / 252 | 183 / 245 / 249 |
| c1-natural | 11.0 / 31.0 / 31.5 | 13.7 / 25.4 / 26.2 | 7.77 / 7.81 / 7.83 | 7.44 / 7.45 / 7.46 | 501 / 1023 / 1025 | 482 / 971 / 972 |
| c8-fixed | 20.7 / 32.9 / 61.1 | 32.7 / 38.2 / 52.4 | 8.88 / 9.30 / 9.36 | 9.06 / 9.46 / 9.71 | 204 / 305 / 311 | 226 / 319 / 325 |
| c8-natural | 21.9 / 41.3 / 86.2 | 37.5 / 42.3 / 63.7 | 8.76 / 8.80 / 9.03 | 8.93 / 9.04 / 9.07 | 571 / 1156 / 1173 | 592 / 1177 / 1181 |
| c16-fixed | 22.4 / 61.1 / 117.4 | 35.3 / 92.2 / 96.6 | 10.55 / 10.92 / 11.73 | 10.23 / 10.93 / 11.49 | 230 / 359 / 413 | 240 / 367 / 408 |
| c16-natural | 30.0 / 69.8 / 182.2 | 40.6 / 96.5 / 131.5 | 9.63 / 10.04 / 10.76 | 9.61 / 9.80 / 9.84 | 632 / 1278 / 1390 | 645 / 1270 / 1320 |
| c32-fixed | 40.4 / 233.2 / 259.9 | 54.4 / 186.1 / 186.9 | 13.54 / 14.24 / 14.66 | 13.52 / 15.12 / 18.16 | 312 / 531 / 658 | 328 / 500 / 567 |
| c32-natural | 33.6 / 269.5 / 380.4 | 62.4 / 192.3 / 240.6 | 11.69 / 14.31 / 15.87 | 10.74 / 11.61 / 12.49 | 765 / 1726 / 1846 | 739 / 1537 / 1555 |

출력 불변성(같은 실행): Riley 15/15, vLLM 6/15.

**판정**: 78절 변경 채택(안정성 8조건 attempt 22, soak attempt 23 통과; 처리량 fixed 계열 +1% 안팎, c1-natural −0.2%, GPU +8–70 MiB). 최소 기준은 미달.
- 처리량 ≥ vLLM: c8-fixed 1.067, c8-natural 1.028, c16-fixed 1.036, c16-natural 1.009, c32-fixed 1.021 충족. c1-fixed 0.986(이번 실행은 vLLM 반복 편차 2.0%), c1-natural 0.958, c32-natural 0.944 미충족.
- TPOT p50 ≤ vLLM: c8-fixed(8.88 vs 9.06), c8-natural(8.76 vs 8.93) 충족. c16-natural +0.02 ms, c32-fixed +0.02 ms, c16-fixed +0.32, C1 +0.33–0.36, c32-natural +0.95.
- TPOT p95/p99: c8 둘 전부, c32-fixed 전부, c16-fixed p95 충족.
- TTFT p50: 8조건 모두 충족. TTFT p95: c1-fixed, c8 둘, c16 둘 충족. TTFT p99: c1-fixed만.
- E2E p50: c8 둘, c16 둘, c32-fixed 충족.
- c1-natural은 126.3 tok/s로 75절(126.6)보다 0.2% 낮은 채 남음(행 수 복사 생략으로는 회복되지 않음 — 원인 미확정).

## 81. decode-only step에서도 LM head가 gathered logits를 직접 쓰기 (commit ee93a0a1) — 미채택, 되돌림

73절의 output-row LM head를 decode-only step까지 넓혀 logits row gather kernel(C32에서 약 15 µs)을 없애는 변경. HF-exact gate attempt 40 통과, 출력 15/15 불변. 기준 = 채택 binary(626a1cb6) attempt49, 새 = attempt50, 같은 세션 연속 측정, Riley 단독 2회. raw `qwen-matrix/lm-head-direct-decode/`.

| 조건 | 기준 tok/s (편차) | 새 tok/s (편차) | 변화 |
|---|---:|---:|---:|
| c1-fixed | 126.1 (0.0%) | 125.9 (0.0%) | -0.1% |
| c1-natural | 126.4 (0.0%) | 126.2 (0.0%) | -0.2% |
| c8-fixed | 807.3 (0.0%) | 806.8 (0.2%) | -0.1% |
| c8-natural | 861.6 (0.1%) | 861.3 (0.0%) | -0.0% |
| c16-fixed | 1343.7 (0.6%) | 1339.3 (0.1%) | -0.3% |
| c16-natural | 1511.6 (0.1%) | 1511.3 (0.2%) | -0.0% |
| c32-fixed | 1921.0 (0.2%) | 1918.0 (0.0%) | -0.2% |
| c32-natural | 2360.7 (0.0%) | 2357.3 (0.2%) | -0.1% |

TPOT p50: c1 7.76 → 7.78, c16-fixed 10.48 → 10.53, c32-fixed 13.47 → 13.51, c32-natural 11.67 → 11.69.

**판정**: 미채택. 전 조건에서 0 ~ −0.3%로 오히려 조금 느립니다(hidden gather와 추가 기록 조합의 비용이 없앤 gather보다 큼). revert 커밋으로 되돌리고 host binary를 다시 빌드(gate attempt 41). 기각 binary는 host `qwen-riley-ee93a0a1-rejected`로 보존.

**graph 재실행의 page extent 비용** (`qwen-attention-v2-diagnostics/attn_bench13.*`, 미채택 조사): 기록된 step은 producer를 workspace capacity(page 64)로 launch합니다. 필요한 page 수 대비 추가 비용은 32행×300 token 21.5 → 22.6 µs, 32행×100 token 10.6 → 13.6, 8행×300 token 8.6 → 12.2, 256행×130 token 62.1 → 70.2. extent를 구간화해 기록 조합에 넣으면 줄일 수 있으나 C32-natural은 긴 요청이 항상 섞여 최대 구간이 되고 조합 수가 상한 32를 넘기 쉬워 착수하지 않음.

되돌린 뒤의 host binary: sha256 `102bd4de…`로 보존해 둔 채택 binary(`qwen-riley-626a1cb6-adopted`)와 동일(재현 빌드), HF-exact HTTP gate attempt 42 통과.

## 82. 현재 채택 구성의 HF token 비교 재실행 (attempt 05) — 11/15, 이전과 동일

raw `qwen-hf-token-reference/attempt05-adopted-626a1cb6/`. 기준은 24절과 같은 HF transformers eager(BF16, greedy, batch 1, 기본 EOS, max_new_tokens = fixture 값). Riley는 현재 채택 플래그 전체, 일반 HTTP 경로, C1(순차)과 C32(전 prompt 동시 반복).

| | 일치 | 불일치 prompt (첫 차이 index / 길이) |
|---|---|---|
| C1 | 11 / 15 | arithmetic 14/24, planning 25/32, sql 4/16, natural-398 69/128 |
| C32 | 44 / 60 | 같은 4개 prompt, 같은 index |

39절(global-max merge 채택 시점)과 불일치 prompt·첫 차이 위치가 모두 같습니다. 그 이후의 채택 변경(step graph, kernel fusion 3종, bit 동일 GEMM 선택, wide RMSNorm, argmax, producer, output-row LM head, 혼합 step graph)이 출력을 바꾸지 않았다는 것과, C1·C32에서 같은 출력을 낸다는 것이 다시 확인됩니다. 불일치 4건은 39절에서 본 대로 logit 차이 0.25 이하의 근소한 경우입니다.

## 83. 2026-10-11 오전 기준 상태 요약 (Qwen2.5-3B, 채택 head 626a1cb6 = f8266c6a 소스)

| goal 항목 | 상태 | 근거 |
|---|---|---|
| 1. 기준·재현성 | 유지 | 측정 전 계획 고정(warmup 32 / retained 192, 2회, 순서 교대, 제외 없음), 기준·기각 binary host 보존, 실패·기각 증거 보존(59·61·72·76·77·81절 등) |
| 2. 정확성 | 부분 | HF-exact 경로: HTTP gate attempt 42까지 매 빌드 통과(P2048/C1/greedy). scheduler 경로: 동시성 불변 15/15, HF token 일치 11/15(82절) |
| 3. 최소 성능 | 미달 | 80절: 처리량 ≥ vLLM 5/8(C8·C16·c32-fixed), TPOT p50 ≤ vLLM 2/8(+0.02 ms 2건), TTFT p50 8/8. 미충족: C1(0.96–0.99), c32-natural(0.944), c16-fixed TPOT, C32 TTFT p95/p99 |
| 4. 안정성 | 통과 | 78절 attempt 22(8조건), 79절 30분 soak |
| 5. 도전 목표 | 미달 | 최소 기준 미달 |

남은 수단: (a) 행렬 곱 기반 prefill attention — prompt 구간 수치 변경, c32-natural 약 +3% 추정, fixed 0 ~ −1%(77절); (b) KV 기록 통합 등 0.04 ms급 미세 변경 — 81절처럼 편차 수준. C1은 가중치 읽기 시간이 step의 약 83%라 수치를 바꾸지 않는 범위에서 0.25 ms 이상 줄일 수단을 찾지 못함(65·72절).

## 84. 행렬 곱 기반 prefill attention 재평가 — sequence들을 padding해 한 번에 계산하면 fixed 계열에도 이득 (microbenchmark, 미구현)

77절의 손익 계산은 "sequence마다 따로 계산"을 전제로 했습니다. 61절에서 불변성 문제로 접었던 "여러 sequence를 padding해 한 번에"를 cuBLASLt로 다시 쟀습니다(`qwen-attention-v2-diagnostics/attn_bench14.*`; 모으기 2 kernel + QK GEMM + causal softmax + PV GEMM, 되돌리기 kernel 약 3 µs는 미포함).

| prefill 형상 | 행렬 곱 경로 (launch 한 묶음) | 현재 경로 (producer + merge, 72·77절) |
|---|---:|---:|
| 1 sequence × 128행, key 328 | 22.9 µs | 약 70 |
| 1 × 224행, key 224 | 26.6 | 약 57 |
| 1 × 174행, key 398 | 28.3 | 약 120 |
| 2 × 112행, key 130 | 25.6 | 약 43 |
| 4 × 56행, key 156 | 26.0 | 약 64 |
| 3 × 60행, key 120 | 22.3 | 약 25 |
| 8 × 28행, key 32 | 23.0 | 약 18 |

sequence 수와 무관하게 묶음당 약 22–28 µs입니다. 혼합 step에서 layer당 15–90 µs를 줄일 수 있어, 추정 효과는 c32-natural TPOT 약 −0.6 ms(11.69 → 약 11.0, vLLM 10.74), c32-fixed 약 −0.5 ms(13.54 → 약 13.0, vLLM 13.52), c16 계열 −0.2~0.3 ms입니다(계산 추정, 미구현).

불변성(같은 sweep, 846 조합 · 7.2M 원소):
- 형상마다 heuristic으로 고른 알고리즘: 61절과 같은 4개 조합에서 1원소씩 차이.
- 기준 형상(key 256 · query 1024 · batch 2)의 첫 heuristic 알고리즘을 고정하고 AlgoCheck가 거부하는 형상만 heuristic으로 대체: **차이 0**. 다만 1,700여 회 중 456회가 대체로 처리됐습니다.
- 기준 형상의 heuristic 후보 8개 중 **모든 probe 형상에서 유효한 것은 없었습니다**(아주 작은 형상 포함). 형상을 최소 크기로 padding하거나 형상 구간별로 알고리즘을 고정하는 설계가 필요하고, 그 구성이 실제로 전 형상 bit 불변인지는 아직 확인하지 못했습니다.

**판정**: 유망(이전 추정보다 큼), 미구현. 다음 단계는 (1) padding된 형상 전체에서 유효하고 bit 불변인 고정 구성 찾기, (2) 그 구성으로 native 연산 + runtime 반영, (3) prompt 구간 수치 변경에 따른 HF token 비교·안정성 재검증.

## 85. 행렬 곱 prefill attention의 불변 조건 규명과 최종 손익 (microbenchmark, 미구현)

84절의 남은 질문("대체 없이 bit 불변인 고정 구성이 있는가")을 쟀습니다(`qwen-attention-v2-diagnostics/attn_bench15–17.*`).

1. **전 형상에서 유효한 구성은 있다**: 행을 8의 배수(최소 8), key를 16의 배수로 padding하면, split-K 없는 (id, tile, stages) 구성 중 94개가 key 16–1024 × query 64–2048 × batch 2–64 전부에서 AlgoCheck를 통과(대체 0).
2. **그래도 불변이 아니다**: 그중 하나(id 5 / tile 20 / stages 7)를 고정해도 846 조합 중 123개(중복 포함)에서 1원소씩 달랐습니다. 차이는 **key 축 길이**(PV의 합산 차원, QK의 출력 행 수)가 달라질 때만 발생합니다.
3. **key 수를 고정하면 불변이다**: key 수를 64 / 256 / 1024로 고정한 sweep(480 조합, 3.8M 원소)에서는 행 수 1–256, 행 padding, batch 2–32를 바꿔도 **차이 0** — 형상별 heuristic과 고정 구성 모두.
4. 따라서 불변을 지키는 설계는 "key 수를 요청 고유의 값(prompt 길이로 정한 class: 64 / 256 / 1024)으로 고정하고, 같은 class의 sequence만 한 번에 계산"입니다. 가장 빠른 유효 구성(QK id 6 / tile 18 / stages 12, PV id 21 / tile 15 / stages 12) 기준 launch 한 묶음의 시간:

| 형상 (key class) | 행렬 곱 경로 | 현재 경로 (근사) |
|---|---:|---:|
| 1 × 128행 @200 (1024) | 31.6 µs | 약 70 |
| 1 × 224행 @0 (1024) | 42.1 | 약 57 |
| 1 × 174행 @224 (1024) | 37.9 | 약 120 |
| 2 × 112행 (256) | 26.7 | 약 43 |
| 4 × 56행 (256) | 27.1 | 약 64 |
| 3 × 60행 (256) | 25.0 | 약 25 |
| 1 × 125행 @0 (256) | 21.1 | 약 25 |
| 8 × 28행 (64) | 26.1 | 약 18 |
| 2 × 16행 (64) | 15.0 | 약 4 |

(되돌리기 kernel 약 3 µs 미포함. class가 다른 sequence가 섞이면 묶음이 class 수만큼 필요.)

5. **수정된 추정**: natural(prompt 16 / 128 / 398 token): 398-token prompt에서 layer당 약 −90 µs, 128-token −4, 16-token +14 → 요청 3건당 약 −2.9 ms, step당 약 −0.35 ms → c32-natural TPOT 11.69 → 약 11.3(vLLM 10.74), 처리량 0.944 → 약 0.97. fixed(prompt 약 125 token, class 256): 혼합 step당 0 ~ −0.5 ms, 평균 약 +1.5%. 84절의 "c32-natural ≈ 동률" 추정은 key를 요청 길이에 꼭 맞춘다는 가정이었고, 불변을 위한 key class 고정을 넣으면 77절 수준(+3%)으로 돌아갑니다.

**판정**: 불변을 지키는 형태가 존재함은 확인. 효과는 c32-natural +3%, fixed +1.5% 수준이며 c32-natural을 vLLM 이상으로 만들지는 못합니다. prompt 구간 수치 변경과 큰 구현(연산 신설, class별 묶음, HF·안정성 재검증)에 비해 이득이 작아 **사용자 결정 전에는 착수하지 않음.**

## 86. 사용자 결정(2026-10-11)과 최종 상태

**결정** (직접 질문에 대한 답):
1. Qwen2.5-3B의 최소 성능 기준을 **현재 측정 수준으로 조정**: C8 이상 처리량 vLLM 이상(c32-natural은 약 0.94×), C1 약 0.96–0.97×, 출력의 동시성 불변, GPU 메모리 vLLM의 약 절반.
2. scheduler 경로의 HF 일치는 **현재 계약을 인정**: Qwen의 수치 reference는 HF-exact 경로(P2048/C1/greedy, 매 빌드 HTTP gate 통과)이고, scheduler 경로의 계약은 "동시성 불변 15/15 + HF token 일치 11/15(불일치 4건은 logit 차이 0.25 이하)"로 문서화.

이에 따라 행렬 곱 기반 prefill attention(85절)과 C1·c32-natural의 추가 추격은 착수하지 않습니다.

**Qwen2.5-3B 채택 구성** (branch `codex/qwen-http-hf-exact`, head f8266c6a = 626a1cb6 소스, host binary sha256 `102bd4de…`, 보존본 `qwen-riley-626a1cb6-adopted`):

`--decode-attention-backend native-bf16-paged-split-gqa-d128-two-stage --decode-attention-merge global-max-v1 --decode-step-graph capture-v1 --gemm-reduction-policy first-heuristic-split-k-v1 --gemm-algorithm-selection bit-equivalent-table-v1 --sampling-backend gpu-greedy --projection-bias-backend merged-qkv-strict-staged-v1 --residual-rmsnorm fused --mlp-residual-norm fused-next-norm-v1 --mlp-activation fused-silu-multiply-v1 --query-key-rope fused-split-v1 --lm-head-rows output-rows-v1 --batch-shape-policy power-of-two --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --batch-token-budget 256 --prefill-chunk-tokens 256 --execution-completion iteration-batch --metadata-transport synchronous --execution-graph-policy disabled --reduction-profile canonical-v1`

**조정된 기준에 대한 판정** (80절 같은-실행 비교, 오류 0, 제외 없음):

| 조건 | 처리량 Riley / vLLM | 비율 | 조정 기준 | TPOT p50 R / V (ms) | TTFT p50 R / V (ms) |
|---|---|---:|:-:|---|---|
| c1-fixed | 126.1 / 127.9 | 0.986 | 충족 (≥ 약 0.96) | 7.76 / 7.40 | 10.7 / 13.2 |
| c1-natural | 126.3 / 131.9 | 0.958 | 충족 (≥ 약 0.96, 경계) | 7.77 / 7.44 | 11.0 / 13.7 |
| c8-fixed | 807.0 / 756.5 | 1.067 | 충족 | 8.88 / 9.06 | 20.7 / 32.7 |
| c8-natural | 861.2 / 837.9 | 1.028 | 충족 | 8.76 / 8.93 | 21.9 / 37.5 |
| c16-fixed | 1335.4 / 1289.6 | 1.036 | 충족 | 10.55 / 10.23 | 22.4 / 35.3 |
| c16-natural | 1512.3 / 1499.3 | 1.009 | 충족 | 9.63 / 9.61 | 30.0 / 40.6 |
| c32-fixed | 1906.1 / 1867.2 | 1.021 | 충족 | 13.54 / 13.52 | 40.4 / 54.4 |
| c32-natural | 2355.0 / 2494.7 | 0.944 | 충족 (≈ 0.94) | 11.69 / 10.74 | 33.6 / 62.4 |

- 출력: Riley 15/15 prompt가 전 조건·반복에서 1종(vLLM 6/15). GPU peak 7.2–8.5 GiB(vLLM 15.1–15.4 GiB).
- 안정성: 8조건 300초(78절 attempt 22) + 30분 soak(79절) 통과, 오류 0, 메모리 평탄.
- 조정 전 원래 기준(8조건 전부 처리량 ≥ vLLM, TTFT/TPOT ≤ vLLM, P95/P99 회귀 없음)으로는 **미달**이라는 사실은 그대로입니다: 처리량 5/8, TPOT p50 2/8(+0.02 ms 2건), C32의 TTFT p95/p99 열세.

**SmolLM2** (17절, gate v1 전체 matrix): 8조건 모두 원래 최소 기준 통과. 도전 목표는 c1-fixed·c32-fixed·c32-natural의 TPOT에서 미달.

**goal 항목별 최종**

| 항목 | SmolLM2 | Qwen2.5-3B |
|---|---|---|
| 1. 기준·재현성 | 완료 | 완료 (기각 증거·binary 보존) |
| 2. 정확성 | exact reference 일치 (HTTP) | HF-exact 경로 gate 통과 + scheduler 경로 계약(불변 15/15, HF 11/15) — 사용자 인정 |
| 3. 최소 성능 | 통과 (원래 기준) | 조정 기준 충족, 원래 기준 미달 |
| 4. 안정성 | 통과 (4·7절) | 통과 (78·79절) |
| 5. 도전 목표 | 미달 (TPOT 3조건) | 미달 |

**이틀간 Qwen 처리량 비율의 변화** (같은-실행 비교): 0.69–0.79(10-08) → 0.81–0.91(10-09) → 0.845–0.930(10-10 오전) → 0.944–1.067(10-11). 10-10 이후의 채택 변경은 모두 출력 bit 불변입니다.

**알려진 한계와 주의**
- 기록된 step graph는 최대 32개(상한), 재실행 step은 command-batch 사용 장부에 buffer를 등록하지 않음(53절부터의 한계).
- native 테스트 `native_bf16_ragged_paged_d128_two_stage_v3_c2048_b8…`가 병렬 실행에서 1회 실패 후 4회 재실행 통과(78절, 원인 미확정).
- c1-natural은 혼합 step graph 도입 후 0.2% 낮음(원인 미확정).
- bit 동일 GEMM 표의 tile·stage id는 cuBLAS 12.8 기준이며, 다른 버전에서는 시작 시 probe가 거부하면 anchor 알고리즘으로 되돌아갑니다(출력은 유지, 속도 이득만 사라짐).
- branch의 변경은 main에 병합하지 않았고, 이 증거 디렉터리는 커밋하지 않은 상태입니다.

## 87. SmolLM2 Batch14 후보 1차: 활성 행 수에 맞춘 decode attention launch grid (진단 측정, 2026-10-11)

도전 목표(TPOT 10% 감소)에 남은 c1-fixed·c32-fixed·c32-natural 중 C1부터 다시 봤습니다. 증거: `smollm2-batch14-diagnostics/` (source patch 2개, native 로그, 실행별 `summary.json`, 기각 patch 2개). 브랜치 `codex/serving-kernel-batch14`(Batch13 `9acd7ee`와 같은 tree에서 시작, commit `7587acfe`, `2c0d9a23`). **모두 단일 lane 진단 측정이며 host gate·vLLM 동시 측정이 없어 성능 근거(qualifying)가 아닙니다.**

병목(Batch11 nsys, C1 fixed): 층당 kernel 9개 ≈ 26.4 µs. 그중 QK `scores`(2.27 µs)는 32행 용량으로 잡은 3072-block grid를 띄우는데, 활성 1행이면 96 block만 일하고 나머지는 즉시 return합니다. graph 재생 microbenchmark(`grid_floor.cu`)로 빈 block 비용을 쟀더니 scores 2.3 → 0.9 µs, AV `independent_values` 1.25 → 0.8 µs였습니다.

| 시도 | 내용 | C1 fixed TPOT p50 | 판정 |
|---|---|---:|---|
| Batch13 | 기준 | 0.881–0.885 ms | |
| b14a | 한 block에 warp 18–24개를 모아 grid를 (4,32)로 축소 (kernel 변경, native bit 일치) | 0.937–0.939 ms | **기각** — 빈 block은 줄지만 소수 SM에 warp가 몰려 6% 느려짐 |
| b14b | kernel은 그대로, 활성 1행 전용 decode capture(attention grid만 1행 크기)를 추가하고 packet의 live=1일 때만 재생 | 0.850–0.851 ms | 채택 후보 |
| b14c | b14b를 live ≤1/2/4/8/16 bucket으로 일반화 | 0.852–0.853 ms | 채택 후보(현재 브랜치 head) |
| b14d | prefill/mixed capture도 총 행 수 64/128/192/256/384 bucket으로 추가 | 0.852–0.853 ms, C1 TTFT·C32 TPOT 변화 없음 | **기각** — 효과 없음 |

b14c vs Batch13, 조건별 2회 교차 순서(C32는 5회), 384 요청, 전 요청 고정 reference(token/text/finish) 일치:

| 조건 | 처리량 | TPOT p50 (ms) | 비고 |
|---|---:|---:|---|
| c1-fixed | 994.8 → 1026.0 (+3.1%) | 0.884 → 0.852 (−3.6%) | 실행 간 편차 0.002 |
| c1-natural | 1015.5 → 1043.3 (+2.7%) | 0.884 → 0.855 (−3.3%) | |
| c8-fixed | 5233.7 → 5227.0 | 1.308 → 1.324 | 실행 간 편차(1.28–1.34)가 차이보다 큼 — 판정 불가 |
| c8-natural | 5796.5 → 5917.6 (+2.1%) | 1.277 → 1.248 (−2.3%) | |
| c16-fixed | 7756.4 → 7840.9 (+1.1%) | 1.781 → 1.757 (−1.3%) | |
| c16-natural | 9575.5 → 9689.6 (+1.2%) | 1.531 → 1.505 (−1.7%) | |
| c32-fixed | 5회 평균 | 2.715 → 2.756 (+1.5%) | 실행 간 편차 2.63–2.87로 차이보다 큼. 회귀 여부 미확정 |
| c32-natural | 5회 평균 | 2.286 → 2.318 (+1.4%) | 실행 간 편차 2.25–2.36으로 차이보다 큼. 회귀 여부 미확정 |

정확성: kernel 산술은 변경 없음(같은 kernel, launch 범위만 축소). native `kernel_batch14_correctness.cu` — 기존 3161건 로그 SHA가 Batch13과 동일(`f9f88464…`), 좁은 grid vs 용량 grid 4800건이 guard·비활성 행 포함 bit 일치. 재생 선택은 packet의 live 행 수 ≤ capture 행 한도일 때만 좁은 capture를 쓰고, 16 초과는 기존 capture를 씁니다.

남은 격차와 다음 실험:

- c1-fixed TPOT 비율은 0.991 → 약 0.955로 추정(이번 측정에 vLLM 없음). 0.90까지 층당 약 1.7 µs가 더 필요합니다. 후보: QKV parts+merge+RoPE를 열 쌍 단위 6-warp block 하나로 합쳐 kernel 1개 제거(추정 층당 −1.0 µs, 새 bit 일치 테스트 필요), scores를 AV kernel에 합치기(추정 −0.5 µs). 둘 다 해도 0.90은 경계선입니다.
- C32는 이번 변경으로 얻는 것이 없고(행이 17개 이상이면 기존 capture), 5회 평균이 두 조건 모두 1.4–1.5% 높게 나왔습니다. 편차 범위 안이라 회귀라고 단정할 수 없지만 배제도 못 합니다. 1행 전용만 넣은 b14b는 같은 실행에서 c32-fixed 2.725(3회, Batch13 2.693)·c32-natural 2.289(Batch13 2.291)였습니다. 정식 측정 전에 반복 수를 늘려 확인해야 하며, 회귀로 확인되면 bucket을 줄입니다. §6의 host 경로(publish 전용 thread, plan/authorize 복사 제거, 추정 2–3%)가 다음 후보입니다.
- 미실행: Batch14의 gate 포함 128-lane 정식 측정, vLLM 동시 비교, 안정성(취소·재요청·soak). 후보가 2개 이상 모이면 한 batch로 실행합니다.

### 87.1 C32 회귀 확인 (8회 추가, 세 binary 순서 회전, `runs/q-c32*`, `run_c32.log`)

| 조건 | binary | 처리량 평균 | TPOT p50 평균 (ms) | 표준편차 | 최소–최대 | TPOT p99 평균 |
|---|---|---:|---:|---:|---:|---:|
| c32-fixed | Batch13 | 10154 | 2.767 | 0.064 | 2.665–2.841 | 3.755 |
| c32-fixed | b14b (1행 전용) | 10218 | 2.715 | 0.057 | 2.644–2.810 | 3.676 |
| c32-fixed | b14c (bucket) | 10245 | 2.721 | 0.071 | 2.651–2.845 | 3.622 |
| c32-natural | Batch13 | 12893 | 2.300 | 0.042 | 2.222–2.346 | 2.696 |
| c32-natural | b14b | 12920 | 2.295 | 0.018 | 2.274–2.322 | 2.645 |
| c32-natural | b14c | 12945 | 2.289 | 0.024 | 2.238–2.313 | 2.714 |

앞의 5회와 합친 13회 평균은 c32-fixed 2.747 → 2.734, c32-natural 2.295 → 2.300으로, 차이가 표준편차보다 훨씬 작습니다. 앞 표의 +1.4–1.5%는 반복 편차였고 **C32 회귀는 없는 것으로 판단**합니다(개선도 없음). 진단 측정이라는 한계는 같습니다.

### 87.2 융합 QKV kernel (commit 3·4번째, 진단 측정)

`qkv_parts_merge_rope`: block 하나가 RoPE가 짝짓는 두 열 tile(dim, dim+32)을 맡아 기존 K192 chunk 계산을 그대로 돌린 뒤, 자기 열만 merge·회전·KV 기록합니다. 층당 kernel 1개가 줄어듭니다(9 → 8). native: 기존 `shared32_qkv_parts` + `qkv_merge_rope` 순서와 parts·회전 query·cache key/value가 guard 포함 bit 일치(270건: 위치 9종 × 행 수 10종 × seed 3), 기존 3161건 로그 SHA 동일.

C1 (2회 교차, 384 요청, reference 전부 일치):

| 조건 | Batch13 | b14c (좁은 grid) | b14e (+1행 capture에 융합 QKV) |
|---|---:|---:|---:|
| c1-fixed TPOT p50 ms | 0.884 | 0.852 | 0.811 (−8.3%) |
| c1-natural TPOT p50 ms | 0.884 | 0.856 | 0.815 (−7.8%) |
| c1-fixed tok/s | 995 | 1026 | 1069 |

C8–C32 (6회, 세 binary 순서 회전, `runs/f-*`, `run_f.log`). b14f = 모든 decode capture에 융합 QKV (현재 브랜치 head, binary `c079a5b5…`):

| 조건 | Batch13 tok/s | b14f tok/s | Batch13 TPOT p50 (평균 ± sd) | b14e | b14f | b14f 변화 |
|---|---:|---:|---:|---:|---:|---:|
| c8-fixed | 5268 | 5450 | 1.294 ± 0.020 | 1.305 ± 0.027 | 1.245 ± 0.017 | −3.8% |
| c8-natural | 5823 | 6068 | 1.269 ± 0.010 | 1.248 ± 0.014 | 1.215 ± 0.012 | −4.3% |
| c16-fixed | 7697 | 7870 | 1.791 ± 0.045 | 1.767 ± 0.071 | 1.750 ± 0.082 | −2.3% (편차 안) |
| c16-natural | 9277 | 9624 | 1.594 ± 0.038 | 1.582 ± 0.038 | 1.526 ± 0.044 | −4.3% |
| c32-fixed | 10173 | 10365 | 2.772 ± 0.067 | 2.697 ± 0.038 | 2.683 ± 0.044 | −3.2% |
| c32-natural | 12954 | 13092 | 2.280 ± 0.035 | 2.310 ± 0.030 | 2.265 ± 0.035 | −0.7% (편차 안) |

TTFT p50와 TPOT p99는 세 binary가 편차 범위 안에서 같습니다. b14f를 Batch14 후보로 둡니다. 여전히 진단 측정이며, vLLM 포함 정식 측정과 안정성은 미실행입니다.

## 88. SmolLM2 Batch14 정식 측정 (gate v1, 96 lane, 2026-10-11)

`smollm2-batch14-gatev1/` — 계획(`batch-plan.json`, 측정 전 고정), controller(`run.py`, Batch13 gate v1 controller에서 엔진 목록과 binary 고정값만 변경), 독립 재계산 script·결과(`analysis.json`, `report.md`), 전체 raw(`kernel-batch14-gatev1-quiet-attempt01.tar.gz`), 검증 script와 실패한 1차 검증 로그.

- 후보: 브랜치 `codex/serving-kernel-batch14` 4개 commit(호스트 commit `57a0d0dd`, 로컬과 tree 동일 `7db86374`), binary `32a06d8e…`. 변경은 §87·§87.2의 두 가지(좁은 decode attention capture, 융합 QKV dispatch).
- 검증: native 3161건(로그의 해당 부분 SHA가 Batch13과 동일) + 융합 QKV 270건 + 좁은 grid 4800건 bit 일치, HTTP exact screen 8조건 60건 일치. 검증 1차 시도는 검증 script의 로그 파서 assertion 오류로 실패(native 실행 자체는 통과)했고 보존했습니다.
- 조건: Batch13(`3dd44e75…`) / Batch14 후보 / vLLM 0.27.1, 8조건 × 4회, 순서 [Batch13, 후보, vLLM] ↔ 역순 교대, warmup 96 / retained 384, 원래 host gate v1(cpu some ≤ 5, io full ≤ 5, memory full ≤ 0.5, 3회 연속, 300초). V52·Batch12 lane은 제외(별도 비교 조건이며 Batch13 attempt와 병합하지 않음).
- 결과: 96 lane 전부 완료, 요청 오류 0, Riley 전 요청 고정 reference 일치, 제외 없음, controller 요약과 독립 재계산 불일치 0.

| cell | engine | tok/s (sd) | TTFT p50/p95/p99 ms | TPOT p50 (sd) /p95/p99 ms | E2E p50/p95/p99 ms | err | GPU MiB | RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | batch13 | 997.6 (1.6) | 3.08/3.26/3.42 | 0.8812 (0.0011) /0.891/0.912 | 20.6/30.5/30.8 | 0 | 1240 | 754 |
| c1-fixed | candidate | 1068.8 (1.8) | 3.07/3.29/3.46 | 0.8102 (0.0014) /0.826/0.837 | 18.8/28.3/28.9 | 0 | 1250 | 763 |
| c1-fixed | vllm | 778.6 (8.0) | 7.73/13.37/17.37 | 0.8891 (0.0014) /1.008/1.143 | 27.7/38.0/43.7 | 0 | 8072 | 3722 |
| c1-natural | batch13 | 1014.4 (2.8) | 3.16/5.79/6.24 | 0.8858 (0.0019) /1.001/1.005 | 59.0/132.9/133.7 | 0 | 1260 | 754 |
| c1-natural | candidate | 1086.9 (1.0) | 3.16/5.76/6.18 | 0.8157 (0.0009) /0.936/0.941 | 54.6/124.7/125.4 | 0 | 1270 | 763 |
| c1-natural | vllm | 881.2 (7.8) | 7.93/13.47/17.41 | 1.0159 (0.0041) /1.094/1.180 | 72.0/143.2/152.2 | 0 | 8094 | 3002 |
| c8-fixed | batch13 | 5178.7 (45.5) | 4.81/6.70/8.92 | 1.3267 (0.0216) /1.497/1.568 | 30.3/49.1/51.9 | 0 | 1268 | 754 |
| c8-fixed | candidate | 5416.2 (37.3) | 4.84/6.85/8.66 | 1.2525 (0.0122) /1.401/1.476 | 29.1/46.9/48.7 | 0 | 1278 | 763 |
| c8-fixed | vllm | 3128.6 (181.1) | 13.04/22.18/30.34 | 1.8414 (0.0205) /2.516/3.122 | 50.7/85.9/101.5 | 0 | 8122 | 2999 |
| c8-natural | batch13 | 5810.1 (26.7) | 7.32/12.42/14.90 | 1.2701 (0.0063) /1.331/1.369 | 87.0/175.7/178.0 | 0 | 1416 | 754 |
| c8-natural | candidate | 6070.6 (37.4) | 6.96/12.15/13.73 | 1.2147 (0.0113) /1.282/1.325 | 83.3/168.1/170.8 | 0 | 1426 | 763 |
| c8-natural | vllm | 4502.0 (70.0) | 11.21/18.90/28.71 | 1.5827 (0.0176) /1.829/2.004 | 111.5/230.3/244.4 | 0 | 8122 | 3016 |
| c16-fixed | batch13 | 7568.0 (268.4) | 5.96/10.33/23.05 | 1.8340 (0.0962) /2.046/2.194 | 40.4/66.7/71.9 | 0 | 1296 | 756 |
| c16-fixed | candidate | 7773.4 (76.1) | 6.27/10.05/22.62 | 1.7579 (0.0273) /2.021/2.204 | 38.5/65.2/71.1 | 0 | 1306 | 764 |
| c16-fixed | vllm | 4777.8 (88.1) | 18.82/33.94/44.95 | 2.3331 (0.0556) /3.171/3.676 | 65.4/106.7/121.3 | 0 | 8074 | 3033 |
| c16-natural | batch13 | 9264.3 (119.6) | 7.84/13.76/28.55 | 1.5967 (0.0339) /1.697/1.811 | 108.0/219.9/223.9 | 0 | 1596 | 756 |
| c16-natural | candidate | 9423.5 (116.5) | 7.73/14.05/30.74 | 1.5692 (0.0271) /1.676/1.746 | 106.4/216.9/222.7 | 0 | 1606 | 765 |
| c16-natural | vllm | 7485.3 (259.5) | 14.82/30.69/41.79 | 1.8322 (0.0386) /2.307/2.604 | 131.4/283.7/308.5 | 0 | 8074 | 3051 |
| c32-fixed | batch13 | 10193.9 (135.9) | 7.58/21.31/42.43 | 2.7281 (0.0654) /3.048/3.481 | 57.8/98.8/113.3 | 0 | 1352 | 759 |
| c32-fixed | candidate | 10204.4 (71.5) | 7.28/21.78/42.42 | 2.7314 (0.0430) /3.151/3.604 | 58.1/100.1/121.8 | 0 | 1362 | 767 |
| c32-fixed | vllm | 6673.6 (657.5) | 34.48/53.42/63.54 | 2.8708 (0.1440) /6.107/7.410 | 86.7/171.7/208.0 | 0 | 8124 | 3019 |
| c32-natural | batch13 | 12791.9 (161.1) | 9.11/30.74/67.54 | 2.3141 (0.0353) /2.444/2.699 | 155.5/314.2/346.6 | 0 | 1956 | 760 |
| c32-natural | candidate | 13067.1 (42.6) | 9.80/28.59/63.93 | 2.2585 (0.0234) /2.425/2.709 | 152.1/307.9/347.7 | 0 | 1966 | 768 |
| c32-natural | vllm | 10922.3 (700.9) | 24.91/62.97/95.81 | 2.4497 (0.1299) /2.967/3.284 | 182.3/379.7/418.2 | 0 | 8124 | 3530 |

| cell | cand/vLLM tok/s | TTFT p50 | TPOT p50 | worst TTFT/TPOT/E2E P95,P99 ratio | minimum | challenge | cand/Batch13 tok/s | TPOT p50 | worst-repeat cand TPOT vs best vLLM |
|---|---:|---:|---:|---:|:-:|:-:|---:|---:|---:|
| c1-fixed | 1.373 | 0.397 | 0.911 | 0.820 | pass | miss | 1.071 | 0.919 | 0.914 |
| c1-natural | 1.233 | 0.398 | 0.803 | 0.870 | pass | pass | 1.071 | 0.921 | 0.809 |
| c8-fixed | 1.731 | 0.372 | 0.680 | 0.557 | pass | pass | 1.046 | 0.944 | 0.697 |
| c8-natural | 1.348 | 0.620 | 0.768 | 0.730 | pass | pass | 1.045 | 0.956 | 0.779 |
| c16-fixed | 1.627 | 0.333 | 0.753 | 0.637 | pass | pass | 1.027 | 0.959 | 0.779 |
| c16-natural | 1.259 | 0.521 | 0.856 | 0.765 | pass | pass | 1.017 | 0.983 | 0.894 |
| c32-fixed | 1.529 | 0.211 | 0.951 | 0.668 | pass | miss | 1.001 | 1.001 | 1.018 |
| c32-natural | 1.196 | 0.393 | 0.922 | 0.831 | pass | miss | 1.022 | 0.976 | 0.989 |

lanes 96; request errors 0; Riley reference all-match True; nonzero exits []; controller-summary mismatches 0; host gate records 288

(값은 4회 반복의 평균, 괄호는 반복 간 표준편차. 비율은 후보/vLLM, 1보다 작을수록 후보가 빠름. "worst … ratio"는 TTFT·TPOT·E2E의 P95·P99 여섯 비율 중 최댓값.)

판정:

- **최소 기준: 8개 조건 모두 통과** (처리량 ≥ vLLM, TTFT/TPOT p50 ≤ vLLM, P95/P99 전부 vLLM 이하).
- **도전 목표: 5/8 통과, 3개 미달** — c1-fixed TPOT 0.911(Batch13 0.991에서 개선, 목표 0.90에 1.1%p 부족), c32-fixed TPOT 0.951, c32-natural TPOT 0.922. 처리량 +15%와 TTFT −10%는 8개 조건 모두 충족.
- Batch13 대비: C1 처리량 +7.1%·TPOT −8%, C8 +4.5%·−5%, C16 +2~3%·−2~4%, c32-natural +2.2%·−2.4%, c32-fixed 변화 없음(1.001/1.001).
- 주의: c32-fixed에서 후보의 TPOT p95/p99(3.151/3.604)가 Batch13(3.048/3.481)보다 3–4% 높습니다. p50의 반복 편차(sd 0.04–0.07)와 같은 크기라 회귀로 단정할 수 없지만 개선은 아닙니다. vLLM(6.107/7.410)보다는 낮습니다. "후보 최악 반복 vs vLLM 최고 반복"으로는 c32-fixed TPOT만 뒤집힙니다(1.018).
- GPU 메모리 최대치는 후보가 Batch13보다 10 MiB 많습니다(capture 추가분), RSS +8~9 MiB.

남은 격차와 다음 실험: c1-fixed는 층당 약 0.3 µs(토큰당 10 µs)가 더 필요합니다 — 후보는 QK score를 AV kernel에 합치기. C32 두 조건은 decode kernel이 아니라 mixed(prefill 동반) step과 host 경로가 지배하므로 §6의 publish 전용 thread·plan/authorize 복사 제거가 다음 후보입니다. 안정성 실행은 §89.

## 89. SmolLM2 Batch14 안정성 (2026-10-11)

`smollm2-batch14-stability/` (계획·controller·lane 로그·분석). Batch13 안정성 controller를 그대로 쓰고 matched attempt와 binary 고정값만 Batch14 정식 측정(§88)으로 바꿨습니다. lane별 원자료는 호스트 `/data/riley-serving-261007/kernel-batch14-stability-*-attempt01`에 있습니다.


지속 부하 300s, idle 취소 20×2(early/decode, 매회 전체 corpus exact 재요청), 부하 중 취소 40회. 제외 없음.

| 조건 | 엔진 | 결과 | 지속 요청 | 오류 | exact | 처리량 첫/끝 round | TPOT p50 첫/끝 | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | RSS 기울기 MiB/min | drain 횟수(최대 ms) | 취소(idle/부하) | 종료 회수 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-fixed | candidate | 통과 | 16128 | 0 | 16128 | 1067.9/1068.2 | 0.811/0.811 | 1250/1250 | 762.9/763.0 | 0.027 | 84 (20.7) | 40/40 | True |
| c1-natural | candidate | 통과 | 4608 | 0 | 4608 | 1088.3/1086.5 | 0.815/0.815 | 1270/1270 | 762.8/763.0 | 0.041 | 84 (20.7) | 40/40 | True |
| c8-fixed | candidate | 통과 | 80640 | 0 | 80640 | 5394.3/5428.4 | 1.261/1.245 | 1278/1278 | 763.1/763.4 | 0.079 | 84 (20.8) | 40/40 | True |
| c8-natural | candidate | 통과 | 24576 | 0 | 24576 | 6109.4/6132.3 | 1.201/1.201 | 1426/1426 | 763.1/763.3 | 0.041 | 84 (20.7) | 40/40 | True |
| c16-fixed | candidate | 통과 | 117120 | 0 | 117120 | 7833.5/7938.3 | 1.755/1.696 | 1306/1306 | 765.1/765.6 | 0.149 | 84 (20.8) | 40/40 | True |
| c16-natural | candidate | 통과 | 38784 | 0 | 38784 | 9475.6/9741.4 | 1.565/1.499 | 1606/1606 | 765.0/765.4 | 0.101 | 84 (20.8) | 40/40 | True |
| c32-fixed | candidate | 통과 | 152832 | 0 | 152832 | 10203.3/10289.2 | 2.766/2.715 | 1362/1362 | 768.5/769.2 | 0.177 | 84 (20.9) | 40/40 | True |
| c32-natural | candidate | 통과 | 52224 | 0 | 52224 | 13121.7/12770.5 | 2.253/2.336 | 1966/1966 | 769.1/769.9 | 0.225 | 84 (20.7) | 40/40 | True |
| c32-natural | vllm | 통과 | 44928 | 0 | 33247 | 10603.1/11024.7 | 2.530/2.486 | 8124/8124 | 3036.9/3067.0 | 7.987 | 0 (—) | 40/40 | True |
| c32-fixed | vllm | 통과 | 102528 | 0 | 97030 | 6454.5/6682.6 | 3.181/3.263 | 8124/8124 | 3020.2/3061.3 | 10.896 | 0 (—) | 40/40 | True |

전체 lane 통과: True

한계: early cancellation closes right after the request write; the server-side phase (queued/prefill) is not observable from the client; memory is sampled once per second; peaks between samples are not proven; no performance qualification

10 lane 모두 통과: 요청 오류 0, Riley 전 요청 exact, 매 취소·재요청 뒤 active/waiting/KV 블록 0으로 drain(84회, 최대 20.9 ms), 종료 시 자원 회수 확인. Riley RSS 기울기 0.03–0.23 MiB/min(Batch13 때 0.05–0.25와 같은 수준), GPU 메모리는 구간 내 변화 없음. vLLM의 "exact" 열은 Riley reference와의 일치 수로, 엔진 간 산술 차이를 기록한 것이며 통과 조건이 아닙니다. 미실행: Batch14 30분 soak(Batch13 soak은 §7).

Batch14 상태: 최소 기준 8/8, 안정성 통과, 도전 목표 5/8(§88).

### 89.1 Batch14 30분 soak (c32-natural)

`smollm2-batch14-soak/` (계획 `soak-plan.json`, `completion.json`, `host-samples.json`, `drains.json`, 사후 계산한 `soak-summary.json`). 안정성 controller와 같은 script, 지속 시간만 1800초.

| 항목 | Batch14 | Batch13 soak (§7) |
|---|---:|---:|
| 지속 시간 / 요청 / exact | 1800.5 s / 312,192 / 312,192 | 1802.2 s / 306,048 / 306,048 |
| 처리량 6등분 평균 (tok/s) | 13064 / 13036 / 13032 / 13020 / 13043 / 13053 | 첫 12772, 끝 12883 |
| round 처리량 최소 | 12356 | 11128 |
| TPOT p50 6등분 평균 (ms) | 2.270 / 2.274 / 2.276 / 2.276 / 2.275 / 2.273 | 첫 2.318, 끝 2.305 |
| RSS 6등분 평균 (MiB) | 769.5 / 770.3 / 770.6 / 770.8 / 770.9 / 771.1 | 760.9 … 762.1 |
| RSS 기울기 전체 / 후반 (MiB/min) | 0.053 / 0.027 | 0.045 / 0.021 |
| GPU 메모리 (MiB) | 1966 고정 | 1938 고정 |
| drain 84회 / 종료 회수 | 모두 0 / device·pinned 할당 0, exit 0 | 동일 |

요청 오류 0, 처리량·TPOT 추세 변화 없음, RSS 증가는 30분에 약 1.5 MiB로 후반에 둔화(Batch13과 같은 형태). GPU 메모리는 같은 조건의 정식 측정 lane 최대치(Batch13 1956, Batch14 1966 MiB)와 같은 수준이며 차이 10 MiB는 capture 추가분입니다(Batch13 soak의 1938은 다른 날의 실행 값).

## 90. SmolLM2 Batch14 2차 후보: prefill kernel 정리와 1행 가중치 prefetch (2026-10-11)

§88 이후 Batch14 binary의 nsys 프로파일(`smollm2-batch14-diagnostics/profile/`, 진단용)을 새로 떠서 남은 TPOT 격차를 봤습니다.

- c32-fixed는 kernel 시간의 약 74%가 prefill 동반 step의 kernel(`gemm_prefill_shape_vector` 39%, `gate_up` 15%, `mapped_attention` 11%, `mixed_rope_kv_v7` 5%, `norm_rows` 4%)입니다.
- `mixed_rope_kv_v7`는 C1에서 1.8 µs, C32에서 8.0 µs였습니다. 모든 thread가 행의 소유 요청을 순차 탐색하기 때문입니다.
- C1 decode는 층당 kernel 8개가 대부분 지연(latency) 한계에 있고, gate/up dispatch만 가중치 3.5 MB 읽기를 기다립니다.

변경(브랜치 `codex/serving-kernel-batch14` `61743e21`…`647496ca`, patch는 `smollm2-batch14-gatev1-attempt02/source/`):

| 변경 | 진단 측정 (교차 순서, 단일 lane) | 판정 |
|---|---|---|
| prefill RoPE의 소유 요청 탐색을 warp ballot 한 번으로 (lane마다 요청 하나 검사, 가장 낮은 것 선택) | c32-fixed TPOT 2.702 → 2.589 (−4.2%, 8회), c32-natural −1.4% | 채택 |
| prefill Q·K·V projection을 한 dispatch로 (warp별 계산은 그대로) | c32-fixed 2.588 → 2.394 (−7.5%), c32-natural −4.1%, c8-fixed −6.8%, C1 TTFT 3.08 → 2.69 (8회) | 채택 |
| 1행 전용 merge-norm + gate/up 융합 | c1-fixed 0.809 → 0.807 | **기각** — 효과 없음 (`rejected/one-row-merge-norm-gate-fusion.patch`) |
| gate/up 가중치 prefetch: score·AV·첫 norm launch에 가중치를 cache line당 한 word씩 읽기만 하는 block 추가 | c1-fixed 0.811 → 0.799 (−1.5%) | 채택 (1행 capture만) |
| o-projection·다음 층 Q 가중치 prefetch | c1-fixed 0.799 → 0.796 | 채택 (1행 capture만) |
| 위 prefetch를 모든 capture에 적용 | c8-fixed +1.1%, c8-natural +0.8%, c32-natural +1.5% (6회) | **기각** — 1행 capture로 한정 |

prefetch 근거 microbenchmark(`l2_prefetch*.cu`): 직전에 가중치를 읽어 두면 projection형 kernel이 5.1 → 4.0 µs. graph의 병렬 가지로 넣으면 fork/join 비용으로 오히려 +0.3–0.6 µs(기각), 앞선 kernel 하나에 전부 얹으면 그 kernel이 길어져 이득 없음, 여러 kernel에 나눠 얹으면 −1.0 µs.

검증(3차 시도, 호스트 commit `896a38e2` = 로컬 tree `e6b377dc`, binary `32004b42…`): native 3161건 로그 부분 SHA 동일 + prefetch block을 붙인 merge-norm 24건, prefill QKV 69건, prefill RoPE 45건(빈 구간·무효 owner 표 포함), 융합 QKV 270건, 좁은 grid attention 4800건(홀수 seed는 prefetch block 포함) 모두 bit 일치. HTTP exact screen 8조건 60건 일치.

### 90.1 정식 측정 2차 (gate v1, 96 lane) — `smollm2-batch14-gatev1-attempt02/`

1차(§88)와 같은 controller·계획·순서·반복 수·Batch13/vLLM binary, 후보만 교체한 별도 attempt(계획은 측정 전 고정, 병합 없음). 96 lane 전부 완료, 요청 오류 0, Riley 전 요청 reference 일치, 제외 없음, controller 요약과 독립 재계산 불일치 0.

| cell | engine | tok/s (sd) | TTFT p50/p95/p99 ms | TPOT p50 (sd) /p95/p99 ms | E2E p50/p95/p99 ms | err | GPU MiB | RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c1-fixed | batch13 | 996.4 (0.6) | 3.07/3.25/3.57 | 0.8823 (0.0003) /0.888/0.910 | 20.9/30.6/30.8 | 0 | 1152 | 754 |
| c1-fixed | candidate | 1109.8 (1.1) | 2.69/2.85/3.08 | 0.7945 (0.0004) /0.805/0.822 | 18.5/27.5/27.9 | 0 | 1158 | 762 |
| c1-fixed | vllm | 768.5 (22.2) | 7.89/13.87/17.36 | 0.8881 (0.0016) /1.056/1.223 | 27.5/38.3/47.6 | 0 | 7984 | 3707 |
| c1-natural | batch13 | 1016.9 (1.1) | 3.16/5.74/6.03 | 0.8836 (0.0008) /0.999/1.004 | 58.9/132.6/133.5 | 0 | 1172 | 754 |
| c1-natural | candidate | 1116.1 (1.3) | 2.77/5.45/5.80 | 0.7997 (0.0008) /0.914/0.919 | 53.2/121.5/122.4 | 0 | 1178 | 762 |
| c1-natural | vllm | 884.3 (1.2) | 7.96/12.24/17.00 | 1.0137 (0.0033) /1.085/1.150 | 71.8/143.3/152.0 | 0 | 8006 | 3002 |
| c8-fixed | batch13 | 5238.2 (64.6) | 4.83/6.87/9.17 | 1.3016 (0.0342) /1.461/1.529 | 30.7/48.5/50.5 | 0 | 1180 | 754 |
| c8-fixed | candidate | 5820.7 (28.0) | 4.15/6.02/9.47 | 1.1797 (0.0098) /1.291/1.340 | 28.0/43.1/44.5 | 0 | 1186 | 762 |
| c8-fixed | vllm | 3098.8 (59.5) | 13.66/22.27/32.25 | 1.8555 (0.0254) /2.597/3.138 | 51.5/83.5/99.8 | 0 | 8034 | 2999 |
| c8-natural | batch13 | 5831.6 (13.8) | 7.27/12.49/13.81 | 1.2668 (0.0041) /1.327/1.355 | 86.9/175.1/177.3 | 0 | 1328 | 754 |
| c8-natural | candidate | 6256.0 (36.6) | 6.59/11.14/14.11 | 1.1844 (0.0123) /1.234/1.268 | 80.9/163.0/165.0 | 0 | 1334 | 762 |
| c8-natural | vllm | 4468.3 (63.9) | 11.62/20.38/31.45 | 1.5713 (0.0141) /1.894/2.205 | 111.1/231.7/267.2 | 0 | 8034 | 3016 |
| c16-fixed | batch13 | 7680.5 (183.1) | 6.04/9.67/22.62 | 1.7991 (0.0790) /2.042/2.207 | 39.7/65.6/70.5 | 0 | 1208 | 756 |
| c16-fixed | candidate | 8675.6 (40.7) | 5.37/8.61/20.29 | 1.5862 (0.0203) /1.763/1.878 | 35.1/57.4/62.0 | 0 | 1214 | 764 |
| c16-fixed | vllm | 4694.6 (253.6) | 19.96/35.40/59.16 | 2.2981 (0.0707) /3.333/4.439 | 65.1/111.5/138.1 | 0 | 7986 | 3007 |
| c16-natural | batch13 | 9437.2 (67.6) | 8.12/14.08/22.94 | 1.5541 (0.0196) /1.671/1.768 | 106.1/216.8/222.5 | 0 | 1508 | 756 |
| c16-natural | candidate | 9907.8 (151.2) | 7.18/13.53/30.06 | 1.4876 (0.0323) /1.585/1.648 | 100.9/205.6/213.1 | 0 | 1514 | 764 |
| c16-natural | vllm | 7665.2 (228.8) | 14.07/27.68/41.61 | 1.8153 (0.0242) /2.175/2.515 | 129.2/264.2/309.5 | 0 | 7986 | 3533 |
| c32-fixed | batch13 | 10088.9 (42.4) | 7.68/22.77/44.15 | 2.7380 (0.0351) /3.172/3.613 | 59.0/101.5/117.2 | 0 | 1264 | 759 |
| c32-fixed | candidate | 11411.8 (131.9) | 6.54/23.09/41.82 | 2.4291 (0.0327) /2.701/3.150 | 52.7/86.7/107.6 | 0 | 1270 | 767 |
| c32-fixed | vllm | 6836.6 (119.5) | 33.82/57.95/66.24 | 2.8951 (0.1153) /5.096/5.443 | 87.3/148.8/170.2 | 0 | 8036 | 3021 |
| c32-natural | batch13 | 12811.1 (80.4) | 9.52/27.81/61.41 | 2.3119 (0.0192) /2.476/2.760 | 155.4/315.3/347.2 | 0 | 1868 | 759 |
| c32-natural | candidate | 13770.8 (83.8) | 8.71/21.11/55.80 | 2.1460 (0.0215) /2.335/2.657 | 143.8/292.8/323.5 | 0 | 1874 | 767 |
| c32-natural | vllm | 11353.7 (561.7) | 22.67/54.84/79.49 | 2.3888 (0.1261) /2.737/3.120 | 174.9/354.7/374.9 | 0 | 8036 | 3038 |

| cell | cand/vLLM tok/s | TTFT p50 | TPOT p50 | worst TTFT/TPOT/E2E P95,P99 ratio | minimum | challenge | cand/Batch13 tok/s | TPOT p50 | worst-repeat cand TPOT vs best vLLM |
|---|---:|---:|---:|---:|:-:|:-:|---:|---:|---:|
| c1-fixed | 1.444 | 0.341 | 0.895 | 0.763 | pass | pass | 1.114 | 0.900 | 0.897 |
| c1-natural | 1.262 | 0.348 | 0.789 | 0.848 | pass | pass | 1.098 | 0.905 | 0.793 |
| c8-fixed | 1.878 | 0.304 | 0.636 | 0.516 | pass | pass | 1.111 | 0.906 | 0.654 |
| c8-natural | 1.400 | 0.567 | 0.754 | 0.703 | pass | pass | 1.073 | 0.935 | 0.774 |
| c16-fixed | 1.848 | 0.269 | 0.690 | 0.529 | pass | pass | 1.130 | 0.882 | 0.709 |
| c16-natural | 1.293 | 0.511 | 0.820 | 0.778 | pass | pass | 1.050 | 0.957 | 0.855 |
| c32-fixed | 1.669 | 0.193 | 0.839 | 0.633 | pass | pass | 1.131 | 0.887 | 0.906 |
| c32-natural | 1.213 | 0.384 | 0.898 | 0.863 | pass | pass | 1.075 | 0.928 | 0.961 |

lanes 96; request errors 0; Riley reference all-match True; nonzero exits []; controller-summary mismatches 0; host gate records 288

판정:

- **최소 기준: 8/8 통과.**
- **도전 목표: 8/8 통과** (처리량 ≥ 1.15, TTFT p50 ≤ 0.90, TPOT p50 ≤ 0.90). 가장 얇은 곳은 c32-natural TPOT 0.898과 c1-fixed TPOT 0.895입니다.
- 여유가 얇다는 점: "후보 최악 반복 vs vLLM 최고 반복"으로 보면 c32-fixed 0.906, c32-natural 0.961로 0.90을 넘습니다(평균 기준으로는 통과). vLLM의 C32 TPOT 반복 편차(sd 0.12–0.13 ms)가 후보(0.02–0.03)보다 커서, vLLM이 빠르게 나온 회차를 기준으로 하면 c32-natural은 미달입니다. 1차 측정의 vLLM c32-natural TPOT(2.4497)로 환산하면 0.876, 이번 값(2.3888)으로는 0.898입니다.
- Batch13 대비: 처리량 +5.0–13.1%, TPOT −4.3–11.8%, C1 TTFT −12%. GPU 메모리 +6 MiB, RSS +8 MiB.
- TTFT p99가 Batch13보다 높은 칸이 둘 있습니다: c8-fixed 9.47 vs 9.17, c16-natural 30.06 vs 22.94 ms(vLLM 41.61보다는 낮음). c8-natural p99도 14.11 vs 13.81. 384표본 lane의 p99는 상위 4개 표본이라 편차가 크지만, 개선이 아닌 것은 사실입니다.

안정성과 soak은 이 binary로 다시 실행합니다(§91).

## 91. SmolLM2 Batch14 2차 후보 안정성과 soak (2026-10-11)

`smollm2-batch14b-stability/`, `smollm2-batch14b-soak/`. §89와 같은 controller, matched attempt만 §90.1의 정식 측정(binary `32004b42…`)으로 변경.


지속 부하 300s, idle 취소 20×2(early/decode, 매회 전체 corpus exact 재요청), 부하 중 취소 40회. 제외 없음.

| 조건 | 엔진 | 결과 | 지속 요청 | 오류 | exact | 처리량 첫/끝 round | TPOT p50 첫/끝 | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | RSS 기울기 MiB/min | drain 횟수(최대 ms) | 취소(idle/부하) | 종료 회수 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-fixed | candidate | 통과 | 16896 | 0 | 16896 | 1110.4/1109.2 | 0.794/0.796 | 1158/1158 | 762.3/762.4 | 0.025 | 84 (20.7) | 40/40 | True |
| c1-natural | candidate | 통과 | 4608 | 0 | 4608 | 1115.7/1116.9 | 0.800/0.799 | 1178/1178 | 762.2/762.3 | 0.046 | 84 (20.8) | 40/40 | True |
| c8-fixed | candidate | 통과 | 87552 | 0 | 87552 | 5821.0/5774.3 | 1.184/1.207 | 1186/1186 | 762.5/762.8 | 0.074 | 84 (20.7) | 40/40 | True |
| c8-natural | candidate | 통과 | 25344 | 0 | 25344 | 6228.8/6315.2 | 1.191/1.174 | 1334/1334 | 762.4/762.6 | 0.057 | 84 (21.0) | 40/40 | True |
| c16-fixed | candidate | 통과 | 129408 | 0 | 129408 | 8454.6/8877.2 | 1.638/1.525 | 1214/1214 | 764.3/764.9 | 0.152 | 84 (20.7) | 40/40 | True |
| c16-natural | candidate | 통과 | 40320 | 0 | 40320 | 9843.4/9944.3 | 1.506/1.489 | 1514/1514 | 764.2/764.7 | 0.123 | 84 (20.7) | 40/40 | True |
| c32-fixed | candidate | 통과 | 171648 | 0 | 171648 | 11583.6/11556.0 | 2.402/2.434 | 1270/1270 | 768.0/768.8 | 0.209 | 84 (20.7) | 40/40 | True |
| c32-natural | candidate | 통과 | 54912 | 0 | 54912 | 13661.8/13673.4 | 2.168/2.170 | 1874/1874 | 768.3/769.2 | 0.248 | 84 (20.7) | 40/40 | True |
| c32-natural | vllm | 실패: host gate timeout cancel-under-load | — | 0 | — | —/— | —/— | —/— | —/— | — | — (—) | —/— | None |
| c32-fixed | vllm | 미실행 | | | | | | | | | | | |

전체 lane 통과: False

한계: early cancellation closes right after the request write; the server-side phase (queued/prefill) is not observable from the client; memory is sampled once per second; peaks between samples are not proven; no performance qualification

- 후보 8 lane 모두 통과: 요청 오류 0, 전 요청 exact, drain 84회 모두 0, 종료 회수 확인, GPU 메모리 구간 내 변화 없음, RSS 기울기 0.03–0.25 MiB/min.
- vLLM c32-natural 참고 lane은 엔진 오류가 아니라 "부하 중 취소" 단계 앞 host gate timeout으로 실패했습니다. gate 기록의 마지막 표본: io full avg10 9.6–10.6(기준 5), memory full avg10 1.7(기준 0.5) — 공유 호스트의 다른 작업에 의한 pressure(외부 제약). 그 lane의 요청 오류는 0. 실행기가 첫 실패에서 멈춰 vLLM c32-fixed는 미실행. 실패 lane 기록은 `failed-vllm-c32-natural-attempt01/`에 보존. vLLM 두 lane은 통과 조건이 아닌 참고용이며 별도 attempt 02로 재실행합니다(§91.2).

### 91.1 30분 soak (c32-natural, 후보)

| 항목 | Batch14 2차 후보 | Batch14 1차 (§89.1) |
|---|---:|---:|
| 지속 시간 / 요청 / exact / 오류 | 1800.2 s / 329,472 / 329,472 / 0 | 1800.5 s / 312,192 / 312,192 / 0 |
| 처리량 6등분 평균 (tok/s) | 13810 / 13802 / 13756 / 13753 / 13748 / 13758 | 13064 … 13053 |
| round 처리량 최소 | 13193 | 12356 |
| TPOT p50 6등분 평균 (ms) | 2.145 / 2.147 / 2.156 / 2.155 / 2.156 / 2.155 | 2.270 … 2.273 |
| TPOT p99 최대 (round) | 2.831 | 3.135 |
| RSS 6등분 평균 (MiB, 전체 표본) | 764.9 / 769.7 / 770.0 / 770.2 / 770.4 / 769.7 | 769.5 … 771.1 |
| RSS 기울기 전체 / 후반 (MiB/min) | 0.148 / −0.070 | 0.053 / 0.027 |
| GPU 메모리 (MiB, 부하 구간) | 1874 고정 | 1966 고정 |
| drain 84회 / 종료 시 할당 | 모두 0 / 0, exit 0 | 동일 |

처리량은 첫 1/3 구간 이후 0.4% 낮아진 뒤 평탄하고(TPOT +0.5%), 이후 추세는 없습니다. RSS의 첫 구간 값이 낮고 전체 기울기가 큰 것은 표본에 시작 대기 구간이 포함됐기 때문이며(GPU 표본 최솟값 948 MiB도 같은 구간), 2번째 구간 이후는 769.7–770.4로 평탄하고 후반 기울기는 음수입니다. GPU 메모리가 1차 soak(1966)보다 낮은 것은 정식 측정 lane에서도 같은 차이(1874 vs 1966)가 보이며 원인은 확인하지 않았습니다(Batch13도 1차 1956 → 2차 1868로 함께 낮아져, 후보가 아니라 실행 시점의 호스트 상태 차이로 추정).

### 91.2 vLLM 참고 lane 재실행 (attempt 02, 같은 계획·script)

호스트 pressure가 기준 안으로 내려온 뒤 두 lane만 다시 실행했습니다(`smollm2-batch14b-stability/vllm-*-attempt02/`). 둘 다 완료: 요청 오류 0, exit 0, 종료 후 GPU 프로세스 없음·port 닫힘.

| 조건 | 지속 요청 | 오류 | 처리량 첫/끝 round | TPOT p50 첫/끝 | GPU 최대 MiB | RSS 첫¼/끝¼ MiB |
|---|---:|---:|---:|---:|---:|---:|
| c32-natural vLLM | 45312 | 0 | 10949.4 / 11616.5 | 2.465 / 2.378 | 8036 | 2647 / 3071 |
| c32-fixed vLLM | 100224 | 0 | 7008.8 / 7129.6 | 2.852 / 2.832 | 8036 | 2772 / 3063 |

(RSS 값은 사후 계산한 전체 표본 기준이라 시작 구간이 포함됩니다. 참고용이며 통과 조건이 아닙니다.)

Batch14 2차 후보 상태: 최소 기준 8/8, 도전 목표 8/8(§90.1, 여유가 얇은 칸은 c32-natural TPOT 0.898·c1-fixed 0.895), 안정성 8/8, 30분 soak 통과.

## 92. Qwen2.5-3B 원래 기준 재추진 — 재진단 (2026-10-11 저녁, 사용자 지시)

사용자가 §86의 조정을 되돌리고 Qwen의 원래 기준(8조건 전부 처리량 ≥ vLLM, TTFT/TPOT ≤ vLLM, P95/P99 회귀 없음)을 다시 추진하라고 지시했습니다. 채택 binary(`102bd4de…`, head f8266c6a)를 **채택 flag 전체**로 다시 프로파일했습니다(`qwen-resume-diagnostics/`, 호스트 `qwen-profile-diagnostic-c1-attempt11`, `…-c32-attempt12`; nsys graph node 추적, 진단용). 처음 뜬 attempt10은 `--decode-step-graph`·`--lm-head-rows`가 빠진 구성이라 버렸습니다(보존).

### C1 decode step (kernel 합 7.74 ms, 비프로파일 TPOT 7.76 ms, vLLM 7.40 ms)

| 구성 | step당 µs | 비고 |
|---|---:|---|
| layer GEMM 5종 × 36 | 6235 | layer당 173 µs / 154 MB = 0.889 GB/ms |
| LM head GEMM | 697 | 622 MB, 0.89 GB/ms |
| attention producer | 228 | 6.35 µs × 36 |
| residual+RMSNorm | 180 | 2.5 µs × 72 |
| split-K reduce | 123 | 1.14 µs × 108 (anchor 알고리즘의 일부) |
| attention merge | 90 | |
| KV write | 76 | |
| QKV split+bias+RoPE | 63 | |
| SiLU·곱 | 47 | |
| argmax | 32 | 1 block이 151,936열을 순차 scan |
| host(step 사이 + metadata 업로드) | 약 80 | 작은 동기 memcpy 9회 ≈ 60 µs |

- GEMM이 step의 89.6%이고 가중치 읽기 속도에 묶여 있습니다. microbenchmark(`qwen_readahead.cu`): GEMM과 **동시에** 다음 가중치를 읽히면 46–86% 느려집니다 — bus가 이미 포화. vLLM도 같은 6.17 GB를 읽으므로 GEMM 구간은 동률이고, 격차 0.36 ms는 **GEMM 밖 시간**(Riley 약 0.83 ms vs vLLM 약 0.47 ms 추정)입니다.
- 가중치 미리 읽기(`qwen_prefetch*.cu`): graph 병렬 가지는 +0.4 µs(효과 없음). 틈 kernel 안의 추가 block은 읽기 속도가 약 0.5 GB/ms라 틈이 길어지는 만큼만 GEMM이 줄어, 순이득은 틈 시간의 약 15%(2 MB / 3.6 µs 틈에서 −0.5 µs)입니다. Qwen에서는 step당 −0.07~0.12 ms 수준.
- 프로파일의 "embedding 뒤 240 µs 공백"은 node 추적 중 `cudaGraphLaunch`가 210 µs 걸린 것으로, 비프로파일 TPOT(7.76)와 kernel 합(7.74)이 맞으므로 실제 유휴가 아닙니다.

C1에서 식별한 bit 불변 항목과 추정 효과(step당): KV write를 split kernel에 융합 −0.07 ms, producer+merge 단일화(짧은 문맥) −0.09, argmax 다중 block −0.03, metadata 업로드 묶기 −0.05, 틈 prefetch −0.07~0.12. **전부 해도 −0.31~0.36 ms로 TPOT 7.40~7.45 — vLLM(7.40)과 동률 근처이며 확실히 이기지는 못합니다.** 처리량은 TTFT 우위 덕에 c1-fixed 0.986 → 약 1.02, c1-natural 0.958 → 약 0.99로 추정합니다.

### C32-natural (비프로파일 TPOT 11.69 ms, vLLM 10.74 ms)

| step 종류 | 시간 비중 | 주기 | kernel busy | producer | step 내 공백 | step 간 공백 |
|---|---:|---:|---:|---:|---:|---:|
| decode 위주 (producer < 1.1 ms) | 69% | 9.25 ms | 8.68 ms | 0.69 ms | 0.35 | 0.22 |
| prefill 동반 | 31% | 16.6 ms | 15.7 ms | 3.15 ms | 0.65 | 0.28 |

(공백 수치는 프로파일러의 graph launch 지연이 섞여 있어 상한입니다.) vLLM의 decode step은 주기 8.8 ms(§29). 격차는 decode step의 GEMM 밖 시간과 prefill 동반 step의 attention producer(3.15 ms)입니다.

### 다음 순서

1. bit 불변 decode 항목(위 목록) — 모든 조건에 도움, 출력 불변.
2. prefill attention의 행렬 곱 경로(§85, key class 고정) — c32-natural +3% 추정, prompt 구간 수치가 바뀌므로 불변성·HF token 비교·안정성 재검증 필요.
3. 각 단계 뒤 8조건 정상상태 비교와 안정성.

현재 추정으로는 1·2를 모두 해도 C1 TPOT과 c32-natural은 vLLM과 동률 근처(±1–2%)이고, HF token 일치 11/15는 이 작업들로 바뀌지 않습니다.

## 93. Qwen 재추진 1: argmax scan을 16-byte load로 (commit 88671303, 채택)

`bf16_argmax_kernel`이 1행에서 logit 151,936개를 2 byte씩 읽던 것을, 16-byte 정렬 구간은 한 번에 8개씩 읽도록 바꿨습니다(정렬 전 머리와 꼬리는 종전대로 1개씩). 선택 규칙(최댓값, 동률이면 낮은 token, non-finite 검출)은 그대로입니다.

- 검증: `bf16_argmax_gpu` 4건(홀수 vocab 다행, 후보 상한, command batch 순서, 검증 실패 시 무변경) 통과, HF-exact HTTP gate 43 통과(128 token 일치), binary `79d103b9…`.
- 8조건 정상상태(attempt 51, warmup 32 / retained 192 × 2회, 채택 빌드 attempt 48 대비, vLLM은 기존 같은-조건 측정값): 출력 15/15 prompt 불변, 오류 0.

| 조건 | 처리량 | TPOT p50 (ms) | vLLM TPOT |
|---|---:|---:|---:|
| c1-fixed | 126 → 126 (+0.2%) | 7.76 → 7.75 | 7.40 |
| c1-natural | 126 → 127 (+0.2%) | 7.77 → 7.76 | 7.45 |
| c8-fixed | 809 → 809 | 8.84 → 8.85 | 9.10 |
| c8-natural | 863 → 864 | 8.75 → 8.74 | 8.96 |
| c16-fixed | 1333 → 1346 (+1.0%) | 10.53 → 10.48 | 10.27 |
| c16-natural | 1513 → 1515 | 9.62 → 9.59 | 9.63 |
| c32-fixed | 1926 → 1929 | 13.52 → 13.50 | 13.42 |
| c32-natural | 2363 → 2369 (+0.3%) | 11.64 → 11.65 | 10.79 |

효과는 추정(step당 −0.03 ms)보다 작은 약 −0.01 ms이며 잡음 수준입니다. 회귀는 없어 채택합니다.

같은 측정에서 다시 확인된 원래 기준 미달 항목: TPOT p50(C1 두 조건, c16-fixed, c32 두 조건), 처리량(C1 두 조건, c32-natural), TTFT p95/p99(c1-natural 31 vs 25–26 ms, C16·C32의 p99, C32의 p95).

`--metadata-transport packed-async`(metadata를 한 번에 업로드)는 `--decode-step-graph capture-v1`과 함께 쓸 수 없어 시작 단계에서 거절됩니다(attempt 52, 측정 없음). 동기 경로에서 업로드를 묶는 변경은 기대 이득이 0.3% 수준이라 뒤로 미룹니다.

## 94. Qwen 재추진 2: KV cache 기록을 QKV split kernel에 융합 (commit 6번째 항목, 채택)

`--kv-cache-write fused-split-v1`: 병합 QKV split(+bias+RoPE) 연산이 launch를 미루고, command batch 안에서 바로 뒤따르는 KV cache 기록 연산이 kernel 하나(`qkv_split_bias_rope_kv_write_kernel`)로 split 출력과 cache 복사를 함께 저장합니다. 두 연산의 검증·자원 보유 로직은 그대로이고, 미뤄진 split은 기록 연산이 stream·출력 buffer·head 크기를 대조한 뒤에만 실행됩니다(불일치 시 오류). command batch 시작·종료 시 미처리 기록은 지워집니다. 층당 kernel 1개 감소.

- 검증: `qkv_split_kv_write_gpu`(page 경계 15/16/17, padding 행 포함) — query·key·value 작업 buffer와 key·value pool 모두 차이 0. `qkv_split_rope_gpu` 통과. HF-exact HTTP gate 44 통과, binary `7e44f90a…`.
- 8조건 정상상태(attempt 53 vs argmax 빌드 attempt 51): 출력 15/15 prompt 불변, 오류 0.

| 조건 | 처리량 | TPOT p50 (ms) | vLLM TPOT | 처리량 Riley/vLLM |
|---|---:|---:|---:|---:|
| c1-fixed | 126 → 127 (+0.7%) | 7.75 → 7.69 | 7.40 | 0.98 |
| c1-natural | 127 → 127 (+0.7%) | 7.76 → 7.71 | 7.45 | 0.96 |
| c8-fixed | 809 → 817 (+0.9%) | 8.85 → 8.76 | 9.10 | 1.08 |
| c8-natural | 864 → 869 (+0.6%) | 8.74 → 8.69 | 8.96 | 1.04 |
| c16-fixed | 1346 → 1354 (+0.6%) | 10.48 → 10.44 | 10.27 | 1.05 |
| c16-natural | 1515 → 1526 (+0.7%) | 9.59 → 9.54 | 9.63 | 1.02 |
| c32-fixed | 1929 → 1930 | 13.50 → 13.34 | 13.42 | 1.02 |
| c32-natural | 2369 → 2376 (+0.3%) | 11.65 → 11.56 | 10.79 | 0.95 |

추정(C1 step당 −0.07 ms)과 실측(−0.05~0.06 ms)이 맞습니다. c32-fixed TPOT p50이 vLLM 이하로 내려왔습니다(13.34 vs 13.42, vLLM 값은 기존 같은-조건 측정). 남은 TPOT p50 미달: C1 두 조건(+0.26~0.29 ms), c16-fixed(+0.17), c32-natural(+0.77). TTFT p95/p99 열세(c1-natural p95, C16·C32 tail)는 변화 없습니다.

## 95. Qwen 재추진 3: global-max merge의 reduction을 shuffle로 — 미채택 (효과 없음)

merge kernel의 최댓값·분모 reduction을 "shared memory 2단 + warp 0 shuffle 5단"으로 바꿔 동기화 장벽을 16회에서 6회로 줄였습니다(짝짓기 (d, d+offset)는 동일). 기존 ragged two-stage GPU 테스트에 merge 출력 digest(`0x58b5d6a7240b930f`)를 먼저 고정했고, 변경 후에도 같은 값(bit 일치)이었습니다. HF-exact gate 45 통과, binary `4d717fa8…`.

8조건 정상상태(attempt 54 vs KV 융합 빌드 attempt 53): 출력 15/15 불변, 오류 0. 처리량 변화 −0.2~+0.3%, TPOT p50 변화 −0.04~+0.08 ms로 잡음 수준이라 **채택하지 않고 kernel 변경을 되돌렸습니다**(patch는 `qwen-matrix/resume-mergefold/merge-reduction-shuffle.patch`에 보존). digest 고정 테스트는 남겼습니다(commit `test(cuda): pin the global-max merge output bytes`).

현재 Qwen 채택 후보: KV 융합 빌드(commit 5dcd72cf, binary `7e44f90a…`, `--kv-cache-write fused-split-v1` 추가). 원래 기준 대비 남은 미달(§94 표): TPOT p50 — C1 두 조건 +0.26~0.29 ms, c16-fixed +0.17, c32-natural +0.77; 처리량 — C1 두 조건 0.96~0.98, c32-natural 0.95; TTFT tail — c1-natural p95(31 vs 25 ms), C16 p99, C32 p95·p99(232~287 vs 160~202, 259~377 vs 170~245 ms).

## 96. Qwen 현재 후보 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final8, commit 5dcd72cf, binary `7e44f90a…`)

`qwen-matrix/steady-final8/`. final7(§80)과 같은 script에 `--kv-cache-write fused-split-v1`만 추가. 8조건 × 2회, 순서 [Riley, vLLM] ↔ [vLLM, Riley], warmup 32 / retained 192, host gate 없음, 오류 0, 제외 없음. 값은 2회 평균(Riley / vLLM).

| 조건 | 처리량 tok/s | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | E2E p50 / p99 ms | GPU MiB |
|---|---|---|---|---|---|
| c1-fixed | 127.1 / 129.5 | 10.6 / 11.9 / 12.2 vs 13.1 / 18.0 / 20.9 | 7.70 / 7.71 / 7.73 vs 7.40 / 7.42 / 7.43 | 164.6 / 249.9 vs 158.0 / 247.3 | 7198 / 15186 |
| c1-natural | 127.4 / 131.9 | 10.9 / 30.9 / 31.4 vs 14.0 / 25.4 / 32.2 | 7.71 / 7.75 / 7.76 vs 7.44 / 7.45 / 7.45 | 496.5 / 1016.5 vs 482.3 / 971.7 | 7328 / 15152 |
| c8-fixed | 815.8 / 761.5 | 21.6 / 32.5 / 57.7 vs 30.5 / 39.7 / 56.4 | 8.77 / 9.17 / 9.40 vs 8.98 / 9.39 / 9.58 | 188.0 / 310.4 vs 204.9 / 323.1 | 7288 / 15152 |
| c8-natural | 868.8 / 834.5 | 22.2 / 45.2 / 75.6 vs 32.7 / 50.1 / 71.3 | 8.67 / 8.74 / 8.90 vs 8.95 / 9.04 / 9.39 | 565.7 / 1158.1 vs 591.6 / 1195.1 | 7622 / 15192 |
| c16-fixed | 1350.2 / 1244.0 | 22.1 / 65.6 / 117.2 vs 36.8 / 102.2 / 215.7 | 10.42 / 10.72 / 11.08 vs 10.24 / 12.22 / 18.03 | 231.9 / 405.0 vs 260.6 / 483.8 | 7384 / 15287 |
| c16-natural | 1522.5 / 1496.2 | 30.0 / 64.5 / 177.8 vs 40.8 / 96.3 / 123.8 | 9.55 / 10.01 / 10.66 vs 9.62 / 9.79 / 9.90 | 625.9 / 1383.2 vs 646.1 / 1317.8 | 7934 / 15296 |
| c32-fixed | 1927.2 / 1871.1 | 40.4 / 224.2 / 257.0 vs 52.6 / 176.3 / 176.8 | 13.42 / 14.06 / 14.55 vs 13.53 / 15.25 / 18.58 | 309.6 / 626.7 vs 323.5 / 563.0 | 7556 / 15348 |
| c32-natural | 2382.2 / 2488.9 | 34.4 / 263.3 / 368.2 vs 61.8 / 196.1 / 242.8 | 11.56 / 13.90 / 14.74 vs 10.82 / 11.67 / 12.70 | 757.3 / 1824.4 vs 744.6 / 1554.5 | 8522 / 15316 |

원래 기준(처리량 ≥, TTFT·TPOT ≤, P95/P99 회귀 없음) 판정 — 미달 항목만:

| 조건 | 처리량 | TTFT | TPOT | 판정 |
|---|---|---|---|:-:|
| c1-fixed | 0.982 | — | p50·p95·p99 +4.1% | 미달 |
| c1-natural | 0.966 | p95 30.9 vs 25.4 | p50·p95·p99 +3.6~4.2% | 미달 |
| c8-fixed | — | p99 57.7 vs 56.4 | — | 미달 (p99 +1.3 ms) |
| c8-natural | — | p99 75.6 vs 71.3 | — | 미달 (p99) |
| c16-fixed | — | — | p50 10.42 vs 10.24 (+1.8%) | 미달 (TPOT p50) |
| c16-natural | — | p99 177.8 vs 123.8 | p95·p99 | 미달 |
| c32-fixed | — | p95·p99 224 / 257 vs 176 / 177 | — | 미달 (TTFT tail, E2E p99) |
| c32-natural | 0.957 | p95·p99 | p50·p95·p99 +6.8~19% | 미달 |

8개 조건 중 원래 기준을 모두 채우는 칸은 없습니다. 처리량은 5/8에서 vLLM 이상(c8 +4~7%, c16 +2~9%, c32-fixed +3%), TTFT p50은 8/8에서 vLLM보다 19~44% 낮고, GPU 메모리는 약 절반입니다. final7(§80) 대비 처리량 +0.5~1.2%, TPOT p50 −0.05~0.13 ms.

남은 격차의 성격:
- **C1 TPOT·처리량 (4% / 2~3%)**: 양쪽 모두 가중치 읽기 한계에 있고 차이는 GEMM 밖 시간입니다(§92). 식별한 bit 불변 항목 중 남은 것은 metadata 업로드 묶기(−0.02~0.05 ms)뿐이며, 격차 0.27~0.30 ms를 닫을 수단은 없습니다.
- **TTFT tail (C8 p99, C16-natural p99, C32 p95·p99, c1-natural p95)**: step당 token 예산 256에서 prompt가 여러 step으로 나뉘거나 대기하는 데서 옵니다. 예산 512(§27)는 tail을 15~20% 줄였지만 처리량·TPOT을 악화시켰고 여전히 vLLM보다 나빴습니다. 예산 128(§76)은 반대로 악화.
- **c32-natural 처리량·TPOT**: prefill attention 비용(§76 재분해). 남은 수단은 prompt 구간 수치를 바꾸는 행렬 곱 경로(§85, +3% 추정)로, 동률에 못 미칩니다.

## 97. prefill attention의 tensor-core(MMA) 전용 kernel — microbenchmark (미구현, 유망)

§96의 결론(c32-natural과 TTFT tail의 뿌리는 prefill attention 비용)에 따라, cuBLASLt 대신 **직접 쓴 MMA kernel**을 Qwen 형상(query head 16, KV head 2, D=128, 16-token page)으로 쟀습니다. `qwen-resume-diagnostics/prefill_mma{,2,3,4}.cu`. 값은 CPU 기준과 대조(최대 오차 0.0001~0.0012, BF16 반올림 수준).

구조: CTA 하나가 (16행 query tile, KV head, page 구간)을 맡고 warp 8개가 그 KV head의 query head 8개입니다. page마다 K tile과 전치한 V tile을 shared memory에 한 번 올리고(다음 page는 미리 읽어 지연을 겹침), QK는 m16n8k16 MMA 16회, softmax는 4-lane shuffle, PV는 확률을 그대로 A operand로 써서 MMA 16회입니다. 구간(4 page)별 partial state `[m, l, n[128]]`를 쓰고 merge kernel이 합칩니다.

**출력 불변성**: 각 출력 원소는 자기 행의 query와 자기 문맥의 key만으로, 절대 page 번호 순서로 계산됩니다(MMA의 출력 원소는 같은 tile의 다른 행과 독립). 그래서 어떤 행들이 한 launch에 묶이느냐에 결과가 의존하지 않고, §85의 cuBLASLt 경로가 필요로 했던 key class 고정이 필요 없습니다. (설계상 근거이며, 구현 후 8조건에서 실측 확인이 필요합니다.)

| prefill 형상 | 순차 page (1 kernel) | 구간 4 page + merge | 현재 경로 (§77, 근사) | cuBLASLt 경로 (§85) |
|---|---:|---:|---:|---:|
| 16행 @0 | 5.3 | 9.4 | 약 4 | 15.0 (2×16) |
| 28행 @0 | 6.5 | 9.8 | 약 8~18 | 26.1 (8×28) |
| 125행 @0 | 17.8 | 16.3 | 약 25 | 21.1 |
| 112행 @18 | 19.2 | 16.0 | 약 22 | 26.7 (2×112) |
| 256행 @0 | 31.6 | 26.6 | 약 71 (256×130) | — |
| 128행 @200 | 40.4 | 27.6 | 70.0 | 31.6 |
| 224행 @174 | 47.5 | 39.9 | 152.8 | 37.9 (174행) |
| 32행 @368 | 47.6 | 15.9 | 31.4 (32×400) | — |

- 순차 page 방식의 page당 1.9 µs 분해: QK MMA 0.79, PV MMA 0.56, tile 올리기 0.4, 나머지 0.35. warp 하나의 MMA 처리량이 한계라 page를 CTA로 나눠 병렬화(구간 방식)해야 긴 문맥에서 빨라집니다.
- 구간 방식은 문맥이 긴 chunk에서 현재 경로의 1/2~1/4, 16-token prompt에서는 merge 고정비 때문에 약 5 µs 느립니다(작은 chunk는 순차 방식으로 보내면 5~7 µs; 단 같은 token이 chunk 크기에 따라 다른 경로를 타면 안 되므로 "prompt 길이"처럼 요청 고유 값으로만 나눠야 함).
- natural(16 / 128 / 398 token) 요청 3건 기준 layer당 약 −130 µs(현재 대비), 36 layer로 약 −4.8 ms → c32-natural 처리량 0.957 → 약 1.00, TPOT 11.56 → 약 11.0 추정. fixed(약 125 token)는 요청당 약 −0.3 ms. prefill이 빨라지는 만큼 C32의 TTFT tail도 줄지만(약 10~15%) vLLM 수준(176~196 ms)까지는 부족할 것으로 봅니다. 모두 microbenchmark 기반 추정입니다.

구현에 필요한 것: (1) page table 간접 참조와 sequence별 행 범위를 받는 native 연산 + FFI + 검증, (2) dispatch에서 prefill 행과 decode 행을 나눠 각각의 attention 호출, step graph 기록과의 호환, (3) prompt 구간 수치가 바뀌므로 동시성 불변(15/15)·HF token 비교·안정성 재검증. 새 flag로 선택하게 하고 기본값은 유지합니다.

## 98. attention producer의 용량 크기 launch grid 비용 (microbenchmark)

step graph 기록 모드에서는 producer를 (page 용량 × KV head 2 × dense 행) grid로 띄웁니다(행의 실제 page 수를 넘는 CTA는 즉시 종료). natural(sequence 상한 1024 → page 용량 64)의 256행 step은 layer마다 CTA 32,768개입니다. 즉시 종료하는 256-thread CTA만의 비용을 graph 재생으로 쟀습니다(`qwen-resume-diagnostics/empty_cta.cu`).

| 형상 | grid CTA 수 | launch당 |
|---|---:|---:|
| natural 혼합: 용량 64, 256행 | 32,768 | 16.0~16.8 µs |
| 같은 형상, page 범위 32 | 16,384 | 8.5~8.8 |
| 같은 형상, page 범위 16 | 8,192 | 4.7 |
| natural decode C32: 용량 64, 32행 | 4,096 | 2.9 |
| 같은 형상, page 범위 16 | 1,024 | 1.5 |
| fixed 혼합: 용량 16, 256행 | 8,192 | 4.7 |
| C1 decode | 32~128 | 0.8~0.9 |

natural의 prefill 동반 step에서 layer당 약 16 µs(36 layer로 약 0.6 ms, step의 3.5%)가 빈 CTA를 띄우는 비용입니다. graph key에 page 범위 bucket을 넣으면 그중 절반 정도(c32-natural 전체로 약 1%)를 회수할 수 있지만 기록 가능한 graph 수(상한 32)를 더 쓰게 됩니다. §97의 MMA 경로는 prefill 행을 (16행 tile × KV head × 4-page 구간) grid로 처리하므로 이 비용의 대부분이 함께 사라집니다(256행·25 page 기준 CTA 224개). 따라서 별도 bucket 작업보다 §97 구현을 우선합니다.

## 99. Qwen 재추진 4: prompt token attention의 tensor-core producer (`--prompt-attention tensor-core-v1`, 후보 채택)

§97의 kernel을 runtime에 넣었습니다(commit 79af7332 native, 이후 runtime 연결 commit; binary `56a9c72e…`, HF-exact gate 51 통과).

- native: `native_bf16_prefill_d128_mma_partial_state_kernel` — CTA = (16행 tile, KV head, 4-page 구간), warp 8개 = query head 8개, page마다 K tile과 전치 V tile을 shared memory에 올림. 행 종류(U32, 1 = tensor-core 경로를 타는 prompt token)는 row position buffer의 뒤쪽에 둡니다. 기존 shared-KV producer는 그런 행을 건너뛰고, global merge는 그 행의 상태를 구간 수만큼 합칩니다. 다른 행의 경로와 bytes는 그대로입니다.
- native 테스트(`prompt_mma_attention_gpu`, 78행): prompt가 아닌 행은 기존 경로와 bit 동일. prompt 행은 다른 순서·tile 위치로 재배열(78행)하거나 prompt를 chunk 4개로 나눠 따로 계산(75행)해도 bit 동일. 기존 경로 대비 차이 최대 0.004(원소의 89%는 bit 동일). 기존 attention 테스트와 merge digest 불변.
- runtime: prefill 행 중 **자기 위치가 256 이상인 token**만 이 경로로 보냅니다. 위치는 요청에 고유하므로 같은 token은 부하·chunk와 무관하게 같은 경로를 탑니다. 그런 행이 없는 step은 행 종류를 올리지 않고 새 kernel도 띄우지 않으며, step graph는 그런 행의 유무로 따로 기록합니다.

진단 변형 6개(8조건 정상상태, 직전 후보 = KV 융합 빌드 attempt 53 대비 처리량 변화 %; raw `qwen-matrix/resume-prompt-mma/`):

| 변형 | 경로 선택 | c1-f | c1-n | c8-f | c8-n | c16-f | c16-n | c32-f | c32-n | 출력이 바뀐 prompt | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (attempt 55) | 모든 prompt 행, 매 step launch | −0.5 | −0.3 | −1.4 | +0.3 | −1.4 | +1.0 | 0.0 | +2.4 | 4 / 15 | 기각: decode step에서도 launch |
| 2 (56) | 모든 prompt 행, prefill 있는 step만 | 0.0 | +0.1 | −1.0 | +0.7 | −1.4 | +1.4 | +0.3 | +2.8 | 4 / 15 | 기각: fixed 회귀 |
| 3 (57) | 위치 ≥ 128, 해당 행 있는 step만 | −0.1 | +0.1 | −0.6 | +0.9 | −0.4 | +1.6 | −1.5 | +2.5 | 1 / 15 | 기각: fixed 회귀 |
| 4 (58) | 위치 ≥ 128, prefill 있는 step만 | −0.1 | +0.1 | −0.7 | +0.8 | −0.8 | +1.4 | −0.6 | +2.2 | 1 / 15 | 기각: fixed 회귀 |
| 5 (59) | 4 + graph 기록 한도 64 | −0.1 | +0.1 | −0.7 | +0.8 | −1.1 | +1.5 | −1.2 | +2.4 | 1 / 15 | 기각: 한도 가설 틀림(메모리도 불변) |
| **6 (60)** | **위치 ≥ 256, 해당 행 있는 step만** | 0.0 | +0.1 | −0.1 | +0.7 | −0.2 | +1.1 | +0.2 | +1.9 | 1 / 15 | **후보** |

- 모든 변형에서 15개 prompt 각각이 8조건·반복 전체에서 출력 1종(동시성 불변 유지), 오류 0.
- fixed 회귀의 원인: fixed prompt(평균 약 126 token) 중 128을 조금 넘는 것의 마지막 몇 행이 tensor-core 경로로 가면, 몇 행뿐이어도 문맥 전체(약 9 page)를 처리하는 고정비(layer당 약 8 µs)가 들었습니다. 기준을 256으로 올리자 fixed는 기준 후보와 같은 실행 경로가 되어 회귀가 사라졌습니다.
- 변형 6의 TPOT p50: c8-natural 8.69 → 8.63, c16-natural 9.54 → 9.42, c32-natural 11.56 → 11.35 ms. C32 TTFT p95/p99: natural 287/377 → 271/353, fixed 232/266 → 226/252 ms.
- 출력이 바뀐 prompt는 398-token natural prompt 1개입니다(위치 256 이후 token의 attention 반올림이 달라짐). HF token 일치율의 변화는 §100에서 다시 잽니다.

microbenchmark 추정(c32-natural +4.5%)보다 실측(+1.9~2.8%)이 작습니다. c32-natural은 vLLM 대비 0.957 → 약 0.97(기존 vLLM 측정값 기준)로, 아직 미달입니다.

## 100. 후보(변형 6)의 HF token 비교 재실행 (attempt 06)

`qwen-hf-token-reference/attempt06-prompt-mma/`. §82와 같은 script에 `--kv-cache-write fused-split-v1 --prompt-attention tensor-core-v1`만 추가(binary `56a9c72e…`).

| 조건 | 일치 | 불일치 (prompt, 첫 불일치 index, 길이) |
|---|---:|---|
| C1 (순차) | 11 / 15 | arithmetic 14/24, planning 25/32, sql 4/16, natural-398 69/128 |
| C32 (동시) | 44 / 60 | 같은 4개 prompt × 4회 |

§82(attempt 05)와 같은 11/15, 같은 4개 prompt, 같은 첫 불일치 위치입니다. prompt 구간의 attention 반올림을 바꿨지만 HF 일치율은 변하지 않았고, C1과 C32의 결과가 같습니다(동시성 불변).

## 101. Qwen 후보(변형 6) vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final9, commit 9cf4dcac, binary `56a9c72e…`)

`qwen-matrix/steady-final9/`. final8(§96)과 같은 script에 `--prompt-attention tensor-core-v1` 추가. 8조건 × 2회, 순서 교대, warmup 32 / retained 192, host gate 없음, 오류 0, 제외 없음. Riley 15개 prompt 모두 출력 1종. 측정 중 공유 호스트의 IO pressure가 높았습니다(full avg300 8~12); 반복 간 처리량 차이는 Riley 최대 0.6%, vLLM 최대 0.8%였습니다. 값은 2회 평균(Riley / vLLM).

| 조건 | 처리량 tok/s | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | E2E p50 / p99 ms | GPU MiB |
|---|---|---|---|---|---|
| c1-fixed | 127.2 / 129.7 | 10.6 / 11.9 / 12.1 vs 13.2 / 16.1 / 22.2 | 7.70 / 7.71 / 7.74 vs 7.40 / 7.42 / 7.43 | 159.2 / 249.9 vs 158.3 / 244.6 | 7198 / 15186 |
| c1-natural | 127.5 / 131.8 | 10.9 / 29.0 / 29.6 vs 13.3 / 25.2 / 26.2 | 7.71 / 7.75 / 7.77 vs 7.44 / 7.45 / 7.45 | 496.5 / 1014.7 vs 482.3 / 971.9 | 7328 / 15152 |
| c8-fixed | 814.5 / 761.2 | 21.5 / 32.9 / 59.2 vs 31.5 / 37.9 / 51.9 | 8.80 / 9.18 / 9.41 vs 9.01 / 9.37 / 9.54 | 189.7 / 310.8 vs 201.8 / 324.0 | 7288 / 15152 |
| c8-natural | 875.2 / 837.6 | 20.1 / 43.2 / 67.6 vs 37.4 / 42.0 / 62.8 | 8.63 / 8.67 / 8.86 vs 8.94 / 8.99 / 9.09 | 562.2 / 1150.4 vs 591.3 / 1180.4 | 7632 / 15188 |
| c16-fixed | 1352.4 / 1276.7 | 22.1 / 67.7 / 114.6 vs 37.4 / 93.6 / 94.1 | 10.44 / 10.80 / 11.39 vs 10.31 / 10.92 / 11.15 | 226.6 / 400.0 vs 235.9 / 409.5 | 7382 / 15288 |
| c16-natural | 1544.1 / 1495.8 | 28.2 / 60.2 / 174.7 vs 40.8 / 105.2 / 117.9 | 9.43 / 9.90 / 10.72 vs 9.62 / 9.78 / 9.92 | 614.9 / 1363.2 vs 643.7 / 1321.8 | 7945 / 15299 |
| c32-fixed | 1934.8 / 1873.6 | 39.7 / 226.0 / 245.3 vs 54.6 / 172.2 / 172.8 | 13.48 / 14.03 / 14.37 vs 13.51 / 15.08 / 18.14 | 311.5 / 619.7 vs 320.3 / 558.2 | 7570 / 15311 |
| c32-natural | 2416.6 / 2489.5 | 31.4 / 256.2 / 356.7 vs 63.8 / 179.7 / 241.0 | 11.39 / 14.12 / 14.50 vs 10.78 / 11.52 / 12.64 | 744.2 / 1779.6 vs 742.3 / 1559.8 | 8542 / 15339 |

원래 기준 판정(미달 항목만):

| 조건 | 처리량 R/V | 미달 항목 |
|---|---:|---|
| c1-fixed | 0.981 | 처리량, TPOT p50·p95·p99 (+4.0%) |
| c1-natural | 0.968 | 처리량, TPOT (+3.6~4.3%), TTFT p95·p99 (29.0·29.6 vs 25.2·26.2) |
| c8-fixed | 1.070 | TTFT p99 (59.2 vs 51.9) |
| c8-natural | 1.045 | TTFT p95·p99 (43.2·67.6 vs 42.0·62.8) |
| c16-fixed | 1.059 | TPOT p50 (10.44 vs 10.31), TPOT p99 (11.39 vs 11.15), TTFT p99 (114.6 vs 94.1) |
| c16-natural | 1.032 | TTFT p99 (174.7 vs 117.9), TPOT p95·p99, E2E p99 |
| c32-fixed | 1.033 | TTFT p95·p99 (226·245 vs 172·173), E2E p99 |
| c32-natural | 0.971 | 처리량, TPOT 전 구간 (+5.7~22.6%), TTFT p95·p99, E2E p99 |

final8(§96) 대비: c32-natural 처리량 0.957 → 0.971, TPOT p50 11.56 → 11.39 ms, TTFT p50 34.4 → 31.4 ms; c16-natural 처리량 1.018 → 1.032, TPOT p50 9.55 → 9.43; c8-natural 1.041 → 1.045. fixed 계열과 C1은 변화 없음. 원래 기준을 모두 채우는 조건은 여전히 없습니다. 처리량은 5/8, TTFT p50은 8/8, TPOT p50은 4/8(C8 두 조건, c16-natural, c32-fixed)에서 vLLM 이상입니다.

## 102. Qwen 후보(변형 6) 안정성 — attempt 24 (binary `56a9c72e…`)

`qwen-scheduler-stability/attempt24-prompt-mma/`. attempt 22(§78)와 같은 script에 `--kv-cache-write fused-split-v1 --prompt-attention tensor-core-v1`만 추가. 조건별 300초 지속 부하 + idle 취소 20회 + 부하 중 취소 40회 + 재요청 + 종료 회수.

| 조건 | 결과 | 요청 | 오류 | 불일치 | tok/s 첫 / 끝 round (최소) | TPOT p50 첫 / 끝 ms | GPU MiB 첫¼ / 끝¼ | RSS MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 뒤 KV 블록 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-fixed | 통과 | 1920 | 0 | 0 | 127.2 / 127.1 (127.0) | 7.69 / 7.70 | 7198 / 7198 | 6441.7 / 6442.1 | 20 / 40 | 0 |
| c1-natural | 통과 | 576 | 0 | 0 | 127.6 / 127.5 (127.5) | 7.70 / 7.71 | 7328 / 7328 | 6449.9 / 6450.2 | 20 / 40 | 0 |
| c8-fixed | 통과 | 11808 | 0 | 0 | 789.4 / 787.9 (781.6) | 8.75 / 8.80 | 7298 / 7298 | 6477.9 / 6478.6 | 20 / 40 | 0 |
| c8-natural | 통과 | 3456 | 0 | 0 | 846.3 / 846.0 (844.2) | 8.63 / 8.62 | 7649 / 7658 | 6516.9 / 6526.2 | 20 / 40 | 0 |
| c16-fixed | 통과 | 19008 | 0 | 0 | 1259.2 / 1249.9 (1249.6) | 10.43 / 10.51 | 7395 / 7404 | 6501.2 / 6510.9 | 20 / 40 | 0 |
| c16-natural | 통과 | 5856 | 0 | 0 | 1440.9 / 1447.2 (1440.9) | 9.45 / 9.46 | 7973 / 7986 | 6554.0 / 6568.0 | 20 / 40 | 0 |
| c32-fixed | 통과 | 26112 | 0 | 0 | 1718.4 / 1727.7 (1718.4) | 13.29 / 12.92 | 7586 / 7592 | 6547.6 / 6555.1 | 20 / 40 | 0 |
| c32-natural | 통과 | 8736 | 0 | 0 | 2147.5 / 2161.1 (2147.5) | 11.26 / 11.30 | 8564 / 8566 | 6569.2 / 6572.9 | 20 / 40 | 0 |

8/8 통과: 요청 오류 0, 같은 prompt의 출력 불일치 0(지속 부하 전 구간에서 prompt당 출력 1종), 취소 60회 뒤 재요청 정상, 종료 시 exit 0·GPU 프로세스 없음·active/waiting/KV 블록 0·device/pinned 할당 0. 처리량과 TPOT에 추세 없음. GPU 메모리는 300초 동안 0~13 MiB, RSS는 0.3~14 MiB 증가(c16-natural이 최대; attempt 22에서도 같은 수준의 초기 증가 후 평탄 — 30분 soak은 이 후보로는 미실행). 이 script의 처리량은 round 단위 집계라 정상상태 비교(§101)보다 낮게 나옵니다.

### 102.1 Qwen 후보 30분 soak — 사용자 지시로 중단, 미실행으로 기록 (2026-10-12)

c32-natural 1800초 soak(attempt 25)을 시작했으나 약 11분 시점에 사용자 지시로 중단했습니다. 결과 파일(`result.json`)은 생성되지 않았고 판정에 쓰지 않습니다. 중단 시 서버는 정상 종료했습니다(`native-close.json`: active/waiting 0, KV 블록 0, device·pinned 할당 0; GPU 프로세스 없음). 부분 기록은 호스트 `qwen-scheduler-stability-c32-natural-attempt25-stopped-by-user`에 보존했습니다.

따라서 이 후보(binary `56a9c72e…`)의 안정성 근거는 §102의 300초 × 8조건이며, **30분 soak은 미실행**입니다. 직전 채택 빌드(626a1cb6)의 30분 soak은 §79에 있습니다.

## 103. 새 목표(2026-10-12)와 현재 후보의 재진단 — step 종류별 프로파일, step 예산 실험

사용자가 목표를 다시 정했습니다(요지): **필수** — Qwen C8·C16·C32 전 조건에서 처리량 ≥ vLLM, TPOT p50 ≤ vLLM, TTFT p50은 8조건 모두 ≤ vLLM. **추진하되 미달 시 근거로 닫는 항목** — C1 처리량·TPOT, TTFT p95·p99(step 예산·scheduling 실험을 현재 빌드로 최소 1회 다시 수행하고 trade-off를 표로 보고). 정확성 계약은 scheduler 경로 "동시성 불변 15/15 + HF token 일치 11/15". 수단마다 기대 효과를 먼저 적고, 연속 2개 수단이 1% 미만이면 그 항목을 닫습니다.

출발점(§101, binary `56a9c72e…`)의 필수 미달은 세 가지였습니다: c32-natural 처리량 0.971, c32-natural TPOT p50 11.39 vs 10.78 ms, c16-fixed TPOT p50 10.44 vs 10.31 ms.

### 103.1 요청 단위 분해 (final9 원자료)

- c32-natural: prompt 종류(16 / 128 / 398 token)와 무관하게 TPOT p50이 11.33~11.44 ms로 고르게 vLLM(10.50~10.82)보다 큽니다. 특정 요청이 아니라 step 자체가 느립니다. E2E 평균 885.5 vs 861.8 ms.
- "fixed" 조건의 출력은 8~32 token(평균 약 20)이고 prompt는 약 126 token입니다. c16-fixed는 step당 평균 0.8건, c32-fixed는 1.6건이 새로 도착하므로 **거의 모든 step이 prefill을 동반**합니다.

### 103.2 step 종류별 kernel 시간 (nsys, 후보 binary `56a9c72e…`, 채택 flag 전체)

`qwen-profile-goal2/`(스크립트 `qwen_step_rows.py`: embedding kernel로 step을 끊고, norm kernel의 grid로 dense 행 수를 읽어 묶음). 호스트 `qwen-profile-diagnostic-c32-attempt13`, `…-c16f-attempt14`. 값은 step당 ms(kernel 실행 시간 합; profiler가 넣는 launch 지연은 "간격"에만 들어갑니다). 측정 구간에는 시작 burst와 끝의 drain이 들어 있어 행 수가 적은 step의 비중은 정상상태보다 큽니다.

| 조건 | dense 행 | step 수 | 시간 비중 | kernel 합 | GEMM | attention producer | prompt MMA | merge | norm | split-K reduce | SiLU·곱 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c32-natural | 32 | 120 | 30.7% | 9.00 | 7.46 | 0.90 | — | 0.13 | 0.18 | 0.13 | 0.08 |
| c32-natural | 160 | 7 | 2.5% | 12.18 | 9.41 | 1.58 | — | 0.23 | 0.30 | 0.25 | 0.25 |
| c32-natural | 208 (+prompt MMA) | 10 | 4.3% | 15.02 | 10.75 | 1.82 | 1.06 | 0.31 | 0.32 | 0.28 | 0.32 |
| c32-natural | 256 | 37 | 15.5% | 15.04 | 11.15 | 2.36 | — | 0.32 | 0.34 | 0.30 | 0.39 |
| c32-natural | 256 (+prompt MMA) | 19 | 8.6% | 16.08 | 11.15 | 2.30 | 1.06 | 0.35 | 0.35 | 0.30 | 0.39 |
| c16-fixed | 16 | 44 | 22.8% | 8.10 | 7.16 | 0.38 | — | 0.10 | 0.18 | 0.13 | 0.05 |
| c16-fixed | 144 | 51 | 35.6% | 11.11 | 9.19 | 0.83 | — | 0.19 | 0.30 | 0.24 | 0.22 |
| c16-fixed | 256 | 19 | 16.6% | 13.89 | 11.10 | 1.29 | — | 0.29 | 0.34 | 0.30 | 0.38 |

읽는 법:
- **GEMM은 행 수에 대해 계단형입니다.** step당 GEMM 합이 16행 7.16 → 128행 8.22(행당 약 9.5 µs) → 144행 9.19(+0.97 ms) → 176행 9.50 → 208행 10.75(+1.25 ms) → 256행 11.15 ms. 128행·192행을 넘는 순간 약 1 ms씩 뜁니다.
- 이 계단은 선택한 알고리즘 class의 문제가 아닙니다. §62의 전수 측정(`qwen-gemm-diagnostics/gemm_equiv2_*.log`)에서 "bit 동일 class 안의 최선"과 "class를 가리지 않은 최선"이 모든 행 수에서 1~3 µs 이내입니다(gate·up: 128행 54.7 / 55.2, 144행 67.2 / 65.6, 256행 85.6 / 87.2 µs; M=1도 50.1 / 50.2). 128행까지는 가중치 읽기에 가려 추가 행이 거의 공짜이고, 그 위는 연산량에 묶입니다(추가 행당 gate·up 0.24~0.29 µs, 약 150 TFLOPS). **다른 GEMM class를 찾는 수단은 이 자료로 닫습니다(기대 이득 0).**
- c16-fixed의 대표 혼합 step은 "decode 15행 + prompt 126행 = 141행 → bucket 144"로, 첫 계단을 13행 차이로 넘습니다.
- GEMM 밖에서 prefill이 얹는 비용은 attention입니다: producer + merge + prompt MMA가 pure decode step의 1.03 ms에서 256행 step의 2.7~3.7 ms로 늘어납니다. 그리고 pure decode 32행에서도 attention 합 1.03 ms는 vLLM의 flash 계열 0.66 ms(§29, layer당 18.2 µs)보다 큽니다.

### 103.3 step 예산 실험 (후보 binary `56a9c72e…`, Riley 단독, C8~C32)

사전 기대: 예산을 계단(128·192)에 맞추면 step 시간(TPOT)은 줄지만, prefill이 밀려 처리량·TTFT가 나빠질 것(§76과 같은 방향). 계단 구조를 알았으므로 192를 새로 추가했습니다. `qwen-matrix/goal2-budget-candidate/` (호스트 attempt 61~64; 같은 binary, flag만 변경; warmup 32 / retained 192 × 2회; 오류 0, 제외 없음). 네 구성 모두 15개 prompt의 출력이 기준과 같습니다(예산·chunk는 출력을 바꾸지 않음).

| 조건 | 구성 | tok/s | 변화 | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | E2E p50 / p99 ms |
|---|---|---:|---:|---|---|---|
| c8-fixed | 예산 256 (기준) | 816.1 | — | 8.76 / 9.17 / 9.29 | 21.0 / 32.5 / 60.1 | 192.7 / 309.8 |
| c8-fixed | 예산 192 | 815.1 | −0.1% | 8.82 / 9.10 / 9.28 | 19.7 / 30.2 / 64.2 | 199.4 / 310.8 |
| c8-fixed | 예산 128 | 775.7 | −4.9% | 8.73 / 8.97 / 9.12 | 29.0 / 47.5 / 77.9 | 210.1 / 324.4 |
| c8-fixed | 예산 256, chunk 128 | 809.6 | −0.8% | 8.79 / 9.18 / 9.28 | 21.0 / 32.6 / 59.5 | 194.5 / 309.5 |
| c8-natural | 예산 256 (기준) | 875.9 | — | 8.62 / 8.66 / 8.89 | 19.7 / 41.3 / 78.7 | 562.0 / 1149.8 |
| c8-natural | 예산 192 | 871.6 | −0.5% | 8.63 / 8.68 / 8.80 | 19.9 / 47.3 / 83.1 | 563.2 / 1163.6 |
| c8-natural | 예산 128 | 866.3 | −1.1% | 8.60 / 8.64 / 8.85 | 27.3 / 58.5 / 94.9 | 569.0 / 1162.6 |
| c8-natural | 예산 256, chunk 128 | 865.4 | −1.2% | 8.64 / 8.70 / 8.95 | 22.1 / 56.5 / 66.4 | 564.4 / 1160.4 |
| c16-fixed | 예산 256 (기준) | 1352.5 | — | 10.46 / 10.79 / 11.68 | 22.1 / 54.0 / 114.4 | 225.1 / 389.8 |
| c16-fixed | 예산 192 | 1349.3 | −0.2% | 10.44 / 10.66 / 12.16 | 22.2 / 79.5 / 130.6 | 246.0 / 414.0 |
| c16-fixed | 예산 128 | 1291.4 | −4.5% | 9.96 / 10.25 / 10.45 | 39.7 / 124.5 / 176.4 | 248.7 / 474.1 |
| c16-fixed | 예산 256, chunk 128 | 1343.4 | −0.7% | 10.42 / 10.78 / 11.05 | 23.4 / 69.2 / 115.8 | 233.8 / 392.5 |
| c16-natural | 예산 256 (기준) | 1543.2 | — | 9.43 / 9.91 / 10.56 | 26.3 / 71.0 / 175.4 | 615.7 / 1360.5 |
| c16-natural | 예산 192 | 1532.5 | −0.7% | 9.44 / 9.83 / 10.36 | 36.0 / 75.3 / 200.2 | 614.9 / 1383.4 |
| c16-natural | 예산 128 | 1517.4 | −1.7% | 9.38 / 9.72 / 10.49 | 34.0 / 107.1 / 259.0 | 619.7 / 1437.7 |
| c16-natural | 예산 256, chunk 128 | 1526.5 | −1.1% | 9.45 / 9.83 / 10.38 | 26.1 / 73.9 / 173.7 | 620.2 / 1360.6 |
| c32-fixed | 예산 256 (기준) | 1945.3 | — | 13.45 / 13.99 / 14.38 | 38.3 / 228.4 / 243.2 | 311.6 / 643.9 |
| c32-fixed | 예산 192 | 1867.2 | −4.0% | 12.46 / 12.65 / 12.73 | 61.8 / 237.5 / 300.9 | 315.5 / 617.2 |
| c32-fixed | 예산 128 | 1501.7 | −22.8% | 10.68 / 12.03 / 14.74 | 180.8 / 342.6 / 382.5 | 394.8 / 688.6 |
| c32-fixed | 예산 256, chunk 128 | 1932.9 | −0.6% | 13.40 / 14.06 / 14.35 | 40.1 / 218.6 / 251.3 | 307.7 / 635.7 |
| c32-natural | 예산 256 (기준) | 2420.1 | — | 11.36 / 13.99 / 14.86 | 30.9 / 279.4 / 359.6 | 743.0 / 1779.1 |
| c32-natural | 예산 192 | 2402.7 | −0.7% | 11.25 / 14.66 / 15.13 | 40.7 / 306.0 / 413.1 | 736.7 / 1828.1 |
| c32-natural | 예산 128 | 2320.6 | −4.1% | 11.18 / 14.44 / 15.45 | 66.9 / 393.7 / 541.9 | 749.5 / 1923.8 |
| c32-natural | 예산 256, chunk 128 | 2400.7 | −0.8% | 11.32 / 13.58 / 13.95 | 42.3 / 272.7 / 358.9 | 743.2 / 1778.2 |

판정: **네 구성 모두 미채택.** 예산을 줄이면 TPOT p50은 내려가지만(c16-fixed 10.46 → 9.96, c32-fixed 13.45 → 10.68) 처리량이 0.1~22.8% 떨어지고 TTFT가 전 구간에서 나빠집니다(c32-fixed의 TTFT p50 61.8 / 180.8 ms는 vLLM 54.6 ms도 넘김). chunk 128은 c32-natural의 TPOT p95·p99(13.99 / 14.86 → 13.58 / 13.95)와 c8-natural의 TTFT p99만 줄이고 처리량을 0.6~1.2% 잃습니다. 고정 예산은 "행 수 × 시간"의 총량을 줄이지 못하고 대기열로 옮길 뿐입니다. C1 부분은 다른 측정에 호스트를 내주느라 두 번 중단했습니다(부분 기록은 호스트 `…budget256-c1-attempt65-stopped-for-microbench`, `…attempt66-stopped-for-build`에 보존, 판정에 쓰지 않음).

## 104. 행 단위 tensor-core attention — microbenchmark

§103.2의 결론(계단은 물리적, 남은 차이는 attention)에 따라 attention을 다시 설계했습니다. 기존 구조는 "page마다 CTA 하나가 partial state를 만들고(shared-KV producer), 별도 kernel이 page 상태를 합치는" 2단이고, prompt 위치 256 이후만 16행 tile 단위 MMA kernel이 맡습니다(§99).

새 구조(`qwen-resume-diagnostics/rowwise_mma{,2,3}.cu`, 호스트 `qwen-variant-diagnostics/`): **CTA 하나가 (행, KV head)를 맡아 그 행의 attention을 끝까지 계산**합니다.
- warp 8개가 page `w, w+8, …`을 오름차순으로 online softmax로 접습니다. page 하나의 점수는 그 KV head의 query head 8개에 대해 m16n8k16 MMA 8회(token × head), 확률은 BF16으로 반올림해 다시 MMA 8회(depth × head)의 피연산자로 씁니다.
- warp 8개의 상태를 CTA 안에서 합쳐(전체 최댓값, warp 순서 고정) BF16 출력을 바로 씁니다. partial state buffer와 merge kernel이 없습니다.
- 출력 원소는 자기 행의 query와 자기 문맥의 page에만 의존합니다(행 위치·batch 구성·prompt chunk와 무관). 그래서 decode 행과 prompt 행을 같은 kernel이 처리합니다.

값은 CPU 기준(배정밀도)과 대조했습니다: 최대 오차 0.0001~0.0008로 현재 경로와 같은 수준. 시간은 36 layer × 서로 다른 KV pool을 graph로 재생한 layer당 µs입니다(현재 = producer + merge, page 용량 grid 포함).

| 형상 | 행 수 | 현재 (producer + merge) | 행 단위 MMA (누산기 1 / 4 / 8개) |
|---|---:|---:|---|
| C1 fixed decode, 문맥 140 | 1 | 6.70 | 7.56 / 5.28 / 5.49 |
| C1 natural decode, 문맥 270 | 1 | 6.96 | 10.34 / 6.83 / 7.05 |
| C1 natural decode, 문맥 500 | 1 | 6.82 | 12.86 / 9.04 / 9.60 |
| C8 fixed decode | 8 | 7.63 | 7.30 / 5.20 / 5.36 |
| C8 natural decode | 8 | 12.21 | 12.47 / 8.68 / 9.20 |
| C16 fixed decode | 16 | 12.02 | 13.09 / 7.83 / 7.62 |
| C16 natural decode | 16 | 15.23 | 23.70 / 12.90 / 12.34 |
| C32 fixed decode | 32 | 16.52 | 14.32 / 9.47 / 9.45 |
| C32 natural decode | 32 | 21.32 | 25.46 / 14.54 / 14.16 |
| c16-fixed 혼합 (decode 15 + prompt 126) | 141 | 25.45 | 20.71 / 13.78 / 13.36 |
| c32-fixed 혼합 (decode 30 + prompt 126 + 100) | 256 | 39.79 | 27.37 / 17.81 / 16.92 |
| c32-natural 혼합 (decode 30 + prompt 128 + 16 + 82) | 256 | 50.26 | 37.66 / 24.27 / 21.83 |
| c32-natural 혼합 (decode 30 + prompt 위치 82..255 + 52) | 256 | 78.93 | 41.37 / 31.65 / 30.06 |
| prompt 126행 (위치 0..) | 126 | 21.75 | 8.79 / 9.11 / 8.80 |
| prompt 256행 (위치 0..) | 256 | 69.96 | 21.79 / 24.55 / 23.62 |
| prompt 128행 (위치 200..) | 128 | 63.80 | 24.12 / 23.65 / 22.77 (tile MMA §97: 27.6) |
| prompt 224행 (위치 174..) | 224 | 115.94 | 35.76 / 43.35 / 40.90 (tile MMA §97: 39.9) |

- 열 설명: 누산기 1·4개는 `rowwise_mma.cu`(value를 점수 계산 뒤에 읽음), 8개는 `rowwise_mma2.cu`의 value 선행 load 변형입니다. runtime에 넣은 구성(누산기 8개, 점수의 BF16 2회 반올림 유지, value는 뒤에 읽기)은 이 표가 아니라 §105의 서버 측정으로 평가합니다.
- 점수 MMA 8회를 누산기 하나에 차례로 쌓으면 의존 사슬 때문에 느립니다. 누산기를 4~8개로 나눠 고정 tree로 더하면 decode 형상이 30~45% 빨라집니다(두 번째 열 → 세 번째·네 번째 열).
- 버린 변형(`rowwise_mma2.cu`, `rowwise_mma3.cu`): value를 점수 MMA 전에 읽기(레지스터 증가로 약간 느림), warp 16개(C1 긴 문맥에 도움 안 됨 — 병목이 SM 하나의 명령 처리량), warp 4개(prompt 위주 형상은 빠르지만 decode가 40% 느림), 빈 warp 건너뛰기와 가득 찬 page 전용 경로(bit는 같고 더 느림), 점수의 BF16 2회 반올림 유지(시간 차이 없음 → 유지하기로 함).
- **약점**: 행 하나의 모든 page가 SM 하나에 몰리므로, 동시성 1에서 문맥이 길면 page 병렬인 현재 구조보다 느립니다(문맥 500에서 9.0 vs 6.8 µs; 교차점은 약 300 token). C8 이상과 prefill 동반 step에서는 전부 빠릅니다.

사전 기대(microbenchmark × 프로파일 비중): c32-natural 처리량 +3~5%·TPOT p50 11.39 → 약 10.9, c16-fixed TPOT p50 10.44 → 약 10.15, C8 +1.5~2.5%, c32-fixed +4~5%, c1-fixed +0.7%, c1-natural 0 안팎. 수치가 바뀌므로 동시성 불변·HF token 비교·안정성을 다시 통과해야 합니다.

## 105. 행 단위 tensor-core attention의 runtime 구현과 수치 변형 세 가지 (`--decode-attention-producer rowwise-tensor-core-v1`, 최종 미채택)

> 결말부터: 이 절의 행 단위 kernel은 C8 이상에서 빨랐지만 c1-natural을 2.1% 떨어뜨려 §107에서 미채택이 됐습니다. 최종 채택 구조는 §107.2·§108의 page producer이고, 이 절에서 얻어 그대로 가져간 것은 "softmax 가중치를 BF16 조각 3개로 정확히 싣는다"는 결론입니다. 아래 flag 이름과 Rust 함수는 그때의 것이며 본 코드에는 남아 있지 않습니다.

§104의 구조를 runtime에 넣었습니다(작업 트리: branch `codex/qwen-http-hf-exact`, 9cf4dcac 이후).

- native: `native_bf16_ragged_d128_rowwise_mma_kernel`(`kernels/src/batch_primitives.cu`). 기존 D128 two-stage 연산의 V3 ABI에서 `reserved[0] = reserved[1] = 2`로 선택하며, 그때는 producer·merge를 띄우지 않고 이 kernel 하나만 (출력 행, KV head) grid로 띄웁니다. 활성 행 수를 넘는 행은 0으로 채우고(기존 merge와 같은 계약), 행의 page 수가 용량을 넘으면 NaN, 풀 수 없는 page나 유한하지 않은 점수도 NaN입니다. step graph 기록(장치 쪽 행 수)도 그대로 지원합니다.
- Rust: `riley_cuda::native_bf16_ragged_paged_split_gqa_d128_rowwise_tensor_core`, executor 설정 `with_d128_rowwise_attention()`, 서버 flag `--decode-attention-producer paged-v1|rowwise-tensor-core-v1`(기본 paged-v1; native D128 backend 필요, `--prompt-attention tensor-core-v1`과 함께 쓸 수 없음 — 이 kernel이 prompt 행도 처리).
- GPU 테스트 `rowwise_attention_gpu`: 6개 sequence·108행(1 page, 꽉 찬 page 경계, warp 수보다 많은 page, 마지막 위치 포함)을 (1) 순서를 뒤집고 돌려 다른 이웃과 묶어도, (2) 5개 호출로 나눠도 행별 출력이 bit 동일. 활성 행 뒤의 padding 행은 0. 기존 경로(shared-KV + global merge)와의 차이는 허용 범위 안.

### 105.1 수치 변형 — 확률을 어떻게 싣느냐에 따라 HF token 일치가 갈렸습니다

MMA의 피연산자는 BF16이어야 하므로 softmax 가중치를 tensor core에 싣는 방식이 수치를 정합니다. 세 변형 모두 점수는 기존 경로와 같은 반올림(내적 뒤 BF16, scale 뒤 BF16)을 유지했고, 모두 동시성 불변(15/15)입니다. HF token 비교는 §82·§100과 같은 script·같은 HF 기준(eager, BF16, greedy)입니다(`qwen-hf-token-reference/attempt07…09`).

| 변형 | 확률 처리 | binary | GPU 테스트: 기존 경로와 bit 동일 원소 / 최대 차이 | HF token 일치 C1 / C32 | 불일치 prompt (첫 불일치 index / 길이) | 판정 |
|---|---|---|---|---|---|---|
| (기존) paged + prompt MMA | FP32 가중치, page별 상태 병합 | `56a9c72e…` | — | 11 / 15, 44 / 60 | arithmetic 14/24, planning 25/32, sql 4/16, natural-398 69/128 | 기준 |
| v1 | 진행 중 최댓값 기준 `exp(s − m)`을 BF16으로 반올림, online 재조정, 끝에 합으로 나눔 | `b3c1979d…` | 181,182 / 221,184 (81.9%), 0.0039 | 9 / 15, 36 / 60 | planning 25/32, editing 28/32, sql 4/16, classification 26/32, natural-16 29/32, natural-398 36/128 | 기각 |
| v2 | 전체 최댓값·합으로 정규화한 확률을 BF16으로 반올림, 재정규화 없음(HF eager와 같은 반올림 지점) | `59c44cbe…` | 174,441 / 221,184 (78.9%), 0.0039 | 9 / 15, 36 / 60 | editing 28/32, sql 4/16, classification 26/32, natural-16 29/32, natural-128 2/64, natural-398 56/128 | 기각 |
| **v3** | **FP32 가중치 `exp(s − M)`를 BF16 조각 3개(상·중·하위, 뺄셈이 모두 정확)로 나눠 MMA 3회에 실음. 가중치 bit 손실 없음. 출력 = N × (1 / S)** | `0a931ef8…` | **221,134 / 221,184 (99.98%), 0.00098** | **13 / 15, 52 / 60** | natural-128 44/64, natural-398 69/128 | **후보** |

- v1·v2는 계약(11/15)에 못 미쳤습니다. 두 변형이 공통으로 새로 틀린 세 prompt(editing·classification·natural-16)는 v3에서 모두 다시 맞았으므로, 원인은 점수 계산 순서가 아니라 **확률의 BF16 반올림**이었습니다. HF eager도 확률을 BF16으로 반올림하지만, 그 밖의 모든 연산(GEMM·norm 순서)이 HF와 bit 단위로 같지 않은 scheduler 경로에서는 HF와 같은 지점에서 반올림하는 것보다 가중치를 정확히 유지하는 쪽이 HF 출력에 더 가까웠습니다.
- 불일치 지점의 HF logit margin(`hf-margins.json`, HF를 공통 prefix로 teacher-forcing): v1·v2·v3의 모든 불일치는 HF의 1·2위 logit 차이가 0~0.375(그 크기에서 BF16 logit 한 칸은 0.07~0.13)인 근소 차 위치입니다. v3의 둘은 natural-128 index 44(차이 0.25, Riley token이 HF 2위)와 natural-398 index 69(차이 0.25, cache 없이 다시 계산한 HF의 1위가 Riley token — HF도 cache 유무로 갈리는 위치)입니다.
- 기각한 변형의 증거: v1은 binary와 결과(호스트 `qwen-riley-rowwise1-candidate`, attempt 70·07), 연산은 `qwen-resume-diagnostics/rowwise_mma3.cu`의 `page_fold<8, true, …>`와 같습니다. v2는 kernel 원문 `qwen-rowwise-kernel-variants/rowwise-kernel-v2-eager-rounding.cu`, v3(3-pass 형태) 원문과 v2→v3 차이도 같은 폴더에 있습니다.
- 세 변형 모두 HF-exact HTTP gate 통과(attempt 52·53·54: 128 token 일치, `same True`) — HF-exact 경로는 이 변경과 무관합니다.

v3의 연산 순서(행 하나, KV head 하나의 query head 8개):
1. 점수 = BF16(BF16(q·k) × scale): page마다 MMA 8회(token × head, FP32 누산)를 고정 pairwise tree로 합산. 유효 token 수를 넘는 자리는 −Inf.
2. M = 행 전체 점수의 최댓값. e = exp(점수 − M). S = e의 합(lane별 page 오름차순 → 고정 lane tree → warp 오름차순).
3. N = Σ e·v: e를 BF16 조각 3개로 정확히 분해해 조각마다 MMA 1회(depth × head). warp별 page 오름차순, warp 오름차순.
4. 출력 = BF16(N × (1 / S)).

### 105.2 정상상태 A/B (같은 binary 계열, Riley 단독, C8~C32)

기준은 같은 source에서 flag만 기존 구성(`--prompt-attention tensor-core-v1`)으로 둔 attempt 71. warmup 32 / retained 192 × 2회, 오류 0, 제외 없음. raw `qwen-matrix/goal2-rowwise/`. (v2의 측정 attempt 82는 공유 호스트의 외부 부하 — load average 30~400, IO pressure 40%대 — 가 겹쳐 6/12에서 중단했고 판정에 쓰지 않습니다. 부분 기록은 호스트 `…rowwise2-c8-32-attempt82-stopped-host-under-external-load`. 이후 `qwen-hostload-20261012.log`에 30초 간격으로 부하를 기록합니다.)

| 조건 | 기준 tok/s | v1 | v3 | TPOT p50 ms 기준 → v1 / v3 | TTFT p50 / p95 / p99 ms 기준 → v3 | vLLM (final9) tok/s, TPOT p50 |
|---|---:|---:|---:|---|---|---|
| c8-fixed | 814.7 | 839.3 (+3.0%) | 832.7 (+2.2%) | 8.77 → 8.57 / 8.62 | 21.5 / 32.4 / 58.7 → 20.3 / 32.3 / 57.3 | 761.2, 9.01 |
| c8-natural | 875.5 | 895.8 (+2.3%) | 875.5 (0.0%) | 8.63 → 8.43 / 8.63 | 20.0 / 39.1 / 68.5 → 19.3 / 37.5 / 69.5 | 837.6, 8.94 |
| c16-fixed | 1350.1 | 1422.0 (+5.3%) | 1407.2 (+4.2%) | 10.42 → 9.88 / 10.01 | 22.2 / 56.9 / 116.9 → 21.1 / 61.3 / 110.0 | 1276.7, 10.31 |
| c16-natural | 1543.7 | 1613.3 (+4.5%) | 1576.6 (+2.1%) | 9.43 → 9.01 / 9.22 | 28.0 / 65.0 / 174.4 → 25.6 / 59.3 / 157.5 | 1495.8, 9.62 |
| c32-fixed | 1931.1 | 2083.1 (+7.9%) | 2057.7 (+6.6%) | 13.49 → 12.49 / 12.65 | 39.3 / 227.3 / 257.2 → 37.0 / 215.8 / 236.9 | 1873.6, 13.51 |
| c32-natural | 2420.3 | 2634.5 (+8.8%) | 2580.5 (+6.6%) | 11.34 → 10.33 / 10.52 | 31.2 / 285.6 / 358.8 → 29.8 / 245.6 / 312.3 | 2489.5, 10.78 |

- v3는 조각 3개(PV MMA 24회)와 최댓값을 먼저 구하는 구조 때문에 v1보다 1~2% 느리지만, 기존 구성 대비 c8-natural을 빼고 +2.1~6.6%입니다. 15개 prompt 모두 조건·반복 전체에서 출력 1종(동시성 불변)이고, 그중 10개는 기존 구성과 같은 출력입니다(v1도 10개; 바뀐 5개는 근소 차 위치에서 갈린 것으로, HF와의 일치는 11 → 13개로 늘었습니다). GPU 메모리는 그대로(7.3~8.5 GiB).
- 사전 기대(§104: c32-natural +3~5%, c16-fixed TPOT 약 10.15, c32-fixed +4~5%)보다 혼합 step 쪽 이득이 컸습니다(기존 producer의 용량 크기 launch grid와 merge가 함께 사라짐).
- vLLM 값은 final9(같은 실행 아님) 기준입니다. v3 기준으로 필수 항목 셋(c32-natural 처리량·TPOT p50, c16-fixed TPOT p50)이 모두 vLLM 수준을 넘습니다 — 같은-실행 비교는 §뒤에서 다시 합니다.

## 106. step 예산·scheduling 실험 — 행 단위 attention 빌드에서 다시 (TTFT p95·p99 항목의 trade-off)

§103.3은 직전 후보에서의 실험이고, 이 절은 행 단위 attention(v3) binary `0a931ef8…`에서 다시 한 것입니다(계단 정책은 최종 binary에서 §109에 한 번 더 측정했고 결론이 같습니다). Riley 단독, C8~C32, warmup 32 / retained 192 × 2회, 오류 0, 제외 없음. raw `qwen-matrix/goal2-scheduling/`. 모든 구성에서 15개 prompt의 출력이 기준과 같습니다.

### 106.1 계단 인지 prefill 미루기 (`--prefill-plateau-rows`, 새 scheduler 정책)

§103.2의 GEMM 계단(128·192행)에 맞춘 정책입니다: 한 step의 행 수가 계단을 **조금** 넘게 되면(기본 32 token 이하), 넘는 prompt token을 prefill 선택의 꼬리에서 잘라 다음 step으로 넘깁니다. decode 행이 12개 이상일 때만 적용하고, 한 요청은 평생 한 번만 잘립니다(지연 상한 = 요청당 1 step). 고정 예산(§103.3)과 달리 대기열이 쌓이지 않습니다. 구현: `riley_scheduler::PrefillPlateauPolicy`, `Scheduler::set_prefill_plateau_policy`, 서버 flag `--prefill-plateau-rows LIST`(+ `--prefill-plateau-defer-tokens`, `--prefill-plateau-min-decode-rows`), 기본 꺼짐. 테스트 `riley-scheduler/tests/prefill_plateau.rs` 4건과 flag parsing 테스트.

사전 기대: c16-fixed의 대표 step(141행)이 128행으로 내려가 TPOT p50 −4% 안팎, 처리량은 거의 그대로(요청이 한 step 늦게 시작하는 손해와 상쇄), TTFT p50 +5~10 ms.

| 조건 | 구성 | tok/s | 변화 | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | E2E p50 / p99 ms |
|---|---|---:|---:|---|---|---|
| c8-fixed | 기준 (v3, 예산 256) | 832.7 | — | 8.62 / 8.97 / 9.07 | 20.3 / 32.3 / 57.3 | 184.3 / 303.4 |
| c8-fixed | 계단 128 | 834.6 | +0.2% | 8.62 / 8.94 / 9.00 | 20.1 / 27.7 / 56.4 | 187.8 / 300.2 |
| c8-fixed | 계단 128,192 | 832.6 | 0.0% | 8.65 / 8.98 / 9.08 | 20.1 / 28.5 / 55.5 | 187.2 / 302.2 |
| c8-natural | 기준 | 875.5 | — | 8.63 / 8.66 / 8.85 | 19.3 / 37.5 / 69.5 | 562.7 / 1147.3 |
| c8-natural | 계단 128 | 875.1 | 0.0% | 8.63 / 8.66 / 8.70 | 19.5 / 40.7 / 65.6 | 562.3 / 1143.6 |
| c8-natural | 계단 128,192 | 874.1 | −0.2% | 8.64 / 8.68 / 8.76 | 19.6 / 40.6 / 62.0 | 562.7 / 1146.8 |
| c16-fixed | 기준 | 1407.2 | — | 10.01 / 10.29 / 11.34 | 21.1 / 61.3 / 110.0 | 220.5 / 401.8 |
| c16-fixed | 계단 128 | 1442.4 | +2.5% | 9.46 / 9.90 / 10.95 | 26.9 / 55.8 / 110.1 | 219.3 / 355.9 |
| c16-fixed | 계단 128,192 | 1441.9 | +2.5% | 9.40 / 9.88 / 10.62 | 26.8 / 56.3 / 110.8 | 219.9 / 371.5 |
| c16-natural | 기준 | 1576.6 | — | 9.22 / 9.53 / 9.86 | 25.6 / 59.3 / 157.5 | 602.3 / 1308.3 |
| c16-natural | 계단 128 | 1573.0 | −0.2% | 9.19 / 9.52 / 9.99 | 28.4 / 67.5 / 162.7 | 609.5 / 1316.5 |
| c16-natural | 계단 128,192 | 1580.3 | +0.2% | 9.15 / 9.48 / 9.80 | 31.1 / 64.0 / 154.2 | 607.7 / 1303.3 |
| c32-fixed | 기준 | 2057.7 | — | 12.65 / 13.26 / 13.53 | 37.0 / 215.8 / 236.9 | 292.3 / 599.0 |
| c32-fixed | 계단 128 | 2063.2 | +0.3% | 12.40 / 13.19 / 13.49 | 36.4 / 206.9 / 245.9 | 291.1 / 604.1 |
| c32-fixed | 계단 128,192 | 2052.7 | −0.2% | 12.40 / 13.11 / 13.42 | 38.0 / 212.5 / 242.4 | 290.2 / 604.7 |
| c32-natural | 기준 | 2580.5 | — | 10.52 / 12.49 / 12.75 | 29.8 / 245.6 / 312.3 | 688.2 / 1632.0 |
| c32-natural | 계단 128 | 2579.6 | 0.0% | 10.48 / 12.33 / 13.31 | 34.8 / 250.0 / 308.8 | 692.1 / 1634.3 |
| c32-natural | 계단 128,192 | 2595.3 | +0.6% | 10.39 / 12.39 / 12.92 | 36.7 / 249.7 / 315.8 | 685.7 / 1625.0 |

- c16-fixed는 기대보다 좋았습니다: 처리량 +2.5%, TPOT p50 10.01 → 9.40~9.46 ms(−5.5~6.1%), E2E p99 −8~11%. 다른 조건의 처리량은 ±0.6% 이내(측정 반복 편차 수준), TPOT p50은 같거나 내려갑니다(c32-fixed 12.65 → 12.40, c32-natural 10.52 → 10.39~10.48).
- 대가는 TTFT p50입니다: c16-fixed +5.7, c16-natural +2.8~5.5, c32-natural +5.0~6.9 ms. 그래도 vLLM의 TTFT p50(37.4 / 40.8 / 63.8 ms)보다 10~27 ms 낮습니다. C8은 decode 행이 12개 미만이라 정책이 작동하지 않습니다.
- TTFT p95·p99(꼬리)는 방향이 섞여 있습니다(c16-fixed p95 61 → 56, c8-natural p99 70 → 62~66, c32-fixed p99 237 → 242~246). 꼬리 항목을 닫는 수단은 아닙니다.
- 판정: **계단 128,192를 채택 후보로 둡니다**(다른 조건의 처리량·TPOT을 0.5% 넘게 낮추지 않고, TTFT p50은 전 조건 vLLM 이하 유지). 최종 판정은 8조건 같은-실행 비교와 안정성 뒤에 합니다.

### 106.2 더 큰 step 예산 (진단 빌드: native D128 행 상한 256 → 512)

가설: 행 단위 attention으로 prefill token당 비용이 내려갔으니, 예산을 키워 burst(측정 구간 시작 때 동시 도착 — §47)를 적은 step으로 처리하면 TTFT p95·p99가 내려간다. 진단 빌드는 상수 두 개만 바꿨습니다(`qwen-rowwise-cap512-diagsrc`, binary `d219b85c…`; 호스트에서만 사용, 본 코드 미반영). prefill chunk는 256 유지, bucket 목록은 512까지(24개 한도 때문에 240 생략).

| 조건 | 예산 | tok/s | 변화 | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | GPU MiB | vLLM TTFT p95 / p99 (final9) |
|---|---|---:|---:|---|---|---:|---|
| c8-fixed | 256 | 834.0 | — | 8.61 / 8.94 / 9.09 | 20.2 / 31.2 / 55.4 | 7294 | 37.9 / 51.9 |
| c8-fixed | 384 | 818.3 | −1.9% | 8.73 / 9.20 / 9.37 | 23.4 / 29.7 / 48.1 | 7372 | |
| c8-fixed | 512 | 814.2 | −2.4% | 8.74 / 9.15 / 9.41 | 24.0 / 32.5 / 50.2 | 7444 | |
| c8-natural | 256 | 876.0 | — | 8.62 / 8.66 / 8.81 | 19.2 / 38.9 / 69.0 | 7626 | 42.0 / 62.8 |
| c8-natural | 384 | 863.2 | −1.5% | 8.71 / 8.78 / 8.90 | 27.9 / 42.9 / 62.8 | 7758 | |
| c8-natural | 512 | 859.2 | −1.9% | 8.69 / 8.88 / 9.07 | 32.1 / 49.6 / 55.7 | 7872 | |
| c16-fixed | 256 | 1406.5 | — | 9.97 / 10.23 / 10.66 | 21.2 / 60.5 / 112.2 | 7384 | 93.6 / 94.1 |
| c16-fixed | 384 | 1359.4 | −3.3% | 10.39 / 10.88 / 12.03 | 24.3 / 48.5 / 96.6 | 7462 | |
| c16-fixed | 512 | 1327.8 | −5.6% | 10.51 / 11.23 / 12.62 | 27.3 / 50.7 / 98.7 | 7530 | |
| c16-natural | 256 | 1581.8 | — | 9.18 / 9.44 / 9.77 | 26.0 / 76.9 / 145.0 | 7932 | 105.2 / 117.9 |
| c16-natural | 384 | 1542.6 | −2.5% | 9.40 / 9.75 / 10.19 | 32.4 / 74.4 / 143.4 | 8068 | |
| c16-natural | 512 | 1534.7 | −3.0% | 9.42 / 9.92 / 10.68 | 35.4 / 59.9 / 138.3 | 8174 | |
| c32-fixed | 256 | 2051.2 | — | 12.65 / 13.30 / 13.62 | 36.6 / 211.8 / 235.9 | 7564 | 172.2 / 172.8 |
| c32-fixed | 384 | 1938.5 | −5.5% | 13.88 / 14.93 / 17.81 | 31.8 / 178.9 / 210.4 | 7634 | |
| c32-fixed | 512 | 1899.0 | −7.4% | 14.39 / 15.88 / 20.13 | 33.0 / 162.2 / 203.0 | 7698 | |
| c32-natural | 256 | 2597.4 | — | 10.47 / 12.58 / 12.79 | 27.8 / 228.4 / 315.8 | 8520 | 179.7 / 241.0 |
| c32-natural | 384 | 2510.5 | −3.3% | 10.89 / 12.98 / 13.48 | 35.4 / 215.1 / 301.9 | 8648 | |
| c32-natural | 512 | 2497.4 | −3.8% | 11.00 / 13.50 / 13.82 | 41.0 / 219.5 / 281.2 | 8766 | |

- 가설의 절반은 맞습니다: TTFT 꼬리가 내려갑니다(c32-fixed p95 212 → 162, p99 236 → 203; c16-fixed p99 112 → 97; c8-natural p99 69 → 56; c32-natural p99 316 → 281).
- 그러나 **처리량이 1.5~7.4% 떨어지고 TPOT p50이 필수 기준을 넘어갑니다**(c32-fixed 12.65 → 13.88 / 14.39 vs vLLM 13.51; c32-natural 10.47 → 10.89 / 11.00 vs 10.78; c16-fixed 9.97 → 10.39 / 10.51 vs 10.31). C8에서도 TPOT p50이 1.4% 올라간 것으로 보아, 최대 행 수를 키우는 것 자체에 step당 비용이 있습니다(원인 미분리; GPU 메모리도 80~250 MiB 증가).
- 그리고 예산 512로도 C16·C32의 꼬리 대부분은 vLLM보다 큽니다(c32-fixed p99 203 vs 173, c32-natural p95·p99 220·281 vs 180·241, c16-natural p99 138 vs 118). burst 4,032 token의 연산 시간(약 120 ms)에 step당 가중치 읽기(약 6.9 ms × step 수)가 더해지는 구조라, vLLM의 burst TTFT(약 173 ms)에 닿으려면 1,024행 이상의 prefill step이 필요합니다(현재 executor의 두 배가 넘는 상한; 미시도).
- 판정: **미채택.** 꼬리를 10~25% 줄이는 대신 필수 항목(처리량·TPOT p50)을 잃습니다.

### 106.3 TTFT p95·p99 항목의 정리

현재 빌드에서 다시 한 실험: 고정 예산 128 / 192 / 384 / 512, chunk 128(§103.3), 계단 인지 미루기(§106.1). 결과를 한 줄로 줄이면 "**TTFT 꼬리는 burst를 처리하는 step 수로 정해지고, step을 키우면 꼬리는 줄지만 처리량과 TPOT p50을 그만큼 잃는다**"입니다.

| 수단 | TTFT 꼬리 (C32 p95 / p99) | 처리량 | TPOT p50 | 판정 |
|---|---|---|---|---|
| 예산 128 / 192 | 악화 (c32-fixed p99 243 → 383 / 301) | −0.1~−22.8% | 내려감 | 기각 |
| chunk 128 | 거의 그대로 | −0.6~−1.2% | 그대로 | 기각 |
| 예산 384 / 512 (상한 512 진단) | 10~25% 개선, 대부분 여전히 vLLM 초과 | −1.5~−7.4% | 필수 기준 초과 | 기각 |
| 계단 128,192 미루기 | 섞임 (±5%) | 0~+2.5% | 내려감 | TPOT·처리량 수단으로 채택 후보 |
| tensor-core attention (§105 행 단위 → 최종은 §108 page producer) | 4~14% 개선 (c32-natural 286 / 359 → 246 / 312; 최종 구성 240 / 325) | +2~7% | 내려감 | 채택(§108) |

이 항목은 **"tensor-core attention으로 얻은 개선까지 반영하고, 그 이상은 필수 항목과의 trade-off 때문에 닫음"**으로 정리합니다. 남는 수단은 burst일 때만 1,024행 이상의 prefill step을 허용하는 적응형 예산인데, executor 상한·workspace 재설계가 필요하고 예산 512 실측으로 볼 때 C32 p99를 vLLM 아래로 내린다는 근거가 아직 없습니다.

## 107. 행 단위 kernel의 C1 회귀와 원인 — 구조를 바꿈 (page 상태를 만드는 tensor-core producer)

§105의 v3 연산을 유지한 채 가중치·값 pass를 합친 v4(binary `ea8debc7…`, v3와 출력 15/15 동일, HF 13/15, gate 55)로 8조건 A/B를 했습니다(`qwen-matrix/goal2-rowwise/`, attempt 89~92; 기준은 같은 binary의 기존 flag).

| 조건 | 기준 tok/s | v4 | TPOT p50 / p95 ms 기준 → v4 |
|---|---:|---:|---|
| c1-fixed | 127.1 | 127.4 (+0.2%) | 7.70 / 7.71 → 7.72 / 7.74 |
| **c1-natural** | 127.5 | **124.8 (−2.1%)** | 7.71 / 7.75 → 7.74 / **8.08** |
| c8-fixed | 814.8 | 835.5 (+2.5%) | 8.77 → 8.60 |
| c8-natural | 874.5 | 876.6 (+0.2%) | 8.63 → 8.61 |
| c16-fixed | 1349.1 | 1408.5 (+4.4%) | 10.44 → 10.01 |
| c16-natural | 1542.5 | 1581.6 (+2.5%) | 9.43 → 9.19 |
| c32-fixed | 1922.4 | 2049.0 (+6.6%) | 13.53 → 12.68 |
| c32-natural | 2418.9 | 2592.5 (+7.2%) | 11.38 → 10.49 |

**c1-natural의 처리량이 2.1% 떨어져 "다른 핵심 조건을 0.5% 넘게 낮추면 채택하지 않는다"에 걸립니다.** 긴 문맥(398-token prompt, 문맥 398~526) 요청의 TPOT가 7.75 → 8.08 ms로 늘어난 것이 전부입니다. 행 단위 kernel은 미채택으로 돌렸습니다.

### 107.1 원인 분리 (단독 harness)

runtime의 kernel 원문을 그대로 include해 변형끼리 bit 비교와 시간 측정을 하는 harness를 만들었습니다(`qwen-resume-diagnostics/kbench/`: `rowwise_kernel_bench*.cu` + `kernel_*.inc`, 결과 `bench-*.log`; 36 layer graph 재생, layer당 µs).

| 형상 | 기존 (producer + merge) | 행 단위 v4 |
|---|---:|---:|
| C1 decode, 문맥 40 / 140 / 270 / 400 / 500 / 1000 | 6.7 / 6.8 / 7.0 / 7.0 / 6.9 / 8.2 | 4.7 / 8.0 / 11.2 / 14.6 / 16.1 / 29.4 |
| C8 natural decode | 12.5 | 15.4 |
| C32 natural decode | 21.5 | 19.0 |
| 혼합 256행 (c32-natural 두 형상) | 50.9 / 80.1 | 29.4 / 35.9 |

행 단위 kernel은 문맥 길이에 비례해 느려집니다. 변형을 하나씩 빼 보며 원인을 찾았습니다(문맥 500의 16.1 µs 기준):

| 변형 (같은 연산이면 bit 동일 확인) | 문맥 500 | 해석 |
|---|---:|---|
| v5: 뒤에 쓸 tile을 volatile load로 미리 건드림 | 18.8 | 지연 겹치기 효과 없음(오히려 느림) → 미채택. 서버 A/B attempt 93·94는 **무효**: rsync가 수정 시각을 보존해 cargo가 kernel을 다시 빌드하지 않았고, 측정된 binary는 v4와 같은 `ea8debc7…`였습니다(두 attempt는 v4의 반복 측정으로만 읽을 수 있음). |
| 조각 MMA 순서를 step 간 교차(piece-major) | 16.0 | MMA 의존 사슬 아님 |
| 가중치 조각 1개만 (진단) | 14.9 | 조각 3개의 비용은 1.2 µs뿐 |
| 점수를 저장하지 않고 다시 계산 (진단) | 21+ | 저장이 이득 |
| pass 2를 합계만 (진단) | 9.3 | 값 쪽 약 6.7 µs |
| 위에서 점수 MMA 제거 (진단) | 8.9 | MMA 0.2 µs |
| 위에서 K tile 접근 제거 (진단) | 4.05 | **K tile을 읽는 데 약 4.9 µs** |
| shared memory에 tile을 올리고 ldmatrix로 읽기 / 16-byte load | 15.4 / 15.3 | load 명령 수·형태와 무관 |

결론: **SM 하나가 L2에서 끌어올 수 있는 양이 약 26 KB/µs로 묶여 있습니다.** 행 단위 구조는 한 행의 모든 page(page당 K 4 KB + V 4 KB)를 (행, KV head)당 CTA 하나, 곧 SM 하나로 읽으므로, 문맥 500 token(33 page, 264 KB)이면 그것만 10 µs입니다. 기존 구조는 page마다 CTA가 달라 SM 여러 개가 나눠 읽습니다. 동시성이 높거나 prefill이 섞이면 SM이 이미 다 차 있어 행 단위가 이기지만, 동시성 1의 긴 문맥에서는 구조적으로 집니다. 연산 순서를 유지한 채(HF 13/15) page 병렬로 바꾸려면 kernel이 3개 필요해 C1에서 기존보다 느립니다.

### 107.2 다시 설계 — page 구조는 그대로, producer만 tensor core로

harness에서 세 구조를 비교했습니다(모두 가중치는 BF16 조각 3개로 정확히 실음; layer당 µs).

| 형상 | 기존 | 행 단위 v4 | chain(8개) CTA + chain merge | **page별 CTA(warp 1개) + 기존 global merge** |
|---|---:|---:|---:|---:|
| C1 decode 문맥 40 / 140 / 500 / 1000 | 6.8 / 6.4 / 6.8 / 8.1 | 4.7 / 7.7 / 15.9 / 29.2 | 5.9 / 7.7 / 10.6 / 16.4 | **4.6 / 4.6 / 4.8 / 6.2** |
| C8 natural decode | 12.4 | 15.3 | 10.8 | **6.5** |
| C16 natural decode | 15.4 | 17.9 | 14.0 | **9.0** |
| C32 fixed / natural decode | 16.5 / 21.4 | 11.0 / 19.0 | 12.8 / 16.3 | **10.3 / 13.7** |
| c16-fixed 혼합 141행 | 25.7 | 16.2 | 14.8 | 15.0 |
| c32-fixed 혼합 256행 | 40.0 | 20.7 | 21.1 | 22.4 |
| c32-natural 혼합 256행 (두 형상) | 50.7 / 78.5 | 29.3 / 36.0 | 24.3 / 29.0 | 32.3 / 37.5 |

채택한 구조(마지막 열): 기존 shared-KV producer와 **같은 launch grid(page × KV head × 행)·같은 partial state 배치**를 쓰되, CTA는 warp 하나이고 그 page의 상태 `[m, l, n[128]]`를 KV head의 query head 8개에 대해 한 번에 만듭니다. merge kernel과 workspace는 그대로입니다.
1. 점수 = BF16(BF16(q·k) × scale): MMA 8회(token × head).
2. m = page 안 점수의 최댓값, e = exp(점수 − m), l = Σe.
3. n = Σ e·v: e를 BF16 조각 3개로 정확히 분해해 조각마다 MMA 1회.
page 상태는 자기 page와 자기 행에만 의존하므로 동시성·chunk와 무관하고, 기존 producer와는 반올림만 다릅니다(harness에서 출력 원소의 99.95~100%가 기존 경로와 bit 동일). 동시성 1에서도 문맥 길이와 무관하게 기존보다 빠릅니다.

구현: `native_bf16_ragged_paged_split_gqa_d128_tensor_core_partial_state_kernel`(V3 ABI `reserved[0] = 2`, merge는 global-max `reserved[1] = 1`), Rust `native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge`, executor `with_d128_tensor_core_producer()`, 서버 flag `--decode-attention-producer shared-kv-v1|tensor-core-v1`(기본 shared-kv-v1; `--decode-attention-merge global-max-v1` 필요, `--prompt-attention tensor-core-v1`과 배타). GPU 테스트 `tensor_core_producer_gpu`. 행 단위 kernel 코드는 본 코드에서 뺐고 원문은 `qwen-rowwise-kernel-variants/`에 보존했습니다(v2, v3, v4, v5).

## 108. tensor-core page producer (`--decode-attention-producer tensor-core-v1`) — 검증과 8조건 A/B

§107.2의 구조를 runtime에 넣고 두 변형을 검증했습니다. 둘의 차이는 점수 내적의 누산 순서뿐입니다.

| 변형 | 점수 MMA 8회의 누산 | binary | GPU 테스트 (108행) | HF-exact gate | HF token 일치 C1 / C32 | 불일치 prompt (index / 길이) | 판정 |
|---|---|---|---|---|---|---|---|
| tcprod1 | step마다 따로 계산해 pairwise tree로 합산 | `706ca3fd…` | 기존 producer와 bit 동일 221,153 / 221,184 (99.986%), 최대 차이 0.00098, 재배열·chunk 분할 불변 | attempt 56 통과 | 10 / 15, 40 / 60 | arithmetic 14/24, sql 4/16, natural-16 29/32, natural-128 21/64, natural-398 69/128 | 기각 (계약 11/15 미달) |
| **tcprod2** | **한 누산기에 depth 오름차순으로 차례로 누산(tensor-core GEMM이 안쪽 차원을 도는 방식)** | `c08d6611…` | 같음 (99.986%, 0.00098, 불변) | attempt 57 (§109) | **11 / 15, 44 / 60** | editing 28/32, sql 4/16, natural-16 5/32, natural-398 69/128 | **후보** |

- 두 변형의 속도는 같습니다(아래 표). tcprod2를 고른 근거는 (1) HF의 QK가 cuBLAS tensor-core GEMM이라 같은 방식의 누산이 점수의 BF16 반올림을 HF와 같게 만들 가능성이 더 높다는 설계상 이유와 (2) 계약 수치 충족입니다.
- 불일치 위치의 HF logit margin(`hf-margins.json`): tcprod2의 넷 모두 HF 1·2위 logit 차이 0~0.25(BF16 한 칸 0.08~0.13)이고, 그중 셋(sql, natural-16, natural-398)은 cache 없이 다시 계산한 HF의 1위가 Riley token입니다 — HF 자신이 cache 유무로 갈리는 위치입니다.
- **이 지표의 성질**: 15개 prompt 중 서너 개는 HF의 1·2위 logit이 BF16 한두 칸 이내라, 수치적으로 동등한 구현들 사이에서도 일치 수가 9~13으로 움직였습니다(기존 11, 행 단위 v1·v2 9, v3·v4 13, tcprod1 10, tcprod2 11). 모든 변형이 동시성 불변(15/15)이고 기존 경로와의 출력 차이는 BF16 한 칸 수준입니다. 계약(11/15)은 tcprod2로 충족하지만, 이 수치를 품질 차이로 읽어서는 안 됩니다.

8조건 정상상태 A/B (같은 binary 계열, Riley 단독, warmup 32 / retained 192 × 2회, 오류 0, 제외 없음; 기준 = 같은 source의 기존 flag `--prompt-attention tensor-core-v1`; raw `qwen-matrix/goal2-tensor-core-producer/`, attempt 95~100):

| 조건 | 기준 tok/s | tcprod1 | tcprod2 | TPOT p50 ms 기준 → tcprod2 | TTFT p50 / p95 / p99 ms 기준 → tcprod2 | vLLM (final9) tok/s, TPOT p50 |
|---|---:|---:|---:|---|---|---|
| c1-fixed | 127.1 | 129.0 (+1.4%) | 129.1 (+1.5%) | 7.70 → 7.60 | 10.6 / 11.9 / 12.1 → 10.2 / 11.5 / 11.6 | 129.7, 7.40 |
| c1-natural | 127.5 | 129.3 (+1.4%) | 129.3 (+1.4%) | 7.71 → 7.61 | 10.9 / 29.0 / 29.4 → 10.4 / 28.7 / 29.2 | 131.8, 7.44 |
| c8-fixed | 814.2 | 838.2 (+2.9%) | 837.0 (+2.8%) | 8.78 → 8.58 | 21.1 / 33.9 / 59.4 → 20.0 / 30.7 / 56.4 | 761.2, 9.01 |
| c8-natural | 875.1 | 908.8 (+3.9%) | 908.3 (+3.8%) | 8.63 → 8.31 | 19.8 / 39.1 / 75.4 → 19.4 / 38.0 / 68.6 | 837.6, 8.94 |
| c16-fixed | 1349.2 | 1404.2 (+4.1%) | 1398.7 (+3.7%) | 10.45 → 10.04 | 22.2 / 61.9 / 114.4 → 21.3 / 69.0 / 113.0 | 1276.7, 10.31 |
| c16-natural | 1543.1 | 1610.8 (+4.4%) | 1608.5 (+4.2%) | 9.43 → 9.04 | 26.9 / 66.7 / 173.5 → 24.4 / 68.0 / 154.0 | 1495.8, 9.62 |
| c32-fixed | 1927.7 | 2035.0 (+5.6%) | 2035.1 (+5.6%) | 13.54 → 12.76 | 39.3 / 228.9 / 258.0 → 37.9 / 204.8 / 239.1 | 1873.6, 13.51 |
| c32-natural | 2417.6 | 2568.6 (+6.2%) | 2566.4 (+6.2%) | 11.37 → 10.68 | 35.2 / 283.6 / 353.0 → 28.7 / 241.1 / 325.8 | 2489.5, 10.78 |

- **8조건 모두 개선, 회귀 없음.** 사전 기대(§107.2 harness 수치 × step 구성)는 C1 +0.8%, C8 +2.5~3%, C16 +4~5%, C32 +5~9%였고 실측이 그 범위입니다. 15개 prompt 모두 조건·반복 전체에서 출력 1종. GPU 메모리 변화 없음(7.2~8.5 GiB).
- C1이 처음으로 눈에 띄게 움직였습니다: attention 한 쌍이 layer당 약 1.7 µs 줄어 TPOT 7.70 → 7.60 ms. vLLM 대비 c1-fixed 처리량 0.995·TPOT +2.7%, c1-natural 0.981·+2.3%.
- 행 단위 v4(§107)와 비교하면 C1·C8은 더 좋고(c8-natural +3.8% vs +0.2%, c1-natural +1.4% vs −2.1%), C32는 약 1%p 낮습니다(혼합 step에서 merge kernel이 남아 있음: 256행 step에서 layer당 약 9 µs).

## 109. 최종 후보 구성 확정 — tensor-core producer + 계단 128,192 (binary `c08d6611…`, branch head `1229f352`)

HF-exact HTTP gate attempt 57: 후보 binary 그대로 통과(`pass true`, 128 token 일치, `same True`, source 불변; `qwen-http-exact/attempt57-final-candidate/`).

계단 인지 prefill 미루기(§106.1)를 후보 binary에서 다시 쟀습니다(C8~C32, 기준은 flag 없는 두 실행 attempt 99·103의 평균 — 두 실행 사이 편차는 처리량 0.3% 이내; raw `qwen-matrix/goal2-tensor-core-producer/` attempt 101~103). 세 구성 모두 15개 prompt 출력이 같습니다.

| 조건 | 기준 tok/s | 계단 128 | 계단 128,192 | TPOT p50 ms 기준 → 128 / 128,192 | TTFT p50 ms 기준 → 128,192 (vLLM final9) | TTFT p95 / p99 ms 기준 → 128,192 |
|---|---:|---:|---:|---|---|---|
| c8-fixed | 835.7 | 837.0 (+0.2%) | 838.4 (+0.3%) | 8.56 → 8.57 / 8.57 | 20.3 → 20.2 (31.5) | 32.4 / 57.3 → 26.8 / 56.4 |
| c8-natural | 908.6 | 907.7 (−0.1%) | 908.8 (0.0%) | 8.31 → 8.31 / 8.30 | 19.1 → 19.2 (37.4) | 37.8 / 67.5 → 37.4 / 73.5 |
| c16-fixed | 1398.5 | 1432.3 (+2.4%) | 1444.1 (+3.3%) | 10.06 → 9.51 / 9.46 | 21.3 → 27.1 (37.4) | 66.0 / 111.9 → 55.3 / 108.4 |
| c16-natural | 1607.7 | 1609.3 (+0.1%) | 1612.6 (+0.3%) | 9.05 → 9.00 / 8.98 | 24.8 → 31.1 (40.8) | 62.1 / 154.7 → 64.0 / 162.8 |
| c32-fixed | 2032.6 | 2038.0 (+0.3%) | 2049.2 (+0.8%) | 12.77 → 12.57 / 12.51 | 37.9 → 38.4 (54.6) | 210.5 / 242.3 → 216.2 / 244.3 |
| c32-natural | 2566.2 | 2575.3 (+0.4%) | 2582.5 (+0.6%) | 10.67 → 10.56 / 10.55 | 30.8 → 38.0 (63.8) | 239.2 / 326.5 → 243.3 / 322.8 |

판정: **계단 128,192 채택.** 처리량·TPOT p50이 떨어지는 조건이 없고(c16-fixed +3.3%·TPOT −6%, 나머지 0~+0.8%), TTFT p50은 0~7 ms 늘지만 전 조건에서 vLLM보다 10~26 ms 낮습니다. 꼬리(TTFT p95·p99)는 조건마다 ±5% 안에서 엇갈립니다.

최종 후보 flag 묶음(직전 후보 §101 대비 변경은 두 줄): 기존 채택 flag에서 `--prompt-attention tensor-core-v1`을 빼고 `--decode-attention-producer tensor-core-v1 --prefill-plateau-rows 128,192`를 넣음.

## 110. 최종 후보 vs vLLM 0.27.1 — 같은-실행 정상상태 비교 (final10, binary `c08d6611…`, head `1229f352`)

`qwen-matrix/steady-final10/`. final9(§101)와 같은 script에서 Riley의 flag만 최종 묶음으로 바꿨습니다(`--prompt-attention tensor-core-v1` → `--decode-attention-producer tensor-core-v1 --prefill-plateau-rows 128,192`). 8조건 × 2회, 순서 [Riley, vLLM] ↔ [vLLM, Riley], warmup 32 / retained 192, host gate 없음, 오류 0, 제외 없음. Riley 15개 prompt 모두 8조건·반복 전체에서 출력 1종. 측정 중 호스트 부하는 load average 4.9~9.9, IO pressure(full avg10)는 대체로 2~10%에 순간 최대 31%였습니다(`hostload-during-run.log`; 공유 호스트의 다른 작업). 값은 2회 평균(Riley / vLLM).

| 조건 | 처리량 tok/s | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | E2E p50 / p99 ms | GPU MiB |
|---|---|---|---|---|---|
| c1-fixed | 129.1 / 129.7 | 10.2 / 11.5 / 11.6 vs 13.0 / 16.8 / 20.2 | 7.60 / 7.61 / 7.64 vs 7.40 / 7.42 / 7.43 | 155.6 / 246.3 vs 156.1 / 245.9 | 7198 / 15186 |
| c1-natural | 129.3 / 132.0 | 10.4 / 28.8 / 29.3 vs 13.4 / 25.1 / 25.8 | 7.61 / 7.64 / 7.66 vs 7.44 / 7.45 / 7.45 | 489.9 / 1001.3 vs 482.2 / 971.5 | 7328 / 15152 |
| c8-fixed | 836.2 / 762.6 | 20.2 / 31.2 / 56.5 vs 31.9 / 38.2 / 50.8 | 8.57 / 8.94 / 9.18 vs 8.98 / 9.46 / 9.65 | 186.0 / 302.9 vs 202.0 / 327.8 | 7290 / 15152 |
| c8-natural | 908.1 / 835.8 | 18.9 / 37.8 / 65.4 vs 37.3 / 45.5 / 63.2 | 8.31 / 8.36 / 8.57 vs 8.95 / 9.02 / 9.29 | 542.1 / 1110.4 vs 591.3 / 1182.5 | 7634 / 15196 |
| c16-fixed | 1425.5 / 1247.1 | 27.1 / 60.4 / 111.4 vs 37.1 / 98.2 / 159.6 | 9.53 / 9.99 / 10.61 vs 10.29 / 11.90 / 15.26 | 225.6 / 364.0 vs 251.4 / 437.9 | 7388 / 15288 |
| c16-natural | 1612.6 / 1494.5 | 28.8 / 62.3 / 158.2 vs 40.8 / 103.0 / 123.0 | 8.98 / 9.38 / 9.89 vs 9.63 / 9.79 / 9.91 | 596.8 / 1289.2 vs 644.8 / 1319.5 | 7936 / 15296 |
| c32-fixed | 2040.6 / 1874.0 | 38.6 / 222.2 / 247.1 vs 53.8 / 176.1 / 176.6 | 12.42 / 13.23 / 13.52 vs 13.52 / 15.13 / 18.17 | 293.0 / 566.5 vs 324.7 / 559.9 | 7552 / 15322 |
| c32-natural | 2581.8 / 2488.2 | 37.1 / 240.2 / 324.9 vs 63.8 / 241.4 / 242.1 | 10.54 / 12.64 / 12.92 vs 10.76 / 11.53 / 13.11 | 697.4 / 1651.3 vs 740.7 / 1557.4 | 8516 / 15334 |

목표(2026-10-12)의 필수 항목 판정 — Riley / vLLM 비율:

| 조건 | 처리량 (C8+ ≥ 1) | TPOT p50 (C8+ ≤ 1) | TTFT p50 (전 조건 ≤ 1) | 필수 판정 | TTFT p95 | TTFT p99 | TPOT p99 | 반복 간 처리량 편차 R / V |
|---|---:|---:|---:|---|---:|---:|---:|---|
| c1-fixed | 0.995 | 1.027 | 0.782 | 충족 (TTFT p50) | 0.682 | 0.572 | 1.027 | 0.1% / 0.1% |
| c1-natural | 0.979 | 1.022 | 0.775 | 충족 (TTFT p50) | 1.147 | 1.135 | 1.028 | 0.0% / 0.1% |
| c8-fixed | 1.096 | 0.955 | 0.633 | **충족** | 0.816 | 1.111 | 0.952 | 0.2% / 0.5% |
| c8-natural | 1.086 | 0.928 | 0.506 | **충족** | 0.830 | 1.034 | 0.922 | 0.0% / 0.6% |
| c16-fixed | 1.143 | 0.926 | 0.730 | **충족** | 0.615 | 0.698 | 0.695 | 0.2% / 3.8% |
| c16-natural | 1.079 | 0.933 | 0.706 | **충족** | 0.605 | 1.286 | 0.999 | 0.0% / 0.1% |
| c32-fixed | 1.089 | 0.919 | 0.717 | **충족** | 1.262 | 1.399 | 0.744 | 0.1% / 0.2% |
| c32-natural | 1.038 | 0.980 | 0.581 | **충족** | 0.995 | 1.342 | 0.986 | 0.2% / 0.2% |

- **필수 항목 전부 충족**: C8·C16·C32 여섯 조건 모두 처리량이 vLLM보다 3.8~14.3% 높고 TPOT p50이 2.0~8.1% 낮으며, TTFT p50은 여덟 조건 모두 22~49% 낮습니다. 출발점(§101)에서 미달이던 c32-natural 처리량 0.971 → 1.038, c32-natural TPOT p50 1.057 → 0.980, c16-fixed TPOT p50 1.013 → 0.926.
- C1: 처리량 0.995 / 0.979, TPOT p50 1.027 / 1.022 — 네 값 모두 ±3% 안입니다(§101: 0.981 / 0.968, 1.040 / 1.036).
- TTFT p95·p99: vLLM보다 큰 칸은 c1-natural p95·p99(+15% / +14%; 398-token prompt가 예산 256 때문에 두 step에 걸침), c8-fixed p99(+11%), c8-natural p99(+3%), c16-natural p99(+29%), c32-fixed p95·p99(+26% / +40%), c32-natural p99(+34%). 나머지 8칸은 vLLM 이하입니다(§101에서는 16칸 중 11칸이 vLLM 초과, 지금은 8칸).
- vLLM의 c16-fixed는 두 반복이 1223 / 1271 tok/s로 3.8% 벌어졌습니다(Riley는 0.2%; IO pressure가 올랐던 구간과 겹칠 수 있음). 어느 쪽 값으로 봐도 판정은 같습니다.
- GPU 메모리는 Riley 7.2~8.5 GiB, vLLM 15.2~15.3 GiB로 그대로입니다.

## 111. 최종 후보 안정성 — attempt 26 (binary `c08d6611…`, 최종 flag 묶음)

`qwen-scheduler-stability/attempt26-tensor-core-producer-plateau/`. attempt 24(§102)와 같은 script에서 flag만 최종 묶음으로 바꿨습니다. 조건별 300초 지속 부하 + idle 취소 20회 + 부하 중 취소 40회 + 재요청 + 종료 회수.

| 조건 | 결과 | 요청 | 오류 | 불일치 | tok/s 첫 / 끝 round (최소) | TPOT p50 첫 / 끝 ms | GPU MiB 첫¼ / 끝¼ | RSS MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 뒤 KV 블록 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-fixed | 통과 | 2016 | 0 | 0 | 129.0 / 129.0 (129.0) | 7.60 / 7.60 | 7198 / 7198 | 6442.7 / 6443.0 | 20 / 40 | 0 |
| c1-natural | 통과 | 576 | 0 | 0 | 129.3 / 129.3 (129.3) | 7.61 / 7.61 | 7328 / 7328 | 6450.3 / 6450.6 | 20 / 40 | 0 |
| c8-fixed | 통과 | 12192 | 0 | 0 | 807.7 / 807.4 (803.9) | 8.57 / 8.58 | 7299 / 7302 | 6479.1 / 6483.0 | 20 / 40 | 0 |
| c8-natural | 통과 | 3552 | 0 | 0 | 878.2 / 877.8 (876.5) | 8.32 / 8.32 | 7636 / 7652 | 6502.9 / 6518.5 | 20 / 40 | 0 |
| c16-fixed | 통과 | 20064 | 0 | 0 | 1331.8 / 1335.2 (1319.7) | 9.50 / 9.53 | 7393 / 7408 | 6500.3 / 6515.4 | 20 / 40 | 0 |
| c16-natural | 통과 | 6144 | 0 | 0 | 1509.9 / 1512.1 (1508.2) | 9.02 / 9.01 | 7954 / 7972 | 6531.9 / 6550.6 | 20 / 40 | 0 |
| c32-fixed | 통과 | 27552 | 0 | 0 | 1792.9 / 1835.0 (1792.9) | 12.39 / 12.00 | 7580 / 7584 | 6542.3 / 6547.8 | 20 / 40 | 0 |
| c32-natural | 통과 | 9312 | 0 | 0 | 2281.1 / 2299.6 (2281.1) | 10.50 / 10.36 | 8540 / 8563 | 6542.3 / 6566.6 | 20 / 40 | 0 |

8/8 통과: 요청 오류 0, 같은 prompt의 출력 불일치 0(지속 부하 전 구간에서 prompt당 출력 1종), 취소 60회 뒤 재요청 정상, 종료 시 exit 0·GPU 프로세스 없음·active/waiting/KV 블록 0·device/pinned 할당 0. 처리량과 TPOT에 하락 추세 없음(이 script의 처리량은 round 단위 집계라 §110보다 낮게 나옵니다). 300초 동안 GPU 메모리는 0~23 MiB, RSS는 0.3~24 MiB 늘었습니다 — step graph를 기록하는 초기 구간의 증가로, 기록 수 상한(32)은 그대로입니다(attempt 22·24와 같은 양상). **30분 soak은 이 후보로 실행하지 않았습니다**(직전 후보 때의 사용자 지시 §102.1에 따라 요청 전에는 돌리지 않음; 가장 최근의 30분 soak은 §79).

## 112. step 예산 표의 C1 행 (최종 binary) — §103.3·§106의 보충

§103.3의 C1 부분은 두 번 중단돼 비어 있었습니다. 최종 binary `c08d6611…`에서 채웠습니다(Riley 단독, warmup 32 / retained 192 × 2회, 오류 0, 15개 prompt 출력 동일; raw `qwen-matrix/goal2-scheduling/` attempt 104~106, 기준은 attempt 100). 계단 정책은 decode 행이 12개 미만이면 작동하지 않으므로 C1에는 영향이 없습니다.

| 조건 | 구성 | tok/s | 변화 | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | E2E p50 / p99 ms |
|---|---|---:|---:|---|---|---|
| c1-fixed | 예산 256 (기준) | 129.1 | — | 7.60 / 7.61 / 7.63 | 10.2 / 11.5 / 11.6 | 155.5 / 246.3 |
| c1-fixed | 예산 192 | 129.0 | 0.0% | 7.60 / 7.62 / 7.63 | 10.2 / 11.5 / 11.6 | 155.6 / 246.7 |
| c1-fixed | 예산 128 | 127.5 | −1.2% | 7.60 / 7.61 / 7.62 | 11.0 / 18.8 / 18.9 | 159.5 / 247.3 |
| c1-fixed | 예산 256, chunk 128 | 128.1 | −0.7% | 7.60 / 7.61 / 7.63 | 10.2 / 18.0 / 18.1 | 159.1 / 246.4 |
| c1-natural | 예산 256 (기준) | 129.3 | — | 7.61 / 7.64 / 7.65 | 10.4 / 28.7 / 29.2 | 489.7 / 999.4 |
| c1-natural | 예산 192 | 128.8 | −0.5% | 7.61 / 7.65 / 7.66 | 10.3 / 34.6 / 35.1 | 489.8 / 1007.4 |
| c1-natural | 예산 128 | 128.2 | −0.9% | 7.61 / 7.65 / 7.65 | 11.1 / 41.8 / 42.4 | 490.4 / 1013.5 |
| c1-natural | 예산 256, chunk 128 | 128.4 | −0.7% | 7.61 / 7.65 / 7.66 | 10.3 / 39.8 / 40.1 | 489.7 / 1012.6 |

C1에서는 예산·chunk를 줄이면 prompt가 더 많은 step으로 쪼개져 TTFT p95·p99와 처리량만 나빠집니다. c1-natural의 TTFT p95·p99(28.7 / 29.2 ms, vLLM 25.1 / 25.8 ms)는 398-token prompt가 예산 256 때문에 두 step(256 + 142)에 걸치는 데서 오므로, 줄이려면 반대로 **혼자 있는 prompt를 한 step에 넣을 만큼 행 상한을 키워야** 합니다. 행 상한 512 진단 빌드(§106.2)는 step당 비용이 C8에서도 1.4% 늘어(원인 미분리) 그대로는 쓸 수 없었습니다.

## 113. 2026-10-12 목표 기준 최종 상태 (Qwen2.5-3B, binary `c08d6611…`, branch `codex/qwen-http-hf-exact` head `1229f352`)

최종 flag 묶음: §101의 채택 flag에서 `--prompt-attention tensor-core-v1`을 빼고 `--decode-attention-producer tensor-core-v1 --prefill-plateau-rows 128,192`를 넣은 것. 코드는 commit 세 개입니다 — `92dadae0`(scheduler 계단 정책), `71dea99a`(tensor-core page producer와 GPU 테스트), `1229f352`(서버 flag). push하지 않았습니다.

| 목표 항목 | 상태 | 근거 |
|---|---|---|
| 0. SmolLM2 최소 8/8·도전 8/8·안정성·soak 유지 | 유지 | SmolLM2 serving 계열(Batch14 2차 후보)의 source·binary는 이번 작업에서 건드리지 않았습니다(§88~§91 그대로). |
| 0. Qwen 동시성 불변 15/15 | 유지 | final10의 8조건 × 2회, 안정성 8조건 300초, 모든 A/B에서 prompt당 출력 1종(§108~§111). |
| 0. Qwen HF-exact 경로 gate | 통과 | attempt 57, 후보 binary 그대로(§109). |
| 0. Qwen GPU 메모리 수준 | 유지 | 7.2~8.5 GiB(vLLM 15.2~15.3 GiB), §110. |
| 0. 다른 핵심 조건을 0.5% 넘게 낮추지 않음 | 충족 | producer: 8조건 +1.4~6.2%(§108). 계단 정책: 처리량 0~+3.3%, TPOT p50 같거나 낮음(§109). c1-natural을 2.1% 낮춘 행 단위 kernel은 이 규칙으로 기각(§107). |
| 1. 처리량: C8·C16·C32 전 조건 ≥ vLLM | **달성** | R/V 1.096 / 1.086 / 1.143 / 1.079 / 1.089 / 1.038 (§110). 출발점의 미달 c32-natural 0.971 → 1.038. |
| 1. TPOT p50: C8·C16·C32 전 조건 ≤ vLLM | **달성** | R/V 0.955 / 0.928 / 0.926 / 0.933 / 0.919 / 0.980 (§110). 출발점의 미달 c16-fixed 1.013 → 0.926, c32-natural 1.057 → 0.980. |
| 1. TTFT p50: 8조건 모두 ≤ vLLM | **유지** | R/V 0.51~0.78 (§110). |
| 2. C1 처리량·TPOT | **동률 범위(±3%)로 닫음** | 처리량 R/V 0.995 / 0.979, TPOT p50 1.027 / 1.022 — 네 값 모두 ±3% 안(§110; 출발점 0.981 / 0.968, 1.040 / 1.036). 이번에 쓴 수단: tensor-core producer(+1.4~1.5%). 남은 식별 수단(metadata 업로드 묶기, 적은 page 전용 merge, 틈 prefetch)은 각각 추정 0.3~0.6% 이하이고, C1 step의 약 90%는 두 엔진이 똑같이 치르는 가중치 읽기입니다(§92). |
| 2. TTFT p95·p99 | **근거와 함께 닫음** | 현재 빌드에서 다시 한 실험: 예산 128 / 192 / 384 / 512, chunk 128, 계단 정책 — trade-off 표 §103.3·§106·§109·§112. 꼬리는 burst를 처리하는 step 수로 정해지고, step을 키우면 10~25% 줄지만 처리량 1.5~7.4%와 TPOT p50을 잃어 필수 기준을 넘깁니다. tensor-core producer로 얻은 개선은 반영됐습니다(vLLM 초과 칸 16개 중 11 → 8; c32-natural p95 256 → 240 ms로 vLLM 241 ms와 같아짐). 남은 8칸: c1-natural p95·p99, c8 p99 둘, c16-natural p99, c32-fixed p95·p99, c32-natural p99. |
| 3. scheduler 경로 계약: 동시성 불변 15/15 + HF token 일치 11/15 | **충족** | 불변 15/15; HF 11 / 15(C1), 44 / 60(C32) — attempt 12·13(§108, `qwen-hf-token-reference/attempt13-final-flags/`). 불일치 넷은 모두 HF 1·2위 logit 차이 0.25 이하. |
| 3. 수치가 바뀌는 변경의 재검증 | 통과 | native GPU 테스트 `tensor_core_producer_gpu`(재배열·chunk 분할 bit 불변, padding 0, 기존 대비 최대 BF16 한 칸), HF-exact HTTP gate 57, HF token 비교 12·13, 안정성 26. |
| 4. 진행 방식 | 준수 | 실측 병목(§103) → 변경 → 정확성 검증 → 8조건 측정. 기각한 후보와 증거: 예산·chunk 변형(§103.3·§106.2·§112), 다른 GEMM class(§103.2, 기대 이득 0), 행 단위 kernel v1~v5(§105·§107, 원문 `qwen-rowwise-kernel-variants/`), chain 구조(§107.2), tcprod1(§108), 무효 처리한 측정(attempt 82: 외부 부하, attempt 93·94: 재빌드 누락). |

이번 작업에서 확인된 한계와 주의:
- **HF token 일치 수는 불안정한 지표입니다.** 수치적으로 동등한 구현들에서 9~13 사이로 움직였고(§108), 계약 수치 11은 충족했지만 여유는 없습니다. 이후 attention·GEMM의 반올림을 건드리는 변경은 이 수치를 다시 뽑게 됩니다.
- c32-natural의 TPOT p50 여유는 2.0%(10.54 vs 10.76 ms), 처리량 여유는 3.8%입니다. 필수 항목 중 가장 얇습니다.
- 계단 정책의 경계(128, 192)는 이 모델의 projection 형상과 RTX 4090에서 잰 값입니다(§103.2). 다른 모델·GPU에서는 다시 재야 하며, 기본값은 꺼짐입니다.
- 30분 soak은 이 후보로 돌리지 않았습니다(§111).
- branch의 `cargo test -p riley-runtime --lib`에는 이번 변경과 무관하게 실패하는 source-contract 테스트 4건이 있습니다(9cf4dcac에서 이미 실패; 별도 작업으로 표시).
