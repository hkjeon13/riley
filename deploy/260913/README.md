# 260913 — 연구 기반 serving 최적화 PR 계획

상태: **구조별 구현·검증 진행 중, 전체 serving 목표 미달**. 현재 작업 브랜치는 `codex/260913-serving-integration`이다. PR02 실행 ticket·staging·KV 부분 commit, PR03 attention 모델 실험, PR05/07/19 실행 후보의 구현 및 측정 이력이 각 카드와 결과 문서에 있다. 후보별 correctness·성능 실패와 승격 보류를 구분한다. PR20 rolling decode의 모델·serving 통합을 완료했고, PR21 GQA staging은 serving 개선 미확인으로 승격하지 않았다. PR22 prefill projection의 GC 통제·tmpfs C32 비교를 완료했고 vLLM throughput/latency 목표는 미달이다. 같은 조건의 C8 검증까지 완료했으며 shared의 제한된 최소 지표는 만족하지만 unique는 미달이다. C64 검증이 남아 있다. Rust serving 경로에 Python을 도입하지 않는다. 계획 커밋은 `894f0713`이며, 기존 deploy 계획의 완료 여부를 소급 변경하지 않는다.

## 목표와 현재 증거

동일 모델·하드웨어·워크로드에서 vLLM 이상의 throughput, 동등 이하 TTFT/TPOT, 높은 concurrency에서 tail·안정성 유지가 목표다. 목표 개선폭은 throughput +15%, TTFT/TPOT 각각 −10%다. 개별 PR 완료와 전체 목표 달성을 구별한다.

원격 V56 기준 revision은 `030c207565eda3ec45533a166e6d9a201c8d1831`이다. 로컬 checkout과 원격 실험 코드는 같다고 가정하지 않는다. 착수 시 revision·dirty state를 확인하고 필요한 변경만 이식한다. 현재 구현에는 continuous batching, paged KV, mixed execution, CUDA Graph, GPU sampling이 이미 있으므로 다시 만드는 계획으로 취급하지 않는다.

기존 V56 serving 측정은 C32 natural에서 vLLM 대비 throughput 약 −12.69%, TPOT도 불리했다. 전체 목표는 미달이다. V57은 primitive 검사이며 application serving 개선 증거가 아니다. 이번 계획 작성 직전 FlashInfer 직접 비교에서는 context 1만 exact였고 16/398/4096은 Riley와 달랐다. 이는 수치 호환성 문제를 확인한 제한적 probe이며 full-model 품질 판정이 아니다.

근거: [V56 serving](../../benchmarks/results/20260912-serving-optimization/V56_SERVING_RESULTS.md), [측정 종료](../../benchmarks/results/20260912-serving-optimization/MEASUREMENT_CLOSEOUT_V57.md), [FlashInfer 직접 비교 raw](../../benchmarks/results/20260912-serving-optimization/raw/flashinfer-source-audit/riley-compare.log).

## 실행 순서와 PR 크기

01로 공통 계약을 정한 뒤 **02 비동기 실행**과 **03 attention adapter**를 첫 구조 batch로 진행한다. 04·05·06은 자원 중첩과 memory/work 분할을 다룬다. 07은 큰 실행 구조의 feasibility PR이며 긍정 결과 후 전 layer serving 확장을 새 PR로 구체화한다. 08~12는 장비 확장 경로다. 13~15는 모델·정밀도별 조건부 경로다. 16의 측정 계약은 첫 batch부터 적용하며 마지막에 조합 전체를 다시 판정한다.

숫자는 의존성이 없는 PR까지 직렬로 강제하는 순서가 아니다. 각 파일은 2~5개 연관 변경을 묶은 리뷰 단위다. 개별 파일 안의 변경을 다시 몇 줄짜리 최적화 PR로 쪼개지 않는다. 반대로 전체 serving engine·compiler·cluster control plane을 하나의 PR로 만들지 않는다. 구현 상태와 남은 범위는 각 카드의 최신 검증 결과를 기준으로 확인한다.

| PR | 범위 | 선행 |
|---|---|---|
| 01 | [실행 backend 계약과 하드웨어 검증 기반](01-execution-contract-and-hardware-validation.md) | 없음 |
| 02 | [비동기 iteration 실행과 응답 처리 중첩](02-asynchronous-serving-execution.md) | 01 |
| 03 | [FlashInfer paged attention 실행층 통합](03-attention-backend-adapter.md) | 01; 02와 독립 개발 가능 |
| 04 | [POD 실행과 시간 예산 기반 mixed batching](04-mixed-prefill-decode-policy.md) | 03 |
| 05 | [메모리 수명 기반 FFN tile 실행](05-memory-aware-ffn-execution.md) | 01 |
| 06 | [Stream-K·LeanAttention 방식 작업 분할](06-work-balanced-decode.md) | 03 |
| 07 | [persistent task graph의 한 layer 실행 실험](07-persistent-layer-execution.md) | 01, 05; 03 attention 계약 재사용 |
| 08 | [Hopper 비동기 attention backend](08-hopper-attention-backend.md) | 01, 03 |
| 09 | [Blackwell attention pipeline](09-blackwell-attention-backend.md) | 01, 03; 08은 필수 의존 아님 |
| 10 | [KV export/import와 공유 prefix 소유권](10-kv-transfer-and-prefix-ownership.md) | 01; 02의 ticket 수명과 호환 |
| 11 | [고정 P/D worker의 분리 serving](11-prefill-decode-disaggregation.md) | 10, 04 |
| 12 | [Dense 모델 tensor parallel 실행](12-dense-tensor-parallel.md) | 01; 10과 transport 원칙 공유하되 필수 의존 아님 |
| 13 | [Draft-target speculative decoding](13-speculative-decoding.md) | 01, 03; draft/target 모델 로딩 지원을 착수 전에 확인 |
| 14 | [KV 양자화와 fused attention](14-quantized-kv-attention.md) | 01, 03, 10의 layout identity와 호환 |
| 15 | [저정밀 weight와 fused dequant GEMM](15-weight-quantized-gemm.md) | 01, 05 |
| 16 | [실제 serving 평가와 후보 승격](16-serving-evaluation-and-promotion.md) | 01; 후보 PR마다 재사용, 마지막에 전체 조합 평가 |

- [17 MoE expert dispatch/combine](17-moe-expert-parallel.md): 01, 12의 rank·collective 수명 계약. 추가 선행: 지원 MoE 모델 하나의 loader, router, 단일 device reference forward. 이 선행이 없으면 EP를 동작한다고 주장하지 않으며 모델 지원을 별도 PR로 먼저 구체화한다.
- [18 SLO 기반 aggregated/P-D routing](18-adaptive-serving-routing.md): 10, 11; 정책 비용을 정할 고정 aggregated/split 측정 자료. 장비가 없으면 비용 모델을 주입한 상태 전이 검증과 구현은 진행한다.
- [19 Persistent layer 실행의 전 layer·serving 통합](19-persistent-serving-integration.md): 01, 02, 07. 07의 task·buffer 계약이 구현되어야 한다. 장비 부재로 07 GPU 검증이 skip되어도 비활성 경로 구현은 진행할 수 있으나 기본값 승격은 실제 검증 후다.

## 공통 하드웨어·테스트 지침

4090은 현재 검증 장비이며 연구·구현 지원 범위의 상한이 아니다. 필요한 Hopper·Blackwell·multi-GPU 기능은 개발하고, 실제 장비가 필요한 테스트만 skip한다. required capability, observed capability, skip 이유, 실제 실행 backend를 기록한다. compile toolchain조차 없으면 compile 미검증도 명시한다. 테스트를 실행하여 실패한 경우에는 장비 부재 skip으로 바꾸지 않는다.

CPU 상태 전이·ABI·plan·가능한 compilation과 4090 지원 경로 회귀는 계속 실행한다. 전용 장비 CI에서는 요구 장비가 있는데도 테스트가 skip되지 않는지 확인한다. 기능 구현과 해당 GPU에서 검증 완료를 따로 기록한다. 검증 대기 feature는 기본 비활성으로 둘 수 있으며 후속 개발은 계속할 수 있다.

multi-GPU에서는 GPU 총수·SKU·memory·peer access·NVLink/PCIe/network topology를 비교 엔진과 맞춘다. mock이나 한 GPU의 가상 rank 검증은 실제 통신 및 성능 증거가 아니다. Hopper/Blackwell 기능은 제품군 이름만으로 지원을 판정하지 않고 instruction·toolchain 요구를 검사한다.

## 공통 correctness·측정·승격 계약

- 현행 exact 경로는 보존한다. 새로운 reduction 순서나 quantization은 별도 모드로 구분하고 허용 오차·품질 기준을 결과를 보기 전에 명시한다. 특정 tolerance 숫자를 근거 없이 이 계획에서 발명하지 않는다. exact 불일치 backend는 기존 exact 모드에서 선택하지 않는다.
- 각 optimization batch는 변경 전 frozen baseline과 같은 날의 vLLM에 비교한다. TRT-LLM·SGLang도 지원되는 같은 조합으로 포함하며 미지원 조합은 이유를 남긴다. 모든 수치를 동일 precision 결과처럼 합치지 않는다.
- 기존 SmolLM2-135M BF16 C16/C32 fixed/natural을 회귀 축으로 유지한다. 큰 dense 모델·긴 context·높은 concurrency·공유 prefix·open-loop 부하를 추가하되, 모델과 runtime 지원을 확인한다. C64 이상의 실제 지원 구현 없이 측정했다고 주장하지 않는다.
- 측정 전에 checkpoint/tokenizer hash, dtype/KV dtype, EOS·sampling·출력 길이, warmup/retained request 수, 도착률, TTFT/TPOT SLO, tail 집계 단위, 반복 횟수, soak 시간과 실패 허용 기준을 manifest에 고정한다. 이 수치가 미정이면 성능 측정을 시작하지 않는다.
- 교차 순서 반복으로 분산과 불확실성을 보고하고, 미량 차이를 승리로 단정하지 않는다. profiler·build는 timed run과 분리한다. throughput과 latency를 tradeoff한 결과는 동시 목표 달성으로 처리하지 않는다.
- PR 결과는 구현 여부, hardware validation, correctness, performance promotion을 별도 기록한다. 검증 대기는 pass가 아니다. 성능 향상이 없으면 원인을 분석하고 not-promoted/rejected로 남긴다. 원본 raw와 revision·binary hash·argv를 보존한다.
- 최종 판정은 16의 serving 평가다. microbenchmark·모의 실행·compile 성공·논문 배수로 목표를 완료 처리하지 않는다.

Blender는 최신 사용자 지시에 따라 GPU 검증·측정에 필요할 때만 잠시 종료하고, 검증 완료 후 GPU가 불필요한 연구·소스 작업 동안에는 복구한다. 기존 세 작업의 파일·시작 스크립트·포트(9876/9911/9887)를 보존하고 복구 후 프로세스와 포트 응답을 확인한다. 과거 benchmark의 Blender-down 기록은 당시 측정 조건으로 유지한다. GUI 및 다른 작업을 건드리지 않으며 기존 unrelated dirty 변경과 live data를 보존한다. 이전 문서의 별도 승인 문구를 이 계획에 재도입하지 않는다. 각 카드의 완료 기준과 실제 결과를 대조하기 전에는 구현·성능 승격을 주장하지 않는다.

## 연구와 PR 연결

| 연구 묶음 | 적용 PR |
|---|---|
| SGLang / TensorRT-LLM overlap, vLLM graph 실행 | 01, 02, 03 |
| FlashInfer / FlashAttention / POD / Sarathi / NanoFlow | 03, 04, 06, 12 |
| HBM IO·Welder·MCFuser·Stream-K·LeanAttention | 05, 06 |
| MPK / Mirage / Ada-MK / ThunderKittens | 07, 19 |
| FA3 / FA4 | 08, 09 |
| RadixAttention / Mooncake / NIXL / DistServe / Dynamo / TaiChi | 10, 11, 18 |
| ParallelKittens / DeepEP | 12, 17; custom fused collective는 기본 TP 검증 이후 후보 |
| EAGLE-3 / speculative decoding | 13 |
| KIVI / QServe / TurboQuant / FLUTE | 14, 15 |
| FlashInfer-Bench 및 실제 serving 평가 | 16 |

17~19는 후속 구조 PR이다. MoE 모델 지원은 17의 명시적 선행이며 현재 dense 모델 범위와 구별한다. 18은 고정 worker pool의 신규 요청 routing, 19는 iteration 단위 persistent 실행으로 제한한다. 전체 GPU serving scheduler나 클러스터 autoscaler를 한 PR에 넣지 않는다. Fast-TurboQuant는 초록 수준 선별이므로 본문·구현 검토 이후 적용 계획을 정한다.

상세 근거는 [전체 연구](../../benchmarks/results/20260912-serving-optimization/RESEARCH_20260913.md), [HBM·CUDA 연구](../../benchmarks/results/20260912-serving-optimization/HBM_CUDA_RESEARCH_20260913.md), [현재 코드 접점](../../benchmarks/results/20260912-serving-optimization/RESEARCH_CODE_FIT_20260913.md)을 참조한다. 각 PR은 위 연구의 Riley 적용 제안이며 논문 주장을 현재 구현의 사실로 취급하지 않는다.

## Serving 언어 경계 — 사용자 확정 제약

Serving 실행 경로에 Rust ↔ Python 호출을 도입하지 않는다. Scheduler, model 실행, attention/KV 관리, sampling 등 요청 처리 중 Python interpreter, PyO3, Python subprocess, Python worker/RPC에 의존하지 않는다. 외부 kernel 라이브러리는 Rust → C/C++ ABI → CUDA 경로로 연결한다. Python은 필요한 빌드·오프라인 연구·검증 도구에서만 사용할 수 있다. 이는 2026-09-13 사용자 지시이며 후속 PR에도 적용한다.

## 비교표 보고 주기 — 사용자 확정 지시

작은 수정마다 반복하지 않고 Scheduler·attention·KV 등 의미 있는 구현/도입 묶음이 serving에서 실행·검증 가능한 수준에 도달할 때, 직전 Riley baseline·새 Riley·vLLM을 동일 조건에서 비교하고 결과를 표로 남긴다. Throughput, TTFT/TPOT, P95/P99 latency, 오류율, correctness, revision·모델·하드웨어·workload·수치 profile·반복 횟수 및 raw 증거를 포함한다. 새 측정이 없으면 미측정으로 표시하며 primitive/모델 검사 시간을 serving 수치로 대체하지 않는다. 보고 형식은 [비교표 양식](BENCHMARK_REPORT_TEMPLATE.md)을 따른다.

## 추가 통합 PR

- [PR20 — Rolling decode pipeline](20-rolling-decode-pipeline.md): KV·scheduler·runtime·server 통합 및 C8/C32/C64 screen 완료. Opt-in이며 기본값 승격·전체 qualification은 미완료다. C8 shared 결과를 다른 workload/concurrency의 성공으로 확대하지 않는다.
- [PR21 — GQA shared staging](21-gqa-shared-kv-staging.md): native/full-model gate 및 C8/C32/C64 serving 완료. Native 이득이 serving에서 재현되지 않아 승격하지 않았다.
- [PR22 — Prefill projection pipeline](22-prefill-projection-pipeline.md): native/full-model 및 긴 C32 비교 완료. 기존 Riley 대비 throughput 개선은 확인됐지만 vLLM/전체 latency 목표에는 미달이다. C8/C64 확대 전에 측정 클라이언트 GC와 결과 기록 IO 영향을 통제한 전체 비교를 실행 중이다.

[Client GC 대조 실험](../../benchmarks/results/20260914-client-gc-intervention/README.md)은 오프라인 Python 부하 생성기의 측정 조건 검사다. Riley 서버의 GC 검사 또는 서버 최적화 이득이 아니다. Riley의 큰 P99 일부는 GC 비활성 시 감소했지만 vLLM의 한 반복에는 GC 없이도 큰 tail과 host IO pressure가 남았다. 전체 run을 보존하고 중앙값만으로 목표 달성을 주장하지 않는다. 다음 비교는 모든 엔진에 같은 `gc-phase-disabled-v1` 및 tmpfs 원본 응답·로그 기록을 적용하며, 완료된 수치와 혼합하지 않는다.

## Blender 복구 — 2026-09-14

사용자 로그인 없이 별도 Xvfb 가상 디스플레이에서 기존 세 `.blend` 파일과 시작 스크립트로 복구했다. 기존 MCP 애드온은 `blender -b`를 거부하므로 가상 화면에서 기존 이벤트 루프를 유지한다. 포트9876/9911/9887 모두 `get_scene_info` 성공 응답을 확인했다. 시스템 패키지를 변경하지 않고 Ubuntu 패키지를 별도 폴더에 추출했다.

원격 `/data/riley-serving-260913-recovery/blender-restoration-260914`의 `launch.json`, `xvfb-launch.json`, `verified.json`에 실행 및 검증 기록이 있다. 인증 cookie는 이 원격 폴더의 private 파일로만 보관하며 저장소에 복사하지 않는다. 다음 측정은 해당 receipt의 프로세스 identity와 현재 상태를 재확인한 뒤 기존 세 작업만 종료하고, GPU 작업 종료 시 복구한다. 가상 디스플레이를 사용하는 현재 세션은 이전 실제 화면의 GUI 세션이나 미저장 상태 복구를 의미하지 않는다.

`3d.fin-ally.net`은 별도 정적 웹 뷰어다. Cloudflare origin은 localhost:31840이며 기존 `/home/psyche/lotte-tower/output/scripts/serve_local.py --directory /home/psyche/lotte-tower/output/web/site --port 31840`로 복구했다. 공개 HTTP200, GLB Range206, 실제 브라우저의 타워 렌더·탐색 가능 상태를 확인했다. 해당 웹 서버는 Blender MCP나 서버 GPU에 의존하지 않으므로 이후 GPU 측정 중에도 유지한다. 실행 기록은 위 원격 폴더의 `viewer-launch.json`이다.

비교 뷰어도 복구했다: `3dsol.fin-ally.net`→127.0.0.1:31970 (`codex_gpt5_20260908_132327/output/web/site`), `3dfable.fin-ally.net`→127.0.0.1:32010 (`claude_fable51_20260908_1410/output/web`). 각 기존 문서의 Python 정적 서버 명령을 사용했다. 두 공개 사이트 HTTP200 및 브라우저 모델 로딩 완료·실제 타워 렌더를 확인했다. 실행 기록은 `3dsol-launch.json`, `3dfable-launch.json`이다. 세 웹 서버 모두 GPU 측정 대상에서 제외하고 계속 유지한다.

## Projection GC 통제 concurrency matrix 완료

[C8/C32/C64 전체 표](../../benchmarks/results/20260914-projection-serving-matrix/README.md): 393216 retained 요청을 검증했다. C64 후보 throughput은 vLLM 대비 shared −21.46%, unique −17.26%다. C8 shared의 제한된 최소 지표 통과 외 전체 목표는 미달이며 기본값 승격은 보류한다. C64 shared의 동일 binary control 대비 E2E P95/P99 및 ITL P99 회귀도 보존했다. 다음은 고정 후보/control의 bounded C32 Nsight profile로 남은 prefill·decode·graph 대기를 분석한 후 영역 단위 optimization batch를 결정한다.
