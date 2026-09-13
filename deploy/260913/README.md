# 260913 — 연구 기반 serving 최적화 PR 계획

상태: **계획만 작성**. 이 요청으로 application 구현·commit·배포·새 serving 측정을 수행하지 않는다. 기존 deploy 계획의 완료 여부를 소급 변경하지 않는다.

## 목표와 현재 증거

동일 모델·하드웨어·워크로드에서 vLLM 이상의 throughput, 동등 이하 TTFT/TPOT, 높은 concurrency에서 tail·안정성 유지가 목표다. 목표 개선폭은 throughput +15%, TTFT/TPOT 각각 −10%다. 개별 PR 완료와 전체 목표 달성을 구별한다.

원격 V56 기준 revision은 `030c207565eda3ec45533a166e6d9a201c8d1831`이다. 로컬 checkout과 원격 실험 코드는 같다고 가정하지 않는다. 착수 시 revision·dirty state를 확인하고 필요한 변경만 이식한다. 현재 구현에는 continuous batching, paged KV, mixed execution, CUDA Graph, GPU sampling이 이미 있으므로 다시 만드는 계획으로 취급하지 않는다.

기존 V56 serving 측정은 C32 natural에서 vLLM 대비 throughput 약 −12.69%, TPOT도 불리했다. 전체 목표는 미달이다. V57은 primitive 검사이며 application serving 개선 증거가 아니다. 이번 계획 작성 직전 FlashInfer 직접 비교에서는 context 1만 exact였고 16/398/4096은 Riley와 달랐다. 이는 수치 호환성 문제를 확인한 제한적 probe이며 full-model 품질 판정이 아니다.

근거: [V56 serving](../../benchmarks/results/20260912-serving-optimization/V56_SERVING_RESULTS.md), [측정 종료](../../benchmarks/results/20260912-serving-optimization/MEASUREMENT_CLOSEOUT_V57.md), [FlashInfer 직접 비교 raw](../../benchmarks/results/20260912-serving-optimization/raw/flashinfer-source-audit/riley-compare.log).

## 실행 순서와 PR 크기

01로 공통 계약을 정한 뒤 **02 비동기 실행**과 **03 attention adapter**를 첫 구조 batch로 진행한다. 04·05·06은 자원 중첩과 memory/work 분할을 다룬다. 07은 큰 실행 구조의 feasibility PR이며 긍정 결과 후 전 layer serving 확장을 새 PR로 구체화한다. 08~12는 장비 확장 경로다. 13~15는 모델·정밀도별 조건부 경로다. 16의 측정 계약은 첫 batch부터 적용하며 마지막에 조합 전체를 다시 판정한다.

숫자는 의존성이 없는 PR까지 직렬로 강제하는 순서가 아니다. 각 파일은 2~5개 연관 변경을 묶은 리뷰 단위다. 개별 파일 안의 변경을 다시 몇 줄짜리 최적화 PR로 쪼개지 않는다. 반대로 전체 serving engine·compiler·cluster control plane을 하나의 PR로 만들지 않는다. 현재는 모든 PR이 미구현이다.

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

Blender는 사용자 지시에 따라 내려둔 상태를 유지한다. GUI 및 다른 작업을 건드리지 않으며 기존 unrelated dirty 변경과 live data를 보존한다. 이전 문서의 별도 승인 문구를 이 계획에 재도입하지 않는다. 이번 산출물은 계획이며 구현 승격 증거가 아니다.

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
