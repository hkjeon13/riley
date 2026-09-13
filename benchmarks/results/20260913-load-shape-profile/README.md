# C32/C64 matched-request serving profile

현재 `a471d457` runtime에서 C32와 C64의 실행 구성을 비교했다. **C64에서 prefill 비중이나 graph 사이 대기가 크게 증가했다는 근거는 찾지 못했다.** 두 조건 모두 paired의 prefill/mixed가 GPU graph envelope 시간의 약53%를 차지한다. 다음 영역은 PR05의 prefill FFN data-transfer pipeline이며, 다음 pair 예약 구조를 바로 확장하는 방향은 우선순위를 낮춘다.

## 조건과 완료 증거

RTX4090, SmolLM2-135M, 동일 frozen natural requests384개/lane, active32, client concurrency32/64, single/paired. Binary SHA는 네 lane 모두 `a5036ff769a960d48dbd96fc971b6a5a3899c32c267c86da72137df76c0f0eaf`다. 각 lane의 launch JSON에 argv·fixture/client/controller hash·요청 수·client concurrency·active capacity가 있다. 측정 도구에 client concurrency/요청 수를 명시하는 bounded 옵션을 추가했으며 runtime은 바꾸지 않았다.

네 lane 총1,536개 HTTP 요청이 frozen reference exact이고 모두 exit0, 잔여 소유 process0이다. Peak owned-tree RSS는 약1.52~1.54GB였다. 기존8GiB sampled RSS/360초 watchdog과 native owner별 종료를 유지했다. 측정 후 GPU compute process가 없는 것을 확인했다. Controller 소유 process 및 기존 paired trace 분석 테스트3개 통과. `raw-manifest.json`의 JSON 복사 및 analysis/area SQLite SHA를 대조했다. 원본 Nsight/SQLite/full HTTP rows는 `ai-assistant:/data/riley-serving-260913-recovery/load-shape-c{32,64}-trace-v1`에 보존한다.

## Trace 결과

아래는 각 trace의 launch 개수 기준 중간80%다. 시작·종료를 정확히 제거한 steady state가 아니며 profiler와 client pacing이 포함된다. 동일384요청이어도 선택된 batch/launch 경계가 달라질 수 있다. Trace window를 나누어 serving throughput 개선율을 계산하지 않는다.

| Client C | Lane | Launch 수 | GPU window ms | Graph 사이 gap ms | Prefill/mixed graph ms | Graph 시간 중 prefill/mixed |
|---|---|---:|---:|---:|---:|---:|
| 32 | single | 790 | 2474.15 | 498.54 | 1081.97 | 54.77% |
| 32 | paired | 810 | 2337.94 | 390.52 | 1036.43 | 53.22% |
| 64 | single | 789 | 2477.99 | 489.46 | 1108.33 | 55.74% |
| 64 | paired | 790 | 2317.88 | 396.72 | 1026.10 | 53.41% |

[전체 집계](comparison.json). Paired의 future graph는 선택된 decode launch의 C32 43.39%, C64 42.43%다. 이 비율은 selected successor launch 비율이며 전체 요청의 pair eligibility나 정확한 row occupancy를 뜻하지 않는다. C64의 작은 unprofiled 이득을 실행 구성 변화만으로 설명할 수 없다. [직전 unprofiled C32/C64 결과](../20260913-overlap-successor-preparation/README.md)가 serving 성능 판정의 근거이며 이번에는 새 serving 성능표를 만들지 않는다.

C64 paired prefill/mixed의 kernel duration 합은 attention35.35%, 기타 projection/QKV/RoPE31.63%, FFN27.46%, norm3.87%다. Kernel duration의 합은 graph envelope나 wall time과 다르며 HBM bandwidth를 측정한 것이 아니다. 대표 FFN kernel은 fused `riley_prefill51::gate_up<4>`131.24ms와 `gemm_prefill_shape_vector<576,1536,320,2,true>`139.04ms다. 둘을 합친270.28ms는 현재 선택 구간의 구체적인 개선 대상이다.

## 다음 batch 선택

`mixed_model_v49.cuh`의 prefill FFN은 이미 gate/up/SwiGLU를 fuse한다. 이를 새 fusion으로 주장하지 않는다. 현재 `prefill_fused_gate_v51.cuh`와 `prefill_shape_projection.cuh`는 각16-wide K step에서 register로 global load한 다음 MMA를 수행한다. 다음에는 기존 MMA 순서·BF16 rounding을 보존하면서 CTA 공유 입력 staging과 asynchronous copy/MMA pipeline을 gate/up와 down 두 kernel에 함께 연결한다. Logical load/staging 구조가 실제 HBM traffic이나 속도를 줄인다는 주장은 검증 전에는 하지 않는다.

[PR05의 구체적인 범위와 gate](../../../deploy/260913/05-memory-aware-ffn-execution.md)에 기록했다. Native/모델/serving과 SM90a/SM100a compile·runtime skip을 구분하고, C32뿐 아니라 client C64/active32에서 직전 후보 및 vLLM을 비교해야 한다. 이미 모델 품질/성능 gate를 실패한 FlashInfer compensation·FP16 변환 계열이나 회귀한 monolithic attention→FFN 확대는 이번 증거로 재개하지 않는다.

새 prefill pipeline은 아직 미구현이다. Default single, opt-in paired 상태를 유지하며 최종 serving 목표는 미달성이다.
