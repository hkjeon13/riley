# Riley 추론 성능 격차 축소 종합 계획

작성일: 2026-10-20 KST. 상태: 조사·실행 계획. 이 문서 작성으로 새 최적화나 GPU 벤치마크를 실행한 것은 아니다.

실행 업데이트: [v10 구현·검증 결과](qwen-prefill-norm-and-gap-v10-20261020.md). Prefill RMSNorm 배치화·CUDA 실행 구조를 개선하고 기존 정확도와 GPU6/7 홀드를 유지한 C16/C32 fresh5회 비교를 완료했다. 종합 계획 전체의 완료를 뜻하지 않으며 남은 범위는 실행 보고서에 구분했다.

실행 업데이트 v11: [배치 projection·attention·Qwen decode 본체 Graph 구현·검증 결과](qwen-batched-attention-whole-graph-v11-20261020.md). 기존 정확도와 GPU6/7 홀드를 유지한 C16/C32 fresh5회에서 v10 대비 처리량이 각각17.93%·17.59% 증가했다. 실제 새 경로 선택을 trace로 확인했으며, +20% 목표 및 vLLM 격차 해소에는 미달했다.

실행 업데이트 v14 (2026-10-21 KST): [구조 변경 실험·미채택 판정](qwen-mixed-attention-structural-v14-20261021.md). mixed 두-pass 실행, R1 전체 Graph, embedding report 지연 확인, R2~16 merged Gate/Up 및 선택적 softmax/PV 융합을 실제 구현·검사했다. 마지막 D의 정상5회 비교는 v13B 대비 C16 +0.76%, C32 -0.27%였고 반복별2% 초과 tail 플래그3개가 남았다. D의 정확도는 통과했지만 처리량 개선을 입증하지 못해 기본 코드에 반영하지 않는다. 기본 소스는 검증된 v13B를 유지하며 미적용 patch와 실패 표본을 보존한다. GPU6/7의 원래 holder context는 계속 유지했다.

실행 업데이트 v13 (2026-10-21 KST): [mixed MLP piecewise Graph·decode Gate/Up 구현·검증 결과](qwen-piecewise-graph-merged-gateup-v13-20261020.md). 직전 v12B와의 fresh5회 비교에서 처리량은 C16 +1.40%·C32 +2.37%, C16 TPOT p95·p99는 -2.00%·-3.29%였다. 반복별 TTFT/TPOT p95·p99의 2% 초과 악화는 없었다. 원래 정확도와 GPU6/7 context 홀드를 유지했다. vLLM 대비 처리량은61.49%·55.40%로 격차가 크며, R2~16 Gate/Up과 mixed attention·host 실행 구조가 남아 있다.

## 1. 목표와 판단

목표는 original native/HF 정확도 기준과 GPU 6·7의 기존 CUDA context 홀드를 유지하면서 C16 초기 TTFT p95를 안정화하고, C16·C32의 vLLM 대비 처리량 격차를 줄이는 것이다. 개별 커널의 소폭 개선을 반복하는 대신 **실제 활성 행의 가중치 재사용 → prefill 투영 → 수치 경계를 보존하는 융합 → 실행 범위 확대** 순서로 진행한다.

Riley에는 continuous batching, paged KV, shape bucket, Graph, staged upload, 여러 수치 보존 배치 커널이 이미 있다. 새로 도입할 기능 이름보다 현재 strict 경로에서 실제로 얼마나 선택되는지, 메모리 왕복과 GPU 시간이 얼마나 줄었는지를 기준으로 개선한다.

기존 [10월 19일 라이브러리 조사](inference-library-research-and-plan-20261019.md)의 최초 실행 순서를 최신 v9 증거로 갱신한 후속 계획이다. 과거 후보·실패 로그와 원래 정확도 oracle은 변경하지 않는다.

## 2. 출발점: 확인된 결과와 아직 모르는 것

근거: [v9 최종 보고서](qwen-initial-prefill-and-launch-v9-20261020.md), [C16 원자료](../benchmarks/evidence/20261020-latency-v9/runs/final-c16/result.json), [C32 원자료](../benchmarks/evidence/20261020-latency-v9/runs/final-c32/result.json).

Qwen3-4B BF16, A100-SXM4-40GB, C16/GPU6·C32/GPU7, 각 엔진 fresh 3회에서 얻은 지표별 중앙값이다. 다른 모델·GPU에 일반화하지 않는다.

| 지표 | C16 Riley v9 | C16 stock vLLM | C32 Riley v9 | C32 stock vLLM |
|---|---:|---:|---:|---:|
| output tok/s | 574.188 | 1126.664 | 735.891 | 1671.550 |
| Riley/stock 처리량 | 50.96% | — | 44.02% | — |
| TTFT p95 ms | 105.854 | 83.666 | 339.077 | 164.610 |
| TTFT p99 ms | 220.887 | 110.866 | 461.062 | 217.645 |
| TPOT p95 ms | 26.418 | 12.893 | 41.840 | 17.778 |

별도 C32 진단 trace의 Riley 커널 시간 합은 1762.015ms이며 Gate/Up 391.585ms, Down 218.446ms, K/V 228.090ms, QK/PV 206.595ms, RMSNorm 162.902ms다. 앞의 두 MLP 투영은 합계 약 34.6%다. KV gather는 8.602ms, 약 0.5%이므로 독립 최적화 우선순위를 낮춘다. 이 비율은 GPU kernel-duration 합의 구성이고 HTTP 지연 기여도가 아니다.

| 관찰 | 계획에 미치는 영향 |
|---|---|
| Decode Gate/Up은 C16 319.948→319.698ms, C32 223.837→223.818ms | 가장 큰 미해결 투영을 우선한다 |
| Decode Down은 C32 143.282→123.498ms | v9를 보존하고 같은 아이디어의 반복보다 남은 병목으로 이동한다 |
| C16 prefill Gate/Up·Down에서 뚜렷한 개선 없음 | 초기 TTFT와 decode를 별도 과제로 관리한다 |
| C16 p95 인접 요청은 측정 시작 후 0.43~1.14ms에 유입 | 초기 cohort의 admission·prefill 완료 경로를 추적한다 |
| C16 sync 1208→306회지만 CPU API 대기 합 1122.685→1105.675ms | 호출 수를 성공 지표로 삼지 않는다 |
| C32 Riley 103761 kernels/1762.015ms, stock 32433/846.620ms | 실행 분할·GPU 계산·메모리 왕복을 함께 다룬다 |
| 기존 부분 Graph·500µs cohort 대기는 성능/꼬리 지연 이점 부족 | 동일 후보를 설정만 바꿔 재시도하지 않는다 |

아직 미확인인 항목은 정확한 GPU idle 원인, projection별 DRAM/L2 이용률·Tensor Core 효율, 실제 행 분포에 따른 padding 비용, 익명 stock GEMM별 연산 대응이다. 커널 수만으로 launch가 전체 격차의 주원인이라 하거나 대역폭 포화라고 단정하지 않는다.

## 3. 외부 구현에서 가져올 설계

2026-10-20 공식 문서·프로젝트 소스를 확인했다. 아래는 설계 참고이며 Riley에서의 속도·비트 일치 검증 결과가 아니다. `latest`·`main` 링크는 변할 수 있으므로 구현 착수 시 참고 commit SHA와 문서 snapshot/hash를 실험 manifest에 고정한다. 최신 upstream과 현재 비교 대상 stock vLLM 0.27.1의 기능을 동일시하지 않는다.

| 라이브러리 | 확인한 구현 원리 | Riley 적용 계획 | 조건·제약 |
|---|---|---|---|
| vLLM | Qwen3의 QKV 합친 projection, 재사용하는 Qwen2 MLP의 Gate/Up merged linear와 SiluAndMul | 요청 축·출력 타일 병렬화, 입력 재사용, Gate/Up epilogue 결합을 분리 평가 | 합친 GEMM이 native M1 누적 순서를 보존하는지 별도 검증 [S1][S2] |
| vLLM | 배치 특성·backend 적격성에 따른 full/piecewise/eager Graph 선택 | strict backend별 적격성 표와 고정 주소·형상 cache, 준비 단계에서 capture | 기존 느린 segment Graph 재활성화로 대체하지 않음 [S3] |
| vLLM | decode 우선 chunked prefill과 token budget 조절 | 초기 prefill cohort와 mixed 부하의 정책을 분리하고 공정성·TTFT/TPOT를 함께 평가 | budget 증가는 별도 실험이며 동일 자원 비교 결과와 구분 [S4] |
| SGLang | GPU forward와 CPU batch 준비·이전 결과 처리의 중첩 | GPU idle이 CPU 준비에 귀속될 때 제한된 다음 iteration 준비를 도입 | EOS·취소·KV commit/abort·출력 순서 의존성을 보존 [S5] |
| TensorRT-LLM | padding 없는 packed input, context/generation in-flight batching, paged KV 기반 chunked context | 실제 토큰 수 중심 실행과 prefill/decode별 계획을 참고 | chunking은 attention geometry·수치를 바꿀 수 있어 strict gate 필요 [S6] |
| FlashInfer | attention의 plan/run 분리, workspace·metadata 재사용, Graph용 고정 버퍼 | 준비 비용을 hot path에서 분리하고 paged 접근·요청 병렬화를 단계적으로 검증 | online softmax나 fixed split 결정성이 기존 BF16 일치를 뜻하지 않음 [S7] |
| CUTLASS | 계층적 tiling, software pipeline, 사용자 정의 epilogue | SM80에서 가중치 재사용·copy/compute 중첩·반올림을 명시한 epilogue 설계 참고 | Hopper/Blackwell 전용 TMA·WGMMA 예제를 A100에 직접 적용하지 않음 [S8] |

전면 runtime 교체보다 기존 strict backend에 이 원리를 적용한다. FP8/INT4, KV 양자화, 수치 허용오차 확대는 이번 경로에 포함하지 않는다. Prefix cache는 현 비교에서 off이므로 재사용 workload의 별도 과제로 두며 이번 격차 축소 성과에 합치지 않는다.

## 4. 변경하지 않을 계약

1. **정확도:** original native oracle과 고정 환경의 HF teacher를 유지한다. 기존 full-vocabulary BF16 바이트, token IDs·text·usage·finish reason·원래 EOS 검사를 유지한다. v9를 새 정답으로 삼지 않는다. 결정성·배치 불변성·토큰 일치·BF16 바이트 일치를 구분한다.
2. **수치 경계:** 각 연산의 K 누적 체인, split-K merge 순서, BF16 중간 반올림, scale/mask/softmax 순서를 기록한다. 메모리에 저장하는 대신 레지스터에서 BF16 cast를 수행할 수 있는지는 별도 증명·검증한다. 대수적으로 같은 식만으로 동치라 판단하지 않는다.
3. **GPU:** 6·7만 사용한다. 기존 holder CUDA context는 종료·재시작·해제하지 않고 기존 controller로 idle 20GiB/실행 64MiB를 전환한다. 재사용할 때 holder 프로세스를 다시 띄우지 않는다. 홀드는 배타 잠금이 아니므로 foreign PID 유입을 계속 감시한다.
4. **자원:** 기존 own+holder 22528MiB 한도, 수명·context·span·allocation/use/replay ledger와 회수 검증을 유지한다. 외부 작업은 중단하지 않는다. GPU0~5는 사용하지 않는다.
5. **저장:** 모델·빌드·스크립트·벤치·scratch는 EFS에 둔다. 이미 승인된 C02 테스트 4개의 임시 fixture만 private `/tmp` 예외이며 범위를 확대하지 않는다.
6. **비교:** 같은 GPU 안에서 baseline/candidate/stock의 입력·환경·자원을 맞춘다. stock 기본 compile/Graph를 유지한 실사용 비교와 기능을 통제한 원인 분석을 별도 표로 보고한다.

Holder의 마지막 검증 상태는 v9 보고서의 상태다. 이 문서 작성 중 새 live 상태를 검사한 것은 아니며, 다음 실행 전 PID/start ticks·UUID·context·foreign PID·메모리 상태를 다시 읽어야 한다. holder가 사라졌다면 기존 context가 유지됐다고 보고하거나 자동 재생성하지 않는다.

## 5. 작업 패키지와 실행 순서

### P0 — 기준선 동결과 시간 분해

**목적:** 후보 탐색을 실제 시간 절감 가능성이 큰 구간에 집중한다.

- 현재 dirty tree와 기존 후보를 보존하고 v9 소스 manifest·바이너리·선택 옵션을 기준선으로 동결한다. 새 layer는 별도 patch와 manifest로 저장한다.
- v9 후보 바이너리 SHA256: `d9a1e6042861e432e08f4ac4b533e0f2dcc0a45a95f13b8713f2969941734424`. 성능 baseline은 v9, 정확도 oracle은 원본 native/HF다.
- 요청 수신→queue/admission→batch 선택→H2D→prefill 완료→첫 sample→SSE 전달의 시간을 같은 request/iteration ID로 연결한다. GPU event와 CPU clock은 별도로 보정하고 중첩 시간을 중복 합산하지 않는다.
- 레이어·phase·projection·실제 M/N/K·bucket/padded M·선택 kernel·fallback 사유·H2D/D2H·Graph hit/miss를 기록한다. C16/C32의 행 수별 호출 빈도와 GPU 시간 가중 분포를 만든다.
- CPU scheduling/metadata 준비, launch 사이 bubble, GPU compute를 Nsight Systems로 나누고 대표 hot kernel만 Nsight Compute로 측정한다. profiler 없는 정상 측정을 별도로 유지한다.
- roofline은 실제 byte/연산/하드웨어 counter가 있을 때 작성한다. Kernel 합·span·CPU wait를 HTTP 시간에 단순 가산하지 않는다.

**완료 산출물:** 단계별 시간표, 초기 cohort 요청 추적, 행 분포, hot kernel counter, 실제 경로 선택 감사. 차기 후보마다 예상 총효과를 `측정된 영향 구간 비중 × 해당 구간 절감률`로 대략 추정하고 중첩·배치 변화 한계를 명시한다.

### P1 — C16 초기 prefill TTFT와 prefill 투영

**수정 후보:** `crates/riley-runtime/src/llama/forward.rs`, `executor/dispatch.rs`, `executor/owner.rs`, `kernels/src/cublas_gemm_probe.cu`. 정책 변경이 필요한 경우에만 `crates/riley-scheduler/src/scheduler.rs`, `plan.rs`, `config.rs`를 다룬다.

1. 초기 cohort의 첫/마지막 요청 prefill completion과 p95 인접 요청을 추적해 queue 대기와 GPU prefill 중 어느 부분이 큰지 확정한다.
2. 현재 125~130 token prompt와 실제 prefill row-group을 기준으로 Gate/Up·Down의 recipe/fallback/padding을 조사한다. Prompt 길이가 곧 GEMM M이라고 가정하지 않는다.
3. native-equivalent K 누적을 고정한 채 request/token row tile과 N 타일의 병렬화·가중치 재사용을 확대한다. Full-prefill과 chunked-prefill은 다른 geometry로 관리한다.
4. chunk 크기·token budget 변경은 계산 경로 개선과 분리한다. 먼저 budget576을 고정하고, 이후 메모리 한도 내 정책 sweep을 동일 조건의 stock과 비교한다. 짧은 prompt에 무조건 chunk를 늘리지 않는다.
5. 별도 cohort 대기를 추가하지 않는다. 초기 요청을 모으는 정책을 재검토하려면 P0에서 원인과 예상 이득이 먼저 확인되어야 한다.

**판정:** 초기 cohort를 포함한 전체 TTFT p95/p99, 첫/마지막 cohort completion, steady-state 지표를 모두 보존한다. Prefill 컴포넌트와 full-vocab 검사를 통과한 뒤 정상 C16 실험에서 반복 악화가 없는지 확인한다. TTFT만 낮추고 TPOT·처리량·공정성을 악화시키면 채택하지 않는다.

### P2 — Decode Gate/Up의 실질적 배치화

**수정 후보:** `forward.rs`의 row-group dispatch, `executor/config.rs`·`owner.rs`, `crates/riley-cuda/src/gemm.rs`·`owned_mlp.rs`, `kernels/src/cublas_gemm_probe.cu`.

- R1..16 native 잔류 구간을 우선 조사하고 R17..32 기존 v8 kernel도 같은 분포로 비교한다. 지원 경계 R4/5,16/17,31/32와 실제 빈도가 높은 행 수에 집중한다.
- 후보 A: native K-chain을 그대로 유지하면서 독립 request row·Gate/Up output tile을 같은 CTA/launch에서 처리해 입력과 가중치 재사용을 늘린다.
- 후보 B: SM80 async copy·double buffering·tile 배치로 copy/compute 중첩을 조정한다. 레지스터 압박·shared memory·occupancy·spill을 함께 측정한다.
- 후보 C: Gate/Up 투영의 각각의 BF16 출력 반올림과 SiLU BF16 경계를 명시적으로 수행한 뒤 곱셈까지 epilogue에서 처리한다. 전역 메모리 intermediate 제거와 수치 경계 제거를 혼동하지 않는다.
- 기존 21개 K32 slice 변형·8-warp 후보와 무차별 Lt 검색을 그대로 반복하지 않는다. 실패 사유와 counter를 연결해 다른 가설이 있을 때만 탐색한다.
- checkpoint·환경·shape·alignment·수명 적격성에 따라 명시적으로 dispatch하고 미검증 geometry는 v9로 fallback한다. 실제 선택률을 성과와 함께 보고한다.

**판정:** native M1과 실제 가중치·다층 활성값·seed·경계값의 컴포넌트 byte 검사, 배치 순열·활성 행 변화·padding 비활성 행 오염 검사, whole-model 검사 후 C16/C32 정상 처리량을 측정한다. Microbench 승리가 전체 이득으로 이어지지 않으면 기본 선택에서 제외한다.

### P3 — Down·Norm·Residual 융합과 중간 저장 축소

- v9 decode Down의 개선은 유지한다. 새 Down 후보는 남은 reducer launch와 workspace 트래픽이 실제 비용인지부터 확인한다.
- Down reduction→BF16 cast→residual add, residual→RMSNorm의 결합을 각각 독립 후보로 둔다. 기존 residual 연산의 반올림과 norm의 reduction·epsilon 순서를 유지한다.
- Gate/Up activation 융합과 Down 융합을 한 번에 바꾸지 않고 단독·결합 ablation을 남긴다.
- Prefill Down은 decode와 별도 tile/dispatch로 설계한다. 모든 행 수에 하나의 커널을 강제하지 않는다.
- Workspace 증가·weight 재배치 비용·cold preparation 시간·동시 context memory peak를 함께 보고한다. 이미 fused인 SiLU를 새 성과로 중복 집계하지 않는다.

**완료:** 전체 byte gate와 정상 성능 gate를 통과한 융합만 선택한다. Kernel 수 감소와 함께 GPU 시간·intermediate bytes·HTTP 이득이 확인되어야 한다.

### P4 — K/V·QK/PV 요청 병렬화와 paged 접근

**수정 후보:** `executor/dispatch.rs`, `crates/riley-cuda/src/decode.rs`·`attention.rs`, `kernels/src/attention_cublaslt.cu`·`cublas_gemm_probe.cu` 및 기존 softmax helper.

1. K/V projection의 수치 보존 row-group 범위를 확장한다. QKV 병합은 Q·K·V 각각의 누적 순서를 보존하는지 따로 평가한다.
2. QK/PV를 요청·head 축으로 묶되 각 요청의 K reduction, scale/mask, softmax, PV reduction 순서를 유지하는 batched 실행부터 시도한다.
3. Paged KV를 직접 읽는 후보는 별도로 둔다. Gather 제거 자체보다 후속 attention의 locality·전체 시간 감소가 있는지 검증한다.
4. FlashInfer의 plan/run 분리와 metadata/workspace 재사용을 참고한다. 외부 kernel 직접 호출은 지원 SM80·shape·ABI·workspace·동기화·byte gate를 확인한 뒤의 선택지다.
5. Online softmax/FlashAttention 대체는 수치 위험이 높으므로 작은 동치성 실험을 먼저 한다. 최초 차이가 나면 허용오차로 통과시키지 않고 현 strict 기본 경로에서 제외한다.

**검증:** 지원 head dimension/GQA, 길이1·page 직전/직후·partial page·최대 지원 길이, 비연속 block table, mixed context, prefill→decode, 취소 후 block 재사용·자원 부족 rollback. 지원하지 않는 geometry는 기존 거부/fallback 계약을 유지한다.

### P5 — 전체 decode 실행 범위와 Graph dispatcher

P0의 idle 분석을 바탕으로 P2~P4에서 안정화한 실행 경로를 묶는다. 현재 segment Graph가 느렸던 이유와 달라진 capture 범위·metadata 전략을 명시해야 착수한다.

- Decode uniform batch와 prefill/mixed batch를 구분하고 backend/shape/dtype/KV layout/환경을 cache key·적격성 조건에 반영한다.
- 고정 device/pinned 주소와 preallocated workspace를 사용한다. Attention plan·allocation·host-dependent 분기는 capture 밖에서 준비한다.
- 실제 자주 쓰는 shape를 먼저 사전 capture하고 측정 중 capture 수·miss·padding waste·메모리 peak를 공개한다. 전체 shape 무차별 capture를 기본값으로 만들지 않는다.
- 첫 단계는 안정된 decode 경로의 더 넓은 capture다. Attention을 포함하지 못하면 남는 launch·동기화 경계를 수치로 보고한다. Piecewise와 whole-step을 같은 workload에서 비교한다.
- Graph는 GPU kernel 계산량을 줄이는 기법과 별개다. GPU 시간이 지배적이면 Graph만으로 격차가 해소될 것으로 기대하지 않는다.

**판정:** eager/capture/first replay/repeated replay의 full-vocab·자원 수명을 확인한다. 정상 HTTP의 초기 cohort·cold 비용을 숨기지 않고, capture 완료 후 steady-state 결과를 별도로 보인다.

### P6 — SGLang식 CPU/GPU 중첩과 일반화

**착수 조건:** 남은 GPU idle이 CPU scheduling·metadata 준비·결과 처리 때문에 발생한다는 P0 증거가 있을 때다.

- 현재 scheduler의 단일 inflight 계약을 먼저 명세한다. 준비 단계만 다음 iteration으로 앞당기는 최소 변경부터 검토하고, token 결과가 필요한 결정을 임의로 추측하지 않는다.
- Event 기반으로 완료를 알리고 metadata를 double buffer한다. KV reservation과 commit/abort를 iteration ID에 결부하고, 취소·EOS·실패 시 후속 준비를 폐기할 수 있게 한다.
- 출력 순서·backpressure·stream 종료·요청별 natural EOS를 보존한다. Overlap이 불필요한 다음 decode를 실행하거나 메모리 회수를 지연시키는지 검사한다.
- Qwen3-4B C16/C32 통과 후 C1/C8, heterogeneous prompt/output, 지원하는 다른 모델·head geometry로 확대한다. A100 통과를 RTX4090 통과로 보고하지 않는다.

## 6. 단계별 우선순위와 중단 기준

작업 기간은 담당자 1명 기준 초기 추정이며 GPU 실험·정확도 실패에 따라 변한다. 병렬 agent 실행을 전제하지 않는다.

| 단계 | 순서·의존성 | 초기 작업량 | 검토 시점 |
|---|---|---|---|
| P0 | 먼저 수행 | 1~2 작업일 | 실제 병목·행 분포로 후보 우선순위 확정 |
| P1 | P0 후, 최우선 tail 과제 | 2~4 작업일 | C16 초기 p95/p99와 prefill GPU 시간 |
| P2 | P0 후, 최우선 처리량 과제 | 3~5 작업일 | Gate/Up GPU 시간·실제 선택률·C16/C32 처리량 |
| P3 | P1/P2의 수치 계약 확정 후 | 2~4 작업일 | 중간 bytes·GPU 시간·결합 ablation |
| P4 | P0와 안정된 투영 baseline 후 | 3~5 작업일 | K/V·QK/PV 및 전체 step 시간 |
| P5 | 안정된 실행 경로와 idle 분석 후 | 2~4 작업일 | Graph hit/miss·cold/steady 성능 |
| P6 | CPU 기인 idle 확인 후 | 3~5 작업일 | 실제 overlap·EOS/취소·자원 회수 |

한 후보의 탐색 예산은 처음에 고정한다. 초기 상한은 구조가 다른 3개 가설, 가설당 대표 hot shape 12개 설정이다. 정확도 실패를 기록한 뒤 임의의 허용오차를 넣지 않는다. 정확도 통과 후보가 GPU 시간·정상 처리량을 개선하지 못하면 중단하고 다음 가설로 이동한다. 같은 구조의 상수 조정만으로 예산을 계속 늘리지 않는다.

P1/P2 종료 시 중간 평가를 한다. 둘 다 정상 성능 이득이 없으면 P3~P6를 자동으로 계속하기 전에 수치 계약별 비용과 실패 원인을 재분석한다. 큰 이득이 없다는 결과도 그대로 보고하고, 정확도 완화는 별도 사용자 결정 없이는 제안 경로에 섞지 않는다.

## 7. 성능 목표와 채택 기준

아래 수치는 **실험 목표**이며 달성 예측이나 보장이 아니다. 현재 gap에서 목표까지 필요한 개선량을 보이기 위해 둔다. Stock도 매번 새로 측정한다.

| 목표 | 초기 목표 | v9 기준 의미 |
|---|---|---|
| 첫 유효 묶음 | C16/C32 처리량 각각 +10% 이상 | 약632/810 tok/s, 기존 소폭 개선에서 벗어나는 확인점 |
| C16 초기 TTFT | p95 중앙값 -10% 이상 | 약95.27ms 이하; 초기 cohort 포함 |
| 종합 중간 목표 | stock 대비 C16 60%, C32 55% 이상 | 고정된 과거 stock 값으로 환산 시 약676/919 tok/s |
| 장기 방향 | 같은 정확도·자원에서 지속적인 격차 축소 | 목표 달성 여부는 fresh 측정으로만 판정 |

채택은 정확도와 자원 검사를 선행하고 처리량·tail을 함께 평가한다. 빠른 화면 검사는 paired 3회로 후보를 줄인다. 최종 후보는 C16/C32 각각 독립 서버 시작을 포함한 fresh 5회 이상, 순서를 균형 있게 교대해 비교한다. 최소 64 warm+192 measured/arm을 유지하고 p95/p99 표본·반복 변동이 크면 요청 수를 늘린다.

- 정상 처리량 개선의 paired 변화가 측정 변동보다 큰지 확인한다. 5회만으로 작은 차이의 통계적 유의성을 보장하지 않는다. 요청 간 의존성을 무시한 단순 bootstrap 대신 반복 실행을 단위로 분석한다.
- 채택 후보의 C16/C32 처리량 중앙값이 개선되고 TTFT p95/p99·TPOT p95에 일관된 악화가 없어야 한다. p95/p99 2% 초과 악화가 한 반복이라도 있으면 자동 채택하지 않고 원인과 추가 paired 실행으로 확인한다. 2% 이내도 반복적인 악화는 미해결로 남긴다.
- 첫 cohort, steady state, 전체 측정을 모두 공개한다. 보고 후 유리한 구간만 primary metric으로 바꾸지 않는다.
- p99의 작은 변화는 샘플 수가 부족할 수 있으므로 192요청 한 회의 p99만으로 통과·회귀를 단정하지 않는다. Warm된 프로세스의 초기 cohort와 완전 cold start도 구분한다.

## 8. 벤치마크·정확도 매트릭스

| 축 | 필수 범위 |
|---|---|
| Primary | Qwen3-4B BF16, C16/GPU6·C32/GPU7, 각 lane에서 v9/후보/stock 교대 |
| 기존 비교 조건 | budget576, max sequence1024, KV block16/blocks=C×64, prefix off, 동일12 prompts125~130 tokens, cap8/16/24/32, greedy·원래 EOS |
| 보조 | C1/C8, 실제 행1..32 및 tile 경계, batch 순열·요청 입출입 |
| 일반화 | 지원 범위 내 짧은/긴 prompt, heterogeneous output, 긴 decode, burst·지속 부하, prefill/decode 혼합 |
| 정확도 | 컴포넌트 BF16 → 다층 활성값 → 전체 vocabulary teacher → greedy 전체 생성·SSE 재구성 |
| 생명주기 | 취소·EOS·timeout·allocation 실패·KV 고갈·rollback·Graph 파기·재실행·회수 |
| 조합 | 각 후보 단독 → pairwise 주요 상호작용 → 최종 결합; eager와 선택 Graph 구분 |
| Profile | 정상 성능 측정과 별도 Nsight Systems/대표 Nsight Compute; 계측비용 확인 |

Batch/recipe를 바꾸면 final logits뿐 아니라 최초로 달라지는 레이어·projection·원소를 찾을 수 있도록 진단한다. 지원 geometry 전체에 대한 수학적 증명과 유한 corpus의 byte 검증을 구분한다. 새로운 외부 backend는 원래 oracle을 통과할 때만 strict 후보로 남긴다.

SGLang·TensorRT-LLM을 새 비교 엔진으로 실행하는 것은 선택적 진단이다. 같은 GPU·모델·입력·자원 예산을 맞출 수 있을 때만 별도 lane 시간에 수행하고, 버전·선택 kernel·지원하지 않는 옵션을 공개한다. 구현 아이디어 참고만으로 해당 엔진을 실제 benchmark했다고 쓰지 않는다.

## 9. 재현 자료와 완료 정의

각 단계는 별도 candidate patch·source manifest·원본 baseline 참조·build command·바이너리 hash·실제 argv/config·driver/CUDA/cuBLAS·checkpoint/tokenizer hash를 남긴다. 외부 source를 차용하면 commit·라이선스·변경 범위를 함께 남긴다.

결과에는 raw 응답/SSE·요청 시간·실제 output token 수·EOS·오류·profile·kernel selection·memory peak·holder 전후 PID/start ticks/UUID와 검증 manifest를 포함한다. 성능 baseline·정확도 oracle·stock 설치 파일은 덮어쓰지 않는다.

완료 조건은 **기존 정확도 통과 + fresh 정상 비교의 처리량/tail 이득 + 자원 회수 + 원래 GPU context 홀드 유지**다. 검토 가능한 patch와 보고서로 묶고 기존 dirty/untracked 변경을 보존한다. 계획 단계에서 성과를 미리 기입하거나 commit/push하지 않는다.

## 10. 공식 참고 자료

- **S1** [vLLM Qwen3 구현](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py): QKVParallelLinear와 Qwen2 MLP 재사용.
- **S2** [vLLM Qwen2 MLP 구현](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py): MergedColumnParallelLinear·SiluAndMul·Down 실행.
- **S3** [vLLM CUDA Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/): backend·batch에 따른 Graph 실행 모드와 dispatcher.
- **S4** [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill): chunked prefill·token budget의 지연/처리량 관계.
- **S5** [SGLang 팀의 overlap scheduler 설명](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/): CPU 준비·결과 처리와 GPU forward 중첩의 설계 원리. 2024년 설계 설명이며 성능 수치를 Riley에 전용하지 않는다.
- **S6** [TensorRT-LLM Attention 문서](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html): packed in-flight batching, chunked context, paged KV. 문서의 backend별 적용 조건을 구현 시 다시 확인한다.
- **S7** [FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html): plan/run, workspace 재사용, CUDA Graph용 metadata·버퍼 제약.
- **S8** [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html), [GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html): tiling·pipeline·mainloop/epilogue 설계.


## 2026-10-20 v12 실측 반영: packed prefill projection

Q/K/V/O의 요청별 prefill을 native-M128 arithmetic family를 보존하는 총 토큰 배치로 묶었다. v11 E를 기준으로5회 새 프로세스 비교를 수행했다. TTFT p95는 C16 -5.84%, C32 -5.66%였으나 처리량은 각각 -0.12%, +1.95%였다. C16의 한 반복은 TPOT p95/p99가 +2.556%/+4.501% 악화돼 모든 pair에서2% 이내 tail 비악화라는 점검 기준을 충족하지 못했다. 큰 처리량 개선으로 승격하지 않는다.

기존 정확도 기준을 유지했다. CPU523개, original-native/HF full-vocabulary BF16 파일2464개, 실제 packed Lt 선택, 정상30 arms의 raw SSE6042개 및 GPU 자원 회수·원래 GPU6/7 holder context 유지를 검증했다. 세부 recipe·실측·제외한 구간·실패 시도와 원자료는 [v12 결과](/Users/psyche/PycharmProjects/riley/docs/qwen-packed-projection-v12-and-vllm-20261020.md)에 기록했다.

다음 큰 처리량 작업은 mixed/prefill piecewise Graph와 native-family decode Gate/Up GEMM·epilogue 재설계다. 최신 vLLM MRV2의 pinned staging/GPU metadata 및 SGLang·TensorRT-LLM의 overlap 설계는 executor 소유권·EOS·취소·embedding-before-KV 실패 계약과 함께 적용해야 한다. 단순 launch 설정이나 동기화 삭제로 완료 처리하지 않는다. 이번 v12의 prefill 부분 이득을 해당 구조 변경의 완료로 간주하지 않는다.

## 2026-10-21 v14 실측 반영과 다음 실험의 조건

이번 실험은 큰 처리량 격차를 줄이지 못했다. 구조나 launch 수가 바뀌었다는 이유만으로 단계 완료·성능 개선으로 판정하지 않는다. 코드 기본값은 v13B snapshot이며, v14는 재현 가능한 미적용 실험 패치다.

- 동일 buffer/동일 길이 hot 구성요소 검사만으로 kernel을 채택하지 않는다. 이종 context, pipeline cache 상태 및 실제 batch 노출 비중을 먼저 확인한다. v14의 R24~32 softmax/PV 이득이 정상 처리량에 유지되지 않았다.
- Graph 내부 kernel attribution부터 보완한다. C16/C32 D trace에 각각703.478/468.320ms가 미분류로 남았으며 이를 Gate/Up·Down으로 임의 배분하지 않는다. 실제 pipeline의 DRAM/L2·SM counter 없이는 memory-bound라는 결론을 내리지 않는다.
- 다음 큰 연산 후보는 native BF16 중간 경계와 원래 reduction family를 유지하는 **prefill attention fusion**이다. native/HF full-vocabulary gate 및 C16 초기 cohort TTFT/TPOT/E2E를 함께 검증한다. decode의 작은 조각만 빨라지는 것을 전체 backend 개선으로 계산하지 않는다.
- mixed 전체 실행/host overlap은 **단일 iteration의 완료·publish 경계와 stable metadata 수명**부터 설계한다. v14의 두-pass 방식은 C16 약5~6% 느렸으므로 중간 동기화를 추가하는 형태로 재도입하지 않는다. GPU-resident metadata나 복수 in-flight 실행은 EOS·취소·KV reservation 회수·pinned buffer 재사용 계약을 먼저 증명해야 한다.
- R2~16 Gate/Up은 full pipeline에서 recipe와 fused epilogue를 비교한다. layer0의 반복 측정에서0.8% 빠른 Lt가 서버 전체를 빠르게 만들었다고 가정하지 않는다. dtype/tolerance/출력 조건을 완화하지 않는다.

사용자 지시에 따라 **큰 실험 단위가 끝나면 코드·MD·재현 가능한 증거를 커밋하고 푸시한다.** 실패 실험도 미채택 사유와 전체 반복 표본을 보존한다. 마지막에는 committed source와 qualified source manifest의 일치 및 원격 tip을 확인한다. 다음 GPU 작업에도 원래 GPU6/7 holder를 종료하거나 context를 재시작하지 않는다.

### 2026-10-21 v15A prefill 구조 실험 checkpoint

[v15A 보고서](qwen-prefill-host-v15a-20261021.md)의 prefill-only 후보는 K/V GQA 확장과 Q/output 재배치를 제거하고 native BF16 경계의 causal softmax를 융합했다. CPU524, full-vocabulary BF16 2,464 files, attention 구성요소156조합을 통과했다. 한 번의 선별에서 C16+4.78%, C32+5.21%였지만 C16 TPOT p99+2.38%이고 C32에는 host trace가 있으므로 정상5회 결과로 채택하지 않는다. 재현 패치와 압축 증거를 커밋·푸시로 보존한다.

기존 baseline의 Graph node 재분류로 other는 C16 407.503→96.909ms, C32 375.545→89.720ms가 됐다. Gate/Up+Down이 전체 GPU kernel 시간 합계의 약41~43%다. prefill attention 변경과 host resource index·completion 경계 개선을 결합한 후보를 독립 빌드하고 C16/C32의 before/after/stock 5회 정상 비교를 진행한다. 정확도·tail·GPU6/7 holder 기준은 유지하며, 최종 정상 수치는 루트 README에 별도로 반영한다.

### 2026-10-21 v15B prefill+host checkpoint

[v15B 보고서](qwen-prefill-host-v15b-20261021.md): prefill 구조와 O(1) resource index·embedding report completion 경계를 합쳤다. 정상 fresh5회 처리량은 C16 +3.06%, C32 +5.45%; TTFT p95는 -6.81%, -9.82%였다. C32 반복2 TPOT p99+2.40% 플래그와 일반 BF16 view의 정렬 적격성 보완이 남아 기본 코드는 v13B로 유지한다. 미적용 patch·30 정상 arms·5 별도 profiles·원래 HF/native bytes와 whole-device cap의5개 재검사를 압축 증거로 보존한다.

실제 after trace의 prefill compact gather는 각 동시성에서6,912회, pointer-batched GEMM은13,824 NVTX ranges였다. 기존 Q gather/output scatter/scale-mask/BF16 scatter는0이다. generic context 복원 변경은 관측된 API 비용상 우선순위가 낮아 도입하지 않았다. 최종 C 후보는 비정렬 view를 기존 경로로 보내는 가드와120개의 실제 GPU view 검사를 더하고 같은 정확도·C16/C32 fresh5회 성능 gate를 다시 수행한다. +10% 처리량 목표 달성으로 승격하지 않는다.
