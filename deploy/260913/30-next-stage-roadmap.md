# 다음 단계 계획 — 2026-09-15

상태: N01 로컬 계약·lifecycle 기반과 N02 첫 synthetic control 구현을 시작했다. 원격 GPU 실행, native-BF16 품질 기준, materialized 1.7B/3B 자산, full-model·serving benchmark는 아직 없다. 이전 종료 기록과 성능 미달 판정은 유지한다. 이 문서의 순서는 기존 PR01~29를 대체 구현한 목록이 아니라, 최신 결과를 반영한 다음 착수 우선순위다.

후속 실행 계획: [PR 단위 새 계획](next-stage/README.md). 실행 순서와 Blender 운영 정책은 이 후속 문서를 우선한다.

## 방향

다음 목표는 특정 135M 모델의 tile·threshold를 더 조정하는 것보다, 실제 serving에 적용할 실행 기반과 평가 범위를 넓히는 것이다. **측정 운영 정리 → 모델/수치 계약 → 범용 BF16 실행 경로 → workload별 serving 정책 → 하드웨어 확장** 순서로 진행한다. 각 구현 PR은 2~5개의 관련 변경을 묶고, 의미 있는 실행 경로가 완성됐을 때 비교한다.

이 방향이 더 빠를 것이라는 결론은 아직 없다. 표준 native backend가 기존 전용 kernel보다 빠른지, 새로운 모델에서도 이득이 유지되는지를 확인할 수 있도록 만드는 계획이다. 모델 지원 확장 자체를 성능 개선으로 계산하지 않는다.

## 연구 우선 gate — 후속 요청 반영

[새 방법 도입 가능성 조사31](31-method-feasibility-research.md)에 따라 **소스·수치·용량 판정(R0) → 제한된 native operator/layer 실험(R1) → 선택한 후보만 full-model/serving(R2)** 순서를 N03/N04의 선행 조건으로 추가한다. N01/N02는 이 판정에 필요한 범위부터 진행하고, 범용 실행층을 크게 구현한 뒤 가능성을 확인하는 순서로 진행하지 않는다. 우선 후보는 긴 context GQA와 작업 분할이며, 공유 prefix state merge와 RMSNorm–GEMM 재배치는 조건부다. 아래 N03/N04는 gate를 통과했을 때의 구현 계획이다.

## 1차 실행 범위 확정 — RTX 4090 / 20GB / BF16

사용자의 후속 조건에 따라 N01~N05의 1차 실험은 **RTX 4090 한 장, 전체 GPU 사용량 20GB 이내, BF16**으로 제한한다. Hopper/Blackwell/multi-GPU는 이후 확장 축이며 1차 결과의 선행 조건이 아니다. 현재는 이 계약에 따라 N01/N02의 제한된 구현과 검증을 진행한다.

### 메모리·정밀도 계약

- “20G”는 보수적으로 **20,000,000,000 bytes(약 18.63GiB)**를 전체 GPU 사용량 상한으로 잡는다. MiB/GiB 표시를 GB와 혼용하지 않고 manifest에는 bytes를 기록한다.
- 가중치와 KV cache는 BF16을 유지한다. GEMM/attention의 누적 정밀도는 별도 profile에 명시한다. 20GB에 맞추려고 FP8/INT4로 바꾸거나 CPU weight/KV offload를 섞지 않는다. BF16이라는 dtype만으로 기존 strict bitwise 일치를 가정하지 않는다.
- `가중치 + resident KV + activation/scratch + graph pool + CUDA/library context + weight packing/로딩 중복 + 외부 GPU 사용량`의 peak를 포함한다. 설계 추정치에 추가 불확실성 여유 1GB 이상을 넣어 20GB 이하인지 먼저 확인한다. 엔진 allocator 카운터만 보고 전체 GPU peak가 통과했다고 판정하지 않는다.
- CPU RAM은 별도 점검한다. 20GB VRAM 예산이 호스트 RAM 부족·swap 문제를 해결하지는 않는다. 엔진은 순차 실행하며 비교 엔진마다 같은 전체 메모리 상한과 workload를 적용한다.
- 로딩·weight packing·graph capture·최대 prefill·steady decode·취소/회수·재사용을 포함한 peak를 검증한다. 전체 가중치의 추가 GPU 사본이 필요한 구현은 그 peak까지 예산에 포함하거나 streaming packing/원본 해제를 설계해야 한다.
- 예산 초과 예상 시 KV pool/active capacity/graph bucket/prefill chunk를 줄이고 그 동일 조건으로 모든 엔진을 비교한다. 모델 precision을 몰래 바꾸거나 일부 엔진만 짧은 context로 돌리지 않는다. admission의 resident-token 예산으로 용량을 제한하고, 가능한 조합만 실행한다.

### 모델 순서

| 역할 | 후보 | BF16 가중치 단순 추정 | 1차 범위 |
|---|---|---:|---|
| 기존 회귀 | SmolLM2-135M | 약 0.27GB | 기존 strict 결과 유지·새 경로 회귀 확인 |
| 연결 검증 | SmolLM2-1.7B-Instruct | 약 3.4GB | 같은 계열에서 shape 확장; 총 sequence 길이 8,192 이하 |
| 주 평가 | Qwen2.5-3B-Instruct | 약 6.18GB | 더 큰 GEMM·GQA·긴 context·긴 생성 비교 |
| 선택적 후속 | 7B급 BF16 | 대략 14~16GB | 3B 완료 후 실측 peak가 20GB 안일 때만, 낮은 active capacity부터 |

가중치 추정은 파라미터 수×2 bytes이며 실제 checkpoint tensor와 tied weight·packing에 따라 달라진다. **로딩 가능 또는 serving 지원이 검증됐다는 의미가 아니다.** N02에서 checkpoint revision·tensor shape/hash·tokenizer·RoPE·QKV bias·head partition을 고정한다. 현재 Qwen tokenizer 소스는 0.5B artifact profile을 명시하므로 3B 호환을 자동 가정하지 않는다. 7B는 1차 필수 목표가 아니며, 가중치만 들어간다는 이유로 장문·고동시성까지 가능하다고 주장하지 않는다.

Qwen 3B를 주 평가 후보로 둔 이유는 가중치 여유뿐 아니라 GQA의 KV 크기다. 아래는 공식 config의 layer/KV-head/head-dim을 사용한 **payload 계산**이다. 할당·page rounding·prefix 보존·workspace는 별도다.

`KV bytes = 2(K,V) × layers × KV_heads × head_dim × 2(BF16 bytes) × resident_tokens`

- SmolLM2-1.7B: 24 layers, 32 KV heads, head_dim 64 → **196,608 bytes/token(192KiB)**. 8,192 tokens×4 active에서 약 6.44GB의 KV payload다.
- Qwen2.5-3B: 36 layers, 2 KV heads, head_dim 128 → **36,864 bytes/token(36KiB)**. 8,704 tokens×8 active에서 약 2.57GB의 KV payload다.

따라서 작은 파라미터 모델이 반드시 긴 context·동시성에서 메모리를 덜 쓴다고 가정하지 않는다. 공유 prefix로 절약될 것을 전제로 admission 상한을 잡지 않고, 비공유 resident token을 기준으로 먼저 산정한다. 사용 후 cache에 남는 prefix pages도 총 예산에 포함한다.

### 3B workload 확장 순서

| 축 | 입력 tokens | 최대 출력 tokens | active capacity 후보 | 목적 |
|---|---:|---:|---|---|
| 짧은 대화 | 512 | 128 | 1 → 8 → 32 | 낮은 부하 latency와 높은 부하 처리량 |
| 일반 대화 | 2,048 | 256 | 1 → 8 → 32 | prefill/decode 혼합 비용 |
| 긴 입력 | 8,192 | 512 | 1 → 4 → 8 | 긴 prefill과 resident KV 비용 |
| 긴 문맥·생성 | 16,384 | 1,024 | 1 → 2 → 4 | context 증가와 긴 decode의 영향 |

모든 조합의 Cartesian product를 한 번에 돌리지 않는다. 기본 correctness·메모리 검사 후 대표 축부터 진행하고, actual tokenizer 길이를 고정한다. 위 값은 3B의 제안 matrix이며 메모리/실행 지원 검사 후 확정한다. SmolLM2-1.7B에는 8,192 total context 한계를 적용하므로 이 장문 matrix를 그대로 복사하지 않는다. C64 이상의 client concurrency는 별도 대기열 부하 검사이며 active capacity와 구분한다. output 예산·EOS 처리 정책도 엔진 간 동일하게 고정한다.

### 1차 단계 완료 기준

1. 기존 135M strict 경로 보존과 별도 BF16 실행 profile의 사전 correctness/품질 계약 충족.
2. 1.7B 연결 확인 및 3B 대표 workload에서 20GB 전체 peak 예산 충족. 미지원 조합은 명시하고 지원 완료로 표시하지 않는다.
3. 같은 4090·모델·길이·정밀도·메모리 조건에서 기존 Riley(지원 시)/후보/vLLM 및 지원되는 비교 엔진의 serving 표.
4. 큰 모델/긴 context 지원 자체와 성능 우위를 구분한다. Hopper/Blackwell 없이도 이 1차 평가를 완료할 수 있도록 설계한다.

구성 확인 출처(2026-09-15, 실행 전 immutable revision으로 재고정): [SmolLM2-1.7B config](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/raw/main/config.json), [Qwen2.5-3B config](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json), [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). 이 자료는 모델 구조·payload 추정의 근거이며 Riley 지원이나 20GB 실측 증거가 아니다.

## 현재 근거와 반복하지 않을 작업

- [마무리 결과](29-serving-closeout.md): C8 공유에서 CTA 실험판은 vLLM 대비 throughput +13.98%, TTFT −14.94%, TPOT −7.71%. 기존 Riley 대비 throughput은 정순/역순 +0.58%/+1.00%에 그친다. C32 공유 TPOT는 vLLM보다 48.26%, 비공유 TTFT는 27.45% 느리다. C32 환경 변동과 C8 비공유 미완료를 전제로 하며 병목 원인을 이 표만으로 확정하지 않는다.
- [PR13 최신 결과](13-speculative-decoding.md): speculative serving까지 이미 연결했고, short-query attention 후에도 frozen prior보다 느렸다. 이를 아직 도입하지 않은 새 기능으로 제안하지 않는다.
- [PR19 최신 결과](19-persistent-serving-integration.md): attention→FFN cooperative 결합은 V7 대비 throughput −18.01%였다. fusion 범위 확대 자체를 다음 해법으로 삼지 않는다.
- [PR03](03-attention-backend-adapter.md): FlashInfer 통합·정밀도·보정 후보는 여러 수치/품질 gate 실패 또는 serving 회귀가 있었다. 같은 실패 경로의 tile 탐색을 반복하지 않는다.
- [PR08 최신 항목](08-hopper-attention-backend.md): FA3는 primitive뿐 아니라 model recorder와 CLI 선택까지 연결됐다. 남은 것은 실제 Hopper 실행·수치·serving 검증이다. 문서 앞부분의 오래된 상태 문구만 보고 다시 구현하지 않는다.
- `crates/riley-runtime/src/llama/variable_session.rs`의 측정 대상 buffers에는 SmolLM2 전용 shape 및 49152×576 head 설정이 있다. `kernels/src/decode_shape.cuh`는 원래 K16 recurrence와 BF16 부분합 반올림을 명시한다. 이는 **현재 측정 경로**의 제약이며 저장소 전체가 다른 모델을 지원하지 않는다는 주장이 아니다.
- Continuous batching, paged KV, prefix cache, CUDA Graph, rolling decode, double buffering은 이미 존재한다. 다음 PR은 기존 구현 재사용과 남은 비용을 명확히 구분한다.

## 우선순위와 단계별 종료점

| 순서 | PR 묶음 | 직접 목적 | 비교/종료점 |
|---|---|---|---|
| 1 | N01 측정 수명과 자산 고정 | 긴 기동 대기·누락 자산·부분 완료 혼동 해소 | 기존 엔진의 제한된 기준 실행 한 번; 성능 개선 주장 없음 |
| 2 | N02 모델 shape 및 수치 profile 계약 | 전용 shape 밖에서 무엇을 검증할지 고정 | 지원 모델 1개 선정, 오차/품질 기준 사전 고정 |
| 3 | N03 native GEMM 실행층 | 표준 BF16 matmul의 적용 비용/가능성 검증 | 대표 shape와 layer 검사; full serving 주장 없음 |
| 4 | N04 native attention + full-model 경로 | 같은 수치 profile로 prefill/decode 실행 통일 | 첫 구조 milestone: 기존 Riley·후보·vLLM·가능한 SGLang/TRT-LLM 비교 |
| 5 | N05 SLO 기반 mixed batching | TTFT/TPOT tradeoff를 요청 부하에 맞게 제어 | 두 번째 serving milestone: 지연 제한 내 처리량·tail·과부하 회복 |
| 별도 확장 | H01 Hopper / H02 Blackwell / D01 tensor parallel | 장비별 연산 및 다중 GPU 지원 | 필요한 장비에서만 성능 판정; 없는 장비의 실행 테스트만 skip |
| 조건부 | Speculation·KV/weight 양자화·P/D 분리 | 연산 횟수·메모리 트래픽·자원 간섭 감소 | 아래 재진입 조건 충족 시 별도 PR 작성 |

**첫 착수 범위는 N01~N02까지 권장한다.** 그 결과로 N03~N04의 수치 계약과 모델 범위를 확정한다. 모든 후보를 동시에 구현하거나 매 PR마다 전체 matrix를 실행하지 않는다. 기반 PR은 관련 correctness/계약 검사, 실제 경로가 완성되는 milestone은 serving 비교로 검증한다.

## N01 — 재현 가능한 측정 수명과 자산 관리

대상: `benchmarks/analysis/projection_cta_serving_screen.py`, 부분/전체 verifier, runtime asset manifest 및 lifecycle 도구. 기존 스크립트를 보강하며 새 측정 시스템을 통째로 만들지 않는다.

묶음:
1. 영구 Python/model/toolchain 경로와 hash/version/argv를 실행 전에 검증하고, 누락 자산을 GPU 중단 전에 발견한다.
2. CPU import/asset 준비, GPU 초기화, warmup, timed serving, 종료 단계를 분리한다. 초기화 시간을 throughput/latency에 섞지 않고 별도 기록한다. GPU 초기화가 필요한 준비는 GPU 소유권 확보 후 실행한다. N01의 `nvidia-smi` 값은 sampled observation으로만 기록하고 warmup/timed-serving의 실행 중 poll이 없으면 20GB 관측 gate를 통과시키지 않는다.
3. 완료·실패·사용자 중단 상태를 명시하고, controller의 stop/cancel 요청 수와 verifier의 concurrency 계약을 일치시킨다. C8 부분 종료 검증을 full-matrix pass로 해석할 수 없게 한다.
4. 모든 엔진에 같은 사전 준비·메모리/KV 예산·실행 순서를 적용한다. 호스트 PSI는 관측으로 남기며 임의 임계값 때문에 작업 전체를 반복 중단하지 않는다.

완료: 작은 단위의 정상/기동 실패/사용자 중단에서 원본 보존과 `leave_stopped_no_restore` 종료 정책이 검증된다. 각 실행에 bounded timeout을 두고 같은 실패의 무제한 재시도를 금지한다. 기존 C8 비공유만 끝없이 보충하지 않고 다음 milestone matrix에 미완료 축을 포함한다.

롤백: 기존 controller와 보존된 asset manifest로 복귀. 기존 evidence 파일은 덮어쓰지 않는다.

## N02 — 모델 shape와 수치 계약의 명시화

대상: `crates/riley-model/src/config.rs`, `ir.rs`, runtime variable-session plan 및 graph identity. 기존 loader/IR 지원 목록을 먼저 확인한다.

묶음:
1. 현재 135M 회귀 모델을 유지하고, 위 4090/20GB 범위의 1.7B 연결 후보와 3B 주 평가 후보에 대해 loader, RoPE/GQA, tokenizer, KV/head layout, BF16 peak 메모리 예산을 확인한다. 후보 지정과 실행 지원 검증을 구별하며, 미지원이면 필요한 구현 범위를 먼저 확정한다.
2. 해당 실행 경로의 hidden/intermediate/vocab/head/context/capacity를 검사된 descriptor로 표현하고, 기존 Smol 전용 실행의 dispatch·allocation·출력을 그대로 유지한다. 임의 모델 전체 지원은 이 PR 범위가 아니다.
3. 기존 `strict` profile과 제안하는 `native-bf16` profile의 입력·누적·반올림·연산 경계를 문서화한다. descriptor와 numerical profile을 graph/cache identity에 포함한다.
4. reference 데이터와 평가 입력을 고정한다. 동일 history의 logits, 자유 생성, 길이·batch invariance, NLL/KL 및 실제 task 품질의 역할을 분리한다. 독립 평가 입력을 사용한다.

중요한 결정: vendor GEMM/attention은 같은 BF16이라도 기존 K320 부분합·BF16 probability 반올림과 bitwise 같지 않을 수 있다. **기존 strict 기준을 낮추지 않는다.** 새 profile은 별도 연구 경로이며, 모델 품질 허용 기준은 구현 성능 결과를 보기 전에 수치로 고정하고 사용자와 합의해야 한다. 합의가 없으면 진단까지만 가능하며 기존 정확 경로의 대체·승격은 불가하다. 현재 계획은 그 기준을 승인한 것으로 간주하지 않는다.

완료: 검증할 모델·dtype·수치 profile·reference·품질 기준이 명확하다. N02 자체를 성능 개선으로 보고하지 않는다. 모델 미지원은 장비 부재 skip이 아니다.

롤백: descriptor를 사용한 새 profile을 비활성화하고 기존 fixed-shape strict 경로 유지.

## N03 — native BF16 GEMM adapter

대상: `crates/riley-cuda`의 기존 prepared GEMM과 native ABI, projection/FFN 실행 plan. 이미 사용하는 head GEMM owner를 재사용할 수 있는지 먼저 확인한다.

묶음:
1. QKV/output projection/gate-up/down의 대표 M/N/K에 native cuBLASLt 또는 CUTLASS adapter를 연결한다. 첫 구현은 한 backend만 선택하고 다른 하나는 제한된 대조군으로 사용한다.
2. packed weight와 workspace 수명을 모델/세션에 귀속하고 graph capture 가능한 사전 준비를 제공한다. timed path에서 JIT·allocation·weight repack을 하지 않는다.
3. activation/residual/norm 경계와 stride/layout 변환을 명시하며, 전송·scratch·추가 kernel 비용까지 측정한다. 무조건 fusion을 늘리지 않는다.
4. capability/shape/profile별 선택 및 명시적 기존 경로 fallback을 연결한다. 기존 strict profile로 새 연산을 몰래 dispatch하지 않는다.

검증: N02 reference 기준의 operator/layer 검사와 supported/unsupported shape, padding, workspace 재사용, bounded sanitizer. 동일 BF16 조건의 성능 대조를 수행한다. kernel 배속을 serving 배속으로 환산하지 않는다.

중단 조건: 품질 계약 미확정, 변환/메모리 비용이 이득을 상쇄, 대표 shape 대부분 회귀. 이런 경우 layer 전체와 server까지 확장하지 않는다.

롤백: 새 GEMM profile 선택 해제; in-flight 작업을 완료시킨 뒤 기존 graph로 복귀.

## N04 — attention과 full-model serving의 일관된 native profile

선행: N02 계약과 N03의 타당성. 대상: 기존 FlashInfer/FA3 native adapter, model recorder, retained workspace, `crates/riley-server/src/engine.rs`.

묶음:
1. 기존 라이브러리 연결을 재사용해 선택된 profile의 prefill·mixed decode·pure decode 연산을 일관되게 연결한다. 이미 실패한 FP16/잔차보정 후보를 그대로 재승격하지 않는다.
2. paged KV의 ownership/COW·workspace·graph identity를 통합하고, 실제 layout conversion 또는 scatter 비용을 계상한다.
3. N03 GEMM과 기존 norm/RoPE/sampling을 조합한 full-model 경로를 하나의 실험 선택으로 노출한다. 변경된 profile 간 prefix/KV 혼용을 금지한다.
4. stop/cancel·batch 변경·새 slot 재사용·OOM/실행 오류 후 수명 및 ordinary fallback을 검증한다.

완료: 선택 모델의 전체 모델 correctness/품질 gate 이후 첫 serving milestone. 비교 대상은 같은 모델·dtype·KV 예산·workload의 기존 Riley(지원 시), 후보, 실행 시점에 고정한 vLLM, 지원되는 SGLang/TRT-LLM이다. 새 모델을 기존 Riley가 지원하지 않으면 기존값을 지어내지 않고 미지원으로 표시한다. 해당 조합의 타 엔진 미지원도 별도로 기록한다.

중단 조건: 모델 품질 실패 또는 큰 workload 축에서 throughput/latency 동시 개선 부재. 이때 새로운 attention tile 탐색을 무기한 반복하지 않는다.

롤백: profile 비활성화, 세션 drain 후 기존 backend. 새 품질 profile을 기존 strict 성능과 동일한 정확도 결과처럼 합치지 않는다.

## N05 — SLO 기반 mixed batching

선행: N04 milestone 또는 현재 strict 경로의 신뢰 가능한 workload별 측정. 작은 135M 결과만으로 compute/메모리 병목을 단정하지 않는다.

묶음:
1. prefill/decode 시간과 KV 여유를 사용하는 비용 모델을 기록된 workload에서 보정한다. 별도 held-out 요청 분포로 검증한다.
2. prefill chunk/token budget, decode 대기 시간, admission/대기열 상한을 하나의 정책으로 연결한다. 기존 continuous batching과 rolling decode를 새로 구현하지 않는다.
3. aging/fairness, 긴 prompt의 starvation 방지, burst 후 회복, 과부하에서 bounded rejection과 cancellation 자원 회수를 구현한다.
4. 고정 정책 fallback과 선택 이유/cost counter를 제공한다. live timed 결과를 보며 유리한 정책만 고르는 방식은 금지한다.

완료: open-loop에서 TTFT/TPOT SLO를 만족한 요청의 goodput과 전체 throughput·P95/P99·실패율을 함께 보고한다. SLO goodput은 원래 throughput 목표를 대체하지 않는 보조 지표다. 정책의 학습/조정 구간과 평가 구간은 분리한다.

중단 조건: 한 latency 개선을 다른 latency 악화로 얻거나 CPU scheduling 비용이 GPU 이득을 상쇄. 기존 고정 정책으로 롤백한다.

## 하드웨어 확장 — 각각 별도 PR

- **H01 Hopper:** 이미 연결된 FA3 recorder/CLI/owner를 재사용한다. 고정 fixture 복구, 실제 SM90a attention/graph 수치 검사, error/수명 검사, matched serving을 묶는다. 새 native profile과 기존 strict의 호환 판정을 구분한다. Hopper가 없으면 실행 검증만 대기하며 FA3를 다시 구현하지 않는다.
- **H02 Blackwell:** pinned native/AOT attention adapter, TMEM/비동기 pipeline 자원 수명, graph/capability 선택, 모델 연결을 묶는다. source·license·compiler 요구를 먼저 확인한다. Python/CuTe DSL이 있다면 offline AOT 경계를 입증해야 하며 Riley runtime에 Python 호출을 넣지 않는다. 지원 compiler도 없으면 compile 미검증까지 표시한다. 이 PR에서 FP4까지 동시에 추가하지 않는다.
- **D01 두 GPU dense TP:** N02 descriptor 위에서 weight/head/KV partition, 표준 NCCL collective 및 stream ordering, rank failure/teardown을 묶는다. CPU partition 검사·가능한 build는 진행하고, 실제 2-GPU 동치/통신/serving은 장비 없을 때만 skip한다. GPU 수와 topology가 같은 vLLM/SGLang/TRT-LLM과 비교하고, 총 처리량 및 GPU당 효율을 모두 보고한다. 단일 GPU에 안 맞는 모델에 잘못된 fallback을 하지 않는다.

장비 부재는 구현 전체를 멈추는 이유가 아니다. 반대로 compile/mock 성공은 실장비 지원·성능 증거가 아니다. 장비가 있는데 발생한 실패는 skip으로 바꾸지 않는다. 새 경로들은 검증 전 기본 비활성으로 두고, 오류 시 해당 backend/TP 선택을 해제한다.

## 조건부 재진입 후보

| 후보 | 다시 검토할 조건 | 그대로 반복하지 않을 것 |
|---|---|---|
| Speculative decoding | 더 큰 target의 낮은/중간 부하에서 `draft+verify+settlement 시간 / 실제 emitted tokens`가 ordinary decode보다 유리하다는 비용 증거 | 135M C32의 낮은 call count나 높은 acceptance만으로 재추진 |
| FP8/FP4 weight·KV 양자화 | 메모리 이동/용량의 실측 병목과 별도 사전 품질 예산; 같은 정밀도의 비교군 확보 | BF16 baseline과 precision을 섞은 성능 승리 주장 |
| P/D disaggregation | 긴 prefill과 decode 간섭이 남고, 추가 GPU 배치 이득이 KV 전송/라우팅 비용보다 큼 | 전송·CPU orchestrator 비용을 뺀 kernel 수치로 채택 |
| 새로운 수학적 attention 근사 | 적용 가능한 가정·오차 상한·fallback 비용과 독립 품질 기준이 함께 제시됨 | 선형대수/통계/위상수학이라는 분야명만으로 구현 선정 |

## Serving milestone 공통 계약

1. 135M은 기존 회귀 축으로 유지한다. 확장 모델 1개, 짧은/긴 prompt, 짧은/긴 출력, 공유/비공유, C1/C8/C32/C64, closed-loop 및 open-loop를 단계적으로 포함한다. 고정된 한 모델·32-output-token 결과만으로 최종 목표를 종료하지 않는다.
2. N01~N02에서 실제 지원 한계와 메모리 예산을 확인한 후 정확한 길이·도착률·SLO·샘플 수를 manifest에 고정한다. 아직 미정인 값을 확정됐다고 가장하지 않는다. C64 client concurrency와 active capacity 32 제한은 다른 값으로 표기한다.
3. 주요 승격 비교는 최소 세 번의 균형 잡힌 엔진 순서 반복을 계획한다. warmup/retained 수는 tail 표본 수와 전체 소요를 근거로 사전 결정한다. 신뢰구간은 요청을 독립 표본이라고 가정하기보다 run 간 변동을 우선 반영하고, 불확실하면 판정 유보한다.
4. 전체 요청·오류·취소·거절·timeout을 기록한다. 유리한 일부 lane만 선택하거나 실패 실행을 합쳐 완전한 matrix로 만들지 않는다. 30분 이상 soak를 최종 승격 전에 별도로 계획하고 memory 증가·queue 회복·P99를 확인한다.
5. 각 의미 있는 milestone의 표: 기존 Riley/후보/vLLM/지원 비교군, throughput, TTFT/TPOT P50/P95/P99, E2E P95/P99, error/reject 비율, GPU/CPU/메모리, 초기화 시간, numerical profile. 우위·미달·미검증을 구별한다.
6. 기존 목표는 그대로다: 동일 조건에서 throughput 최소 동등(목표 +15%), TTFT/TPOT 동등 이하(목표 −10%), 높은 concurrency의 tail·안정성 유지. 어떤 허용 tail/오류 예산을 적용할지는 측정 전에 고정한다.
7. Blender 세 작업은 사용자 지시에 따라 종료 상태를 유지하며 종료/실패/사용자 중단 후에도 복구하지 않는다. 다른 서비스를 종료하지 않는다. raw profile은 별도 보존하고 timed run 중 빌드·다운로드·profiling은 하지 않는다.

## 참고한 공식 자료와 적용 한계

2026-09-15 확인. 아래 문서는 적용 방향의 근거이며 Riley 성능 개선의 증거가 아니다.

- [SGLang attention backends](https://docs.sglang.io/docs/advanced_features/attention_backend): backend를 하드웨어·형상·지원 기능별로 검토하는 대조 자료. N04/H01/H02의 지원 matrix와 연결한다.
- [vLLM speculative decoding 문서, v0.20.2](https://docs.vllm.ai/en/v0.20.2/features/spec_decode.html): 중저 QPS의 memory-bound 조건을 명시한다. 이번 조회에서 확인한 버전의 설명이며 현재 vLLM 전체 기능 목록이나 Riley의 미래 이득을 보장하지 않는다. Riley의 기존 speculation 회귀와 함께 조건부 재진입에 사용한다.
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html): KV 전송과 독립 요청 계산 중첩을 다룬다. D01 뒤의 선택지이며 KV 전송·layout·라우팅 비용을 빼고 평가하지 않는다.

계획 작성 단계에서는 커밋·푸시·서버 변경·새 GPU 실행을 하지 않는다. 다음 구현 착수 시에는 이 문서의 첫 묶음과 남은 수치 계약 결정을 기준으로 범위를 정한다.
