# Hugging Face Rust 도입: serving 교체가 아닌 정확성·checkpoint 경계 — 2026-09-16

상태: 조사, Qwen2.5-3B raw-logit 대조, P0/V1 layer-0 pre-attention trace와 P0/V2 Q/K/V bias-boundary trace를 완료했다. V2는 현재 Riley GEMM 결과가 HF no-bias shadow endpoint와 exact이고, 불일치가 post-bias 경계에서 시작함을 보였다. 다음 correctness 진단은 V2b의 명시적 staged-bias reference다. 이 문서는 Rust serving 경로에 Python을 넣지 않는다. HF Python은 외부 create-only correctness artifact를 만드는 오프라인 기준으로만 남는다.

## 결정

Hugging Face의 Rust 생태계는 도입한다. 다만 구성요소마다 역할을 분리한다.

| 구성요소 | Riley에서의 역할 | serving hot path 여부 | 1차 판단 |
|---|---|---:|---|
| `candle-core` / `candle-transformers` | Qwen2 독립 raw-logit discriminator | 아니오 | 별도 diagnostic binary로 시험 |
| `hf-hub` | immutable revision checkpoint 준비·검증 | 아니오 | 별도 `riley-checkpoint` CLI로 도입 후보 |
| `tokenizers` | Qwen tokenizer parity와 CPU 비용 대조 | 요청 전 CPU 단계만 | parity 후 선택 |
| `safetensors` | format 상호운용성 대조 | startup만 | 기존 strict loader와 동등성이 증명될 때만 검토 |

Candle을 Riley serving engine 전체의 대체물로 사용하지 않는다. Candle Qwen2 구현은 layer별 KV를 보관한 뒤 decode에서 `Tensor::cat`으로 이어 붙이고, KV를 repeat한 뒤 일반 QKᵀ–softmax–PV를 수행한다. Riley의 paged KV, continuous batching, request ownership, CUDA Graph, long-lived workspace와는 실행 구조가 다르다. 이 경로를 serving에 바로 넣으면 vLLM보다 높은 throughput이라는 목표를 검증할 수 없다. 근거: [Candle Qwen2 source](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2.rs).

반면 이번 대조에서 Riley의 cache-free reference와 paged-KV reference는 같은 BF16 행을 만들었지만, HF eager와 addressable vocabulary 영역부터 달랐다. 독립 Rust implementation은 이 차이가 Riley 공통 forward인지 HF eager의 연산 계약인지 분리하는 데 직접 도움이 된다.

### Candle compatibility boundary

현재 Riley workspace의 MSRV는 Rust 1.85다. 최신 Candle 0.11은 transitive `zip`의 Rust 1.88 요구 때문에 이 기준에서 빌드되지 않았고, 0.10도 Rust 1.85에서 사용할 수 없는 API를 사용한다. Qwen2가 포함된 Candle **0.9.1**은 Rust 1.85 CPU build와 CUDA 12.8 compile probe를 통과했다. 따라서 HF-R0은 `=0.9.1`을 독립 nested workspace에 pin한다. 이는 최신 Candle을 production에 고정한다는 뜻이 아니다. CUDA 13/Hopper/Blackwell qualification은 MSRV를 다시 판정한 최신 Candle lane에서 별도로 수행한다.

Candle 0.9.1 Qwen2는 BF16 RoPE table, BF16 softmax, Candle CUDA RMSNorm을 사용하고, 첫 forward에 internal KV cache를 기록한다. pinned HF eager의 F32 RoPE/softmax/RMSNorm 및 `use_cache=False`와 같은 수치 계약이 아니다. 그러므로 Candle row hash는 **수치 삼각측량 관측치**이며 byte-equality gate가 아니다. Candle/HF/Riley가 같거나 다르다는 사실만으로 어느 구현의 오류를 단정하지 않고, 후보 ordering·tolerance·layer trace를 다음 판정에 사용한다.

## 현재 수치 근거

동일 Qwen2.5-3B-Instruct revision, 2,048개의 token ID `3409`, BF16, RTX 4090 조건이다. HF는 eager/cached-off, Riley는 cache-free reference와 reference-paged KV를 사용했다. 이 표는 serving 성능 비교가 아니라 raw-logit correctness 대조다.

| 실행 | 전체 BF16 행 SHA-256 | addressable 151,665 tokens SHA-256 | top-1 | logit(304) | logit(374) |
|---|---|---|---:|---:|---:|
| HF Transformers eager | `66b5f0bb…ffcfca9d` | `ab814763…57debb24` | 304 | 14.0625 | 13.75 |
| Riley cache-free reference | `a935ebd4…e69d7712` | `da7a87cc…0787c174` | 304 | 13.9375 | 13.6875 |
| Riley reference-paged KV | `a935ebd4…e69d7712` | `da7a87cc…0787c174` | 304 | 13.9375 | 13.6875 |
| existing vLLM serving oracle | raw row 미수집 | raw row 미수집 | 374 | 미수집 | 미수집 |

따라서 KV paging은 현 차이의 원인이 아니다. Riley와 HF의 후보 토큰 순서는 대체로 겹치지만, 실제 생성 가능한 vocabulary 영역에서도 값과 hash가 다르다. vLLM의 `374`는 별도로 설명해야 한다. HF Python artifact와 Rust test artifact의 pre/post I/O PSI는 외부 evidence에 보존했으며, 높은 I/O PSI를 결과 필터·가중치·보정에 사용하지 않는다.

### HF-R0 execution receipt — 2026-09-15

`13fe4fd7`에서 Candle 0.9.1 CUDA diagnostic을 RTX 4090에서 한 번 실행했다. artifact는 checkout 밖의 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-candle-oracle-20260915T171457Z/qwen3b-candle-raw-logits.json`에 create-only로 저장했고 SHA-256은 `1eb8e8b8…c193f73ba`다. Rust 1.85의 CPU test/clippy와 원격 CUDA 12.8 compile은 통과했다. 원격 first run의 296초에는 debug binary build와 checkpoint GPU materialization이 포함되어 있으므로 forward latency나 throughput으로 사용하지 않는다.

| 실행 | 전체 BF16 행 SHA-256 | addressable 151,665 tokens SHA-256 | top-1 | logit(304) | logit(374) |
|---|---|---|---:|---:|---:|
| Candle 0.9.1 Qwen2 CUDA first forward | `3b3facf2…18bd78d5` | `bec504d4…44196f72` | 11 (`304`와 14.1875 동률) | 14.1875 | 14.125 |

Candle과 HF top-32의 교집합은 30개다. `304`, `11`, `374`는 모두 두 결과의 가장 높은 후보군에 남았지만 raw hash는 다르다. Candle의 BF16 RoPE/softmax/RMSNorm과 internal KV write 때문에 이 row를 HF/Riley byte-equality 판정에 사용하지 않는다. run 전후 GPU idle memory는 모두 335 MiB였고 I/O PSI는 artifact 옆에 보존했다. I/O PSI는 이 numerical observation을 제외·가중·보정하는 데 사용하지 않는다.

### P0/V1 — Qwen3B P2048 layer-0 pre-attention trace receipt — 2026-09-16

외부 create-only artifact는 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-layer0-trace-20260915T175259Z-r2/`에 있다. Qwen2.5-3B-Instruct `aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16, RTX 4090, P2048의 token ID `3409`, `use_cache=false`, materialized reference attention으로 실행했다. source revision은 `37594b05`이며 trace source 범위는 clean이었다. comparison JSON SHA-256은 `96dd23bdd8441e001c725df004776006b55d8b22c91d0a4af43ec5baec5c9987`, HF manifest는 `aeacdf2ba3561fce238aaed91f761f270c2dbfe4011d658bacfdf005961472ff`, sidecar는 `b613146b1819891773272d2fffb0dde47545676b81fec2582f10897b71543d34`다.

이 artifact는 `performance_claim_eligible=false`다. shared-host I/O full PSI가 높은 상태에서 발생한 load wall time은 serving throughput, TTFT, TPOT, vLLM 비교 결과가 아니다.

| Stage | BF16 exact | 불일치 / 전체 | 최대 절대차 | 판정 |
|---|---:|---:|---:|---|
| `embedding` | 예 | 0 / 4,194,304 | 0 | exact |
| `layer0.input_norm` | 예 | 0 / 4,194,304 | 0 | exact |
| `layer0.q_proj` | 아니오 | 1,083,392 / 4,194,304 | 0.0625 | 첫 불일치 |
| `layer0.k_proj` | 아니오 | 110,592 / 524,288 | 0.125 | projection 이후 차이 |
| `layer0.v_proj` | 아니오 | 155,648 / 524,288 | 0.00390625 | projection 이후 차이 |
| `layer0.q_rope` | 아니오 | 1,051,621 / 4,194,304 | 0.125 | upstream projection 차이를 포함 |
| `layer0.k_rope` | 아니오 | 102,154 / 524,288 | 0.25 | upstream projection 차이를 포함 |

V1의 Q/K/V projection capture는 Rust에서 `execute_projection_bias` 뒤에 수행된다. 따라서 projection GEMM의 reduction/layout 차이와 BF16 row-bias addition 차이를 이 결과만으로 구분할 수 없다.

다음 P0/V2는 동일 pinned contract에서 Q/K/V bias 경계를 별도 trace ID와 create-only artifact로 기록한다. Riley 쪽은 실제 standalone GEMM 직후와 별도 bias 연산 뒤를 각각 capture한다. HF 쪽의 `*.unbiased_linear`은 fused `nn.Linear`의 관측 불가능한 내부값이라고 주장하지 않고, unmodified module output을 먼저 capture한 뒤 같은 input·weight로 별도 실행한 `torch.nn.functional.linear(input, weight, bias=None)` shadow endpoint로 정의한다. V1 post-bias artifact는 보존한다.

shadow no-bias endpoint에서도 차이가 시작되면 GEMM 실행·reduction·layout 후보를, shadow endpoint가 exact이고 real post-bias에서만 차이가 생기면 HF fused epilogue와 Riley의 BF16→FP32 add→BF16 staged rounding 경계를 다음 진단 대상으로 삼는다. 후자의 경우 row-bias kernel 오류로 단정하지 않고, 먼저 명시적 staged reference를 추가하는 V2b를 수행한다. 어느 경우에도 이를 성능 개선 또는 serving correctness pass로 승격하지 않는다.

### P0/V2 — Qwen3B P2048 Q/K/V bias-boundary trace receipt — 2026-09-16

외부 create-only artifact는 `/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-qkv-bias-boundary-20260915T185511Z/`에 있다. source revision은 `aa2b9879daa3f36881e65a25a71d5a5245510a13`이며, HF manifest SHA-256은 `0a08b97e4a44784ac7ac80b616f94fe783f36ef117c3bd5d903af4541b670dbe`, safetensors sidecar는 `2305388416e03caf33480a79519b24815247ed5624849092b0af5dcec53f3702`, Riley comparison JSON은 `21647bc2ee227d6a65882128061c30806ab6debfa64b94581b105ba4cd54fbb1`다. pinned Qwen2.5-3B-Instruct revision, BF16, RTX 4090, P2048, `use_cache=false`, materialized reference attention contract는 V1과 같다.

HF의 `*.unbiased_linear`은 module forward를 수정하지 않은 뒤 동일 input·weight에 `torch.nn.functional.linear(..., bias=None)`을 적용한 shadow endpoint다. Riley의 동명 capture는 standalone GEMM 직후다. `*.q_proj`/`k_proj`/`v_proj`는 각각 원래 HF module output과 Riley의 BF16→FP32 bias-add→BF16 output을 비교한다. 따라서 fused `nn.Linear` 내부 activation을 관측했다고 주장하지 않는다.

| Stage | BF16 exact | 불일치 / 전체 | 최대 절대차 | 평균 절대차 | 판정 |
|---|---:|---:|---:|---:|---|
| `layer0.q_proj.unbiased_linear` | 예 | 0 / 4,194,304 | 0 | 0 | 현재 GEMM 경계 exact |
| `layer0.q_proj` | 아니오 | 1,083,392 / 4,194,304 | 0.0625 | 0.0010425765 | post-bias에서 시작 |
| `layer0.k_proj.unbiased_linear` | 예 | 0 / 524,288 | 0 | 0 | 현재 GEMM 경계 exact |
| `layer0.k_proj` | 아니오 | 110,592 / 524,288 | 0.125 | 0.0022181869 | post-bias에서 시작 |
| `layer0.v_proj.unbiased_linear` | 예 | 0 / 524,288 | 0 | 0 | 현재 GEMM 경계 exact |
| `layer0.v_proj` | 아니오 | 155,648 / 524,288 | 0.00390625 | 0.0003593564 | post-bias에서 시작 |

Rust GPU test는 38.82초에 통과했고 close 뒤 CUDA allocation zero도 검증했다. 다만 이 시간에는 checkpoint I/O와 initialization이 포함되어 있으므로 serving latency나 kernel performance가 아니다. run 전후 GPU idle memory는 모두 335 MiB였고, I/O PSI는 artifact에 기록했지만 결과의 제외·가중·보정에 사용하지 않았다. artifact 자체도 `performance_claim_eligible=false`다.

판정: 현재 standalone GEMM/reduction/layout은 이 exact no-bias endpoint에서 원인이 아니다. 세 projection 모두 real post-bias에서만 달라지므로, 다음 작업은 fused HF epilogue와 Riley의 staged BF16 rounding 사이를 가르는 **V2b explicit staged-bias reference**다. V2b는 같은 no-bias BF16 row와 FP32 bias로 `round_bf16(no_bias.to(float32) + bias.to(float32))` endpoint를 별도 기록한다. Riley가 V2b staged endpoint와 exact이면 현 불일치는 HF fused epilogue contract 차이로 분류하고, 그렇지 않으면 bias row/layout/operator를 점검한다. 그 전에는 GEMM kernel 교체나 serving 성능 주장으로 진행하지 않는다.

## PR HF-R0 — Candle Qwen raw-logit diagnostic

대상: [`tools/candle-qwen3b-oracle`](../../tools/candle-qwen3b-oracle)의 독립 nested diagnostics workspace와 Cargo lock. Riley server, scheduler, CUDA ABI, serving API에는 의존성을 연결하지 않는다.

묶음:

1. `candle-core`, `candle-nn`, `candle-transformers`를 `=0.9.1`로 pin하고, 기본 workspace build에는 Candle CUDA dependency가 들어오지 않도록 별도 nested crate로 격리한다. Rust 1.85 compile과 CUDA 12.8 compile을 별도로 확인한다.
2. local-only Qwen2.5-3B checkpoint, pinned config/receipt, P2048 workload만 받는 create-only CLI를 만든다. token ID는 직접 Tensor로 만들며 tokenizer·Hub 네트워크·sampling·serving socket을 호출하지 않는다.
3. Candle CUDA가 가능한 환경에서는 BF16 first forward를 실행해 full/addressable/tail BF16 hash, top-32, 304/374/3409 probes, source/dependency/GPU provenance를 JSON artifact로 기록한다. Candle의 internal KV write와 BF16 numerical contract를 artifact에 명시한다. CUDA가 없는 환경은 compile/schema/unit test만 수행하고 GPU result를 주장하지 않는다.
4. artifact schema, local-only binding, checkpoint regular-file/receipt contract, output create-only, CPU-only validator를 테스트한다. Candle 결과를 Riley serving quality gate나 performance result로 승격하지 않는다.

판정:

- Candle이 HF 또는 Riley의 candidate ordering/tolerance window에 가까워도, BF16 intermediate 차이를 먼저 고려한 layer trace를 만든다.
- Candle이 어느 쪽과도 다르거나 세 결과가 모두 다르면, full-row hash만으로 원인을 단정하지 않고 norm, RoPE, projection, attention, reduction의 layer trace artifact를 별도 PR로 만든다.

완료: Rust-only numerical triangulation diagnostic이 raw output과 provenance를 남기고, 4090에서 지원되면 HF/Riley와 한 표에 비교된다. Candle의 single-request 결과가 serving throughput 개선이나 exact-HF correctness pass로 표시되지 않는다.

롤백: diagnostics crate/feature만 제거한다. Riley server와 model loader에는 runtime dependency가 없으므로 serving rollback이 필요하지 않다.

## PR HF-R1 — `hf-hub` checkpoint preparation CLI

대상: server와 분리된 `riley-checkpoint` CLI.

묶음:

1. `hf-hub` blocking client로 model ID, immutable revision, explicit allowlist를 받는다. revision 없는 branch/tag와 implicit network fallback을 거부한다.
2. Hub cache에서 바로 serve하지 않는다. staging directory에 regular files로 materialize하고, expected size/SHA-256을 확인한 뒤 `riley-checkpoint.json`을 atomically publish한다. symlink cache entry는 Riley loader의 existing policy대로 거부한다.
3. download, checksum mismatch, partial staging, concurrent publish, offline validation을 검사한다. 모델 download 시간은 serving throughput/TTFT/TPOT에 포함하지 않는다.

완료: 기존 Riley loader가 검증한 regular-file checkpoint만 받아들이며, runtime server는 Hub client·token·network permission을 갖지 않는다. 공식 crate는 Hub API client이지 inference runtime이므로 이 PR은 수치 mismatch를 해결하는 작업이 아니다. 근거: [hf-hub](https://github.com/huggingface/hf-hub).

## PR HF-R2 — tokenizer parity와 선택

대상: Qwen tokenizer parity harness와 request-preparation benchmark.

묶음:

1. Hugging Face `tokenizers` Rust implementation으로 Qwen raw prompt IDs와 special-token policy를 독립 비교한다. chat template rendering policy는 Riley가 계속 소유한다.
2. Korean, code, long prompt, added token, truncation/error cases에서 exact IDs를 비교한다.
3. parity가 유지되는 경우에만 CPU request-preparation latency/allocations를 기존 tokenizer와 비교한다. gain이 반복 변동보다 작으면 도입하지 않는다.

완료: 동일 input IDs와 chat rendering contract가 증명되고 CPU 비용이 의미 있게 줄어들 때만 production tokenizer candidate로 승격한다. GPU decode 성능 개선이라고 주장하지 않는다. 근거: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers).

## 성능 경계

HF-R0~R2는 정확성·모델 준비·CPU pre-processing을 위한 PR이다. 이들만으로 vLLM serving 성능 목표를 달성했다고 표시하지 않는다. Candle diagnostic 결과와 strict/native-BF16 numerical profile을 확정한 뒤에만 기존 N03/N04 native GEMM·attention serving 작업으로 진행한다. 그 milestone에서 Riley baseline, candidate, pinned vLLM, 지원 가능한 SGLang/TRT-LLM을 같은 모델·workload·GPU·concurrency로 표에 기록한다.
