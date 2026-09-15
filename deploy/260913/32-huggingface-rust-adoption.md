# Hugging Face Rust 도입: serving 교체가 아닌 정확성·checkpoint 경계 — 2026-09-16

상태: 조사와 Qwen2.5-3B raw-logit 대조를 완료했다. 이 문서는 Rust serving 경로에 Python을 넣지 않는다. HF Python은 외부 create-only correctness artifact를 만드는 오프라인 기준으로만 남는다.

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

## 현재 수치 근거

동일 Qwen2.5-3B-Instruct revision, 2,048개의 token ID `3409`, BF16, RTX 4090 조건이다. HF는 eager/cached-off, Riley는 cache-free reference와 reference-paged KV를 사용했다. 이 표는 serving 성능 비교가 아니라 raw-logit correctness 대조다.

| 실행 | 전체 BF16 행 SHA-256 | addressable 151,665 tokens SHA-256 | top-1 | logit(304) | logit(374) |
|---|---|---|---:|---:|---:|
| HF Transformers eager | `66b5f0bb…ffcfca9d` | `ab814763…57debb24` | 304 | 14.0625 | 13.75 |
| Riley cache-free reference | `a935ebd4…e69d7712` | `da7a87cc…0787c174` | 304 | 13.9375 | 13.6875 |
| Riley reference-paged KV | `a935ebd4…e69d7712` | `da7a87cc…0787c174` | 304 | 13.9375 | 13.6875 |
| existing vLLM serving oracle | raw row 미수집 | raw row 미수집 | 374 | 미수집 | 미수집 |

따라서 KV paging은 현 차이의 원인이 아니다. Riley와 HF의 후보 토큰 순서는 대체로 겹치지만, 실제 생성 가능한 vocabulary 영역에서도 값과 hash가 다르다. vLLM의 `374`는 별도로 설명해야 한다. HF Python artifact와 Rust test artifact의 pre/post I/O PSI는 외부 evidence에 보존했으며, 높은 I/O PSI를 결과 필터·가중치·보정에 사용하지 않는다.

## PR HF-R0 — Candle Qwen raw-logit diagnostic

대상: 새 optional diagnostics crate/binary와 Cargo lock. Riley server, scheduler, CUDA ABI, serving API에는 의존성을 연결하지 않는다.

묶음:

1. `candle-core`, `candle-nn`, `candle-transformers`를 정확한 버전으로 pin하고, 기본 workspace build에는 Candle CUDA dependency가 들어오지 않도록 `candle-oracle` feature 또는 별도 crate로 격리한다. Rust 1.85 compile을 우선 확인한다.
2. local-only Qwen2.5-3B checkpoint, pinned config/receipt, P2048 workload만 받는 create-only CLI를 만든다. token ID는 직접 Tensor로 만들며 tokenizer·Hub 네트워크·sampling·serving socket을 호출하지 않는다.
3. Candle CUDA가 가능한 환경에서는 BF16 eager single forward를 실행해 full/addressable/tail BF16 hash, top-32, 304/374/3409 probes, source/dependency/GPU provenance를 JSON artifact로 기록한다. CUDA가 없는 환경은 compile/schema/unit test만 수행하고 GPU result를 주장하지 않는다.
4. artifact schema, local-only binding, checkpoint regular-file/receipt contract, output create-only, CPU-only validator를 테스트한다. Candle 결과를 Riley serving quality gate나 performance result로 승격하지 않는다.

판정:

- Candle≈HF, Riley≠Candle: Riley common forward의 norm, RoPE, projection, attention 또는 BF16 reduction trace를 layer 단위로 좁힌다.
- Candle≈Riley, HF≠Candle: Candle의 kernel/dtype/rope/attention contract을 먼저 확인하고 HF와의 차이를 기존 Riley bug로 단정하지 않는다.
- 세 결과가 모두 다름: full-row hash만으로 원인을 단정하지 않고 layer trace artifact를 별도 PR로 만든다.

완료: Rust-only third oracle가 raw output과 provenance를 남기고, 4090에서 지원되면 HF/Riley와 한 표에 비교된다. Candle의 single-request 결과가 serving throughput 개선으로 표시되지 않는다.

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
