# PR 07 — Llama-compatible 기준 Forward

**상태:** Active

**선행 조건:** [PR 06](06-core-primitives.md)  
**다음:** [PR 08 — Prefill Attention](08-prefill-attention.md)

[← 이전](06-core-primitives.md) | [목차](README.md) | [다음 →](08-prefill-attention.md)

## 목적

최적화보다 정확성을 우선하여 한 Llama-compatible checkpoint의 **cache 없는 full-sequence forward와 logits**를 완성한다.

이 단계가 Gate B다.

## 실행 그래프

```text
Token IDs
→ Embedding
→ N × Decoder Block
   → RMSNorm
   → Q/K/V projection
   → correctness-first attention
   → output projection
   → residual
   → RMSNorm
   → gated MLP
   → residual
→ Final RMSNorm
→ LM Head
→ Logits
```

## Attention reference

이 PR에서는 느린 분리 구현을 허용한다.

- QK matmul
- scale
- causal mask
- softmax
- AV matmul

목적은 attention interface와 tensor layout의 정확성을 검증하는 것이다. FlashAttention 대체는 PR 08에서 수행한다.

## Weight와 execution plan

- model load 후 immutable execution plan 생성
- layer별 weight binding 검증
- workspace 크기 사전 계산
- forward 중 weight name lookup 금지
- forward hot path에서 JSON/hash map 접근 금지

## 검증 레벨

1. embedding output
2. 첫 decoder block 주요 checkpoint
3. 중간 layer checksum
4. final hidden state
5. logits slice와 통계
6. top-k tokens

가능하면 PyTorch hook으로 생성한 golden fixture와 비교한다.

## 메모리 검증

- 반복 forward에서 allocation count 증가 없음
- peak VRAM 기록
- weight alias/tied head 확인
- 오류 발생 시 intermediate buffer 회수

## 구현 결과

### 범위와 review 단위

PR 06까지는 검증된 BF16 primitive, cuBLASLt GEMM과 GPU weight owner는 있었지만 이를
decoder graph로 묶는 immutable plan, correctness-first attention, full-sequence logits
경로가 없었다. PR 07은 다음 review commit으로 Gate B 경계를 추가했다.

1. `5bffa57` — allocation-free CUDA download와 명시적 transfer completion
2. `25ed2d9` — fixed-sequence immutable Llama execution plan과 direct physical binding
3. `f377b70` — materialized GQA reference attention과 causal softmax CUDA ABI
4. `1b5705c` — pinned Hugging Face BF16 18-point trace producer와 manifest contract
5. `f4d033f` — 30-layer cache-free forward, logits, owner/cleanup/error state machine
6. `8befe4e` — Hugging Face와 같은 BF16 RMSNorm/RoPE staging 경계
7. `93c1d33`, `1c5eab4` — residual diagnostic/final-hidden gate 분리와 strict lint 마감
8. `962d530` — nested GEMM poison 전파와 최초 divergence 진단 순서 감사 수정

검증된 구현 snapshot은 `962d5300a72b7af2bcc5819b4838758a7989f0bf`다. PR 06 parent
`5a5b366` 대비 전체 diff는 26개 파일 `+8,489/-17`줄이며, KV cache, decode,
optimized attention과 batching은 계속 비범위다.

### Execution plan과 production 경계

- `LlamaExecutionPlan`은 검증된 CUDA weight upload 뒤 fixed sequence length, 30개 layer의
  연속성, model-wide dimension/scalar, bias 부재, BF16 dtype, 각 weight의 shape/byte
  length와 tied LM-head physical identity를 검증한다. 성공한 plan은 weight name이나 hash
  map이 아니라 prebound physical index만 보관한다.
- `PreparedLlamaForward`는 uploaded weights, hidden/KV/intermediate/score/logit buffers,
  RoPE table, pinned I/O staging과 hidden/KV/intermediate/down/LM-head용 5개 deterministic
  GEMM plan을 하나의 owner로 묶는다. `upload_tokens`, `execute`, `download_logits`,
  `download_last_logits`, `forward`, `close`가 이 owner의 명시적 상태 전이를 구성한다.
- hot `execute`는 direct-indexed weight와 사전 할당 buffer만 사용한다. Native
  execution/completion 실패 또는 nested GEMM owner poison은 `ExecutionSite(layer, op)`를
  보존하고 outer owner도 poison하여 stale output 재사용을 막는다. Caller가 유발한
  pre-native validation 및 token/trace preflight는 nested owner가 healthy일 때만 retry
  가능하다. Native-return validation으로 GEMM이 poison되면 outer에도 전파한다. Cold
  preparation의 부분 할당은 회수하고 explicit `close`는 첫 실패 뒤에도 모든 resource의
  close를 끝까지 시도한 후 첫 오류를 반환한다.
- reference attention은 CUDA C++/C ABI 뒤에서 QK, scale+causal mask, row-wise softmax,
  AV를 분리 실행한다. GQA의 query-head→KV-head mapping을 명시하고 BF16 score/probability
  matrix를 materialize한다. 이는 정확성 기준 경로이며 PR 08의 online-softmax backend로
  대체될 대상이다.
- RMSNorm은 FP32 normalize 결과를 BF16으로 staging한 뒤 BF16 weight를 곱하고, RoPE는
  table과 각 product의 BF16 boundary를 Hugging Face 5.15.1 실행과 맞췄다. 이 수정으로
  초기 BF16 semantic mismatch를 제거해 final-hidden/token gate를 통과했으며 token
  불일치를 tolerance 확대로 숨기지 않았다.
- production forward는 Rust/CUDA/cuBLASLt만 사용한다. Python/PyTorch/Transformers는
  아래의 외부 golden producer에만 존재하며 native runtime이 import하거나 subprocess로
  호출하지 않는다.

### Golden trace와 수치 gate

고정 입력은 token ID
`[504, 2365, 6354, 16438, 11139, 253, 1890]`이고 checkpoint는
`HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`다.
producer는 clean Git tree, pinned Python/dependency/model/GPU contract와 Transformers Llama
source hash를 확인한 후 18개 BF16 tensor를 safetensors sidecar로 exclusive-write한다.
Rust gate는 schema/artifact/trace ID, token IDs, `S=7`, `use_cache=false`, sidecar SHA-256,
18개 key/dtype/shape/byte range와 data의 gap 없는 전체 coverage를 다시 확인한다. 전체
producer/model provenance의 재검증은 producer 책임이며 Rust consumer의 범위가 아니다.

18개 checkpoint 중 embedding은 byte-exact, 16개는
`cosine/max_abs/mean_abs` numeric gate, unnormalized `final_norm.input`은 divergence 위치를
보존하는 diagnostic-only checkpoint다. 실제 LM-head 입력이자 이 PR의 final hidden state는
`final_norm.output`이며 계속 numeric gate 대상이다. `layer0.output`과 `last_logits`에는
PR 01 E0 v2에서 미리 정한 세 공통 metric threshold만 재사용하며 relative-error metric을
검사했다고 주장하지 않는다. PR 01의 전체 corpus/alternate-reduction gate를 이 단일
S=7 trace가 대체하거나 다시 활성화한 것도 아니다.

| Checkpoint | cosine | max abs | mean abs | 판정 |
|---|---:|---:|---:|---|
| embedding | — | 0 | 0 | BF16 byte-exact |
| layer0 input norm ~ gated | 1.000000000000 | 0 | 0 | 11개 checkpoint exact |
| layer0 down | 0.999997353690 | 0.125000000 | 0.001529663 | strict gate 통과 |
| layer0 output | 0.999997297914 | 0.125000000 | 0.001516325 | predeclared 3-metric gate 통과 |
| layer14 output | 0.999999897207 | 2.000000000 | 0.079835634 | cumulative-hidden gate 통과 |
| final norm input | 0.999723488471 | 6.000000000 | 0.462789263 | diagnostic-only |
| final norm output | 0.999631281846 | 0.687500000 | 0.032231574 | final-hidden gate 통과 |
| last logits | 0.999943663085 | 0.803710938 | 0.264808974 | predeclared 3-metric gate 통과 |

각 threshold `(cosine_min, max_abs_max, mean_abs_max)`는 strict early
`(0.999, 0.5, 0.02)`, attention probability `(0.999, 0.02, 0.002)`, layer0 output
`(0.999983706829855, 0.3884272575378418, 0.008509292567237658)`, cumulative hidden
`(0.998, 3.0, 0.35)`, last logits
`(0.9979035305495393, 5.852936458587647, 1.151280319263363)`로 고정했다.

마지막 logits 행의 top-1은 rank까지 exact하고 top-10은 set-exact다. top-10 내부 순서까지
exact하다고 주장하지 않는다. full logits를 두 번 실행했을 때 byte-equal하며, 공통 prefix
4개와 다른 suffix를 가진 두 S=7 입력의 prefix logits도 byte-equal하다. 실제 greedy
token은 `4052`, ranked golden top-10은
`[4052, 655, 28, 2654, 6354, 970, 29, 198, 979, 5372]`다.

### 검증과 결과

실제 model/tokenizer/CUDA/GPU 실행은 로컬에서 하지 않았다. 로컬에서는 model-free CPU
test와 정적 검사만 수행했다. 실제 golden 생성과 native 실행은 clean Git checkout을
`server-4096`에 전송한 뒤 RTX 4090에서 수행했다. Golden producer는 고정 local cache와
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 사용했고, native CUDA container는
`--network none`으로 실행했다.

```text
local model-free Rust tests:        117 passed, 0 failed, 1 remote-only ignored
local fmt / non-CUDA strict Clippy: passed
remote reference unittest:          65 passed, 0 failed
remote workspace --all-features:    112 passed, 0 failed, 21 explicit GPU/model ignored
remote strict Clippy:               workspace/all-targets/all-features `-D warnings` passed
remote full SmolLM2 forward:        1 passed (18 captured/compared; 17 gated)
remote attention tests:             2 passed (S=1,2,5,7,31,32,33)
remote primitive regressions:       3 passed
100 hot execute calls:              rustinfer-owned CUDA allocation counters unchanged
failure/cleanup checks:             cold partial cleanup, foreign stream, explicit close passed
Compute Sanitizer forward:          1 passed / 0 errors / 0 bytes leaked
Compute Sanitizer attention:        2 passed / 0 errors / 0 bytes leaked
Compute Sanitizer primitives:       3 passed / 0 errors / 0 bytes leaked
device-wide VRAM samples (50 ms):   239 samples; 235 MiB min, 910 MiB peak, 675 MiB delta
```

Attention 전용 fixture는 staged-BF16 CPU reference와 QK/scale/mask/softmax/AV를 비교하고
모든 future probability가 exact zero인지 확인한다. S=1, 2, 5, 7, 31, 32, 33은 한 warp
안쪽과 32-token 경계를 모두 포함한다. Full-forward test는 100회 전후 CUDA allocation
statistics equality, tied head, trace/public logits equality, deterministic rerun, causal common
prefix, 실패한 cold prepare 뒤 zero allocation, foreign-context rejection, 최종 zero-allocation
close를 함께 검증한다. VRAM 값은 process-owned allocation report가 아니라 같은 GPU의
device-wide 관측치이므로 baseline과 delta를 함께 기록한다.

S=7 plan은 273 logical/272 physical weight binding과 269,030,016 weight bytes, 825,262
graph bytes/18 graph allocations, 기본 4,194,304-byte pinned I/O staging을 고정한다. 실제
device total에는 선택된 GEMM workspace가 추가된다. Materialized attention은 SmolLM2에서
`18 × S²` bytes이므로 기본 512 MiB attention budget은 model maximum 8192와 별개로
`S≤5461`까지만 허용하며, 더 긴 reference forward에는 budget을 명시적으로 늘려야 한다.

100회 equality는 rustinfer가 소유한 CUDA device/pinned allocation counter 기준이다.
성공 경로의 Compute Sanitizer는 zero-error/zero-leak이지만 arbitrary device-fault injection
뒤의 driver/cuBLAS 내부 allocation 회수까지 증명하지 않는다. Ambiguous native completion은
stale handle을 재노출하지 않고 fail-closed retain할 수 있으며, 실제 device-fault 격리와
복구 gate는 PR 16 범위다.

검증 환경은 RTX 4090(compute capability 8.9), driver `580.173.02`, CUDA runtime `12.8.1`,
nvcc `12.8.93`, Rust `1.85.0`, Compute Sanitizer `2025.1.0.0`이다. Container image ID는
`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`다.
Golden lane은 Python `3.13.15`, PyTorch `2.13.0`, Transformers `5.15.1`, safetensors
`0.8.0`을 사용했다.

외부 evidence는 다음 append-only 위치에 보존했다.

```text
source commit:       962d5300a72b7af2bcc5819b4838758a7989f0bf
source archive sha:  a585305235538c6725455268d1ee12cd7a8c814fcd304dbc237239ffbb32901a
Git bundle sha:      6b9d35502ea2cac421facc5b63c662dbefe648ee77ffd4a189b0acc3d37c1877
golden manifest sha: 6f4214250c3ada145aa99d73943ffd5eff63346c6a90a3de5d0296fac89c07dc
golden sidecar sha:  569860e3f6b24f4ed4ec9e9aad2867b598a2ff2cfe6c154a2e9724b1e767d214
evidence root:       server-4096:/home/psyche/rustinfer-artifacts/pr07/962d5300a72b7af2bcc5819b4838758a7989f0bf/
SHA256SUMS sha:      2e70342c470f228c7345dbfb298f32f30ddb2a5a6181159682e254c6e371f792

golden/golden-produce.log
golden/hf-bf16-s7-manifest.json / golden/hf-bf16-s7.safetensors
validation/python-reference-unittest.log / validation/golden-topk.txt
validation/workspace-all-features.log / validation/clippy-all-features.log
validation/llama-forward-golden-gpu.log / validation/llama-forward-vram-sampled.log
validation/attention-gpu.log / validation/primitives-gpu.log
validation/compute-sanitizer-forward.log
validation/compute-sanitizer-attention.log / validation/compute-sanitizer-primitives.log
validation/forward-vram-samples.csv / validation/forward-vram-summary.txt
environment/gpu.txt / environment/image-id.txt / environment/toolchain.txt
environment/source-commit.txt / environment/source-status.txt
environment/source-transport-sha256.txt
```

### 롤백과 PR 크기 예외

전체 diff는 권장 hand-written production 크기를 넘지만, 한 가지 질문인 “검증된
checkpoint의 cache-free Llama logits를 native CUDA로 재현할 수 있는가”를 닫는다.
Review는 immutable plan(`25ed2d9`), attention(`f377b70`), golden contract(`1b5705c`),
full forward(`f4d033f`), BF16 semantic fix(`8befe4e`)로 나눌 수 있다. 문제 발생 시 이
main commit만 골라 되돌리지 않고 PR 07 구현 range `5a5b366..962d530`의 11개 commit
전체 또는 최종 merge commit을 단위로 revert해야 PR 06의 primitive/weight-upload 경계로
돌아간다. 이 rollback은 PR 08 착수 전에는 독립적이지만, 이후 단계가 이 interface에
의존하기 시작하면 downstream commit도 함께 되돌려야 한다.

## 비범위

- KV cache
- token generation loop
- optimized attention
- batching
- Qwen
- server

## 완료 기준

- [x] S=7 동일 token IDs의 마지막 `[V]` logits 행이 허용 오차 내 일치
- [x] 마지막 logits 행의 greedy next token이 golden result와 일치
- [x] attention primitive의 S=1,2,5,7,31,32,33에서 causal mask 정확
- [x] token upload 뒤 100회 hot `execute`에 rustinfer-owned allocation 증가 없음
- [x] graph 실행 오류가 layer 또는 global op 위치를 표시

**중단 조건:** token 결과가 일치하지 않으면 tolerance를 확대하지 말고 최초 divergence layer를 찾는다.

구현 gate는 통과했다. 이 문서는 선행 PR과 함께 merge되기 전까지 `Active`, merge 후
`Complete`로 전환한다.

[← 이전](06-core-primitives.md) | [목차](README.md) | [다음 →](08-prefill-attention.md)
