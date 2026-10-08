# PR 11 — Logits Processing, Sampling, Generation Loop

**상태:** Implemented — `d018c2fb0f74417df65a2faa2dcd66151a78fe47` 원격 GPU 검증 완료<br>
**선행 조건:** [PR 10](10-paged-kv-manager.md)  
**다음:** [PR 12 — Qwen 호환성](12-qwen-compatibility.md)

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)

## 목적

prefill, paged KV decode와 token 선택을 연결해 단일 요청의 완전한 autoregressive generation을 만든다. 이후 rejection-sampling 기반 speculative decoding을 정확하게 추가할 수 있도록 **request-local RNG와 sampling 의미 계약**을 먼저 고정한다.

이 단계가 Gate C다.

## 구현 결과

실행 및 검증 기준 revision은 `d018c2fb0f74417df65a2faa2dcd66151a78fe47`다.

- `6173af5` — caller-owned buffer로 raw token bytes와 whole-sequence bytes를 decode하는
  tokenizer streaming API
- `ed1ee39` — normative Philox request RNG와 preallocated CPU logits processing/sampling
- `bd2aebe` — model-independent `GenerationRequest`, `GenerationState`, stop/text state와
  structured finish/error contract
- `8573455` — reusable CUDA Llama prefill/decode owner와 sampling/streaming generation 연결
- `5cc9a4b` — 31-case golden, fixed-seed, cancellation, callback failure와 cleanup 원격 GPU gate
- `d018c2f` — arbitrary ByteLevel raw tail의 순서·byte 무손실 보존과
  `min_new_tokens` no-cross-gate stop watermark

`GenerationState`는 model과 CUDA를 참조하지 않는다. Token history, request-local RNG,
sampling parameters, stop state와 finish reason만 소유한다. CUDA adapter는 이 state를
`PreparedLlamaDecode`와 연결하며 한 요청이 끝날 때 reset하고 다음 요청에 같은 prepared
owner를 재사용할 수 있다.

## 정책 범위

초기 구현 순서:

1. greedy
2. temperature
3. top-k
4. top-p
5. repetition penalty
6. min/max new tokens
7. EOS/stop token

frequency/presence penalty와 복잡한 grammar constraint는 뒤로 미룬다.

## CPU와 GPU 경계

첫 버전은 정확성을 위해 CPU sampling을 허용할 수 있다. 단, logits 전체 copy 비용을 측정한다.

그 후 같은 PR에서 무리하게 GPU sampler를 추가하지 않는다. GPU sampling이 필요하면 작은 후속 commit 또는 PR 15 최적화 대상으로 둔다. CPU와 GPU sampler는 동일한 logits processing 순서와 RNG contract를 구현해야 한다.

## Sampling pipeline 순서

동일한 parameter 집합이라도 처리 순서가 달라지면 분포가 달라질 수 있으므로 순서를 명시한다.

```text
raw logits
→ token constraints
→ repetition or other penalties
→ temperature
→ top-k
→ top-p
→ normalization or equivalent sampling
→ categorical sample
```

각 단계는 입력·출력 dtype과 masking 값의 의미를 문서화한다.

구현된 stable contract는 다음과 같다.

```text
pipeline ID = bf16ne-constraints-unique-repetition-temperature-top-k-top-p-f64-v1
pipeline version = 1
categorical ID = u32-midpoint-token-id-ascending-categorical-v1

native-endian BF16 logits
→ token constraints
→ sign-aware repetition penalty, unique history token당 정확히 한 번
→ positive temperature scaling
→ deterministic top-k
→ deterministic top-p minimal prefix
→ F64 normalization
→ ascending token-ID categorical traversal
```

`temperature == 0`은 별도 greedy 경로다. Numeric logit tie는 낮은 token ID가 이기고 RNG를
소비하지 않는다. Stochastic categorical 성공은 U32 word를 정확히 하나 소비한다. Parameter,
shape, non-finite logit 또는 all-masked 오류는 draw 전에 반환한다.
Top-k/top-p의 정렬 tie-break도 token ID라 batch나 heap ordering에 의존하지 않는다.

## RNG 계약

- explicit seed
- request별 RNG state
- batch 순서 변화가 다른 request RNG를 오염시키지 않음
- deterministic mode와 high-performance mode 구분 가능
- RNG algorithm과 version을 결과 metadata에 기록
- snapshot, restore, fork 기능을 정의
- rejected 또는 cancelled branch의 RNG 소비 정책을 명시

권장 interface:

```rust
trait RequestRng {
    type Snapshot;

    fn snapshot(&self) -> Self::Snapshot;
    fn restore(&mut self, snapshot: &Self::Snapshot);
    fn fork(&self, domain: RngDomain) -> Self;
    fn next_uniform(&mut self) -> f32;
}
```

`fork`는 draft, target correction, user-visible sampling처럼 서로 다른 확률 경로가 같은 request 안에서도 독립적인 stream을 갖도록 한다.

Normative algorithm ID는 `riley.philox4x32-10.v1`이다. Random123 Philox4x32-10 core,
SHA-256 seed/stream/domain derivation, counter/word order, F64 open-interval mapping,
snapshot JSON과 fork derivation은 [`benchmarks/RNG.md`](../benchmarks/RNG.md)에 고정했다.
Greedy는 0 draw, user-visible stochastic sample 성공은 1 draw다. Snapshot/restore/fork는
draw를 소비하지 않는다. 마지막 `2^66`번째 word 뒤에는 counter를 wrap하지 않고 exhausted
error를 반환한다.

## Sampling 결과 계약

```rust
struct SamplingResult {
    token_id: u32,
    token_logprob: Option<f32>,
    finish_reason: Option<FinishReason>,
}
```

추후 speculative verifier가 필요로 하는 기능을 위해 sampler 내부에는 다음 primitive를 분리한다.

- processed logits 생성
- 특정 token의 probability 또는 log-probability 조회
- categorical sample
- 두 분포의 positive residual에서 sample할 수 있는 확장 지점

PR 11에서는 residual distribution sampler를 구현하지 않는다. API가 ordinary sampling과 강하게 결합되지 않도록 경계만 정한다.

## Generation state

```rust
GenerationRequest
GenerationState
SamplingParams
StopState
GeneratedToken
FinishReason
```

Finish reason:

- eos
- stop token/string
- length
- cancelled
- error

종료 우선순위는 `EOS → stop token → stop string → length`다. EOS와 stop token ID는 generated
history에는 남지만 자신의 decoded bytes는 text에서 숨긴다. Stop string은 UTF-8 문자열을
raw tokenizer bytes에서 찾고 matched bytes와 그 뒤 bytes를 text에서 제외한다.

`min_new_tokens`가 닫혀 있을 때 EOS/stop token은 ordinary visible token으로 취급하고 stop
string 검색도 비활성화한다. String stop은 gate가 열린 뒤 accepted token의 decoded bytes에서
시작해야 한다. Gate 이전에 시작한 prefix나 invalid raw tail 안의 과거 match는 나중에
재활성화하지 않는다.

Streaming `text`와 delta는 stop-prefix disambiguation이 끝나 안전하게 내보낼 수 있는 strict
UTF-8 prefix다. `StopState::pending_bytes`는 아직 possible stop prefix인 valid bytes도 보류한다.
Tokenizer가 incomplete 또는 definitively invalid UTF-8 bytes를 만들면 첫 unrepresentable
byte부터 raw tail 전체도 순서와 byte 그대로 보존한다. Replacement character를 삽입하거나
뒤의 valid-looking bytes를 앞으로 내보내지 않는다. `text.as_bytes()`와 `pending_bytes`를
이어 붙인 값은 matched stop 이전에 retained된 decoded raw output이다.

## CUDA generation 경계

`PreparedLlamaGeneration`은 prompt shape와 output capacity별로 decode owner, full BF16 logits
host staging, vocabulary-sized sampling scratch, tokenizer byte buffer와 CUDA timing event를
prepare한다. Token hot path에서 이 storage를 resize하거나 CUDA allocation을 만들지 않는다.

N개의 output token을 만들 때 model call은 prefill 1회와 최대 N-1 decode다. 마지막 sampled
token 뒤에는 사용하지 않을 logits를 만들기 위한 decode를 실행하지 않는다. Cancellation은
model 전과 model 완료 후/D2H 전에 각각 검사하며 두 경로 모두 새 RNG word를 소비하지 않는다.

각 token event는 다음 timing boundary를 제공한다.

- model stage와 CUDA-event GPU milliseconds
- model host wall
- full-logits D2H bytes와 wall
- constraints/logits processing/sampling CPU
- raw detokenize/UTF-8/stop CPU
- callback 직전 token total wall

Request summary는 위 합계와 전체 generate-call wall을 제공한다. Callback 오류는 accepted token과
소비한 RNG word를 되돌리지 않고 state를 `error`로 만든 뒤 KV owner를 reset한다. Recoverable
request/callback/cancellation 경로는 owner를 재사용하고, device/cleanup fatal 오류는 owner를
poison/close하는 fail-closed 정책이다.

## 검증

- greedy golden token exact match
- fixed seed 반복 결과
- snapshot 후 sample, restore 후 재실행 결과 일치
- request RNG fork 간 독립성
- batch 처리 순서 변경 시 request별 결과 계약 확인
- top-k/top-p boundary
- temperature 0 처리
- all-masked distribution 오류
- EOS 즉시 종료
- max token 종료
- UTF-8 decode boundary
- cancellation 시 cache와 RNG state 회수
- callback/consumer 오류 시 cleanup

## 검증 결과

Exact source snapshot `d018c2f…`을 read-only mount한 `server-4096`에서 검증했다. RTX 4090
(sm89, 24,564 MiB), driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93, Rust 1.85.0이며
container network는 비활성화했다. 모든 Cargo command는 `--locked --offline`이었다. 로컬에서는
model load, CUDA feature build 또는 inference를 실행하지 않았다.

- Strict CUDA all-features Clippy 통과
- Workspace all-targets/all-features `168 passed, 0 failed, 48 ignored`
- Workspace all-features doctest `13 passed, 0 failed`
- Runtime model-free unit `78 passed, 0 failed`
- Pinned SmolLM2 fixture 31/31: cache-on fixture = cache-off fixture = native adapter token IDs
- 30 length finish, 1 EOS finish; greedy RNG draw는 모든 case 0
- Prompt shape 9개: `1/2/10/48/54/128/1024/4096/8064`
- 16-token case는 모두 prefill 1 + decode 15
- Terminal raw UTF-8 tail 11 cases/184 bytes: incomplete 6, invalid 5; 무손실 보존
- Fixed-seed stochastic 8 tokens/8 draws; token/text/final snapshot 반복 exact
- Pre/post-model cancellation 0 draw; callback error 뒤 reset/reuse; KV와 CUDA allocation cleanup
- Full BF16 vocabulary logits D2H는 token당 98,304 bytes
- 독립 static review 2회에서 P1/P2 finding 없음

전체 evidence는
[`20260825T104035Z-rustinfer-generation-pr11-run001`](../benchmarks/results/20260825T104035Z-rustinfer-generation-pr11-run001/README.md)에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr11/d018c2f/full
source tar sha256=4cd807fcb2e5a59723b0ab1f3d559d68e62e195071c52d0223bb9ffd8b2fb341
SHA256SUMS.v3 sha256=46491762270d014b87421f954a6e6c71a576791cf613c0f72cc008c729b7b7ca
```

Debug test binary, 고정하지 않은 clock과 benchmark warmup/measurement protocol 때문에 timing은
boundary 수집의 기능 증거로만 사용한다. TTFT, ITL, throughput 또는 전후 성능 개선을 주장하지
않는다. CPU sampler가 매 token full logits row를 D2H copy하는 비용은 PR 15 최적화 후보다.

## Gumbel 계열에 대한 범위 결정

Gumbel-Softmax는 categorical variable의 학습용 연속 relaxation이며, 일반적인 inference sampling의 연산량을 직접 줄이는 기본 경로로 사용하지 않는다. 따라서 PR 11 범위에 포함하지 않는다.

향후 candidate-tree 또는 특수 stochastic decoding에서 Gumbel-Max/Top-k가 필요해지면 ordinary sampler와 분리된 연구 PR로 검토한다.

## 비범위

- continuous batching
- HTTP/SSE
- beam search
- speculative decoding
- residual distribution correction
- Gumbel-Softmax
- structured output

## 완료 기준

- [x] 단일 요청이 token stream을 생성
- [x] greedy golden sequence 일치
- [x] fixed seed sampling 재현
- [x] RNG snapshot/restore/fork unit test 통과
- [x] sampling transform 순서와 알고리즘 version이 문서화됨
- [x] 모든 종료 경로에서 KV block과 RNG state 회수
- [x] 생성 상태가 model runtime과 분리됨
- [x] per-token CPU/GPU timing이 수집됨

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)
