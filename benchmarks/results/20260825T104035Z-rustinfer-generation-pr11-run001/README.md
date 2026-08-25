# PR 11 sampling and generation evidence — run001

`d018c2fb0f74417df65a2faa2dcd66151a78fe47`의 request-local Philox RNG,
CPU logits processing/sampling, model-independent generation state와 CUDA Llama generation
adapter를 RTX 4090/sm89에서 검증했다. 모든 CUDA compile, checkpoint load와 model inference는
`server-4096`의 network-disabled container에서 수행했다. 로컬에서는 model-free unit test,
format, lint와 source review만 수행했다.

## 결론

PR 11 Gate C를 통과했다. Pinned SmolLM2-135M fixture 31개에서 cache-on과 cache-off golden
token sequence가 먼저 서로 일치했고, native generation adapter의 token sequence도 두
fixture와 모두 exact였다. Greedy는 RNG word를 소비하지 않았다. 16-token 요청은 prefill
1회와 decode 15회만 실행해 마지막 token 뒤의 불필요한 model step을 만들지 않았다.

Fixed-seed stochastic generation은 token, text와 최종 RNG snapshot이 반복 실행에서
일치했다. 8개 token에 정확히 8개 RNG word를 소비했다. Pre-model 및 post-model cancellation은
추가 RNG word를 소비하지 않았고, callback 오류 뒤 owner/KV state를 reset해 다음 요청에서
재사용했다. 명시적 close 뒤 runtime CUDA allocation accounting은 0이었다.

## 고정된 계약

- RNG: `rustinfer.philox4x32-10.v1`; request-local stream, snapshot/restore/fork, 최대 `2^66`
  word 뒤 fail-closed exhaustion
- Sampling pipeline:
  `bf16ne-constraints-unique-repetition-temperature-top-k-top-p-f64-v1`
- Categorical algorithm: `u32-midpoint-token-id-ascending-categorical-v1`
- 처리 순서: constraints → unique-history repetition penalty → temperature → top-k → top-p →
  F64 normalization → ascending-token-ID categorical traversal
- Greedy: `temperature == 0`, lower token ID tie-break, RNG 0 draw
- Stochastic: 성공 시 U32 word 정확히 1개 소비; validation/all-masked 오류는 0 draw
- 종료 우선순위: EOS → stop token → raw-byte stop string → length
- `min_new_tokens`: gate가 열린 뒤 accepted token에서 시작한 stop string만 유효하다. Gate가
  닫힌 동안 시작한 prefix는 이후 재활성화하지 않는다.
- Streaming text: stop-prefix disambiguation이 끝나 safely emittable한 strict UTF-8 prefix만
  `text`/delta로 노출한다. `pending_bytes`는 possible stop prefix 또는 첫 unrepresentable
  byte 이후 raw tail을 순서와 byte 그대로 보존한다. 둘을 이어 붙이면 matched stop 이전의
  retained decoded raw output이 된다.

ByteLevel tokenizer는 token sequence에 incomplete뿐 아니라 definitively invalid UTF-8 raw
tail도 만들 수 있다. Golden 31건 중 terminal raw tail은 11건/184 bytes였고, 첫 오류 기준
incomplete 6건과 invalid 5건이었다. Replacement character를 삽입하거나 뒤 byte를 앞으로
재배치하지 않는다.

## 원격 GPU 검증

| Gate | 결과 |
|---|---|
| CUDA all-features strict Clippy | pass |
| Workspace all-targets/all-features | 168 passed, 0 failed, 48 ignored |
| Workspace all-features doctest | 13 passed, 0 failed |
| 31-case greedy golden | 31/31 exact; cache-on/off/native adapter |
| Greedy RNG | 모든 case 0 draw |
| Finish reason | length 30, EOS 1 |
| Prompt shapes | 9: 1, 2, 10, 48, 54, 128, 1,024, 4,096, 8,064 |
| Fixed-seed stochastic | 8 tokens, 8 draws, token/text/snapshot repeat exact |
| Cancellation | pre/post-model 모두 0 draw |
| Callback failure | 1 accepted stochastic draw 뒤 error; owner/KV recovery와 reuse pass |
| Model steps | N sampled tokens에 prefill 1 + 최대 N-1 decode |
| Logits copy | BF16 vocabulary row 98,304 bytes/token |
| Timing | token별 model GPU/wall, D2H, CPU sampling, detokenize/stop, total wall 수집 |
| Cleanup | KV reset; explicit close 뒤 CUDA allocation accounting 0 |

`generation-lifecycle.log`는 처음 `--ignored`를 빠뜨려 compile만 하고 test 0개를 실행한
진단 기록이다. 실제 lifecycle gate의 authoritative log는
`generation-lifecycle-executed.log`이며 1 passed를 기록한다. 실패/무실행 기록을 통과
증거로 덮어쓰지 않았다.

## 환경과 명령

- Host: `server-4096`, Intel Core i7-13700K, 24 logical CPUs, 67,185,598,464 bytes RAM
- Host kernel: Linux 6.8.0-138-generic; container OS: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- Driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- Image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- Model: `HuggingFaceTB/SmolLM2-135M` revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16, batch 1
- Container network disabled; source/checkpoint/Cargo registry read-only; target/evidence만 writable

주요 command는 모두 `--locked --offline`로 실행했다.

```text
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test --workspace --all-targets --all-features --locked --offline
cargo test --doc --workspace --all-features --locked --offline
cargo test -p rustinfer-runtime --test llama_generation_gpu --features cuda --locked --offline <golden-test> -- --ignored --exact --nocapture
cargo test -p rustinfer-runtime --test llama_generation_gpu --features cuda --locked --offline <lifecycle-test> -- --ignored --exact --nocapture
```

## Artifact와 provenance

이 디렉터리의 `raw-events.jsonl`은 두 authoritative summary marker를 보존하고,
`metadata.json`은 구현, environment, validation과 limitation 계약을 기계가 읽을 수 있게
기록한다. 전체 stdout과 exact source tar, GPU/driver, image inspect, host, toolchain,
checkpoint file hashes 및 normalized command record는 다음 append-only 원격 root에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr11/d018c2f/full
payload entries: 28
regular files: 25
regular-file bytes: 49,135,087
artifact-root apparent bytes: 49,151,471
SHA256SUMS.v3 sha256: 46491762270d014b87421f954a6e6c71a576791cf613c0f72cc008c729b7b7ca
source tar sha256: 4cd807fcb2e5a59723b0ab1f3d559d68e62e195071c52d0223bb9ffd8b2fb341
```

초기 `SHA256SUMS`는 먼저 수집한 source와 validation logs를 고정한 기록으로 그대로 남겼다.
Raw environment/command metadata를 append한 `SHA256SUMS.v2`도 보존한다. Checkpoint manifest와
config 원문까지 append한 뒤 final canonical `SHA256SUMS.v3`가 앞선 checksum 두 개와 전체
payload를 검증한다. 기존 evidence file을 덮어쓰지 않았다.

앞선 `8573455`, `32b69ca`, `fce0e01`, `5cc9a4b`, `fcf3ba3`, `32ca63c` 원격 roots는
compile/lint/UTF-8 정책을 교정한 append-only 진단 기록이며 통과 evidence가 아니다.

## 제한

- 단일 request, batch 1이다. Scheduler, continuous batching, HTTP/SSE와 speculative
  decoding은 후속 PR 범위다.
- 첫 sampler는 CPU implementation이며 매 token 전체 BF16 logits row 98,304 bytes를
  D2H copy한다. GPU sampler나 성능 최적화 결과를 주장하지 않는다.
- Strict text만 필요한 caller는 `text`를 사용한다. Arbitrary raw tokenizer bytes를 완전히
  보존해야 하는 caller는 terminal `pending_bytes`도 함께 소비해야 한다.
- Debug test binary, 고정하지 않은 GPU clock, warmup/measurement protocol 부재 때문에 수집된
  timing은 boundary 존재를 검증하는 기능 증거다. TTFT/ITL/throughput 성능 수치나 전후 개선을
  보고하지 않는다.
- Runtime allocation accounting은 runtime 소유 CUDA buffer 기준이며 process peak VRAM이 아니다.
- sm89와 기록된 driver/toolkit/model revision에서만 GPU generation을 검증했다.
