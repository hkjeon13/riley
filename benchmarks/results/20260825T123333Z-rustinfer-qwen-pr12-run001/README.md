# PR 12 Qwen2.5 compatibility evidence — run001

`842b1cdc68b46bc32ed2e72041c91e62c6e25fc7`에서 dense Qwen2/Qwen2.5
adapter, tokenizer/chat-template profile, projection bias, tied embedding과 공유 Llama execution
modules를 RTX 4090/sm89에서 검증했다. 모든 CUDA compile, checkpoint load와 model inference는
`server-4096`의 network-disabled container에서 수행했다. 로컬에서는 source 편집과 model-free
검사만 수행했다.

## 결론

PR 12 compatibility gate를 통과했다. Pinned `Qwen/Qwen2.5-0.5B-Instruct`의 English,
Korean, Rust code 세 case에서 8-token direct decode와 generation token sequence가 BF16 eager
golden과 exact였고 cache-on/cache-off도 exact였다. 첫 prefill의 raw top-1 token과 top-10
token set이 golden과 exact였으며, 고정 probe/top-10 logit 값은 기존 E0 maximum-absolute-error
bound 안에 있었다. Addressable tokenizer domain 밖의 padded 271개 model-vocabulary token은
greedy 선택에서 mask되었다.

같은 process에서 SmolLM2를 닫은 뒤 Qwen2.5를 load/execute/close했고 각 close 뒤 runtime CUDA
allocation accounting은 0이었다. 기존 SmolLM2 31-case generation, reference forward, online
prefill golden과 100-run byte determinism gate도 다시 통과했다.

## Oracle과 execution 의미

Exact compatibility oracle은 Hugging Face Transformers BF16 eager fixture다. Runtime gate는
기존의 일반 `PreparedLlamaForwardConfig::with_reference_attention()`와
`PreparedLlamaDecodeConfig::with_reference_decode_attention()`을 명시적으로 선택해 prefill과
decode 모두 materialized reference semantics로 실행한다. 이는 Qwen 이름을 검사하는 hot-path
분기가 아니라 caller가 선택하는 일반 semantic policy다.

Optimized online attention은 계속 E0 tolerance와 performance 검증을 위한 path이며 이 exact eager
oracle로 재분류하지 않았다. Model family 차이는 config/tokenizer/weight adapter와 canonical IR의
일반 parameter로만 표현된다. 실행 hot path에는 model-name 기반 Qwen 분기가 없다.

## 호환성 계약

- Model: dense `Qwen2ForCausalLM`, 24 layers, hidden 896, intermediate 4,864
- Attention: 14 query heads, 2 KV heads, head dimension 64, Q/K/V projection bias
- Vocabulary: model rows 151,936; addressable tokenizer tokens 151,665; padded tail 271
- Context: runtime maximum sequence length 32,768
- Weights: 291 logical bindings; LM head aliases the token embedding
- Tokenizer: pinned Qwen2 NFC + Split/ByteLevel BPE profile and exact pinned chat template
- Unsupported sliding-window, scaled/multi-axis RoPE, MoE, VL, Qwen3, quantized and remote-code
  variants fail closed
- Row bias: BF16 in-place primitive with bounded shapes, context ownership and overflow checks

## 원격 GPU 검증

| Gate | 결과 |
|---|---|
| CUDA all-features strict Clippy | pass |
| Workspace all-targets/all-features | 184 passed, 0 failed, 55 ignored |
| Workspace all-features doctest | 13 passed, 0 failed |
| Qwen metadata/checkpoint/sequential load | 3 passed |
| BF16 row-bias GPU | 2 passed |
| Deterministic GEMM matrix | 20 shapes passed |
| Qwen GEMM at 16 MiB cap | down projection M=1/30/40/46, N=896, K=4,864 passed |
| Qwen exact compatibility | English/Korean/code, 3 × 8 tokens direct + generation exact |
| Raw-logit rank | top-1 exact, top-10 token set exact |
| Cross-model CUDA owner | SmolLM2 → Qwen2.5; allocation accounting 0 after each close |
| SmolLM2 generation regression | 31 cases exact |
| SmolLM2 forward regression | reference golden passed |
| SmolLM2 online-prefill regression | golden and 100-run byte determinism passed |
| Row-bias compute-sanitizer memcheck | 0 errors, 0 leaked bytes |
| Row-bias compute-sanitizer racecheck | 0 hazards, 0 errors, 0 warnings |

The GEMM gate includes odd shapes, existing SmolLM2 projections and the Qwen down-projection decode
and three golden-prompt prefill shapes. The corrected deterministic cuBLASLt selection admits
normalized non-split-K candidates within the unchanged 16 MiB workspace cap.

## 환경과 고정 artifact

- Host: `server-4096`, Linux 6.8.0-138-generic
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- Driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- Image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, batch 1
- Golden SHA-256:
  `42cc7f3fd04098bc4d70836ee9d18dbf919f158a010da3da6fdaa3d9deeceab7`
- Container network disabled; source/checkpoints/Cargo registry read-only; target/evidence only
  writable

Checkpoint payloads were pinned by exact byte length and SHA-256:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 659 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| `tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | 7,305 | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `model.safetensors` | 988,097,824 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |

주요 Cargo command는 모두 `--locked --offline`로 실행했다.

```text
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test --workspace --all-targets --all-features --locked --offline
cargo test --doc --workspace --all-features --locked --offline
cargo test -p rustinfer-model --test qwen_loading --locked --offline -- --ignored
cargo test -p rustinfer-cuda --test row_bias_gpu --features cuda --locked --offline -- --ignored
cargo test -p rustinfer-cuda --test gemm_gpu --features cuda --locked --offline
cargo test -p rustinfer-runtime --test qwen_compat_gpu --features cuda --locked --offline <exact-test> -- --ignored --exact --nocapture
compute-sanitizer --tool memcheck <row-bias-test-binary> --ignored
compute-sanitizer --tool racecheck <row-bias-test-binary> --ignored
```

## Artifact와 provenance

이 디렉터리의 `raw-events.jsonl`은 authoritative Qwen case/summary, cross-model owner와
SmolLM2 regression marker를 보존한다. `metadata.json`은 source, environment, model,
oracle, validation과 limitation 계약을 기계가 읽을 수 있게 기록한다. 전체 stdout, exact
source tar, GPU/driver, image, host/toolchain, checkpoint hashes와 normalized command record는
다음 append-only 원격 root에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr12/842b1cd/full
checksum payload lines: 24
regular files including SHA256SUMS: 25
regular-file bytes: 29,707,038
artifact-root apparent bytes: 29,711,134
SHA256SUMS sha256: 863090e529e8f4a860ce1fad4008f56915f2097aef132044aa9305b284717b63
source tar sha256: cc3cd6afeb59d3303cb682b8ab1435a36675d6bfd2b5bcc7eff9569e89076a70
```

앞선 `cb70581`, `a6004d6`, `243b3ff` 원격 roots에는 cuBLASLt plan selection과 attention
semantic 차이를 분리해 낸 실패/진단 기록이 append-only로 남아 있다. 통과 evidence로
덮어쓰지 않았으며 이 디렉터리는 `842b1cd` final root만 authoritative로 취급한다.

## 제한

- 이 artifact는 reference/functional validation이다. 고정 clock, warmup/measurement protocol이
  없으므로 latency, throughput 또는 성능 개선을 주장하지 않는다.
- Exact Qwen oracle은 명시적인 general reference prefill/decode policy다. Optimized attention은
  기존 E0 tolerance/performance path로 별도 검증한다.
- 단일 request, batch 1이다. Scheduler와 continuous batching은 PR 13 범위다.
- Dense pinned Qwen2.5 revision만 검증했다. Qwen-VL, MoE, Qwen3, quantization, remote code와
  sliding/local attention은 범위 밖이다.
- Runtime CUDA allocation accounting은 runtime 소유 buffer 기준이며 process peak VRAM이 아니다.
- GPU compatibility는 기록된 sm89 driver/toolkit/checkpoint 조합에서만 검증했다.
