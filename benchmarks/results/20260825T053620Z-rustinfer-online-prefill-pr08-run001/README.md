# PR 08 online prefill evidence — run001

`73cbdad9e0f6b04dd46a9e719be33ae050aa4836`의 native CUDA online-softmax
prefill backend는 이 실행의 target인 RTX 4090/sm89에서 correctness, 성능, 메모리
검사를 통과했다. 모든 CUDA·모델 실행은 `server-4096`의 network-disabled container에서
수행했으며 로컬에서는 실행하지 않았다.

## 결론

Online backend는 dense BF16 BSHD, D64 MHA/GQA causal prefill을 한 kernel launch로
처리한다. QK와 scaled score의 기존 BF16 rounding boundary는 register에서 재현하지만,
FP32 `(m,l,n)` state와 numerator는 마지막 BF16 context cast까지 유지한다. 따라서
materialized reference와 bit-exact라고 주장하지 않는다. `[S,S]` score/probability는
HBM에 쓰지 않으며 attention workspace와 layout copy는 모두 0 byte다.

| S | reference median | online median | speedup | reference score/workspace | online score/workspace |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.131209 ms | 0.050771 ms | 2.5843× | 294,912 B | 0 B |
| 1,024 | 2.804055 ms | 0.848067 ms | 3.3064× | 18,874,368 B | 0 B |
| 4,096 | 43.465677 ms | 10.008649 ms | 4.3428× | 301,989,888 B | 0 B |

SmolLM2 S=128의 prepared full-forward `execute + stream.synchronize` proxy는 reference
7.565736 ms에서 online 5.125984 ms로 줄어 1.4760× 빨랐다. 이 경계에는 이미 준비된
30-layer prefill forward가 포함되지만 tokenization, model load, decode와 sampling은
포함되지 않는다. 실제 serving TTFT로 해석하면 안 된다.

## Correctness

Pinned S=7 SmolLM2 last logits의 online 대 HF BF16 golden 결과는 cosine
`0.999985581919`, max abs `0.312500000`, mean abs `0.058640185`였다. Greedy token은
`4052`, top-10 set은 exact, 첫 causal row는 backend 간 byte-exact였다. 같은 입력을
100회 실행하면서 매번 full logits를 내려받아 비교했고 모두 byte-exact였으며 SHA-256은
`7660d1ef201e6342fe84d5deb19594690a78507c21de71f5c6d0ea9f37f6257d`다.

S=128 full-model reference/online 비교도 top-1 `6354`를 보존했고 cosine
`0.999901240213`, max abs `0.40625`, mean abs `0.111404342`였다. Direct kernel 비교는
cosine `0.999996006219`, max abs `0.001953125`였다. S1/7/8/9/31/32/33, batch 1/2,
target batch 4, MHA/GQA, S128/1K/4K, local window 0/1/3/9, 큰 finite score gap,
context mismatch와 invalid workspace를 포함한 remote GPU test 8개가 통과했다.

PR01 E0 v2에서 사전 고정한 final-logits cosine/max/mean threshold 세 개를 pinned
trace에 재사용했다. 이 실행은 PR01의 FP32 comparator, relative metrics, 31-prompt
worst-case corpus를 다시 실행하거나 전체 gate를 재활성화하지 않았다.

## HBM과 layout

Nsight Compute 2025.1.1에서 동일 실행 중 한 call을 profile했다. 표는 각 backend를
구성하는 kernel들의 `dram__bytes_read.sum + dram__bytes_write.sum` 합이다. Profiler
replay duration은 raw latency와 비교하지 않았다.

| S | reference kernels | reference DRAM | online kernels | online DRAM |
|---:|---:|---:|---:|---:|
| 128 | 4 | 1,167,488 B | 1 | 260,096 B |
| 1,024 | 4 | 58,658,304 B | 1 | 1,980,544 B |
| 4,096 | 4 | 2,330,735,104 B | 1 | 7,878,784 B |

Online kernel은 40 registers/thread, 8,192 B static shared memory(9,216 B allocated)를
사용했다. Interface는 contiguous dense BSHD만 받으며 내부 conversion kernel이나 copy를
실행하지 않는다. Selection trace와 benchmark metadata의 `layout_copy_bytes`는 0이다.
Non-contiguous 입력은 cold validation에서 지원되지 않는 것으로 처리한다.

## 검증 환경과 명령

- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- container network disabled; Python/Python3 absent

주요 inner command는 다음과 같았다. 모든 Cargo command에는 `--locked --offline`을
사용했고 source는 read-only, target과 사전 복사한 Cargo registry만 writable하게 mount했다.

```text
cargo test --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --release -p rustinfer-cuda --features cuda --test prefill_attention_gpu -- --ignored --test-threads=1 --nocapture
cargo test --release -p rustinfer-cuda --features cuda --test attention_gpu -- --ignored --test-threads=1 --nocapture
cargo test --release -p rustinfer-runtime --features cuda --test llama_forward_gpu pinned_smollm2_online_prefill_matches_reference_without_score_storage -- --ignored --exact --test-threads=1 --nocapture
cargo test --release -p rustinfer-runtime --features cuda --test llama_forward_gpu benchmark_pinned_smollm2_reference_vs_online_prefill_execute_proxy_ttft -- --ignored --exact --test-threads=1 --nocapture
```

Compute Sanitizer memcheck는 direct online shape matrix, fully-masked/local-window와
SmolLM2 S=7 100회 경로에서 `0 errors`, `0 bytes leaked`였다. Shared K/V tile racecheck는
`0 errors`, `0 warnings`였다. Nsight Compute는 online 1개 kernel과 reference 4개
kernel을 S128/1K/4K에서 각각 capture했다.

## Artifact와 실패 이력

Version-controlled [raw-events.jsonl](raw-events.jsonl)은 54개 measured latency sample과
summary/correctness/profile event를 lossless decimal 또는 integer로 보존한다.
[metadata.json](metadata.json)은 환경, source, 요약과 제한을 담는다. 원문 stdout,
sanitizer, Nsight CSV와 source metadata는 다음 append-only 경로에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr08/73cbdad9e0f6b04dd46a9e719be33ae050aa4836
SHA256SUMS sha256=e4dd63012044b84532990bc8150e000bb61201a6deedb296a47e4a4605411212
```

기존 실패 증거를 덮어쓰지 않았다. `0ac1864…`는 CUDA-feature strict Clippy가 host-only
helper cfg를 발견했고, `5edd00c…`는 calibration 근거 없는 pairwise mean-abs 0.25 gate가
실패했다. `771a75a…`는 사전 고정 3-metric/golden 직접 비교로 교체한 뒤 top-10 cutoff의
한 token swap을 검출했다. Register-only 두 score rounding boundary를 복원한
`6218614…`가 이를 해소했고, every-iteration determinism과 parseable marker까지 포함한
최종 snapshot이 `73cbdad…`다.

## 제한

- CUDA Graph capture, split-K/partial native merge, non-contiguous views는 지원하지 않는다.
- Online은 causal-local을 지원하지만, 다른 capability 때문에 online이 거절된 causal-local
  요청은 reference가 대신할 수 없어 명시적으로 실패한다.
- GPU allocation accounting의 live bytes는 observed peak VRAM이 아니다.
- Cold load latency와 운영 fallback 비율은 측정하지 않았다. Native archive는 AOT link되며
  availability/capability fallback은 cold selector unit test로 검증했다.
- sm89와 기록된 driver/toolkit만 검증했다. 다른 architecture는 해당 AOT target으로 다시
  build하고 같은 gate를 실행해야 한다.
- NaN/±Inf 동작은 diagnostic API contract에 정의돼 있지만 model E0 evidence는 finite input을
  대상으로 한다.
