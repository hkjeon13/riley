# Native model coverage and A100 qualification

Source adapters, a successful build, numerical correctness and competitive
performance are separate stages. The current campaign runs only on
`aws-a100` physical GPUs 2 and 3, compiled for `sm_80` with CUDA 12.4.
GPU processes belonging to other users must not be stopped by the campaign.

| HF configuration family | Native path | Current campaign evidence |
|---|---|---|
| Llama / SmolLM2 | Existing dense decoder and ByteLevel BPE | Pinned 135M owned profile matches HF at C1/C8/C32 with CUDA 12.9 libraries; CUDA 12.4 retains one differing prompt; performance pending |
| Qwen2 / Qwen2.5 | Existing dense decoder, Q/K/V bias and Qwen tokenizer | Pinned 3B owned profile now matches HF at C1/C8/C32 through 128 tokens after bias split-K correction; performance pending |
| Qwen3 dense | Existing per-head Q/K RMSNorm and independent head dimension | Pinned 1.7B owned profile matched HF at C1/C8/C32 through 128 tokens; performance pending |
| Mistral, full attention | New cold adapter and native Metaspace/byte-fallback BPE; shared dense runtime | Evidence below; full performance acceptance pending |
| Sliding-window Mistral, Gemma, Phi, MoE and multimodal models | Not implemented by these adapters | Unsupported; do not reinterpret their metadata as Llama |

## Mistral v0.3

The pinned checkpoint is `mistralai/Mistral-7B-Instruct-v0.3` revision
`c170c708c41dac9275d15a8fff4eca08d52bab71`. Its weight shards are verified
against Hub LFS SHA-256 values and loaded through the regular checksum-bound
checkpoint loader. A native Rust tokenizer executes the validated Metaspace,
ranked BPE, byte-fallback and BOS/decoder pipeline without Python in serving.
It shares the existing ranked merge implementation with ByteLevel BPE.

`sliding_window` must be explicitly null. A missing field is rejected because
Mistral's default window changes attention semantics. Attention/MLP biases,
unknown fields, scaled RoPE and incompatible architecture metadata are rejected.
This is a full-attention adapter, not general sliding-window support.

The tokenizer has matched independently generated `tokenizers==0.23.3`
references for 1,195 encoding and 67,939 decoding cases. Cases cover every
vocabulary ID, special-token offsets, multilingual text, arbitrary fallback
bytes and invalid UTF-8 runs. Streaming prefix state belongs to each request;
skipped special tokens do not consume the initial-space rule. Raw fallback
bytes remain subject to the runtime's existing strict UTF-8 emission contract.

The explicit `owned-hf-eager-bf16-v1` profile now supports bias-free
`hidden=4096, Q=32, KV=8, head_dim=128` blocks. The request-group RMSNorm
supports width 4096 with epsilon `1e-5`, retaining the original width
2048/2560 epsilon `1e-6` contracts. All 36 PyTorch CUDA norm cases matched
BF16 output bytes, including different request group sizes.

On A100 the owned profile matched HF eager generated tokens and response text
on all 15 tested prompts at C1 and all 45 responses at each of C8 and C32,
with at most 128 output tokens per prompt, including 128-token generations.
The explicit bias-free iteration-completion candidate also matched all
15/45/45 samples. The optimized existing profile matched tokens on 13/15
at the longer bound; vLLM matched 12/15 in the earlier 32-token probe;
these are diagnostic samples, not a full correctness or accuracy qualification.
Additional quality and performance acceptance remain required. Do not describe
these results as vLLM performance parity.

## Owned completion candidate

The explicit owned HF arithmetic profile retains per-operation completion by
default. A new `--execution-completion iteration-batch` candidate uses a
separate owned cuBLAS constructor: the enclosing command batch retains plan
and device-buffer leases until confirmed completion. The original diagnostic
constructor still rejects command batches, and graph capture stays unsupported.
Biased projections are rejected during cold preparation for this candidate;
their separate epilogue path has not been qualified for batching.

A100 GPU tests passed exact integer GEMM results, grouped-row reuse, cross-stream
buffer exclusion, allocation reuse and the original probe rejection contract.
Both synchronous and batched owned Attention passed the independent uniform
causal oracle at six shapes, including M=8192/T=8704 and T=9216. The candidate
server passes its completion-configuration contracts and the pinned Mistral
128-token C1/C8/C32 tests (15/45/45 exact responses). Performance acceptance
remains in progress.

## RMSNorm warp-reduction candidate

The general BF16 wide kernel keeps its original floating-point addition tree.
Its final five shared-memory reduction levels use warp shuffles followed by
one publication barrier. This changes synchronization, not the reduction
order or BF16 rounding. GPU comparison against the independent 256-thread
path passed for widths 512, 1024, 1536, 2048, 2560, 3072 and 4096, including
4096 with epsilons 1e-6 and 1e-5, plain norm, residual output and fused norm.
All BF16 bytes matched. Complete generation equivalence and throughput
measurements are running before this candidate is accepted.

Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` also
matched the fresh HF eager 128-token reference on 15/45/45 responses at
C1/C8/C32 with the original owned per-operation profile. These results do not
qualify every Qwen3 size or establish its performance against vLLM.

## Benchmark environment diagnosis

The initial native optimized Mistral screen is not an acceptance result.
On this A100 host, vLLM 0.27.1 spends substantial time in CUDA event waits.
An isolated PyTorch BF16 4096-by-4096 multiplication probe (ten GEMMs per
sample, fifty samples) measured median 5.46 ms with nonblocking events and
100.53 ms with blocking events. These are event/matmul diagnostic timings,
not model throughput. Three alternating-order repetitions reproduced approximately 5.44–5.46 ms
versus 100.5–100.7 ms. The shared vLLM installation remains unchanged.

Setting OMP_NUM_THREADS=1 and removing CPU affinity did not improve the
initial vLLM C1 result. The supported `--no-async-scheduling` option measured
9.94 output tokens/s in its first retained run, versus roughly 19.9 with the
default scheduler. It is therefore not a demonstrated fix. Results from any
process-local workaround must be labelled as a modified diagnostic baseline,
kept separate from default vLLM results, and cannot establish parity alone.

The owned-profile Nsight attempt completed requests but its trace import
failed with `Wrong event order`; the raw trace is retained. No kernel-time
conclusion is drawn from that failed import. The native optimized-path trace imported successfully: GPU busy share was
92.8%, with most time spent in GEMM. A process-local event-wait workaround
measured vLLM at approximately 79.2–79.3 output tokens/s at C1 and 459.0–459.1
at C8, compared with the current native screen at 70.4 and 409.1–411.0.
The native path therefore has a demonstrated throughput gap against this
modified diagnostic baseline; default-baseline wins are not accepted as parity.
Merged gate/up projection is being evaluated as a related optimization.

## Numerical qualification and resolved differences

Fresh 128-token Qwen2.5-3B probes match HF tokens/text on 10/15, 30/45 and
30/45 responses at C1/C8/C32. SmolLM2-135M matches 12/15, 36/45 and 36/45.
The original v3 and new v5 binaries produce the same tokens/text on all
15 requests for each model, so these differences predate owned completion.
These are retained baseline results. The Qwen2.5 difference is resolved below;
SmolLM2 results for its norm correction are recorded below.

The source-bound `owned_teacher_logits_gpu` diagnostic replays the same forced
prefixes, verifies checkpoint/reference hashes, checks finite full-vocabulary
logits and reports cosine, absolute errors, KL divergence and choice margins.
Its 26 sampled choices reproduce the native HTTP choices exactly. Metrics are
ungraded; the diagnostic does not fit thresholds to observed errors.

Runtime mapping inspection confirms that HF loads cuBLAS 12.9.1.4 and CUDA
runtime 12.9.79 while the default native binary loads toolkit 12.4 libraries.
A process-local native run with the 12.9 libraries keeps Qwen2.5 at 10/15 and
improves SmolLM2 to 13/15. The library difference therefore does not explain
all remaining errors. NVIDIA documents that [cuBLAS bitwise reproducibility
is not guaranteed across toolkit versions](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility);
matching library versions is a diagnostic control, not a correctness proof.
No shared library or vLLM installation was modified.

A second RMSNorm candidate reconstructs the 128/64/32 reduction levels in
warp-zero registers with the same addition tree. All eight width/epsilon
cases remain byte-identical to the independent path, and all 15 long native
Mistral outputs match the previous merged-projection candidate. The completed
C1 screen measures 78.94/79.01 versus 79.12/79.20 output tokens/s for modified
vLLM. It does not establish parity.

An additional bit-preserving attention merge reduction passed its frozen GPU
fingerprint and preserved all 15 native Mistral outputs. Its completed screen
measured C1 79.21/79.18 versus 79.34/79.32 tokens/s, and C8 460.80/462.31
versus 458.87/459.32. Native C8 TPOT remained 14.80/14.78 ms versus
14.36/14.35 ms. The candidate did not demonstrate a repeatable improvement
and was reverted; its patch and immutable binary remain as rejected evidence.

### Qwen2.5 bias split-K correction

Forty independent first-layer norm, SiLU, attention and Q/K/V/O/MLP linear
cases use actual pinned checkpoint activations for prefill and cache-on decode.
The previous candidate matched 36/40 cases; only Qwen2.5 K/V projections
failed. A process-local cuBLASLt trace shows HF selecting FP32 `COMPUTE_TYPE`
partial reduction at split-K 8 (prefill) and 16 (decode), while Riley rewrote
these to no-split. The new explicit bias-only policy preserves those
heuristics. Plain GEMM and existing strict reduction policies retain their
previous rejection bounds. The selected heuristic also mirrors actual
A/B/C/D pointer alignment preferences from the pinned Torch source.

With CUDA 12.4 native libraries, all 40 component cases now match BF16 bytes.
The immutable `riley-compute-reduction-bridge-v11` binary (SHA-256
`e99c19131bb63d991304b3e01915040b2b5f693dc6d6b2b8ab0ccd6bb2100c26`)
matches HF tokens and text on 15/15, 45/45 and 45/45 responses at C1/C8/C32
with the same fifteen prompts and at most 128 generated tokens. This qualifies
that model/workload correctness cell, not arbitrary Qwen2 checkpoints or speed.
Feature regression tests pass: CUDA 77, runtime 293, server library 72 and
server binary 32; ignored GPU tests are reported separately.

### SmolLM2 A100 norm selection

Eight actual-input RoPE application cases match HF exactly. Eight additional
cases using native cold frequencies and CPU/CUDA table generation also match.
Inspection found the owned A100 SmolLM2 profile still selecting canonical
RMSNorm because the automatic HF norm selection was restricted to SM 8.9.
The explicit owned profile now selects the reviewed HF norm on its already
admitted A100 geometry and groups by actual request rows. Ordinary automatic
profile selection stays unchanged. All 44 component cases, including padded
parent descriptors and untouched inactive rows, pass byte-exact comparison.
The immutable `riley-smol-request-norm-v14` binary (SHA-256
`513dd54ecb259797d1d7b03b06cbb93cb30135b77fe1e4f11c9d8b1c2aaf14b9`)
with native CUDA 12.4 libraries improves to 14/15, 42/45 and 42/45 exact
responses; `natural-16` still diverges at generated token index 26. With
process-local cuBLAS 12.9.1.4 and runtime 12.9.79 library paths, the same binary
matches HF on 15/15, 45/45 and 45/45 responses at C1/C8/C32. Both per-operation
and explicit iteration-batch completion pass under that library control.
All fifteen prompts use the original EOS and at most 128 generated tokens.
Library paths and loaded dependency resolution are retained with each run.
This is a conditional correctness result: the default CUDA 12.4 runtime is
still not fully HF-exact for this model. Performance qualification is pending.

## Owned-path performance diagnosis and fused activation candidate

The SmolLM2 exact owned CUDA 12.9 iteration-batch screen remains slower than
the process-local modified event-spin vLLM baseline: C1 270.41/279.75 versus
431.08/446.95 output tokens/s and C8 380.89/385.64 versus 1419.40/1453.40.
Two repeats of fixed prompts are diagnostic only. GPU greedy sampling keeps
all C1/C8/C32 generation matches but yields only a small speed improvement.

An independent direct-cuBLAS strided-batched GEMM experiment compares 66 cells
across actual HF projection inputs and batch sizes 2/8/32. Thirty-eight cells
have differing BF16 outputs versus separate HF calls; the experiment is not
adopted as an exact-path optimization. See `strided-gemm-probe-v1/result.json`.

The explicit owned profile now admits the existing
`--mlp-activation fused-silu-multiply-v1` option. Its kernel preserves the BF16
SiLU intermediate rounding before multiplication. Forty-eight component
oracle cases pass, including four actual-input fused-activation comparisons.
The immutable `riley-owned-fused-silu-v15` binary (SHA-256
`d04579eb308191641e9475b4b49535e25ef6527020356e7a298dd642e79c0ddf`)
with CUDA 12.9 libraries and GPU greedy sampling preserves all SmolLM2
C1/C8/C32 HF generation matches. The initial C32 attempt refused admission
when another Riley process occupied GPU3; it was retained and a fresh
post-exit attempt passed. Feature regression tests pass 77/293/72/32 with
zero failures. The performance screen is running; the option is not defaulted
or accepted as meeting vLLM parity. Other model sizes need generation checks.

A valid Nsight trace of the v14 owned batch/GPU-greedy C1 path contains
174593 kernels totaling 604.24 ms across a 1408.79 ms kernel span. CUDA kernel
launch APIs consume 692.44 ms in the diagnostic trace. The GPU is busy for
about 43 percent of that span; repeated launch/host submission is a measured
bottleneck. This motivates graph-capable ownership and shape handling work;
simply enabling graph flags remains forbidden by the current owned contract.

## Validation commands

Run on the designated host with the CUDA 12.4 toolchain, a dedicated Cargo
target, `RILEY_CUDA_ARCHITECTURES=80` and an allowed GPU UUID. The artifact
directories below must contain the independent pinned reference files.

```sh
cargo test --locked -p riley-model
RILEY_MISTRAL_ARTIFACT_DIR=/path/to/mistral-oracle \
  cargo test --locked -p riley-model --test mistral_tokenizer_oracle \
  native_encoding_batch_and_incremental_decoding -- --ignored --nocapture
RILEY_MISTRAL_CHECKPOINT=/path/to/verified/checkpoint \
  cargo test --locked --release -p riley-model --test mistral_tokenizer_oracle \
  complete_mistral_checkpoint -- --ignored --nocapture
RILEY_WIDE_NORM_ORACLE_PATH=/path/to/norm-oracle \
  cargo test --locked --release -p riley-cuda --features cuda-cublas-gemm-probe \
  --test wide_norm_oracle_gpu -- --ignored --nocapture
cargo build --locked --release -p riley-server --features server,owned-hf-eager
```

Before accepting an optimization, pin model/tokenizer revisions, sampling and
EOS settings, source/binary hashes, hardware and baseline dependencies. Check
correctness, then run fresh-process AB/BA HTTP comparisons on the same GPU,
with warmup excluded and at least five independent runs at C1/C8/C16/C32.
Require throughput at least equal to vLLM and no TTFT/TPOT p50/p95/p99
regression per required cell. Keep diagnostic screens, fixed-output workloads
and natural-EOS results distinct. A single favorable aggregate is insufficient.


## HF graph and native profile integration (2026-10-16)

The validated v26 worktree changes have been integrated into the primary checkout, preserving the concurrent owned decode/bias work. See [native profile and measurements](hf-native-performance.md) and [owned MLP graph contracts](owned-hf-graphs.md). The unbuilt v27 wide residual-norm experiment is retained only under `artifacts/hf-worktree-integration-20261016/unfinished-worktree`; it is not enabled in the runtime. GPU validation has not been repeated for the combined checkout.
