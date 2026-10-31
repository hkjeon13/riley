# The Riley history book

How Riley was built between August and October 2026: the idea we started
from, the order we built things in, and the points where it got better. It is
written for people who want to understand why the engine looks the way it
does, or who are attempting something similar.

All measurements were taken on one RTX 4090 with BF16 weights. Ratios against
vLLM are Riley divided by vLLM.

## 1. The idea

Riley began on 2026-08-24 under the working name *rustinfer*, with a short
statement of intent: a Rust-native, CUDA-first LLM inference engine, and
explicitly not "vLLM rewritten in Rust".

Four convictions shaped it.

**Rust for the host, CUDA for the arithmetic.** Rust does not make a GPU
kernel faster. What it can do is remove what surrounds the kernels: the
interpreter, the garbage collector, unpredictable allocation, and scheduling
overhead between GPU steps. The serving loop, the scheduler and the KV-cache
manager are state-heavy and concurrent, which is where ownership rules pay
off. The production server contains no Python at all; Python was used only to
produce reference outputs.

**Patterns, not models.** Transformer families differ far less than their
class names suggest. A manual review of about 17 families in Hugging Face
Transformers confirmed that a few building blocks (RMSNorm, RoPE, grouped-query
attention, SwiGLU, a KV cache, sampling) cover the decoder-only models we cared
about. Riley reads a model into a small canonical description and runs it
through shared kernels; no hot path branches on a model's name.

**Do not fight the vendor at GEMM.** Matrix multiplication goes to cuBLASLt.
Our own kernels cover what sits between the multiplications: normalisation,
position encoding, attention over a paged cache, sampling, and fusions of
those.

**Systems first, kernels second; correctness before both.** Measure, remove
host overhead, and only then fuse the kernels that measurement points to.
Every optimisation had to state what it preserves, sit behind a switch with
the original path still available, and never have its tolerance widened after
the results were in.

The first scope was deliberately narrow: one GPU, Llama- and Qwen-compatible
models, BF16, interactive workloads.

## 2. Building it in order (late August)

The engine was built as a fixed sequence of small steps, each with a completion
test, and none started before the previous one passed.

1. **Baselines before any code.** Hugging Face Transformers and vLLM 0.27.1
   were measured first, five independent runs each, with variance limits and
   hashed result files. Nothing could later be called "faster" without a
   pinned number to compare against.
2. **Foundations.** A Cargo workspace with strict crate boundaries; a narrow C
   ABI to ahead-of-time compiled CUDA; tensor and memory types whose lifetimes
   make a use-after-free unrepresentable in safe code; a strict loader for
   Hugging Face config, tokenizer and safetensors files.
3. **A slow, exact forward pass.** Logits had to match the reference before
   anything was optimised.
4. **Generation.** Online-softmax attention, a contiguous and then a paged KV
   cache (16-token blocks), sampling with a random stream owned by each
   request, and a second model family (Qwen) without model-specific branches.
5. **Serving.** A scheduler with continuous batching, then an
   OpenAI-compatible completions API with streaming and leak-free
   cancellation.

Two early results set the tone for everything after:

- The first custom kernel paid off where expected: online-softmax prefill
  attention was 2.6 to 4.3 times faster than the materialised reference and
  removed a 302 MB workspace at 4,096 tokens.
- The first large win was not a kernel at all. A decode step ran 517 kernels
  and waited for the GPU after each one. Waiting once per step instead raised
  throughput by 22% (113.5 to 139.0 tok/s) with GPU time unchanged.

On 2026-08-27 the project was renamed Riley and tagged 0.1.0-rc1, then rc2 the
same day: rc1 could hand a token to the wrong request when prompt and decode
work shared a step. That bug is why "the same prompt always gives the same
output, whatever else is in the batch" became a test in every later
measurement.

At that point Riley was correct and far behind: 140 tok/s against vLLM's 770
on a 135M-parameter model, with 7.2 ms per token against 1.1 ms.

## 3. Closing the gap on a small model (September)

Profiling showed where the time went, and almost none of it was arithmetic:
every step was padded to 512 rows for one real row, token metadata was uploaded
in seven separate copies, the full logit row was copied to the host for every
token, and the CPU scanned the vocabulary to pick it.

Fixing those four (batch shapes sized to the live rows, one packed upload,
token selection on the GPU, attention that loads each key/value page once)
took the engine's own harness from 140 to 244 tok/s.

The first measurement through the real HTTP server was sobering all the same:
62 tok/s against vLLM's 894. The fixes again followed the profile:

| Change | Effect (SmolLM2-135M, one client, over HTTP) |
|---|---|
| Whole-prompt prefill instead of one prompt token per step | 62 to 353 tok/s; first token after 24 ms instead of 441 ms |
| Removing a 25 ms polling wait in the HTTP path | 353 to 395 tok/s |
| Packed projections, parallel attention, wider prefill | 395 to 917 tok/s |

By 2026-09-12 a single client got 919 tok/s from Riley against 886 from vLLM.

Then the same engine was given prompts of different lengths and 4 to 32
concurrent clients, and it restarted at under a quarter of vLLM's throughput.
The single-request path had been tuned around one shape. Rebuilding it for mixed
batches (several prompts in one prefill step, prompt and decode rows in the
same step, 32-row decode) brought the 32-client case from 0.42 to 0.90 of
vLLM. The round closed there, short of the goal.

Lessons from this period:

- **A microbenchmark is a hypothesis.** A faster LM-head kernel in isolation
  was slower inside the model (476 against 278 µs), because isolated runs have
  warm caches that a real step does not.
- **Faster is not enough.** A shared-row GEMM beat vLLM at 8 clients but 40 of
  480 responses changed with batch composition. It was rejected.
- **Measurement needs its own tests.** One round was thrown away because the
  client mis-parsed a valid streaming frame; another because the server had
  been started with the wrong capacity.

## 4. A larger model and a stricter goal (October)

In October the work resumed with two models and one written goal: on eight
workloads (1, 8, 16 and 32 clients; short fixed and mixed natural prompts),
match or beat vLLM on throughput, time per token and time to first token,
without giving up output consistency, stability or memory.

### SmolLM2-135M

The remaining gap was on the host side and in launch overhead: prompt
tokenisation moved to the submitting thread, ownership checks became bitmap
operations, attention launches were sized to live rows, and three small kernels
were fused into one. The result met both the minimum and the stretch target on
all eight workloads: 1.21 to 1.88 times vLLM's throughput, with 1.2 to 1.9 GiB
of GPU memory against about 8 GiB. This work lives on a separate branch,
preserved in [`archive/smollm2-batch14-source`](../archive/smollm2-batch14-source).

### Qwen2.5-3B-Instruct

A 3-billion-parameter model changes the problem. Each generated token reads
about 6 GB of weights, so a step is limited by memory bandwidth rather than by
launch overhead, and both engines pay that cost equally. What is left to win is
everything else.

Riley's first Qwen2.5-3B number came from its reference-exact route: 19 tok/s
against vLLM's 122. Moving to the regular scheduler and removing obvious waste
(a cuBLAS handle created on every call cost 33 ms of a 53 ms step) gave the
starting line: 0.18 to 0.56 of vLLM across the eight workloads.

From there the ratio moved in small, individually verified steps:

| Stage | Throughput, Riley / vLLM (range over 8 workloads) | Main changes |
|---|---|---|
| Start | 0.18 – 0.56 | Scheduler route |
| | 0.54 – 0.68 | Unbounded HTTP worker pool, 256-token steps, a faster cuBLASLt reduction order |
| | 0.69 – 0.79 | Token selection on the GPU (removing a 9.6 MB download per step), one GEMM algorithm for all row counts |
| | 0.74 – 0.84 | Attention loads each key/value page once instead of eight times |
| | 0.81 – 0.91 | One GEMM for Q, K and V; fused residual and norm; 21 batch shapes; one-kernel page merge |
| | 0.85 – 0.93 | Decode steps recorded as CUDA graphs and replayed |
| | 0.89 – 1.00 | Fused activation and RoPE kernels; per-row-count GEMM algorithms, admitted only if they reproduce the reference bytes |
| | 0.94 – 1.06 | Wider norm and argmax kernels; logits only for rows that emit a token |
| | 0.97 – 1.07 | Graphs for mixed steps; cache write fused into the projection split; tensor-core attention for long prompts |
| Final (2026-10-12) | 0.98 – 1.14 | Tensor-core attention for every page; step sizes that respect GEMM cost plateaus |

Final position: at 8 to 32 clients Riley is ahead on throughput and median time
per token; at one client the engines are within 3%; median time to first token
is 22 to 49% lower with Riley everywhere; GPU memory is about half. vLLM keeps
a better tail time to first token when 32 clients start at once. The full table
is in [performance.md](performance.md).

Things we learned on the way:

- **The cost of a step is a staircase.** A cuBLASLt GEMM on this model costs
  about the same from 1 to 128 rows, then jumps at 129 and again at 193. A
  scheduler that knows this can leave a few prompt tokens for the next step
  and come out ahead (up to 3.3%), with no change to any output.
- **CUDA graphs are worth what the gaps cost.** vLLM's kernels start 64 ns
  apart inside a graph; ours started about 600 ns apart when launched one by
  one. Replaying recorded steps closed about 0.4 ms per step.
- **Batch-invariant numerics constrain everything.** cuBLASLt may pick a
  different algorithm for a different row count, and then the same request
  gives different bits in a different batch. We fixed one reference algorithm
  and allowed a faster one for a given row count only after a start-up probe
  showed identical output bytes. A per-row-count heuristic that was 11% faster
  kept only 7 of 15 prompts consistent, and was dropped.
- **Rounding matters at near-ties.** A tensor-core attention kernel that
  rounded its softmax weights to BF16 lost agreement with the reference on
  several prompts. Carrying each weight as three exact BF16 pieces restored
  it, at the price of three multiply passes instead of one.
- **One streaming multiprocessor is slow at fetching.** An attention kernel
  that handled a whole row on one multiprocessor was fastest at 32 clients and
  2% slower at one, because a single multiprocessor pulls only about 26 KB per
  microsecond from cache. Splitting the work per page kept the gain and
  removed the loss.
- **Agreement with a reference is a noisy number.** Token-for-token agreement
  with Hugging Face moved between 9 and 13 of 15 prompts across implementations
  that are numerically equivalent, because the differing prompts sit on
  near-tied logits. We kept the measure, and stopped reading it as quality.

## 5. How we worked

The method mattered as much as any single change.

- **Write the criterion first.** Each goal named its workloads, metrics and
  thresholds before the run. Tolerances were never adjusted afterwards, and no
  request was excluded after the fact.
- **One change, one question.** A change stated its expected gain in advance,
  then went through the same sequence: native test, reference check through
  the real HTTP server, A/B measurement, five minutes of load on all eight
  workloads with cancellations. A change that cost any workload more than 0.5%
  was not adopted, whatever it gained elsewhere.
- **Compare in the same run.** Riley and vLLM were measured back to back on the
  same machine with alternating order. Numbers borrowed from an earlier run
  were labelled as such and never used for a conclusion.
- **Three kinds of correctness, kept apart.** *Reference-exact*: a dedicated
  route reproduces Hugging Face output byte for byte. *Batch-invariant*: a
  prompt gives one output whatever the concurrency. *Reference agreement*: how
  many prompts match the reference token for token on the fast path.
- **Keep the failures.** Rejected candidates, invalid runs and stopped
  attempts were stored next to the successes with the reason. Several later
  decisions were made by rereading them. Among the invalid ones: a run taken
  while another tenant loaded the machine, and two runs where a file sync kept
  old timestamps so the kernel was never rebuilt.
- **Stop when the evidence says so.** When two consecutive attempts on a
  metric moved it by less than 1%, the item was closed with its evidence
  instead of being pursued further.

## 6. What is still open

- Tail time to first token under simultaneous arrival. It needs larger steps
  for prompt bursts without slowing the requests already decoding.
- Single-client throughput on large models, where the remaining 2 to 3% is
  spread over small per-step costs.
- The small-model kernel branch has not been merged into `main`.
- BF16 only, no quantisation. The A100 campaign (October) added 24 qualified
  models and a chat-completions endpoint; tensor parallelism over two A100s
  runs Qwen2.5-32B but is still 1–5 % behind vLLM at one and eight clients.
- Quality checks (GSM8K and HF agreement, October 31) found measurement
  mistakes rather than engine errors: vLLM applied the checkpoint's sampling
  defaults, the HF eager reference disagreed with sdpa and FP32, some
  manifests lacked `generation_config.json`, and DeepSeek-R1-Distill's BOS
  was dropped by transformers 5.19. The corrected numbers are in
  [model coverage, section 6](model-coverage-a100.md#6-품질과-큰-모델-2026-10-31).
- The 30-minute soak has not been repeated on the final Qwen2.5-3B build.

The measurement logs behind this book were removed from the tree to keep the
repository small. They remain in the git history at commit `f90683ec`, under
`benchmarks/results/`.
