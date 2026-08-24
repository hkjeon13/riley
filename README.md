# rustinfer

**A Rust-native, CUDA-first LLM inference engine focused on low latency, predictable memory management, and reusable transformer kernels.**

> Project status: design / early prototype

Implementation follows the numbered [deployment plan](deploy/README.md). See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR, validation, benchmark, and `unsafe`/FFI review contract.

## Production workspace

The production runtime is an explicitly bounded seven-crate Rust workspace; its
responsibilities and dependency direction are documented in
[`crates/README.md`](crates/README.md). Python reference tools and Triton
experiments are intentionally outside that workspace and are never production
Cargo features or build-script inputs.

The repository pins Rust 1.85.0. A CPU-only checkout needs no CUDA toolkit:

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo test --locked --workspace --no-default-features
cargo doc --locked --workspace --no-deps --no-default-features
```

The default member is `rustinfer-server`. A CUDA environment builds the empty
native ABI boundary and version binary with the exact root command below; PR 02
does not initialize a CUDA device or run inference.

```bash
cargo build --locked --release --features cuda,server
./target/release/rustinfer --version
```

CUDA compilation/link validation is separate from the mandatory CPU gate. See
[`ci/README.md`](ci/README.md) for the reproducible local and container commands.

## 1. Vision

`rustinfer` aims to build a lightweight, high-performance LLM inference engine with a different emphasis from general-purpose engines such as vLLM, SGLang, or TensorRT-LLM.

The core idea is:

1. **Use Rust for the runtime and serving control plane**
2. **Use CUDA / CUTLASS / cuBLASLt for GPU execution**
3. **Normalize Hugging Face Transformer model architectures into a common intermediate representation**
4. **Identify frequently recurring operators and subgraphs across models**
5. **Implement and reuse high-value kernels rather than maintaining model-specific execution code**
6. **Optimize first for low-latency interactive inference, then expand toward broader throughput-oriented workloads**

The project should not begin as “vLLM rewritten in Rust.”

Instead, the goal is an **architecture-driven native inference runtime** that reduces CPU-side overhead and minimizes the cost of supporting new transformer architectures.

---

## 2. Why Rust + CUDA?

Rust does not make CUDA kernels inherently faster than kernels launched from Python or C++.

The GPU ultimately executes CUDA/PTX/SASS regardless of which host language schedules the work.

The expected benefit of Rust is primarily on the **host/runtime side**:

- No Python GIL
- No Python garbage-collector pauses
- Predictable memory ownership
- Efficient native concurrency
- Fine-grained control of allocation and reuse
- Lower request/scheduler overhead
- Easier construction of lock-free or low-contention structures
- Memory safety compared with large C++ runtime codebases
- Strong async ecosystem for HTTP/gRPC serving
- Good fit for a state-heavy KV-cache manager

The expected GPU-side performance comes from:

- CUDA
- cuBLASLt
- CUTLASS
- Tensor Cores
- FlashAttention-compatible kernels
- Custom fused kernels
- CUDA Graphs
- Workload-specific kernel selection

The desired architecture is therefore:

```text
Client
  |
  v
Rust API Server
  |
  v
Request Manager
  |
  v
Scheduler / Continuous Batcher
  |
  v
KV Cache Manager
  |
  v
Model Runtime
  |
  v
Kernel Planner
  |
  +-------------------+
  |                   |
  v                   v
Custom CUDA       cuBLASLt / CUTLASS
  |                   |
  +---------+---------+
            |
            v
          NVIDIA GPU
```

---

## 3. Main Design Principle

### Model-centric design

A conventional model runtime can grow like this:

```text
Llama
  -> LlamaModel

Qwen
  -> QwenModel

Gemma
  -> GemmaModel

Mistral
  -> MistralModel

DeepSeek
  -> DeepSeekModel
```

This approach often causes model support code to grow continuously.

### rustinfer approach

Instead, `rustinfer` should analyze model architectures and normalize them into reusable primitives.

```text
Hugging Face Transformers
          |
          v
Architecture Analyzer
          |
          v
Canonical IR
          |
          +-------------------------+
          |            |            |
          v            v            v
       RMSNorm        RoPE        SwiGLU
          |            |            |
          +------------+------------+
                       |
                       v
                Pattern Matcher
                       |
                       v
                 Kernel Planner
                       |
                       v
                 CUDA Runtime
```

The number of model architectures may be very large, but the number of common high-value computation primitives is much smaller.

---

## 4. Architecture Census

One early research task should be an automated analysis of Hugging Face `transformers`.

Target source:

```text
transformers/models/*/modeling_*.py
```

The analyzer should identify how often common components appear.

Candidate dimensions:

- RMSNorm
- LayerNorm
- RoPE
- partial RoPE
- ALiBi
- MHA
- MQA
- GQA
- Sliding Window Attention
- SwiGLU
- GeGLU
- GELU
- SiLU
- MoE
- QK Norm
- KV-cache layout
- attention bias variants
- normalization placement
- residual topology

Example output:

```text
Architecture Census

RMSNorm              281
LayerNorm            106

RoPE                  259
Absolute Position      41
ALiBi                   17

SwiGLU                221
GELU                   97

GQA                    181
MHA                    143
MQA                     23
```

These numbers above are only illustrative. The actual project should generate them directly from the target Transformers version.

---

## 5. Primitive-Centric Kernel Library

The first reusable layer should consist of common kernels.

Candidate primitives:

```text
RMSNorm
LayerNorm
RoPE
SiLU
GELU
SwiGLU
GeGLU
KV-cache write
KV-cache gather
Paged decode attention
Prefill attention
Sampling
Top-k
Top-p
Repetition penalty
MoE routing
Quantize / dequantize
```

However, primitive kernels alone are unlikely to create a large performance advantage.

The more valuable optimization target is frequently recurring **subgraphs**.

---

## 6. Pattern-Centric Fusion

Instead of only asking:

> Which operation is common?

`rustinfer` should also ask:

> Which sequence of operations is common enough that fusing it has ecosystem-wide value?

Examples:

### Attention preparation

```text
RMSNorm
   |
   v
QKV Projection
   |
   v
RoPE
   |
   v
KV Cache Write
```

Possible optimization:

```text
Fused Attention Prepare
```

### MLP path

```text
Gate Projection
      |
      v
     SiLU
      |
      x
      |
Up Projection
      |
      v
Down Projection
```

Possible optimization:

```text
Fused SwiGLU Path
```

### Residual / normalization

```text
Residual Add
     |
     v
RMSNorm
```

Possible optimization:

```text
Fused Residual + RMSNorm
```

Potential high-value fused patterns:

- RMSNorm + QKV preparation
- RoPE + KV-cache write
- residual + RMSNorm
- activation + gate multiplication
- logits processing + sampling
- quantization + cache write
- MoE routing + dispatch metadata preparation

---

## 7. Canonical Intermediate Representation

A model should first be converted into a compact canonical representation.

Illustrative Rust structure:

```rust
struct Transformer {
    norm: NormKind,
    attention: AttentionConfig,
    mlp: MlpConfig,
}

enum NormKind {
    RMSNorm,
    LayerNorm,
}

struct AttentionConfig {
    kind: AttentionKind,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope: Option<RopeConfig>,
}

enum AttentionKind {
    MHA,
    MQA,
    GQA,
    SlidingWindow,
}

struct RopeConfig {
    theta: f32,
}

enum MlpConfig {
    SwiGLU {
        hidden_size: usize,
        intermediate_size: usize,
    },
    GELU {
        hidden_size: usize,
        intermediate_size: usize,
    },
}
```

The purpose of the IR is not to mirror every Hugging Face class.

Its purpose is to reduce many model implementations into a small number of execution patterns.

---

## 8. Kernel Planner

The fastest implementation of an operator depends on more than the operator name.

Relevant dimensions include:

- GPU architecture
- dtype
- batch size
- hidden size
- head dimension
- number of heads
- sequence length
- prefill vs decode
- cache layout
- quantization format
- tensor-parallel configuration

Illustrative key:

```rust
struct KernelKey {
    op: Operation,
    gpu_arch: GpuArch,
    dtype: DType,
    hidden_size: usize,
    batch_class: BatchClass,
}
```

Conceptually:

```text
Model
  |
  v
Shape / DType / GPU / Workload
  |
  v
Kernel Registry
  |
  v
Best Known Kernel
```

This creates an opportunity for an auto-tuned kernel database.

---

## 9. Auto-Tuning

The engine should eventually benchmark multiple implementations and persist the best result for each relevant hardware/workload class.

Example:

```text
RMSNorm + QKV

Implementation A
separate kernels          38 us

Implementation B
fused implementation      29 us

Implementation C
custom CUDA               21 us

Implementation D
CUTLASS-based fusion      19 us

Result:

SM90 + BF16 + hidden=4096
-> implementation D
```

Over time, this becomes a valuable asset:

> a hardware-aware LLM inference kernel database.

---

## 10. Runtime Architecture

Proposed top-level modules:

```text
rustinfer/
|
+-- api/
|   +-- OpenAI-compatible HTTP API
|   +-- streaming
|
+-- tokenizer/
|
+-- scheduler/
|   +-- request queue
|   +-- continuous batching
|   +-- latency-aware policy
|   +-- cache-aware policy
|
+-- kv/
|   +-- block allocator
|   +-- prefix cache
|   +-- eviction
|   +-- prefetch
|   +-- offload
|
+-- model/
|   +-- HF config parser
|   +-- safetensors loader
|   +-- canonical IR
|   +-- architecture registry
|
+-- planner/
|   +-- graph normalization
|   +-- pattern matching
|   +-- kernel selection
|   +-- auto tuning
|
+-- runtime/
|   +-- CUDA streams
|   +-- CUDA graphs
|   +-- memory allocator
|   +-- execution
|
+-- kernels/
|   +-- norm
|   +-- rope
|   +-- attention
|   +-- kv
|   +-- mlp
|   +-- sampling
|   +-- moe
|   +-- quantization
|
+-- benchmark/
|
+-- tools/
    +-- transformers architecture census
```

---

## 11. KV Cache Manager

The KV-cache manager is a particularly good fit for Rust because it maintains complex mutable state under concurrency.

Conceptual state:

```text
Request A
  +-- block 92
  +-- block 17
  +-- block 81

Request B
  +-- block 45
  +-- block 33
  +-- block 78
```

Operations:

- allocate
- free
- reference count
- prefix share
- reuse
- evict
- offload
- prefetch
- pin/unpin
- move between memory tiers

Illustrative type:

```rust
struct KvBlock {
    id: BlockId,
    ref_count: std::sync::atomic::AtomicU32,
    state: BlockState,
}
```

This should be designed around predictable allocation and minimal synchronization overhead.

---

## 12. Scheduler

The scheduler is another area where a custom engine may outperform a general-purpose runtime for a constrained workload.

Baseline policies:

- FCFS
- priority

Possible `rustinfer` policies:

- cache-aware
- latency-aware
- TTFT-aware
- SLA-aware
- KV-pressure-aware
- estimated-compute-aware
- starvation-safe

Conceptual scoring:

```text
score =
    waiting_time
  + request_priority
  + cache_affinity
  + TTFT_urgency
  - estimated_compute_cost
  - KV_pressure_cost
```

The exact formula should be treated as an experimental policy, not a fixed design.

---

## 13. CPU/GPU Overlap

A major goal is to avoid GPU idle gaps caused by host-side scheduling or memory movement.

Bad pipeline:

```text
CPU batch preparation
████

GPU execution
    ███████████

CPU preparation
               ████

GPU execution
                   ███████████
```

Desired pipeline:

```text
GPU batch N
████████████

CPU prepares N+1
     █████

GPU batch N+1
          ████████████

CPU prepares N+2
               █████
```

The runtime should therefore support:

- asynchronous batch preparation
- asynchronous KV prefetch
- CUDA streams
- pinned host memory where useful
- CUDA Graphs for stable decode paths
- minimized synchronization
- preallocated metadata buffers

---

## 14. CUDA Graph Strategy

Small-batch decode can be sensitive to kernel-launch overhead.

CUDA Graphs may reduce launch overhead, but they can also increase memory reservation and create shape-management complexity.

Therefore `rustinfer` should avoid treating CUDA Graph usage as a binary global option.

Potential future policy:

```text
low concurrency
-> latency priority
-> aggressive graph reuse

high concurrency / KV pressure
-> capacity priority
-> selective graph use
```

This can eventually become part of the Kernel / Execution Planner.

---

## 15. GEMM Strategy

Do **not** begin by attempting to beat NVIDIA's matrix multiplication stack.

For mainstream GEMM workloads, prefer:

- cuBLAS
- cuBLASLt
- CUTLASS
- Tensor Cores

`rustinfer` should focus custom engineering effort on areas where integration and fusion matter more:

- RMSNorm
- RoPE
- KV-cache operations
- fused attention preparation
- sampling
- MoE routing
- quantization
- layout transforms
- cache management
- scheduler/runtime overhead

---

## 16. Dependency Philosophy

Long-term target:

```text
Rust Runtime
   |
   +-- CUDA Driver / Runtime API
   +-- cuBLASLt
   +-- CUTLASS
   +-- custom CUDA
   +-- safetensors
   +-- tokenizer
```

PyTorch should ideally **not be required in the production runtime**.

Python may still be useful for:

- analysis tools
- kernel prototyping
- benchmark comparison
- developer utilities
- optional client SDK

The inference server itself should remain Rust-native where practical.

---

## 17. Initial Scope

The first version should be intentionally narrow.

Recommended target:

```text
Hardware:
NVIDIA single GPU

GPU family:
choose one first
(e.g. Ampere, Ada, Hopper, or Blackwell)

Models:
Llama-compatible
Qwen-compatible

Precision:
BF16 first

Workload:
interactive chat

Concurrency:
1-16 initially

Primary metrics:
TTFT
TPOT / ITL
E2E latency
GPU utilization
CPU overhead
memory usage
throughput
```

Avoid trying to support every Hugging Face model immediately.

---

## 18. MVP Roadmap

### Phase 0 — Architecture Census

- [ ] Select target `transformers` version
- [ ] Scan `modeling_*.py`
- [ ] Build primitive frequency report
- [ ] Build subgraph/pattern frequency report
- [ ] Select first model families

### Phase 1 — Minimal Rust Runtime

- [ ] Rust workspace
- [ ] CUDA FFI / runtime wrapper
- [ ] safetensors loader
- [ ] tokenizer integration
- [ ] GPU memory abstraction
- [ ] basic model config parser

### Phase 2 — Single Model Forward

- [ ] Llama/Qwen-compatible decoder-only model
- [ ] BF16
- [ ] single GPU
- [ ] cuBLASLt GEMM
- [ ] baseline RMSNorm
- [ ] baseline RoPE
- [ ] attention backend
- [ ] greedy sampling

### Phase 3 — Autoregressive Runtime

- [ ] KV cache
- [ ] decode loop
- [ ] streaming output
- [ ] CUDA stream management
- [ ] preallocated buffers
- [ ] benchmark harness

### Phase 4 — Serving Engine

- [ ] OpenAI-compatible API
- [ ] request manager
- [ ] continuous batching
- [ ] scheduler
- [ ] cancellation
- [ ] backpressure
- [ ] metrics

### Phase 5 — Kernel Optimization

- [ ] fused residual + RMSNorm
- [ ] RoPE + KV write
- [ ] optimized decode attention
- [ ] fused logits processing / sampling
- [ ] CUDA Graph decode path
- [ ] kernel registry

### Phase 6 — Architecture Normalization

- [ ] Canonical IR
- [ ] HF architecture mapper
- [ ] pattern matcher
- [ ] kernel planner
- [ ] reusable model composition

### Phase 7 — Auto-Tuning

- [ ] kernel benchmark database
- [ ] hardware signature
- [ ] shape-aware selection
- [ ] startup or offline tuning
- [ ] persisted best-kernel map

### Phase 8 — Advanced Runtime

- [ ] prefix caching
- [ ] cache-aware scheduling
- [ ] predictive KV prefetch
- [ ] KV offload
- [ ] FP8 / INT8
- [ ] tensor parallelism
- [ ] multi-GPU

---

## 19. First Benchmark Target

The first milestone should not claim:

> rustinfer is faster than vLLM.

Instead:

> For a narrowly defined workload, rustinfer achieves lower host-side overhead and competitive or better small-batch latency.

Suggested benchmark matrix:

```text
Engine:
- rustinfer
- vLLM
- SGLang
- TensorRT-LLM where practical

Concurrency:
1
2
4
8
16

Prompt length:
128
1K
4K
8K

Output length:
32
128
512

Metrics:
TTFT
median TPOT
p95 TPOT
p99 latency
tokens/s
GPU utilization
CPU utilization
VRAM usage
scheduler time
kernel launch count
```

All comparisons should use:

- same model
- same dtype
- same GPU
- same prompt/output lengths
- same sampling configuration
- warm and cold measurements reported separately

---

## 20. Potential Differentiators

`rustinfer` should aim to differentiate through:

### 1. Rust-native runtime

Lower and more predictable host-side overhead.

### 2. Architecture normalization

New model support through canonical building blocks instead of large duplicated model implementations.

### 3. Pattern-driven kernels

Optimize recurring transformer subgraphs rather than isolated operators only.

### 4. Hardware-aware kernel planning

Choose implementations based on shape, dtype, workload, and GPU architecture.

### 5. Low-latency fast path

Specialize aggressively for small-batch interactive inference.

### 6. Cache-aware scheduling

Treat KV-cache state as a first-class scheduling signal.

### 7. Predictive memory movement

Overlap KV prefetch and CPU preparation with GPU execution.

---

## 21. Non-Goals for the First Version

The MVP should **not** attempt to:

- support every Transformers architecture
- beat cuBLASLt at general GEMM
- support every NVIDIA GPU architecture
- implement distributed inference immediately
- support every quantization format
- replicate all vLLM APIs
- optimize training
- replace DeepSpeed or Megatron-LM
- provide multimodal support immediately

---

## 22. Risks

### Kernel maintenance cost

Custom CUDA kernels create substantial testing and hardware compatibility burden.

### Architecture diversity

Transformer models often contain subtle differences that are easy to incorrectly normalize.

### Correctness before speed

Small numerical differences can cascade during autoregressive decoding.

### Benchmark bias

An engine optimized for batch=1 can look excellent while performing poorly under high concurrency.

### CUDA / driver compatibility

Native CUDA integration requires explicit compatibility testing.

### Scope explosion

Serving, model loading, scheduling, kernels, distributed execution, and API compatibility are each large projects.

The project should therefore preserve a narrow initial target.

---

## 23. Success Criteria

An early version can be considered successful if it demonstrates all of the following:

- Rust-native request-to-GPU execution path
- no PyTorch dependency in the serving runtime
- working Llama/Qwen-compatible BF16 inference
- streaming generation
- reusable KV cache
- continuous batching
- reproducible benchmark suite
- at least one custom/fused CUDA kernel
- at least one measurable advantage over a general-purpose baseline under a clearly defined workload

The advantage does not have to be total throughput.

Examples of valid wins:

- lower batch-1 TTFT
- lower host CPU utilization
- lower scheduler overhead
- lower p99 ITL
- better cache-heavy workload latency
- lower memory fragmentation
- faster model startup

---

## 24. Guiding Principle

The project should optimize **systems architecture first, kernels second**.

A useful order is:

```text
Measure
  |
  v
Find host/runtime bottleneck
  |
  v
Reduce scheduling / allocation overhead
  |
  v
Improve CPU-GPU overlap
  |
  v
Optimize KV-cache behavior
  |
  v
Profile kernels
  |
  v
Fuse only high-value recurring patterns
```

This avoids spending months writing CUDA kernels for operations that are not actually limiting end-to-end latency.

---

## 25. Working Definition

> **rustinfer is a Rust-native, CUDA-first inference engine that converts transformer architectures into reusable execution primitives and fused patterns, with an initial focus on predictable memory behavior and ultra-low-latency LLM serving.**
