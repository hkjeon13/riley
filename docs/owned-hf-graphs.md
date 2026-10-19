# Owned HF graph experiments

The graph work is isolated at `codex/hf-owned-graphs`, based on the verified
v15 source snapshot (`bb85494c14b792d35543af1f5a766267a349641e95b8eb9a0dd79af2f391ad22`).
Concurrent work in the primary checkout is preserved. Following user-confirmed
parallel Riley work, new GPU experiments use **aws-a100 GPU 2 only**.
GPU 3 performance cells with another process are excluded from acceptance.

## Evidence

Both tests use CUDA 12.4 AOT compilation for SM80 and process-local cuBLAS
12.9.1.4 / CUDA runtime 12.9.79. No shared library installation changes.

`owned_hf_graph_feasibility_gpu` repeats one frozen HF projection 180 times.
Six alternating-order samples preserve the exact BF16 output. Median elapsed
batch time is 1.173 ms eager versus 0.585 ms graph, approximately 2.00x.

`owned_hf_mlp_graph_gpu` composes separate request-row gate/up GEMMs,
BF16-rounded fused SiLU/multiply, and separate request-row down GEMMs. The
independent reference uses actual checkpoint weights and post-norm activations,
with a separate HF M1 projection for each request. Each sample executes thirty
MLP segments; six alternating-order samples are retained per request count.

| Requests | Eager median ms | Graph median ms | Segment speedup | BF16 HF output |
|---|---:|---:|---:|---|
| 1 | 0.703 | 0.474 | 1.48x | Exact |
| 2 | 1.320 | 0.849 | 1.55x | Exact |
| 8 | 4.819 | 3.055 | 1.58x | Exact |
| 32 | 18.544 | 11.909 | 1.56x | Exact |

These are fixed-address segment diagnostics. The test holds operands and plans
alive until graph destruction, verifies every sample against the independent
reference, and checks allocation reuse and zero retained allocations at exit.
The tests do not prove production cache ownership, dynamic request transitions,
full-model latency/throughput, or vLLM parity. Repeated segments reuse the same
weights and have different cache behavior from a full model.

## Managed v22 integration

`CudaOwnedHfMlpGraph` now owns a bounded graph cache and two private direct-cuBLAS
plans. Graph keys retain opaque allocation and stream identity tags, actual M1
request count, span offsets and extents. Replacing an allocation cannot reuse a
stale entry even if CUDA reuses its address. Every replay re-registers all eight
operands and both plans in the command-batch resource ledger. Cache exhaustion
runs the identical eager MLP; changing streams retires old graphs before cuBLAS
workspace rebinding. Execution failures poison the owner; graphs close before
plans and enclosing model resources.

The explicit server option is:

```sh
--numerics-profile owned-hf-eager-bf16-v1 \
--execution-completion iteration-batch \
--mlp-activation fused-silu-multiply-v1 \
--owned-mlp-graph pure-decode-v1
```

Attention, norm and residual operations remain eager. Only independent decode
M1 rows, up to 32, use the cache; prefill and mixed work preserve the existing
path. The cache has at most 1024 entries per model owner. The option defaults
off. Current isolated source still rejects biased owned iteration completion;
the concurrent primary-checkout bias work is deliberately not imported here.

The initial server attempt exposed an integration error: the caller's 512-row
parent spans exceeded the private 32-row plan's exact native descriptors. The
v22 implementation validates parent ranges and passes bounded child spans.
A GPU test with 64-row parents, 32-row plans and 1/2/8/32 active rows passes.
It also verifies hit/miss behavior, bounded fallback, retained output leases,
allocation replacement, stream retirement, and zero explicit retained buffers.

The immutable `riley-owned-mlp-graph-v22` binary has SHA-256
`45719ea59d372889554108c7a46ff48eef69907f5c46875826a8e44ad947940d`.
With the qualified CUDA 12.9 library paths, SmolLM2-135M matches HF tokens/text
on 15/15, 45/45 and 45/45 responses at C1/C8/C32, at most 128 generated tokens.
Qwen3-1.7B and full-attention Mistral-7B-Instruct-v0.3 also match HF tokens/text on 15/15, 45/45 and 45/45 responses
at C1/C8/C32 with the same candidate and runtime paths.
Ten full-vocabulary teacher-forced BF16 logit rows are byte-identical with and
without managed MLP graphs. The model executor reports **30 captures, 2760
hits and zero cache-capacity fallbacks**, proving actual model graph replay.

CPU-only checks pass. CPU runtime/server tests pass 276/72/32 with zero failures;
CUDA-feature regressions pass 77/293/72/33 with zero failures (40 ignored).
The dedicated GPU2 modified-vLLM screen completed without errors or foreign GPU
processes. Native C1 measured 303.79/307.02 versus 433.33/425.15 output tokens/s;
C8 measured 436.36/425.96 versus 1407.18/1425.21. Full-model vLLM parity is not
achieved. GPU3 v15 screen cells with foreign processes remain excluded.

A separate causal native comparison uses the **same immutable v22 binary**,
changing only the MLP graph option, for five fresh-process alternating-order
pairs at C1 and C8. All paired output hashes match, all errors are zero and
all foreign GPU process maps are empty. Median paired throughput improves
6.41 percent at C1 and 9.72 percent at C8. TPOT p50 falls 7.01/9.68 percent;
TTFT p50 changes +0.45/-1.83 percent. This proves an improvement for these
fixed-workload native cells, not a no-regression result for every latency
quantile or vLLM parity. Natural EOS and C16/C32 performance still need qualification. Other models
have correctness evidence above, but no accepted performance matrix.

A follow-up pointer-array `cublasGemmBatchedEx` diagnostic tested 66 actual
projection/batch cases. 38 differ from sequential HF BF16 results (4,352,256
elements total). It is not admitted to the owned HF numerical path.

## Remaining qualification

Complete model-level vLLM comparisons, add other model sizes/architectures,
qualify more request transitions and capture failures, and improve remaining
attention/projection submission costs. The graph implementation and tests are
in the isolated worktree; they have not been merged over concurrent primary
changes. Full-model vLLM parity is still unproven.

The [cuBLAS graph support documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cuda-graphs-support)
allows capture in most cases and describes scalar capture and workspace/memory
node restrictions. The experiments use host alpha/beta values captured as
constants and warm cuBLAS state before recording.
