# Owned BF16 eager profile

`--numerics-profile owned-hf-eager-bf16-v1` explicitly selects caller-owned CUDA
BF16 QK/probability/AV, FP32 row-wise softmax, actual-request GEMMs, and the
model's matching normalization path. It has no Torch runtime dependency.

Build with `cargo build --locked --release -p riley-server --features server,owned-hf-eager`.
The profile supports compute capability 8.0 and 8.9 and bounded D64/D128 geometry.
Model/device qualification is recorded separately from build support.
The current geometry families are SmolLM2-135M, Qwen2.5-3B, Qwen3-1.7B, and Qwen3-4B;
other geometries fail before GPU weight upload. Qualification binds exact pinned
checkpoint revisions, not every model that shares a geometry.

The profile defaults to synchronous per-operation execution and power-of-two
capacity buckets. Each actual request retains its full prefill GEMM row count.
The maximum batch/prefill budget is 8192, and the serving sequence bound is at
most 9216 and the checkpoint context. A checkpoint with an 8192-token context
cannot serve an 8192-token prompt plus 512 new tokens.

For example, a qualified 8k/512 run uses `--batch-token-budget 8192
--prefill-chunk-tokens 8192 --max-sequence-tokens 8704 --max-output-tokens 512`.
Weights, KV blocks, workspace, packing, loading copies and external GPU use must
fit the independently fixed server/GPU budget before startup.

## HTTP admission contract

The existing `/v1/completions` JSON and SSE response formats remain unchanged.
For this explicit profile, prompt tokens must fit both `--batch-token-budget`
and `--prefill-chunk-tokens`. A larger prompt receives the existing HTTP 400
invalid-request response before scheduler submission or KV allocation. Splitting
that prompt would change GEMM geometry; the server rejects it instead.
Cached single-token decode remains within the configured total context.
The default `existing` profile retains its original prompt/chunk behavior.

Full-step Graph capture, packed metadata and unqualified component overrides
are rejected for this profile; there is no fallback to another numerical backend.
The explicit `--owned-decode-graph segments-v1` option reuses fixed decode
segments around eager Attention with iteration-batch completion.
`--owned-decode-batch grouped-v1` groups unchanged M=1 cuBLAS calls in native
submission. These opt-ins have separate model/device qualification; see
[Graph and batch validation](owned-graph-batch-optimization.md).
The default remains per-operation completion. Explicit
`--execution-completion iteration-batch` retains the arithmetic and request-row
geometry while keeping resources leased through one iteration completion.
This also supports the admitted Qwen2.5 attention-bias projection path;
owned MLP biases remain rejected. See
[A100 completion measurements](inference-completion-optimization.md) for the
qualified model, numerical controls and measurement scope.
C02's older exact-change artifact schema is a separate contract and is not
selected with this profile. `RILEY_NUMERICS` records the actual prepared
attention, QKV, MLP, output, LM-head and normalization identifiers.
The four former numerical diagnostic environment selectors are rejected;
reproduction commands must use the explicit CLI profile.

Verification uses fixed request counts and independent same-device GPU
references. Passing functional controls is not a performance measurement or
proof of long-term stability. No soak tests are included.

The qualified memory layout: BF16 QK rows are read into the existing F32 query tile before those exact rows are overwritten with BF16 probabilities. The full QK/AV GEMM shapes, row order, F32 softmax, BF16 rounding and per-operation synchronization are unchanged. One full H*M*T*2 buffer is removed. Native validation permits only the exact score/probability role pair (8,11), with identical owner/range; all other overlaps remain rejected. The use ledger deduplicates the shared allocation.
The owned wide-query Qwen3-4B path keeps its separate Q/K/V weights and output bitpacking; it does not upload a second merged QKV weight tensor that its execution path never reads. The established merged path outside this explicit profile remains unchanged.
Reproduction must explicitly unset CUBLAS_WORKSPACE_CONFIG (the qualified default), and preserve the pinned CUDA toolchain and memory budget. Forced :4096:8 creates additional per-handle library workspace and is a separate unqualified configuration.

## Admission wait budget

`--admission-timeout-ms` sets the bounded scheduler queue and engine-command admission wait in milliseconds (1..3600000). The default remains30000ms. A request that expires after SSE headers still receives the existing timeout error event and `[DONE]`; capacity/overload and resource recovery contracts are unchanged. For fixed concurrent benchmark workloads that legitimately queue behind the identical finite KV budget, choose an explicit600000ms admission budget and the same600s client waiting limit on both engines. This changes waiting policy only; it does not enlarge weights, KV, workspace, active/waiting capacity, or alter numeric kernels or expected outputs.

The optional `hf-exact-probe` reference route receives the same explicit duration for its internal scheduler and engine command admission. Its existing Rust `prepare` entry point keeps the30s default; the CLI uses the checked explicit-budget constructor. This extends waiting-policy consistency only and does not extend the reference route's qualified geometry or generation limits.


## RTX4090 qualification checkpoint (2026-10-16)

The Source28 candidate was built directly on `ssh ai-assistant` with Rust1.85.0,
NVCC12.8.93, SM89, and features `server,cuda,bench,owned-hf-eager`.
The physical device was one RTX4090 (24564MiB, CC8.9), driver580.178.04.
Candidate binary SHA256:
`cf923d13c3c02a20a28cbbf7ae61c1ddf3df047301c8620e582f738bc33e97c4`.
The exact927-file candidate manifest SHA256 is
`5ff6b0fe5d992db16f0bbdc4cd8b3f15fb7fc6a6f8c48c8a773bc60138313621`.
This document update is outside compiled code; integrated-main binaries must
be rebuilt and revalidated independently.

CPU workspace:917 passed,33 ignored; native CUDA:30 tests and12 in-place
operator executions passed. Fixed functional HTTP requests:1484 prescribed
outcomes passed across configuration, GPU sampling and CPU sampling campaigns.
Both sampler paths execute model math on CUDA. Expected invalid requests,
cancellation and overload responses are included in that count.
Generation checks require exact token IDs, text, finish reason and usage.
Cancellation/disconnect/retry/overload recovery, KV return/reuse, owned process
and IPC cleanup and before/after memory are recorded separately.

Qualified pinned checkpoints:

| Checkpoint | Revision | Tested long prompt/output limit |
|---|---|---|
| SmolLM2-135M |93efa2f097d58c2a74874c7e644dbc9b0cee75a2|7680/512|
| Qwen2.5-3B-Instruct |aa8e72537993ba99e69dfaafa59ed015b17504d1|8192/512|
| Qwen3-1.7B |70d244cc86ccca08cf5af4e1e306ecf908b1ad5e|8192/512|
| Qwen3-4B |1cfa9a7208912126459214e8b04321603b3df60c|8192/512|

Qwen2.5 uses the manifest-recorded canonical generation-EOS transformation.
Chat/thinking inputs are rendered into `/v1/completions`; a native
`/v1/chat/completions` endpoint is not implemented.

The same-device comparison completed40 backend arms, five paired process runs,
28 model/workload rows and9120 measured requests, plus3800 warmup requests.
Both engines used the same GPU,22GiB owned GPU ceiling,24GiB server-process-group
RSS ceiling, CPU affinity,544 usable KV blocks and eager BF16 configuration.
Graph allocation was zero and cross-request prefix caching disabled.
The fixed budget includes weights, KV, workspace, packing and loading copies;
external GPU usage and driver-reserved memory were inventoried before admission.

The profile is adopted for its strict generation agreement. It is not adopted
as a general performance improvement: vLLM had higher throughput in25 of28
rows. Riley/vLLM paired throughput ratios range0.037 to1.490. Shared-host IO
pressure, individual runs and dispersion are retained without post-hoc filtering.
vLLM's reference disagreements are recorded independently from protocol success;
its attention arithmetic differs, so these are not semantic-quality judgments.

The completed campaign retains10 valid arms from a prior terminal campaign and
runs30 remaining arms without resubmitting measurements. Only pre-request
startup readiness changed from240 observations to a900s/900GET bound for both
engines. The failed zero-POST setup and exit1 remain preserved. Results describe
warm HTTP/SSE performance across these segments, not uniform cold-load latency.
TPOT is a response mean, not measured per-token ITL; client concurrency32 does
not prove32 GPU-resident sequences. NVML500ms samples can miss transient peaks.

Raw comparison: `source28-regular03-complete-evidence-01.tar.gz`,17099 files,
SHA256 `a42f1361d3d754581c9b36d8c89be0f1b739938ad7707edbc5d257c6fe5715ff`.
Detailed results, raw manifests, inventory, build commands, mismatch diagnoses,
source manifests and reproduction scripts are preserved under
`artifacts/rtx4090-validation-20261015-v24/` and the matching private remote run
root `/home/psyche/riley-validation-rtx4090-20261014-v13/`.

This qualification does not establish A100 model correctness or performance,
multi-GPU behavior, other checkpoints, graph execution, general chunked long
prefill, or long-term stability. No soak or time-based sustained load was run.
A100 has build support, but this candidate has no complete A100 qualification.
