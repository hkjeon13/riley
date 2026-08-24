# vLLM baseline adapter

`rustinfer-vllm-benchmark` runs exactly one matrix cell and independent run per
fresh process. Warmups and measured warm trials reuse that process's model;
cold runs contain one measurement and no within-run model reuse. It is
cache-only by default; `--allow-download` permits fetching only the pinned
model revision. Before vLLM/tokenizer construction, the adapter verifies exact
SHA-256 values for `model.safetensors`, `config.json`, `merges.txt`,
`special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, and
`vocab.json`, plus the canonical aggregate tokenizer hash.

Gate A applies the 5% throughput CV threshold only to warm cells. A cold cell
still reports throughput CV as a diagnostic, while its gate is based on model
load CV, peak VRAM, failures, and token identity; one first-request observation
per fresh process is not treated as a stable within-run throughput p50.

The adapter sets `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. This
selects vLLM's non-JIT PyTorch sampling path so the pinned lane does not depend
on a host CUDA toolkit or `nvcc`. A conflicting inherited value is rejected.
Prefix caching is disabled so repeated warm trials do not reuse prompt KV.
Request statistics are enabled as an independent timing sanity check; missing
statistics invalidate a trial.
The canonical lane environment also pins `VLLM_NO_USAGE_STATS=1`,
`VLLM_DO_NOT_TRACK=1`, `DO_NOT_TRACK=1`, and
`HF_HUB_DISABLE_TELEMETRY=1`; conflicting ambient runtime overrides are rejected
by the outer runner.

Per-request timing uses the synchronous public `LLMEngine` exposed by
`LLM.llm_engine` in pinned vLLM 0.27.1. Startup inspects the pinned
`add_request`, `step`, `abort_request`, and `RequestOutputKind.DELTA` contract
and fails closed if it differs. Each exact `TokensPrompt` request is timestamped
with the host monotonic clock at arrival, at every one-token DELTA, and at its
finished DELTA. Separately, `LLMEngine.add_request(arrival_time=...)` receives
`time.time()` because vLLM 0.27.1 computes engine statistics from wall-clock
arrival. Wall and monotonic absolute timestamps are never subtracted. TTFT is
monotonic arrival to first DELTA, request E2E is arrival to that
request's finished DELTA, and `itl_ms` preserves each adjacent token interval in
generation order. `RequestOutput.metrics` durations are sanity evidence only;
they must not be meaningfully later than the corresponding host-observed
durations. The trial's `batch_wall_ms` remains separate. A step that
returns zero or multiple tokens for one request is rejected because assigning a
single host timestamp to multiple tokens would fabricate an ITL distribution.
Errors abort every pending request and shut down the engine core.

GPU utilization and peak used VRAM are sampled device-wide through NVML. CPU
utilization is derived from psutil CPU times for the frontend and its recursive
worker process tree. The benchmark must run only after the primary-host
preflight confirms that no unrelated CUDA compute process is present.

The canonical repeatability runner performs offline dependency synchronization
and unmeasured fresh-process cache primes for all three distinct vLLM gate
profiles before any of the 20 measured subprocesses. Prime raws are evidence
only and never enter repeatability statistics.

Example (placeholders must be replaced with one selected matrix cell):

```text
uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark \
  --matrix benchmarks/matrix.yaml --prompts benchmarks/prompts.jsonl \
  --result-dir <new-directory> --run-index <1..5> --run-id <id> \
  --warm-state <cold|warm> --concurrency <n> \
  --prompt-tokens <n> --output-tokens <n>
```

The example assumes the outer runner has already completed
`uv sync --frozen --offline` into its fresh repository-external
`UV_PROJECT_ENVIRONMENT`. Direct use without that preparation is noncanonical.
