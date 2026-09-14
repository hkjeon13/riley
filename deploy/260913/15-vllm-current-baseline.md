# Refresh the current vLLM serving baseline

The official latest release checked on 2026-09-14 is [v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0), released September 9. It defaults to Model Runner V2 and provides CUDA 13.0 wheels. Existing local benchmark results use v0.27.1 and cannot establish a latest-release comparison.

After the short-prefill screen finishes, create a separate pinned environment for vLLM 0.29.0; preserve the 0.27.1 environment and every frozen Riley binary. Record resolved packages, Python/Torch/CUDA versions, driver/GPU, exact command and model file identities. Do not install, build or run competing GPU work during timed serving.

Use the same model/checkpoint, BF16, tokenizer, prompt corpus, token budgets, KV capacity and prefix-cache policy. Preserve default supported vLLM execution optimizations; do not force legacy Model Runner V1 merely to resemble the older comparator. Use `vllm serve` (the older api_server module entry point is deprecated in 0.29.0). Verify startup backend selection and completed responses before benchmarking. Any unsupported option or changed default must be recorded rather than silently omitted.

Compare frozen best Riley, the selected candidate, and both vLLM versions in reversed orders. First C32 shared and unique screens, then sufficient sustained C8/16/32/64 load with errors, throughput, TTFT, TPOT and P95/P99. Maintain exact Riley reference checks and stop/cancel/recovery. Report vLLM token agreement separately from protocol completion. Version-refresh results do not substitute for multi-model, multi-GPU or Hopper/Blackwell qualification.

Rollback consists of selecting the preserved 0.27.1 environment; no destructive in-place package upgrade is required. Status: official version verified; environment installation and measurements pending.
