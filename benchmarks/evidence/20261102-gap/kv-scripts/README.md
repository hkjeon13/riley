Fixed KV split, 2026-11-03.

- `kv/`: split-size screen on GPUs 4 to 7, Riley only.
- `kv2/`: parity, API checks and Qwen3-4B long check are valid. `kv2/cmp` is not
  a valid vLLM baseline: vLLM on GPU 4 hit the known 20 tok/s cap.
- `kv3/`: Riley and vLLM on GPU 7, Qwen2.5-7B long check.
- `kv4/`: soak on the split build, long check on two GPUs.
- `curve/`: TTFT against prompt length on GPU 6.
