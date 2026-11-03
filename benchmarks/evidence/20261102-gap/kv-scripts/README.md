Fixed KV split, 2026-11-03.

- `kv/`: split-size screen on GPUs 4 to 7, Riley only.
- `kv2/`: parity, API checks and Qwen3-4B long check are valid. `kv2/cmp` is not
  a valid vLLM baseline: vLLM on GPU 4 hit the known 20 tok/s cap.
- `kv3/`: Riley and vLLM on GPU 7, Qwen2.5-7B long check.
- `kv4/`: soak on the split build, long check on two GPUs.
- `curve/`: TTFT against prompt length on GPU 6.
- `grid/`: strided attention grid against `kv1`; `c32-*` repeat closed-c32.
- `merge/`: per-thread merge (`merge1`) against the strided grid (`grid1`).
- `prof/`: Nsight Systems summaries (`summary*.txt`, `steps-*.txt`, `layer-*.txt`).
