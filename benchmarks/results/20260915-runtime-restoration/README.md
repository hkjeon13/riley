# Runtime asset restoration — 2026-09-15

The C8 v2 lifecycle failed before server launch because the vLLM virtualenv's Python symlink pointed into a missing `/data/riley-vllm-interim.CfrT9T` directory. The old Riley checkpoint directory was also absent. The cause of disappearance is unknown. No timed comparison was produced by that attempt; its remote spool and lifecycle records are retained.

Restored assets live under `/data/riley-serving-260913-recovery/runtime-assets-20260915`. `downloads.json` records hashes and sources; `workload-update.json` records the workload manifest before/after hashes and unchanged natural fixtures.

- Restored CPython 3.13.15 from the Astral 20260901 standalone release, verified against the release asset SHA-256. This preserves the Python version but replaces the base interpreter distribution; it is not proof of byte-identical historical Python runtime.
- Safe extraction initially rejected a terminfo symlink escaping the extraction destination. A fresh extraction directory excludes `python/share/terminfo/` and retains the safe data filter. The failed partial extraction is preserved remotely.
- Repointed the existing vLLM virtualenv interpreter and pyvenv.cfg; packages were not reinstalled. Original link/config are backed up remotely. `pip check` exited 0 with no broken requirements. Metadata reports vLLM 0.29.0, torch 2.13.0, transformers 5.17.0, flashinfer-python 0.6.18. Metadata checks alone do not prove server startup.
- Restored SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Config, safetensors and tokenizer size/hash match the checked-in checkpoint fixture. Both engines use this restored model directory.
- Frozen prior/candidate Riley binaries are unchanged. This recovery adds no Python dependency to the Riley serving runtime.

C8 v3 uses 64 warmup and 2,048 retained requests per lane, two reversed engine orders, and shared/unique workloads. Host quiet timeout is explicitly zero; PSI remains observational and cannot be used to correct rates. GPU ownership, memory and correctness checks remain enabled. Results must be compared within the new run, without pooling historical measurements across runtime restoration. Projection CTAs remain opt-in pending serving evidence.

At the time this restoration note was written, Riley reference generation had completed; vLLM preflight and all timed comparisons remained pending. This directory contains recovery evidence, not a successful benchmark receipt.
