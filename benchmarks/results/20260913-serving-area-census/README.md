# Serving area census — 2026-09-13

This is profiling evidence to choose the next optimization batch, not another serving-performance result. The [unprofiled comparison](../20260913-flashinfer-serving-screen/README.md) remains authoritative for throughput and latency.

## Measured scope

Same frozen binary `189d05728de55b3e54d660ccff89730add89adcc45f38ddc4f74070e1bd3d5cb`, V7 and experimental FlashInfer v2, SmolLM2-135M BF16, RTX4090, C32 natural 16/128/398 prompts. Each process served96 streaming requests; both receipts record exact agreement with this frozen reference and exit0. This is not resolution of the separate quality failures. Nsight CUDA graph-node tracing adds overhead. The census selects the middle80% of launches by count; this is not a steady-state window or an isolation of pure prefill from mixed iterations.

| Profile / stage | Launches in window | Attention kernel share | FFN gate/activation/down share | Other projection/QKV/RoPE share | Residual/norm share |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / decode | 203 | 33.59% | 32.90% | 19.96% | 8.12% |
| baseline / prefill_or_mixed | 43 | 34.74% | 27.81% | 31.85% | 3.89% |
| flashinfer / decode | 199 | 28.14% | 35.47% | 21.42% | 8.74% |
| flashinfer / prefill_or_mixed | 47 | 29.47% | 29.19% | 34.06% | 4.13% |

Shares are within summed kernel duration of each selected stage. They exclude graph gaps/copies and cannot be interpreted as end-to-end removable fractions. Different profiles produced different graph counts; do not subtract total durations as an A/B speedup. Every raw demangled kernel and duration is retained in areas.json. Unclassified kernels account for less than1% in every stage. CPU launch/device activity mapping, stream ownership, and coverage are checked by the existing overlap analyzer.

## Decision and memory contract

Proceed with PR05 as one FFN memory-pipeline batch, measured against exact V7. The current V7 already fuses gate/up/SwiGLU and merge/norm, so removing those launch boundaries again would duplicate implemented work. Gate/activation/down still accounts for32.90% of V7 decode kernel time and27.81% of prefill/mixed kernel time. This supports an area choice, but does not prove the area is bandwidth-bound; hardware memory-traffic counters were not collected.

At32 rows, the unique BF16 activated intermediate is32×1536×2 =98,304 bytes. Its producer write plus one logical consumer read is196,608 bytes/layer. The three FFN weight matrices contain3×576×1536×2 =5,308,416 bytes/layer; these are logical tensor sizes, not observed DRAM bytes or traffic ceilings. The existing five-way down projection also writes a5×32×576×4 =368,640-byte partial tensor before merge/norm reads it. Caches, repeated CTA reads and profiler effects are not inferred from those sizes.

A fully resident16-row activation tile alone needs49,152 shared bytes. One CTA per16-row tile would expose only two CTAs at32 rows; partitioning output columns would instead recompute producers or require communication. Therefore full gate-to-down fusion is not chosen merely to remove the96KiB intermediate. Keep phase boundaries for this batch and pipeline global-to-shared transfers against existing MMA operations.

The concrete implementation contract and tests are appended to [PR05](../../../deploy/260913/05-memory-aware-ffn-execution.md). No kernel implementation or performance improvement is claimed by this census.

## Reproduction / evidence

`controller.py` invokes the frozen native binary through Nsight; its Python client is offline. Raw `.nsys-rep` and SQLite files remain at `/tmp/riley-opt-260912/serving-areas-v1-{baseline,flashinfer}/c32.*` on `ai-assistant`. Each areas.json contains the source SQLite SHA256 and launch coverage. Run `python3 benchmarks/analysis/serving_area_census.py TRACE.sqlite OUTPUT.json`; retain the sibling `overlap_headroom.py` import. No Blender restoration was performed.
