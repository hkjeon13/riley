# Decode projection row-parallelism native screen

Candidate batch: QKV, attention output projection and FFN down preserve K16 recurrence, BF16 partial boundaries and the existing scratch layout while placing each M16 tile in an independent warp. Compare two warps in one CTA (root logs) against separate CTAs (`cta/`). Baseline is the current adaptive projection in both independent runs.

| Active rows | Two warps in one CTA: duration change | Separate CTAs: duration change |
|---:|---:|---:|
| 1 | +22.66% | +5.65% |
| 8 | +18.05% | -0.04% |
| 16 | +8.73% | -13.31% |
| 17 | -7.74% | -25.71% |
| 24 | -6.19% | -18.85% |
| 32 | -6.19% | -18.38% |

Negative duration change is faster. Each value compares medians of four alternating-order measurements within its own run; this is not a cross-run baseline comparison or a confidence interval. Each measured graph has 90 operator calls spanning 30 synthetic weight regions (106,168,320 bytes total), with 20 warmup replays and 100 timed replays per sample. CUDA events measure GPU graph execution. No full model, client/server, vLLM, TTFT, TPOT or tail-latency claim is made.

Both variants passed 102 graph bitwise comparisons (active rows 0..33 across three input seeds), inactive/guard checks, and bounded memcheck with 34 comparisons and zero errors. Correctness uses the first layer weight region; timing traverses all 30 regions. Input distribution is synthetic. These are not actual-model quality or serving qualification. Both lifecycle receipts confirm restoration of the three Blender instances; the CTA run is the later receipt.

Decision: retain the separate-CTA variant for actual-model validation. It improves the three-operator graph duration by 13.31–25.71% at 16–32 rows, is effectively flat at eight rows, and regresses 5.65% at one row. Do not promote it unconditionally. The two-warps-per-CTA variant has larger small-row regressions and smaller high-row gains, so stop that variant. A full-model/serving check must determine whether low-row fallback is needed and quantify its own dispatch overhead.

`selection.json` ties the experiment to an existing serving profile. The affected operators account for 27.80% of the selected shared decode graph span. Multiplying this fraction by the measured 32-row CTA duration saving gives about 5.11% of that graph span as a hypothetical saving, not a serving prediction. Full serving includes other GPU work, host costs and varying batch sizes.

Reproduce using CUDA 13.0 nvcc with `-std=c++17 -O3 -arch=sm_89`, and add `-DRILEY_PROJECTION_CTA_SPLIT` for separate CTAs. The shared probe and optional header are in the repository. `sources-v1/` preserves the initial two-warp experiment sources before introducing the CTA alternative. `provenance.json` records frozen binary hashes and the final CTA source dependency hashes, all checked against this checkout. Lifecycle scripts are archived for reproducibility and require fresh authoritative Blender process identities.

Production dispatch remains unchanged. Hopper/Blackwell/multi-GPU execution is untested. Native results do not overcome the previously recorded available-memory guard that prevented the separate owner-index serving attempt.
