# Batch4 operation costs and batch5 decision

The isolated diagnostic binary `8840bed5743be3e69308182941f5ffff7f59acb9c6acdea29c6892d03f13b59b` was built from frozen batch4 `53677519082135c42a5d044044c5b31cad4a146a`. Production serving binaries were unchanged. All 18 fresh diagnostic processes validated six P128/O32 HTTP requests; the first three requests were excluded from timing summaries, leaving three prefill and 93 decode replays per process. GUI512MiB, idle/temperature checks and raw telemetry are retained per process.

The 15-case screen selected layers 0, 15 and 29 in separate captures. The following values are medians of each replay's sum across those three layers, not sums of independently computed medians.

| Decode family | Three-layer CUDA span (us) |
|---|---:|
| Attention | 136.192 |
| Gate/up | 23.552 |
| Down | 16.384 |
| QKV | 15.360 |
| O | 13.312 |
| RoPE/KV | 10.240 |
| Post-attention norm | 10.240 |
| Next/final norm | 10.240 |
| SwiGLU | 9.216 |

The separately measured single head projection was 64.512us. First-layer P128 projection probes measured Q 41.984us, gate 99.328us and down 108.544us. These prefill probes cover one layer and do not establish whole-prefill family totals.

The follow-up measured attention across all 30 layers. Its per-replay sum was **1.366016ms**, with layer medians between 44.032 and 46.080us. All 93 retained decode records had complete intervals. Whole decode graph span was 2.117952ms with the 60 external event nodes; off-before and off-after were 1.986304 and 1.993248ms. The added events perturb timing by about 0.125–0.132ms, so 1.366ms is a diagnostic interval sum, not an exact share of uninstrumented runtime. Attention remains the dominant measured family after allowing for this perturbation. No GPU event span is added to the host synchronization wait, which already includes GPU execution.

Batch5 groups three related attention changes: one warp per query head, two independent output-block partitions per head, and reuse of the original FP32 exponentials for the MODE6 denominator. The existing attention kernel remains the M1/P128 oracle. Only packed decode selects the new kernel. Reverse tile order, per-output MMA accumulation, denominator addition order and BF16 rounding must remain byte-exact. The candidate is not accepted until full logits/status/KV, retained reuse/cancel/rejection, allocation cleanup and HTTP qualification pass, followed by matched HTTP and engine serving campaigns against current batch4 and vLLM.

Raw evidence: `raw/operator-screen/`, `raw/attention-full-profile/`, `raw/operator-profile-build.json`, instrumentation receipt and source/tool hashes. Each case retains original warmups, filtered retained logs, capture/event inventories and the exact-output validation receipt. Blender was restored after each session; latest successful receipt is `raw/blender-round5-restore-verified.json`.

## Batch5 serving outcome

The candidate passed fresh integrated GPU/HTTP qualification and completed both five-pair serving campaigns. HTTP throughput increased55.673% from420.459 to654.542tok/s; engine TPOT decreased44.201% from1.989926 to1.110364ms. Engine TTFT stayed13.725ms. Batch5 is accepted as the new c1 baseline, while vLLM remains faster at894.322HTTPtok/s and0.920084ms engine TPOT. See `batch5-comparison.json` for process-level distributions and metric boundaries. These results use uninstrumented binaries, not the operator traces above.
