# Batch7 decode operator profile

Attention remains the largest measured decode family at **314.368µs across all 30 layers**, followed by packed gate/up at **244.736µs**. Whole decode graph medians with operator timing disabled were **0.969760ms before** and **0.970976ms after** the screen. These diagnostic traces identify optimization candidates; serving throughput and latency require the separate uninstrumented campaigns.

The exact 12-case order was off-before, QKV, RoPE/KV, attention, O, post-attention norm, gate/up, SwiGLU, down, next/final norm, head, off-after. Each case used a fresh process with SmolLM2-135M BF16, P128/O32, greedy sampling and concurrency one. All six HTTP requests per process matched the reference text and length finish. The first three were discarded: raw replay IDs 1–192 became retained IDs 97–192, containing three prefill replays at position 127 and 93 decode replays at positions 128–158. Totals are **72 validated requests, 36 retained requests, 2,304 raw replays and 1,152 retained replays**. Every retained family interval was measured; no failed replays or missing timings were recorded.

Each operator value below is the median of the **per-replay sum across layers 0–29**, rather than a sum of layer medians. Head is one global LM-head GEMM interval (`layer: null`). Whole graph columns are medians from that same case; “off” disables operator intervals while retaining whole graph timing.

| Case, in execution order | Operator span (µs) | Whole decode graph (ms) | Whole prefill graph (ms) |
|---|---:|---:|---:|
| Off-before | — | 0.969760 | 4.787040 |
| Packed QKV | 156.672 | 1.084608 | 4.780288 |
| RoPE/KV write | 107.520 | 1.105856 | 4.791040 |
| Two-warp attention | 314.368 | 1.062144 | 4.607680 |
| O projection | 140.288 | 1.076576 | 4.599392 |
| Post-attention norm | 105.472 | 1.070208 | 4.791424 |
| Packed gate/up | 244.736 | 1.109056 | 4.800320 |
| SwiGLU | 93.216 | 1.061184 | 4.606304 |
| Down projection | 169.984 | 1.077856 | 4.793472 |
| Next/final norm | 108.544 | 1.087904 | 4.796768 |
| LM head, single interval | 64.512 | 0.972192 | 4.787072 |
| Off-after | — | 0.970976 | 4.784320 |

The off decode medians differ by 1.216µs, approximately 0.125%. Selecting a layer family adds **60 external event-record graph nodes**; head adds two; off adds zero. Every prefill capture has zero operator event nodes, and every capture has zero event-wait nodes. Selected layer-family whole decode spans exceed the two off baselines by 90.208–139.296µs, approximately 9.29–14.36%; the head difference is 1.216–2.432µs. These differences include instrumentation effects and run variation, so they are not a universal overhead correction. Prefill medians vary from 4.599392 to 4.800320ms even though no prefill operator events were captured.

Each family was measured in a separate process and compared with both off logs. **Do not add family medians, divide them into an exact uninstrumented time breakdown, or infer an equivalent serving gain.** The first input norm, embedding, argmax and transfers are outside operator coverage. The original completion synchronization is retained, with zero added synchronizations; GPU spans must not be added to CPU synchronization waits that already include GPU execution. Summary ratios retain `matched_runtime_provenance_verified: false` and `performance_claim_eligible: false`; the campaign separately checks source, tool and binary pins.

All cases used the retained-GUI diagnostic condition `serving-gui-retained-512mib-cool48-v1` on GPU `GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`. Recorded idle memory was 306MiB, start temperatures were 43–45°C, and no foreign compute process was present at each preflight. This is outside the canonical headless 256MiB condition. Per-case telemetry, raw warmups, retained logs, capture inventories, HTTP responses and their hashes are preserved in [raw/decode7-operator-screen](raw/decode7-operator-screen/).

The base is frozen batch7 commit `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`. The diagnostic copy is intentionally instrumented (`source_clean: false`) and uses a separate binary; it does not replace the serving build. The pinned SHA256 identities are:

| Artifact | SHA256 |
|---|---|
| Batch7 build receipt | `4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395` |
| Original `graph_resources.cu` | `8586726646729333e9bcea17c419b530aeb30e47cc1727189b701d9468d5dad5` |
| Instrumented `graph_resources.cu` | `8d616c3cf2649c5bafd3f487af84b19ee870cdd0a7aea02150fd6e009caeccf5` |
| Adapted decode operator tool | `6cb953a66df57150834e9df38d1a65fa82de976d2af1c583ffa2db3c912509d0` |
| Historical whole graph tool | `9f0f3a526ed1a47c3e6a37e33111a21356ac5b2d58294ff797ec0a40a5091b78` |
| Diagnostic `riley` binary | `4f13752623c47082649b09a124f10c49444491c170ee6d26eed4cf27462c1f41` |
| Diagnostic build receipt | `4ce666b6437149b68f7add65b7b5b04577f2f7d4c0f54fa21db5218eb72995bf` |
| Round11 restore helper | `916ae1ae972915b29cd895d47ffefe033a0876365c68a03cb9c8ac4df929dc9e` |
| Round11 verified restoration | `40930e8fa4a76759029e79624c44751bbdc3b0d570dba98c17b12814c66b2124` |

At round11 completion, the restore receipt verified successors **4128432 / 4128433 / 4128445** on ports **9876 / 9911 / 9887**, with matching commands and GUI environment. This records that restoration event, not current process liveness. Screen completion was written only after restoration verification.

The values above were checked against the synced [screen summary](raw/decode7-operator-screen/summary.json), [completion receipt](raw/decode7-operator-screen/completion.json), [preparation pins](raw/decode7-operator-screen/preparation.json), [diagnostic build receipt](raw/decode7-profile-build.json) and [round11 restoration receipt](raw/blender-round11-restore-verified.json). Local checks also matched every case's raw/retained/HTTP hashes, exact replay IDs, successful statuses and event-node inventory. No new GPU work was performed to write this report.
