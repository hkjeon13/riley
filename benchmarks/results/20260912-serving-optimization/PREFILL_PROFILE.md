# Full-layer P128 profiling and batch6 decision

Frozen batch5 source `a791d63081dcdb319c8e45c79ec9301166ee199a` was profiled with a separate binary. Tool SHA256 is `462e9696921f95f58b2edbd773f23b88e905f11289cadc3be7dcaeff1136ea10`; instrumented graph source is `47bb614cf4310403d50f2d530632b8779d2521e07f5d308756aae1f655f57c9a`. Production source and serving binaries stayed unchanged.

Seven fresh processes ran off, QKV, O, gate/up, down, attention, off. Each validated six exact-output P128/O32 HTTP requests. The last three requests were retained: three prefill and 93 decode replays. Every selected family produced three complete records spanning all 30 layers. Original warmups, response receipts, source/binary/tool hashes, capture/replay bindings, CUDA event inventories, preflight and telemetry are preserved in `raw/prefill-operator-profile/`.

| Family | Median per-replay sum over 30 layers (ms) | Whole prefill graph with events (ms) |
|---|---:|---:|
| QKV | 2.107392 | 13.835520 |
| O | 1.274880 | 13.824672 |
| Gate/up | 5.982208 | 14.003104 |
| Down | 3.236192 | 13.825920 |
| Attention | 1.179648 | 13.823840 |

Off-before and off-after whole-prefill spans were 13.676288 and 13.680736 ms. Each selected capture adds 60 event-record nodes, with whole-graph perturbation of approximately 0.14–0.16 ms for most families and 0.32–0.33 ms for gate/up. These are instrumented diagnostics from separate processes; adding family medians does not produce a measured request total. Nevertheless, the projection families clearly dominate. They explain why batch5's uninstrumented TTFT remained 13.725 ms despite faster decode.

Batch6 implements three related projection changes: populate all 16 MMA input rows, specialize the five shape/rounding combinations, and use 1/2/4 warps per CTA for output widths 192/576/1536. Every output retains its ordered K16 MMA chain and existing BF16 chunk rounding. Only packed-parent P128 prefill selects the new internal wrapper; prior row/M1 kernels and every decode path remain unchanged. No new GPU parents or allocations are introduced. Packed graph identity is F106/g10; the existing numerical gate was freshly qualified below.

Batch6 build and correctness qualification completed on clean source `67a4197b007d4ec6c9f7ad473bdf1c4f2a636111`, a direct child of frozen batch5. The source diff contains exactly the precise projection kernel, internal declaration, packed-prefill dispatch and F106 graph-signature files. The build and test receipts pin the actual release binaries and recheck source and binary identities before and after execution. The numerical gate remains `g04-vllm-smol-p128-v1`; the new implementation is `g10-prefill-m16-v1`.

The checkpoint and tokenizer retain their existing SHA256 pins: weights `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`, tokenizer `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`. The standalone projection probe copies each BF16 weight tensor's raw bytes without conversion, regenerates six row-dependent input fixtures, and reconstructs the exact native case index during validation. It compiles the frozen production precise translation unit with the actual CMake CUDA flags, pinned include response file and shared runtime; runtime version 13000 and SM89 are required.

All 630 real-GPU projection cases passed: seven projections × 30 checkpoint layers × three distinct 128-row patterns. Every output byte matched the preserved rows128 oracle and was finite. Input, weight, position and oracle buffers stayed unchanged; all guards remained intact. All 3,150 device allocations were freed, with zero live allocations, live bytes or cleanup errors and successful stream destruction. This is correctness evidence with no performance timing.

Fresh full-model tests also passed on this source: three prompt comparisons with exact full logits, argmax/status and 96 initialized KV snapshots covering every decode continuation; six retained requests with cancellation after prefill and decode, 42 invalid-shape/stage cases, exact reused outputs/final KV and allocation cleanup; both legacy scheduler-block-mapping checks; and profile unit tests. Fresh release HTTP checks passed for CPU and GPU-greedy sampling, including exact P128/O32 text and token usage, SSE completion, HTTP 400 rejection, disconnect before text and after first text, subsequent reuse, and graceful shutdown. Historical test receipts were not substituted for these executions.

The first projection preparation/build attempt was preserved as attempt1 after a CMake cache parser error. Its multiline regular expression misread cache entries across lines; the corrected driver parses each line separately. The failed attempt stopped before native probe compilation and GPU execution. Fresh fixtures and receipts pin the corrected driver; the failed attempt contributes no passing proof.

| Completed artifact | SHA256 |
|---|---|
| `batch6-build.json` | `39e30dacda86c41c104aa211ba3c9603f36e478d0558773968d88de22334459d` |
| `batch6-qualification.json` | `de72a7dadbda9a73a2e869c6981853c2e2946a6c84a4cac5e95cc73f753619d7` |
| `batch6-projection-probe/receipt.json` | `2c26da3eee39533f0608d541bff5312412fe942bdeea4a91b25455c729505e89` |
| `batch6_projection_probe.py` | `e306157697d5584b1f5d59826326f1682e6be57c224108fed68e8a2c3146e3d9` |
| `batch6_projection_probe.cu` | `d8881a5d6f40a8bc2065f8b75fc0dd306afcca096ee8e38485a3b336712fe34b` |

Round8 matched HTTP and engine measurements completed after successful qualification. Batch6 is accepted as the current Riley c1/P128/O32 baseline. HTTP throughput is **804.884386 tok/s**, E2E **39.568668 ms** and first text **5.180778 ms**, compared with vLLM **882.943510 tok/s**, **35.359594 ms** and **7.476989 ms**. Against batch5, HTTP throughput improves **22.969%**, engine TTFT decreases **66.323%**, and TPOT changes **-0.053%**, effectively unchanged.

Engine measures **819.966638 tok/service-s**, TTFT **4.6222465 ms**, TPOT **1.109772 ms** and E2E **39.024704 ms**, versus vLLM **828.068963**, **7.108036 ms**, **0.936472 ms** and **36.434081 ms**. TTFT is lower, but all five HTTP throughput ratios are below one (**0.905–0.922**) and median TPOT remains higher. The vLLM engine rate spans **826.1–861.5 tok/service-s**; tail requests affect the sum of service times, so the close rate medians do not establish parity or contradict the median E2E gap. HTTP first text is not token TTFT, and HTTP TPOT remains unmeasured. P95/P99 are nearest-rank estimates from 30 retained requests per process, not high-concurrency stability proof. Full process ranges and paired ratios are in `batch6-comparison.json`.

The separate batch6 whole-graph diagnostic measured prefill **4.789680 ms** and decode **1.143168 ms**, with 397 prefill and 277 decode kernel nodes. Its existing summary includes six requests and all 6 prefill / 186 decode replays, including warmups; these spans are not serving timings. The prior **139,353,984-byte** packed weight/output allocation cost remains unchanged. The completed serving corpus now contains **14 campaigns, 70 pairs, 140 process runs and 4,200 retained requests**.

Round9 then measured residual decode attention in a separate off/attention-all/off diagnostic. The full 30-layer attention sum was **0.482304 ms** over 93 retained decode replays, with no missing or failed intervals. Whole decode spans were **1.148064 ms**, **1.227936 ms** and **1.107456 ms**; 60 extra event nodes cause approximately **0.080–0.120 ms** of perturbation. Attention remains material. Batch7 implementation now targets independent QK, two-warp PV and exact softmax publication by warp 0, with no correctness or performance result claimed yet.

Blender restoration after round7 profiling and round8 serving remains preserved in their receipts. The latest round9 restoration was verified in `raw/blender-round9-restore-verified.json`: PIDs **4029998 / 4029999 / 4030027**, ports **9876 / 9911 / 9887**, with matching commands and GUI environment. A future authorized pause must reverify these exact successors by identity and start time. The GUI 512 MiB measurement envelope remains distinct from canonical 256 MiB qualification, and the overall serving goal remains open.
