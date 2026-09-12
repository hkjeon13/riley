# Batch7: residual decode attention

Batch6 is the accepted serving baseline. Matched c1/P128/O32 HTTP throughput is 804.884 tok/s versus vLLM 882.944; engine TTFT is 4.622 versus 7.108 ms, while TPOT remains 1.109772 versus 0.936472 ms. This profile investigates that remaining decode cost and does not claim overall goal completion.

An isolated binary from source `67a4197b007d4ec6c9f7ad473bdf1c4f2a636111` ran three fresh processes: events off, attention across all 30 layers, events off. Each process validated six exact HTTP outputs; three warmup requests were excluded, leaving three prefill and 93 decode replays. All intervals and output checks passed.

| Diagnostic | Prefill graph ms | Decode graph ms |
| --- | ---: | ---: |
| Off before | 4.784032 | 1.148064 |
| Attention events | 4.619808 | 1.227936 |
| Off after | 4.614912 | 1.107456 |

The median sum of all 30 attention spans is **0.482304 ms** across 93 retained decode replays. Each layer is approximately 16 microseconds. The 60 event nodes perturb the whole graph by approximately **0.080–0.120 ms** relative to the two brackets. This is diagnostic evidence, not a serving speedup. Even allowing for that overhead, attention remains a material optimization area. Raw records, capture identity, exact output validation, warmup filtering and telemetry are under `raw/attention-batch6-full-profile/`.

Diagnostic tool SHA256 is `44cf0fd8ea5797592518ef8f332567a041663a52cb3ba4f9900002047bad840b`; instrumented graph SHA256 is `da06b2d2862d6391b2be9bf5370c2c3754445f70c2094c2166f78fb05fced45e`; diagnostic binary SHA256 is `12cc2428c349646d3460cad240c50ca4b044347b44c207959e24a2f6c5a32831`. No historical projection hooks were enabled because they bypass the M16 prefill branch.

Batch7 implements three related changes in a separate packed-decode kernel: two warps process independent QK token groups; each warp owns two complete PV output-block chains; warp zero publishes the exact lane-specific FP32 rescaling and inverse values. Reverse tile order, per-output MMA order, BF16 rounding and denominator additions remain unchanged. Cross-warp barriers cover every tail, including positions where one QK warp has no tokens. The accepted kernel remains an unchanged prefix and an independent oracle. Only packed decode selects the new wrapper; P128 prefill and device allocations stay unchanged. The candidate is implemented and built from clean source `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`. Full-model logits/KV, retained lifecycle, legacy GPU, profile CLI and both HTTP samplers passed. The standalone GPU probe passed all 8,640 synthetic cases against both oracles, full 576-word comparisons and mapping invariance, with 60,480 allocations freed and zero live bytes or cleanup errors. Five fresh HTTP and five engine pairs now support acceptance as the current baseline. HTTP throughput is 916.998713 versus vLLM 894.053067 tok/s (all five paired ratios above one), while engine TPOT is 0.948323 versus 0.938211 ms; the overall goal remains open.

Acceptance requires full 576-word exact comparison against both prior attention oracles for positions 128–159, three deterministic input patterns and three physical block mappings, followed by fresh full-model logits/KV, retained lifecycle and HTTP validation. Then run matched HTTP/engine comparisons against accepted batch6 and fresh vLLM controls. A standalone probe cannot replace the serving result.

Round9 completed and restored the three authorized Blender processes to PIDs **4029998 / 4029999 / 4030027**, ports **9876 / 9911 / 9887**, with matching commands and GUI environment. The public receipt is `raw/blender-round9-restore-verified.json`. Existing authorization covers only these verified successors for subsequent measurement pauses.

Batch7 build receipt SHA256: `4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395`; independent attention receipt: `cd448172460496201177099071152e60f5160fd757994775e46e19e02d28f94b`; full-model GPU receipt: `64f8b49e6fcf345293d18f18d50ca7f966a69b35014b77a5f1491a8ca276d0d0`; HTTP receipt: `aa23a573e677a1fe13935c24fa31de97cf65413a1f515bfeeb85c63d39c92288`. The attention fixture layer indices are synthetic seeds, not actual checkpoint activations; actual model validation is the separate integrated GPU test.

Serving comparison: `batch7-comparison.json`. HTTP throughput improves 13.929% and engine TPOT improves 14.548% versus batch6, while TTFT changes +0.044%. Separate whole-graph diagnostics measure 4.779152 ms prefill and 0.961136 ms decode, including warmups. Round10 restored current successors 4087125 / 4087139 / 4087140 on the same three ports; see `raw/blender-round10-restore-verified.json`.
