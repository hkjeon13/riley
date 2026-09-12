# vLLM output diagnosis across seven settings

All seven fresh vLLM processes completed their planned **123 correctness-only requests**, with Blender retained and the private 580.173.02 compute runtime. This experiment measures no serving performance and does not establish concurrent numerical correctness. Raw response, process, startup, runtime-mapping and cleanup evidence is in `raw/vllm-output-matrix-diagnostic/`; `vllm-output-matrix-analysis.json` verifies the response hashes and counts.

Each setting used two nonstream waves at offered C, one offered-C1 request on the same server capacity, one streaming C-wave with token IDs and final usage, and a separately labeled C-wave requesting top-1 logprobs. All use the unchanged P128/O32 input, weights/tokenizer, greedy sampling and default vLLM numerical policy. The extra logprob wave is a diagnostic execution condition and is not a timing control.

| Offered C / vLLM token budget | Requests | Exact c1-reference IDs | Distinct continuations | First differing indices (request count) |
| --- | ---: | ---: | ---: | --- |
| 1 / 128 | 5 | 5 | 1 | none |
| 2 / 128 | 9 | 1 | 3 | 8 (4), 29 (4) |
| 2 / 256 | 9 | 9 | 1 | none |
| 4 / 128 | 17 | 5 | 4 | 8 (8), 29 (4) |
| 4 / 512 | 17 | 1 | 3 | 8 (16) |
| 8 / 128 | 33 | 1 | 5 | 8 (12), 29 (20) |
| 8 / 1024 | 33 | 0 | 3 | 8 (32), 29 (1) |

Every response had the exact 128 input IDs, 32 generated IDs, consistent 128/32/160 usage and a length finish. Each streaming response contained 32 one-ID generated frames, one usage-only frame and one DONE. Each wave reached its declared overlapping client count. All 29 logprob requests, totaling 928 selected tokens, reported the selected token at the returned top-1 probability. This is a self-reported greedy observation, not an independent full-logit oracle.

Budget 256 removes the observed C2 difference in this sample. Larger budgets do not generally restore c1 equality: the capacity-eight/budget1024 server also differs on its single offered-C1 request. Execution capacity, CUDA graph shape and scheduling can therefore matter beyond the instantaneous client count. This experiment does not isolate the responsible kernel or establish a numerical error bound.

The previous Round12 text-reference gate and failure remain unchanged. These observations support an initial **strict-reference C1 and C2/budget256** token-aware measurement, subject to fresh warmup and retained-request checks. A nine-request diagnostic cannot guarantee that a longer C2 run will remain exact. C4/C8 need an explicit comparison policy that separately reports candidate numerical preservation, HTTP publication correctness and cross-engine output equality; do not substitute an arbitrary continuation or silently mark equality true.

The helper SHA256 is `4d21bf1d35aeaf9637e5ff6a188a50ed5aaed8415cf4633ad39daad8b64f9566`. Whole-experiment completion SHA256 is `f75a4608c7958f201d2cd15352926d83e4c41dfac56c3d02fd93cf75b7307848`. The diagnostic left all three Blender sessions unchanged and cleaned every owned server process group.
