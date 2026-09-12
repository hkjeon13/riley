# First paired HTTP token measurements

This is the next measurement contract, not a result. The numerical engine remains accepted batch7; optional HTTP token observations are a new server binary and require their own GPU/publication proof. Historical first-text and engine TPOT receipts remain unchanged.

Initial settings are offered C1 with vLLM active1/budget128, then offered C2 with vLLM active2/budget256. Riley retains active1, waiting64 and eight HTTP workers. Both lanes use the same P128/O32 reference, greedy sampling, disabled prefix cache, token IDs enabled and final streaming usage. vLLM keeps its existing CUDA graph and chunked-prefill policy. The C2 choice follows the seven-setting output diagnostic; all warmup and retained responses must still pass the exact reference checks during measurement.

Use five fresh AB/BA process pairs per setting, five nonstream and five streaming warmups per worker, then 1,000 closed-loop refill streaming requests per process. Preserve each process distribution and paired ratios. These are initial throughput/token-latency measurements, not a high-concurrency or P99-stability qualification. A failed warmup or retained response stops refill, drains owned in-flight requests within total deadlines, preserves raw data and marks the setting incomplete; never retry or silently replace a divergent output.

Each client records the time a complete SSE data payload becomes available before JSON parsing. Require exactly one generated ID per token frame, the exact prompt IDs once, 32 exact output IDs, expected text, a length finish, final usage 128/32/160 and DONE. Blank text still counts as a token. A final token-plus-finish frame and a separate tokenless finish frame are both valid. Finish, usage and DONE carry no token observation.

- Token TTFT: first generated-frame arrival minus request start.
- Token TPOT: last minus first generated-frame arrival, divided by 31.
- Token ITL: adjacent generated-frame arrival differences.
- E2E: DONE arrival minus request start.
- Throughput: successful output tokens divided by the common retained interval from the first request start to the last terminal arrival.

These are client delivery measurements. Multiple complete frames in one socket read share its observation time; TCP buffering can produce zero observed ITL without implying simultaneous engine generation. Nonstream warmups establish response correctness, not per-token timing. First visible text remains a separate metric.

Before pausing Blender, freeze source/binary/model/request/reference, all helper dependencies, the new raw-token qualification, the private compute/GL runtime and the complete measurement plan. Use the new round13 restoration helper, including its pinned runtime journal/watchdog and actual vendor-library checks. Every retained lane requires no foreign CUDA compute, GUI idle memory at most 512 MiB and start temperature at most 48°C. Run no other GPU work, builds or large transfers during retained measurement. Restore the exact three authorized sessions in the finalization path.

For C4/C8, the current comparator can change continuation under default batching. They are outside this initial strict-reference measurement. A later comparison must declare its numerical policy and preserve cross-engine equality as a separate observed result. The seven-setting diagnostic is neither a timing baseline nor proof that arbitrary 32-token responses are correct.
