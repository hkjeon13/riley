# Paired HTTP token delivery measurements, grouped SSE revision

This is the next measurement contract, not a result. The first round13 attempt is preserved as incomplete: vLLM grouped IDs 314/338 into one valid SSE frame on retained request124, after123 successful responses and ten successful warmups. The V1 client stopped at that frame; no full-response correctness or performance can be recovered from that partial stream. The numerical engine remains accepted batch7; optional HTTP token observations are a new server binary and require their own GPU/publication proof. Historical first-text and engine TPOT receipts remain unchanged.

Initial settings are offered C1 with vLLM active1/budget128, then offered C2 with vLLM active2/budget256. Riley retains active1, waiting64 and eight HTTP workers. Both lanes use the same P128/O32 reference, greedy sampling, disabled prefix cache, token IDs enabled and final streaming usage. vLLM keeps its existing CUDA graph and chunked-prefill policy. The C2 choice follows the seven-setting output diagnostic; all warmup and retained responses must still pass the exact reference checks during measurement.

Use five fresh AB/BA process pairs per setting, five nonstream and five streaming warmups per worker, then 1,000 closed-loop refill streaming requests per process. Preserve each process distribution and paired ratios. These are initial throughput/token-latency measurements, not a high-concurrency or P99-stability qualification. A failed warmup or retained response stops refill, drains owned in-flight requests within total deadlines, preserves raw data and marks the setting incomplete; never retry or silently replace a divergent output.

Each client records the time a complete SSE data payload becomes available before JSON parsing. Accept one or more ordered generated IDs per token frame, bounded by the remaining output count. Require the exact prompt IDs once, all 32 exact output IDs, expected text, a length finish, final usage 128/32/160 and DONE. Assign every ID in the same frame its observed arrival time; do not interpolate hidden generation times. Preserve frame token counts and report grouped tokens and within-frame zero ITLs separately from distinct frames coalesced by a socket read. Blank text still counts as a token. A final token-plus-finish frame and a separate tokenless finish frame are both valid. Finish, usage and DONE carry no token observation.

- Token TTFT: first generated-frame arrival minus request start.
- Token TPOT: last minus first generated-frame arrival, divided by 31.
- Token ITL: adjacent token delivery arrival differences, including zero within a grouped frame.
- E2E: DONE arrival minus request start.
- Throughput: successful output tokens divided by the common retained interval from the first request start to the last terminal arrival.

These are client delivery measurements. Multiple IDs within one frame and multiple complete frames in one socket read share their observation time; TCP buffering can produce zero observed ITL without implying simultaneous engine generation. Nonstream warmups establish response correctness, not per-token timing. First visible text remains a separate metric.

Before pausing Blender, freeze source/binary/model/request/reference, all helper dependencies, the new raw-token qualification, the private compute/GL runtime and the complete measurement plan. Use the new round14 restoration helper, including its pinned runtime journal/watchdog and actual vendor-library checks. Every retained lane requires no foreign CUDA compute, GUI idle memory at most 512 MiB and start temperature at most 48°C. Run no other GPU work, builds or large transfers during retained measurement. Restore the exact three authorized sessions in the finalization path.

For C4/C8, the current comparator can change continuation under default batching. They are outside this initial strict-reference measurement. A later comparison must declare its numerical policy and preserve cross-engine equality as a separate observed result. The seven-setting diagnostic is neither a timing baseline nor proof that arbitrary 32-token responses are correct.

The V2 measurement client has separate CPU and recorded-frame replay qualification. Existing baseline and batch8 API/model proofs retain their frozen V1 validators and source identity; their successful responses are not relabelled as V2 measurements. The new controller pins both client modules and the separate client qualification. Each fresh response is checked with V2 during the new campaign.
