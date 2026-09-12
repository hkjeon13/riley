# Grouped token delivery: Round13 failure and Round14 measurement

Batch8 remains a correctness-qualified candidate pending serving comparison. The current comparison uses SmolLM2-135M BF16, the fixed P128/O32 reference, offered C1/C2, Riley active1 and vLLM active1/budget128 or active2/budget256. It does not establish broad workload or high-concurrency success.

## Actual Round13 failure

The first V1 token campaign completed five nonstream and five streaming warmups, then123 retained vLLM C1 responses. Retained request124 contained an SSE frame with `token_ids:[314,338]` and text `" is that"`. The V1 client rejected more than one ID per frame. This is a valid grouped delivery shape; it does not establish a numerical output error. The client had recorded21 IDs before that frame and closed the connection on the parser exception. The saved stream has no finish/usage/DONE, so full-request correctness and timing cannot be recovered.

No comparison pair completed. Preserve `raw/token-serving-round13/` as incomplete. Plan SHA256 is `0ca01ebc7b16f9422db84f0f5afa233c6518773bb5e7c141e3ef9d42ed328de1`; finalization SHA256 is `2f61901debee8bf5c9cb9096766b48193572d6a8d9d98229c6df524fc276ced0`.

The controller gracefully stopped its owned vLLM process and restored all three authorized Blender sessions. Their new PIDs2234550/2234607/2234700, ports9876/9911/9887, commands, GUI environment and actual private GL/compute mappings were verified. `raw/blender-round13/verified.json` has SHA256 `9bf3b9d644015fec626122271d973e63fcab70f4ee9427320757d009375e1fb0`.

## V2 client and qualification

`serving_token_client_v2.py` accepts an ordered group bounded by the remaining output count. Every ID in that frame receives its actual arrival timestamp; no hidden generation times are inferred. Text is appended once per frame. The parser still requires exact prompt/output IDs, text, finish reason, usage, DONE and complete transport. It retains deadline/EOF/cleanup behavior. Schema versions distinguish V2 observations/phases from archived V1 evidence.

TTFT is first token delivery minus request start. TPOT is last minus first token delivery divided by31. ITL includes zero within each grouped frame; grouping counts distinguish these from separate frames received by one socket read. These are client delivery metrics, not CUDA or engine generation timings. First visible text stays separate. See `TOKEN_BENCHMARK_CONTRACT_V2.md`.

The new client passed26 CPU/parser/HTTP-loopback tests locally and on the remote host. The separate CPU-only qualifier replayed all133 successful Round13 responses through both clients and found their common token/text/usage/timestamp/metric fields unchanged. Replaying the failed frame through V2 produced the exact23-token prefix, still incomplete and unqualified. It did not invent terminal events or promote the failed campaign.

| Artifact | SHA256 |
| --- | --- |
| V1 client, unchanged | `a2a4a35569d6b892542097c119e2be8a6402beb60ab1565aa95658794c364766` |
| V2 client | `2bc9238265666b99654f4f4e456ca0bb10c45b8bed8f28493452b89c1f450fcf` |
| V2 test suite | `0f963a9cb4a37d031f2c12ef94788b97ddc3598022be527d66743ebb6f0aa353` |
| Grouped-client validator | `314bfa0e9e5f8c76c5c0c0d2ed870a06b148b9debd7ffaf042b62ff4b72bc983` |
| Remote client qualification | `749a372eae0c278cf6386c6c7c7c2b58457b8d8abfcf97b1db0cb51cf4254263` |

Existing baseline/Batch8 model and184 combined API checks retain their original V1 validators. No serving source, binary or vLLM configuration changed for this client correction. The new controller independently pins the V2 qualification and both client modules. Every fresh warmup/retained response uses strict V2 validation.

## Round14

The new plan uses five fresh AB/BA pairs for each of baseline/vLLM, candidate/baseline and candidate/vLLM, separately at C1 and C2. Each process receives five warmups per worker per transport and1,000 retained streaming requests:60 fresh processes and60,000 planned retained requests. This is the requested measurement inventory, not a completed count.

Plan SHA256: `199d70ebf61e03d3db00ed557b454632daf2aad934ae56d6c4e027b37de7ed7b`. ControllerV4 SHA256: `4b51034247b4301e390897788ffa3f7c0f1ce26bb313a9fc718afebc6524ee0e`. Readiness passed with5,484 immutable pins and573 transitive artifacts. Controller tests19 and session tests5 passed; independent reviews found no remaining blocker.

The initial plan-generator attempt interpreted a historical model inventory's relative `config.json` entry as a current artifact path and stopped before writing a plan or pausing Blender. Its log remains preserved. The final generator and controller freeze the grouped qualifier's explicit inputs/raw/log bytes; archived plan/preparation/finalization documents are historical provenance. This boundary applies only to the exact validated grouped qualification. All new source/model/runtime dependencies are independently qualified and remain fully pinned. No generic relative-path exception was introduced.

Measurement started at approximately11:05 KST. No performance result is claimed in this document yet. During retained measurement, other GPU work, builds and large transfers are excluded. The round14 helper binds the exact three predecessor receipts and retains pidfd identity checks, private journals, independent restoration watchdog and final vendor-library verification. Finalization must restore those sessions before the campaign can complete.
