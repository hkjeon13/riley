# FP32 context-split native control

Experimental BF16-KV/FP32-probability partial and merge implementation. **Not an accepted model backend or serving result.** No Rust runtime or default profile was changed.

## Measured scope

The candidate reuses the existing score layout, computes independent maxima and unrounded FP32 probabilities, accumulates scalar FP32 P×V partials and merges rescaled partials. Scratch is fixed at 2,433,024 bytes for 32 rows × 9 heads × 32 partitions. The caller owns scratch and validates allocation/page extents. Host null/extent errors are rejected before enqueue; invalid device row/count metadata leaves output untouched. This header is an isolated prototype, not a new public C ABI.

The probe initializes deterministic BF16-representable score/value data and a noncontiguous bijective page mapping. Scores are supplied directly, not generated from model Q/K. Timings compare existing `independent_values` against candidate partial+merge, excluding the common QK producer. Each graph receives 5 warmups and 40 timed replays using CUDA events; one fixed-order sweep is an exploratory screen, not a stable crossover estimate. The table selects the fastest of four measured split counts retrospectively and cannot define production dispatch.

| Context | Active rows | Existing P×V µs | Best measured split count | Candidate partial+merge µs | Time change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 3.379 | 4 | 5.171 | +53.03% |
| 128 | 8 | 3.482 | 4 | 5.248 | +50.74% |
| 128 | 32 | 4.557 | 4 | 6.477 | +42.13% |
| 512 | 1 | 7.814 | 16 | 6.554 | -16.13% |
| 512 | 8 | 8.013 | 16 | 7.168 | -10.54% |
| 512 | 32 | 11.185 | 16 | 11.878 | +6.20% |
| 4096 | 1 | 45.517 | 32 | 12.646 | -72.22% |
| 4096 | 8 | 46.566 | 32 | 20.122 | -56.79% |
| 4096 | 32 | 74.906 | 16 | 50.381 | -32.74% |

## Numerical and lifecycle evidence

120 combinations cover capacities 1/17/129/257/1024/4096, active counts 0/1/3/32/33 and split counts 1/2/4/32. Rows have differing lengths. Poisoned scratch exercises empty partitions; two graph replays, inactive sentinels, invalid launch/device extents, finite active outputs and exact one-token selection passed. The probe compares three dimensions for each of three heads per active row with independent CPU FP64 softmax/P×V. The maximum sampled absolute error is 0.00164395566939, including final BF16 rounding. No numerical tolerance was chosen after this result, and this number is not a model gate pass.

Across repeated shape/split combinations, 144,944 BF16 outputs differ from the existing implementation. This counts comparison events, not unique tokens or model predictions. The implementation belongs to a separate experimental numerical profile; existing exact compatibility is not claimed.

Bounded memcheck and racecheck each ran 64 combinations (maximum context 257, active 3 plus invalid active 33): 0 errors and 0 hazards/0 warnings respectively. The larger native cases were executed without sanitizer. SM89, SM90a and SM100a builds passed; only SM89 ran. Hopper/Blackwell runtime tests are skipped because that hardware is absent.

All three recorded Blender processes were restored and their scene RPCs succeeded. No viewer process was stopped. Archive verification checked regular members, five local/remote source hashes, three build receipts, all execution exits, sanitizer summaries and restoration. The archive includes the lifecycle controller, native/timing logs, build logs and binary digest; raw profiler data is not involved.

## Decision and remaining work

Keep this as a numerical control and long-context integration candidate. Preserve the existing short-context path: its measured launch/merge overhead dominates at context 128, and even context 512 regresses at active 32. Do not hard-code the retrospective best split counts from this single sweep. Next batch must include graph/scratch ownership, explicit profile identity and full-model teacher forcing/generation/cache invariance, followed by matched long-context serving against current Riley and vLLM. The existing immutable NLL/KL and strict generation gates remain required.

No new vLLM comparison is presented because this is not a serving run. The overall throughput/latency objective is still unfulfilled. See [all timings and verification](verification.json), [evidence archive](evidence.tar.gz) and [fixed contract](../../../deploy/260913/06-attention-split-numerical-contract.md).
