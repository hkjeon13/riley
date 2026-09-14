# Request-pair regression attribution

Nsight Systems trace: context 512, 32 rows, shared-prefix mode 1. Native checks for scale 1 and 32 plus graph replay passed. Trace includes correctness, warmup and timed launches; means below are not steady-state-only or serving latency. Original has 222 launches per kernel, pair has 224 due to candidate-only replay checks. Raw Nsight and SQLite remain remote-private under /data/riley-serving-260913-recovery/decode-request-pair-profile-evidence-v1.

| Kernel | Mean us | Registers/thread | Static shared B | Local B/thread | Theoretical resident CTAs/SM |
|---|---:|---:|---:|---:|---:|
| Original QK | 4.924 | 39 | 0 | 0 | 24 |
| Pair QK | 4.966 | 40 | 0 | 0 | 24 |
| Original PV | 10.916 | 40 | 2304 | 0 | 16 |
| Pair PV | 23.511 | 72 | 3072 | 0 | 9 |

The regression is concentrated in PV, while paired QK does not provide a useful gain in this case. Occupancy API values are theoretical limits, not measured achieved occupancy. No local-memory allocation is reported. More registers, serial per-request normalization, additional shuffles and reduced grid parallelism are consistent with the slowdown, but this trace does not isolate each causal contribution. Do not label it an HBM bottleneck without memory counters.

Decision: stop this paired-warp design. Any further shared-KV backend should separate independent request normalization from shared matrix work without retaining both states in every consumer lane; it must first demonstrate whole-attention gain with grouping overhead. Existing serving remains unchanged. Profiling lifecycle succeeded and all three Blender processes were restored.
