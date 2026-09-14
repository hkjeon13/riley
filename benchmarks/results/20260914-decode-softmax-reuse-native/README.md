# Decode softmax reuse: rejected native screen

SM89 build succeeded. Native oracle: 208 cases passed bitwise output and graph replay comparison; bounded memcheck passed with zero errors. Lifecycle exited successfully and restored all three Blender processes. This is not model/serving correctness qualification.

The experimental three-kernel path removes repeated softmax preparation across eight output blocks but adds a launch and materialized probability/alpha/inverse scratch. Two reversed timing orders measure the complete captured QK + attention path. Timings use the scale=32 input remaining after correctness checks; they are synthetic, not model-distribution measurements.

| Context | Rows | Page sharing | Prior us | Candidate us |
|---|---:|---|---:|---:|
| 512 | 1 | no | 10.276 | 12.452 |
| 512 | 32 | no | 16.276 | 17.669 |
| 512 | 32 | yes | 16.379 | 18.678 |
| 4096 | 32 | no | 97.505 | 97.423 |
| 4096 | 32 | yes | 97.633 | 100.941 |

Decision: do not integrate into serving. Most shapes regress; the near-flat long-context case is not a useful gain. The experiment does not establish whether launch overhead, scratch traffic, or serialized preparation dominates; instruction-level attribution requires separate profiling. Further work should explore reuse within a fused execution structure rather than promote this split merely because it removes repeated arithmetic.

The prototype remains optional and unreachable from production dispatch. Runtime Python is not introduced. No model-level validation or vLLM serving comparison was run for this failed candidate.
