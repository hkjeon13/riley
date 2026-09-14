# Request-pair QK and PV: rejected native prototype

Compilation for SM89 succeeded. All 390 native bitwise output and graph replay checks passed; bounded memcheck passed with zero errors. Blender restoration succeeded. This is not model or serving qualification.

Adjacent row pairs were supplied by the host outside the timed graph. Mode 0 uses original QK+PV; mode 1 uses paired QK+PV. Shared=0 means distinct per-row maps; shared=1 shares full prefix tiles with a distinct last tile; shared=2 also shortens the second row by up to 17 tokens. Tests include singleton/odd row counts and context boundary cases. Timing uses scale=32 synthetic Q after correctness checks. Group construction cost is excluded, so a regression here cannot become a serving win by adding grouping overhead.

| Context | Rows | Shared mode | Original us | Pair us |
|---:|---:|---:|---:|---:|
| 512 | 1 | 0 | 10.358 | 17.659 |
| 512 | 1 | 1 | 10.368 | 17.695 |
| 512 | 1 | 2 | 10.342 | 17.700 |
| 512 | 3 | 0 | 10.394 | 31.631 |
| 512 | 3 | 1 | 10.404 | 25.467 |
| 512 | 3 | 2 | 10.470 | 25.472 |
| 512 | 8 | 0 | 10.793 | 31.928 |
| 512 | 8 | 1 | 10.737 | 25.723 |
| 512 | 8 | 2 | 10.808 | 25.779 |
| 512 | 16 | 0 | 11.638 | 32.865 |
| 512 | 16 | 1 | 11.822 | 26.291 |
| 512 | 16 | 2 | 11.863 | 26.317 |
| 512 | 32 | 0 | 16.312 | 35.686 |
| 512 | 32 | 1 | 15.980 | 28.385 |
| 512 | 32 | 2 | 15.616 | 27.592 |
| 4096 | 1 | 0 | 51.953 | 113.142 |
| 4096 | 1 | 1 | 51.937 | 113.183 |
| 4096 | 1 | 2 | 51.932 | 113.275 |
| 4096 | 3 | 0 | 52.511 | 217.001 |
| 4096 | 3 | 1 | 52.495 | 156.867 |
| 4096 | 3 | 2 | 52.521 | 156.831 |
| 4096 | 8 | 0 | 53.760 | 218.644 |
| 4096 | 8 | 1 | 53.714 | 158.075 |
| 4096 | 8 | 2 | 53.847 | 158.060 |
| 4096 | 16 | 0 | 57.477 | 222.464 |
| 4096 | 16 | 1 | 57.431 | 160.010 |
| 4096 | 16 | 2 | 57.421 | 159.995 |
| 4096 | 32 | 0 | 94.909 | 243.548 |
| 4096 | 32 | 1 | 97.556 | 177.910 |
| 4096 | 32 | 2 | 97.690 | 178.171 |

Decision: reject this implementation; do not integrate or run a vLLM serving comparison. Reuse across requests does exist in the measured workload, but this pairing design does not exploit it efficiently. It serializes two softmax states within each warp, adds sharing decisions and accumulator lane shuffles, and changes CTA parallelism. These are candidate explanations, not individually profiled causes. No shared-prefix optimization is claimed. Before another variant, measure instruction/resource costs and reconsider warp scheduling rather than extend this prototype with further ad hoc branches.
