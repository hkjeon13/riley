# Fused decode softmax reuse: rejected native batch

SM89 compilation succeeded. Widths 2, 4 and 8 each pass 208 bitwise output/graph-replay comparisons and bounded memcheck with zero errors. All six lifecycle steps exit zero and Blender restoration succeeds. This is synthetic native evidence, not model or serving qualification.

| Width | Context | Rows | Shared pages | Prior us | Candidate us |
|---:|---:|---:|---|---:|---:|
| 2 | 512 | 1 | 0 | 10.286 | 14.408 |
| 2 | 512 | 1 | 1 | 10.291 | 14.423 |
| 2 | 512 | 8 | 0 | 10.778 | 14.792 |
| 2 | 512 | 8 | 1 | 10.803 | 14.756 |
| 2 | 512 | 16 | 0 | 11.648 | 15.555 |
| 2 | 512 | 16 | 1 | 11.945 | 15.478 |
| 2 | 512 | 32 | 0 | 16.292 | 17.669 |
| 2 | 512 | 32 | 1 | 16.394 | 17.930 |
| 2 | 4096 | 1 | 0 | 54.630 | 92.780 |
| 2 | 4096 | 1 | 1 | 54.630 | 92.687 |
| 2 | 4096 | 8 | 0 | 56.612 | 94.131 |
| 2 | 4096 | 8 | 1 | 56.525 | 94.126 |
| 2 | 4096 | 16 | 0 | 59.197 | 96.568 |
| 2 | 4096 | 16 | 1 | 57.370 | 92.687 |
| 2 | 4096 | 32 | 0 | 95.002 | 105.989 |
| 2 | 4096 | 32 | 1 | 97.587 | 107.428 |
| 4 | 512 | 1 | 0 | 9.795 | 16.097 |
| 4 | 512 | 1 | 1 | 9.815 | 16.092 |
| 4 | 512 | 8 | 0 | 10.271 | 16.440 |
| 4 | 512 | 8 | 1 | 10.327 | 16.451 |
| 4 | 512 | 16 | 0 | 11.100 | 17.172 |
| 4 | 512 | 16 | 1 | 11.387 | 17.142 |
| 4 | 512 | 32 | 0 | 15.529 | 19.057 |
| 4 | 512 | 32 | 1 | 15.590 | 19.702 |
| 4 | 4096 | 1 | 0 | 52.014 | 107.412 |
| 4 | 4096 | 1 | 1 | 51.978 | 107.930 |
| 4 | 4096 | 8 | 0 | 54.006 | 109.507 |
| 4 | 4096 | 8 | 1 | 53.929 | 109.276 |
| 4 | 4096 | 16 | 0 | 57.713 | 112.850 |
| 4 | 4096 | 16 | 1 | 57.431 | 112.317 |
| 4 | 4096 | 32 | 0 | 95.048 | 125.199 |
| 4 | 4096 | 32 | 1 | 97.249 | 126.484 |
| 8 | 512 | 1 | 0 | 9.810 | 20.511 |
| 8 | 512 | 1 | 1 | 9.795 | 20.485 |
| 8 | 512 | 8 | 0 | 10.266 | 20.854 |
| 8 | 512 | 8 | 1 | 10.322 | 20.833 |
| 8 | 512 | 16 | 0 | 11.116 | 21.642 |
| 8 | 512 | 16 | 1 | 11.377 | 21.571 |
| 8 | 512 | 32 | 0 | 15.514 | 23.188 |
| 8 | 512 | 32 | 1 | 15.611 | 23.496 |
| 8 | 4096 | 1 | 0 | 51.999 | 143.386 |
| 8 | 4096 | 1 | 1 | 52.024 | 143.442 |
| 8 | 4096 | 8 | 0 | 54.021 | 144.927 |
| 8 | 4096 | 8 | 1 | 53.939 | 144.855 |
| 8 | 4096 | 16 | 0 | 57.825 | 148.485 |
| 8 | 4096 | 16 | 1 | 57.503 | 148.296 |
| 8 | 4096 | 32 | 0 | 95.181 | 158.930 |
| 8 | 4096 | 32 | 1 | 97.367 | 160.369 |

Two reversed run orders; medians are per-variant and must not be compared across width runs as a controlled timing pair. Timed inputs retain scale=32 from correctness checks. Full captured QK+PV execution is measured.

Decision: reject all three variants; keep production unchanged. Reuse reduces redundant exponentials but also reduces output-block parallelism and adds live accumulator/value fragments. The results rule out extra kernel launch as the sole explanation for the earlier split-path regression. They do not independently prove register pressure or occupancy as the dominant cause. Stop tuning this family; return to cross-request KV reuse or a different measured execution bottleneck.

No vLLM serving comparison was run for these failed native candidates.
