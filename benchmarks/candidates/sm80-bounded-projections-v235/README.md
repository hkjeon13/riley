# SM80 bounded projection candidate (private v235)

This patch is preserved for review and continued qualification. It is **not applied to the runtime in this commit**. It applies to the actual164 source manifest saved under `benchmarks/evidence/20261019-optimization-checkpoint/`.

From the repository root, `git apply --unidiff-zero --check benchmarks/candidates/sm80-bounded-projections-v235/candidate.patch` checks applicability. Applying it adds explicit mode6 (`bounded-projections-sm80-v1`, alias `bounded-gateup-sm80-v1`) while retaining existing modes and defaults.

The candidate restricts SM80/108SM, CUDA runtime12040, cuBLAS/Lt120405 and the qualified workspace/MPS environment. Gate and Up use separate GEMMs with at most16 request rows per call. Qwen3-4B merged QKV N6144/K2560 uses a cold-prepared, zero-workspace Lt6/tile18/stages16/split1/reduction0 matrix recipe for2..32 rows; R1 retains the original call. Immutable layouts, stream/plan ownership and completion semantics are retained. SM89 qualification does not authorize this path.

Gate/Up-only v225 passed both models'104 fresh same-A100 HF full-vocabulary rows and all32 actual Graph row keys. QKV recipe v234 passed component comparisons for rows1..32 on original, zero and permuted inputs. The v235 CUDA build and518 CPU regressions passed. Qwen3 whole-model C1/all-row Graph helpers completed, but the independent actual-QKV-call audit failed to find inspection records. Whole-model QKV qualification is incomplete; Mistral new-candidate checks and actual heterogeneous API/performance comparisons remain pending at this pause. Call-count reductions do not establish throughput improvements.

Raw finite receipts, failed results, binaries and diagnostics remain in the local ignored `artifacts/multimodel-parity-20261017/` and the existing A100 experiment directory. They are not deleted or treated as release evidence. See `docs/optimization-checkpoint-20261019.md` for the paused state.

## 2026-10-19 execution-path correction

The subsequent W0 audit matched the current 935 source hashes to the retained
SM80 baseline and ran Qwen3-4B through its actual HTTP server on A100 GPU 2.
The server selected `owned-hf-separate-qkv-bitpack-v1`. Captured cuBLAS API
arguments in C1/C8 showed separate Q `(N=4096,K=2560)` and K/V
`(N=1024,K=2560)` calls, including M=1 decode. No merged
`(N=6144,K=2560)` call was observed in those runs. The owned runtime prepares
separate projections when query width differs from hidden width.

Consequently, the v234 merged-shape component pass does not prove that v235
accelerates the currently selected Qwen3-4B QKV route. The missing merged-QKV
inspection records are consistent with an execution-path mismatch; they are
not evidence of a numerical mismatch. Do not bypass the separate-QKV path
to make that candidate execute. Gate/Up-only v225 was compared separately after the user moved experiments
from occupied GPUs 2/3 to GPUs 6/7. Clean GPU7 screens demonstrated C8
throughput gains for both models and led to explicit opt-in integration.
Qwen latency regressions prevent declaring an all-latency release gate pass. The merged-QKV extension remains unapplied.
Fresh-binary serving qualification is tracked in the W0 report.

See [W0 execution report](../../../docs/a100-execution-path-and-bottleneck-20261019.md)
and `benchmarks/evidence/20261019-w0-a100/` for compact receipts.
