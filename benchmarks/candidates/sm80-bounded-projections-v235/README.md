# SM80 bounded projection candidate (private v235)

This patch is preserved for review and continued qualification. It is **not applied to the runtime in this commit**. It applies to the actual164 source manifest saved under `benchmarks/evidence/20261019-optimization-checkpoint/`.

From the repository root, `git apply --unidiff-zero --check benchmarks/candidates/sm80-bounded-projections-v235/candidate.patch` checks applicability. Applying it adds explicit mode6 (`bounded-projections-sm80-v1`, alias `bounded-gateup-sm80-v1`) while retaining existing modes and defaults.

The candidate restricts SM80/108SM, CUDA runtime12040, cuBLAS/Lt120405 and the qualified workspace/MPS environment. Gate and Up use separate GEMMs with at most16 request rows per call. Qwen3-4B merged QKV N6144/K2560 uses a cold-prepared, zero-workspace Lt6/tile18/stages16/split1/reduction0 matrix recipe for2..32 rows; R1 retains the original call. Immutable layouts, stream/plan ownership and completion semantics are retained. SM89 qualification does not authorize this path.

Gate/Up-only v225 passed both models'104 fresh same-A100 HF full-vocabulary rows and all32 actual Graph row keys. QKV recipe v234 passed component comparisons for rows1..32 on original, zero and permuted inputs. The v235 CUDA build and518 CPU regressions passed. Qwen3 whole-model C1/all-row Graph helpers completed, but the independent actual-QKV-call audit failed to find inspection records. Whole-model QKV qualification is incomplete; Mistral new-candidate checks and actual heterogeneous API/performance comparisons remain pending at this pause. Call-count reductions do not establish throughput improvements.

Raw finite receipts, failed results, binaries and diagnostics remain in the local ignored `artifacts/multimodel-parity-20261017/` and the existing A100 experiment directory. They are not deleted or treated as release evidence. See `docs/optimization-checkpoint-20261019.md` for the paused state.
