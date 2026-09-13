# Same-input prefill precision screen

The preceding prefill-only full-model candidate failed strict greedy equivalence and the fixed natural-language KL gate. This screen tests whether changing attention arithmetic precision is worth a full-model experiment. It does **not** identify the cause of that model-level failure or establish serving performance.

## Controlled comparison

The offline generator `benchmarks/analysis/prefill_precision_screen.py` compiles three native probes:

1. Existing V7 `riley_mixed_attention::attention_body<8>` from the actual repository header (including its small-query path).
2. Pinned FlashInfer BF16 prefill with the verified warp-barrier header overlay.
3. The same FlashInfer template instantiated with FP16 Q/K/V/output; input values are rounded to BF16 **before** conversion to FP16, and output values are rounded back to BF16 before scoring.

The final lane is an isolated dtype experiment, not a serving ABI. Its raw output dump contains FP16 bit patterns whose values have been rounded to BF16; the manifest labels this explicitly. Do not load that dump as BF16 or attach that adapter to the existing BF16 model buffers.

All three lanes receive numerically identical Q/K/V. Each writes the input values as float32 and the generator verifies bitwise equality by SHA256 for every scale. No input quantization difference is hidden in the comparison. FP64 dot product, softmax and value accumulation form the same CPU oracle for every lane.

Fixed seed and ragged cases are inherited from the native probe: Q lengths [1,17,33], [16,32], [127,128,129], and 32 requests totaling 1,024 queries; KV context up to 4,096, noncontiguous pages, invalid suffix and graph replay. Each lane evaluates 868,032 output values per scale. Q and K are multiplied by 0.125, 1 or 4; V retains the original [-0.25,0.25] distribution. These are synthetic inputs, **not captured model activations**.

## Results: weighted RMSE against FP64

| Q/K scale | Existing V7 | FlashInfer BF16 | FP16 computation → BF16 output | Change vs FI BF16 |
|---|---:|---:|---:|---:|
| 0.125 | 2.42045e-5 | 2.39778e-5 | 2.03862e-5 | −14.98% |
| 1 | 2.88054e-5 | 2.85481e-5 | 2.15791e-5 | −24.41% |
| 4 | 1.66557e-4 | 1.65473e-4 | 1.63991e-4 | −0.90% |

This is an **error reduction**, not a speedup. Both existing and FlashInfer BF16 kernels already round attention probabilities to BF16 before P×V. The dtype experiment changes more than that one operation, so these results cannot attribute the entire difference to probability rounding alone.

The unchanged 0.01 synthetic maximum-absolute-error bound passes for all nine lane/scale combinations. The final FP16 scale-4 probe passes native memcheck (0 errors) and racecheck (0 hazards, 0 errors, 0 warnings); sanitizer output dumps match the normal run bitwise. The FI BF16 scale-1 output remains bitwise identical to the previously verified primitive dump. These are primitive sanitizer results, not full-model FP16 results.

## Reproducibility and limits

Final run: RTX 4090/SM89, nvcc 13.0.88. `comparison.json` includes build commands, source/binary/input/output hashes, overlay provenance and per-case error metrics. `verification.json` includes the repository header/generator hashes and sanitizer checks. Generated sources, executables and raw inputs/outputs remain at `/tmp/riley-opt-260912/prefill-precision-v3`. Earlier v1/v2 runs were initial single-scale runs; the committed comparison is the completed three-scale run.

No production kernel, runtime profile or default changed. No new Hopper/Blackwell build or runtime result is claimed. The preceding full-model greedy/KL failures remain failures. No throughput, TTFT, TPOT or vLLM comparison was performed here.

## Next optimization batch

The evidence supports evaluating a native precision variant, with these requirements kept together:

- Preserve BF16 persistent KV storage. Investigate explicit conversion at kernel load/compute boundaries rather than copying the entire cache on every iteration. Upstream mixed dtype behavior must be verified from the instantiated code, not inferred from template parameter names.
- Define and enforce FP16 representability, including overflow and subnormal loss; the synthetic inputs above do not test the full BF16 range. Do not reinterpret BF16 bits as FP16 or silently change the model's storage contract.
- Keep a separate experimental graph identity and Rust/native lifetime ownership, and include all conversion/temporary-memory costs in later measurements.
- Re-run the unchanged full-model quality gates before acceptance, then compare actual serving with existing Riley and vLLM. If the quality/cost tradeoff fails, retain the evidence and move to another attention/backend design rather than optimizing this synthetic probe further.
