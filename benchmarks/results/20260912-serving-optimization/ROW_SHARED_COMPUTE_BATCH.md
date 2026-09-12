# Shared-weight decode compute batch: evidence and integration gates

Status: native candidate only; frozen V9 remains the serving optimization base. The original exact M1 numerical profile and tests remain intact.

## Evidence

The V9 actual-serving trace attributes about 78.7% of decode kernel time to GEMV. A dense [N,K] × [K,B] multiplication can share weights across B requests; this changes floating-point accumulation and must be qualified independently.

Fixtures use the pinned SmolLM2 weights and last-token projection inputs from CPU FP32 HuggingFace eager execution of eight synthetic held-out token streams of lengths 16/32/64/128/192/256/384/512. Inputs are then rounded to BF16. All30layers' QKV, gate/up, O, down plus head yield121operators and1,637,376outputs. These are realistic activation magnitudes from synthetic inputs, not a representative serving dataset, nor actual Riley cache-state activations.

Independent CPU FP64 dot products compare the current strided GEMV configuration and shared-row cuBLASLt algorithm21. The primitive screening bound was set before the runs: |Y - dot64| <= 2*K*2^-24*sum(|w*x|) + |dot64|/256 + 1e-30. This conservative finite-input accumulation/rounding screen is not a model-level quality acceptance threshold. It does not establish exact BF16 equivalence, exceptional-input behavior, general numerical stability or generation quality.

Default heuristic produced1447bound violations and56695BF16 differences, concentrated in down projection. Explicit split-K1/reduction0 removed all observed bound violations; workspace became0. There are590BF16 differences:53closer and537farther from FP64 than the original. Worst relativeL2 per shape remains about0.0017–0.0019, but rounded summary equality is not exact error equality. On head, both implementations match the FP64 argmax in7/8rows; that count alone does not prove they choose the same token or explain the remaining row.

Nonexclusive repeated graph timings (hot weights, not serving acceptance), median across30layers where applicable:

| Projection | Original N8 us | Shared rows us | Decision |
|---|---:|---:|---|
| QKV |8.16|4.27|Integrate as candidate|
| gate/up |21.61|4.71|Integrate as candidate|
| O |5.53|4.23|Defer to keep first batch focused|
| down |7.25|7.54|Keep original|
| head |262.3|21.6|Integrate as candidate; whole-model cache history must be measured|

## Coupled implementation scope

1. Add an explicitly distinguished shared-row GEMM plan for QKV and gate/up; retain ownership, compact/padded stride validation, cold algorithm preparation, workspace0 and no split-K/reduction guarantees.
2. Add shared-row head execution with the same explicit numerical plan identity; preserve compact greedy output and existing nonfinite detection/tie handling.
3. Bind shared-row plans through a separate numerical execution profile and catalog fingerprint. Do not disguise B>1 GEMM as the exact strided-M1 implementation or alter its metadata/tests. Support active/bucket masking and B1/2/4/8 behavior with separate qualification.

## Required qualification before serving adoption

- Test row independence, stride/padding preservation, incomplete buckets, invalid binding, graph mode switching, shutdown/partial capture cleanup and cancellation/KV reuse. Check the selected algorithm metadata, rather than assuming heuristic output stays constant.
- Establish held-out full-model teacher-forced comparison to CPU/FP32 reference before evaluating candidate timings: logits error distribution, top1 margins, probability divergence, accumulated KV/state error and sequence-level loss. Report original and candidate against the same reference. Decide and record a model-level acceptance policy before viewing candidate performance; primitive bound alone is insufficient.
- Diagnose any head/top-token disagreement using the common input and high-precision scores, retaining failures. Do not silently mark exact-reference mismatches as passes.
- Profile actual model execution. Previous head hot-cache prediction failed in V8; do not repeat that inference.
- Benchmark original/current candidate and vLLM under matched model/hardware/workload, reporting numerical qualification separately. Variable prompt lengths, high concurrency and long-run tails remain required; this native fixture does not fulfill those serving gates.

Artifacts: raw/projection-accuracy-v1/{manifest.json,cases.tsv,results.jsonl,results-v2.jsonl,results-v3.jsonl}. Full fixture binaries remain remotely under /tmp/riley-opt-260912/projection-accuracy-v1 and are individually SHA256-pinned in manifest.json. Reproduction scripts are adjacent to this document.
