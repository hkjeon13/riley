# Isolated F8 history specialization candidate

This directory contains one source candidate for the measured F8 combined RoPE/KV/attention regression. It is not a production change or an accepted optimization. No CUDA compilation, GPU run, remote operation or performance measurement was performed while preparing it.

The [paired campaign](../ROUND15_PROFILE_RESULTS.md) measured combined-region medians of 360448/359936 ns for accepted separate RoPE+attention and 422400/422912 ns for F8. The roughly 62–63 us difference tracked the 60–66 us whole-decode difference. Those intervals justify examining this family; they do not attribute the regression to a particular instruction. Event instrumentation itself perturbed those measurements. The archived [comparison](../raw/paired-rope-attention-round15/comparison.json) is SHA256 `05be2a65ad78aa78f2e3346118c812d01e8172e82e1b094f43696ffaed1bc2bb`.

## Source and candidate

`baseline_graph_numerics.cu` is the complete frozen F8 translation unit, SHA256 `edb7990577187d9574bf6d7c751713581529e4da1b835d92ca4a012518de5552`. `graph_numerics.cu` is the full generated replacement. `candidate.patch` changes only that translation unit; no headers, resources, owners, allocations or public APIs change. `generate.py` rejects any other baseline. `invariants.json` records the generated source identity and preserved segments.

The source contains a shared runtime tile loop and repeated current-token selection/cache indexing in QK and PV. No locally saved PTX/SASS for the actual fused kernel was available in this preparation. The compiler may already hoist addresses, prove some branches or combine BF16 loads; source occurrence does not establish executed instruction cost. Existing RoPE lookup tables and all explicit RoPE arithmetic remain unchanged.

One batch contains three linked changes:

1. Two explicit `fused_history_tile<false/true>` calls specialize the tail `[128, position]` and then full history `[0,127]`. Only tail can select current local K/raw V or masked zeros. The validated position range makes exactly two tiles sufficient.
2. QK computes the key/cache base once outside its depth loop. Historical PV computes one physical page expression for its four strided V operands. The two independent output-block chains retain their original order; no new page array or reordered MMA schedule is introduced.
3. Existing local Q/K and probability arrays receive explicit four-byte alignment. Adjacent BF16 Q/K/probability operands load raw 32 bits. Global V stays a strided BF16 gather: adjacent token values are 64 BF16 elements apart.

The accepted Batch7 complete prefix remains byte-identical (SHA256 `a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e`). F8 RoPE helpers, initial Q/K/V publication, final denominator/output code and wrapper are preserved. The existing fast-math translation unit and separate precise oracle remain separate compilation units.

## Numerical and synchronization contract

| Property | Retained contract / CPU evidence |
|---|---|
| Launch geometry | 9 heads × 2 output halves, 64 threads; no GQA topology change |
| Current token | Rounded K is CTA-local, V is raw; no current-position global cache read |
| QK | Same token groups, four depth-16 MMA operations per score, unchanged `.125F` scaling |
| Softmax | Exact copied maximum/shuffle, exponentials, FP32 score reuse, per-lane MODE6 `j/z` denominator chain |
| PV | Same two output-block chains, alpha multiply, increasing depth-token MMA order; tail then history |
| Synchronization | Initial local-head barrier + three barriers per tile + final inverse barrier = eight; same two warp barriers |
| Publication | Q scratch576, K192 and V192 BF16 words written exactly once; all576 output words written once |
| Raw32 alignment | Shared arrays aligned4; all pair indices even; global K base/page/head/token strides and pair offsets divisible by4 bytes |
| Raw bits | CPU little-endian pair check includes every BF16 bit pattern, signed zero, subnormals and NaN payloads; no conversion is added |
| Bounds | CPU enumeration covers positions128–159, all lane/warp/head/output-half mappings and the probe's three physical-page maps |

The inline PTX uses the matching shared/global address space and `volatile` plus a `memory` clobber for indirect reads. NVIDIA documents natural alignment as mandatory for memory access width, and documents these inline-assembly memory/optimization constraints. These are language/ISA premises, not proof that this candidate compiles or improves performance. [PTX memory operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands), [inline PTX constraints](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#incorrect-optimization).

Code specialization can increase instruction footprint or register pressure. Explicit loads/clobbers can constrain optimizations. Inlining, register allocation and code generation can also affect floating-point execution despite preserved source chains. Actual exact-output qualification is mandatory; no tolerance or fallback is introduced.

## Local checks and existing GPU probe reuse

Reproduce source/address checks without CUDA or external packages:

```sh
python3 -m unittest discover -s benchmarks/results/20260912-serving-optimization/fusion-history-candidate-v1 -p 'test_*.py' -v
```

Regenerate into a fresh directory with `generate.py --output-dir NEW_DIRECTORY`; outputs use exclusive creation. CPU tests check exact source/hash segments, deterministic artifacts, addresses, pair bits, operation order and publication coverage. They do not emulate tensor-core arithmetic or device synchronization.

After the root agent explicitly selects a fresh isolated snapshot/build, the unchanged frozen `batch8_fusion_probe.py` supports this candidate's existing wrapper. It requires the actual new snapshot/build receipt and explicit source SHA; reusing the historical Batch8 build receipt would be invalid. Example placeholders below are intentionally not executable paths:

```sh
python3 benchmarks/results/20260912-serving-optimization/batch8_fusion_probe.py prepare \
  --source-root NEW_CANDIDATE_SNAPSHOT \
  --build-receipt NEW_CANDIDATE_BUILD_JSON \
  --oracle-source ACCEPTED_BATCH7_GRAPH_NUMERICS_CU \
  --candidate-entry enqueue_compiled_packed_decode_rope_attention \
  --candidate-source-sha256 CANDIDATE_SHA_FROM_INVARIANTS_JSON \
  --output-dir FRESH_FUSION_PROBE_DIRECTORY
```

Then use its existing `build --manifest ... --nvcc ...` and `run --manifest ... --driver-library-dir ...` commands. That probe derives production flags, including `--use_fast_math` on this full TU, and separately compiles the unchanged full precise TU. It covers8640 synthetic cases:30 independent layer seeds ×32 positions ×3 raw-bit patterns ×3 noncontiguous maps. The independent oracle is precise RoPE/KV followed by accepted Batch7 two-warp attention. It compares complete Q scratch, K/V storage and576 attention outputs with guards and input immutability. It is not actual model activation coverage.

Before interpreting timing, inspect baseline and candidate compiled kernel PTX/SASS from those exact artifacts for retained history predicates, generic-pointer handling, address instructions, load widths, register usage/spills and code size. Attribute only observed differences. Exact standalone proof, fresh full-model logits/KV and HTTP proof, then matched serving plus bracketing graph measurements remain required. No source or graph identity for a future engine integration is changed here.
