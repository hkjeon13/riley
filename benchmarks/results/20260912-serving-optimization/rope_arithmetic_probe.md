# RoPE arithmetic experiment (local preparation; GPU execution pending)

This experiment tests whether either explicit FP32 multiply/FMA order, compiled with the accepted attention translation unit's `--use_fast_math` flags, matches the accepted packed decode RoPE/KV operation bit for bit. It does not implement fusion, qualify a fused kernel, or measure performance.

The oracle is `enqueue_compiled_packed_decode_rope_kv` from the complete, unchanged `graph_numerics_precise.cu` in accepted batch7 commit `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`. The generic primitive is not the oracle: its intermediate BF16 rounding contract differs. The driver requires the frozen batch7 build receipt SHA `4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395`, source and serving-binary pins, a clean snapshot, and the exact generated CMake rules. Existing batch6/batch7 probe modules are imported for their precise/attention flag parsers without modifying them.

Copy these five files together to a new tools directory:

- `rope_arithmetic_probe.py`
- `rope_arithmetic_probe.cu`
- `rope_arithmetic_candidates.cu`
- unchanged `batch6_projection_probe.py`
- unchanged `batch7_attention_probe.py`

Run on the measurement host only when its owner has confirmed the GPU is available. Each output directory is exclusive; a failed build/run keeps its evidence and is retried with a new prepared directory.

```sh
python3 rope_arithmetic_probe.py prepare \
  --build-receipt /tmp/riley-opt-260912/batch7-build.json \
  --output-dir /tmp/riley-opt-260912/rope-arithmetic-diagnostic

python3 rope_arithmetic_probe.py build \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-diagnostic/fixtures.json \
  --nvcc /usr/local/cuda-13.0/bin/nvcc

python3 rope_arithmetic_probe.py inspect \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-diagnostic/fixtures.json \
  --cuobjdump /usr/local/cuda-13.0/bin/cuobjdump

python3 rope_arithmetic_probe.py run \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-diagnostic/fixtures.json \
  --driver-library-dir /tmp/riley-opt-260912/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu

python3 rope_arithmetic_probe.py validate \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-diagnostic/fixtures.json
```

The CUDA tool paths above must refer to the compiler selected in the frozen CMake build. `prepare` and `validate` are CPU-only. `build` and `inspect` invoke compiler/binary utilities but execute no GPU kernels. Only `run` uses the GPU. It requires device 0/SM89/CUDA runtime 13000 and reports the device UUID. The optional driver directory pins `libcuda.so.1` and precedes the exact CMake-selected shared cudart directory in `LD_LIBRARY_PATH`; inherited `LD_PRELOAD` is rejected. Session/process/driver management belongs to the measurement owner and is absent from this helper.

## Raw fixtures and checks

There are 436 cases. Four phases each visit all 65,280 finite BF16 encodings exactly once across Q/K, with different affine permutations and half-pairings. Three additional phases cover equal halves, opposite signs, and mixed extremes at every position 128–159. Across all cases, V covers all 65,536 uint16 patterns, including NaN payloads, which must be copied unchanged. Identity, reverse and affine physical block maps cover all 96 position/map combinations.

All input files are packed little-endian integers. No host float operation or text conversion changes their bits. The 62 FP32 table patterns include both signed zeros, subnormals, normal/subnormal boundaries and BF16 rounding ties with values in [-1,1]. These are synthetic arithmetic stress tables, not checkpoint activations or the model's generated sin/cos table.

| Buffer | Layout |
| --- | --- |
| Q/K/V | 1920 bytes: Q[576] at 0, K[192] at 1152, V[192] at 1536, all BF16 words |
| Cos / sin | Each 160×32 FP32 words, 20,480 bytes |
| Metadata | 116 bytes: position byte 4, live blocks byte 12, physical IDs byte 16, valid counts byte 80, sequence slot byte 112 |
| Rotated Q | 576 BF16 words, 1152 bytes |
| K / V caches | Each 16 physical blocks×3 heads×16 tokens×64 dimensions, 98,304 bytes |

Every case allocates 13 buffers, each with 256-byte guards on both sides and a 256-byte-aligned payload. Q and active K output slots start with distinct NaN poisons. Unused K/V slots start with fixed finite sentinels. The harness compares all Q words and both entire caches for each candidate against the oracle, checks exact raw V scatter, checks untouched cache slots, checks every input and guard, rereads oracle outputs after candidates, and explicitly frees every allocation. Output infinities from finite extreme arithmetic are allowed; an unwritten NaN poison fails coverage. Complete execution requires 5,668 allocations freed, no live device bytes, no cleanup errors, and stream destruction.

`run/native.jsonl` contains a device record, 436 case records and one summary. Each candidate has Q/K/V mismatch counts and the first differing word/bits. Numerical mismatch is an expected possible result: exit 0 and `complete:true` mean the experiment completed; inspect `candidate_equal`. Guard, input, coverage, CUDA or cleanup failures return nonzero. `fusion_qualified` and `performance_claim_eligible` always remain false. Final validation rechecks fixture bytes, tool/source/build hashes, exact compile commands, runtime selection, raw record coverage, mismatch aggregation and cleanup.

## Inspecting the actual contraction order

`inspect` writes full SASS dumps and selected RoPE function blocks for both frozen serving executables, the separately compiled full precise oracle object and the candidate object. Those are actual binary instructions. It also emits separate PTX files regenerated from the pinned sources, replacing only the validated SM89 code-generation option with `--gpu-architecture=compute_89` for PTX emission. The receipt labels this distinction and records every command/tool/artifact hash.

Inspect each first/second rotation expression independently. Trace BF16 table conversion and widening, product operands/signs, `FMUL`/`FFMA` or PTX `mul`/`fma` order, final BF16 rounding, and any denormal-flush modifiers. The two candidates explicitly use `mul.rn.f32` and `fma.rn.f32` without `.ftz`, raw BF16 widening and raw sign-bit negation. Candidate0 rounds the sine product and fuses the cosine product; candidate1 rounds the cosine product and fuses the sine product. A compiler may choose different contraction orders for the two output expressions, so neither whole candidate is assumed to match.

Even if one candidate matches all fixtures and its instruction chain matches the oracle, a future fused attention implementation still needs actual-model RoPE/table inputs, full tensor/KV parity and serving measurement under its own source identity. This experiment supplies arithmetic evidence only.

Local preparation checks completed: fixture coverage and bitspace enumeration; exact CMake rule reconstruction and rejection of changed arithmetic flags; host harness compilation using CUDA API doubles with `clang++ -Wall -Wextra -Werror`; 436 mock cases; signed-zero mismatch reporting; guard/input/allocation-failure cleanup. These checks do not prove CUDA compilation or GPU arithmetic.
