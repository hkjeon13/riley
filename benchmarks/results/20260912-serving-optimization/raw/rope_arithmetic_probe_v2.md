# RoPE arithmetic probe v2

This is an isolated numerical diagnosis. It changes no production source and
does not qualify fusion or measure serving performance. The v1 helpers,
fixtures, executable and receipts remain unchanged.

The actual v1 oracle and both accepted production binaries have identical
selected RoPE SASS (`4ce1a5725a741475c69c599699059b3ae8dff2ef7c10e74c6421a7aabed8c433`).
Tracing its registers gives `a=R12`, `b=R13`, `c=R4`, `s=R7`:

| Address | Operation | Meaning |
|---|---|---|
| `0x02a0` | `FMUL R15,R7,R13` | Round `s*b` |
| `0x02b0` | `FMUL R6,R4,R13` | Round `c*b` |
| `0x02c0` | `FFMA R15,R4,R12,-R15` | First result: `fma(a,c,-mul(b,s))` |
| `0x02d0` | `FFMA R6,R7,R12,R6` | Second result: `fma(a,s,mul(b,c))` |

V1 candidate0 used the correct first expression but rounded `a*s` in the
second. Candidate1 used the correct second expression but rounded `a*c` in the
first. Neither reproduced the complete pair. The displayed `RZ` in
`F2FP.BF16.F32.PACK_AB R4,RZ,R6` is a zero source register; it is not a
round-toward-zero modifier. Both v1 candidates already use packed final output
conversion. Table narrowing remains a separate scalar-versus-packed codegen
difference.

V1 completed 436 cases with 67/75 differing words in 30/31 cases. All recorded
first differences were oracle `0x0000` versus candidate `0x8000`. A local CPU
float32/`fmaf` calculation reproduced every per-case region count and every
stored first difference; its inferred complete histograms were 67/75 such
zero-sign differences. This inference is not proof of all v1 GPU words because
the v1 log retained only the first bit pair per case.

V2 compares two candidates with the same corrected mixed FMA pair:

| Candidate | Table conversion | Arithmetic |
|---|---|---|
| 0 | Unchanged scalar `cvt.rn.bf16.f32` | Explicit non-FTZ `mul.rn.f32` and `fma.rn.f32` |
| 1 | `cvt.rn.bf16x2.f32`, high input +0, low input table value | Identical to candidate0 |

The packed conversion places its second source in the low half; shifting that
result left by 16 restores the BF16 value as raw FP32 bits. No zero-sign
normalization, FTZ, saturation or tolerance is introduced. The operand layout
follows the [NVIDIA PTX conversion specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt).

All 436 Q/K/V, cosine/sine and metadata fixtures are byte-identical to v1,
including finite BF16 encodings, signed zeros, subnormals, exponent extremes,
FP32 table rounding ties and positions128–159. They are synthetic inputs, not
checkpoint activations or an actual model trigonometric table. Full Q and both
physical KV caches are compared, with the same guards, poison checks,
immutability checks and 5,668 allocation/free accounting. Every mismatching
word contributes to an exact bit-pair histogram. Per-case and aggregate
histograms must reconcile; signed-zero differences remain numerical failures.

The versioned driver retains the complete frozen precise TU as oracle and the
actual attention-TU flags, including `--use_fast_math`, for candidates. It pins
the accepted build, compiler, shared CUDA runtime, helpers, generated fixtures,
objects, executable and raw results. It additionally pins the v1 GPU receipt,
raw result, successful inspection receipt and selected SASS/PTX that motivated
the change. The v2 native/fixture/build/run schemas use suffix `.v2`.

Run only through the campaign controller, which supplies the verified private
driver environment and retains the three Blender sessions:

```sh
python3 /tmp/riley-opt-260912/rope_arithmetic_probe_v2.py prepare \
  --build-receipt /tmp/riley-opt-260912/batch7-build.json \
  --output-dir /tmp/riley-opt-260912/rope-arithmetic-probe-v2
python3 /tmp/riley-opt-260912/rope_arithmetic_probe_v2.py build \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-probe-v2/fixtures.json \
  --nvcc /data/riley-g04-cuda13/bin/nvcc
python3 /tmp/riley-opt-260912/rope_arithmetic_probe_v2.py run \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-probe-v2/fixtures.json \
  --driver-library-dir /tmp/riley-opt-260912/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu
python3 /tmp/riley-opt-260912/rope_arithmetic_probe_v2.py validate \
  --manifest /tmp/riley-opt-260912/rope-arithmetic-probe-v2/fixtures.json
```

`inspect` accepts the same manifest and `--cuobjdump` as v1. Include
`/data/riley-g04-cuda13/bin` in the child PATH so cuobjdump finds nvdisasm. Actual
binary SASS and separately regenerated PTX are labelled distinctly.

Local syntax and five CPU contract tests passed: complete fixture parity,
histogram coverage/tamper rejection, signed-zero classification without
equality weakening, actual v1 basis pins, and precise/fast-math/shared-runtime
command preservation. No v2 CUDA compilation or GPU execution was performed
during preparation; candidate equality remains unmeasured.
