# M2/M4 projection arithmetic probe

This is a local-prepared, separately executed numerical feasibility experiment.
It makes no throughput, latency, full-model multisequence, scheduler, KV ownership,
or serving qualification claim. Only the `run` command launches GPU work. There
are no production source changes, remote controls, server launches, or Blender
controls in these helpers.

`multisequence_projection_probe.py` uses the frozen batch8 child V2 prerequisite
validator to reconstruct the exact batch8 build, full-model logits/KV proof and
independent fused RoPE/attention proof. It also checks the accepted precise TU
hash. That validator preserves private-runtime library aliases correctly. The
new probe uses batch8 source `8329c1aeec6e013f581128888c536e15f8bf7300` and serving
binary `4c0876db…60c741`; it does not run the imported HTTP helper or depend on a
new HTTP receipt.

## Coverage and arithmetic gates

There are 1,089 cases: 121 actual checkpoint weight groups × 3 deterministic
input patterns × `(M2, active2)`, `(M4, active4)`, `(M4, active3)`.

| Group | Count | N | K | M1 custom option |
|---|---:|---:|---:|---:|
| QKV concatenated in that order | 30 | 960 | 576 | 74 |
| Gate/up concatenated in that order | 30 | 3072 | 576 | 74 |
| O | 30 | 576 | 576 | 74 |
| Down | 30 | 576 | 1536 | 75 |
| Final head, following model config's actual tied binding | 1 | 49152 | 576 | 89 |

The fixture driver copies source BF16 bytes without conversion. Every copied
piece has an exact source tensor name, shape, byte range and hash; validation
reconstructs all packed bytes from the checkpoint. Tied heads use the embedding
tensor, matching the model loader; any optional physical head must be identical.
Input patterns have distinct row fingerprints, cancellation, signed zeros and
raw BF16 subnormals. M4/active3's last row is all positive-zero bits. All active
and inactive output words are compared.

The host C++ harness links the existing production native static archive. It
prepares five strict M1 anchors under the same 16 MiB preparation cap and checks
every field of their recorded identity: algorithm13, the custom option above,
backend1, tile0/stages0/swizzle0, splitK1, reduction0, deterministic1,
flags131585, actual workspace0, SM89, CUDA runtime13000, cuBLASLt130101. The two
child descriptors for each group use only the public anchored-plan API, with a
zero workspace cap. They cannot request a shape-specific heuristic fallback.

A child `NOT_SUPPORTED` admission is a completed experimental observation. Its
cases are explicitly skipped, and `arithmetic_equal` is false. Other admission
errors or metadata drift fail the run. Admission alone never establishes
numerical equality. Numerical mismatches also remain completed observations,
with each row's full comparison count and first differing BF16 words retained.
No tolerance or signed-zero normalization is applied.

Each batched input/output and each M1 oracle row input/output has its own
allocation with an aligned 256-byte subspan and 256-byte guards on both sides.
The immutable weight allocation is shared by the two algorithms. Each output is
poisoned with BF16 NaNs before execution. The harness checks every output,
input/weight immutability, all guards and finite results, then closes each case's
buffers. Plans, stream, pinned staging and context close on success and errors;
ambiguous native failures remain failed observations rather than being declared
clean. Public allocation statistics prove unchanged live counts/bytes across
execution and zero live storage at final close; they do not expose allocation
attempt counts or prove hidden cuBLAS allocator behavior.

## Build provenance and execution

The driver reads the frozen production CMake cache/compile options and generated
shared-library path files, verifies that built and installed archives match, and
compares every archive member against its production object. It newly pins these
objects, archive, compiler, source/header build rules and shared libraries. The
old full-model receipt did not hash the archive, which is stated explicitly in
the new compile receipt. The harness contains no GPU arithmetic and does not
recompile any production translation unit. Its host compiler is independently
pinned. Run receipts verify the actual GPU UUID and loaded private libcuda,
cuBLASLt and cudart maps at both ends, plus the existing private-runtime receipt,
file/symlink inventory and loaded kernel version.
The native build string must exactly identify ABI1 and nvcc13.3.73, as in the
frozen production compiler receipt. This string does not contain F108/g12;
those identities are checked separately in the full batch8 qualification.

Root owns execution after the current serving experiment is finished. Example
commands on that authorized host (not executed during local preparation):

```sh
PY=/data/riley-vllm-interim.CfrT9T/venv/bin/python
ROOT=/tmp/riley-opt-260912
$PY "$ROOT/multisequence_projection_probe.py" prepare \
  --root "$ROOT" --base /tmp/riley-g04-vllm-profile-260911 \
  --output-dir "$ROOT/multisequence-projection-probe"
$PY "$ROOT/multisequence_projection_probe.py" build \
  --manifest "$ROOT/multisequence-projection-probe/fixtures.json" \
  --cxx /usr/bin/c++ --nvcc /data/riley-g04-cuda13/bin/nvcc
$PY "$ROOT/multisequence_projection_probe.py" run \
  --manifest "$ROOT/multisequence-projection-probe/fixtures.json" --timeout 1800
$PY "$ROOT/multisequence_projection_probe.py" validate \
  --receipt "$ROOT/multisequence-projection-probe/run/receipt.json"
```

Every output directory/file is created exclusively. Repeated prepare/build/run
commands require a fresh experiment directory; they do not overwrite evidence.
`validate` is read-only and launches no CUDA work. Native stdout is JSONL with
one device record, 15 plan records, 1,089 case records, final library maps and a
summary. Complete numerical equality is a prerequisite for investigating true
batched row kernels, graph/KV ownership and full-model parity, not their proof.

Local tests include all metadata fields, complete and rejected-plan record
matrices, signed-zero mismatch preservation, missing/duplicate cases, inactive
row coverage, exact fixture bits/head binding, bounds/nonfinite rejection and
cleanup/guard failures. Host syntax uses the real public ABI with only the CUDA
UUID declaration stubbed; this is not CUDA linkage or numerical validation.
