# True-row attention feasibility probe

This standalone experiment tests one component of the proposed multi-sequence
batch: independent-row attention over a shared physical KV pool. It changes no
production source, owner, scheduler, allocation ABI or serving configuration.
No CUDA compilation or GPU execution was performed while preparing these files.

The future batch still needs scheduler/admission, owner/metadata, the remaining
true-row GPU paths, and bulk result publication. The projection feasibility
probe is also a separate prerequisite. A passing attention experiment alone
does not qualify M2/M4 serving or establish a performance improvement.

## Exact accepted lineage

The baseline is accepted **Batch7/F107**, source commit
`1a2be0df01fe49daa4d4db155ad5c44f34ead6df`, build receipt SHA-256
`4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395`.
Its complete `kernels/src/graph_numerics.cu` is pinned to
`a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e`.
Fusion8 is rejected by the source pin.

`lineage/accepted_batch7_graph_numerics.cu` contains those exact accepted bytes,
recovered locally from the unchanged prefix preceding the fusion8 addition and
verified against the accepted build's source hash. It exists for CPU source
tests, not as a replacement build receipt. `prepare` requires the actual clean
Batch7 source snapshot, exact build receipt and its binary/source identities.

The unchanged accepted TU is compiled as the independent M1 oracle. `probe.py`
extracts its exact `exponential`, `pair`, `mma`, and `cache_index` helpers and
two-warp kernel into a second TU under namespace
`riley_multisequence_attention_probe`. The separate namespace prevents duplicate
host/device symbols. It preserves the original headers and copies the exact
accepted `ffi_internal.hpp` beside the generated TU. The original oracle TU is
compiled from the accepted source path without alteration.

`candidate_preview.cu` is the mechanically generated candidate for inspection;
CPU tests require it to equal the transform of the pinned lineage file. Runtime
`prepare` regenerates `candidate.cu` from the actual accepted snapshot and pins
the generated file and header in `fixtures.json`.

The candidate changes only row selection and an inactive-row gate:

- Grid `(9, 2, bucket)`, 64 threads per CTA; `blockIdx.z` selects the request row.
- The cold bucket fixes the grid; each replay reads its fresh active count from
  common header `packet[6]` (byte24). The wrapper has no active-count scalar,
  host readback, synchronization or graph-parameter update.
- Q and Y row strides are 576 BF16 words / 1,152 bytes.
- Each active row uses the validated C10 prefix at packet byte
  `128 + 128*row`; its position and physical block table are independent.
- An inactive CTA writes its own 576 outputs as positive zero through the
  existing 9-head × 2-half geometry and returns uniformly before row metadata/Q/KV
  loads or barriers. Warp0 writes 32 outputs per half; warp1 returns without
  writing. Only the common header is read before this gate. There are no KV
  writers in this attention-only experiment.

After removing the injected mapping setup and restoring the kernel name and
signature, the generator requires the accepted per-row kernel to be identical
byte-for-byte. QK depth/token order, the MODE6 denominator's j/z order, FP32
alpha/inverse publication, all barriers, PV MMA chains and BF16 rounding remain
unchanged. This is source evidence; only the future GPU comparison can establish
numeric equality after code generation.

## Fixtures and comparisons

The fixed 34,560 cases comprise 30 independent synthetic layer seeds × 32
position tuples × 3 BF16 bit-pattern families × 3 map permutations × 4 modes:
`(bucket, active) = (1,1), (2,2), (4,3), (4,4)`. Synthetic layer numbers are seeds,
not checkpoint activation captures. Each row's position is
`128 + ((tuple + 11*row) % 32)`, so every row covers all 128–159 tail positions
and multi-row cases have heterogeneous positions.

Position159 is **numerical-only**. Its generated index32 is outside public
P128/O1..32 serving admission. The host fixture validator checks numerical C10
geometry and synthetic ownership; it does not claim the earlier Rust descriptor
codec would admit this row. The probe does not link that Rust prototype.

The physical pool has64 blocks, 393,216 bytes each for K and V. Each active row
owns all ten reserved logical blocks through an independent host ownership
table, including its unused future block when the live table has only nine.
The identity, reverse and affine `(7*index+3)%64` permutations are bijections;
both all ten reserved blocks and every live prefix are checked for disjointness.
All physical KV slots are initialized and compared unchanged after attention,
including unmapped and future positions.

The full 1,280-byte canonical packet follows the frozen descriptor contract
SHA `5bed6d70342dec4f30c85ba67dd41cf560e6b7f4b0de5dedfe59424cecc9b641`.
The host validates its fixed headers, rows, C10 tables, identity/slot fields and
zero padding against separately constructed synthetic rows and ownership. It
then deliberately poisons the **device copy's inactive row metadata** with
`0xff`; inactive Q contains BF16 NaNs. This diagnostic exception tests that the
inactive gate ignores those inputs and overwrites output with zero. It is not a
production packet-admission rule. The source's early uniform return supplies
the no-read argument; poison/output checks do not trace individual GPU loads.

Every active row's full 576 BF16 output words are compared exactly with an
independent invocation of accepted M1 attention using its own 256-byte-aligned
Q, 80-byte C10 metadata and output allocations. Both paths read the same
immutable KV pool. Candidate output is poisoned before execution, including the
inactive padded row. Each allocation has distinct256-byte prefix/suffix guards;
the probe checks all guards, original/candidate inputs, every KV byte, earlier
oracle outputs, inactive positive-zero output and cross-map output invariance.
It records the first unequal row/word and both raw BF16 values without tolerance.

Each case performs one actual N-row candidate launch and `active` independent
M1 oracle launches; there is no candidate serial-M1 bridge. At most17 guarded
device allocations are live per case; all432,000 allocations across the fixed
matrix must be freed. These allocations and synchronizations belong to the
diagnostic harness, not a proposed serving hot path.

Three additional records test a retained bucket4 graph, separately from the
34,560 arithmetic cases. The same five candidate parents and twelve independent
M1 parents remain allocated throughout A3 → A4 → A3. The graph is captured and
instantiated once, checked to contain exactly one kernel node, and replayed
without node-parameter updates. Fresh H2D uploads occur outside capture to the
same allocations. Active counts, Q/K/V contents, row identities, positions and
physical mappings change between replays:

| Replay | Active | Positions | Layer seed / pattern / map |
| --- | --- | --- | --- |
| 1 | 3 | 128, 139, 150 | 0 / bounded fingerprint / identity |
| 2 | 4 | 145, 156, 135, 146 | 15 / signed zero impulses / affine |
| 3 | 3 | 133, 144, 155 | 29 / tail cancellation / reverse |

Every replay repeats full M1 comparison, input/KV immutability and guard checks;
the padded row returns to positive zero after previously being active. Separate
graph/executable creation/destruction counts must each be one. These17 extra
buffers bring the total required frees to432,017. The graph and executable are
destroyed before their parents, including exception unwinding. This tests
synthetic packet freshness with fixed device addresses; it does not test the
future owner, descriptor admission or actual serving graph lifecycle.

## Separate prepare, build, run and validate steps

Only the parent should execute the following steps after the serving campaign
and prerequisite gates allow it. All output paths must be fresh and outside the
accepted source snapshot.

```sh
python3 probe.py prepare \
  --build-receipt /tmp/riley-opt-260912/batch7-build.json \
  --conventions /tmp/riley-opt-260912/batch7_attention_probe.py \
  --descriptor-contract /tmp/riley-opt-260912/MULTISEQUENCE_DESCRIPTOR_CONTRACT.md \
  --output /tmp/riley-opt-260912/multisequence-attention-probe

python3 probe.py build \
  --manifest /tmp/riley-opt-260912/multisequence-attention-probe/fixtures.json \
  --nvcc /absolute/path/to/the/matching/production/nvcc

python3 probe.py run \
  --manifest /tmp/riley-opt-260912/multisequence-attention-probe/fixtures.json \
  --driver-library-dir /tmp/riley-opt-260912/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu \
  --timeout 3600

python3 probe.py validate \
  --result /tmp/riley-opt-260912/multisequence-attention-probe/result.json
```

The frozen compile-conventions helper is pinned to SHA
`bf8f8deca087d6db53c6cce9238d05b2d014328b3f0000f864cbf30ebc56e7f1`.
It reads the accepted CMake cache, include response files, compile rule and
complete production options. Both numerical TUs use the same flags, including
SM89 code generation and per-source `--use_fast_math`. Host fixture code is
compiled separately with exceptions enabled. Linkage uses the exact shared
cudart selected by that production build; no static-runtime substitution is
permitted.

`run` requires the pinned private `libcuda.so.1` content, clears inherited
library-path selection to the explicit driver/cudart directories, and rejects
`LD_PRELOAD`. Native device records must match the fixed GPU UUID and CUDA13.0
runtime. Before/after `/proc/self/maps` records must identify those exact loaded
driver/runtime files and current inode/device identities. These runtime checks
are part of future execution; they have not been observed by this preparation.

The wrapper owns one fresh native process/session. The native source contains
no child-process spawning. Timeout/interruption cleanup targets that session,
tolerates an already-exited leader, reaps its process, and records primary and
cleanup failures independently. Failure cannot create a completed result.
Existing run artifacts are never overwritten. Raw records survive mismatches
and failures for diagnosis.

A complete, correctly cleaned-up numeric mismatch can produce a result with
`attention_bitwise_equal=false`. `completed=true` means the full experiment
finished, not that arithmetic passed. Passing all output/guard/immutability
checks still means only this synthetic attention component passed; it cannot
select a serving candidate, bypass projection admission, or change accuracy
acceptance policy.

## CPU validation

```sh
python3 -m unittest discover \
  -s benchmarks/prototypes/multisequence_attention_v1 -p 'test_*.py' -v
```

The tests cover exact accepted-source lineage and reverse transformation,
inactive-before-barrier ordering, complete active/inactive output coverage,
full reserved/live block disjointness, every row's tail-position coverage,
unchanged synthetic sampling, complete compile-command construction, full raw
case validation, mismatch preservation and owned-process timeout/cleanup races.
They also reject incomplete retained-graph transition coverage, graph updates,
unexpected node/capture counts and unfreed graph/executable state. A source
contract test verifies the active count is a device header read and that the
same bucket4 wrapper is used for both A3 and A4.
They do not compile the C++/CUDA sources or execute CUDA. Native compilation,
real BF16 comparison, complete model/KV checks and serving measurements remain
separate pending gates.
