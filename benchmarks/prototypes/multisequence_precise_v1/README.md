# Independent-row precise operations, prototype v1

This is a local preparation for two missing components of a future genuine N-row decode batch. It does not change production sources, establish model correctness, or measure performance. CUDA compilation and GPU execution have not been performed as part of this prototype's local validation.

The accepted Batch7 precise translation unit is the M1 oracle, byte-for-byte unchanged (`fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f`). `probe.py` extracts the packed RoPE/KV and SwiGLU kernel bodies into a separate namespace. Reversing only the new signatures and row-pointer prefixes reconstructs the original kernels exactly. Both the unchanged oracle translation unit and generated candidate use the production CMake flags for `graph_numerics_precise.cu`, including its optimization and SM89 target. Unlike the attention translation unit, this precise source has no per-source `--use_fast_math`. Linking explicitly uses the selected shared CUDA runtime.

The row operations have these explicit layouts:

| Operation | Input and output layout | Launch |
| --- | --- | --- |
| Packed RoPE/KV | QKV stride 960 BF16 values; Q/K/V offsets 0/576/768; Q output stride 576; shared full physical K/V pools; C10 row view at byte `128 + 128*r` | grid `(2, bucket)`, 256 threads |
| Packed SwiGLU | GU stride 3072 BF16 values; gate/up offsets 0/1536; product output stride 1536 | grid `(6, bucket)`, 256 threads |

`bucket` is the only captured row-shape argument (1, 2, or 4). Both kernels read active count from freshly uploaded common-header word `packet[6]`, byte 24. An inactive CTA reads no row metadata or packed inputs; it writes positive zero to its whole output row. No host active-count scalar, graph parameter update, or recapture is used.

## Coverage and limits

The 3,456 normal cases each run both operations: synthetic seeds **0, 15, 29** × 32 independent position tuples × 3 input patterns × 4 `(bucket, active)` shapes `(1,1), (2,2), (4,3), (4,4)` × 3 physical maps. These are synthetic seed indices, not checkpoint layers or captured model activations. Every active row visits all positions 128–159. Position 159 is numerical coverage only; public P128/O32 serving's last input remains 158.

The pinned attention prototype supplies an exact, narrowly extracted block of CPU fixture, synthetic reservation-authority, guarded buffer, and mapping helpers. Its main, CUDA kernels, wrappers, and retained graph implementation are excluded. The synthetic host ledger validates the descriptor before inactive row metadata is deliberately poisoned. It is not a production reservation registry or scheduler admission implementation.

The precise harness constructs raw packed QKV/GU inputs, exact FP32 RoPE-table bit patterns (including signed zero, subnormal, and BF16 halfway cases), and poisoned current-token K/V slots. Independent, 256-byte-aligned M1 row inputs and outputs avoid accepting a misaligned row subspan as an oracle. Every active Q and product word and the **entire 393,216-byte K and V pools** are compared byte-for-byte, including zero signs. It separately checks all guards, all input/table/metadata bytes, oracle-output immutability after candidate execution, non-current KV slots, finite outputs, inactive positive-zero outputs, and logical output equivalence across all three maps.

A separately counted retained graph contains exactly the two candidate kernel nodes. The same 43 guarded parents and graph executable replay **A3 → A4 → A3** with changed positions, maps, and inputs. All fresh uploads occur outside capture. Each replay is compared against aligned M1 oracles, and the graph and executable are destroyed before parents. The normal matrix uses 107,136 guarded allocations; the retained graph uses 43 more, for **107,179 allocations** to reconcile with frees. The three graph replays do not count as three additional normal cases.

A completed numerical mismatch is retained as `completed: true, precise_bitwise_equal: false`; it cannot become passing evidence. Runtime, incomplete output, or cleanup failures prevent a completed result receipt. No timing fields, serving qualification, checkpoint-wide claim, or production integration claim is produced. Arithmetic body preservation does not prove identical GPU code generation; actual CUDA execution remains required.

## Commands for later execution

Run these only in an explicitly selected isolated environment. The accepted snapshot, production CMake output, compiler, private driver, and shared runtime must still match their pinned bytes. Each output directory and every run artifact is exclusive; retries use a new directory.

```bash
python3 probe.py prepare \
  --build-receipt /tmp/riley-opt-260912/batch7-build.json \
  --conventions /tmp/riley-opt-260912/batch6_projection_probe.py \
  --support-driver /path/to/multisequence_attention_v1/probe.py \
  --descriptor-contract /path/to/MULTISEQUENCE_DESCRIPTOR_CONTRACT.md \
  --output /tmp/riley-opt-260912/multisequence-precise-probe
python3 probe.py build \
  --manifest /tmp/riley-opt-260912/multisequence-precise-probe/fixtures.json \
  --nvcc /data/riley-g04-cuda13/bin/nvcc
python3 probe.py run \
  --manifest /tmp/riley-opt-260912/multisequence-precise-probe/fixtures.json \
  --driver-library-dir /path/to/verified/private-driver \
  --timeout 3600
python3 probe.py validate \
  --result /tmp/riley-opt-260912/multisequence-precise-probe/result.json
```

`prepare` and `build` do not initialize CUDA; only `run` does. `run` owns a fresh native child process and uses a whole-process deadline with bounded TERM/KILL cleanup. Primary and cleanup failures are preserved separately. The result binds the frozen build/source/binary receipts, generated files, helper and descriptor hashes, exact compile commands and CMake artifacts, actual driver/runtime mappings, raw comparison records, and owned-process cleanup.

Local checks are stdlib-only:

```bash
python3 -m unittest discover -s benchmarks/prototypes/multisequence_precise_v1 -p test_probe.py -v
```

The source transformation, complete matrix geometry, inactive coverage, exact compile-command propagation, normal/retained raw accounting, and failure cleanup tests use no GPU. Synthetic raw test records are parser tests, not GPU correctness evidence.
