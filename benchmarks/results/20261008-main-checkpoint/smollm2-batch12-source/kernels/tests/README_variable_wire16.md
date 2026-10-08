# Versioned sixteen-row wire contract (V39)

`variable_wire::Expectation<8>` remains the default V3 type. `variable_wire16::Expectation` aliases `Expectation<16>` and uses V4 encoding. Both share authority validation and completion checking. Their single-record result extents match, but result magic differs and cross-version results are rejected.

| Field | V3 | V4 |
|---|---:|---:|
| Request magic | 0x33444d52 | 0x34444d52 |
| Version | 3 | 4 |
| Result magic | 0x33524d52 | 0x34524d52 |
| Descriptor capacity | 8 | 16 |
| Token offset | 13440 | 26752 |
| Request bytes | 17536 | 30848 |
| Result record bytes | 98432 | 98432 |
| Batch result bytes | 787456 | 1574912 |
| Pinned output offset | 787456 | 1574912 |
| Shared staging bytes | 1574912 | 3149824 |

Only wire capacities8/16 are accepted. Unsupported capacities, invalid live ownership, aliases, duplicate identities/slots, stale replay and row counts above capacity are rejected before packet mutation. Max active rows may be1/2/4/8 (and16 for V4), while selected rows may be any positive count up to that limit. Prefill retains one-row semantics. Every request byte is canonical; unused rows, unused token slots and result records must be zero. Completion validation checks all active rows before returning any result, including full finite BF16 logits and deterministic lowest-ID argmax.

The native parser is structural only. `valid_prefill_shape_packet` retains V3; `valid_v4_shape_packet` selects V4. It cannot establish live ownership or replay acceptance; Rust must supply that authority before native dispatch.

## Verification

Run `cargo test -p riley-runtime --lib variable_wire` with `RILEY_V39_WIRE_FIXTURES` set to an output directory. This exports eleven old/new Rust packets. Compile `variable_wire16_packet_test.cpp` with C++17, `-Ikernels/src -fsanitize=address,undefined -fno-omit-frame-pointer`, then pass all eleven fixture paths. Tests include all-byte mutation safety, upper-row aliases, unused records/descriptors, wrong versions/extents, and a maximum1024-token prefill with no descriptor overlap.

## V40 integration and remaining performance work

V40 connects the canonical token offset26752, result magic0x34524d52, exact graph extents/aliases, retained transfers, M16 head, source digest, scheduler selection and sixteen output slots. Select it explicitly with `--graph-numerics variable-smol-v4`; V3 remains available. See `README_shared16.md` for GPU/HTTP receipts and the scope of Round48 serving comparisons.

The standalone native16 probe deliberately retains its old reference packet/result format. Its default result template uses the old magic; the compiled V4 serving wrapper explicitly chooses0x34524d52. Do not feed its standalone reference setup into V4 serving.

Round48 found high-concurrency gains but low-concurrency regressions and a remaining vLLM throughput/TPOT gap. V40 is not a completed performance goal. Runtime native intervals include GPU execution, synchronization and host readback; they must not be labeled GPU-only timing. Profile these costs and preserve full correctness/ownership checks in subsequent changes.
