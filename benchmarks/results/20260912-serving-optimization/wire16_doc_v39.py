from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/tests/README_shared16.md'
s=p.read_text().replace('- Introduce a distinct 16-row wire/recorder contract or an explicitly parameterized contract; retain the eight-row baseline.', '- The Rust V4 wire and native structural parser now exist (see `README_variable_wire16.md`); the graph recorder is still pending. Retain the eight-row baseline.')
s=s.replace('- Update request encoding, identity/digest/max-row checks, zero unused-row validation, result capacities and CPU-validated argmax slot arrays together.', '- Request encoding, identity/digest/max-row checks, zero unused-row validation and result capacities have V4 coverage. Runtime buffers and CPU-validated argmax slot arrays still need integration.')
p.write_text(s)
(r/'kernels/tests/README_variable_wire16.md').write_text('''# Versioned sixteen-row wire contract (V39)

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

## Pending integration; no serving performance claim

The native16 model/result prototype still uses the old single-reference prefill offset/result magic. It is deliberately not connected to this new wire. Production V3 recorder and runtime remain eight-row. Before V4 dispatch, connect canonical token offset26752, result magic0x34524d52, retained transfer sizes/offsets, M16 head and scratch allocation, source digest, scheduler selection, and output slot capacity together. Then rerun native/Rust-owned GPU parity, cleanup/cancellation, and matched serving benchmarks. Do not use V39 CPU evidence as GPU completion or throughput evidence.
''')
