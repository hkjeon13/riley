# Progress publication validation — 2026-09-13

This is a snapshot of accumulated work for publication to `main`, not a release
qualification or a claim that the overall vLLM performance goal is complete.
No GPU runs or deployment were performed for this publication.

| Check | Observed result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed. |
| `git diff --check` before staging | Passed for tracked changes. |
| `cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings` | Failed: documentation formatting and CUDA wrapper lint diagnostics remain. |
| `cargo test --locked --workspace --all-targets --no-default-features` | Stopped at the first failed target: 511 passed, 1 failed, 24 ignored across the targets reached; subsequent targets were not run. |
| `python3 -m unittest discover -s benchmarks/competitive/scripts/tests -p 'test_*.py' -v` | 45 tests passed. |
| `python3 -m unittest discover -s benchmarks/scripts/tests -p 'test_*.py' -v` | 510 tests ran; 2 failures and 30 errors, including subtest errors. |

The Rust failure is
`graph_decode_exact_device_slab_stays_a_cold_geometry_binding_boundary`.
It searches for the literal `source.layout() != self.layout`, while the source
uses `pure_decode_graph_v1_exact_metadata_layouts_match(self.layout, source.layout())`.
That helper call and the literal assertion are already present in the parent
commit `3bc72c9`; this publication did not repair that source-text contract.

Python failures cover reviewed serve-argument source-hash drift, performance
runner derivation drift, frozen decode/prefill profiler source hashes, and a
`prompts_sha256` mismatch in
`20260911-g04-gui-measurement/engine/pair-01-vllm/vllm/raw.jsonl`.
These checks remain failed; historical receipts were not rewritten to make them
pass against the current source.

Large local artifacts are documented in
[the artifact storage note](ARTIFACT_STORAGE_20260913.md). Source, documents and
textual measurement evidence are included. Files written by ongoing experiments
after the staging snapshot remain outside this commit.
