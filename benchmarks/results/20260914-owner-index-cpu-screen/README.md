# Shared-owner CPU validation screen

Local CPU release-build experiment, not the remote GPU host. Baseline function copied from the pre-change dense-wire validator; inputs are valid 32-row decode descriptors with 1,152 owner/page pairs. Two reversed orders, 2,000 calls each, no separate warmup. The experiment measures structural validation only, not the ownership-retention shortcut or full future encoding.

Owner=16: baseline 12.282/12.485 us, candidate 8.526/8.683 us. Owner=32: baseline 24.528/24.521 us, candidate 22.388/22.590 us. These cases consistently favor the candidate in this small screen. Owner=0 and 4 show substantial order/run variability, so do not claim improvement there.

This supports retaining the candidate for remote serving evaluation, not promoting it. The linked per-page lookup grows with owner count and still needs larger off-batch-owner stress coverage. Serving preflight remains below the fixed 16 GiB available-memory threshold after permitted tmpfs cleanup; both failed preflight attempts restored Blender and launched no serving measurement. No vLLM performance result exists for this candidate.

Run: cargo test -p riley-runtime --release --lib owner_index_bench --offline -- --ignored --nocapture

## Off-batch scaling follow-up: rejected

The subsequent release CPU screen uses 16 active sharing owners and 0/32/128/512 off-batch owners, each owning the same 24 immutable pages. Each measurement runs 200 calls in two reversed orders, without separate warmup. These are local synthetic costs, not remote serving measurements.

| Off-batch owners | Baseline us/call (two orders) | Candidate us/call (two orders) |
|---:|---:|---:|
| 0 | 15.009 / 14.339 | 9.842 / 9.423 |
| 32 | 32.345 / 29.064 | 46.540 / 43.560 |
| 128 | 88.492 / 86.991 | 395.087 / 395.903 |
| 512 | 501.507 / 509.334 | 7978.298 / 7730.693 |

Per-page linked-list duplicate detection scans all earlier owners, producing quadratic construction cost as sharing grows. Reject v1 for production despite small-owner gains. Both production edits (linked owner index and unmeasured ordered-prefix retention shortcut) were removed from the worktree; the exact experiment is preserved in `rejected-owner-index-v1.patch` against the recorded base commit. The retention shortcut may be evaluated independently later; this screen neither proves nor disproves its benefit.

To reproduce, apply the patch in an isolated checkout at the recorded base commit and run the command above. The preserved serving controller targets the rejected frozen binary and is historical, not an instruction to promote or rerun it. GPU serving preflight failed before measurement; no candidate/vLLM comparison exists. No runtime change is promoted by this artifact commit.
