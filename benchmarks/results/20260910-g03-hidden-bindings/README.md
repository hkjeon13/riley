# G03 prerequisite: fixed hidden-buffer roles inside the layer chain

## Delivered scope

The batch dispatch now chooses the two hidden scratch allocations by **layer
ordinal parity**, rather than moving their owners after every layer. Layer 0
uses physical A as current and B as projection; layer 1 reverses these roles.
The next layer therefore consumes the preceding residual output. Norm, Q/O
projection, RoPE, attention residual, down projection and MLP residual all use
the selected references. No CUDA allocation or copy was added.

For odd layer counts, one swap after the loop preserves the existing final-norm
input and externally retained owner roles. Even counts (including SmolLM2 L30)
need no swap. This does **not** establish stable odd-model owner roles across
replays: a future native recorder must implement its own fixed entry/final
mapping and continuation handling. Existing eager API behavior is preserved.

This is the buffer-binding prerequisite for aggregate composition, not a native
aggregate implementation. No native ABI, graph lease guard, inventory admission
or graph replay dispatch was changed. G02H/G03 remain incomplete, inventory
remains 7 Supported / 7 Unknown, and no vLLM performance result is claimed.

## Verification

An instrumented **previous dispatcher** ran first on the remote GPU. Its source
hash is verified against HEAD and its source and binary identity are retained.
Then the new dispatcher ran with the same models, test code and CUDA environment.
Four consecutive iterations contribute every output-logit byte and initialized
K/V region to each receipt. The pre/post hashes and tokens match exactly:

| Model | Layers | Hashed bytes | Generated tokens |
|---|---:|---:|---|
| SmolLM2-135M BF16 | 30 | 623616 | 808, 2775, 288, 536 |
| canonical fixture | 2 | 398336 | 2, 2, 2, 2 |
| canonical odd fixture | 3 | 400896 | 5, 5, 5, 0 |

`before-after-parity.json` contains all three full hashes. Five model receipts
include repeated even-model runs with existing operator audits enabled.
The odd fixture is derived reproducibly from the two-layer fixture by copying
layer 1 into layer 2; its script and manifest are retained. It is synthetic.

- GPU: **14 before / 14 after** C07 tests passed, including actual model parity,
  prior operator audits, invalid-state rejection and allocation cleanup.
- Additional GPU batch regressions: **3 passed**: independent single-request
  forward comparison, fused/separate 16-step exact comparison, and mixed
  prefill/decode with packed/synchronous transport and KV continuation. The
  independent full-sequence comparisons use their existing tolerance criteria;
  they are not relabeled as byte-exact. Packed/synchronous transport reports
  zero raw-logit mismatches and zero live allocations after close.
- CPU: runtime library **259**, architecture **15**, inventory **1** passed.
  The new role test compares the previous swap algorithm with fixed physical
  roles for 0/1/2/3/29/30/31/64 layers across 16 iterations, including physical
  owner identity and both scratch contents.
- `cargo fmt --all -- --check`, `git diff --check`, and normal runtime Clippy
  passed. Existing warnings remain; no new role-helper warnings were emitted.
- An initial test-only SHA formatting compilation error was corrected before
  either baseline or candidate GPU run; its log is preserved.
- Remote source hashes match the overlay. Six unrelated modified model-loader
  files retain their original hashes. No commit, push or deployment occurred.

## Remaining aggregate work

The native owner still needs a deduplicated resource/plan lease ledger, full
ordered recording from H2D through all layers and output/status D2H, guarded
input freshness on every launch, and completion-gated error publication. The
fixed per-layer role mapping above resolves one prerequisite; it does not grant
resource lifetime or capture authority. One retained full M=1 graph must next
pass changing-token/position logits, token, KV, rejection and cleanup checks
before bucket qualification and matched SmolLM2 vLLM measurement.
