# G02D LM head cold graph bridge — 2026-09-10

The actual LM head prepared plan, physical weight, final-normalized input,
logits output and optional workspace are now connected to the selected-no-split
borrowed graph. This is standalone cold correctness evidence, not retained
full decode graph admission or speedup evidence.

The audit requires a healthy completed M=1 iteration and effective no-split
canonical GEMM. It uses the exact normal dispatch mapping:
`hidden_norm × lm_head_weight → logits`. No replacement plan, weight or
workspace is allocated. The existing graph probe performs four replays,
compares against eager execution after poisoning the output, checks finite
values and complete input/weight preservation, verifies unchanged plan
config/algorithm metadata, and restores/readbacks the original logits.
Transaction errors poison the executor; uninitialized use is rejected.

The probe now returns its captured output. Model tests compare that output
byte-for-byte with the full logits of an independent baseline executor before
continuing. Thus the baseline assertion checks graph output itself, in addition
to checking restored output and subsequent decode. Projection audits continue
to reuse the same probe and discard its returned bytes.

## Verified

Remote RTX 4090, CUDA runtime 12.8, cuBLASLt 12.8.4:

- 2 model GPU tests passed, exit 0: actual SmolLM2-135M BF16 and canonical H64/L2.
- Per model: prefill plus 3 decode steps, 4 LM head comparisons / 16 replays.
  Both have vocabulary size 49,152; SmolLM2 K=576, canonical K=64.
- Selected algorithms have split-K=1/NONE and require zero workspace. Actual
  absent workspace is borrowed as `None`; no diagnostic sentinel is used.
- Captured full logits equal independent baseline logits. Restored logits,
  initialized KV and continuation match: SmolLM2 `[808,2775,288,536]`,
  canonical `[2,2,2,2]`. Allocation counts remain stable and close to zero.
- Existing all-layer projection, embedding, normalization, RoPE, KV-write and
  pointwise cold audits run in the same tests.
- Runtime CPU 258, architecture 15 and inventory 1 passed. Format/diff checks
  passed. Normal Clippy exited 0; the previously recorded
  `needless_option_as_deref` warning in the shared cold probe remains, along
  with baseline warnings. No new LM head diagnostic was reported.

No CUDA kernel, ABI or graph lifecycle implementation changed in this stage.
Source hashes match remote scratch; binary/checkpoint identities and model
source preservation are recorded. No commit, push or deployment occurred.

Next: GPU token selection and output/completion, metadata H2D chain, retained
aggregate and full decode graph. Global inventory stays 7 Supported / 7 Unknown,
aggregate Unknown. Actual split-K or positive workspace requirements, larger M
and matched vLLM performance remain outside this receipt.
