# G02P2 layer normalization resource audit — 2026-09-10

Development **cold resource audit implemented and verified**. This is partial
G02P2 evidence: no per-layer in-flight activation trace, retained full-chain
owner, full decode graph, release qualification or performance claim.

## Change

- Every input/post-attention normalization derives its physical uploaded weight,
  BF16 shape, epsilon and reduction profile from the actual prepared layer plan.
  InputNorm maps hidden_current -> hidden_norm; standalone PostAttentionNorm
  maps hidden_rotary -> hidden_norm, matching the current dispatch fields.
- A diagnostic transaction borrows the actual executor buffers and stream,
  captures the existing profile-specific primitive, checks eager/graph byte
  equality, finite output, unchanged input/weights/tails, recaptures and replays,
  closes the graph, then restores the original shared normalized output.
- The complete schedule is validated before mutation. Exact M=1, separate
  residual normalization and an already completed iteration are required.
  Fixed37, fused residual and uninitialized scratch cannot inherit this evidence.
  Any probe/restore failure after entering the transaction poisons the executor.
- Binding includes Final vs Input(layer) vs PostAttention(layer). Layer evidence
  cannot promote FinalNorm. Audit hash receipts are diagnostic values, not graph
  admission tokens. All temporary native graphs are closed before returning.
- The audit is cold and disconnected from hot dispatch. Between-iteration scratch
  contains reused data; it is **not** claimed to hold each layer's historical
  activation, nor to prove the future full-graph buffer-swap schedule. Those
  connections still require G02H/G03 validation. Global inventory remains
  7 primitive Supported / 7 Unknown, aggregate Unknown.

## Verification

Final remote command, exit 0:

```sh
cargo test -p riley-runtime --features cuda --lib c07_ -- --include-ignored --nocapture --test-threads=1
```

- CUDA-enabled C07 suite: **26 passed**, including CPU contracts, both model
  layer audits, FinalNorm model/owner regressions and existing attention/H2D
  ownership/lifecycle coverage. Do not read this as 26 new GPU tests.
- SmolLM2 H576/L30: all 60 norm sites audited after each of four iterations,
  240 site transactions. The canonical H64/L2 fixture contributes 16 more.
- Both models use two independent executors, one-token prefill plus three decode
  steps. Original normalized scratch, full logits, all initialized KV across
  layers/heads, allocation counters and next-token continuation agree. Resources
  close to zero. HF continuation is [808, 2775, 288, 536]; canonical is [2, 2, 2, 2].
- Fused-path and pre-iteration rejection are recoverable and tested before the
  successful continuation. Existing owner matrix additionally covers layer-vs-
  FinalNorm admission, wrong weights/profile, non-finite poisoning and Drop/close.
- Runtime CPU library: 258 passed. Architecture: 15 passed. Inventory: 1 passed.
  Formatting and diff checks passed. Normal CUDA Clippy exit 0 with no diagnostics
  in the changed norm modules; baseline repository warnings remain.

`gpu-final.log` is the final source run. Earlier focused runs and initial lint
output are retained; they are not added to the final suite count. Source overlay
and SHA256s were checked against the remote scratch; runtime binary/checkpoint
manifest/GPU identities are recorded. The model loader validates payload hashes
when loading. Six unrelated local riley-model edits remain byte-for-byte intact.

Remote: ssh ai-assistant; /tmp/riley-g01-native-260910; isolated target
/tmp/riley-g01-native-260910-target. CUDA toolkit /data/cuda-12.8.1/bin;
CMAKE=/data/cmake-3.31.12/bin/cmake. Checkpoint env variables and exact paths are
in runtime-identities.json. Source base is in manifest.json. For the canonical
fixture use the preceding G02C receipt's generate_canonical_fixture.py.

The user selected **SmolLM2 first, then expand the model/workload matrix** for
later matched vLLM benchmarks. Shared GPU activity and high host I/O pressure
exclude timing claims here. Remaining P1/P3-P7, A/B/D-F/H and G03/G04 are pending;
this receipt does not close the user's overall performance roadmap.
