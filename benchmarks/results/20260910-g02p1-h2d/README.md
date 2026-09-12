# G02P1 actual token/metadata H2D bridge — 2026-09-10

The actual packed pinned host slab and device slab are bound to the existing
whole-slab H2D graph, with fresh-source staging before every replay. M=1
SmolLM2/canonical correctness is verified. Full decode composition remains
unimplemented; this receipt does not promote aggregate admission or performance.

## Implementation

`BorrowedH2DGraph` retains the actual stream, pinned source and device target.
It preserves native equal/nonzero whole-slab preflight and the existing guarded
fresh-source API. Each replay checks payload length, stages bytes into the same
retained pinned allocation through `stage_h2d_source`, launches, and waits for
completion. Wrong payload length is recoverable; native stage/launch/completion
failure makes the wrapper terminal. Close/Drop releases the leases.

An initial attempt omitted the native fresh-source stage and was rejected on
both model cases. `initial-freshness-rejection.log` preserves that failure.
The fix invokes the existing freshness authority; no native checks were relaxed
and no native/kernel/ABI changes were made in this stage.

The executor audit derives the current packed layout and independently packs
tokens plus all metadata fields. Before intervention it requires the host mirror,
actual pinned bytes and actual device bytes to equal that request payload.
It admits only M=1 active payloads exactly covering both source and destination
owners; larger-capacity/prefix cases are explicitly rejected.

To rule out a no-op copy, the audit overwrites the device destination with a
sentinel, then performs 16 fresh-staged H2D graph replays. The complete copied
slab and unchanged host/pinned source must equal the expected request. That
restores the original device contents. Failure poisons the executor. No new
pinned/device allocation or synthetic metadata source replaces an actual owner.

## Verification

Final GPU commands exited 0: **5 tests passed**.

- Actual SmolLM2/canonical model cases: 2. Each has four iterations and sixteen
  H2D replays per iteration (64 per model). Full token/metadata slab equality,
  prior operator audits, logits/KV/output/continuation and allocation stability
  pass. SmolLM2 tokens `[808,2775,288,536]`; canonical `[2,2,2,2]`.
  Uninitialized H2D audit is rejected; all allocations close to zero.
- Borrowed H2D fixture: 1. Unequal slabs and short payloads are rejected;
  changing payload bytes are staged repeatedly into the same graph. Exact
  copy/source equality, close/Drop and stable allocations pass.
- Existing owned H2D lifecycle/freshness regressions: 2.

CPU: CUDA library 84 / graph contracts 32, runtime library 258 / architecture
15 / inventory 1. Format/diff checks pass. Normal Clippy exits 0 with no warnings
in the new H2D modules; inherited warnings remain. Source hashes match remote
scratch, binary/checkpoint/GPU identities are recorded, and unrelated model
source is unchanged. Nothing was committed, pushed or deployed.

## Concrete next boundary: native aggregate composition

The current standalone wrappers each acquire exclusive leases on shared
buffers and admit a fixed operation. Keeping all of them alive together is not
a valid composition mechanism. The next implementation needs one native owner
that deduplicates buffer/plan leases and records the complete dependency chain:

1. Fresh packed input H2D and embedding/status.
2. For each layer: norm, Q/K/V GEMMs, RoPE, KV write, attention, output GEMM,
   attention residual, post norm, gate/up, SiLU/multiply, down and MLP residual.
3. Final norm, LM head, output gather, argmax, result/status D2H and completion.

Normal dispatch swaps `hidden_current` and `hidden_projection` after each MLP
residual (`executor/dispatch.rs`). Aggregate recording must express that as
fixed physical-buffer identities indexed by layer parity, not reuse a single
cold snapshot mapping for every layer. SmolLM2 has 30 layers; more general layer
counts require their own reset/continuation proof.

The aggregate also needs exact selected GEMM policies/algorithms and workspace
ownership, packed KV/position/output spans, context/stream identity, one fresh
input authority per launch and completion-gated publication of all error/status
records. Existing standalone status behavior does not prove downstream handling
of embedding/metadata failures inside a full model graph.

Acceptance is one retained full M=1 capture across changing tokens/positions,
independent eager logits/token/KV equality, boundary/rejection/cleanup tests,
and stable allocations. Only then qualify bucket dispatch and run the matched
SmolLM2 vLLM comparison. Global inventory remains 7 Supported / 7 Unknown,
aggregate Unknown; this receipt does not claim G02H/G03 completion.
