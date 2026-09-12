# G02B selected-plan GEMM bridge — 2026-09-10

**SmolLM2 actual-plan cold graph parity is now verified.** The prior policy
boundary is resolved without reselecting or relabeling a GEMM plan. The cold
zero-byte workspace sentinel is removed. Retained aggregate/full decode graph
integration and vLLM performance qualification remain pending.

## Contract and implementation

The additive native entrypoint
`riley_cuda_graph_capture_begin_selected_no_split_gemm_bf16` preserves the
prepared opaque cuBLASLt algorithm and reviewed policy flags 0/1/3. It admits
only effective split-K <= 1 with reduction scheme NONE. A permissive prepare
policy is not evidence of actual split-K execution; real split-K/reduction
algorithms remain rejected. The legacy strict C05-21 and composite C05-22
entrypoints keep their prior policy/allocation contract.

Internal graph state records the selected contract through capture, graph,
exec and cleanup. Native preflight checks exact input/weight/output sizes,
context, aliasing, plan readiness and leases. Workspace is optional only when
the selected plan requires zero bytes. A supplied workspace parent must cover
the required byte-zero prefix and remains entirely leased through completion
and destruction. Null-workspace capture, abort and close omit only that absent
lease. No kernel math or prepared plan is changed.

`BorrowedSelectedGemmGraph` retains exclusive Rust borrows of the plan, stream,
actual I/O/weight allocations and optional workspace parent. It repeats
preflight checks before entering native capture. Failure is terminal for that
graph; native destruction precedes release of the borrows. The strict borrowed
wrapper remains available and retains its original semantics.

The executor cold audit maps Q/K/V/attention output/gate/up/down from existing
per-layer physical weights and dispatch scratch. It uses the original GEMM
plans and `gemm_workspace.as_mut()` directly. For each projection, four replays
are compared with the same plan executed eagerly after output poisoning.
Checks cover byte equality, finite output, full input/weight preservation,
unchanged plan config/algorithm metadata, and output restoration with readback.
A failed transaction poisons the executor. These are scratch snapshots between
completed iterations, not an in-flight layer activation trace.

## GPU evidence

Remote `ai-assistant`, RTX 4090 UUID
`GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`, SM89.
Final GPU commands exited 0, **10 tests passed**:

| Gate | Tests |
|---|---:|
| Actual SmolLM2 and canonical continuation | 2 |
| New selected-plan fixture | 1 |
| Native preflight/abort/parent lease | 1 |
| Existing strict borrowed fixture | 1 |
| C05-21 legacy lifecycle/parity | 2 |
| C05-22 composite lifecycle/parity | 3 |

SmolLM2: 30 layers x 7 projections x 4 iterations = **840 comparisons and
3,360 graph replays**. Canonical H64/L2: 56 comparisons and 224 replays. Both
use their actual absent workspace (`None`), without an extra zero-byte owner.
Full logits, initialized KV contents and continuation match independent
baseline executors: SmolLM2 `[808,2775,288,536]`, canonical `[2,2,2,2]`.
Allocation counts stay unchanged and return to zero at teardown. Prior
embedding/norm/RoPE/KV/pointwise cold audits also execute in these model tests.

The selected fixture covers all three prepare policies and four SmolLM2
projection geometries. It checks policy/algorithm preservation, no-workspace
and supplied 4096-byte parent cases, 16 replays with both explicit close and
Drop, unchanged parent/input/weight, output parity and recoverable short-output
rejection. Legacy strict capture still rejects permissive policies.

The native test bypasses public Rust preflight: it verifies strict native
policy rejection, selected native alias rejection, null/present workspace
capture abort, rejection of plan/input close while captured, and rejection of
parent close while leased. All resources close successfully after abort.

All tested selected algorithms require zero workspace. A supplied nonempty
parent proves lease/preservation behavior for an unused workspace; **positive
nonzero-required-workspace execution is not established by this receipt**.
Actual split-K algorithms and additional models are outside this evidence.

## Other checks and provenance

CUDA CPU library 84 / graph contracts 32; runtime CPU library 258 /
architecture 15 / inventory 1 passed. C ABI signature syntax check,
`cargo fmt --all --check` and `git diff --check` passed. Normal Clippy exited 0;
one new `needless_option_as_deref` warning remains in the cold audit, alongside
baseline warnings. This is not a warnings-as-errors result.

The source overlay and SHA256 map are verified against remote scratch.
Runtime identities record the test binary, checkpoint manifest and GPU.
Unrelated model-loader source is unchanged. No commit, push or deployment.
Shared host I/O pressure prevents treating run duration as performance evidence.

## Remaining

G02B is development-verified for the observed M=1/no-split SmolLM2 plans.
Next: LM head, GPU output/completion, selected metadata H2D chain and retained
aggregate/full decode capture, followed by bucket selection, qualification
and a matched SmolLM2 vLLM benchmark. The global inventory remains 7 Supported /
7 Unknown and aggregate Unknown. This standalone cold audit does not promote it.
