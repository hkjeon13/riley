# G02B strict GEMM bridge and SmolLM2 policy boundary — 2026-09-10

**Partial development result. SmolLM2 GEMM graph capture remains unverified.**
The borrowed strict canonical bridge and all seven projection mappings pass on
the H64/L2 canonical model. The selected SmolLM2 plans are rejected before
mutation because their policy is not the existing graph's strict policy.
This is not full decode graph integration or performance evidence.

## Changes

`BorrowedGemmGraph` retains the actual prepared plan, stream, physical weight,
input, output and exact workspace through graph destruction. It reuses C05-21
preflight/capture/enqueue and preserves its strict policy and exact allocation
contract. Native code and ABI were not changed in this stage. CUDA failures
make replay terminal; explicit close and Drop release native resources before
Rust borrows end.

The cold executor audit maps query/key/value/output/gate/up/down exactly as
normal dispatch does. It neither prepares replacement algorithms nor copies
weights into stand-in allocations. For each canonical layer it performs four
replays, compares bytes with the same plan executed eagerly after poisoning the
output, checks finite output and complete input/weight preservation, then
restores and reads back the original scratch. These inputs are sampled between
iterations, not captured live activations of each layer.

A caller-owned zero-byte allocation is required as a native lease sentinel when
the actual plan requires zero workspace. The executor itself has no GEMM
workspace in the tested models. This extra cold diagnostic owner is explicitly
not a production workspace binding. Nonzero shared workspaces with a different
size are rejected; that case needs a separate parent/span integration.

## Live SmolLM2 finding

RTX 4090, CUDA runtime 12.8, cuBLASLt 12.8.4; M=1:

| Plan | N x K | Selected policy | Actual split-K / scheme | Workspace |
|---|---|---|---|---:|
| hidden (Q/output) | 576 x 576 | allow-in-place-and-output-type-split-k-v1 | 1 / NONE | 0 |
| key/value | 192 x 576 | same permissive policy | 1 / NONE | 0 |
| gate/up | 1536 x 576 | strict-no-split-v1 | 1 / NONE | 0 |
| down | 576 x 1536 | permissive policy | 1 / NONE | 0 |

All selected algorithms are marked deterministic. The policy differs even
though the observed algorithms use no split. Do not describe this as an
observed split-K computation or silently reprepare strict plans. The current
audit preflights all four plan families and rejects the whole SmolLM2 audit;
it does not claim even gate/up graph verification on the real model.

The initial GPU run failed the SmolLM2 case at this boundary (one canonical pass,
one failure). `gpu-model-initial-policy-rejection.log` preserves that result.
The final test explicitly verifies this expected rejection and subsequent
normal continuation; changing the expectation does not constitute a fix for
SmolLM2 capture support.

## Verification

Final GPU commands exited 0, **5 tests passed**, with distinct meanings:

- Borrowed fixture: one test, four SmolLM2-shaped strict GEMMs; 16 replays for
  each of explicit close and Drop, wrong-workspace recoverable rejection,
  finite/eager byte parity, input/weight preservation, no allocation growth.
- Existing C05-21: two lifecycle/parity regression tests.
- Canonical model: seven projections x two layers x four steps = 56 cold
  comparisons / 224 replays, scratch restored; full logits, initialized KV,
  continuation `[2,2,2,2]`, and zero outstanding allocations.
- Actual SmolLM2: policy rejection before mutation at all four steps, healthy
  executor, stable allocations, full logits/KV and continuation
  `[808,2775,288,536]`. **No SmolLM2 GEMM graph replay occurred.** Previous
  embedding/norm/RoPE/KV/pointwise audits also ran in both model cases.

CPU: CUDA library 83, graph contracts 32, runtime library 258, architecture 15,
inventory 1 passed. Format/diff checks passed. Normal Clippy exited 0; one new
`too_many_lines` warning remains for the explicit seven-mapping audit method,
in addition to baseline warnings. No clippy warnings were promoted to errors.

The source overlay/hashes and remote hash verification record tested source;
runtime identities record the GPU, binary and checkpoint-manifest hashes.
Model source preservation covers the unrelated existing model-loader edits.
Nothing was committed, pushed or deployed. High shared-host I/O pressure makes
these runs unsuitable for performance comparisons.

## Next implementation boundary

1. Add a separately identified graph contract that preserves the actual
   prepared policy and opaque algorithm, initially checking effective no-split
   topology. Keep legacy strict C05-21 semantics intact. Do not relabel the
   plan or select a different heuristic.
2. Define production zero-workspace ownership (without the cold sentinel) and
   nonzero shared-workspace spans, including leases and failure cleanup.
3. Prove actual SmolLM2 all-layer GEMM parity, then LM head, output/completion,
   retained aggregate and full decode graph. Matched vLLM comparison follows
   qualification; global inventory stays 7 Supported / 7 Unknown, aggregate Unknown.
