# Owned graph cost diagnosis

`benchmarks/scripts/profile_owned_graph.py` instruments a **separate source copy**.
It does not change the production checkout, build CUDA, run a workload, or provide
a performance comparison. Run the script from the original tooling checkout so
its refusal to patch that checkout remains effective.

```sh
python3 /path/to/tooling-checkout/benchmarks/scripts/profile_owned_graph.py instrument \
  --source-root /path/to/isolated-diagnostic-copy --apply > instrumentation.json
```

Build the isolated copy with the same CUDA/release configuration as the fixed
baseline. Enable diagnostic output for the existing engine/HTTP runner with
`RILEY_OWNED_GRAPH_PROFILE=1` and retain its stderr. The binary requires no extra
build flags; the script inserts the diagnostic code in the copied source only.
The JSON receipt records the original and instrumented file hashes. Existing
instrumentation or changed source anchors are rejected before any write.

```sh
python3 /path/to/tooling-checkout/benchmarks/scripts/profile_owned_graph.py summarize \
  --log diagnostic-stderr.log --prefill-tokens 128 > diagnostic-summary.json
```

Both the original single-graph launch and the specialized dual-graph launch are
supported. The script checks the exact launch/selection/capture-transfer seams;
unknown or ambiguous variants fail before writing. Its receipt identifies the
source topology. A dual owner preserves the first capture ID with the prefill
graph. Each replay selects its diagnostic ID from the **actual launched exec
handle**, and the subsequent read record retains that same ID. Summary groups
include their observed capture IDs so phase-to-graph selection is inspectable.

The batched P128 recorder is also supported, with receipt topology
`dual_graph_prefill128_v1`. Its whole-graph mode uses the same command above;
omit `--projection-events` when measuring the complete graph span. The selector
uses the newly validated row count, and diagnostics still identify the actual
launched exec handle. For P128/O32 one request produces one prefill replay at
position 127 and 31 decode replays at positions 128..158. Summary `replays`
counts launches, not tokens: the prefill span covers all 128 prompt tokens.
Compare per-request phase sums with M1's 128 prefill replays; comparing their
per-replay prefill medians would compare different amounts of work. Position
classification is valid for this fixed contract but is not a scheduler trace.

Each graph capture emits total, kernel, memcpy, and other node counts. These are
the nodes in the retained graph, not a sampled count or proof that a kernel did
useful work. Inventory failure is an explicit nonzero CUDA status. Capture IDs
are process-local; summarize each process log separately. Linear memcpy node
parameters also give exact H2D, D2H, and D2D byte totals. CUDA array operands,
unknown directions, overflow, or parameter-query failures make byte totals null
with `memcpy_bytes_complete=false`; array element counts are never treated as
bytes. Historical records without byte fields remain accepted by the summary.

Each accepted native replay emits its position, host staging bytes/time, host
graph launch time, existing host stream-completion wait time, and a CUDA event
span around the graph. One additional record times the completed pinned-to-Rust
output memcpy. `host_staging_bytes` includes the complete input buffer copied by
the native API; it is **not** the graph's GPU H2D byte count. CUDA graph H2D/D2H
copies are included in the event span. The host metadata pack preceding the FFI
call remains unmeasured. Native validation rejection precedes this timing scope
and must remain in the runner's failure log.

The whole-graph event pair is created per diagnostic replay, with setup time reported
separately, and destroyed afterward. Event recording and stderr output add
overhead. The event span can include submission gaps; it is not summed kernel
execution time. The original single `cudaStreamSynchronize` remains in place:
there is no added per-kernel synchronization and no change to the captured
arithmetic DAG. A timing/launch/completion failure produces a null GPU span,
not zero. The default mode does not measure each kernel's duration.

## Optional first-layer projection intervals

Add `--projection-events` when instrumenting a fresh source copy to include the
optional projection probe in the same diagnostic binary. `RILEY_OWNED_GRAPH_PROFILE=1`
still enables diagnostics. At process startup, set
`RILEY_OWNED_GRAPH_PROJECTION=q`, `gate`, or `down` to select one projection from
layer 0. Omit that variable or set it to `off` for ordinary whole-graph timing
with no projection event nodes. Run each selection in a fresh process because
it fixes the captured topology; no rebuild is needed between selections.

For the selected projection, the original graph records intervals around both
the canonical GEMM and prefill overwrite. The specialized graph records only
its actual operator. Each interval adds two external event-record nodes, so the
original topology adds four nodes and each specialized topology adds two.
Event objects are allocated before capture, retained by the aggregate owner,
queried only after the existing completion synchronization, and destroyed after
both graph/exec pairs. Partial event allocation is cleaned up by the same owner.
There is no additional stream or event synchronization.

Captured timing uses `cudaEventRecordWithFlags(..., cudaEventRecordExternal)`
so replay records actual event timestamps. These are instrumented stream
intervals, not isolated kernel cycles: event nodes and other work may alter them.
Compare the normal whole-graph profile against this mode to quantify its overall
perturbation. Tiny intervals approach event timing resolution and should not be
overinterpreted. See NVIDIA's [event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).

Each `projection` record identifies the capture/replay, position, layer,
projection, canonical/override operator, and elapsed interval or explicit null.
The summary groups these intervals by position phase and operator. Missing
operators are absent, never reported as zero-duration work. A baseline override
at decode positions still measures its real immediate-return kernel launch.

## Hypotheses after removing canonical prefill GEMMs

Deleting work can expose another operation's memory cost. The original canonical
GEMM reads the same weights immediately before the overwrite kernel, potentially
warming cache lines that the overwrite then reuses. Without it, the overwrite
may encounter colder weights and more visible memory latency. This is a source
hypothesis, not established attribution of the observed HTTP regression.

The overwrite kernel launches four warps per block and 32 output columns per
block: Q/down with 576 columns use 18 blocks; gate with 1536 uses 48. Such a small
grid may expose memory latency because it supplies limited parallel work to a
large GPU. Its per-warp weight-load pattern, achieved occupancy, and actual cache
hit rates have not been measured. A useful follow-up is a numerically identical
one- or two-warp block variant, preserving each warp's MMA/reduction sequence,
then measuring exact-output parity and serving results. These are candidate
experiments, not a claim that occupancy or coalescing explains the regression.

Clock/power state or differing runtime conditions could also change small-kernel
latency. Use matched telemetry and all repeated pairs, then compare baseline's
canonical-plus-overwrite interval with the specialized overwrite interval.
Separate cache/parallelism hypotheses from proven costs, following NVIDIA's
[memory and execution guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Summary prefill/decode labels are inferred from the specified position boundary,
not scheduler observations. The log includes warmups unless the caller supplies
only measurement records; the summary does not silently drop requests. Use the
raw per-position records to distinguish prompt positions 0..127, first generated
output at position 127, and decode positions 128..158 for the P128/O32 cell.

The last verified G04 baseline has measured host TTFT and TPOT but no CUDA event
span. Its M=1 graph does 128 prompt and 31 decode replays per request. In numerical
profile 2, `graph_resources.cu` calls canonical GEMM and then a prefill overwrite
for seven projections per layer. The overwrite kernel exits immediately for
position >=128. For SmolLM2's 30 layers this is 210 duplicated GEMM calls for each
prefill replay and 210 no-op kernel launches for each decode replay. These counts
are source-derived; their elapsed cost share is not yet measured.

The next related optimization batch can specialize prefill/decode graph
topologies, remove the prefill canonical computations whose complete output is
immediately overwritten, and omit decode's prefill-only launches. Preserve the
existing arithmetic policy and use a new implementation identity. Both graphs
must share one aggregate resource owner, and runtime stage/position validation
must reject selection of the wrong graph. Recheck exact logits/tokens, KV block
mapping boundaries, cancellation/reuse, and graph/parent cleanup. Actual serving
performance must then be measured with an uninstrumented binary and the matched
baseline/vLLM workload.
