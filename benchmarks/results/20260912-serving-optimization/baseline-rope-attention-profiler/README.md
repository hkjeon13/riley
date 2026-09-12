# Baseline combined RoPE/KV + attention diagnostic

This local tool prepares a matched operator interval for the HTTP-token API baseline, source `a179617070526068b66ba5627ba82a7151da8c64`. It encloses the existing packed RoPE/KV enqueue and the following packed two-warp attention enqueue in **one interval per selected decode layer**, named `rope_attention`. This can be compared with Batch8's fused interval after the separate build and runtime evidence is verified. No operator or serving timings have been measured by this preparation.

The source is pinned by `raw/http-token-build.json` (SHA256 `5576db88790d5a87e976c082df77f985595129b122fde55b2b475ce06e17a4f5`). Its native recorder is byte-identical to the frozen Batch7 archive member, SHA256 `8586726646729333e9bcea17c419b530aeb30e47cc1727189b701d9468d5dad5`. The archive supplies the local test fixture; it does not replace proof of the complete API baseline source or binary.

`receipt.json` pins the new tool, tests, generated diagnostic source, baseline build receipt, source archive, historical helper and parent profiler. The generated source is a diagnostic artifact; production files and frozen profilers remain unchanged. The tool verifies the native source and baseline receipt bytes. A separate builder must verify the complete isolated source revision, build identity and resulting binary.

## Use in a fresh isolated source copy

```sh
python3 benchmarks/scripts/profile_decode_rope_attention_baseline.py instrument \
  --source-root /path/to/isolated-api-baseline \
  --baseline-build /path/to/http-token-build.json
```

Preview is the default. Add `--apply` only for the isolated copy. The tool rejects unknown or already instrumented native source, the tooling checkout, escaping paths, symlinks and shared source inodes. Historical prefill projection hooks are unsupported.

Run each mode in a fresh process, with matching model, request, binary, runtime and GPU conditions:

```sh
RILEY_OWNED_GRAPH_PROFILE=1 RILEY_DECODE_OPERATOR=off
RILEY_OWNED_GRAPH_PROFILE=1 RILEY_DECODE_OPERATOR=rope_attention RILEY_DECODE_OPERATOR_LAYERS=all
```

These are environment settings for the separately controlled server or profile process, not launch commands. Use off/selected/off observations to assess drift and event perturbation. `--baseline-log` accepts an off log from this same diagnostic build; the tool does not itself establish matched runtime conditions between processes or between baseline and Batch8.

```sh
python3 benchmarks/scripts/profile_decode_rope_attention_baseline.py summarize \
  --log /path/to/selected.log --baseline-log /path/to/off.log
```

## Preserved behavior and measurement boundary

The original two enqueue statements, arguments, status guards and order are unchanged. Only their outer interval endpoints are inserted. Nonpacked and prefill branches retain their original statements; prefill receives zero operator event nodes. Selected decode adds two external event nodes per layer, or 60 across all 30 layers. Off adds zero captured timing nodes. The existing whole-graph helper still adds its host timing events when profiling is enabled.

There is exactly one existing stream synchronization and no added synchronization. Events are allocated before capture, retained on partial preparation failure, read after the original synchronization, and closed after graph handles under the owner's current CUDA context. Capture IDs and positions bind decode results; failed or missing intervals leave the replay sum null. The summary sums layer intervals within each replay before computing its median. It never substitutes the sum of separately measured RoPE and attention medians.

The 12 CPU tests pass, including three C++17 stub executions for event lifecycle, composed helper/inventory, and the actual generated enqueue region with each failure boundary. These checks do not compile against CUDA, execute a GPU, establish numerical parity, or qualify serving performance. Actual native build, full output validation, paired runtime conditions and perturbation measurements remain external work.
