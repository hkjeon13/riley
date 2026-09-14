# Explicit GPU verification stage and scheduler integration

The experimental scheduler now executes `[pending, draft...]` through a separate
verification stage (wire value 3). Original prompt length, generated count and
committed KV position are preserved. Ordinary native sessions reject this stage;
only the retained verification-head capability admits it. The adapter binds all
returned token positions to iteration, replay, owner generation, catalog digest,
request identity and cookie before host prefix settlement.

No Python participates in model execution. Python generated the offline control
fixture and orchestrated evidence capture/restoration.

| Fixture / revision | Output differences | Accepted draft tokens | Serial model calls | Speculative model calls | Result |
|---|---:|---:|---:|---:|---|
| 8 frozen natural requests, v1 | 0 / 256 | Not instrumented | 33 | 35 | Exact output, no performance claim |
| Natural + 4 repetition controls, v2 | 0 / 384 | 0 | 36 | 38 | **Acceptance gate failed** |
| Same 12 requests, v3 candidate selection fix | 0 / 384 | 144 | 36 | 63 | Exact output and real multi-token KV settlement |

V2 examined only the first four candidates and returned ordinary fallback on
missing drafts, hiding eligible later owners. V3 scans the selected decode
candidates for up to four eligible owners. A CPU regression test preserves this
behavior. V2's failing logs and zero-acceptance receipt remain in the archive.

These are generation correctness fixtures, **not serving benchmarks**. Call
counts are not throughput or latency measurements. Repetition controls must not
be used as representative workload performance. V3 increases total model calls:
its four-owner verification limit fragments the ordinary wider batch. The full
normal result still copies roughly 3MB in addition to the compact auxiliary
result. This implementation is not enabled by the serving CLI.

Validation: both normal and full-model memcheck runs reproduce the recorded
outputs; memcheck reports zero memory errors for v1/v2/v3 (v2 exits 101 from the
acceptance assertion). Rust request/wire tests: 54 passed, one existing ignored;
scheduler library: 69 passed. Native packet probe checks ordinary-session
rejection and ten corrupted fields under AddressSanitizer/UBSan. Selector
compiles for SM89/SM90a/SM100a; only SM89 executes. GPU racecheck is not rerun for
this host-contract change; no full-model racecheck is claimed.

All three Blender RPC endpoints were restored after every GPU gate. The three
viewer loopback endpoints returned 200 after v3; this is not public visual proof.

Reproduce evidence checks:

```sh
python3 benchmarks/analysis/verify_speculative_gpu_stage.py
```

Next optimization batch: widen verification across active owners, eliminate full
normal logits transfer, integrate shared-prefix ownership and serving selection,
then compare matched serving workloads against current Riley and vLLM. Keep
strict generation gates and retain serial fallback.
