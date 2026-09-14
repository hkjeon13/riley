# Wide verification and compact completion

Measured prototype bottlenecks were four-owner fragmentation (63 model calls
versus 36 serial calls on the 12-request generation gate) and a full normal-logit
copy of 3,145,728 bytes per verification. This opt-in batch admits up to 32
owners and 256 query positions, including owners without a draft, and returns
4,096 bytes of normal completion plus 4,096 bytes of verification records.
Ordinary decode retains its 32-row head; verification uses a separate captured
256-row head. Runtime execution remains Rust → C ABI → CUDA.

| Correctness fixture | Serial calls | Wide calls | Accepted draft tokens | Output differences |
|---|---:|---:|---:|---:|
| 8 natural + 4 repetition controls | 36 | 33 | 144 | 0 / 384 |
| 8 natural + 24 repetition controls | 44 | 33 | 581 | 0 / 1,024 |

Both rows reproduce under full-model memcheck with zero memory errors. These
are generation correctness gates, **not serving benchmarks**. The artificial
controls do not establish representative acceptance or speedup. Fixed M256
head work may outweigh saved calls at low acceptance. The head has a distinct
numerical catalog identity; exact generated tokens here do not prove identical
full BF16 logits on every workload.

Native selector: 20 cases, two replays each, normal/memcheck/racecheck pass.
SM89, SM90a and SM100a compile; only SM89 runtime tested. Native packet rejection
passes ten corruptions under ASan/UBSan. Scheduler library: 70 tests pass. Wide
wire contract tests: 20 pass. Wide record parser tests cover slot and padding corruption through position 255.
No full-model racecheck was run. All three Blender RPC processes were restored
following both model gates. Source hashes match the tested remote checkout.

The serving CLI does not select this path yet. Shared-prefix verification is
still rejected. Remaining work: shared-prefix ownership/COW, opt-in serving
integration and serial fallback, then matched serving throughput/TTFT/TPOT/tail
measurements against the existing Riley and vLLM baselines. Query-size buckets
remain a profiling decision. No default promotion or new vLLM speedup is claimed.

Recheck retained evidence with:

```sh
python3 benchmarks/analysis/verify_speculative_wide.py
```
