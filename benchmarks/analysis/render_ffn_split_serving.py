"""Render only the fully verified split-FFN comparison; never promote defaults."""
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);data=json.loads((root/'comparison.json').read_text());verified=json.loads((root/'archive-verification.json').read_text())
assert data['retained_requests']==131072 and len(data['per_run_measurements'])==16
assert verified['archives']==17 and verified['all_archive_hashes_verified'] and verified['all_original_file_hashes_match_materialization'] and verified['lifecycle_and_restoration_passed']
lines=['# Split FFN graph serving — C32','',
'RTX4090, SmolLM2-135M BF16, context1024, client/active32, chunk512, prefix512, KV payload720MiB per engine. Each lane uses256 warmup +8192 retained requests. Shared reuses32 prompts; unique has distinct first16-token pages. Four engines run in each order for each workload.', '',
'Prior is the frozen unified adaptive FFN binary. Control uses original M16 FFN in the current binary. Split selects original M16 below192 total packed rows and separate M32 kernels at192 or above. All Riley lanes enable the same projection pipeline and rolling decode. The runtime remains Rust → C ABI → CUDA; Python is the external measurement client.', '',
'The first attempt stopped on vLLM readiness timeout and is preserved in [failed-v1](failed-v1/README.md). These results use a fresh complete V2 run with a common600-second readiness limit for every engine. No V1 partial measurements are merged. Client GC is disabled only during timed phases, and response evidence is spooled to tmpfs.', '',
'## Serving results','',
'Values are medians of two run-level estimates, not pooled percentiles. Latencies use actual HTTP SSE arrivals without interpolation.', '',
'| Workload | Engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
keys=['throughput_tokens_s','ttft_0.5_ms','tpot_0.5_ms','e2e_0.95_ms','e2e_0.99_ms','itl_0.95_ms','itl_0.99_ms']
for row in data['comparison']:lines.append('| '+row['workload']+' | '+row['lane']+' | '+' | '.join(f'{row[k]:.3f}' for k in keys)+' |')
lines+=['','## Descriptive changes','']
for kind,assessment in data['descriptive_assessment']['workloads'].items():
 for baseline,change in assessment['comparisons'].items():
  regress=[k for k,v in change['reported_latency_change_percent'].items() if v>0]
  lines.append(f"- {kind}, split versus {baseline}: throughput {change['throughput_change_percent']:+.2f}%; increased latency metrics: {', '.join(regress) or 'none'}.")
lines+=['','These ratios do not establish statistical significance, tail stability or overall qualification. Inspect both orders and host pressure below. Defaults are unchanged; future architecture runtime, other concurrency levels and open-loop sustained loads are not proven by this C32 run.','',
'## Run-level measurements and host pressure','',
'| Run | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P99 ms | IO some % | IO full % |','|---|---:|---:|---:|---:|---:|---:|']
for row in data['per_run_measurements']:
 psi=data['host_pressure_by_retained_lane'][row['name']]['fraction_percent']
 lines.append(f"| {row['name']} | {row['throughput_tokens_s']:.3f} | {row['ttft_ms']['0.5']:.3f} | {row['tpot_ms']['0.5']:.3f} | {row['e2e_ms']['0.99']:.3f} | {psi['io_some']:.2f} | {psi['io_full']:.2f} |")
lines+=['','Host PSI is global and is not causal attribution to an engine. Readiness and client preparation are outside timed request phases.','',
'## Startup observations','',
'| Run | Launch to readiness ms | Server RSS KiB | Global GPU memory and temperature at readiness (nvidia-smi) |','|---|---:|---:|---|']
for name,row in data['startup_by_lane'].items():lines.append(f"| {name} | {row['ready_wall_ms']:.3f} | {row['server_rss_kib']} | {row['gpu_at_ready']} |")
lines+=['','Global GPU readings include display allocations. These are not isolated CUDA graph allocation measurements, and launch time includes dependency/model loading and compilation.','',
'## Verification','',
f"All17 archives/{verified['files']} regular files/{verified['uncompressed_bytes']:,} uncompressed bytes passed archive and materialization hash checks, snapshot identity, lifecycle restoration and credential-pattern scans.", '',
'The semantic exporter checks131072 retained requests/4194304 output tokens and4096 warmup requests;98304 retained Riley responses must match the frozen reference. It verifies all16 exits, backend flags, KV capacity, rolling decode/drains, SSE-derived statistics,32 timed-phase GC receipts and96 each stop/cancel/recovery cases. Stored rows do not preserve the raw `[DONE]` marker; protocol completeness also relies on the hash-bound client. vLLM reference agreement is reported separately in comparison.json.', '',
'Reproduce with `verify_ffn_split_archives.py`, then `export_ffn_split_serving.py`, then this renderer against the result directory. The prior [model gate](../20260914-ffn-split-model/README.md) remains a separate full-logit and memcheck proof.','']
lines += ['## Decision and next optimization area','',
'Split adds less than1% throughput over the frozen unified adaptive backend in both workloads in this run. Against the current M16 control, unique improves6.98%, but shared improves only0.41% and changes direction across orders (+0.93%/−0.10%). All reported run-median latency metrics improve versus the unified prior, but candidate throughput remains13.77%/16.83% below vLLM. No broad performance goal is met.', '',
'The four-lane median launch-to-readiness is4757.68ms for split versus1002.17ms for control. Global GPU memory readings are1916 versus1900 and median server RSS is803986 versus795000KiB. These show a startup/resource cost in this experiment; global memory does not isolate graph allocations. Host IO PSI some ranges4.37–27.37% during retained phases, so small gains are not treated as stable causal proof.', '',
'Do not promote split by default and do not continue FFN threshold/tile micro-variants. Preserve the frozen binary and evidence as an experimental comparison point. The next meaningful area is the remaining PR06 attention work distribution: assess actual decode/prefill shape occupancy, bounded scratch/graph ownership, and arithmetic-order compatibility before implementing a linked batch. The existing128-token online-softmax recurrence and BF16 probability rounding mean mathematical split/merge equivalence does not prove the current numerical contract. An executed numerical failure stays a failure; tolerances are not relaxed after observing results.', '',
'Blender scene RPC on9876/9911/9887 was checked again after measurement, and local viewer endpoints31840/31970/32010 returned200. This is service verification, not a new public visual check.', '']
(root/'README.md').write_text('\n'.join(lines))
