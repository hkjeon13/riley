# Exact R2 global loads: Qwen3-4B, RTX4090

The reviewed six-file change adds an explicit `exact-r2-global-v4` option for at most two simultaneous sequences. Existing mode3 retains its GPU algorithms. CLI and runtime admission reject larger mode4 batches. This is a throughput option with a small measured TTFT tradeoff; it is not an all-model or stock-vLLM parity claim.

The Down and attention-output projections share each weight load across two independent request rows, without shared-input barriers. Each row retains its original FP32 FMA order, lane-reduction tree and final BF16 rounding. Original complete-prompt GEMM extents are retained. Existing exact R2 Gate/Up, borrowed validated CSR and HF head-normalization/RoPE/cache fusion remain. No new hot buffer or workspace allocation.

## Same-device finite comparison

One fresh matched native/native/stock triple: RTX4090 SM89/128SM, Qwen3-4B pinned, BF16, C2, context1024, tokenbudget512, KV128 blocks of16,22GiB process cap, prefix cache off, neutral greedy and original EOS. Natural prompt lengths16/128/397 and output caps32/64/128;48 measured requests and3584 output tokens per engine after16 warm requests and full-reference graph-count warm waves. No graph captures during measurement. Native CUDA12.8 versus stock vLLM0.27.1 CUDA13; same CPU affinity and OMP4.

| Metric | Before143 mode3 | After151 mode4 | Stock vLLM |
|---|---:|---:|---:|
| Output tok/s | 131.704 | 156.467 | 186.939 |
| TTFT p95 ms | 52.549 | 52.625 | 46.407 |
| TTFT p99 ms | 52.660 | 52.740 | 47.270 |
| Request-mean TPOT p99 ms | 14.782 | 12.302 | 10.187 |
| E2E p99 ms | 1922.873 | 1607.747 | 1326.787 |

Throughput improved 18.80%. TTFT p99 increased by 0.081ms; do not describe this as a zero-latency-regression pass. TPOT percentiles above are percentiles of per-request mean intervals, not token ITL. The earlier unrestricted149 algorithm repeated a throughput gain in all three fresh native pairs (median+18.89%), but its strict TTFT gate failed and C8 TTFT p95 regressed by3.91%;149 was not adopted. Mode4 is therefore explicit and limited to C1/C2. QKV direct output-placement147 was exact but gained only0.20..0.30% and was not adopted.

The 15-unique-prompt correctness set is repeated three times for C2: native45/45 token/text sequences matched pinned HF; stock39/45 matched. Measured natural requests also had equal token counts, while stock token sequences differed. These are speed comparisons under matching input/settings, not a general quality ranking.

## Numerical and actual-runtime evidence

- Same-SM89 real-weight component:72 selected Down/output weights,5184 owned-CAPI Graph conditions,changed-input replays,halos/leases and zero tracked hot allocations; qualified150 native/safe-GEMM sources are identical in151.
-151 whole-model HF104 full-vocabulary BF16 rows are byte-exact for C1/C2 mode4. Actual driver Graph inspection proves72 global R2 projection nodes. Existing mode3 C2/C8 is separately byte-exact and has zero global R2 nodes.
-Early hidden-GPU CLI test rejects mode4/C8 before startup; runtime owner also limits max_rows/max_output_slots to2. Original SM89/CUDA/cuBLAS/model geometry gates remain.
-Actual main snapshot:935 source hashes match the private151 snapshot. Rebuilt with pinned CUDA12.8 and Rust1.85. CPU tests:517 passed;40 ordinary GPU tests ignored and replaced by dedicated hardware gates, not counted as inference passes.
-Actual rebuilt main C1/C2 Graphs retain HF104 byte equality. Actual Qwen4B C2/mode4 and Mistral7B C8/mode2 HTTP servers passed195 full response values,owned-PID/memory accounting,zero measured captures and disconnect/retry recovery. These final runtime timings are not extra paired benchmark confirmations.

## Use

Use the qualified model and runtime with:

```sh
riley serve --model /path/to/pinned-Qwen3-4B --model-id m \
  --max-active-sequences 2 --max-sequence-tokens 1024 --max-output-tokens 128 \
  --kv-blocks 128 --numerics-profile owned-hf-eager-bf16-v1 \
  --execution-completion iteration-batch --owned-decode-graph segments-v1 \
  --owned-decode-batch exact-r2-global-v4
```

Larger batches retain the existing `exact-r2-v3` option. Mode4 has not been qualified on A100/SM80, different SM counts, CUDA versions or arbitrary model geometries. A100 GPU2/3 owners were rechecked after every finite GPU experiment; foreign work was preserved and A100/RTX results are kept separate. No deployment,commit,push or soak.

Artifacts: `artifacts/multimodel-parity-20261017/v151-serving-pilot-audit.json`, `v151-scoped-qualification-result.json`, `v151-adoption-receipt.json`, `v151-final-source-resource-receipt.json`. Remote root: `/data/riley-multimodel-parity-20261017` on `ssh ai-assistant`. Actual rebuilt binary: `/data/riley-multimodel-parity-20261017/riley-candidate-v151-integrated`,SHA256 `a63d11eb9a68eb9368ef6ef95aa9093a434951e6f793bdc0c1d4f651f4314889`.
