# Proposed long-term goal

Complete Riley as a GPU HTTP serving engine competitive with vLLM while preserving model-specific correctness and lifecycle ownership guarantees.

1. Reconcile source/binary/model/tokenizer/environment/raw evidence before continuing; preserve rejected candidates and archives.
2. Fix and separately validate SmolLM2 performance and Qwen cache-on ordinary-HTTP correctness under fixed numerical policies, sampling/EOS and actual input/output lengths.
3. Require all C1/C8/C16/C32 fixed/natural cells to meet throughput at least vLLM, TTFT/TPOT no worse, and no P95/P99 regression. A favorable subset is never overall completion.
4. Verify sustained load, cancellation, re-request, KV reuse and complete owned-resource reclamation before accepting a candidate.
5. Repeat baseline → profiling → measured bottleneck analysis →2–5 related changes → correctness → actual HTTP benchmark. Separate kernel/transfer/host/scheduler/HTTP measurements and do not add overlapping intervals.
6. Freeze repetitions/order and preserve every failure; do not discard slow samples, enlarge tolerances or shorten output to obtain speed.
7. Report absolute V52/candidate/vLLM metrics, errors, memory, repeat SD and raw evidence. Separate version changes and correctness-only results from serving results.
8. Minimum completion requires every core performance cell plus model correctness and stability. Then pursue throughput+15% and TTFT/TPOT each10% lower as a distinct stretch outcome across the complete core matrix.

Current priority is to secure reproducible host admission, complete Batch12's full matrix, then select measured remaining gaps. C32 natural throughput/TPOT/tails and C32 fixed/C1 fixed TPOT were outstanding in the last complete Batch11 screen; Batch12 does not yet establish their status.
