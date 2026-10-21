# SM80 bounded Gate/Up candidate (v225)

This seven-file patch is the Gate/Up-only candidate, separated from v235's unselected merged-QKV recipe. It is now integrated as an explicit opt-in mode. Fresh-binary serving checks passed; defaults are unchanged. This remains an opt-in candidate, not a release gate pass.

From the original baseline, `git apply --check benchmarks/candidates/sm80-bounded-gateup-v225/candidate.patch` checks applicability. In the current working tree, `git apply --reverse --check` with that patch verifies the integrated delta.

The explicit `bounded-gateup-sm80-v1` mode keeps Gate and Up separate and groups singleton decode rows into GEMM calls of at most 16 rows. It accepts at most 32 simultaneous sequences, the qualified Qwen3-4B/Mistral7B geometries, SM80/108SM, CUDA runtime 12040 and cuBLAS 120405, with unmodified workspace/MPS settings. Unknown environments fail closed. Existing modes and defaults are unchanged. Q/K/V, O, Down, LM head, prefill extents, and intermediate BF16 boundaries retain their existing arithmetic.

Source provenance and the current binary are bound in the remote `candidate-provenance.json` and `cohort-provenance.json`; candidate binary SHA256 is `353d0ce2da8ec99ff15a3e39f16c83a8fd0debdd130b52dcf89f80dc25bd84b3`.

The retained source passed independent HF full-vocabulary teacher checks (104 rows/model), actual Graph row keys 1..32, component checks, and 518 CPU regressions. The new GPU6/7 cohort regenerates HF serving references and checks complete token/text/prompt/finish/usage values against fresh eager responses. These checks remain distinct from throughput evidence.

On the new GPU7 Mistral C8 diagnostic profile, Gate/Up kernel time fell from 1588.77 to 465.38ms and total kernel time from 3295.37 to 2170.37ms, with identical 12-request/240-output-token work and 45 correctness responses matching. These are profiler measurements, not unprofiled throughput. Two clean GPU7 Mistral C8 pairs improved unprofiled throughput from a median 70.38 to 109.10 tok/s (+55.0%), with all recorded latency quantiles improving in both pairs. Stock vLLM measured 436.32 tok/s at C8, so this does not establish parity with vLLM. Clean GPU7 Qwen with the independently rebuilt binary improved C8 from 109.81 to 152.35 tok/s (+38.7%); C1 was 84.92 to 84.71 tok/s. Qwen C8 TTFT p95 regressed about 2.7% (153.70 to 157.91ms), despite improved TPOT/E2E and TTFT p99. All-latency non-regression gates are therefore not passed; this mode is not a global default promotion. GPU6 profiles with the authorized small coexisting job are labeled diagnostic and excluded from clean adoption evidence.

See the [execution report](../../../docs/a100-execution-path-and-bottleneck-20261019.md) and [compact profiles](../../evidence/20261019-w0-a100/gpu67/).

An independent build from the exact 935-file candidate manifest produced SHA256 `844dde2bac465c19f7540c6546575808a6389466ffed9813915c345be2cd4391`; all 935 manifest files match the current working tree. It passed 514 CPU tests in EFS and the remaining four C02 fixture tests with the user-approved private `/tmp` exception and the original `umask 077` (518 passed total, 40 existing ignored tests). The temporary fixture tree was removed. The new binary also passed Mistral C8 serving equality at 109.11 tok/s. Clean GPU7 Qwen before/after serving checks also passed all native response-value and HF-token comparisons; performance limitations are recorded above.

Stock Qwen GPU7 measured 115.70 tok/s at C1 and 656.92 tok/s at C8 (two arms each), leaving a C8 candidate/stock ratio of 0.232. Both holders remain idle with 20GiB reserved after experiments; see the final hold receipt.

A subsequent [Qwen TTFT investigation](../../../docs/qwen-ttft-p95-investigation-20261019.md) measured the existing explicit `--sampling-backend gpu-greedy` option on the same binary. Two matched 96-request C8 pairs reduced TTFT p95 from 124.63 to 121.47ms (median) and improved throughput from 153.52 to 158.57 tok/s. TTFT p99 non-regression remains unproven; defaults stay unchanged. These samples are separate from the earlier 48-request screen.

The subsequent [SM80 projection v2 layer](../sm80-bounded-projections-v2/README.md) adds a separate mode7 above this patch. On a stacked tree, use its reverse-check or exact source manifest; the v225-only reverse check does not validate the whole stack. Mode6 behavior is retained.
