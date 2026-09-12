# SmolLM2 first-layer prefix correctness evidence

## Result

Actual remote CUDA runs completed for 1, 128 and 136 input tokens. All embeddings are byte-identical. The first difference against an isolated compiled replay of the loaded vLLM model is layer 0 RMSNorm: 149/576, 19072/73728 and 20291/78336 BF16 elements respectively.

An isolated Riley norm variant removes the intermediate BF16 rounding before multiplication by the norm weight. It exactly matches compiled-prefix norm outputs in all three cases and all recorded Q/K/V outputs for 1 and 136 tokens. This is diagnostic evidence, not an adopted runtime change. The canonical candidate binaries remain unchanged.

For the 128-token case, with equal uncompiled norm inputs, QKV behavior is independently sensitive to projection grouping and reduction settings:

| vLLM isolated prefix projection | Q unequal | K unequal | V unequal |
| --- | ---: | ---: | ---: |
| Packed QKV, default reduction | 29312 | 8064 | 11136 |
| Packed QKV, BF16 reduced reduction disabled | 24576 | 0 | 0 |
| Separate Q/K/V, default reduction | 0 | 0 | 0 |
| Separate Q/K/V, BF16 reduced reduction disabled | 24576 | 0 | 0 |

Counts compare with canonical Riley. This controls projection grouping and the PyTorch reduction flag; it does not identify a specific cuBLAS algorithm or prove its internal accumulation sequence. Disabling reduced precision alone does not guarantee matching outputs.

## Evidence boundaries

- Riley uses `PreparedLlamaForward` with explicit reference attention. The five exported tensors are all before the first attention operation. This is not a paged decode history trace.
- vLLM uses the actual loaded model through `worker_extension_cls`, with a scoped `enable_torch_wrap(False)` context to replay the selected native prefix under Inductor. This is not a tensor tap inside the full compiled serving graph.
- `eager` labels an uncompiled prefix call, not a complete `enforce_eager=True` serving run. The `*_fp32` labels disable BF16 reduced-precision reduction; tensors remain BF16.
- Exposed-output and opaque compiled QKV results are byte-identical. Default full-model generation before and after the diagnostic produced the same 32 tokens. No insecure callable serialization was enabled.
- Each Riley CUDA test passed with three cases and zero remaining device/pinned allocations. Both canonical and norm-variant test logs are retained.
- No performance trials ran. Cross-engine 32-token qualification remains blocked at zero-based index 8 from prior evidence. The norm-only variant does not establish complete output equivalence.
- GPU memory remained 743 MiB with unrelated Blender processes present; exclusive-GPU preflight remains unavailable. Those processes were not stopped.

## Sources and reproduction

Canonical source: `/tmp/riley-g04-followup-source-260911`, commit `9b53ffa14cea7c066fda8eff65b977c860b12afb`, still clean. Additive trace source: `/tmp/riley-g04-prefix-source-260911` at the same base, with only the new `prefix_trace_gpu.rs` test. `source-binary-hashes.txt` records both test copies and preserved release binaries. The test SHA256 is `b55172a340d01a1dfd0aa419a08cbfba853d9437447909ad6084fa6ca6f671fb`.

The norm variant uses `/tmp/riley-g04-norm-experiment-260911`; `norm-base.txt` and `norm-variant.patch` preserve its older base and full tracked delta. `norm-vs-candidate.patch` isolates its primitive kernel difference from the current candidate. It was not applied to the main workspace.

Remote evidence root: `/tmp/riley-g04-prefix-trace-260911`. Python: `/data/riley-vllm-interim.CfrT9T/venv/bin/python`. vLLM 0.27.1, torch 2.13.0+cu130, CUDA 13, RTX 4090. The capture script contains explicit model/checkpoint paths and generation parameters. Raw tensor files are little-endian BF16; the comparison script is standard-library-only.

```sh
# Recompute all comparisons from the included raw tensors, locally:
python3 benchmarks/results/20260911-g04-prefix-trace/compare.py

# On the remote host, rerun vLLM diagnostic with the local worker extension:
cd /tmp/riley-g04-prefix-trace-260911
PYTHONPATH="$PWD" /data/riley-vllm-interim.CfrT9T/venv/bin/python capture.py
```

The Rust ignored test requires `RILEY_REAL_CHECKPOINT`, `RILEY_TRACE_CASES`, and `RILEY_TRACE_OUTPUT`; run `cargo test -p riley-runtime --features cuda --test prefix_trace_gpu --release -- --ignored --nocapture` with the CUDA 13 toolkit/runtime and source-specific target directory configured.

## Remaining gate

Use these controls to trace downstream operations on the same common prefix and establish a coherent numerical policy that passes full generated-token qualification. Then revalidate that policy on the integrated decode graph. Only after that and exclusive-GPU preflight can the matched performance campaign run. No measurement-readiness claim is made by this artifact.
