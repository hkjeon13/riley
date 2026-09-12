# SmolLM2 MLP tail: controlled-input correctness results

Status: 51 operation comparisons verified from raw BF16 data; cross-engine full-output qualification unresolved; performance trials 0.

## What was verified

The remote CUDA test exported seven layer-0 tensors for 1, 128 and 136 tokens. Each vLLM operation then received the corresponding **same Riley input tensor**, so upstream errors cannot explain these individual comparisons. This isolates operation behavior; it does not reproduce the end-to-end vLLM layer input distribution.

| Operation / control | 1 token unequal | 128 tokens unequal | 136 tokens unequal |
| --- | ---: | ---: | ---: |
| Post-attention RMSNorm, uncompiled call | 0 | 0 | 0 |
| Post-attention RMSNorm, isolated compiled call | 149 | 19584 | 20223 |
| Gate/up, packed or split, either reduction flag | 0 | 0 | 0 |
| Down projection, default BF16 reduced reduction enabled | 0 | 0 | 0 |
| Down projection, BF16 reduced reduction disabled | 0 | 26368 | 27156 |
| Selected vLLM SwiGLU, uncompiled or compiled | 404 | 50816 | 54892 |
| SwiGLU with BF16 intermediate SiLU result | 0 | 0 | 0 |
| Residual addition | 0 | 0 | 0 |

Norm/down/residual widths are 576; gate/up/SwiGLU widths are 1536. Gate/up table rows summarize eight comparisons per case. Full counts, maximum absolute errors and tensor hashes are in `comparison.json`.

Both selected vLLM SwiGLU calls exactly equal FP32 SiLU and multiplication followed by one BF16 conversion for these inputs. Riley exactly equals SiLU rounded to BF16 before the multiplication. `verify.py` independently verifies these byte identities, all 51 recorded comparisons, generation invariance and restoration of the reduction setting.

Here `swiglu_eager` means the actual loaded default model's selected activation called without an outer `torch.compile`. It does not mean native custom-op selection or a full serving run with `enforce_eager=True`. Those prior experiments used different selection settings. `*_reducedFalse` disables BF16 reduced-precision reduction; input/output tensors remain BF16.

## Scope and limitations

- The Riley trace executes explicit reference attention, then records its MLP inputs/outputs. It is not the paged decode graph trace.
- vLLM loads the actual checkpoint and uses a local worker extension. The scoped `enable_torch_wrap(False)` context permits isolated native-IR expression compilation. This is not a tap inside the full compiled serving graph.
- The standalone residual-add match does **not** verify the fused residual-add-and-RMSNorm path: the latter may normalize an unrounded sum. Attention, RoPE and that fused boundary remain to be compared directly on common inputs.
- Default vLLM generation before/after the diagnostic is byte-for-byte the same 32 token IDs. The previously observed Riley/vLLM output divergence at zero-based index 8 remains unresolved.
- This evidence does not justify changing an independent SiLU primitive into a pass-through operation, nor changing the global GEMM reduction setting. The latter introduced down-projection differences in this experiment.

## Source, environment and checks

Only the additive ignored test `crates/riley-runtime/tests/tail_trace_gpu.rs` and this artifact directory were added in this step. Runtime kernels and the qualified candidate were not changed. Remote candidate `9b53ffa14cea7c066fda8eff65b977c860b12afb` remains clean; release binary hashes match the previous qualification artifact. See `provenance.txt`.

Remote trace source `/tmp/riley-g04-prefix-source-260911`, source-specific target `/tmp/riley-g04-prefix-target`, evidence `/tmp/riley-g04-tail-trace-260911`. CUDA 13 / RTX 4090, vLLM 0.27.1 and torch 2.13.0+cu130, same SmolLM2 checkpoint and inputs as the preceding prefix artifact. `capture.py` pins the generation arguments and loads the existing checkpoint receipt.

The Rust GPU test passed for all three cases and asserted zero remaining device and pinned allocations. Rust formatting, Python syntax and `git diff --check` passed. Local CPU verification:

```sh
python3 benchmarks/results/20260911-g04-tail-trace/verify.py
```

Remote reproduction uses the CUDA 13 toolkit/runtime and:

```sh
# In the trace source, with source-specific CARGO_TARGET_DIR:
RILEY_REAL_CHECKPOINT=/data/riley-benchmark/20260827T051948Z-d7ad713a/model \
RILEY_TRACE_CASES=/tmp/riley-g04-tail-trace-260911/prefixes.json \
RILEY_TRACE_OUTPUT=/tmp/riley-g04-tail-trace-260911/riley \
cargo test -p riley-runtime --features cuda --test tail_trace_gpu --release -- --ignored --nocapture

cd /tmp/riley-g04-tail-trace-260911
PYTHONPATH="$PWD" /data/riley-vllm-interim.CfrT9T/venv/bin/python capture.py
```

GPU memory remained 743 MiB with the same three unrelated Blender processes. They were not interrupted. No preflight requirement or measurement gate was relaxed, and no performance campaign was run.

## Next implementation boundary

The prefix and tail results now distinguish projection grouping, norm rounding and activation rounding. The remaining numerical investigation must cover RoPE/attention and fused residual normalization before selecting a coherent profile. Any new profile must preserve existing HF primitive contracts, pass full generated-token qualification, and then pass integrated graph and HTTP lifecycle checks. Exclusive-GPU preflight remains required before measurement.
