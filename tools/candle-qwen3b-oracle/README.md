# Candle Qwen2.5-3B numerical triangulation

This is an offline-only Rust diagnostic built on Hugging Face Candle. It is
not a Riley serving backend, has its own nested Cargo workspace and lockfile,
and is never linked into the root workspace or the server binary.

It accepts only the immutable N06-A Qwen2.5-3B P2048 workload and its pinned,
local Qwen checkpoint. It feeds the recorded `u32` token IDs directly to
Candle, writes one create-only JSON raw-logit fingerprint outside the checkout,
and records checkpoint, source, CUDA, and numerical-contract provenance.

The tool is intended to add an independent Rust implementation to the
investigation of the current full-forward discrepancy. It does **not** make a
serving-performance claim.

## Important numerical boundary

Candle 0.9.1 Qwen2 uses BF16 RoPE tables, BF16 softmax, and a Candle CUDA
RMSNorm path. The pinned Transformers eager reference uses F32 intermediate
paths for those operations. Candle also always writes its internal KV cache on
the first forward. Therefore its full BF16 row hash is a numerical
triangulation observation, never an HF eager equality gate. A mismatch must
lead to a tolerance or layer-trace investigation; it does not assign blame to
Riley, Candle, or Transformers.

This package is intentionally pinned to Candle 0.9.1 because it compiles with
the root project's Rust 1.85 baseline. Its CUDA dependency lane supports the
current CUDA 12.8 test host. A CUDA 13/Hopper/Blackwell qualification must use
a newer Candle lane after its MSRV is separately qualified.

## Build and run

Run the CPU-only parser and artifact tests without CUDA:

```sh
cargo test --manifest-path tools/candle-qwen3b-oracle/Cargo.toml --locked
```

On a qualified CUDA host, build and run only the diagnostic. The output parent
must already exist and be outside the checkout. The checkpoint must stay
immutable through the run; the two huge shards are receipt/size checked and
then memory mapped, rather than being re-hashed a second time.

```sh
export PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:$PATH
export CUDA_HOME=/data/cuda-12.8.1
export CUDA_ROOT=/data/cuda-12.8.1
export CUDAToolkit_ROOT=/data/cuda-12.8.1
export CUDACXX=/data/cuda-12.8.1/bin/nvcc
export CARGO_BUILD_JOBS=1

cargo run --locked --manifest-path tools/candle-qwen3b-oracle/Cargo.toml \
  --features cuda -- \
  --checkpoint /absolute/path/to/Qwen2.5-3B-Instruct \
  --workload /absolute/path/to/qwen3b-c8-p2048-o128.json \
  --repo-root /absolute/path/to/riley \
  --output /absolute/external/results/qwen3b-candle-raw-logits.json \
  --device 0
```

The binary checks compute capability 8.0 or higher before opening CUDA. It
does not start a server, open a network connection, tokenize text, sample, or
benchmark throughput.
