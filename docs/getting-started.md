# Getting started

This page takes you from a fresh checkout to a streaming completion.

## Requirements

| | Tested with |
|---|---|
| OS | Linux x86_64 (Ubuntu 22.04) |
| GPU | One NVIDIA RTX 4090 (24 GB); kernels are compiled for `sm_89` by default |
| CUDA Toolkit | 12.8.1 (`nvcc`, cuBLASLt, cuBLAS runtime libraries) |
| Rust | 1.85.0, pinned by `rust-toolchain.toml` |
| Build tools | CMake 3.22 or newer, a C/C++ compiler |

Python is not needed to build or run the server.

Other GPU generations are untested. The compiled architecture list can be
changed with `RILEY_CUDA_ARCHITECTURES` (CMake syntax, for example `86;89`),
but every result in this repository was measured on `sm_89`.

## Build

```bash
export CUDA_HOME=/usr/local/cuda          # where nvcc and lib64 live
cargo build --locked --release --features cuda,server
./target/release/riley --version
```

The binary links cuBLASLt and the CUDA runtime dynamically, so they must be on
the loader path when it runs:

```bash
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

A checkout without a CUDA toolkit can still build and test everything that
does not touch the GPU:

```bash
cargo test --locked --workspace --no-default-features
```

## Prepare a checkpoint

Riley loads Hugging Face files directly: `config.json`, `tokenizer.json` and
safetensors weights. It refuses to load anything that is not listed, with its
byte length and SHA-256, in a `riley-checkpoint.json` manifest in the same
directory. The manifest also records which Hub commit the files came from.

Supported models:

| `model_type` | Architecture | Weights | Used in this repository |
|---|---|---|---|
| `llama` | `LlamaForCausalLM` | BF16 | SmolLM2-135M |
| `qwen2` | `Qwen2ForCausalLM` | BF16 | Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct |

1. Download the model files at a fixed commit, as regular files (not Hub cache
   symlinks). For Qwen2.5-3B-Instruct these are `config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json`
   and the two `model-0000N-of-00002.safetensors` shards.

2. Write the manifest. Its shape is:

   ```json
   {
     "format": "riley-checkpoint-v1",
     "source_model": "Qwen/Qwen2.5-3B-Instruct",
     "source_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
     "converter_revision": null,
     "transforms": [],
     "dtype": "bf16",
     "files": [
       { "path": "config.json", "bytes": 661, "sha256": "<64 hex digits>" },
       { "path": "tokenizer.json", "bytes": 7031645, "sha256": "<64 hex digits>" }
     ]
   }
   ```

   `source_revision` is the full 40-character commit. `files` lists the
   config, the tokenizer (plus `tokenizer_config.json` for Qwen) and exactly
   one weight layout: either `model.safetensors`, or
   `model.safetensors.index.json` with every shard it names.

3. Let the checkpoint tool copy the files into a new directory, check every
   hash and load the result with the same loader the server uses:

   ```bash
   cargo build --locked --release --manifest-path tools/riley-checkpoint/Cargo.toml
   tools/riley-checkpoint/target/release/riley-checkpoint prepare \
     --manifest plan.json --source /path/to/downloaded/files \
     --destination /models/qwen2.5-3b-instruct
   tools/riley-checkpoint/target/release/riley-checkpoint verify \
     --checkpoint /models/qwen2.5-3b-instruct
   ```

   The destination must not exist yet. See
   [`tools/riley-checkpoint/README.md`](../tools/riley-checkpoint/README.md)
   for the download mode and the exact guarantees.

## Start the server

A minimal start, with every option at its default:

```bash
./target/release/riley serve --model /models/qwen2.5-3b-instruct --model-id qwen3b
```

It listens on `127.0.0.1:8080`. The defaults favour the most conservative
numerical path. For Qwen2.5-3B, the configuration that produced the numbers in
[performance.md](performance.md) is listed in
[configuration.md](configuration.md#recommended-configuration-for-qwen25-3b).

Wait until the server is ready:

```bash
curl -s http://127.0.0.1:8080/readyz
```

## Send a request

```bash
curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3b", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

Add `"stream": true` to receive server-sent events. The endpoint takes raw
text: for a chat model, apply the chat template on the client. The full
request and response reference is in [api.md](api.md).

## Stop the server

Send `SIGINT` or `SIGTERM`. The server stops accepting requests, finishes or
cancels the active ones, releases its GPU memory and exits with status 0.
