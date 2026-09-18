# riley-checkpoint — HF-R1

An independent Rust 1.85 workspace for **checkpoint preparation**, not serving.
Default builds have no Hub client. The production workspace and its lockfile are
unchanged. No Python, subprocess, CUDA initialization, tokenizer substitution or
numerical/profile change is introduced.

## Contract and commands

Obtain a reviewed `riley-checkpoint-v1` manifest independently of the download.
Its existing production schema is the explicit allowlist: `source_model`, a full
40-character lowercase commit `source_revision`, `dtype`, empty `transforms`, and
`files` entries containing `path`, `bytes`, and lowercase `sha256`. Config,
tokenizer and exactly one supported weight layout must be included. This tool
**does not trust-on-first-use downloaded hashes**, resolve branches/tags, guess
files, transform weights, or invent provenance. Metadata authenticity remains
an operator responsibility.

```sh
manifest=tools/riley-checkpoint/Cargo.toml
cargo build --locked --manifest-path "$manifest" --release
bin=tools/riley-checkpoint/target/release/riley-checkpoint
"$bin" prepare --manifest /trusted/qwen-plan.json \
  --source /trusted/regular-files --destination /models/qwen-new
"$bin" verify --checkpoint /models/qwen-new
```

Source and destination ancestors must be real directories, not symlinks. Source
files must be regular files, not Hub snapshot links. The destination parent must
exist; the final directory **must not exist**, even as an empty directory or a
partial previous attempt. The tool never overwrites/resumes an existing target.

Online transport is separately compiled and separately authorized:

```sh
cargo build --locked --manifest-path "$manifest" --release --features hub
"$bin" download --manifest /trusted/qwen-plan.json \
  --destination /models/qwen-new --allow-network
# For gated models, optionally append --token-env HF_TOKEN.
# Set the secret outside shell history; its value is never a CLI argument.
```

`HF_HUB_OFFLINE=1` forbids download, even with `--allow-network`. `prepare` and
`verify` never access Hub, including in a Hub-enabled binary. Endpoint, cache
location and ambient cached credentials are not configurable: only the official
Hub endpoint and a fresh private transport cache are used. No implicit `.get()`
network fallback occurs. Errors omit transport URLs and credential details.

## Atomicity, ownership and filesystem limits

The tool atomically reserves the destination with an exclusive directory create
(mode 0700 on Unix). An owned private staging directory receives streamed copies
(mode 0600), checked against the expected byte length and SHA-256. The **actual
`LoadedModel::load`** then validates dtype, config, tokenizer, weights and exact
manifest coverage. Hash correctness alone is insufficient.

Validated payloads are linked from our own stage into the reserved destination;
`riley-checkpoint.json` is linked **last** as an atomic create-only commit marker.
External source files are copied, not hard-linked. Existing files cannot be
replaced. Files and directory entries are synchronized on Unix. A pre-publication
error cleans only the destination successfully reserved by this invocation.
A process kill may leave a partial directory without a manifest: it is not a
checkpoint, and a later invocation refuses to reuse it. Inspect/clean that
orphan explicitly, or choose another destination. Once the manifest is visible,
subsequent sync/cleanup failure leaves the valid checkpoint intact; run `verify`
before serving it. A leftover private stage may need manual cleanup.

All input trees, the destination parent, and published output must be trusted
and remain immutable during use. Like the existing portable loader, observed
symlink rejection is **not** protection against hostile same-user concurrent
path replacement. Local filesystems supporting atomic exclusive directory
creation and same-filesystem hard links are required. No atomicity claim is made
for arbitrary network filesystems; power-loss durability outside Unix is not
qualified. No automatic serving or deployment is performed.

The pinned `hf-hub` client normally produces snapshot symlinks on Unix. Inside
its freshly created private cache only, we decode the exact
`../../blobs/<40-or-64-lowercase-hex>` pointer and copy from the independently
identified regular blob. Arbitrary cache links, chained blob links, returned
revision/path mismatches and external cache import are rejected. **No symlink
is published or passed to the model loader.** Transport cache is removed before
publication. Download disk/time cost is not serving throughput, TTFT or TPOT.

## Validation

```sh
cargo fmt --manifest-path "$manifest" -- --check
cargo clippy --locked --manifest-path "$manifest" --all-targets --no-default-features -- -D warnings
cargo test --locked --manifest-path "$manifest" --all-targets --no-default-features
cargo clippy --locked --manifest-path "$manifest" --all-targets --features hub -- -D warnings
cargo test --locked --manifest-path "$manifest" --all-targets --features hub
```

Tests use the same tiny Llama artifact contract as
`crates/riley-model/tests/python_free_loading.rs`: successful production-loader
validation, source-copy independence, checksum/length failures, injected transport
interruption, hash-correct invalid model rejection, existing/partial destination
preservation, concurrent publishers, immutable revision and strict manifest
validation, symlink rejection, private Hub pointer confinement, and CLI errors.
No real model download, remote GPU run, performance gate or release promotion is
performed by these tests. Actual Hub/gated-model download qualification remains
a separate explicit operation. Rollback: remove this independent tool and its
CI lane; production binaries and runtime dependencies do not change.
