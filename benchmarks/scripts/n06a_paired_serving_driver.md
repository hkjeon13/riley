# N06-A paired Qwen2.5-3B D128 serving driver

`n06a_paired_serving_driver.py` is an offline benchmark controller. It does
not add a Python path to Riley: it starts an owned Rust `riley serve` process
and an owned vLLM process one at a time on loopback, sends OpenAI
`POST /v1/completions` streaming requests, then cleans up each process before
starting the other lane. Its only stdout on a successful timed attempt is the
two `riley-n06a-d128-serving` records read by
`n03b_n06a_d128_repeat_summary.py`.

Use it through `n01_repeat_control.py`. N01 passes
`N01_REPEAT_CONTROL_PHASE=warmup|timed` and a one-based
`N01_REPEAT_CONTROL_INDEX` to its child without modifying the parent shell.
Timed odd attempts run Riley then vLLM; even attempts run vLLM then Riley.
The driver rejects a direct timed invocation without those variables, so an
order cannot be chosen accidentally.

Each fresh server runs a strict per-server warmup phase after readiness. That
is separate from N01's outer warmup process, which can warm host caches but
cannot warm a server that is started and stopped inside a later timed process.
Only the retained phase produces marker metrics. All raw request rows, the
warmup rows, startup snapshots, lane commands, log hashes, and cleanup receipts
are written create-only under `--artifact-root/<phase>-<index>/`.

The driver accepts only `CUDA_VISIBLE_DEVICES=0` and requires the Docker vLLM
argv to contain exactly `--gpus device=0`. This makes Riley's `--device 0`,
the Docker container, and the host `nvidia-smi --id=0` sampler refer to the
same physical GPU. A sampled peak is post-lane eligibility evidence, not a
live allocation limiter: a sample above the 19,000,000,000-byte ceiling causes
the lane to retain its failure receipt and prevents markers, but cannot undo a
transient allocation that has already occurred.

## Required workload artifact

The workload is a bounded, immutable JSON document. It must be prepared from a
pinned Qwen2.5-3B reference before a comparison; do not invent the raw IDs or
reuse the 0.5B fixture.

```json
{
  "schema_version": "riley.n06a-d128-serving-workload.v1",
  "case": "qwen3b-c32-p2048-o128",
  "model_id": "Qwen/Qwen2.5-3B-Instruct",
  "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
  "prompt": "<the exact rendered 2048-token prompt>",
  "prompt_token_ids": [151644, 872, 198],
  "output_token_ids": [151645, 198],
  "output_text": "<exact generated text>",
  "finish_reason": "length",
  "sampling": {"temperature": 0.0, "top_p": 1.0}
}
```

The shortened arrays above are schematic; the real artifact contains exactly
the pinned prompt/output ID counts. Arrays contain JSON integers, not strings.
The driver requires the returned
first-chunk prompt IDs, every output ID, text, terminal usage, and `length`
finish reason to exactly match this document. It also requires one generated
ID in every SSE event. A grouped, partial, missing-usage, or mismatched stream
is retained as a failure and cannot produce a paired marker.

## Required model identity manifest

Every invocation requires `--model-identity-manifest` and, unless the local
Git executable is elsewhere, `--model-git /usr/bin/git`. The manifest binds
the exact `--model-path`; it is copied create-only into each attempt together
with the result of `git -C <model-path> rev-parse HEAD` and
`git -C <model-path> lfs ls-files -l`. This avoids rereading multi-gigabyte
weights for every lane while still binding the revision and large-shard LFS
objects.

The JSON object has exactly these top-level fields. Paths are relative regular
files below `model_path`; replace every schematic value with the observed value
from the pinned local checkout.

```json
{
  "schema_version": "riley.n06a-model-identity-manifest.v1",
  "model_id": "Qwen/Qwen2.5-3B-Instruct",
  "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
  "model_path": "/absolute/path/to/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1",
  "metadata_files": [
    {
      "path": "config.json",
      "size_bytes": 661,
      "sha256": "<64-lowercase-hex-of-the-small-file>"
    }
  ],
  "shards": [
    {
      "path": "<relative-safetensors-shard>",
      "size_bytes": 3968658944,
      "lfs_oid_sha256": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2"
    },
    {
      "path": "<other-relative-safetensors-shard>",
      "size_bytes": 2203268048,
      "lfs_oid_sha256": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d"
    }
  ]
}
```

Each `metadata_files` entry has exactly `path`, `size_bytes`, and `sha256`; it
is rehashed on the host. Each `shards` entry has exactly `path`, `size_bytes`,
and `lfs_oid_sha256`; the driver checks its size and the Git-LFS long-listing
OID without rereading the full shard. Include every small metadata file and
every serving shard once. A manifest whose revision, local path, small-file
hash, shard size, Git HEAD, or LFS OID differs is rejected before a lane
starts.

## Docker vLLM command template

The vLLM launcher is a JSON argv array, never a shell fragment. It supports
`{port}`, `{model_path}`, `{model_id}`, `{concurrency}`,
`{batch_token_budget}`, `{max_model_len}`, `{max_output_tokens}`, and
`{kv_blocks}`. `--vllm-command-json` must use `{port}`, `{model_path}`, and
`{model_id}` exactly once. For the current Docker baseline, make a new external
file such as `/var/tmp/riley-n06a/vllm-v0.29.0.json`:

```json
[
  "/usr/bin/docker", "run", "--rm", "--gpus", "device=0", "--network", "host", "--ipc", "host",
  "-v", "{model_path}:/model:ro",
  "vllm/vllm-openai@sha256:<replace-with-the-64-hex-inspected-digest>",
  "--model", "/model",
  "--served-model-name", "{model_id}",
  "--host", "127.0.0.1", "--port", "{port}",
  "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
  "--enable-chunked-prefill", "--no-enable-prefix-caching",
  "--max-model-len", "{max_model_len}",
  "--max-num-seqs", "{concurrency}",
  "--max-num-batched-tokens", "{batch_token_budget}",
  "--gpu-memory-utilization", "0.65"
]
```

Before writing the template, inspect the pulled
`vllm/vllm-openai:v0.29.0-cu129` image and replace the placeholder with its
actual `sha256:<64-lowercase-hex>` repo digest. Pass that identical value to
`--vllm-image-digest`. The driver requires this immutable image pin once in the
argv, records the image digest and all non-default launch arguments in the
attempt receipt, and rejects an explicit attention-backend flag: vLLM's auto
selector must make the choice.

The semantic Docker envelope is fixed: the image argument is exactly
`vllm/vllm-openai@<digest>`; Docker receives one separate `--network host`
pair, one `--ipc host` pair, and one `--gpus device=0` pair; the only model
mount is `{model_path}:/model:ro`; and the image entrypoint receives exactly
`--model /model` and `--served-model-name {model_id}`. The driver injects its
own unique `--name`, requires `--rm`, and rejects `--detach`, another model
mount, a different image argument, or a forced attention backend.

vLLM 0.29 logs the actual selection in this source-verified shape. Pass this
exact regex, with exactly one named capture `backend_resolved`:

```text
--vllm-backend-receipt-regex 'Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:'
```

The driver snapshots both vLLM stdout and stderr immediately after readiness,
requires exactly one regex match, and uses that captured value in the marker.
It does not label the OpenAI endpoint as an attention backend. At least one
`--vllm-startup-required-fragment` is also mandatory; supply the literal
startup-log fragments that attest the exact graph/compile/KV configuration you
intend to compare, after inspecting the image's real logs. The command and
snapshot are preserved even when an attestation fails, but that attempt cannot
emit a marker. Pass an optional version/image-inspection argv JSON through
`--vllm-version-command-json`; its stdout/stderr hashes are retained with the
launch-template and host-launcher hashes. For Docker, an inspect command such as
`["/usr/bin/docker", "image", "inspect", "vllm/vllm-openai@sha256:<digest>"]`
is appropriate.

The template deliberately enables chunked prefill for P2048/M32, disables
prefix caching because every request has the same prompt, uses BF16 weights and
KV cache, keeps the attention backend auto-selected, uses Docker host IPC, and
sets GPU memory utilization to `0.65`. It also requires Docker `--rm`,
`--network host`, `--gpus device=0`, the read-only `/model` mount, the exact
image/`--model`/`--served-model-name` contract, `--max-model-len {max_model_len}`,
`--max-num-seqs {concurrency}`, and
`--max-num-batched-tokens {batch_token_budget}` exactly once. The driver
rejects a template missing any of these fixed-comparison controls.

## C=32, M=32 invocation

This example writes new artifacts outside the checkout. It assumes an already built CUDA
Riley executable, a pinned local Qwen2.5-3B directory, an exact workload
artifact, and a vLLM startup regex verified for the actual image.

```bash
OUT=/var/tmp/riley-n03b-n06a/$(date -u +%Y%m%dT%H%M%SZ)-qwen3b-c32
mkdir -p "$OUT"

python3 benchmarks/scripts/n01_repeat_control.py \
  --output-dir "$OUT" \
  --working-directory "$(pwd)" \
  --warmups 1 --repeats 6 --timeout-seconds 1800 \
  --n06a-timeout-cleanup-artifact-root "$OUT/paired-driver" \
  --n06a-timeout-cleanup-docker-launcher /usr/bin/docker \
  -- python3 benchmarks/scripts/n06a_paired_serving_driver.py \
       --artifact-root "$OUT/paired-driver" \
       --workload /var/tmp/riley-n06a/qwen3b-c32-p2048-o128.json \
       --model-path /data/psyche/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1 \
       --model-identity-manifest /var/tmp/riley-n06a/qwen3b-model-identity.json \
       --model-git /usr/bin/git \
       --riley-binary /data/riley-serving-260915-n01/target/release/riley \
       --vllm-command-json /var/tmp/riley-n06a/vllm-v0.29.0.json \
       --vllm-image-digest sha256:<the-inspected-64-hex-digest> \
       --vllm-version-command-json /var/tmp/riley-n06a/vllm-v0.29.0-inspect.json \
       --vllm-backend-requested vllm-auto \
       --vllm-backend-receipt-regex 'Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:' \
       --vllm-startup-required-fragment ' attention backend out of potential backends:' \
       --concurrency 32 --batch-token-budget 32 \
       --retained-requests 1000 --warmup-requests 32 \
       --max-model-len 32768 --kv-blocks 4352 \
       --riley-port 18080 --vllm-port 18081 \
       --nvidia-smi /usr/bin/nvidia-smi --gpu-memory-sampling-interval-seconds 0.25
```

Do not wrap the N01 timed command in `ionice` or `nice`: a daemon-owned Docker
vLLM process does not inherit that wrapper, so it would make the AB/BA lanes
asymmetric. Host preparation may use a low-priority wrapper outside the N01
timed interval, but it is not part of this comparison command. The two
`--n06a-timeout-cleanup-*` options must be supplied together and point to the
same paired-driver artifact root and Docker launcher used by the vLLM template.
`--n06a-timeout-cleanup-command-timeout-seconds` is an optional per-Docker-
command bound; it defaults to 30 seconds and cannot exceed 30 seconds.
On a timeout or controller interruption N01 records its attempted cleanup of
only the derived, owned vLLM container and recorded descendants; its sidecar is
part of the retained receipt even when cleanup cannot be proven. If cleanup is
unproven, N01 does not start another child: every remaining planned index is a
`not-started-after-failed-cleanup` placeholder that binds the blocking
kind/index/reason and deliberately has no log, PSI, or cleanup sidecar.

`--batch-token-budget` is independent of offered concurrency and is bounded to
32 for the native D128 path. It must be at least concurrency. Both marker lines
bind `batch_token_budget` and `max_model_len`, so the summary rejects a lane or
repeat that changes either execution envelope. For B8 experiments, pass
`--concurrency 8 --batch-token-budget 32`; do not silently reduce M to 8.

For each owned lane, the whole-GPU sampler takes an initial sample before
server launch, polls at the configured 0.25 to 0.5-second cadence while the
server is alive, then takes a final sample after owned-process cleanup. Each
raw row contains query start/end timestamps; a query longer than 0.5 seconds,
a start-to-start gap above `configured interval + 0.5 seconds`, a missing
sample overlapping server lifetime, a sample error, or a peak above 19 GB
makes the lane ineligible. The marker binds the raw sample JSONL and peak
receipt hashes, plus a hashed per-lane provenance receipt that records launch
argv, cleanup, lifecycle timestamps, and snapshots. The vLLM marker separately
binds the frozen stdout and stderr startup snapshot paths/hashes; the summary
replays the source-verified auto-selector regex from those snapshots.

The driver also records CPU, I/O, and memory pressure in a dedicated,
create-only `<lane>.lane-psi.json` for each launched lane. Its `pre` snapshot
is captured immediately before sampler start/server launch; its `post` snapshot
is captured after owned process/container cleanup and sampler termination,
before the next lane's GPU idle census. The provenance receipt names that file
and its SHA-256, and the existing marker binds the provenance receipt. The
snapshot source is `/proc/pressure/{cpu,io,memory}`. High pressure is retained;
unavailable and malformed sources are also retained with their observation
status. None of these values is a quietness gate, changes marker eligibility,
filters a pair, reweights/adjusts a performance metric, or changes promotion.
The offline N03b output exposes the validated lane records as
`lane_pressure_covariates` and an `order_by_lane_pressure_sensitivity` view by
AB/BA order and lane position. That view is descriptive only: it has no PSI
bootstrap, pressure-adjusted effect, or causal interpretation.

Before starting the next lane, the driver also records a separate GPU-0 idle
census. It must show no compute-process PID and `memory.used <= 512 MiB` after
the owned lane cleanup. The census is evidence only: it never terminates an
unrelated GPU process. A noncompliant or malformed census retains a failed lane
receipt and prevents the attempt from producing a comparable marker.

After the controller completes, produce a separate offline summary:

```bash
python3 benchmarks/scripts/n03b_n06a_d128_repeat_summary.py \
  --serving-receipt "$OUT/n01-repeat-control-receipt.json" \
  > "$OUT/n06a-summary.json"
```

A summary remains scoped to its pinned Qwen3B workload and recorded vLLM launch
provenance. It is not a general vLLM superiority claim. A failed outer attempt,
including a vLLM startup-attestation failure, remains in the N01 receipt and
its PSI covariates but contributes no numerical serving sample.

This document defines the N06-A launch and evidence contract. It does not
report an actual Riley-versus-vLLM comparison: no serving result is a claim
until an executed AB/BA receipt passes raw-stream replay, lifecycle, cleanup,
model-identity, GPU-peak, and post-lane-idle validation.
