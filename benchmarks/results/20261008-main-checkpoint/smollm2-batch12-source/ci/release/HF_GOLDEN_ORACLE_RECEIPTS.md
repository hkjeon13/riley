# PR16 HF golden-oracle receipts

This is a development-only review workflow. It does not run in, ship in, or
authorize the release image. Its sole output is independent provenance for the
eight-token greedy completion used by the Python-free E2E and soak gates.

The producer accepts only the local immutable
`HuggingFaceTB/SmolLM2-135M` snapshot at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It pins the current reference
dependency lock SHA-256
`101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30`,
Linux x86_64 CPython 3.13.15 executable SHA-256
`ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866`,
PyTorch 2.13.0, Transformers 5.15.1, tokenizers 0.22.2, CUDA runtime 13.0,
driver 580.173.02, and designated RTX 4090 UUID
`GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`. The model is loaded from the
supplied directory with `local_files_only=true`, `trust_remote_code=false`,
BF16, and eager attention. Network access is disabled through the recorded HF
and Transformers offline settings.

Run the producer only during the exclusive remote GPU maintenance window. The
canonical full-model-tree digest is an independently reviewed input, not a
value copied from the producer. Each output path must be absent. Invoke the
command twice; each invocation must be a new pinned Python process:

```sh
export CUDA_VISIBLE_DEVICES=GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0
export PINNED_PYTHON=/external/reference-environment/bin/python3.13
export MODEL_DIR=/models/reviewed-smollm2-135m
export MODEL_TREE_SHA256=<independently-reviewed-canonical-full-tree-sha256>
export ORACLE_ROOT=/append-only-evidence/pr16-hf-golden-oracle

"$PINNED_PYTHON" -I ci/release/write_hf_golden_oracle_receipt.py \
  --output "$ORACLE_ROOT/run-01.json" \
  --model-dir "$MODEL_DIR" \
  --dependency-lock tools/python/reference/uv.lock \
  --expected-model-tree-sha256 "$MODEL_TREE_SHA256" \
  --expected-dependency-lock-sha256 101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30

"$PINNED_PYTHON" -I ci/release/write_hf_golden_oracle_receipt.py \
  --output "$ORACLE_ROOT/run-02.json" \
  --model-dir "$MODEL_DIR" \
  --dependency-lock tools/python/reference/uv.lock \
  --expected-model-tree-sha256 "$MODEL_TREE_SHA256" \
  --expected-dependency-lock-sha256 101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30
```

Within each fresh process, the producer runs the exact prompt
`Explain why deterministic benchmarks need immutable inputs.` first with the
HF cache enabled and then disabled. Both paths must emit exactly eight token
IDs and identical decoded UTF-8. The receipt closes over the process/run
identity, timestamps, normalized invocation, dependency and Python hashes,
CUDA/driver/GPU identity, canonical full model tree and individual immutable
model/tokenizer hashes, prompt bytes, all generation settings, ordered token
IDs and their u32le digest, and exact completion text and its UTF-8 digest.

After copying the two immutable receipts to a review host, run the
standard-library-only checker. Its expected values must come from the review
record rather than from either receipt:

```sh
python3 -I ci/release/check_hf_golden_oracle_receipts.py \
  --receipt /review/run-01.json \
  --receipt /review/run-02.json \
  --output /review/hf-golden-oracle-approval.json \
  --expected-model-tree-sha256 "$MODEL_TREE_SHA256" \
  --expected-dependency-lock-sha256 101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30 \
  --expected-python-executable-sha256 ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866 \
  --expected-gpu-uuid GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0
```

The checker requires two different nonces, run IDs, file identities, and
Linux process identities. It independently replays the closed canonical JSON
schema and requires all four cache paths to equal token IDs
`[198,198,504,44771,9577,359,260,9577]`, u32le SHA-256
`d9b9a665ea62ae4e21235b347973ee811267bcf205f090e376c6ed71be2c8ba4`,
and completion-text UTF-8 SHA-256
`e79401a64f79f3a3bf47c04cb0d0d0c0116eb97ee10e7caef4c60dc716831d47`.
The approval is create-only and byte-binds both receipts.

The two receipts and their approval remain outside the closed final release
evidence root as review provenance. They are not final-candidate evidence and
must not be inserted into the E2E or soak archive inventories. Only after the
final native E0 correctness report passes may a reviewer materialize
`correctness-golden.json`. That final golden must bind the clean final `REV`,
the exact final native report SHA-256, the already reviewed model/tokenizer
anchors, prompt and eight-token limit, and the approved completion-text hash.
Changing `REV`, the native report bytes, or any oracle anchor requires a new
golden review; the HF receipts can never self-authorize those candidate-side
bindings.

Local validation is limited to standard-library unit tests and syntax checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  ci/release/test_hf_golden_oracle_receipts.py -v
python3 -m py_compile \
  ci/release/write_hf_golden_oracle_receipt.py \
  ci/release/check_hf_golden_oracle_receipts.py \
  ci/release/test_hf_golden_oracle_receipts.py
```
