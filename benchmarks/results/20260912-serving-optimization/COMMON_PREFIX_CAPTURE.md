# Prepared independent common-prefix logits diagnostic

The new `capture_common_prefix_logits.py` is prepared locally and has not run on
the GPU. It uses the unchanged repository `HuggingFaceCalibrationBackend.load`
and `_capture_numeric`: exact input IDs, explicit positions, HF eager B1,
cache-off, no free generation or retokenization. FP32 and BF16 run in separate
owned child processes. This supplies an independent numerical observation; it
does not capture the actual vLLM serving graph, identify a responsible kernel,
set a tolerance, replace a golden reference, or qualify concurrent serving.

The frozen two cases are the identical histories immediately before the first
observed difference at generated index8 or29. The runner verifies206 input/raw
provenance files and confirms every cited response shares the specified prefix.
It stores all49,152 logits and log probabilities plus the full first-layer
hidden tensor, retaining original dtype bytes. Ranking reports include tied
values, reference/alternative choices and gaps from the maximum without an
accuracy pass/fail rule. See `NUMERICAL_DIVERGENCE_NEXT.md` for evidence and limits.

Every run requires the pinned Round14 plan, complete serving-process cleanup
and verified Blender restoration. The parent sets the already qualified
private driver environment before importing CUDA in its child. The unchanged
HF backend enforces checkpoint/config/tokenizer and interpreter/dependency
versions. The new runner additionally records installed implementation files,
binds actually imported modules and the concrete model class to those files,
checks actual private NVIDIA mappings and CUDA/BLAS library bytes, and verifies
inputs/source files again after capture. These installed-library hashes are
new provenance, not a retroactive claim about an old canonical oracle.

Nine CPU tests passed, including real raw/SSE provenance, changed common
prefixes, corrupted hashes, full finite vocabulary, tie-aware ranks, and child
or cleanup failures. Independent review found no remaining concrete blocker.
Cleanup failures retain both the original error and cleanup error and prevent
a completion receipt. No source in the reference package was edited.

| Artifact | SHA256 |
| --- | --- |
| Capture helper | `413fa06c6e86bee1b2b525854b837c25e19f650ffc5dd38f74040d56a95b129a` |
| CPU tests | `5e82a4a1166d993fa1f05f88c8b59b7480bed4728d3932d993a7106c610768cb` |
| Exact reference source inventory | `8a8b46e0eb6071356a2452c3e029e8276fbc0b2409357ec248645324374dfa03` |
| Common-prefix cases | `e8acc4dac75b97a3a3820cbf8dbddb3903c4e6af440ea3ed08b9c8052685e9f6` |

After the serving campaign ends, root may copy the helper, source inventory,
cases, prior analysis and the inventoried `riley_reference` package to the
remote campaign directory. The actual GPU commands are separate from this
local preparation, for example:

```sh
/data/riley-vllm-interim.CfrT9T/venv/bin/python \
  /tmp/riley-opt-260912/capture_common_prefix_logits.py run \
  --root /tmp/riley-opt-260912 \
  --reference-package /tmp/riley-opt-260912/common-prefix-reference-source/riley_reference \
  --output /tmp/riley-opt-260912/common-prefix-fp32 \
  --dtype fp32 --hf-home /data/riley-vllm-interim.CfrT9T/hf \
  --path-map /Users/psyche/PycharmProjects/riley/benchmarks/results/20260912-serving-optimization/raw=/tmp/riley-opt-260912 \
  --path-map /Users/psyche/PycharmProjects/riley/benchmarks/results/20260912-serving-optimization=/tmp/riley-opt-260912 \
  --path-map /Users/psyche/PycharmProjects/riley/benchmarks/results/20260911-g04-attention-trace=/tmp/riley-g04-attention-trace-260911
```

Then use a fresh `common-prefix-bf16` output and `--dtype bf16`. The original
trace `prefixes.json` must match its pinned hash at the mapped remote path.
Default timeout is600 seconds, with owned-child cleanup on failure. Exclusive
outputs preserve failed attempts. Blender remains running for these numerical
observations; no timing comparison is inferred. Copy these small output
directories explicitly after execution because the general campaign sync
excludes `*.bin`, while these raw tensors are necessary for independent analysis.
