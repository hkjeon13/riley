# Common-prefix logits analysis

`analyze_common_prefix_logits.py` analyzes the two fresh executions produced by
the frozen `capture_common_prefix_logits.py` (SHA-256
`413fa06c6e86bee1b2b525854b837c25e19f650ffc5dd38f74040d56a95b129a`).
It imports only Python's standard library and does not execute torch or CUDA.

The input is **HF eager, cache-off, teacher-forced B1**. These are not the logits
of an actual vLLM graph. The analyzer assigns no numerical tolerance, accuracy
acceptance, candidate selection, kernel cause or performance decision.

## Required files and explicit path mapping

For each dtype, retain `completion.json`, `worker-receipt.json`,
`finalization.json`, `launch.json`, `process.json`, the two case JSON sidecars,
and all six raw tensor files. Keep each capture in its own directory. A worker
log is useful for manual diagnosis but is not a required mathematical input.
FP32 and BF16 captures must have distinct directory and PID receipts and match
the exact two frozen teacher-forced prefixes: lengths 136 and 157, generated
indices 8 and 29.

The local capture helper, frozen cases file (`e8acc4da…`) and source inventory
(`8a8b46e0…`) must match their complete hard-coded hashes. CLI overrides change
their locations only; they cannot change the expected hashes.

Every file reference inside a capture receipt requires an explicit absolute
source-to-local mapping. There is no fallback to another directory or to an
existing unmapped path. The longest source prefix wins; duplicate source rules,
relative paths, `..`, and local symlink escapes are rejected. For captures copied
under the campaign's local `raw` directory, an example is:

```sh
python3 benchmarks/results/20260912-serving-optimization/analyze_common_prefix_logits.py \
  --fp32-completion /absolute/local/raw/hf-prefix-fp32/completion.json \
  --bf16-completion /absolute/local/raw/hf-prefix-bf16/completion.json \
  --path-map /tmp/riley-opt-260912=/absolute/local/raw \
  --output /absolute/local/common-prefix-analysis
```

Use the actual preserved remote capture directory names. The output directory
must not already exist. To analyze files still at their recorded paths, supply
an explicit identity mapping such as `/tmp/riley-opt-260912=/tmp/riley-opt-260912`.
Path maps recorded in the capture's launch arguments are used only to reconstruct
its historical case-input dictionary; they never authorize analyzer file reads.

## What is checked

- Completion references must hash-match worker and finalization files in the
  same run. Worker success, backend close and owned-session cleanup must all be
  complete; a failed or ambiguous cleanup is rejected.
- Launch dtype, tool, output directory, offline environment, campaign gate and
  owned-session receipt must agree with the worker. Both runs bind the same
  model/runtime metadata, installed implementation inventory and frozen source
  inventory. Receipt freshness is based on launch/PID/cleanup artifacts; this
  capture format does not independently record historical process start times.
- Captured inputs must equal the two frozen teacher-forced prefixes. Case JSON
  sidecars must equal the records inside the hash-bound worker receipt.
- All 49,152 logits and log-probabilities, and every first-layer hidden element,
  are read again from raw bytes. Exact shapes, byte sizes, little-endian format,
  hashes and finiteness are mandatory. Logits/hidden use their run's native
  FP32/BF16 dtype; log-probabilities use FP32 in both runs.
- BF16 decoding shifts each raw 16-bit word into an FP32 word without arithmetic
  conversion, preserving signed zero and subnormal bits. Stored top choices,
  ranks/counts and margins are recomputed from all raw logits and must agree.
- Inputs and receipts are rehashed again before writing analysis output, so
  changed files cannot silently become the final analyzed evidence.

The worker's recorded dependency/native-map inventories are checked for internal
hash consistency, concrete implementation membership, private CUDA presence,
mapped inode relationships and GPU identity. Dtype-specific lazy loading may
produce additional library paths; these are reported, while shared paths must
retain the same identity. No unavailable remote installation is recursively
rehashed by this analyzer. Historical source responses, serving finalization and
restoration files, reference package files and installed/native dependencies
remain **receipt provenance**, distinct from locally rehashed inputs. These
newly recorded implementations are not presented as an old canonical oracle.

## Output interpretation

`analysis.json` contains full-array descriptive statistics for logits,
log-probabilities and hidden states: exact represented-value and expanded-FP32
bit differences, signed-zero differences, maximum/mean absolute difference,
RMS difference, signed difference extrema and the largest differences. It also
records top-10 choices, reference/observed serving choices, margins, equal-value
counts and rank changes. Hidden indices are flattened row-major indices with
shape `[input_length, 576]`.

Each case also gets a 49,152-row `*.vocabulary.tsv`. Its columns include both
logits, both log-probabilities, their signed differences and both ranks.
Difference signs are **BF16 minus FP32**. Ranks are one-based, descending by
logit, with the lower token ID first on ties; a positive rank difference means
that token is ranked lower in BF16. `strictly_greater_count + 1` identifies the
first tied rank, while `equal_count` reports the whole tie group.

The sum of exponentiated log-probabilities is a descriptive check only. No
threshold is attached to it or to any difference metric. A complete analysis
means artifact validation and calculation completed; it does not mean either
dtype's selected token is accepted as more accurate.

## Local tests

```sh
python3 -m unittest benchmarks/scripts/tests/test_analyze_common_prefix_logits.py -v
```

The tests use full-size synthetic FP32/BF16 files plus the actual frozen prefix
manifest. They exercise successful TSV reconstruction and rejection of damaged
hash chains, truncation, wrong shapes/dtypes, altered stored scores/prefixes,
cross-run references, incomplete cleanup, launch mismatches, changed provenance,
ambiguous path maps and nonfinite values. Signed-zero/subnormal decoding and tie
ranking are checked directly. Synthetic fixtures establish analyzer behavior,
not model or GPU correctness; the real captures have not been run by this task.
