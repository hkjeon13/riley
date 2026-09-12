# Common-prefix numerical divergence diagnosis

The saved seven-setting matrix contains **two first-divergence sites**, at
zero-based generated indices8 and29. Their exact pre-choice inputs have136 and
157 tokens. Full-vocabulary independent logits at these two inputs are the next
bounded diagnostic. The present evidence cannot establish a small top-two gap,
identify the responsible kernel, or rank continuation quality. No strict
serving gate, frozen campaign, or `VLLM_BATCH_INVARIANT` setting is changed.

## What the saved responses establish

I re-read all123 responses, checked their record and raw-body hashes, reconstructed
IDs from JSON/SSE bodies, and checked the exact128 prompt IDs,32 output IDs,
128/32/160 usage and length finish. Results agree with
[VLLM_OUTPUT_MATRIX.md](VLLM_OUTPUT_MATRIX.md) and its saved analysis. The new
[extraction](numerical-divergence-extraction.json) keeps phase counts and all29
logprob-response references with positions0/8/29 explicitly labelled by whether
their input prefix still matches the reference.

| Capacity / budget | Requests | Exact IDs | First difference at8:1443→2341 (`few`→`lot`) | At8:1443→1838 (`few`→`little`) | At29:253→638 (`a`→`how`) |
|---|---:|---:|---:|---:|---:|
| 1 /128 | 5 | 5 | 0 | 0 | 0 |
| 2 /128 | 9 | 1 | 4 | 0 | 4 |
| 2 /256 | 9 | 9 | 0 | 0 | 0 |
| 4 /128 | 17 | 5 | 8 | 0 | 4 |
| 4 /512 | 17 | 1 | 4 | 12 | 0 |
| 8 /128 | 33 | 1 | 8 | 4 | 20 |
| 8 /1024 | 33 | 0 | 4 | 28 | 1 |

Token labels above omit their leading space for readability; raw token strings
and IDs remain in the extraction. At index8 every response has the same eight
preceding output IDs: `[28,339,5248,253,1838,3241,282,253]`, or
`, I'm a little bit of a`. At index29, only responses whose first difference is
29 share the entire reference prefix ending `...I'm going to show you`.
An index29 value from a response that already diverged at8 is branch-conditioned
and cannot be compared as the same input. Counts72 and29 below include only
first-divergence observations.

Each of the four C-wide waves has the same first-choice count distribution
within its setting, although worker/request indices can exchange roles.
The separately offered-C1 request is exact except on the capacity8/budget1024
server, where it first chooses638 at29. These observations implicate the
configured execution/scheduling shape as a variable; they do not isolate it.
Client overlap does not prove a particular GPU batch or graph bucket.

All928 logprob-token entries contain a **singleton** `top_logprobs` dictionary.
Selected probability equals that sole returned value. The runner-up probability
and top1–top2 margin are **unavailable**, not zero. Selected logprobs at the
common index8 illustrate the available observations:

| Capacity / budget | Selected token: logprob range in its separate logprob wave |
|---|---|
| 1 /128 | 1443: −3.372966 |
| 2 /128 | 1443: −3.405563;2341: −3.286737 |
| 2 /256 | 1443: [−3.429972,−3.359410] |
| 4 /128 | 1443: [−3.348758,−3.338951];2341: [−3.311964,−3.294096] |
| 4 /512 | 1838: −3.313375;2341: −3.316864 |
| 8 /128 | 1443: [−3.417086,−3.329723];1838: −3.328965;2341: [−3.301623,−3.295803] |
| 8 /1024 | 1838: −3.275320;2341: −3.263840 |

These are different distributions' selected values; subtracting them does not
produce a choice margin. Distribution variation also precedes token divergence:
at index0 all inputs are exactly128 copies of19556 and all choose28, but the
capacity8/budget1024 logprob wave reports that token from−2.280190 to−1.223557.
Even capacity2/budget256, whose nine sequences are exact, reports index0 from
−1.223557 to−0.909200. Equal output IDs alone do not establish equal logits.
Logprob collection is a separate diagnostic condition and supplies no timing
control or independent numerical oracle.

## Ready input cases and prior evidence

[numerical-divergence-cases.json](numerical-divergence-cases.json) is ready for a
new capture runner. It contains exact integer inputs and every first-divergence
source response's original path/hash, setting, phase and request index.

| Case ID | Input length / last position | Generated index | Reference ID | Observed alternative IDs | Source responses |
|---|---|---:|---:|---|---:|
| `common-prefix-136-output-08` | 136 /135 | 8 | 1443 | 1838,2341 | 72 |
| `common-prefix-157-output-29` | 157 /156 | 29 | 253 | 638 | 29 |

Each input is `original_prompt_ids + reference_output_ids[:generated_index]`;
the predicted token itself is excluded. Do not decode and retokenize these
arrays or let an oracle generate its own prefix. The cases' input IDs exactly
match every listed source up to the first difference. Provenance paths remain
original evidence identities; a copied bundle needs an explicit path mapping.

The historical
[attention trace](../20260911-g04-attention-trace/README.md) already has exactly
the same `common136` IDs in its `prefixes.json`, which I checked directly. Its
saved outputs and worker are layer0 attention/RoPE/residual-normalization
replays on fixed tensors, with historical before/after generated-token checks.
There are **no full-model logits in that directory**. Reuse its input and
operation-level context; it does not replace the requested full-logit capture
or establish the current serving kernel's behavior.

## First diagnostic: existing independent HF full logits

Use the unchanged
[`HuggingFaceCalibrationBackend.load`](../../../tools/python/reference/riley_reference/hf_calibration.py)
and `_capture_numeric`. The loader pins Python3.13.15 and its executable hash,
torch2.13.0, transformers5.15.1, safetensors0.8.0, RTX4090/SM89, checkpoint and
tokenizer hashes. It sets deterministic algorithms, disables TF32, loads eager
attention with `trust_remote_code=False`, and uses local artifacts. The model
is SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, weights
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`, config
`1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`.
The reference environment is separate from the vLLM environment; fail on pin
differences rather than updating either environment during the campaign.

Run FP32 and BF16 in separate fresh processes. The existing source seam is:

```python
from riley_reference.calibration import FP32_ORACLE_KIND, BF16_ORACLE_KIND
from riley_reference.hf_calibration import HuggingFaceCalibrationBackend

backend = HuggingFaceCalibrationBackend.load(
    artifact_kind=FP32_ORACLE_KIND,  # BF16_ORACLE_KIND in a separate process
    device="cuda:0", local_files_only=True,
)
torch = backend._torch
ids = torch.tensor([case["input_token_ids"]], device=backend._device,
                   dtype=torch.long)
first_layer, logits, log_probs = backend._capture_numeric(ids, torch.ones_like(ids))
assert tuple(logits.shape) == (49152,)
```

This private helper already supplies explicit positions0..L−1, `use_cache=False`,
`logits_to_keep=1`, a full49152-element raw last-position logit tensor, and its
FP32 log-softmax. A new thin runner should save tensors with the existing
safetensors mechanism, retaining actual dtype/raw bytes, shape and hashes. Save
the first-layer tensor if useful; do not replace full logits with top-k summaries.
FP32 here uses the same checkpoint values widened to FP32, not separate
unrounded training weights or exact real-number arithmetic.

Start with the two cases in each dtype: four full-logit captures. Report all
nonfinite counts, deterministic top1/top2 (lower-ID tie order), top1–top2 gap,
and each of `{1443,1838,2341}` or `{253,638}`'s logit, rank, logprob and gap to
top1. For same-input FP32/BF16 tensors report element counts, max/mean absolute
error, RMS error and raw hashes, with no new tolerance/pass threshold. Retain
both outcomes if they disagree. A fresh repeat can establish whether a result
is stable before attributing a small difference to one implementation.

`constants.py` explicitly records HF cache-on/cache-off greedy divergence at
output17 and a predeclared exact golden window16. Therefore the index29
teacher-forced numerical observation is valid as a fixed-input comparison, but
is **not** an extension of that exact16-token golden-generation qualification.
Neither HF BF16 nor FP32 is assumed to share the frozen g04 continuation.

The runner's future CLI should accept `--cases`, a role `fp32|bf16`, and a fresh
`--output`; invoke it with the validated reference Python and
`PYTHONPATH=<frozen-source>/tools/python/reference`. Retain the established
child-only private driver environment, source/runner/dependency/model/case pins,
device UUID, actual loaded-library maps and complete process cleanup. This
document creates inputs/design only; it does not execute that future command.

## Then isolate serving-shape differences

Independent HF logits can show which observed choices are close or far under
another implementation. They alone cannot explain which vLLM kernel caused
the difference. If that question remains, prepare an **isolated diagnostic
vLLM process** from the pinned installed source, keeping the matrix's default
BF16 numerical policy, prefix caching off, model length160, seed0, capacities
1/2/4/8 and budgets128 or128×C. Do not enable batch-invariant/eager serving as a
replacement control. The archived launch/startup receipts are the exact argv
and configuration source; do not infer GPU shape from client C.

Use two explicitly different experiments:

1. **Actual-path observation:** replay the original128-token input and record
   raw49152-element logits immediately before sampling, actual request-to-row
   IDs, full consumed prefix, absolute position, prefill/decode, scheduled token
   count, graph bucket, and selected attention/backend. Capture the decisive
   steps and index0; export raw logits before penalties, temperature, top-k,
   masking or any teacher-forcing. Since every original first-divergence prefix
   is already common, these rows need no forced choice to observe the first
   difference. Require exact prefix hashes before comparing them to HF.
2. **Controlled history:** for subsequent cache-dependent tests, replay P128
   then feed the same manifest continuation one token at a time to every row.
   Export the unmodified logits first and force only the next committed input
   token afterwards. Reset each request's KV/sequence state between trials.
   No independently generated alternative may enter the common-prefix cache.
   A one-shot136/157-token prompt plus one output is a useful separate prefill
   probe, but is not a substitute for the original P128+decode execution.

Existing vLLM adapter seams are `TokensPrompt(prompt_token_ids=...)`,
`LLMEngine.add_request/step`, and pinned `SamplingParams`; the old attention
trace demonstrates `worker_extension_cls` plus `collective_rpc` for exporting
worker-owned data. **Those repository helpers do not expose raw serving logits
or teacher-forcing.** Before implementation, inspect/hash the installed
v0.27.1 worker/model-runner and sampler call site; choose a strict source anchor
outside the model CUDA graph, with correct request/row mapping and raw-logit
lifetime. Do not invent a public full-logit API from `logprobs=1`, or treat a
standalone call to the loaded model as an actual paged-serving tap. Compare
uninstrumented versus diagnostic token outputs/scheduling and record any
perturbation. Export/allocation/synchronization costs are diagnostic, never
performance samples. Missing or mismatched row/prefix records invalidate a
comparison rather than being replaced by another request's data.

With matched full tensors, distinguish equal-argmax/different-logits,
rounding-level rank changes, and large distribution changes. Compare the
reference token and every observed alternative under each independent tensor;
only then consider first differing layer probes. This bounded repeated-Hello
case can diagnose a numerical sensitivity; it cannot establish general model
quality or qualify C4/C8 serving. The current strict campaign keeps its original
exact IDs/text/usage gates regardless of this diagnostic's outcome.

## Artifact identities

* Cases SHA256: `e8acc4dac75b97a3a3820cbf8dbddb3903c4e6af440ea3ed08b9c8052685e9f6`.
* Extraction SHA256: `573089760f40a9b2c0879c189a9a23f7c8c7ceddad7bd6291fb3d12e2eb1375e`.
* Matrix completion SHA256: `f75a4608c7958f201d2cd15352926d83e4c41dfac56c3d02fd93cf75b7307848`.
* Matrix helper SHA256: `4d21bf1d35aeaf9637e5ff6a188a50ed5aaed8415cf4633ad39daad8b64f9566`.

No GPU capture, new numerical gate, performance result, or production change
was made in this extraction/design task.
