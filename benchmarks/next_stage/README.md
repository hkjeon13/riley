# N01 stage contracts

These artifacts pin the N01 planning inputs without changing the frozen
135M benchmark contract or adding a Python dependency to the serving path.
They are a preflight gate, not a measurement receipt or a performance claim.

Run the validator from the repository root:

```sh
python3 benchmarks/next_stage/validate_stage_contract.py
python3 -m unittest discover -s benchmarks/next_stage/tests -v
```

`model-descriptors-v1.json` fixes three roles: the locked 135M strict
regression, a SmolLM2-1.7B connection target whose local artifacts still need
materialization, and Qwen2.5-3B whose local checkpoint and tokenizer receipt
are still required.  The Qwen descriptor deliberately remains unsupported for
serving because the current execution owner supports head dimension 64 only;
the descriptor records its D128 geometry for the later capability gate.

Every scenario accounts for BF16 K and V payloads, a conservative planning
weight ceiling, and at least 1,000,000,000 bytes of uncertainty inside the
20,000,000,000-byte whole-GPU limit.  The remaining allowance must cover
activation/scratch, graph pools, CUDA/library context, packing or load-time
duplicates, and external GPU use.  It is not an observed peak.

`numerical-profiles-v1.json` separates the existing frozen 135M strict gate
from the native BF16 candidate.  Native BF16 has null arithmetic-placement and
quality thresholds until its reference corpus, replay policy, tensor/teacher
forced metrics, perplexity/workload criteria, and free-generation criteria are
frozen.  The validator rejects a qualified status while that profile is
`criteria-pending` and `blocked_on_reference`.

The companion lifecycle controller is offline-only. Before it starts any
phase, it verifies file bytes, validates these two documents with this
validator, and checks the selected model/profile linkage. N01 v1 is limited
to physical GPU index 0. Its `nvidia-smi` values are sampled observations;
warmup and timed-serving require an in-process poll, and even a passing
receipt does not establish a continuous device high-water mark.
