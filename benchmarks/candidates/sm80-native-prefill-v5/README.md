# SM80 native-equivalent prefill v5 (mode10)

Qwen3-4B / SM80 108SM / CUDA12.4 / cuBLAS and Lt120405 only. Explicit
`--owned-decode-batch native-prefill-sm80-v5`; defaults and modes0..9 retain their
arithmetic. This candidate includes the qualified mode9 vector GQA staging and
BF16-scaled softmax load.

Separate Gate and Up keep N9728/K2560 and their original weights. Only adjacent
original request groups M64..130 can be combined into M161..576. Native M131..160
has a different result and is excluded. Request boundaries stay intact for
Attention/KV/output metadata. Decode Gate/Up<=16 and Down<=4 remain unchanged;
Down prefill is not merged. There are no new row copies or scratch buffers for
this row batching.

`candidate.patch` is the seven-file layer on the frozen mode9 source at
`/mnt/efs/nlp/psyche/riley-sm80-mlp-graph-v4-20261019/source`.
`combined-mode8-to-mode10.patch` includes the nine-file mode9+mode10 layer on
`/mnt/efs/nlp/psyche/riley-native-kv-unroll-20261019/source`.
Both reverse-check against the current qualified tree. Older mode7/8/9 layer
patches are historical prerequisites and need their corresponding frozen source.

Final source/binary/raw evidence:
`/mnt/efs/nlp/psyche/riley-sm80-prefill-v5-20261019`.
The source manifest covers935 files, matching the local checkout. The report
contains the measured scope, exactness gates, rejected numerical/latency
candidates, remaining Down/QK-PV work, and GPU6/7 persistent holders:
[qualification report](../../../docs/qwen-c16-tail-and-attention-20261019.md).

Selected C16/C32 recipe uses mode10, GPU greedy, token budget576 and the existing
separate SiLU, with Graph off. The bounded MLP Graph implementation is separately
qualified but is not selected for the C16 latency recipe. This is an opt-in
candidate; C1 small per-repeat tail regressions preclude a universal performance
non-regression claim.
