# SM80 native Attention v4 (mode9)

Frozen attention-only source/binary is at
`/mnt/efs/nlp/psyche/riley-c16-tail-v4-20261019`; the separately qualified bounded
MLP Graph source/binary is at
`/mnt/efs/nlp/psyche/riley-sm80-mlp-graph-v4-20261019`.

`attention-only.patch` adds the vector GQA gather and BF16-scaled softmax input
load to the frozen mode8 source. `candidate.patch` also contains the bounded
Gate/Up<=16 / Down<=4 MLP Graph and private workspace replay ledger. These are
historical layers; the current checkout additionally contains mode10, whose
[patch](../sm80-native-prefill-v5/candidate.patch) reverse-checks at the top layer.

All numerical candidates retain the HF/native full-vocabulary byte-exact gate.
The MLP Graph combination had small C16 p95 regressions in two repeats and is not
the selected latency recipe. Mode10 additionally batches only independently
qualified prefill row ranges; see the
[final report](../../../docs/qwen-c16-tail-and-attention-20261019.md).
