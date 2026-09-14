# Short ordinary prefill query reuse: native gate

Candidate for the existing split-FFN serving path. Stage-0 prefill with 2–16 queries uses ordered eight-query attention tiles. The existing wire directory remains valid; unused entries return without duplicate writes. Single-query, decode, verification and longer prefill retain their existing dispatch. This gate does not establish serving improvement.

768 cases: owners 1/8/32 × query counts 1–16 × contexts 32/127/128/129/512/1024/2048/4096 × finite/NaN V. Two CUDA Graph replays per case compare the whole BF16 output buffer, including inactive sentinels, exactly against legacy dispatch. KV pages are shared across owners in this native fixture. A reduced gate (counts 1/8/9/16) passes memcheck and racecheck. SM89/90a/100a compile; only RTX 4090 / SM89 executes. Blender restored after native tests.

The source snapshot binds this native test independently of later integration edits. Full-model logits equality and full-model memcheck pass, including adjacent repeated cached tails of every size 1–16; evidence is in `model-evidence.tar.gz` with `model-manifest.json`. Real serving qualification remains pending. No vLLM performance claim follows from this artifact.
