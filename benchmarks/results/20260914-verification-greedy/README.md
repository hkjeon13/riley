# GPU greedy verification output

Experimental target verification now optionally returns 32 fixed 16-byte GPU
argmax records, bound by the retained iteration/replay/owner generation/catalog.
The diagnostic full-logit mode remains available with a distinct catalog digest.
There is no Python in the runtime execution path.

| Check | Result |
|---|---|
| Serial target vs multi-position head BF16 logits | 72 rows, 0 differences |
| GPU argmax vs CPU argmax of serial target logits | 72/72 exact |
| Auxiliary verification D2H | 3,145,728 → 512 bytes |
| Native graph replay probe | 15 cases × 2 replays |
| Native memcheck / racecheck | 0 errors / 0 hazards |
| Full model memcheck | 0 errors |
| Rust speculative policy and record tests | 6 passed |
| SM89 / SM90a / SM100a native compilation | Passed; execution only SM89 |
| Blender restoration | All 3 RPC endpoints restored |

The reduction applies only to the auxiliary verification transfer. Ordinary
result transfer remains full sized. This is correctness evidence, not a serving
benchmark, throughput improvement, or scheduler integration claim.

The native probe covers lowest-ID ties, negative logits, NaN, positive/negative
infinity, upstream status, inactive padding, and repeated CUDA graph execution.
The Rust parser validates every record's slot, validity, status and vocabulary
range, including inactive records. Model tests reject reads before completion,
after settlement and with the wrong result mode. Tokens do not authorize KV
commit; the caller still needs exact execution ownership and quiescence.

The current head supports 1–4 pure-prefill owners with 1–8 inputs per owner.
Thus a future `[pending, draft...]` round on this head must cap drafts at **7**,
even though the standalone policy supports 8. Mixed decode owners, shared-tail
COW, private append execution and multi-token scheduler accounting remain work.
No stochastic policy is supported by this argmax path.

Recheck the compressed raw pair, token output, logs and source hashes with:

```sh
python3 benchmarks/analysis/verify_verification_greedy.py
```

Viewer loopback HTTP checks are recorded separately from Blender RPC restoration;
these checks are not a new public browser/visual acceptance test.
