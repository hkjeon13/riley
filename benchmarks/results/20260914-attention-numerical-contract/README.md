# Attention numerical gate preparation

This is an offline check of historical metric receipts, not a new GPU or serving run. The candidate context-split backend is not implemented and no performance gain is claimed.

The new checker recalculates all eight passage aggregates, validates counts/finite metrics/logit digest syntax and refuses inconsistent pass flags or quality claims. It preserves the existing rule: candidate NLL and FP32-reference KL must both be at most baseline. It does not read raw logits or prove model execution, cleanup or provenance beyond the input metric file digest.

| Historical evidence | Checker exit | Interpretation |
| --- | ---: | --- |
| FlashInfer natural screen | 1 | NLL improves but KL fails; rejected |
| Residual natural screen | 0 | Relative metrics pass only; general quality and serving remain unqualified |

[Machine receipt](verification.json) includes recalculated aggregates and hashes of both input reports. The prior residual experiment's generation and serving failures remain authoritative; this check does not overturn them.

CPU tests cover the real historical failure, an identity control, and ten mutations including forged pass/quality flags, missing targets, duplicate indices, non-finite metrics, invalid argmax counts, altered aggregate and invalid logit digest. Three unittest methods pass, including all ten mutation subtests. `git diff --check` passed. GPU tests are not applicable to this offline checker; no GPU processes were changed.

The [implementation contract](../../../deploy/260913/06-attention-split-numerical-contract.md) fixes the next backend batch and preserves existing model acceptance rules before candidate results. Hardware runtime skips and serving qualification remain separate.
