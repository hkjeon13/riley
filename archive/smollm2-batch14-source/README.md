# SmolLM2 kernel branch (`codex/serving-kernel-batch14`)

The kernel work that produced the SmolLM2-135M results in
[docs/performance.md](../../docs/performance.md) was done on a branch that
shares no history with `main`. It has not been merged; its complete history is
kept here as a git bundle.

- `codex-serving-kernel-batch14.bundle`: head
  `647496ca0a329cbcb5d12d29e9b6db561a33e214`, sha256
  `32a021575fec9f7500710184edf4288dcbb7a130125f4a77cd8755dbaead7e55`.
- Restore it with:

  ```bash
  git fetch archive/smollm2-batch14-source/codex-serving-kernel-batch14.bundle \
    codex/serving-kernel-batch14:codex/serving-kernel-batch14
  ```
