# SM80 native Down and paged QK/PV v6

Layer on the qualified `sm80-native-prefill-v5` / mode10 snapshot, not on clean HEAD. The original mode10 binary is SHA256 `b704f82694a32edc791cdf035c6bb203c7cf3b79932c51ea713d32545821bab5`.

`candidate.patch` changes six source files. `source-layer-manifest.json` binds their before/after hashes. The patch reverse-checks against the current workspace. Existing unrelated work remains intact.

- Opt-in CLI remains `--owned-decode-batch native-prefill-sm80-v5` with the owned HF BF16 profile and iteration-batch completion. Startup explicitly reports the v6 paths. Defaults are unchanged.
- Native SM80/CUDA12.4/cuBLAS120405/108SM image qualification is required.
- Down: R12..32, K32-interleaved sliced FP32 accumulation, BF16 split partials, asynchronous K64 tiles and16-row weight reuse. Uses the already-owned Lt workspace and its existing use/replay ledger.
- Attention: independent decode chunks R2..32 and T24..216, pinned per-request QK/PV arithmetic, original BF16 score/probability boundaries. Direct paged reads and at most six launches per chunk. Cold chunk scratch is bounded by3MiB. Query/KV spans require16-byte alignment.
- Singletons, prefill and unsupported T retain the existing attention path. Rows1..11 retain the existing Down path. SM89 eligibility is unchanged.

The broader T24..304 Attention experiment produced8 changed BF16 outputs at T240/304. Its logs and deterministic reproducer remain under the research EFS root. It is excluded; no tolerance was introduced.

See [the Korean report](../../../docs/qwen-down-qkpv-and-vllm-20261020.md) for full-model exactness, fresh C16/C32 stock vLLM comparisons, tail-latency limits and evidence. Final HTTP/profile audits are recorded there after completion.

Research: `/mnt/efs/nlp/psyche/riley-down-qkpv-v6-20261020`.
Final frozen source/build: `/mnt/efs/nlp/psyche/riley-down-paged-v6-20261020`.
