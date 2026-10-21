# v15B prefill and host prototype

Unapplied patch against the v13B runtime at `301c9bf93`. This is an experimental
checkpoint; v13B remains the default at this commit. Five-repeat throughput
improves, but a C32 tail flag and general BF16-view alignment require follow-up.
See [the report](../../../docs/qwen-prefill-host-v15b-20261021.md).
