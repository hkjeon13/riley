# HF-R1 Rust checkpoint preparation implementation — 2026-09-18

Scope: the checkpoint-preparation batch in
[32-huggingface-rust-adoption.md](32-huggingface-rust-adoption.md), not the parallel
CUDA numerical qualification work. Implementation: `tools/riley-checkpoint`, an
independent Rust 1.85 workspace. Production Cargo members, lockfile, serving
API, tokenizer and numerical defaults are unchanged.

The batch includes a strict immutable manifest/allowlist, offline preparation and
verification, optional explicit `hf-hub` transport, streaming size/SHA-256 checks,
real production-loader validation, exclusive destination ownership and a final
atomic create-only manifest publication. See the tool README for its trusted
filesystem assumptions, pinned private-cache pointer handling and interruption
recovery contract. It must not serve from a Hub snapshot or call Python.

CPU regression tests and a separate CI lane cover success, corruption, invalid
model semantics, transport interruption, symlinks and concurrent publication.
Qualification status must be read from the actual PR checks, not inferred from
this implementation note. This work does not claim real Hub/gated-model download
qualification, GPU parity, Gate E/F completion or any serving performance gain.

Remaining independent work: HF-R2 tokenizer parity/CPU cost evaluation, explicit
Hub transport qualification on the target environment, and the already tracked
CUDA cache-on/full-forward numerical and serving gates. Do not remove Python
reference/oracle tools merely because the preparation CLI is Rust.
