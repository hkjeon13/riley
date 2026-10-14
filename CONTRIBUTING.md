# Contributing to Riley

## Before you open a pull request

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo test --locked --workspace --no-default-features
```

These run without a GPU. Changes to `kernels/` or to code behind the `cuda`
feature also need the GPU tests on a machine with CUDA 12.8:

```bash
cargo test --locked --release --features cuda,server
```

## What we ask of a change

- **One question per pull request.** A change should be reviewable and
  revertible on its own. Separate correctness fixes from optimisations.
- **Say what the change preserves.** An optimisation is one of: bit-identical
  to the existing path; an exact rewrite with different rounding; or an
  approximation. Approximations are never enabled by default.
- **Keep the old path.** New execution paths go behind a server option, with
  the previous behaviour still selectable.
- **Keep output independent of batching.** A request must produce the same
  tokens whatever else shares its step. Add or extend a test that shows it.
- **Measure in the same run.** Performance claims need a before/after
  measurement taken back to back on the same machine, with the criterion
  written down before the run. Do not exclude samples afterwards.

## Unsafe code and FFI

`unsafe` is denied across the workspace and allowed only at the CUDA boundary
in `riley-cuda`. Each `unsafe` block needs a comment stating the invariant it
relies on. Changes to `kernels/include/riley_cuda.h` must be additive: existing
fields and functions keep their meaning.

## Dependencies

The server builds with a small, pinned set of crates. Python and its packages
may be used by tools under `tools/python`, never by the workspace crates.
