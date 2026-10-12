## What and why

<!-- The problem, and the one question this change answers. -->

## What it preserves

<!-- One of: bit-identical to the existing path / exact rewrite with different
rounding / approximation (opt-in only) / not applicable. Say how you checked. -->

## Validation

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings`
- [ ] `cargo test --locked --workspace --no-default-features`
- [ ] GPU tests, if `kernels/` or `cuda`-gated code changed
- [ ] Output is still independent of batch composition (test added or extended)

## Results

<!-- For performance changes: before/after from the same run, with the
criterion written before measuring. -->

## Rollback

<!-- The option or commit that restores the previous behaviour. -->
