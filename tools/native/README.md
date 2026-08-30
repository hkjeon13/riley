# Native development tools

This directory is reserved for standalone native inspection and conversion tools.
It remains outside the Cargo workspace.

Native tools may consume stable artifacts or inspect production binaries. They
must not reverse the production dependency graph, become an implicit build input,
or hide runtime behavior behind an external process. Any tool that is needed to
build a production crate belongs in a reviewed Rust crate or the CUDA CMake build.

The runtime-integrated native calibration evidence contract is owned by the
non-default development workspace member `crates/riley-native`, not this
directory. That member shares the root `Cargo.lock` but is not a production
crate or a root default build target.

`gate-e-platform-preflight/` is a standalone C11 inspection tool for the
future RC3 Gate E root-boundary review. It is deliberately no-action and has
no release, runtime, privilege, or qualification authority; its local
README defines its one allowed invocation and fixture-only test command.

`gate-e-root-bundle-authenticator/` is a separate standalone C11,
fixed-path, held-`openat2` object-observation precursor for the future guardian
audit leaves. It is deliberately non-authoritative: it does not install or
execute a bundle, establish a pre-loader trust root, create a lock/cgroup,
touch GPU/Docker/evidence, or become a release, Gate E, freeze, rollback, or
qualification input. Its local README defines its one allowed invocation,
fixed output boundary, and fixture-only C11 checks.
