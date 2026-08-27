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

PR 02 adds no executable here.
