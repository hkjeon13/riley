# Native development tools

This directory is reserved for standalone native inspection and conversion tools.
It is outside the production Cargo workspace unless a future PR explicitly adds a
tool as a separately reviewed workspace member.

Native tools may consume stable artifacts or inspect production binaries. They
must not reverse the production dependency graph, become an implicit build input,
or hide runtime behavior behind an external process. Any tool that is needed to
build a production crate belongs in a reviewed Rust crate or the CUDA CMake build.

PR 02 adds no executable here.
