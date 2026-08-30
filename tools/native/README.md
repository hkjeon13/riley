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

`gate-e-sealed-leaf-snapshot/` is a third standalone C11 **library-only**
precursor. It takes no paths or configuration and has no CLI/output protocol:
given an already-held, upstream-authenticated regular leaf FD, it can create a
bounded anonymous `MFD_NOEXEC_SEAL` data memfd with an exact immutable seal
mask. It does not authenticate the source path or interpreter/runtime closure,
install or execute a bundle, create a lock/cgroup/child, or touch
GPU/Docker/evidence. It is not a guardian, launcher, release, Gate E, freeze,
rollback, or qualification input; its local README defines the strict Linux
kernel requirement and fixture-only checks.

gate-e-execution-closure-manifest-parser/ is a fourth standalone C11
in-memory library-only precursor. It receives raw canonical sidecar bytes only
and returns bounded declared interpreter/loader/runtime labels plus their raw
manifest SHA-256; it does not open or authenticate any declared path. It does
not extend the root-bundle grammar, create an FD handoff, inspect ELF or loader
behavior, execute anything, or touch guardian/lease/GPU/Docker/evidence. It is
not a guardian, launcher, release, Gate E, freeze, rollback, or qualification
input; its local README defines the exact byte grammar and fixture-only checks.

gate-e-guardian-control-packet-validator/ is a fifth standalone C11
in-memory library-only precursor. It compares one bounded canonical READY or
NO_ACTION_COMPLETE packet with caller-owned active-session digest/generation
bindings and returns a reason only. It does not receive a socket message,
authenticate a sender or phase, inspect credentials/cgroups/FDs, retain a
lease/ledger, acquire a path, or execute anything. It is not a guardian,
launcher, release, Gate E, freeze, rollback, or qualification input; its local
README defines the fixed wire grammar and fixture-only checks.
