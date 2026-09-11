# Gate E root-bundle sealed leaves v1

This Linux C11 library is a **source-only composition primitive** for a future
RC3 Gate E native guardian. It is an excluded `tools/native` utility, not a
Riley runtime component, installer, service, launcher, release input, or
qualification producer. It has no `main`, CLI, path, JSON, configuration, or
root-bundle acquisition surface.

Its only public operation,
`gate_e_snapshot_held_root_bundle_leaves_v1()`, borrows a caller-owned
`gate_e_root_bundle_held_v1` and produces two caller-owned
`gate_e_sealed_leaf_snapshot` outputs: the fixed future guardian bootstrap and
core as anonymous no-exec data memfds. It neither acquires, closes, changes,
nor transfers the held root-bundle handle. The caller remains responsible for
its later recheck and close.

The fixed sequence is deliberately narrow:

```text
held root-bundle recheck
→ sealed bootstrap data snapshot
→ held root-bundle recheck
→ sealed core data snapshot
→ held root-bundle recheck
```

Each snapshot independently enforces the held leaf's exact SHA-256 and byte
length, source identity/race checks, `O_RDONLY|O_NOATIME|CLOEXEC` input policy,
and `MFD_NOEXEC_SEAL` plus the full immutable seal mask. If any step fails, the
pair output is closed and cleared; the borrowed root-bundle handle remains with
the caller. The three rechecks retain the fixed parent/name and held-object
identity binding, but do not make the composition an atomic bundle transaction
or re-authenticate every manifest/ACL/filesystem property.

The public output contains only two data snapshots. This library does **not**
place FDs 31/32, execute a leaf, call `execveat`, authenticate an
interpreter/dynamic loader/runtime closure, open or re-resolve a path, create a
child, socket, lock, ledger, cgroup, pidfd, GPU/Docker action, evidence,
receipt, freeze, Gate E result, or C02 qualification result. In particular, a
no-exec Python leaf remains data for a separately authenticated interpreter;
this library must never be presented as a secure-exec or producer authority.

The checked fixed guardian bundle grammar remains intentionally distinct from
the older v3 Python execution-anchor layout. This bridge does not install,
rename, or make either layout compatible.

## Checks

Run on an executable Linux temporary filesystem with a C compiler and Linux
6.3+ no-exec memfd ABI:

```bash
cd tools/native/gate-e-root-bundle-sealed-leaves
make test
make analyze
```

`make test` compiles the root-bundle and sealed-leaf production libraries as
separate objects, then links them with the bridge. Its private current-UID
`/tmp/.../opt/riley/...` fixture manually constructs a **pre-authenticated**
held handle only to exercise the three public APIs together; it does not claim
to test the production root acquisition path. That path remains covered by the
root-bundle authenticator's own suite. The bridge checks successful sealed
pair bytes/seals/digests, input-FD offset preservation, borrowed-handle
preservation, recheck rejection, core-digest drift cleanup, live/uninitialized
output rejection, idempotent close, header/object linkage, exact root-library
exports, and the bridge's lack of direct acquisition/path/process dependencies.
