# Gate E native root-bundle authenticator

This Linux C11 program is a **source/audit object-observation precursor** for a
future RC3 Gate E native guardian. It is an excluded `tools/native` utility,
not a Riley runtime component, root installer, service, launcher, release
input, or qualification producer.

Its only public form is:

```text
gate_e_root_bundle_authenticator --authenticate-root-bundle-v1
```

For a future separately reviewed native guardian, the source directory also
exports the C11 `gate_e_root_bundle_held_v1.h` library ABI. Its only acquire
entrypoint is `gate_e_root_bundle_acquire_fixed_v1()`: it accepts no path,
owner, leaf, manifest, lock, GPU/Docker, or configuration input and returns
caller-owned read-only `CLOEXEC` descriptors for the fixed root, descendants,
manifest, bootstrap, and core only after the same policy checks succeed.
`gate_e_root_bundle_held_v1_recheck()` retains the fixed parent/name and held
object identity checks, while `gate_e_root_bundle_held_v1_close()` closes and
clears every handle. The diagnostic CLI immediately closes its handle before
printing the unchanged non-authoritative JSON report. This retained-object API
does not execute any leaf or turn the checkout-built dynamic binary into a
static guardian, secure-exec launcher, installation path, or producer.

It takes no paths, owner IDs, manifest names, bootstrap names, lock paths,
GPU/Docker/evidence arguments, or configuration. For a root caller it can only
read the future audit tree rooted at `/opt/riley/rc3-gate-e-v1`. It opens `/`
and every descendant through held `openat2` descriptors, with no `open` or
`openat` fallback. `O_NOATIME` is required so the leaf hash reads do not turn
into atime writes. The tool then requires root ownership, exact modes, one
regular-file link, no POSIX ACL, an approved local filesystem (`ext4`, `xfs`,
or `btrfs`), and no bootstrap file capability before comparing the two leaf
hashes and byte lengths against this exact newline-terminated manifest form:

```json
{"bootstrap":{"byte_length":N,"filename":"rc3_gate_e_guardian_bootstrap_v1.py","sha256":"<64 lowercase hex>"},"core":{"byte_length":N,"filename":"rc3_gate_e_guardian_no_action_core_v1.py","sha256":"<64 lowercase hex>"},"schema_version":"riley.rc3-gate-e-root-bundle.v1"}
```

The future manifest and leaves are intentionally distinct from the old v3
`execution-anchor.json` / FD 7–10 Python template. This tool does not install
either bundle and must not be used to bridge or revive that template.

Its JSON always has `status: "not-established"`. At most,
`object_observation_status: "checked"` says that the fixed objects were
observed consistently during this one read-only invocation. It does **not**
establish host-initial namespace membership, the dynamic-loader/pre-Python
trust boundary, same-object execution, interpreter/runtime closure, a guardian
lease, execution authority, an actual Gate E producer, GPU/Docker action,
evidence, receipt, freeze, rollback, or qualification. A normal non-root
server invocation therefore fails closed with exit status `2` and
`effective-uid-gid-not-root`; this repository does not ask for `sudo` or create
`/opt` paths.

The qualification-input policy accepts only canonical JSON bytes. This tool's
terminal-newline diagnostic line is rejected there as noncanonical before
schema dispatch; a canonical reserialization of either this diagnostic schema
or the root-bundle manifest schema receives an explicit denial reason. Neither
form is a qualification input.

Run the source-only checks with a host C compiler:

```bash
cd tools/native/gate-e-root-bundle-authenticator
make test
make analyze
```

`make test` builds its binaries below one fresh `mktemp` directory, tests only
a current-UID private fixture plus the fixed public CLI rejection, and removes
those exact temporary files afterwards. It never invokes the valid root-path
form. `make analyze` performs a separate C compiler static analysis build.
The compiled authenticator binaries and held-object library do not open a GPU,
create a lock/cgroup/socket/child, install a bundle, write evidence, or start a
Gate E workload. The Make recipes themselves naturally invoke the compiler and
small test utilities only. `make test` also links the public header against a
library-mode object, checks the four exported ABI symbols, and confirms that
the library object does not export `main`. `make test` needs an executable
temporary filesystem; a deliberately `noexec` scratch mount can still run
`make analyze`, but must use an explicitly supplied executable `TMPDIR` for the
fixture binary.

A checkout-built dynamic executable necessarily starts after its own loader.
It is therefore not the pre-loader trust root and cannot perform the future
guardian's required `execveat`/sealed-memfd handoff, durable PID1 ledger,
cgroup/pidfd lifecycle enforcement, or producer authorization. Those remain a
separately reviewed and root-installed native boundary.
