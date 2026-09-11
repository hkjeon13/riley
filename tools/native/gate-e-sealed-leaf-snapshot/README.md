# Gate E sealed leaf snapshot

This Linux C11 library is a **source/audit-only data-snapshot primitive** for
a future RC3 Gate E native guardian. It is an excluded `tools/native` utility,
not a Riley runtime component, installer, service, launcher, release input, or
qualification producer. It has no `main`, CLI, JSON output, path input, or
configuration surface.

Its only public operation is `gate_e_snapshot_held_leaf_v1()`. It copies an
already-held regular source FD into a newly created anonymous data-only memfd.
On success, the caller receives one FD (always at least `3`) together with its
expected byte length and SHA-256. The returned FD is `CLOEXEC`, anonymous
(`st_nlink == 0`), has no executable mode bits, and has this exact seal mask:

```text
F_SEAL_EXEC | F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL
```

The primitive makes no temporary file and has no `MFD_EXEC`,
`F_SEAL_FUTURE_WRITE`, pathname, `open`, `openat`, `exec`, fork, socket, GPU,
Docker, lock, cgroup, signal, evidence, receipt, or release/qualification path.
The only payload write is fixed-offset `pwrite` to the just-created private
memfd before it is sealed.

## Contract and failure boundary

The input must already be a held, read-only, `O_NOATIME`, single-link regular
object of the expected nonzero length (at most 2 MiB). Its pathname,
root/owner/ACL/local-filesystem provenance, and caller-side `CLOEXEC` discipline
are authenticated by the upstream native root-bundle acquisition boundary, not
by this primitive.

The first source operation is `F_DUPFD_CLOEXEC(..., 3)`. After that, the
caller-owned numeric FD is not used again: all source reads are fixed-offset
`pread` calls on the private duplicate, so neither the caller offset nor a
shared open-file-description offset changes. The source clone is closed before
the result is published; the caller-owned FD is never closed. Before and after
the copy, the clone is checked as the same regular object using device, inode,
mode, link count, UID/GID, size, mtime, and ctime (deliberately not atime).
Those checks are conservative race detection, not path or execution authority.

The implementation checks the soft `RLIMIT_FSIZE` before creating/writing the
memfd. A finite limit below the requested length fails closed rather than
risking `SIGXFSZ`. A trusted, single-threaded guardian is still required while
the operation is in progress: a generic in-process adversary can otherwise
alter its FD table or process-wide resource limits.

The output requires Linux `MFD_NOEXEC_SEAL` / `F_SEAL_EXEC` support (Linux 6.3
or newer). The old userspace UAPI-header gap is handled by conditional numeric
compatibility definitions, then checked against the running kernel with exact
`F_GET_SEALS` validation. `ENOSYS`, `EINVAL`, or `EPERM` from this required
no-exec memfd ABI is a hard `GATE_E_SEALED_LEAF_MEMFD_UNAVAILABLE` failure;
there is no executable, weak-seal, temporary-file, or older-kernel fallback.

For a valid initialized empty output object, every failed call leaves it empty
(`descriptor == -1`, zero length, zero digest). A misuse call that presents a
live or uninitialized output object is rejected without touching its caller-
owned descriptor. Internal descriptors are closed once and never retried after
a close error; a close failure discards the result and returns
`GATE_E_SEALED_LEAF_CLOSE_FAILED`. The caller closes a successful result with
`gate_e_sealed_leaf_snapshot_close()` before reinitializing the output object.

## Explicit non-authority

This only establishes an immutable copy of supplied bytes. It does **not**
authenticate a source path, root ownership, ACLs, filesystem, held-object
token, manifest closure, interpreter/dynamic-loader/runtime closure, host
namespace, secure `execveat`, Python invocation, guardian lease, PID1 ledger,
cgroup/pidfd lifecycle, GPU/Docker action, evidence, receipt, freeze, rollback,
or C02 qualification. In particular, Python leaves in a no-exec memfd are data
for a separately authenticated interpreter; this API must never be repurposed
as a future native executable handoff.

## Checks

Run on an executable Linux temporary filesystem with a C compiler and a Linux
6.3+ kernel:

```bash
cd tools/native/gate-e-sealed-leaf-snapshot
make test
make analyze
```

`make test` builds below one fresh `mktemp` directory, exercises only a
current-UID private fixture, checks source-FD reuse resistance, low
`RLIMIT_FSIZE` rejection, exact seals, no executable output, immutable writes/
truncation/shared-mapping rejection, and failure cleanup, then removes its
exact temporary build files. The test source itself creates that fixture; the
production library does not create filesystem objects. `make analyze` performs
a separate compiler static-analysis build. A deliberately `noexec` scratch
mount can run `make analyze`, but needs an explicitly supplied executable
`TMPDIR` for `make test`.
