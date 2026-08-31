# Gate E execution-closure held FDs v1

This Linux C11 library is a **source-only binding precursor** for a future RC3
Gate E static guardian. It parses one canonical execution-closure sidecar and
binds its fixed loader/interpreter/runtime role order to already-borrowed
linked regular file descriptors. It is an excluded tools/native utility, not
a Riley runtime component, installer, service, launcher, release input, or
qualification producer.

Its gate_e_bind_execution_closure_held_fds_v1() call borrows raw sidecar bytes
and a caller-owned descriptor set only for the call. The library first parses
the canonical sidecar with the existing in-memory parser, requires the runtime
descriptor count and order to match it exactly, rejects reused numeric
descriptors and device/inode object aliases, then makes private
F_DUPFD_CLOEXEC duplicates. Each duplicate must be a CLOEXEC, O_RDONLY,
non-append, non-direct, non-O_PATH linked regular file with the declared
size. A generic leaf may have more than one link, but an already-unlinked
object is rejected. It is SHA-256 checked using fixed-offset reads and full
fstat identity before and after hashing. Success returns only caller-owned
output duplicates, their copied declarations and identity, and the raw
sidecar's SHA-256.

The caller's input FDs are never closed, seeked, or changed. The caller must
also serialize FD-table ownership: no concurrent thread may close, reuse, or
change any input or output descriptor slot during bind(), recheck(), or
close(). The output has a separate recheck() operation that privately re-pins
and rehashes every owned FD. It neither stores the raw sidecar bytes nor
authenticates them: the raw SHA-256 is only a value for a later static guardian
to bind under a separate trust boundary. Recheck is not an atomic multi-object
snapshot, and no point-in-time identity/digest check can stop a later writer
without the future root-owned provenance and installation policy.

This library intentionally does **not** require O_NOATIME, root ownership, ACL
absence, a filesystem type, or a single link for generic loader/runtime leaves.
Those are upstream provenance and future administrator/static-guardian policy
decisions, not new policy invented by this binder.

It has no path/openat2 operation, no sidecar acquisition, no root/ACL/filesystem
authentication, no ELF PT_INTERP/DT_NEEDED/RPATH inspection, no dynamic loader
resolution, no complete-runtime proof, no execveat, no FD 31/32 placement, and
no child, socket, ledger, cgroup, PID1, GPU/Docker, evidence, receipt, freeze,
Gate E, or qualification action. A successful bind is not a secure-exec,
installed guardian, complete execution closure, or producer authority.

## Checks

Run on Linux with GCC or a compatible compiler/linker:

    cd tools/native/gate-e-execution-closure-held-fds
    make test
    make analyze

The checked-in target expects the sibling
`../gate-e-execution-closure-manifest-parser/` source directory; callers with
a different checked-out layout can set `PARSER_DIRECTORY` to that directory.

The private current-UID fixture covers canonical binding, copied declaration
order and raw sidecar SHA, multi-leaf runtime ordering, CLOEXEC duplicate
ownership, borrowed-input preservation, output survival after input close,
multi-read hashing, recheck, malformed sidecars, count/numeric/object aliases,
unsafe descriptor flags and zero-link objects, digest and bind-time identity
drift, live/uninitialized output rejection, idempotent close, API linkage, and
an export/forbidden-dependency boundary. It does not inspect the host's real
loader/interpreter/runtime, create /opt, request sudo, or use a GPU/Docker
operation.
