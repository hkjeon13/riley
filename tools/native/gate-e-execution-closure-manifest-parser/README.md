# Gate E execution-closure manifest parser v1

This Linux-independent C11 library is a source/audit-only canonical-byte parser
for the future RC3 Gate E execution-closure sidecar. It is an excluded
tools/native utility, not a Riley runtime component, installer, service,
launcher, release input, or qualification producer. It has no main function,
CLI, JSON output, path input, file descriptor, or configuration surface.

Its only public operation,
gate_e_parse_execution_closure_manifest_v1(), accepts a caller-owned raw byte
span and a caller-owned initialized result structure. It accepts only the one
newline-terminated canonical v1 grammar already described by the checked-in
schema:

- root members in the exact dynamic_loader, interpreter, runtime_leaves, and
  schema_version order;
- leaf members in the exact audit_path, byte_length, and sha256 order;
- ASCII canonical audit paths, positive plain-decimal bounded lengths, and
  nonzero lowercase SHA-256 values;
- a nonempty, bytewise strictly sorted runtime list with no overlap with the
  interpreter or dynamic-loader labels; and
- the 64 KiB manifest, 128 runtime-leaf, 512 MiB per-leaf, and 2 GiB total
  declaration bounds.

On success, the fixed-size result contains copied labels, declared lengths,
decoded digests, and runtime_closure_sha256: standard SHA-256 of the exact
supplied raw bytes including the one terminal newline. There is no heap
allocation and no result pointer into caller input. A failed parse clears a
valid initialized result so a stale successful declaration cannot be reused.

## Explicit non-authority

This library does not open, re-resolve, read, stat, hash, or otherwise inspect
any audit path. It does not authenticate root ownership, ACLs, filesystem,
namespaces, device/inode identity, held-object tokens, an ELF interpreter,
PT_INTERP, DT_NEEDED, RPATH/RUNPATH, loader resolution, or Python's standard
library. It has no openat2, descriptor handoff, MFD_NOEXEC_SEAL, execveat,
loader, child, socket, signal, lock, cgroup, PID1 ledger, GPU, Docker,
evidence, receipt, freeze, rollback, Gate E producer, or qualification
behavior.

It deliberately does not extend the fixed bootstrap/core root-bundle v1
grammar or integrate with its read-only authenticator. It also must not use the
sealed-leaf snapshot primitive for interpreter or dynamic-loader execution:
that primitive produces a no-exec data memfd only. A later separately reviewed
native guardian must authenticate held objects and bind this parsed sidecar
digest to its session before any acquisition or same-object launch claim.

## Checks

Run on a C11 host compiler:

~~~
cd tools/native/gate-e-execution-closure-manifest-parser
make test
make analyze
~~~

make test builds beneath one fresh /tmp directory, uses only in-memory
canonical and hostile fixtures, checks the raw Python-contract digest,
canonical-shape/number/path/SHA/sort/budget rejection, input non-mutation, and
failure clearing, then removes its exact temporary build files. It also checks
the standalone parser object for forbidden filesystem, descriptor, process,
loader, and allocator symbols. make analyze performs a separate compiler
static-analysis build. Neither target opens an interpreter/loader/runtime leaf
or performs GPU/Docker work.
