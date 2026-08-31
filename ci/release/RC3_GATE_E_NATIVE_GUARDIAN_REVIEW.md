# RC3 Gate E native guardian review boundary v1

## Status

This is a C02-P2 design-review prerequisite, not an implementation,
installation guide, service unit, launch command, or qualification input. It
records the decisions that must be reviewed before a root-installed static
guardian can be written. It grants no authority to create /opt or
/var/lib/riley, run a child, access a GPU or Docker, capture evidence, write a
receipt, freeze a candidate, or qualify RC3.

The checked-in source precursors are intentionally narrower:

- gate-e-root-bundle-authenticator retains authenticated bootstrap/core
  audit-leaf descriptors from the fixed v1 bundle.
- gate-e-root-bundle-sealed-leaves turns those two data leaves into sealed,
  no-exec snapshots.
- gate-e-execution-closure-held-fds binds caller-held loader, interpreter,
  and runtime files to a canonical sidecar, but does not authenticate the raw
  sidecar or inspect ELF.

They do not compose into a native guardian yet.

## Blocking decisions

### 1. Immutable bundle revision

The existing root-bundle v1 grammar authenticates only the manifest,
bootstrap, and core. It does not bind an execution-closure sidecar, a static
guardian binary, an interpreter, a dynamic loader, or runtime leaves.

Before implementation, reviewers must approve a new versioned immutable bundle
format and fixed installation location. It must not silently repurpose the v1
path or mutate its grammar. Its root manifest must bind, at minimum:

| Object | Required immutable binding |
| --- | --- |
| static guardian | byte length, digest, owner/mode/link and static-link review result |
| bootstrap and core | byte length, digest, immutable data-leaf policy |
| execution-closure sidecar | exact raw byte length and SHA-256 |
| interpreter | byte length, digest, executable-object policy |
| dynamic loader | byte length, digest, loader strategy identifier |
| runtime closure | raw sidecar SHA-256 and an approved resolution-policy revision |

The future guardian must acquire every listed item from the reviewed fixed
bundle through held no-follow descriptors. Passing arbitrary raw
execution-closure bytes to the current held-FD binder is not a substitute for
that authentication.

### 2. Dynamic-loader execution strategy

The sidecar dynamic-loader audit path is a declaration only. A held FD for
that loader does not by itself prove how the kernel will resolve a dynamic
ELF PT_INTERP during execveat, nor does it prove DT_NEEDED, RPATH/RUNPATH,
cache, namespace, or standard-library closure behavior.

Before an execveat call is implemented, a reviewer must select and document
exactly one strategy:

1. a fully static reviewed interpreter; or
2. a reviewed same-object dynamic-loader method with an explicit proof of
   every loader and dependency-resolution step.

The selected strategy must define the supported ELF class, machine,
interpreter-header relation, dependency-resolver rules, rejection rules, and
the exact test matrix. No source helper may infer that selection from a
pathname, ldd, host loader cache, environment variable, or a successful local
launch.

### 3. Secure-handoff ABI

The successor handoff must be specified before implementation, including:

- guardian input descriptors and ownership;
- an empty raw environment, with no LD family, Python family, locale, path, or
  inherited configuration fallback;
- close/duplicate order that leaves the bootstrap stage with only 0, 1, 2,
  sealed bootstrap 31, and sealed core 32;
- a post-handoff worker transition that consumes and closes core 32, then
  leaves the worker with only 0, 1, and 2;
- no_new_privs, capability clearing, signal disposition, UID/GID, namespace,
  and cgroup handoff requirements; and
- fail-closed behavior for every duplicate, close, seal, exec, or child-exit
  failure.

The current launch-isolation matcher validates caller-normalized claims only.
It does not inspect an FD table, call prctl, duplicate FDs, create a child, or
execute an object, so it cannot serve as this ABI implementation.

### 4. PID1 admission and durable state

The secure handoff is not an admission decision. The static-guardian design
must remain separate from the PID1 controller/warden design and define:

- a non-delegated cgroup acquisition identity;
- atomic or conservatively closed ledger ordering for acquire, commit, and
  crash recovery;
- DRAINING rehydration for every incomplete or active record;
- pidfd terminal and fresh same-object empty observation before controller-only
  release; and
- fault injection for each auth, handoff, ledger, cgroup, restart, and terminal
  cutover.

No file-lock availability, guardian loss, or warden loss is an admission
release condition.

## Required review artifacts

Implementation review cannot start until it has all of the following:

1. a threat model covering path replacement, descriptor reuse, loader
   injection, environment inheritance, namespace confusion, PID reuse, crash
   windows, cgroup delegation, and administrator rollback;
2. a versioned immutable bundle schema and canonical manifest examples;
3. a static-build recipe with pinned compiler, libc/toolchain, source and
   output digests, reproducibility procedure, and revocation/rollback owner;
4. a static-ELF inspection policy that rejects an unexpected PT_INTERP or
   dynamic-dependency surface for the selected execution strategy;
5. a syscall/FD state table for guardian, bootstrap, worker, warden, and
   PID1 controller; and
6. a CPU-only hostile-path matrix plus a separately authorized installed
   no-GPU acceptance plan.

The installed acceptance plan must be administered outside this checkout. It
may not be simulated by root Docker, a mutable checkout copy, a user-owned
directory, or a GPU-enabled test run.

## Implementation gate

Until the blocking decisions and artifacts above are approved, the only valid
repository work is source/design review and CPU/static verification. In
particular, the repository must not introduce a partial execveat launcher,
pretend that a dynamic-loader FD establishes same-object execution, add a
root-service unit, create the fixed directories, or treat GPU/Docker access as
guardian authorization.

On server-4096, the required root bundle and lock path are absent and
non-interactive sudo is unavailable. Therefore administrator provisioning,
installed no-GPU acceptance, and every GPU/Docker/producer operation remain
external prerequisites.
