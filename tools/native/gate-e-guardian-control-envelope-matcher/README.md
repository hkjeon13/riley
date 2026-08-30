# Gate E guardian control-envelope matcher v1

This standalone C11 library is a source-only, in-memory matcher for the typed
claims surrounding a future Gate E READY or NO_ACTION_COMPLETE control packet.
It is a sibling of the raw control-packet validator and deliberately accepts
no packet bytes, JSON, schema, kind, phase, or session digest/generation.

Callers initialize one expected binding from an already authenticated active
session: an unprivileged worker PID/start-time/pidfd-token/UID/GID identity and
a non-delegated held-cgroup device/inode/token identity. For each normalized
report, the matcher requires exact field equality and an explicit EMPTY
ancillary claim. Opaque token bytes are claims only, not live pidfds or FDs.
The library stores no accepted event or phase state.

It does not receive a socket message, perform SO_PEERCRED or SCM_RIGHTS
inspection, count/close FDs, inspect a cgroup or pidfd, resolve a path, parse
ELF, open a ledger, transition admission state, or install a guardian. A
successful result means only that caller-provided normalized values match. It
does not authenticate a sender or cgroup, authorize a release, or make
NO_ACTION_COMPLETE a release signal. Root, GPU, Docker, evidence, receipt,
freeze, rollback, and qualification operations remain out of scope.

Run fixture-only checks in a Linux C11 environment:

    make test
    make analyze

The test target permits only memory primitives and compiler sanitizer/stack
instrumentation as unresolved library-object symbols; it rejects every other
operating, transport, process, filesystem, allocator, or loader surface.
