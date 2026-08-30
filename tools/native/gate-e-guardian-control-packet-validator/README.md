# Gate E guardian control-packet validator v1

This standalone C11 library is a source-only, in-memory parity precursor for
the future Gate E guardian control channel. It accepts one caller-owned raw
byte span and compares it to a caller-owned active-session binding. It is not
a socket receiver, guardian, launcher, or release input.

The only accepted packets are nonempty canonical JSON objects of at most 4 KiB
with no terminal newline. Their exact key order is boot_id, contract_sha256,
generation, kind, lease_id, nonce, and schema_version. The only wire kinds are
ready and no_action_complete. Every digest is a nonzero lowercase SHA-256 text
field and generation is a positive canonical uint64 decimal, including
18446744073709551615.

Before validation, callers initialize and populate a binding with the binary
boot ID, lease ID, nonce, guardian-contract digest, and generation from an
already authenticated active session. The library validates all binding inputs
at its public boundary, then accepts only exact equality with the raw packet.
It returns a reason only; it deliberately retains no accepted packet state.

It does not receive bytes from a socket, observe SO_PEERCRED or SCM_RIGHTS,
check worker credentials or cgroups, track phase or draining, read a durable
ledger, acquire a path or FD, parse ELF, execute a loader, or install a
guardian. It has no root, GPU, Docker, evidence, receipt, freeze, rollback, or
qualification authority. A later native guardian must separately authenticate
the sender, phase, worker/cgroup identity, transport framing, zero ancillary
FDs, and all held execution objects.

Run fixture-only checks in a Linux C11 environment:

    make test
    make analyze

The test target permits only memory primitives and compiler sanitizer/stack
instrumentation as unresolved library-object symbols; it rejects every other
operating, transport, process, filesystem, allocator, or loader surface.
