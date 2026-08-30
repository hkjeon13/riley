# RC3 Gate E external execution anchor

`/home/psyche/rustinfer-vllm-roadmap-serial` is a development checkout owned
by the service user. It is intentionally not an authority to start a GPU
capture, even when its source-bound no-action probes pass.

Before a future actual Gate E producer is introduced, a system administrator
must provision the following fixed locations outside the checkout:

```text
/opt/riley/rc3-gate-e-v1/
  execution-anchor.json
  run_remote_rc3_gate_e_session_v3.py
  rc3_gate_e_private_raw_core_v1.py
/var/lib/riley/rc3-gate-e/lock/
```

Every ancestor from `/` through both final directories must be root-owned and
not group/world writable. The anchor root must be mode `0755`, the lock
directory mode `0700`, and all three files must be root-owned, single-link,
regular files with no group/world write bit. The bootstrap must be owner
executable. The manifest is canonical JSON with a terminal newline and
contains exactly the fixed schema version, the bootstrap/core filename,
SHA-256, byte length, and the fixed lock-directory path.

`ci/release/verify_rc3_gate_e_execution_anchor_v1.py` accepts only this
fixed contract, under the reviewed isolated interpreter:

```sh
/usr/bin/python3.10 -I -S -E -B \
  ci/release/verify_rc3_gate_e_execution_anchor_v1.py \
  --anchor-contract-probe
```

It opens the directory chain and leaves through no-follow file descriptors,
hashes the opened bootstrap/core bytes, and checks them against the
root-owned manifest. It does not execute either file, acquire the GPU lock,
query a GPU, invoke Docker, create evidence, or publish a receipt. Until an
administrator installs a reviewed v3 bundle, this probe is expected to fail
closed because the fixed anchor paths are absent.

Its `checked` JSON is an installation preflight only. The verifier itself is
still source in a mutable checkout, so it does not establish verifier-source
integrity, host mount-namespace identity, or POSIX ACL write prohibition. It
must never be accepted as launch authority, a producer normal-return, a
semantic receipt input, or qualification evidence. A future root-installed
bootstrap must repeat those checks inside its own immutable host context.

The v3 bootstrap/core bundle is deliberately not installed by this repository
change. Installing an empty directory, copying mutable checkout files, or
changing the verifier's fixed paths does not authorize a capture. A later
reviewed bundle must carry its own core digest in the root-owned bootstrap,
retain the parent-only lock, and use the locked FD stack directly; it must not
reuse the retired Bash runner, the v1 smoke probe, the v2 source probe, or the
aggregate replay record as authority.

## v3 private-core no-action template

`ci/release/rc3_gate_e_private_raw_core_v1.py` is an audit/source template for
the private core named by the future anchor. It is deliberately **not** an
installed bundle and is not a public command: every direct checkout invocation
fails before it opens a control socket, inspects a lock, or creates a child.

The template accepts only the future bootstrap's fixed isolated handoff:

```text
/usr/bin/python3.10 -I -S -E -B /proc/self/fd/8 --sealed-no-action-core
```

The bootstrap must first authenticate the root-owned manifest and its own/core
held descriptors. It may then copy the verified core bytes into a sealed
anonymous `memfd` on FD 8, place one canonical sealed configuration `memfd` on
FD 9, and pass one private `AF_UNIX SOCK_SEQPACKET` endpoint on FD 10. The
child permits exactly `0,1,2,8,9,10`; in particular it rejects the parent lock
FD 7 and every other inherited descriptor, then marks its internal FDs
close-on-exec. It verifies the core digest and length against the configuration,
checks its isolated pinned Python and immediate parent's
PID/start-time/credentials/executable, restores an unblocked default
`SIGTERM`, sets `PDEATHSIG`, and requires `SO_PEERCRED` plus a per-packet
`SCM_CREDENTIALS` record.

The nonce- and configuration-digest-bound exchange is only
`INIT -> READY -> RUN_NO_ACTION -> COMPLETE`.
Linux may autobind an unbound `SO_PASSCRED` endpoint to a short abstract name;
that non-filesystem kernel form is accepted, while ordinary filesystem or
arbitrary abstract socket names are rejected. Packets are bounded canonical
JSON and reject truncation, duplicate/unknown ancillary records, passed FDs,
or a wrong nonce.

This first core version has no lock acquisition, GPU query, Docker execution,
filesystem output, raw producer, semantic replay, receipt, or qualification
capability. Its CPU-only test copies the template into temporary sealed memfds
and uses a test-only socketpair; it does not create `/opt` or `/var/lib/riley`
paths and does not touch the shared GPU lock. A later root-installed bootstrap
must independently enforce host mount-namespace and ACL policy, retain the
parent-only lock, and treat a normal `COMPLETE` as no more than this protocol
mechanism result—not producer authority, a receipt, or qualification evidence.
