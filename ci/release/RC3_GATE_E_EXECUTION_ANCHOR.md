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
  gate-e-v3.lock
```

Every ancestor from `/` through both final directories must be root-owned and
not group/world writable and must carry no POSIX ACL. The anchor root must be
mode `0755`, the lock directory mode `0700`, the bootstrap mode `0755`, and
the core and manifest mode `0644`. Those three anchor files must be root-owned,
single-link regular files. `gate-e-v3.lock` must already exist as a root-owned,
single-link, zero-byte regular file with exact mode `0600`; the bootstrap never
creates it. The manifest is canonical JSON with a terminal newline and contains
exactly the fixed schema version, bootstrap/core filename, SHA-256, byte length,
and fixed lock-directory path. The installed public bootstrap additionally
allows only `ext4`, `xfs`, or `btrfs` for every held ancestor/final directory;
network, overlay, and unknown rich-ACL filesystems fail closed rather than
pretending the POSIX mode/xattr checks capture their write policy.

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

## v3 root-bound no-action bootstrap template

`ci/release/run_remote_rc3_gate_e_session_v3.py` is the matching audit/source
template for a future **root-installed** bootstrap. Its only public form is
the exact fixed invocation below, launched by a reviewed root-owned service or
narrow privileged launcher through an empty `execve` environment:

```text
/usr/bin/python3.10 -I -S -E -B \
  /opt/riley/rc3-gate-e-v1/run_remote_rc3_gate_e_session_v3.py \
  --bootstrap-core-smoke-test
```

The checkout copy is intentionally not callable: it rejects before opening an
anchor, lock, socket, or child. The eventual installed copy independently
checks the pinned interpreter, fixed argv/path, raw empty environment, initial
`{0,1,2}` FD set, root UID, unblocked HUP/INT/TERM and default `SIGCHLD`, PID 1
mount/user namespaces and the single full initial `0 0 4294967295` UID/GID
identity maps. It then repeats held
`openat`/no-follow/ACL checks for the anchor, authenticates
the canonical manifest and bootstrap/core digests, and compares the core to a
compiled-in SHA-256 and byte-length review pin. The mutable-checkout verifier
is neither imported nor consulted by this path.

For this template's no-action smoke path only, the parent opens the existing
fixed lock on parent FD 7 and obtains a nonblocking exclusive `flock`. It
copies the verified core and canonical configuration into independently sealed
anonymous memfds on FDs 8 and 9, creates a credential-authenticated private
`AF_UNIX SOCK_SEQPACKET` endpoint on FD 10, then forks. The child explicitly
closes FD 7 and every unapproved inherited FD, restores default/unblocked
termination signals, sets `PDEATHSIG(SIGTERM)`, and clean-environment-execs
the sealed core. The parent forwards HUP/INT/TERM as child SIGTERM, reaps it,
then releases the parent lock. A normal result is only
`bootstrap-core-no-action-smoke-test-only`; it creates no receipt or evidence.

This does not grant a privileged execution path or actual producer authority.
`-I -S -E -B` begins only after the dynamic loader: it cannot neutralize
`LD_PRELOAD`, `LD_AUDIT`, or another caller-supplied loader injection. The
installation contract must therefore use a native secure-exec guardian or a
root service/launcher that independently guarantees `execve(..., envp={})` and
does not accept an untrusted caller environment. Because Python has to load the
bootstrap leaf before this source can inspect it, that guardian or launcher
must also authenticate the bootstrap's held leaf FD, approved local filesystem,
and reviewed digest before it execs Python (or execute pre-sealed bytes); the
in-Python bootstrap check is defense in depth for the core/manifest/lock, not
the bootstrap's initial trust root. A parent-only `flock` can also be released
after a parent SIGKILL before the child finishes its `PDEATHSIG` shutdown. Any
future GPU/raw producer must use that separately reviewed launcher plus a
guardian or lease design that closes the lifetime gap. It must not treat this
template's `COMPLETE` as a capture, semantic receipt, or qualification result.

### Guardian/lease v1 model — not an installed v3 extension

`ci/release/RC3_GATE_E_GUARDIAN_LEASE.md` and
`rc3_gate_e_guardian_lease_contract_v1.py` now fix the **CPU-only,
non-authoritative** contract that a later native guardian/warden/PID1
controller must satisfy.  They do not install a guardian, change this v3
bundle layout, execute a bootstrap, or grant a launch/evidence/receipt/Gate E
authority.  In particular, the model's future sealed bootstrap/core audit
leaves and FD 31/32 handoff are a versioned successor contract, not an argv
change or compatibility claim for this v3 FD 7/8/9/10 template.

The future controller, not `flock` availability, must retain admission closed
while the exact held non-delegated worker cgroup may be populated.  It needs a
root-controller-authenticated durable ledger and must rehydrate every active
lease to `DRAINING`; only a fresh same-object `populated=false` observation,
the registered terminal worker pidfd, and explicit controller release may
return to `IDLE`.  The Python model does not create, sign, or read that ledger,
so this is a later native/PID1 implementation prerequisite rather than a host
guarantee.

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
and uses a test-only socketpair; the bootstrap's companion test instead uses a
temporary current-UID anchor and lock fixture to prove the FD 7/8/9/10 handoff.
Neither test creates `/opt` or `/var/lib/riley` paths or touches the shared GPU
lock. A normal `COMPLETE` remains no more than a protocol-mechanism result—not
producer authority, a receipt, or qualification evidence.
