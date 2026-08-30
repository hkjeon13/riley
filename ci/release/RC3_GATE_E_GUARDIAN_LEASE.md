# RC3 Gate E guardian/lease v1 contract

## Status and scope

`rc3_gate_e_guardian_lease_contract_v1.py` is a **CPU-only model** for the
future Gate E native guardian, lease warden, and PID1/system-manager admission
boundary.  It is not an installed service, an executable launcher, an
administrator provisioning recipe, or authority to start a producer.

It deliberately performs no system call for a lock, cgroup, socket, child,
signal, GPU, Docker daemon, evidence root, receipt, semantic replay, candidate
freeze, or qualification.  Its only outputs are in-memory parsed values and
state transitions whose scope is `guardian-lease-contract-only`, authority is
`not-authoritative`, and installation status is `not-installed`.

This contract is a prerequisite for a later, separately reviewed root-owned
implementation.  It does not change the fact that the current administrator
bundle is absent from `server-4096`; the current v3 checkout templates remain
fail-closed audit/source material only.

## Why the current v3 template is insufficient

The v3 bootstrap template has a parent-only FD 7 `flock`; its child/core does
not inherit that FD.  A `SIGKILL` of the parent can release that file lock
before a child observes `PDEATHSIG` and exits.  A next producer must not use
the resulting lock availability as evidence that the prior workload is gone.

Also, a Python wrapper runs only after the platform dynamic loader has begun
to load its interpreter and script.  Its own path/digest checks therefore
cannot authenticate the bootstrap leaf before Python is loaded.  The future
native guardian, not a checkout Python script, owns this pre-Python boundary.

The future guardian contract intentionally names new audit leaves
`rc3_gate_e_guardian_bootstrap_v1.py` and
`rc3_gate_e_guardian_no_action_core_v1.py`.  They are not installed files and
are not aliases for `run_remote_rc3_gate_e_session_v3.py` or
`rc3_gate_e_private_raw_core_v1.py`.  The latter's FD 7/8/9/10 handoff cannot
be silently re-used as this future boundary.

## Trust boundary and immutable handoff

The later implementation has four distinct identities, each bound by PID,
start-time ticks, an opaque pidfd token, UID, and GID:

| Actor | Required role | Authority retained |
|---|---|---|
| native guardian | root, static `native-root-guardian` | validates held bootstrap/manifest/interpreter/runtime objects before Python and makes the same-object handoff |
| lease warden | root, `native-lease-warden` | retains the private lease and controls the non-delegated worker cgroup |
| controller | PID 1 root `pid1-system-manager` | the sole admission authority; survives guardian/warden loss and records durable fencing state |
| worker | non-root, no capabilities | no-action bootstrap/core work only; receives no lease or cgroup-control capability |

All four identities must be distinct.  The controller requires the full
initial `0 0 4294967295` UID and GID maps, rather than a user-namespace claim
of UID 0.  Future native code must also validate host/mount/cgroup namespace
context, root ownership, ACL absence, link count, approved local filesystem,
and manifest closure from held no-follow descriptors.  A pathname is only an
audit label after that point: it must not be resolved again to execute the
object.

The model binds the root-owned manifest to the exact digest, byte length, and
opaque held-object token of the bootstrap, core, interpreter, and runtime
closure.  The interpreter must use the same verified executable object
(`execveat`/secure-exec in the future implementation).  The bootstrap is
derived from the guardian's held object into a sealed memfd; it is not opened
again by pathname.

The bootstrap stage may hold only stdio plus the sealed bootstrap FD 31 and
sealed core FD 32.  The core FD is consumed from that same verified object and
closed before the unprivileged worker stage.  The worker stage has exactly
`0,1,2`, a raw empty `execve` environment, `no_new_privs`, zero capabilities,
and no inherited lease or cgroup-control FD.  This document does not provide
that handoff implementation.

## Durable admission and cgroup lease

The safe lifetime condition is not “a file lock is currently held.”  It is:

> A new admission is forbidden while the prior session's exact held worker
> cgroup may still be populated.

The cgroup identity is `{st_dev, st_ino, held_fd_token, non_delegated:true}`;
no path string, cgroup name, PID alone, or reopened descriptor is sufficient.
The controller maintains an authenticated durable admission ledger with the
boot ID, monotonic/fencing generation, active session/lease/nonce, current
controller identity, held cgroup identity, and most recent population
observation.  The model represents the required recovery input with
`rehydrate_controller_from_durable_ledger()`; it does not read or sign an
actual ledger.

On controller restart, any retained active ledger restores only to
`DRAINING`, even if its last stored observation said empty.  It cannot restore
directly to `IDLE`.  A fresh authenticated observation of the same held
cgroup with `populated=false`, plus the registered terminal worker pidfd token,
is required before explicit controller release.  The durable fencing record
survives release so the same boot/generation cannot be replayed.  A different
boot ID requires an authenticated recovery record in the future native PID1
implementation.  The public Python model deliberately exposes
`initial_state()` as a synthetic test fixture and cannot prevent an arbitrary
caller from constructing a blank value; it must never be used as a post-crash
recovery mechanism or enforcement authority.

This depends on a future PID1/system-manager implementation that actually
persists and authenticates the ledger.  Until that implementation is reviewed
and installed, this is a design invariant, not a host guarantee.

## State machine

```text
IDLE
  -> PREFLIGHT
  -> LEASED_EMPTY
  -> BOOTSTRAP_STARTING
  -> NO_ACTION_LIVE
  -> DRAINING
  -> EMPTY_VERIFIED
  -> IDLE
```

`START` first validates the complete static no-action contract and closes
admission in `PREFLIGHT`.  A `NATIVE_PREFLIGHT_OK` must bind the registered
guardian, warden, controller, and a fresh exact cgroup with
`populated=false`.  Only then can `BOOTSTRAP_EXECED` bind the registered
non-root worker to that cgroup.  `READY` and `NO_ACTION_COMPLETE` are bounded
canonical JSON packets from that exact worker identity, and each packet binds
the boot ID, lease ID, generation, nonce, and guardian contract digest.

`STOP`, timeout, malformed control input, ancillary FDs, bootstrap exit,
guardian/warden/controller loss or `SIGKILL`, cgroup read failure, pidfd
failure, every unsupported event, and every action-like event move any leased
state to `DRAINING`.  In `DRAINING`, a populated observation is a self-loop.
Only exact cgroup empty plus exactly the registered terminal worker pidfd token
moves to `EMPTY_VERIFIED`; only the current controller can then release to
`IDLE`.

`RUN_CAPTURE`, GPU query, Docker, evidence/receipt write, qualification, and
release-from-bootstrap are structurally rejected.  There is no success path
for them in v1.

### Native acquisition/cutover ordering

`PREFLIGHT` in this model means that no native lease or cgroup has yet been
acquired.  Its transition back to synthetic `IDLE` is safe only under that
condition.  A future implementation must first durably persist closed
admission/session fencing, and must atomically commit (or conservatively retain
as active) the exact held cgroup record before any cgroup/lease acquisition can
become observable or a worker can be launched.  Once a lease FD or cgroup is
acquired—or if a crash makes that fact uncertain—it must never report
`PREFLIGHT_FAIL` or return to `IDLE`; restart recovery must use the active
durable record and enter `DRAINING` until fresh empty/pidfd checks complete.

Future hostile tests must inject guardian/warden/controller loss at every
cutover: before durable intent, after intent, after lease/cgroup acquisition,
after active-record commit, during bootstrap handoff, and after `READY`.
They must prove that none of these windows permits a second admission while a
prior held cgroup could still contain a descendant.

## Control packets and failure policy

Packets are UTF-8 canonical JSON, at most 4 KiB, with no duplicate keys or
ancillary FD.  They contain only the fixed control schema, packet kind,
boot ID, lease ID, generation, nonce, and contract SHA-256.  The controller
also compares SCM-like credentials to the registered worker PID/start-time/
pidfd/UID/GID identity and compares the cgroup's held device/inode/token.
PID reuse, stale generation, stale nonce, a mismatched cgroup, boolean-as-
integer generation, noncanonical JSON, or any packet shape drift drains rather
than releasing admission.

The modeled durable ledger is deliberately strict: malformed or incomplete
active records are rejected; a claimed empty ledger cannot retain a cgroup;
and an active record always rehydrates closed.  These are CPU-only hostile-path
checks, not proof that a platform's cgroup or pidfd APIs work.

## Required future implementation review

Before any root installation or actual producer, a new review must provide all
of the following as native/PID1-enforced behavior:

1. held `openat2`/no-follow object authentication, local-filesystem/ACL/
   ownership checks, and immutable manifest closure before the loader/Python;
2. same-object `execveat`/sealed-memfd transfer with no post-check path
   re-resolution, loader-injection boundary, or extra inherited capabilities;
3. PID1-owned durable authenticated ledger with an acquisition/commit ordering
   that either atomically records the held cgroup or conservatively recovers it
   closed after every crash window;
4. a non-delegated cgroup whose population is rechecked after guardian,
   warden, controller, or worker loss, plus pidfd-based terminal observation,
   cgroup-empty verification, and exact controller-only release;
5. failure-injection coverage for every durable-ledger/lease/cgroup/bootstrap
   cutover, including an unknown-after-acquire crash; and
6. a separate authorization review before adding any GPU, Docker, capture,
   evidence, receipt, freeze, semantic replay, or qualification operation.

Until all six exist in a separately installed and reviewed native design,
this contract must not be used as a launch, receipt, Gate E, or C02
qualification input.

## Native platform preflight v1 — not a guardian

`tools/native/gate-e-platform-preflight/gate_e_platform_preflight.c` is a
standalone C11 **platform-observation-only** inspection tool. It accepts only
`--observe-linux-platform-v1`, has no caller path/configuration/anchor/lock
input, and makes read-only observations of the fixed Linux `/proc` and `/sys`
surfaces. In particular, it requires a full `0 0 4294967295` UID/GID map,
compares the current user/mount/cgroup namespace objects with PID 1, checks PID
1's `systemd` identity and a root-owned non-writable cgroup-v2 root, and probes
the `openat2` ABI with no `openat` fallback. These are PID1-relative
observations, not proof of a host-initial namespace. It does not create or mutate a
cgroup, lock, socket, child, signal, GPU, Docker action, evidence root,
receipt, or ledger.

Its transient JSON has schema
`riley.rc3-gate-e-native-platform-preflight.v1`, scope
`platform-observation-only`, authority `not-authoritative`, installation
`not-installed`, and qualification `not-run`. A `checked` observation means
only that those fixed host traits were observed at that instant. It does not
authenticate the immutable bootstrap, establish secure `execveat`, create the
PID1 durable ledger, reserve or recheck a non-delegated worker cgroup, observe
a worker pidfd, cover any crash cutover, or authorize GPU/Docker/evidence/
receipt/freeze/semantic/qualification work. Thus all six future implementation
requirements above remain **not established** by this tool.

## Native root-bundle authenticator v1 — not a guardian

`tools/native/gate-e-root-bundle-authenticator/gate_e_root_bundle_authenticator.c`
is a second standalone C11 source/audit precursor. Its only public form is
`--authenticate-root-bundle-v1`; it has no caller-controlled paths, owner IDs,
manifest names, lock names, GPU/Docker/evidence arguments, or configuration.
For a root caller it reads only the future audit-leaf directory
`/opt/riley/rc3-gate-e-v1`, holding `openat2`/no-follow descriptors from `/`
through the fixed manifest/bootstrap/core leaves. It requires `O_NOATIME`,
root ownership, exact modes, single regular-file links, ACL absence, approved
local filesystem type, bootstrap capability absence, exact canonical manifest
grammar, held-object identity rechecks, and bounded SHA-256/length closure.
It closes every descriptor afterwards and never installs or creates that
directory.

This is a read-only object observation only. Its JSON keeps top-level
`status` at `not-established`; an `object_observation_status` of `checked`
records only one consistent observation. It does not establish host-initial
namespace identity, dynamic-loader or pre-Python trust, same-object
`execveat`, interpreter/runtime closure, a lease/cgroup/pidfd controller,
execution authority, a Gate E producer, GPU/Docker action, evidence, receipt,
freeze, rollback, or qualification. Its schema and the future root-bundle
manifest schema are explicitly denied when presented as canonical documents to
the qualification-input policy; its actual terminal-newline diagnostic line is
rejected even earlier as noncanonical input. Neither form is an input channel.

The checkout-built dynamic program itself begins after its own loader, so it
cannot be the required pre-loader trust root or substitute for the separately
reviewed static native guardian. It also does not install the future FD 31/32
bootstrap/core leaves and is not a compatibility path for the old v3 FD
7/8/9/10 Python template.

## Native sealed leaf snapshot v1 — not a guardian

`tools/native/gate-e-sealed-leaf-snapshot/` is a source-only C11 library
primitive for one much narrower part of the future FD 31/32 model. Given an
already authenticated and held regular source descriptor, it first pins a
private `CLOEXEC` duplicate, copies bounded bytes with fixed-offset I/O, and
returns only an anonymous data-only `MFD_NOEXEC_SEAL` memfd after exact
`F_SEAL_EXEC|F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL` verification.
It closes its private source duplicate before publishing the result and never
reopens a path. It requires the Linux 6.3+ no-exec memfd ABI; no weaker seal or
temporary-file fallback exists.

This is not integrated with the root-bundle authenticator, which deliberately
closes its audit descriptors after read-only observation. The snapshot does not
authenticate the source path/manifest/object token/owner/ACL/filesystem,
interpreter or runtime closure, host namespace, loader boundary, or same-object
execution. It creates no FD 31/32 arrangement, no child, lease, cgroup, PID1
ledger, GPU/Docker action, evidence, receipt, freeze, rollback, or
qualification result. A no-exec Python leaf remains data for a separately
authenticated interpreter, never an `execveat` substitute. Thus it is only an
audit implementation precursor, not a launch or authority edge.

## CPU-only verification

Run from the repository after the source file is present:

```bash
/usr/bin/python3.10 -I -S -E -B \
  ci/release/test_rc3_gate_e_guardian_lease_contract_v1.py
```

The tests exercise the normal no-action state path, parent/guardian/warden
loss, cgroup population/identity mismatch, pidfd mismatch, stale and malformed
control packets, FD handoff drift, manifest/object binding drift, durable
ledger rehydration, controller restart, replay fencing, and structural action
rejection.  They do not contact the GPU host or create an operational lock.
