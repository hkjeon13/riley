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
