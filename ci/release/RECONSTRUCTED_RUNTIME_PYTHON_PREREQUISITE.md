# Reconstructed runtime Python prerequisite

The reconstructed-runtime A/B materializer delegates to reviewed PR16 replay
code that imports `tomllib`. The remote host's ambient `/usr/bin/python3.10`
must therefore remain a fail-closed controller only; it cannot perform a full
materializer replay.

`check_reconstructed_runtime_python_prerequisite_v1.py` verifies a **previously
provisioned** interpreter. Its controller code does not download Python or uv,
install packages, run `uv sync`, create an environment, write a receipt, invoke
Docker, access a GPU, or run the materializer. The fixed probe necessarily
executes the supplied external runtime, however, so those controller limits are
not an operating-system sandbox or a network/Docker/GPU isolation guarantee.

## Fixed target

The target is the repository's existing Linux x86_64 CPython pin:

- version: `3.13.15`
- executable SHA-256:
  `ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866`
- `.python-version` SHA-256:
  `861b3dd8083d28f336ef70f6755bc399538ddad627b1d095820ca34cb953cf14`

The related uv pin (`uv 0.12.5`, Linux x86_64 SHA-256
`b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46`) is
reference-only here. The preflight intentionally does not read `uv.lock`, run
uv, or let a stale dependency-lock pin decide materializer readiness.

## Safe use after toolchain provisioning is authorized

An operator must first provision the pinned interpreter in a durable,
user-owned external location such as `/home/psyche/.local/share/...`; `/tmp`,
`/var/tmp`, `/dev/shm`, a source checkout, a symlink, a hard-linked binary, and
group/world-writable executable are rejected. Toolchain acquisition changes
remote state and is deliberately outside this checker.

This is a readiness diagnostic, not a trust bootstrap or sandbox. **Before
allowing the checker to execute the supplied path**, the operator must
independently establish that the complete runtime tree (the executable,
stdlib, dynamic-loader/shared-library dependencies, and their ancestor
directories) is trusted for execution, and prevent same-UID writers from
mutating it. The path, mode, ownership, no-follow, and executable-hash checks
are useful input filters; they do not establish full-tree integrity or writer
exclusion.

Run the ambient controller in isolation, substituting the already-provisioned
absolute interpreter path:

```bash
/usr/bin/python3.10 -I -S -E -B \
  ci/release/check_reconstructed_runtime_python_prerequisite_v1.py \
  --python /home/psyche/.local/share/riley-python/cpython-3.13.15/bin/python3.13
```

The checker opens and hashes the supplied executable through no-follow held
FDs. After the hash matches, it asks the held-descriptor `/proc/self/fd/...`
path to run a fixed clean-Python-configuration `-I -S -E -B` probe. The held
descriptor prevents pathname replacement, but it does not attest the exact
post-hash executable bytes, stop a same-inode modification before execution,
or validate the stdlib/dynamic-loader/runtime tree. The Python flags are not a
sandbox. The executable leaf is rejected before hashing above 128 MiB, which
bounds ordinary byte volume but is not a host-resource or hashing-time
isolation guarantee for an externally provisioned runtime. The probe requires CPython/Linux/x86_64/3.13.15 plus `tomllib`,
`tarfile`, `hashlib`, `lzma`, `bz2`, and `sqlite3` to import.

On success, stdout is a canonical transient `checked/not-run` JSON diagnostic.
It does **not** prove the exact post-hash executable bytes were run, the whole
stdlib/runtime tree is trusted or immutable, same-UID writers were excluded, a
later materializer receives the same FD, or that the probe had no network,
Docker, or GPU side effect. It also does not prove a capture ran, A/B
materialization passed, or any freeze/Gate E/qualification result. Before
actual replay, the operator must separately verify the provisioned runtime-tree
manifest and invoke the materializer with the same explicit absolute Python
path; the old raw-capture supervisor pinned to `/usr/bin/python3` must not be
changed.
