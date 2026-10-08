# Gate E native platform preflight

This Linux C11 tool is a **source/audit inspection tool**, not a Riley runtime
component. It remains under the excluded `tools/native` root and is never a
Cargo dependency, release-bundle input, systemd unit, root installer, or Gate E
producer.

Its only public form is:

```text
gate_e_platform_preflight --observe-linux-platform-v1
```

It takes no paths, configuration, anchor, lock, GPU, Docker, evidence, or
receipt arguments. It makes only read-only observations of fixed `/proc` and
`/sys` paths: effective root identity, a full `0 0 4294967295` UID/GID map,
equality to PID 1's user/mount/cgroup namespaces, PID 1's `systemd` identity,
cgroup v2 root ownership/mode, and the Linux `openat2` ABI. These are
PID1-relative observations, not proof that the caller is in a host-initial
namespace. The tool has no `openat` fallback when that ABI is unavailable.

The one-line JSON diagnostic is deliberately fixed to
`scope=platform-observation-only`, `authority=not-authoritative`,
`installation=not-installed`, and `qualification_status=not-run`. Even a
`status=checked` result does not authenticate a bootstrap, grant execution
authority, create a cgroup or lock, launch a process, query a GPU, invoke
Docker, write evidence, or publish a receipt.

Run the fixture-only tests with a host C compiler:

```bash
cd tools/native/gate-e-platform-preflight
make test
```

The test target compiles only below a fresh `mktemp` directory and removes that
exact temporary binary and directory after a successful test. It does not need
root and never opens a GPU or changes a cgroup. A real native guardian and its
root-owned systemd/PID1 installation remain a separately reviewed future
boundary.
