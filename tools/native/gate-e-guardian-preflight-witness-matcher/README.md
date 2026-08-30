# Gate E guardian preflight-witness matcher v1

This excluded, source-only C11 library matches one future normalized
`NATIVE_PREFLIGHT_OK` witness against caller-owned active-session identities.
It fixes only the in-memory claim boundary before native acquisition or
controller integration is designed.

The binding setter accepts declared root guardian and warden identities plus a
declared PID 1/root controller identity, requiring the three to be distinct.
The matcher compares normalized guardian/warden/controller claims, requires a
new non-delegated cgroup claim to be structurally valid, and accepts only an
explicit `EMPTY` population declaration.

All values are caller-owned fixed-width claims. This library does **not** open,
create, reserve, or inspect a cgroup, pidfd, socket, or FD; parse JSON; retain
phase or durable-ledger state; change admission; or establish a lease. It does
not prove that the cgroup is fresh or empty, authenticate the source of a
claim, or make `NATIVE_PREFLIGHT_OK` a guardian installation, Gate E action,
freeze/rollback action, or qualification input.

The code has no CLI, path/configuration input, allocation, child/process,
loader, or file/network surface. Invalid expected bindings are cleared so a
stale active-session identity cannot be reused.

Run the fixture and static checks in the reviewed Linux builder:

```sh
make test
make analyze
make test CFLAGS='-O1 -g -fsanitize=undefined -fno-omit-frame-pointer' \
  LDFLAGS='-fsanitize=undefined'
```

The test target additionally whitelists its object file's undefined symbols,
so syscall, socket, process, filesystem, allocator, and loader dependencies
fail closed.
