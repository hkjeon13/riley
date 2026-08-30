# Gate E guardian drain-witness matcher v1

This excluded, source-only C11 library matches one future normalized
`CGROUP_EMPTY` witness against caller-owned active-session claims. It exists
only to fix the fieldwise boundary before any native transport/controller
integration is designed.

The binding setter accepts a declared PID 1/root controller identity, a
registered worker terminal-pidfd token, and a declared non-delegated held
cgroup identity. The matcher accepts a normalized controller/cgroup report,
an explicit `EMPTY`/`PRESENT` population declaration, and a declared terminal
token count plus token. It succeeds only when the controller and cgroup exactly
match, population is explicitly `EMPTY`, and exactly one terminal token matches
the registered worker token.

All values are caller-owned fixed-width claims. This library does **not** open
or inspect a cgroup, pidfd, socket, SCM_RIGHTS message, or FD; parse JSON;
retain phase or durable-ledger state; change admission; or authorize controller
release. A successful match is only an in-memory typed-claim comparison. It
does not make `CGROUP_EMPTY` a release action, a guardian installation, or a
Gate E/freeze/rollback/qualification input.

The code has no CLI, path/configuration input, allocation, child/process,
loader, or file/network surface. Invalid expected bindings are cleared so a
stale active-session claim cannot be reused. This still does not authenticate
the source of a caller-provided claim.

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
