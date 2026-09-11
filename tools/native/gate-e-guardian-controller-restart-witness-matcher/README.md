# Gate E guardian controller-restart witness matcher v1

This excluded, source-only C11 library matches one future normalized
`CONTROLLER_RESTART` empty-witness against caller-owned active-session claims.
It fixes only the typed-claim boundary before native PID1, transport, durable
ledger, or cgroup integration is designed.

The binding setter accepts declared root guardian/warden identities, one
unprivileged registered worker identity, and a declared non-delegated held
cgroup. The matcher requires a declared PID 1/root replacement controller that
is distinct from the registered guardian, warden, and worker; the same held
cgroup; explicit `EMPTY` population; and exactly one terminal worker-token
declaration matching the registered worker token.

All values are caller-owned fixed-width claims. This library does **not** open,
create, reserve, or inspect a cgroup, pidfd, socket, or FD; parse JSON; retain
phase or durable-ledger state; change admission; or authorize controller
release. It does not prove that the controller actually restarted, that the
cgroup is fresh or empty, or that a supplied claim came from a kernel object.
Success is not `EMPTY_VERIFIED`, guardian installation, a Gate E action,
freeze/rollback action, or qualification input.

The code has no CLI, path/configuration input, allocation, child/process,
loader, or file/network surface. Invalid expected bindings are cleared so a
stale active-session claim cannot be reused.

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
