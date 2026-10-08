# Gate E guardian launch-isolation witness matcher v1

This excluded, source-only C11 library compares one caller-normalized sealed
launch profile with separately supplied bootstrap/core held-object tokens. It
fixes only the descriptor-isolation claim boundary before a native secure-exec
guardian is designed.

The fixed profile is the future bootstrap argv shape, an empty environment,
the bootstrap descriptor set 0,1,2,31,32, the worker descriptor set 0,1,2,
sealed bootstrap/core claims, and a worker that has no lease/cgroup-control
descriptor, no capabilities, and no-new-privileges. Both descriptor masks bind
their total normalized descriptor count and highest descriptor number, so an
extra descriptor outside the 64-bit mask fails closed. Raw list order and
duplicate rejection remain the upstream canonical-session parser's
responsibility. It requires FD numbers 31 and 32 plus exact bootstrap/core
token equality, but deliberately does not require those two nonzero tokens to
differ.

All inputs are normalized fixed-width claims. This library does not inspect an
actual argv, environment, descriptor table, seals, capabilities, process, or
filesystem object; call prctl or execveat; open a path; or change a
phase/ledger/admission state. Success does not prove same-object execution,
secure launch, guardian installation, a Gate E action, freeze/rollback action,
or qualification input.

Failure to set a valid binding clears its reusable token claims. The code has
no CLI, path/configuration input, allocation, child/process, loader, or
file/network surface.

Run the fixture and static checks in the reviewed Linux builder:

    make test
    make analyze
    make test CFLAGS='-O1 -g -fsanitize=undefined -fno-omit-frame-pointer' \
      LDFLAGS='-fsanitize=undefined'

The test target additionally whitelists its object file's undefined symbols,
so syscall, socket, process, filesystem, allocator, and loader dependencies
fail closed.
