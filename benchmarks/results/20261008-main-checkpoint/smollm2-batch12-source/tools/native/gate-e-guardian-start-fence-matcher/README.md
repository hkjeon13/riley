# Gate E guardian START fence matcher v1

This excluded, source-only C11 library compares one future normalized START
candidate with caller-owned boot/generation high-water claims. It fixes only
the typed anti-replay relation before a native PID1 durable ledger is designed.

An unfenced binding requires a zero boot-id claim and zero highest generation.
It accepts any valid candidate boot-id with a positive generation. A fenced
binding requires a nonzero opaque boot-id claim: a candidate must have the same
boot-id and a generation strictly greater than the stored high-water value.
A different boot-id requires durable recovery rather than a local START.

All values are fixed-width caller-owned claims. This library does not parse a
session, read or write a ledger, rehydrate recovery state, advance a phase,
create a lease, or inspect a PID1 process, cgroup, pidfd, FD, socket, or
filesystem object. Success is not session admission, ledger mutation,
guardian installation, a Gate E action, freeze/rollback action, or
qualification input.

Failure to set a valid binding clears its reusable claims, preventing a stale
fence from being silently retained. The code has no CLI, path/configuration
input, allocation, child/process, loader, or file/network surface.

Run the fixture and static checks in the reviewed Linux builder:

    make test
    make analyze
    make test CFLAGS='-O1 -g -fsanitize=undefined -fno-omit-frame-pointer' \
      LDFLAGS='-fsanitize=undefined'

The test target additionally whitelists its object file's undefined symbols,
so syscall, socket, process, filesystem, allocator, and loader dependencies
fail closed.
