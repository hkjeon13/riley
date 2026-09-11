# PR-Q — C02-P2 native qualification trust-boundary cards

**목적:** mutable checkout의 source checker를 actual Gate E execution authority로 오인하지 않도록,
candidate qualification 전에 필요한 native guardian, secure execution, durable ledger, cgroup/pidfd
control과 administrator boundary를 PR 크기로 닫는다.

이 문서만으로 root 설치, GPU/Docker 사용, raw capture, candidate freeze, semantic replay 또는
qualification을 승인하지 않는다.

## 현재 source 출발점

Q-track은 새로 처음 만드는 것이 아니라 아래 source precursor를 조합·승격하는 작업이다.

```text
ci/release/RC3_GATE_E_NATIVE_GUARDIAN_REVIEW.md
ci/release/RC3_GATE_E_GUARDIAN_LEASE.md
ci/release/gate_e_native_guardian_review_contract_v1.py
ci/release/rc3_gate_e_guardian_lease_contract_v1.py
benchmarks/release/candidates/gate-e-native-guardian-review-v1.schema.json
tools/native/gate-e-root-bundle-authenticator/
tools/native/gate-e-root-bundle-sealed-leaves/
tools/native/gate-e-execution-closure-{manifest-parser,held-fds}/
tools/native/gate-e-guardian-*-matcher/
tools/native/gate-e-guardian-control-{packet-validator,envelope-matcher}/
```

각 precursor의 `checked`/`accepted`는 source fact일 뿐 installed guardian authority가 아니다.

## provisional installation contract

Q01 review가 version을 올리지 않는 한 현재 fixed anchor는 다음이다.

```text
/opt/riley/rc3-gate-e-v1/
  execution-anchor.json
  run_remote_rc3_gate_e_session_v3.py
  rc3_gate_e_private_raw_core_v1.py
/var/lib/riley/rc3-gate-e/lock/
  gate-e-v3.lock
```

anchor root는 root-owned `0755`, lock directory는 `0700`, bootstrap은 `0755`, core/manifest는
`0644`, lock file은 single-link zero-byte `0600`이다. `/`부터 final object까지 local filesystem,
root ownership, no group/world write, POSIX ACL 부재와 link count를 검증한다. future guardian binary,
ledger/cgroup paths와 FD ABI는 Q01 review에서 exact version으로 확정한 뒤에만 provision한다.

## 고정 보안 성질

후속 구현은 다음을 하나의 연결된 경계로 만족해야 한다.

1. loader/Python 전에 root-owned bundle을 held no-follow descriptors로 인증한다.
2. 인증한 동일 object를 pathname 재해석 없이 clean environment/FD set으로 실행한다.
3. PID1 controller가 lease와 generation을 durable ledger에 기록하고 crash 후 `DRAINING`으로 복구한다.
4. registered worker pidfd terminal과 동일 held cgroup의 fresh empty 관측 뒤에만 admission을 연다.
5. authentication, acquire, commit, move, terminal, release의 모든 cutover를 fault-inject한다.
6. no-GPU acceptance와 GPU/Docker/capture 권한을 별도 승인으로 유지한다.

## PR-Q01 — review contract와 immutable ABI freeze

**한 가지 목적:** 구현 전에 trust root, static/dynamic-loader strategy, bootstrap/worker FD ABI,
guardian/warden/PID1 state machine, ledger/cgroup release predicate를 reviewed 문서와 closed schema로 고정한다.

**변경 범위:** `ci/release/`의 review 문서/schema/checker와 CPU fixtures만. native launcher, installation,
GPU/Docker option, success receipt는 추가하지 않는다.

**검증:** missing artifact digest, unknown field, noncanonical bytes, ambiguous loader strategy,
FD collision, incomplete state transition을 fail-closed한다.

**완료:** reviewer/administrator가 같은 canonical review input digest를 승인한 외부 record가 있어야 한다.
source checker의 `accepted`만으로 완료하지 않는다.

## PR-Q02A — pre-loader held-object authentication

**선행:** Q01 승인.

**한 가지 목적:** fixed root bundle과 interpreter/loader/runtime closure를 held descriptors로 인증해
single-owner immutable handle로 반환한다.

**변경 범위:** 기존 `tools/native/gate-e-*` precursor의 authentication composition과 hostile-path
tests. exec, PID1 ledger/cgroup, GPU/Docker, evidence writer는 제외한다.

**필수 계약:** `openat2`/no-follow, local filesystem/owner/mode/ACL/link-count/digest, canonical manifest,
pre/post identity와 alias denial. check 뒤 pathname reopen은 금지한다.

**완료:** symlink/rename/replace/ACL/owner/mode/digest/FD race를 모두 거부하고 held authenticated
objects의 identity/digest를 재검증할 수 있다.

## PR-Q02B — same-object secure exec

**선행:** Q02A.

**한 가지 목적:** Q02A가 인증한 held object만 post-check pathname 재해석 없이 successor에 넘긴다.

**필수 계약:** `execveat` 또는 approved sealed handoff, empty env, FD scrub, `no_new_privs`,
capability-free, exact bootstrap/worker FD ABI, loader injection denial.

**완료:** authenticated/executed object identity가 같고 argv/env/FD/capability hostile fixtures를 거부한다.

## PR-Q03A — PID1 durable ledger

**선행:** Q01 state machine freeze. Q02A/B와 source 구현은 병렬 가능하다.

**한 가지 목적:** one active qualification lease를 crash/restart에도 보수적으로 유지한다.

**변경 범위:** native ledger library, schema, file durability/fault model. cgroup observation, actual GPU child와
benchmark producer는 제외한다.

**필수 상태:** `EMPTY → ACQUIRING → ACTIVE → DRAINING → EMPTY`; boot identity, monotonic generation,
lease nonce, held cgroup identity, worker pid/start/pidfd token, fsync/commit phase를 record한다.

**검증:** acquire 전/후, ledger temp/write/fsync/rename/dir-fsync와 controller restart cutover.
malformed/incomplete/unknown-after-acquire는
`DRAINING`으로 남고 admission을 열지 않는다.

## PR-Q03B — cgroup/pidfd release controller

**선행:** Q03A.

**한 가지 목적:** registered worker pidfd terminal과 same held non-delegated cgroup의 fresh empty 관측 뒤에만
controller가 ledger release를 commit한다.

**검증:** cgroup move/observation, worker exit, PID reuse, pidfd mismatch, controller restart, empty-before-terminal,
terminal-before-empty, release cutover fault injection.

## PR-Q04 — guardian/warden/controller integration

**선행:** Q02A/Q02B/Q03A/Q03B.

**한 가지 목적:** credential-authenticated control transport와 exact state transitions로 secure child와
durable controller를 결합한다.

**검증:** sender credential mismatch, stale generation/epoch, ancillary FD injection, guardian/warden loss,
PID1 restart, duplicate terminal, orphan cgroup, signal race. success packet 하나를 release 신호로 쓰지 않는다.

**완료:** controller만 release를 결정하며 worker/guardian이 사라져도 admission이 자동으로 열리지 않는다.

## RUN-Q05 — administrator provisioning과 no-GPU acceptance

**선행:** Q01~Q04 source review와 reproducible native binary digest.

**절차:** reviewed immutable bundle, service/PID1 integration, ledger root, cgroup root와 lock을
administrator가 provision한다. 설치 직후 installed-object inspection, secure-exec, restart/ledger recovery,
cgroup/pidfd, complete cutover fault matrix를 GPU/Docker 없이 실행한다.

**완료:** create-only no-GPU acceptance record와 explicit operational authorization decision이 모두 있다.
no-GPU pass는 GPU/Docker/capture 권한이 아니다.

## RUN-Q06 — 별도 GPU/Docker operation authorization

authority holder가 exact host, GPU UUID, Docker operation, evidence root, candidate freeze scope와
time window를 승인한 뒤에만 [05](05-candidate-gpu-and-competitive-campaign.md)를 시작한다.

## 중단 조건

- dynamic-loader strategy 또는 FD ABI가 review에서 확정되지 않음
- installed bundle과 reviewed digest 불일치
- ledger의 crash window에서 admission이 열릴 수 있음
- same-object execution을 pathname 재해석 없이 보장하지 못함
- administrator/no-GPU/GPU-Docker 승인 중 하나가 없음

이 경우 `blocked`로 기록하며 checkout script나 container-root로 host-root boundary를 우회하지 않는다.
