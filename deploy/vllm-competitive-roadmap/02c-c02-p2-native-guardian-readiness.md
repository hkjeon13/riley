# C02-P2 — Native Guardian과 Root Provisioning Readiness

**상태:** In progress — 이 문서는 C02-P1의 source-only/native typed-claim 경계 뒤에 남은 실제 실행 신뢰 경계를 독립 선행 조건으로 고정한다. guardian·warden·PID1 controller·durable ledger 또는 root bundle은 아직 설치하지 않았고, GPU/Docker/capture/evidence/receipt/freeze/semantic replay/qualification 권한도 부여하지 않는다.

**의미 등급:** `reference` + C02 release-gate corrective prerequisite
**한 가지 목적:** mutable checkout과 CPU-only 선언을 실행 권한으로 오인하지 않도록, 실제 Gate E producer 전 필요한 native guardian·root provisioning·no-GPU acceptance·별도 GPU/Docker 승인 순서를 검토 가능한 계약으로 만든다.

[이전: C02-P1](02b-c02-p1-provenance-v2.md) | [목차](README.md) | [다음: C02](02-rc3-candidate-qualification.md)

## 1. 왜 별도 C02-P2가 필요한가

C02-P1은 raw provenance, immutable input closure, source-only C11 matchers와 CPU/static hostile-path를 닫았다. 그러나 caller가 정규화해 넘긴 claim을 비교하는 matcher나 checkout에서 실행한 no-action probe는 다음 사실을 만들지 못한다.

- loader/Python보다 먼저 root-owned bootstrap leaf를 같은 held object로 인증했다는 사실
- 인증 뒤 path를 다시 해석하지 않는 secure-exec handoff와 깨끗한 inherited-FD/environment 경계
- crash/restart에도 admission을 보수적으로 닫는 PID1-owned durable ledger
- 실제 non-delegated cgroup, worker pidfd, controller-only release의 live 관측
- GPU/Docker/capture/evidence/receipt/freeze/qualification에 대한 별도 승인

따라서 C02-P2는 source helper를 더 추가하는 작업이 아니라, 실제 native/PID1 구현을 시작하기 전에 authority와 installation boundary를 명시하는 계획 PR이다. 이 문서 또는 그 안의 checklist가 launch, receipt, Gate E, C02 pass 권한을 대체하지 않는다.

## 2. 구현 전 반드시 닫아야 할 native workstream

후속 구현/리뷰는 아래 여섯 항목을 **하나의 연결된 native/PID1 보안 성질**로 제공해야 한다. 하나의 CPU model, preflight JSON, 또는 standalone matcher의 `checked`/`accepted` 결과는 어느 항목도 충족하지 않는다.

1. **Pre-loader trust root.** `/`부터 fixed anchor leaves까지 held `openat2`/no-follow descriptor로 열고, local filesystem, ACL 부재, ownership, exact mode, link count, canonical immutable manifest와 digest를 loader/Python 시작 전에 인증한다.
2. **Same-object secure exec.** 인증한 object를 post-check pathname 재해석 없이 `execveat` 또는 review된 sealed-memfd 전달로 넘긴다. raw `envp={}`, FD scrub, loader-injection 차단, `no_new_privs`, capability-free handoff와 exact FD 31/32 successor ABI를 함께 검증한다.
3. **PID1 durable ledger.** root controller가 held cgroup/lease acquisition을 인증·원자 기록하고, 어느 crash window에서도 active record를 `DRAINING`으로 보수적으로 rehydrate한다. malformed/incomplete ledger와 unknown-after-acquire 상태는 fail closed한다.
4. **Live cgroup/pidfd control.** non-delegated cgroup의 same-object population을 guardian/warden/controller/worker loss 뒤에도 재확인하고, registered terminal worker pidfd와 fresh empty observation 뒤에만 controller가 admission을 release한다.
5. **Cutover fault injection.** anchor authentication, secure-exec, ledger acquisition/commit, cgroup move/observation, controller restart, worker terminal, release의 모든 cutover에서 failure injection을 실행한다. unknown-after-acquire crash도 포함한다.
6. **Operational authorization separation.** GPU, Docker, raw capture, evidence, receipt, candidate freeze, semantic replay, qualification은 native design review와 별도로 명시 승인한다. no-GPU acceptance는 이 승인이나 실제 producer 실행을 뜻하지 않는다.

### 현재 source precursor

`gate-e-root-bundle-authenticator`는 fixed future guardian bundle을 검사한 뒤 곧바로 닫던 기존 CLI 정책을 보존하면서, 같은 held root/ancestor/manifest/bootstrap/core `CLOEXEC` descriptor와 metadata/digest를 caller-owned handle로 돌려주는 `gate_e_root_bundle_held_v1` ABI를 추가했다. `recheck`와 single-owner `close`도 포함한다. 이어 `gate-e-root-bundle-sealed-leaves`는 이 handle을 빌려 bootstrap/core를 before/between/after recheck와 exact digest/length 검증 뒤 no-exec sealed data pair로 복사한다. 두 source API 모두 checkout-built dynamic binary이며 `execveat`, FD 31/32 placement, interpreter/runtime closure, PID1/cgroup/ledger, launch 또는 GPU/Docker authority를 만들지 않는다.

별도 execution-closure sidecar는 parser만으로는 declaration에 머물렀다. 이제
gate-e-execution-closure-held-fds는 caller가 이미 보유한 loader/interpreter/runtime
FD를 canonical role order, linked regular CLOEXEC duplicate, exact length/SHA-256,
pre/post identity, numeric/device-inode alias denial과 묶어 retained result로 만든다.
caller는 bind/recheck/close 중 input/output FD-table ownership도 serialize해야 한다.
이 결과도 sidecar 또는 FD의 root/ACL/filesystem provenance를 인증하지 않고, ELF
closure, same-object execveat, FD 31/32, launch, PID1/cgroup/ledger, GPU/Docker
authority를 만들지 않는다.

## 3. 설계·리뷰 산출물

native implementation 또는 root installation 전에 다음 산출물을 source review와 administrator review에서 각각 독립적으로 확인한다.

- guardian/warden/PID1 state machine, syscall/FD ABI, trust boundary, crash-recovery threat model
- reproducible static-native build, source/binary digest pin, signing/approval ownership, rollback/revocation 정책
- root service/PID1 integration, UID/GID·namespace·capability·signal·environment contract
- durable ledger schema, key/material ownership, fsync/commit ordering, rehydrate and corruption handling
- cgroup-v2 ownership/protocol, pidfd lifecycle, credential-authenticated control transport, controller release predicate
- six workstream을 cover하는 hostile-path/fault-injection matrix와 no-GPU system acceptance record
- administrator-only installation checklist, installed-object inspection record, and an explicit GPU/Docker authorization decision

각 산출물은 mutable checkout output과 독립이어야 한다. source template, static analyzer, native root-bundle authenticator, sealed-leaf snapshot, witness matcher의 결과는 설계 검토 input일 수 있으나 installed guardian의 authority를 대신하지 않는다.

## 4. Administrator provisioning boundary

현재 v3 execution-anchor contract가 요구하는 fixed location은 다음과 같다.

```text
/opt/riley/rc3-gate-e-v1/
  execution-anchor.json
  run_remote_rc3_gate_e_session_v3.py
  rc3_gate_e_private_raw_core_v1.py
/var/lib/riley/rc3-gate-e/lock/
  gate-e-v3.lock
```

`/`부터 final directory까지 root ownership, non-group/world-writable, POSIX ACL 부재가 필요하다. v3 anchor root는 `0755`, lock directory는 `0700`, bootstrap은 `0755`, core/manifest는 `0644`, 세 anchor file은 root-owned single-link regular file, existing lock은 root-owned single-link zero-byte regular file `0600`이어야 한다. approved local filesystem도 검증 대상이다.

이는 **future v3 no-action anchor의 최소 preflight 계약**일 뿐 C02-P2 guardian을 설치하는 지침이 아니다. future FD 31/32 successor, static guardian binary, PID1 service unit, ledger root, cgroup path와 key material은 design review가 exact immutable bundle/ABI로 확정한 뒤에만 administrator가 provision한다. mutable checkout 복사, 빈 directory 생성, verifier path 변경, Docker를 통한 host-root 우회는 설치·launch·capture 권한을 만들지 않는다.

## 5. 완료 순서와 acceptance boundary

1. C02-P2 설계/ABI/threat model과 reproducible build contract를 review한다.
2. administrator가 reviewed immutable root bundle과 PID1/cgroup/ledger boundary를 provision한다.
3. GPU/Docker 없이 native guardian의 installed-object, secure-exec, ledger recovery, cgroup/pidfd, fault-injection acceptance를 수행한다.
4. 별도 authority holder가 GPU/Docker/capture/evidence/freeze/qualification operation을 명시 승인한다.
5. 그 승인 뒤에만 C02의 clean candidate freeze, authenticated actual Gate E producer, GPU raw capture 및 semantic replay를 시작한다.

C02-P2의 **source/design closure**는 이 계획과 후속 reviewed artifacts가 complete한 상태다. C02-P2의 **installation readiness completion**은 administrator-provisioned native guardian의 no-GPU acceptance와 explicit operational-authorization decision이 남긴 검증 가능한 record가 있을 때만 선언할 수 있다. 어느 경우에도 C02 pass 또는 vLLM win을 뜻하지 않는다.

## 6. 현재 host readiness

현재 `server-4096`의 service user는 root-owned anchor/lock paths를 만들거나 검사 가능한 root service context를 갖지 못한다. `/opt/riley/rc3-gate-e-v1`와 `/var/lib/riley/rc3-gate-e/lock`은 아직 없고, non-interactive `sudo`도 허용되지 않는다. GPU와 Docker가 존재하더라도 이는 native guardian의 root provisioning 또는 actual qualification authority가 아니다.

따라서 이 단계에서 안전하게 가능한 작업은 design, source, CPU/static verification, administrator checklist의 준비까지다. root installation 또는 GPU/Docker operation은 위 4단계의 별도 승인 전에는 수행하지 않는다.
