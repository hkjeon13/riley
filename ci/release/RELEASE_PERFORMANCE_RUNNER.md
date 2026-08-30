# Retired legacy runner and source-binding successor

The historical `ci/run_remote_release_performance.sh` body formerly acted as
its own public Bash launcher. It is now a no-action retirement stub: it
contains no GPU lock, Docker, evidence, child-process, or source-loading body.
`BASH_ENV` is evaluated before Bash starts the file, so an in-file `exit` is
not a trust anchor; removing the privileged body means even a shadowed shell
builtin cannot fall through into an old capture path. The legacy runner is retired.
The v3 contract text below remains only for audit and migration; it
is not an executable child body or a future raw-producer template.

`ci/release/run_remote_rc3_gate_e_session_v2.py
--performance-source-contract-probe` is the replacement trust-boundary
foundation. It requires the fixed remote checkout path and the reviewed
`/usr/bin/python3.10 -I -S -E` interpreter, opens the private body with
no-follow/nonblocking component traversal under a held root FD, copies the
bounded bytes into a sealed `memfd`, and reports only source metadata. The
probe establishes a fixed-source snapshot within the existing trusted-checkout
boundary; it does not independently approve the body bytes. It intentionally does **not** open
the GPU lock, invoke Bash/Docker, create evidence, run a performance capture,
or make a qualification decision. It snapshots the retired no-action stub
only to exercise the fixed-root/FD mechanism; a future versioned raw-envelope
producer must introduce a new private core rather than re-enable this body.

## Future immutable execution anchor

The current checkout is user-owned development input, not actual GPU-execution
authority. `verify_rc3_gate_e_execution_anchor_v1.py
--anchor-contract-probe` defines the next prerequisite: a fixed root-owned
and non-group/world-writable external anchor at `/opt/riley/rc3-gate-e-v1`,
plus a root-owned mode-`0700` lock directory at
`/var/lib/riley/rc3-gate-e/lock`. It checks a canonical root-owned manifest
and FD-hashed bootstrap/core bytes but executes neither. The anchor is not
currently provisioned on `server-4096`, so its probe must fail closed until a
system administrator installs a reviewed v3 bundle. Its `checked` output is a
mutable-checkout installation preflight, not launch/receipt/qualification
authority; the future root-installed bootstrap must repeat host-context and
ACL checks. See
`ci/release/RC3_GATE_E_EXECUTION_ANCHOR.md`.

`ci/release/rc3_gate_e_private_raw_core_v1.py` now provides only the v3
private-core **no-action protocol template**. It rejects direct checkout
execution and accepts only a future bootstrap's sealed FD 8 core, sealed
canonical FD 9 configuration, and private FD 10 Unix `SOCK_SEQPACKET` channel
under `/usr/bin/python3.10 -I -S -E -B`. The nonce/config-digest/credential-bound exchange
returns `COMPLETE` with explicit all-false guarantees and has no GPU lock,
Docker, evidence, semantic replay, receipt, or qualification capability. Its
source template and CPU-only memfd/socketpair test are not an installed anchor
or a path to invoke on `server-4096`; the root-installed bootstrap remains a
separate prerequisite.

## Trusted inputs

Select these values from reviewed evidence before starting the measurement.
Do not derive an expected digest from the file that this run happens to find:

- the frozen 40-character source revision and its canonical uncompressed
  `git archive` SHA-256;
- the immutable optimizer image ID in `sha256:<64 lowercase hex>` form;
- the `final/riley-profile` ELF selected by reproducible-build evidence
  and its externally reviewed SHA-256;
- the pinned SmolLM2 model directory and canonical model-tree SHA-256;
- the passed optimizer correctness report and its externally reviewed
  SHA-256. Its source archive, optimizer image, GPU, model, semantic class, and
  exact command-batch tests must match the other inputs.

The runner's `model_tree_sha256` is not an independent operator assertion. It
must exactly equal `model.manifest_sha256` in that submitted optimizer
correctness report, and the raw-archive replay preserves the full runner
manifest so the final RC gate can check the same binding again.

The model tree must contain the reviewed SmolLM2 revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, including weights SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
and tokenizer SHA-256
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

## Archived v3 capture specification

The arguments below describe the historical v3 contract. They are not a public
command and no longer have an executable Bash entrypoint.
The checkout must already be at the selected revision and completely clean.
The output parent must exist, but the output directory itself must not exist
and must be outside the checkout.

For the current CPU-only source-bound probe, use the reviewed Python
interpreter and this exact fixed `server-4096` checkout path:

| Tool | Path | SHA-256 |
|---|---|---|
| mawk | `/usr/bin/mawk` | `dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e` |
| basename | `/usr/bin/basename` | `3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22` |
| bash | `/usr/bin/bash` | `59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4` |
| cat | `/bin/cat` | `210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba` |
| chmod | `/usr/bin/chmod` | `e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8` |
| cmp | `/usr/bin/cmp` | `b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791` |
| cp | `/usr/bin/cp` | `8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2` |
| df | `/usr/bin/df` | `b06fe81669b9383abed94bb5cae1cb7a63c6e02801b1b7dd1c08d7d2c8987e86` |
| dirname | `/usr/bin/dirname` | `674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94` |
| docker | `/usr/bin/docker` | `29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6` |
| env | `/usr/bin/env` | `85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0` |
| find | `/usr/bin/find` | `791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1` |
| git | `/usr/bin/git` | `587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a` |
| grep | `/usr/bin/grep` | `73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96` |
| head | `/usr/bin/head` | `9e457645cdcfd74ee0a9688b25b7b017d8d393233a0c0bdf3bef3c57a1238ce2` |
| hostname | `/usr/bin/hostname` | `d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e` |
| install | `/usr/bin/install` | `519a00d199d07da6028ec5a9800d92c562934582a2ea1793b2cbc378a85c1439` |
| mkdir | `/usr/bin/mkdir` | `bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b` |
| nvidia-smi | `/usr/bin/nvidia-smi` | `22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be` |
| python3 | `/usr/bin/python3.10` | `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86` |
| sed | `/usr/bin/sed` | `42e2ce00721556ff9d371778fc36adcbb7c1697f65c3f996c6c9b28206dba565` |
| sha256sum | `/usr/bin/sha256sum` | `7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3` |
| sleep | `/usr/bin/sleep` | `b9aec374a2b2a175a182f615291ad408820b7fb8c663a184e37fa3492d3f8eff` |
| sort | `/usr/bin/sort` | `0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78` |
| stat | `/usr/bin/stat` | `9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3` |
| tail | `/usr/bin/tail` | `d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0` |
| tar | `/usr/bin/tar` | `fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2` |
| timedatectl | `/usr/bin/timedatectl` | `a1d1298afc514e7143d1a7a4c0039ce1256871faf33fe356fd9063dd283df5d9` |
| tr | `/usr/bin/tr` | `24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c` |
| uname | `/usr/bin/uname` | `37df0311d0e24169abfd166bc6018d40b87306f7ff64d9eec256c8331ac26347` |
| wc | `/usr/bin/wc` | `504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8` |

The capture holds the same host-wide lock as the soak runner:
`/var/tmp/riley-server-4096-gpu-evidence.lock`. A pinned Python supervisor
opens an existing or new mode-`0600`, regular, single-link, owner-controlled
inode with `O_NOFOLLOW|O_APPEND|O_NONBLOCK|O_CLOEXEC`, then takes a
non-blocking exclusive `flock` before any GPU work. The child Bash authenticates
its direct Python parent, the parent executable and lock inode, the parent's
`CLOEXEC`/`NONBLOCK` fdinfo flags, and the kernel `FLOCK ADVISORY WRITE`
receipt, then closes its own temporary copy of the descriptor. The supervisor
sets `PR_SET_PDEATHSIG` and forwards termination. A normal exit preserves the
ordinary receipt lifecycle without a supervisor Docker cleanup pass. After an
abnormal exit or signal, the supervisor retains the lock until every active
(`created`, `running`, `paused`, `restarting`, or `removing`) container bearing
its unguessable per-run label has been removed; `exited` or `dead` diagnostic
evidence is not swept up. A hidden shell marker
alone therefore cannot enter the acceptance path, and no Bash, Docker,
monitor, or candidate descendant inherits the lock.

```sh
/usr/bin/python3.10 -I -S -E \
  /home/psyche/rustinfer-vllm-roadmap-serial/ci/release/run_remote_rc3_gate_e_session_v2.py \
  --performance-source-contract-probe
```

The probe output is a `source-bound-no-action` JSON document; it is neither a
producer receipt nor a performance result. The historical v3 capture mechanics
below are archived reference material only: there is no current callable child
body, and a future producer must introduce a new versioned private core rather
than reuse this specification. Before a future independent run, the standard
preflight would capture and validate
the actual host kernel, CPU topology, RAM, GPU identity and capacity, driver,
clock state, power limit, idle VRAM, and temperature. The reviewed clock/power
receipt is exactly `power_limit_w=450.00`, `graphics_clock_mhz=[N/A]`, and
`memory_clock_mhz=[N/A]`. Only a temperature-only
failure is retried, for at most 41 attempts spaced 30 seconds apart. Any other
drift fails immediately.

`[N/A]` is a pinned observation for this one server lane: driver 580 does not
report fixed application clocks for this card through the reviewed query. It
does not mean "any clock is acceptable" and makes no fixed-clock claim. Every
preflight and every pre-start/running/post-exit monitor sample must contain
the exact `[N/A]` strings; any numeric value or other spelling is lane drift.

Each of the five runs gets a newly created container and anonymous workspace
volume. Docker's resolved `Path`, `Args`, disabled healthcheck, PID/IPC/UTS/
user/cgroup namespaces, runtime, device inventory, GPU request, CPU and memory
limits, mounts, and state must all equal the closed receipt contract. The
container has `network=none`, one exact GPU UUID request, a read-only root,
read-only source/profile/report/model mounts, no proxy, and no restart policy.
Every bind receipt must report Docker's reviewed `Mode=""` and
`Propagation="rprivate"` values for both read-only and writable binds.
It drops every capability, requires the exact Docker inspect value
`SecurityOpt=["no-new-privileges:true"]`, limits pids to 512, and requires the
exact `/tmp` tmpfs value `rw,nosuid,nodev,noexec,size=2147483648` (2 GiB).
On server-4096 Docker 28.3.2, the unset CLI `CapAdd` ListOpts canonicalizes in
daemon inspection as exact `CapAdd=null`; `[]`, a missing field, or any
non-null value is rejected, while `CapDrop` must be exact `["ALL"]`.
`Config.Labels` must be exactly the immutable image's inspected label map plus the one
`org.riley.release-performance-supervisor=<token>` entry. The normalized
UUID device request must use `Driver=""`, `Count=0`, and `Options={}`; omitted,
widened, extra-label, or alternate spellings are rejected.

Input snapshots are installed before the run. Model directories are mode
`0555`, model/report/source files are mode `0444`, and the profile executable
is mode `0555`. Each per-run evidence directory is mode `0733`, which allows
the capability-less container root to traverse and create its one result.
The container process that owns the raw result changes it to mode `0444`;
the host never attempts to chmod that root-owned file and instead installs a
separate mode-`0444` receipt snapshot. Source/profile/report/model hashes and
permissions are rechecked after accepted preflight, immediately before start,
and immediately after exit.

Each run also records canonical `gpu-monitor.csv` samples in exact
`pre_start,running+,post_exit` order. The boundary samples require zero CUDA
PIDs and at most 256 MiB used. During the run every observed CUDA PID must map
through `/proc/<pid>/cgroup` to that run's full container ID. A foreign PID,
power/clock drift, or failure to observe the candidate CUDA process kills or
rejects the run immediately. No later container can overwrite an earlier raw
result. Inside the container, the runner independently observes and pins
Ubuntu `PRETTY_NAME=Ubuntu 22.04.5 LTS`, Linux
`6.8.0-138-generic`, x86-64 CPU/RAM facts, RTX 4090/driver facts, CUDA runtime
12.8.1, nvcc 12.8.93, and cuBLAS 12.8.4.1. It then executes exactly one
candidate workload with 5 warmups and 30 measured iterations at concurrency
1, prompt length 128, output length 32, and greedy sampling.

The supervisor token and pair index derive a fresh 64-hex `capture_id`. That
ID is embedded in every monitor row, the container environment, and the raw
candidate `run_id`. After exit the validator emits one canonical
`riley.release-performance-execution-receipt.v1` per pair. It records the
capture, full container ID, run ID, candidate `recorded_at_utc`, exact SHA-256
of preflight/candidate/GPU-monitor/before-inspect/after-inspect bytes, and exact
Docker `Created`, `StartedAt`, `FinishedAt`, exit code, and OOM flag. Acceptance
requires `Created <= StartedAt <= candidate recorded_at <= FinishedAt`, no
overlap between sequential pair timelines, exit code 0, and `OOMKilled=false`.
The five receipt objects are copied exactly into
`riley.release-performance-runner-manifest.v3`; offline replay rejects a
candidate, monitor, or inspect splice even when the replacement uses the same
source revision and the outer checksum index is recomputed.

The five native runs must also derive the reviewed PR15 canonical request
identity SHA-256
`e6a99a749c41a8227574c96a1d23f8b7d877d6e75b0df4d99154db1b1921a2e6`.
Matching aggregate metrics cannot authorize different prompt or generated
token identities.

After all five exits, the CPU-only validator replays the raw native-profile
schema and metric derivation. It also checks all five before/after Docker
inspect pairs for the immutable image, exact `Entrypoint`/`Cmd` and full
image-base-plus-override environment, exact GPU request, network isolation,
read-only root, reviewed mounts and provenance environment, five distinct
container IDs and workspace volume names, exit code zero, `OOMKilled=false`,
empty error, and restart count zero. Success therefore
requires all five complete raw documents; saved receipts alone are not a
pass.

## Output inventory

The create-only output root contains:

```text
source.tar
inputs/
  docker-config/
  model/
  optimization-correctness-report.json
  riley-profile
gpu.csv
optimizer-image-inspect-before.json
optimizer-image-inspect-after.json
runner-manifest.json
SHA256SUMS
run-{1..5}/
  preflight.txt
  container-inspect-before.json
  container-inspect-after.json
  gpu-monitor.csv
  candidate.json
  execution-receipt.json
run-evidence/run-{1..5}/
  candidate-{1..5}.json
preflight/run-{1..5}/
  attempt-*.stdout
  attempt-*.stderr
  accepted-validation.json
containers/run-{1..5}/
  stdout.log
  stderr.log
  candidate.sha256
raw-validation.json
receipt-replay.json
```

The runner never reuses or overwrites an output root. A failed root is kept as
diagnostic evidence and is not eligible for packaging.

Use the five mode-0444 files under `run-{1..5}/candidate.json` as the
packager's `--run` inputs and pass the output root as
`--runner-receipt-root`. The canonical v3 archive contains exactly
`runner-manifest.json`, `gpu.csv`, both image inspections, the six receipts
for each run (including `execution-receipt.json`), and `SHA256SUMS`, sorted and
encoded as fixed-metadata USTAR.
The checker replays those receipts before returning the five candidate
payloads to any caller, including the final release-candidate gate.
`source.tar` as `--source-archive`, `inputs/riley-profile` as
`--profile-binary`, the model snapshot for weights/tokenizer, and the
snapshotted correctness report. Continue with the command in
`benchmarks/release/README.md`; packaging also requires the separately
reviewed release binary and immutable release-runtime image.

## CPU-only contract tests

These commands do not start Docker or CUDA:

```sh
python3 -m unittest ci/release/test_release_performance_runner.py -v
bash -n ci/run_remote_release_performance.sh
bash -n ci/release/run_release_performance_once.sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.10 -B -m unittest \
  ci/release/test_run_remote_rc3_gate_e_session_v2.py
```

The mode test above needs neither Docker nor CUDA. A reviewer can additionally
probe the capability-less bind semantics on `server-4096` without attaching a
GPU or loading a model. Replace the image placeholder with the already
reviewed optimizer image ID; this command deliberately has no `--gpus` flag:

```sh
ssh server-4096 /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC HOME=/home/psyche \
  /usr/bin/bash -s -- sha256:<optimizer-image-digest> <<'PERMISSION_PROBE'
set -euo pipefail
image=$1
probe=$(/usr/bin/mktemp -d /var/tmp/riley-permission-probe.XXXXXX)
cleanup() {
  /usr/bin/chmod -R u+w -- "${probe}" 2>/dev/null || true
  /usr/bin/find "${probe}" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT
/usr/bin/mkdir -m 0700 "${probe}/model"
/usr/bin/install -m 0444 /etc/hostname "${probe}/model/sentinel"
/usr/bin/chmod 0555 "${probe}/model"
/usr/bin/mkdir -m 0733 "${probe}/evidence"
/usr/bin/docker run --rm --network none --read-only --no-healthcheck \
  --cap-drop ALL --security-opt no-new-privileges:true --user 0:0 \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=2147483648 --entrypoint /bin/bash \
  --mount "type=bind,source=${probe}/model,destination=/model,readonly" \
  --mount "type=bind,source=${probe}/evidence,destination=/evidence" \
  "${image}" -ceu \
  'test -r /model/sentinel; printf passed > /evidence/result; chmod 0444 /evidence/result'
test "$(/usr/bin/stat -c '%a' -- "${probe}/evidence/result")" = 444
PERMISSION_PROBE
```

This is intentionally a single reviewed environment lane, not a portable
benchmark launcher. Kernel, image, driver, CUDA, OS patch release, or hardware
drift must produce a new reviewed lane rather than a relaxed comparison.
