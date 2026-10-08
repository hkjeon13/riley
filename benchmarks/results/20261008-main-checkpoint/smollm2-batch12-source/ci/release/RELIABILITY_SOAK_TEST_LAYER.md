# Python-free reliability-soak test layer

`ReliabilitySoak.Dockerfile` is an evidence-only derivative of an already
reviewed release image. It does not build or replace `riley`. The caller
must pass the release image's local immutable `sha256:...` ID to the launcher;
there is no default or registry fallback. The layer adds only the direct observation/client packages needed by
`ci/run_release_soak.sh`: Bash, coreutils, curl, findutils, gawk, grep, jq,
procps, and util-linux. `nvidia-smi` comes from the host's NVIDIA container
runtime. Python-family runtimes and artifacts, compilers, and build tools are
rejected while the layer is built.

Before `apt-get` runs, the source-bound Dockerfile asks the image's loader to
resolve `/opt/riley/bin/riley` and writes a canonical, bytewise-sorted
TSV of dependency name, resolved absolute path, canonical regular-file target,
and target SHA-256. Exactly one build-time unresolved dependency is allowed:
`libcuda.so.1<TAB>NOT_FOUND<TAB>-<TAB>-`, because NVIDIA injects the driver at
container start. Any other unresolved SONAME, or a missing/duplicate libcuda
row, fails the build and replay. The row must have the same unresolved state
before and after package installation. Loader ASLR addresses are excluded.
Installation uses `--no-upgrade`; after installation the same closure is
recomputed and the build fails unless it is byte-for-byte identical. The pre-install closure is retained
read-only at `/opt/riley-soak/release-runtime-closure.tsv` for audit. This
receipt does not authorize itself: the remote launcher builds the exact
Dockerfile bytes extracted from the independently hashed source archive, and
the static test-layer verifier pins the capture-before/install/capture-after/
compare instruction order and target-byte hashing operation.

BuildKit does not portably support a raw local image ID in `FROM`. The remote
launcher therefore creates a deterministic local base tag whose name contains
the full candidate revision and full release image digest. An existing tag is
accepted only when it resolves to that exact ID. The launcher checks the tag's
`.Id` before and after the build, then verifies that the release image's rootfs
layer list is an exact prefix of the derivative. The tag is only a BuildKit
transport reference: labels, runtime bindings, and evidence retain the
original immutable image ID. The base tag, derived image, and stopped target
container remain present for replay.

## Authorized remote run

Run only from a completely clean checkout of the exact candidate revision on
`server-4096`. All SHA-256 arguments below are independent, reviewed inputs;
do not derive them from the files during the candidate run.

The launcher fail-closes its acceptance-critical absolute host-tool inventory.
Every reviewed path must be a root-owned, non-group/world-writable executable
regular file and not a symlink, with its exact server-4096 SHA-256. This
includes `/usr/bin/stat` and the regular `/usr/bin/mawk`; the symlink
`/usr/bin/awk` is forbidden. The bootstrap `/usr/bin/env` and `/bin/cat` paths
are also members of the closed inventory. Bash, `cat`, `env`, and the Python
lock supervisor are invoked directly by their reviewed absolute paths. The
inventory loop invokes its `stat`, `sha256sum`, and `mawk` validators through
readonly absolute `*_BIN` values; subsequent ordinary calls use readonly
absolute-path wrappers. `find -exec` likewise uses the reviewed absolute
`CHMOD_BIN` value rather than performing a nested `PATH` lookup. The static
verifier requires this executable dispatch set, its absolute-path constants,
the readonly wrapper set, and the digest inventory to be exactly equal. The
closed server-4096 inventory is:

| Command | Absolute path | SHA-256 |
| --- | --- | --- |
| bash | `/usr/bin/bash` | `59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4` |
| basename | `/usr/bin/basename` | `3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22` |
| cat | `/bin/cat` | `210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba` |
| chmod | `/usr/bin/chmod` | `e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8` |
| cmp | `/usr/bin/cmp` | `b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791` |
| cp | `/usr/bin/cp` | `8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2` |
| date | `/usr/bin/date` | `08b85d43067bcd15edb0882d5372a8b5635e211f76b62ccc4d575f2ed4920e18` |
| dirname | `/usr/bin/dirname` | `674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94` |
| docker | `/usr/bin/docker` | `29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6` |
| env | `/usr/bin/env` | `85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0` |
| find | `/usr/bin/find` | `791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1` |
| git | `/usr/bin/git` | `587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a` |
| grep | `/usr/bin/grep` | `73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96` |
| hostname | `/usr/bin/hostname` | `d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e` |
| jq | `/usr/bin/jq` | `858a84f22b39317f13a57b4b91e535925c1b4f819d9bb2864361df4ad6acb00f` |
| mawk | `/usr/bin/mawk` | `dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e` |
| mkdir | `/usr/bin/mkdir` | `bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b` |
| nvidia-smi | `/usr/bin/nvidia-smi` | `22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be` |
| od | `/usr/bin/od` | `8831c6be1e0b0a7c8c01e2f939b03d8d1d144e238c6b8e0a5d9d1a8c367ac910` |
| python3 | `/usr/bin/python3.10` | `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86` |
| sha256sum | `/usr/bin/sha256sum` | `7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3` |
| sort | `/usr/bin/sort` | `0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78` |
| stat | `/usr/bin/stat` | `9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3` |
| tail | `/usr/bin/tail` | `d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0` |
| tar | `/usr/bin/tar` | `fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2` |
| tee | `/usr/bin/tee` | `eb219ccfbdad53064135a4101d4f56f0d9e5f7f1cd20c032b29e3604264cf79b` |
| tr | `/usr/bin/tr` | `24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c` |
| wc | `/usr/bin/wc` | `504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8` |

Every reviewed Python control-plane call uses `-I -S`; neither user/system
site initialization nor `sitecustomize`/`.pth` files execute before the GPU
lock, parent/FD proof, or no-follow input snapshots. Python remains absent from
the derived production runtime layer.

```sh
/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC HOME=/home/psyche \
  /usr/bin/bash ci/run_remote_release_soak.sh \
  --release-image-id sha256:<reviewed-release-image-id> \
  --source-revision <full-40-character-candidate-revision> \
  --source-archive /artifacts/source.tar \
  --expected-source-archive-sha256 <reviewed-sha256> \
  --release-binary /artifacts/riley \
  --expected-release-binary-sha256 <reviewed-sha256> \
  --model-dir /models/SmolLM2-135M-Instruct \
  --expected-model-tree-sha256 <reviewed-sha256> \
  --materialized-manifest /artifacts/reliability-soak-v1.materialized.json \
  --expected-manifest-sha256 <reviewed-sha256> \
  --correctness-golden /artifacts/python-free-e2e-golden.json \
  --expected-correctness-golden-sha256 <independently-reviewed-sha256> \
  --native-correctness-report /artifacts/native-e0-correctness-report.json \
  --expected-native-correctness-report-sha256 <independently-reviewed-sha256> \
  --test-image-tag riley-soak-test:<unique-candidate-run> \
  --output-dir /append-only-evidence/pr16-soak-<unique-run>
```

The source archive must be the reviewed, uncompressed `git archive` tar. The
launcher first copies it create-only into the new mode-`0700` output, verifies
that snapshot's byte hash and embedded commit, then extracts only the
Dockerfile, soak driver, and canonical manifest into a mode-fixed minimal
build context. The manifest, golden, native report, and model are likewise
snapshotted before content validation; immediate pre-use and post-run hashes
detect accidental replacement/restoration races. It never builds the test
layer from the live checkout. The
output directory and test image tag must not already exist. The
materialized manifest may differ from the checked-in template only in its two
golden digest fields. Before the multi-hour run, the launcher independently
hashes the E2E golden and native correctness report and requires:

- the manifest generated-text hash to equal the E2E golden's exact completion
  hash;
- the manifest provenance hash and the golden's correctness-report hash to
  equal the submitted native report's reviewed byte hash;
- the golden and passing native report to bind the candidate revision, model
  ID/revision, config, weights, tokenizer aggregate, and tokenizer JSON, while
  requiring a lowercase SHA-256 for the report's separate development-only
  `riley-native calibrate` executable (it is not the production server
  binary); and
- the manifest's greedy request to bind the golden prompt and token count.

This separate golden/report interface is intentional: a later trusted
materializer can supply the reviewed files and hashes without changing the
test layer, while a candidate manifest cannot authorize its own output.

## Isolation and evidence ownership

The launcher holds the designated host's shared GPU-evidence `flock` for its
entire lifetime through one no-follow, close-on-exec-opened descriptor whose
inode type, owner, link count, mode, and pathname identity are checked. A
minimal Python control-plane supervisor alone retains that descriptor; the
Bash launcher proves its direct parent, parent executable, descriptor inode,
and kernel `FLOCK` record through `/proc`, while `PR_SET_PDEATHSIG` terminates
the launcher if the lock-owning supervisor dies. The descriptor is never
inherited by the launcher or its external commands. The launcher
checks that the compute-app inventory is empty both during
preflight and immediately before `docker start`. The target runs as the
production identity `65532:65532`, with a read-only
root filesystem, a private 64 MiB `/tmp` tmpfs, all Linux capabilities
dropped, `no-new-privileges`, and only the exact GPU UUID. The model and
materialized manifest mounts are read-only. All three binds must retain empty
Docker `Mode` and `Propagation=rprivate`; propagated host submounts are not valid
soak evidence. Runtime networking is `none`; the driver and server communicate
only over `127.0.0.1` inside the container.

The release, derivative, and container receipts preserve exact environment
inheritance. `PATH`, `LD_LIBRARY_PATH`, `NVIDIA_VISIBLE_DEVICES`, and
`NVIDIA_DRIVER_CAPABILITIES` are accepted only at their reviewed production
values. `HOME`, `CURL_HOME`, `XDG_CONFIG_HOME`, every other `LD_*` variable, and
the existing shell/loader injection controls must be absent.

The container deliberately uses `--pid host`. NVML/`nvidia-smi` reports host
PIDs, so the driver, its child `riley` server, `/proc`, and `nvidia-smi`
must share that PID namespace for exact process/VRAM attribution. The launcher
records and rechecks `PidMode=host`; this is why the run is restricted to the
designated, idle evidence host. PID sharing does not grant host networking or
additional capabilities.

The original model is copied into `model-snapshot/` before validation, every
file/directory mode is fixed to `0444`/`0555`, and only that snapshot is
mounted read-only. Its canonical tree hash is checked against the reviewed
digest at initial validation, immediately before start, and after the run.
The top-level output directory stays mode `0700` and contains
host-written build/container logs and trusted input copies. Only
`container-evidence/` is mode `0777` and mounted writable, so the
fixed production UID can create the driver's mode-`0700` raw run. The stopped
target container and original bind-mounted evidence are never removed. After
exit, `docker cp` creates the host-owned, readable
`container-evidence-export/` copy used for checking and packaging. The
container name is retained in `container-name.txt` for audit or recovery.

The designated Unix account is a trusted-host boundary: another malicious
process running concurrently as that same UID can control the account's files
and Docker socket. The clean `/usr/bin/env -i` entry, private Docker config,
safe Git configuration, no-follow lock, create-only snapshots, and repeated
hashes close accidental and lower-privilege interference; they do not claim to
isolate mutually hostile processes that already share the authorized UID.

`runtime-receipts/` is a closed seven-file directory containing exactly
`host-gpu.csv`, `launcher-receipt.json`, `release-runtime-closure.tsv`,
`release-image-inspect.json`, `test-layer-image-inspect.json`,
`container-inspect-pre.json`, and `container-inspect-post.json`. The launcher
copies the retained closure from the actual created container before start,
requires canonical sorted absolute-path/target-digest rows with one loader and
exactly one `libcuda.so.1/NOT_FOUND/-/-` row, and hashes it into the launcher receipt. This
build receipt records the non-NVIDIA build namespace; the existing runtime
container inspection and native/Python-free validation separately prove the
NVIDIA-injected `libcuda.so.1` path and bytes after container start. The
launcher receipt cross-binds the designated
host/GPU, source/archive/binary/model/manifest/golden/native hashes, immutable
image IDs, persistent container ID/name/exit status, and exact byte SHA-256s of
the exported `run.json`, `events.jsonl`, and runtime closure. The v3 receipt is written only after
those exported files and the post-run container state have been checked, so a
receipt set cannot be reused with another execution's stream. The run's strict
UTC start stamp is repeated exactly in `run_id` and must fall within Docker's
validated start/finish lifecycle; the container-name stamp is strictly parsed
and may precede Docker `Created` by at most five minutes. `launcher-SHA256SUMS`
is written after the post-run checks; `completed` is the final filesystem
write.

The exported event stream carries closed per-request transport proof after the
temporary request/response files are removed. The checker reconstructs the
compact key-sorted request bytes from the manifest profile and exact `stream`
action before accepting `request_body_sha256`, including jq 1.6's integer
spelling for integral JSON numbers such as `0.0`. Cancel is non-streaming curl
timeout 28 with an empty response; disconnect is streaming and closes after
exactly 1,024 response bytes with curl write error 23 and byte-limiter status
zero. Other actions have exact non-streaming exit/status contracts. Scenario
start/end intervals must be non-overlapping and follow manifest order.

Run the Python standard-library trust tooling outside the production/test
image against that exported copy:

```sh
python3 benchmarks/scripts/check_reliability_soak.py \
  --manifest "$OUTPUT/materialized-reliability-soak-v1.json" \
  --run-directory "$OUTPUT/container-evidence-export/run" \
  --correctness-golden "$OUTPUT/python-free-e2e-correctness-golden.json" \
  --native-correctness-report "$OUTPUT/native-correctness-report.json" \
  --runtime-receipts-directory "$OUTPUT/runtime-receipts" \
  --report "$OUTPUT/reliability-soak-report.json"

python3 benchmarks/scripts/package_reliability_soak_evidence.py \
  --manifest "$OUTPUT/materialized-reliability-soak-v1.json" \
  --run-directory "$OUTPUT/container-evidence-export/run" \
  --correctness-golden "$OUTPUT/python-free-e2e-correctness-golden.json" \
  --native-correctness-report "$OUTPUT/native-correctness-report.json" \
  --runtime-receipts-directory "$OUTPUT/runtime-receipts" \
  --output "$OUTPUT/reliability-soak-raw-evidence.tar"
```

Do not run the Docker build or soak locally. Local validation is limited to:

```sh
bash -n ci/run_remote_release_soak.sh ci/run_release_soak.sh
python3 ci/release/verify_soak_test_layer.py
python3 -m unittest discover -s ci/release -p 'test_soak_test_layer.py' -v
```
