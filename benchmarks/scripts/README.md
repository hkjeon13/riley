# Baseline scripts

이 디렉터리의 도구는 benchmark 계약을 검증하고 target host의 비교 가능성을 확인한다. Python reference 또는 vLLM 환경은 각 lane의 lock으로 실행하며 production binary의 dependency graph에 포함하지 않는다.

## 정적 계약 검사

```bash
python3 benchmarks/scripts/validate_contract.py
```

이 검사는 matrix, prompt corpus, lane manifest, JSON schema의 구조와 cross-file invariant를 확인한다. 모델을 다운로드하거나 GPU를 요구하지 않는다.

## GPU preflight

clean checkout의 target host에서 실행한다.

```bash
mkdir -p /var/tmp/riley-preflight
RILEY_PREFLIGHT_OUTPUT_ROOT=/var/tmp/riley-preflight \
  benchmarks/scripts/preflight.sh \
    > /var/tmp/riley-preflight/preflight.stdout.txt \
    2> /var/tmp/riley-preflight/preflight.stderr.txt
```

스크립트는 상태를 바꾸지 않는다. GPU 종류·개수·compute capability, idle memory, 온도, compute process와 clean Git revision을 확인하고 비교에 필요한 snapshot을 출력한다. 실패한 run은 측정하지 않는다.
preflight 출력을 checkout 안에 redirect하면 검사가 시작되기 전에 Git tree가 dirty가
될 수 있으므로 artifact는 반드시 repository 밖에 둔다.
현재 RTX 4090 profile은
`host_profiles/rtx4090-ubuntu22-driver580-host-v3.env`이며 profile ID/version과
관측 CPU/RAM/driver/idle-GPU 값을 함께 기록한다. RAM은
67,185,594,368 B의 ±16 MiB 범위, NVIDIA `580.` driver branch, idle GPU ≤512 MiB,
start temperature ≤48°C를 확인한다. 이 값은 profile에 명시된 허용 범위이며,
각 campaign의 실제 관측값도 receipt에서 다시 확인해야 한다.

## Reference fixture

고정 checkpoint를 별도 단계에서 cache하고 checksum을 확인한 뒤 reference lane을 offline으로 실행한다.

```bash
UV_BIN=/absolute/path/to/pinned/uv
export UV_PROJECT_ENVIRONMENT=/var/tmp/riley-project-envs/reference-fixture-001
test ! -e "$UV_PROJECT_ENVIRONMENT"
UV_PYTHON=3.13.15 UV_PYTHON_DOWNLOADS=never \
  "$UV_BIN" sync --frozen --offline --project tools/python/reference
"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference \
  riley-reference generate \
  --prompts benchmarks/prompts.jsonl \
  --repo-root . \
  --output /var/tmp/riley-reference/<fixture-id>.json
```

checkout 밖에서 생성·검증한 뒤 SHA-256과 diff를 검토한 artifact만 version-control
workflow로 `benchmarks/reference/`에 반입한다. 정확한 CLI와 model cache 준비
명령은 `tools/python/reference/README.md`가 권위 있는 문서다.

## N03b/N06-A D128 반복 증적

`n01_repeat_control.py`는 production host의 I/O 압력이 남아 있는 상태에서도
반복을 보존하는 outer-process controller다. 모든 timed attempt의 stdout/stderr,
성공·실패·timeout, CPU/I/O/memory PSI pre/post snapshot을 남긴다. PSI는
quiet-window 선택이나 결과 제외 조건이 아니다. child process에는 controller가 parent shell을 바꾸지 않고
`N01_REPEAT_CONTROL_PHASE=warmup|timed`와 one-based
`N01_REPEAT_CONTROL_INDEX`를 넣는다. N06-A driver는 이 index만으로 timed lane
order를 결정한다.

```bash
OUT=/var/tmp/riley-n03b-n06a/<unique-run-id>
python3 benchmarks/scripts/n01_repeat_control.py \
  --output-dir "$OUT" \
  --working-directory /absolute/path/to/riley \
  --warmups 1 --repeats 6 --timeout-seconds 1800 \
  --n06a-timeout-cleanup-artifact-root "$OUT/paired-driver" \
  --n06a-timeout-cleanup-docker-launcher /usr/bin/docker \
  -- python3 /absolute/path/to/n06a_paired_serving_driver.py \
       --artifact-root "$OUT/paired-driver" \
       <required-N06-A-options>
```

`--n06a-timeout-cleanup-artifact-root` and
`--n06a-timeout-cleanup-docker-launcher` are an all-or-nothing pair. They make
N01 retain a sidecar for the exact owned vLLM container cleanup after a timeout
or interruption; the corresponding `failed-cleanup` attempt remains visible as
an incomplete failure. If that cleanup is unproven, N01 fail-stops: it launches
no later warmup or timed child and records each remaining planned index as an
exact `not-started-after-failed-cleanup` placeholder with no fabricated
logs, PSI, or cleanup sidecar. Do not place `ionice` or `nice` around this timed N01
command: Docker's daemon-created vLLM process would not inherit it, so it is
not a common AB/BA condition. A low-priority wrapper may be used only for host
preparation outside the timed command. The optional
`--n06a-timeout-cleanup-command-timeout-seconds` bounds each cleanup command;
its default and maximum are 30 seconds.

`n03b_n06a_d128_repeat_summary.py`는 새 artifact를 만들지 않고 immutable
receipt와 log를 읽어 median, sample standard deviation, deterministic 95%
bootstrap CI를 계산한다. pooled effect와 Riley-first/vLLM-first order-stratified
effect를 모두 내보낸다. 실패한 attempt는 성능 표본에는 들어가지 않지만 output의
`pair_completion`과 `timed_run_pressure_covariates`에는 항상 남는다. 하나라도
planned timed pair가 실패하면 `promotion_status`는 `incomplete`이고 `0/N`을
포함한 failure reason을 남긴다. 결과 root는 checkout과 기존
`benchmarks/results/20260912-serving-optimization/` 밖의 새 경로여야 한다.
shared host의 CPU를 과도하게 쓰지 않도록 reader는 100,000 resamples 및 metric당
250,000 bootstrap draw를 넘는 receipt를 거부한다.

N06-A는 각 Riley/vLLM lane에 대해 `<lane>.lane-psi.json`을 create-only로 남긴다.
`pre`는 sampler/server launch 직전, `post`는 owned process/container cleanup과
sampler 종료 뒤 및 다음 lane의 GPU idle census 전에 CPU/I/O/memory
`/proc/pressure/*`를 관찰한 값이다. artifact의 path/SHA-256은 lane provenance에
들어가고, 그 provenance는 기존 marker hash로 binding된다. 높은 PSI, unavailable,
malformed 관찰은 모두 보존하며 lane marker, pair eligibility, 성능 표본 선택,
weighting, 보정, promotion에는 사용하지 않는다. Summary의
`lane_pressure_covariates`와 `order_by_lane_pressure_sensitivity`는 lane별 pre/post
covariate와 AB/BA order·first/second position별 descriptive pressure view만
제공한다. 이 view에는 PSI bootstrap, pressure-adjusted effect, correlation 또는
인과 해석이 없다.

```bash
python3 benchmarks/scripts/n03b_n06a_d128_repeat_summary.py \
  --operator-receipt /var/tmp/riley-n03b-operator/<id>/n01-repeat-control-receipt.json \
  --serving-receipt "$OUT"/n01-repeat-control-receipt.json \
  > /var/tmp/riley-n03b-n06a/<unique-run-id>-summary.json
```

`--operator-receipt`는 현재 V2 prepared paged-decode control의 exact
`riley.cuda.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2`
identity를 N03a reader로 검증한다. 이것은 operator 결과이며 full-model 또는
vLLM 결과가 아니다.

`--serving-receipt`의 성공 outer command는 stdout에 Riley와 vLLM 각각 한 개의
`riley-n06a-d128-serving` key=value marker를 남겨야 한다. 양쪽 marker는 model
revision, workload, concurrency, `batch_token_budget`, `max_model_len`,
fixed vLLM prefix-cache/image/memory envelope, model identity manifest and
Git/LFS validation, hashed retained JSONL/phase/attempt-config/whole-GPU
peak/lane-provenance artifacts, vLLM stdout/stderr
startup snapshots, request/output-token count, complete zero-failure accounting, client-observed
one-token SSE TTFT/TPOT/E2E
percentiles와 wall throughput이 일치하는 workload를 가리킨다. `pair_order`는 timed
odd index에서 `riley-vllm`, even index에서 `vllm-riley`여야 한다. Riley marker는
CLI request에는 `native-bf16-paged-split-gqa-d128-two-stage`를, resolve에는 다음
exact runtime implementation을 기록한다.

paired driver는 한 lane의 retained 측정과 owned-server cleanup이 끝난 뒤 그 marker를
stdout에 쓰며, 두 marker의 line order도 선언한 `pair_order`와 같아야 한다. 이 규칙은 shared host의 시간 변동을
engine 순서와 혼동하지 않기 위한 것이다.

```text
riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2
```

또한 Riley marker는 attempt마다 서로 다른 absolute `startup_log_path`와 SHA-256을
포함한다. 이는 live `server.log`가 아니라 ready 직후 `riley serve`의 **stderr**에서
만든 create-only snapshot이어야 한다. summary는 current `riley serve`가 실제로 쓰는
다음 한 줄을 SHA-256과 함께 직접 파싱한다.

```text
RILEY_DECODE_ATTENTION requested_backend=native-bf16-paged-split-gqa-d128-two-stage resolved_ragged_backend=<exact-id> fallback_reason=none query_heads=16 key_value_heads=2 head_size=128 page_size=16 graph=false
```

이 startup receipt가 없거나 D64 fallback으로 바뀌면 serving 통계 자체를 거부한다.
vLLM marker도 ready 직후 create-only stdout/stderr snapshot의 absolute path와 SHA-256,
그리고 lane launch/cleanup provenance path와 SHA-256을 포함한다. summary는 provenance의
argv·cleanup·whole-GPU lifecycle을 marker 및 attempt config와 맞춘 뒤,
`Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:`
auto-selector receipt를 두 startup snapshot에서 정확히 한 번 확인한다.
`runtime_python` 또는 generic JSON backend receipt처럼 실제 server가 아직 만들지 않는
필드는 이 성능 evidence의 요구사항으로 쓰지 않는다. benchmark controller/client는
Python일 수 있지만 Riley serving runtime의 Rust→native ABI→CUDA 경로에는 Python이
들어가지 않는다는 것은 별도의 source/runtime release check로 검증한다.

각 N06-A 실행에는 `--model-identity-manifest`가 필수다. 이 manifest는
`riley.n06a-model-identity-manifest.v1` schema로 pinned model ID/revision,
exact `--model-path`, small metadata file의 size/SHA-256, 그리고 large shard의
size/Git-LFS OID를 기록한다. driver는 small file만 rehash하고
`git -C <model-path> rev-parse HEAD` 및 `git -C <model-path> lfs ls-files -l`로
large shard OID를 검증한다. small metadata file은 파일별 최대 8 MiB이며,
Qwen2.5-3B `tokenizer.json`(7,031,645 bytes)은 각 paired attempt에서 full SHA-256 rehash
대상이다. 이를 넘는 large shard는 payload를 reread하지 않고 size/Git-LFS OID로
검증한다. vLLM template도 exact
`vllm/vllm-openai@<digest>`, separate `--network host`, `--ipc host`,
`--gpus device=0`, `{model_path}:/model:ro`, `--model /model`,
`--served-model-name {model_id}`를 요구한다. 전체 schema와 invocation은
[`n06a_paired_serving_driver.md`](n06a_paired_serving_driver.md)가 권위 있다.

Summary는 N03a operator control과 N06-A full-model paired serving을 별도 object로
내보낸다. operator speedup을 serving speedup으로 재사용하지 않으며, `p99_status`
가 `qualified`인 경우에는 각 lane/cell에 최소 1,000 retained latency samples가
있어야 한다. P95/P99 paired ratio와 delta에는 outer-pair bootstrap CI가 포함되며,
planned pair가 빠졌거나 P99 표본이 부족하면 tail 결과는 descriptive/incomplete로
표시된다.
N06-A driver는 owned lane마다 physical GPU 0의 `nvidia-smi memory.used`를
0.25~0.5초 간격으로 sample한다. sampler는 server launch 전에 시작하고 owned-process
cleanup 뒤에 종료하며, raw row의 query start/end, observed cadence, lifecycle overlap을
peak receipt와 marker hash로 binding한다. query가 0.5초를 넘거나,
`configured interval + 0.5초`보다 긴 sample gap, running-lifetime overlap 누락,
sample error, 또는 `19,000,000,000` bytes 초과가 있으면 marker를 내지 않는다.
driver는 `CUDA_VISIBLE_DEVICES=0`과 Docker `--gpus device=0`을 강제하여 Riley lane,
vLLM lane, host sampler의 physical GPU identity를 일치시킨다. 이 ceiling은 run 중
allocation을 선제 중단하는 live limiter가 아니라, lane 후 marker eligibility를
판정하는 sampled evidence다. N01의 GPU pre/post snapshot과 CPU/I/O/memory PSI는
선택·filter·보정에 쓰지 않는 pre/post covariate일 뿐이다.
각 lane cleanup 뒤와 다음 lane 시작 전에는 별도 GPU-0 idle census가 retained
provenance에 남아야 한다. 이 census는 compute-process PID가 없고
`memory.used <= 512 MiB`임을 보이며, unrelated process를 종료하지 않는다.
그 조건을 증명하지 못하면 lane은 비교 표본이 될 수 없다.
`n06a_paired_serving_driver.py`의 Docker argv template, strict reference
workload, per-server warmup, vLLM startup-log backend/graph/compile/KV
attestation, and C/M examples are
[`n06a_paired_serving_driver.md`](n06a_paired_serving_driver.md)에 있다.

이 문서는 execution contract만 정의한다. 이 문서만으로 Riley와 vLLM의 실제
비교 결과나 우위를 주장할 수 없으며, 실행된 AB/BA receipt가 raw stream replay,
model identity, lifecycle, cleanup, whole-GPU peak, post-lane idle 검증을 모두
통과한 뒤에만 serving result로 읽는다.

## 반복성 gate

표준 runner는 matrix에 고정된 4개 cell을 5개의 독립 run 각각에서 순차
실행한다. 각 20개 cell 실행은 별도 subprocess와 결과 디렉터리를 사용하며,
같은 독립 run의 4개 cell만 `run_id`를 공유한다. output root는 실행 전에
존재하지 않아야 하고 repository 밖에 있어야 한다.

```bash
mkdir -p /var/tmp/riley-cache/{uv,uv-python,huggingface,vllm,torchinductor,triton,cuda}
UV_BIN=/absolute/path/to/pinned/uv
test "$("$UV_BIN" --version)" = 'uv 0.12.5 (x86_64-unknown-linux-gnu)'
test "$(sha256sum "$UV_BIN" | awk '{print $1}')" = \
  b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46
export UV_CACHE_DIR=/var/tmp/riley-cache/uv
export UV_PYTHON_INSTALL_DIR=/var/tmp/riley-cache/uv-python
export HF_HOME=/var/tmp/riley-cache/huggingface
export VLLM_CACHE_ROOT=/var/tmp/riley-cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/var/tmp/riley-cache/torchinductor
export TRITON_CACHE_DIR=/var/tmp/riley-cache/triton
export CUDA_CACHE_PATH=/var/tmp/riley-cache/cuda
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUNNER_PYTHON="$(UV_PYTHON_DOWNLOADS=never "$UV_BIN" python find 3.13.15)"
test "$(sha256sum "$RUNNER_PYTHON" | awk '{print $1}')" = \
  ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866
"$RUNNER_PYTHON" benchmarks/scripts/run_repeatability_gate.py \
  --lane hf-transformers \
  --output-root /var/tmp/riley-repeatability-hf-001 \
  --uv "$UV_BIN" \
  --finalize-to \
    benchmarks/results/20260824T000000Z-hf-transformers-eager-repeatability-run001
```

vLLM lane은 `--lane vllm`으로 선택한다. 다른 위치의 uv executable을 고정할
때는 `--uv /absolute/path/to/uv`를 사용한다. 이 옵션은 lane manifest argv의
첫 번째 literal `uv`만 치환한다.

runner는 실행 전에 `execution-plan.json`을 생성한다. 이 파일에는 matrix,
prompt corpus, lane manifest, dependency manifest/lock의 SHA-256, Git revision,
20개 exact argv, manifest environment override, 모든 artifact 경로와 재현성
allowlist의 exact 값이 기록된다. allowlist는 `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `HF_HOME`, `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`만 포함하며
secret을 포함할 수 있는 전체 inherited environment는 기록하지 않는다.
offline 두 flag는 exact `1`, cache root는 기존의 absolute repository-external
directory여야 한다. runner는 모든 child에 `UV_OFFLINE=1`을 추가하고 `PATH`,
locale/TLS/temp 같은 좁은 system allowlist, 위 cache/offline 값, version-controlled
manifest 값만 담은 exact environment를 plan에 평문 기록한다. ambient environment는
상속하지 않으며 `RILEY_*`, `VLLM_*`, `CUDA_*`, `TORCH_*`, `OMP_*` 등
측정·preflight를 바꿀 수 있는 미기록 override가 부모에 있으면 시작 전에
fail closed한다. 부모가 지정한 `UV_PROJECT_ENVIRONMENT`도 거부한다. 대신
runner가 selected lane, dependency lock SHA-256, gate execution nonce에 바인딩된
fresh 전용 경로를 `UV_PYTHON_INSTALL_DIR/project-environments/`
`<lane>-<lock-prefix>-<nonce>`로 derive한다. 시작 시 nonexistence와 repo 밖임을
검증하고 sync, prime, measured subprocess 모두에 같은 값으로 주입한다. path와
derivation은 plan에 기록되고 managed interpreter symlink와 전체 project tree는
UV Python install inventory fingerprint에 포함된다.
runner는 `UV_PYTHON=3.13.15`, `UV_PYTHON_DOWNLOADS=never`,
`PYTHONDONTWRITEBYTECODE=1`, `CUDA_CACHE_MAXSIZE=4294967296`,
`PYTHONHASHSEED=0`, `TOKENIZERS_PARALLELISM=false`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`DO_NOT_TRACK=1`, HF/vLLM telemetry opt-out 값을 exact child environment에
소유·기록한다. `uv python find 3.13.15` 결과와 project environment의 실제
interpreter가 Linux x86_64 CPython 3.13.15 및 pinned binary SHA-256인지도
각각 검증한다. `uv sync` 뒤 prime/measured 명령은 모두
`uv run --frozen --offline --no-sync`이므로 project environment를 다시
동기화하지 않는다.
각 invocation 직전에 `preflight.sh`를 다시
실행하고 stdout/stderr를 해당 cell 디렉터리에 분리 보존한다. preflight나
lane이 실패하면 즉시 중단하고 `failure.json`과 이미 생성된 로그를 남긴다.
첫 snapshot의 `persistence_mode`, `power_limit_w`, graphics/memory application
clock, 전체 CPU policy 수/governor, VRAM, `driver_version`을 baseline으로
고정하며, driver는 `580.173.02`, VRAM은 `24,564 MiB`, persistence mode는
`Disabled`, 정확히 24개 governor policy 모두 `powersave`여야 한다. 또한
Ubuntu 22.04, kernel `6.8.0-138-generic`, x86_64, Intel Core i7-13700K
(16 physical cores/24 logical threads), RAM `67,185,598,464` bytes를 exact로
확인한다. `NTPSynchronized=yes`와
staging filesystem의 최소 20 GiB 가용 공간도
매 invocation에서 확인한다. 이후 19개 snapshot에서 required key가 빠지거나
중복되거나 비교 설정이 baseline과 다르면 benchmark subprocess를 시작하지
않고 중단한다.
온도만 50°C를 초과한 정확한 preflight 오류는 30초 간격으로 최대 20분 동안
재시도한다. 각 attempt의 stdout/stderr/status snapshot은 모두 보존하며, 온도 외
오류는 즉시 중단한다. 안정화 뒤에도 측정 직전 마지막 full preflight 한 번이
반드시 통과해야 한다.

측정 전에 runner는 별도 `preparation/` stage를 수행한다. 선택 lane을
`uv sync --frozen --offline`으로 준비하고 exact argv, 시작/종료 시각, exit,
stdout/stderr, uv와 lock SHA-256를 보존한다. HF와 vLLM 모두 세 가지 distinct
compile/model profile `(c1,p128,o32)`, `(c1,p4096,o128)`, `(c8,p128,o32)`를 각각 별도 fresh
subprocess에서 unmeasured warm cell로 실행한다. prime raw와 로그는 보존하지만
checker 입력에는 절대 포함하지 않는다. 각 prime raw는 공통 result schema와
cross-file validator를 먼저 통과한 뒤 exact warm cell/run identity, 30개 연속
trial, 전부 success 및 `failure_count=0`인 경우에만 primed로 인정한다. 준비
전/후 external cache inventory는
relative path, file size, mtime과 aggregate SHA-256로 기록한다. post-prime
fingerprint가 measured baseline이며 20개 invocation 각각 직후 다시 계산해
조금이라도 달라지면 fetch/JIT cache fill로 보고 fail closed한다.
전체 entry 목록은 `cache.inventory.{before,after}.json.gz`에 canonical compact
JSON을 gzip level 9, `mtime=0`으로 압축해 저장한다. 압축은 Git artifact 크기만
줄이며, summary의 root별 count/bytes/fingerprint와 원본 entry 증거를 모두
보존한다.

이 계약의 `cold`는 process/model-state cold이다. 모든 independent run은 새
process에서 model을 새로 load하지만 immutable model/tokenizer, uv wheel, OS
page cache와 vLLM/TorchInductor/Triton compile disk cache는 preparation 뒤 같은 external
path를 재사용한다. 따라서 “완전한 filesystem/OS cache cold start” 결과로
해석하면 안 된다.

20개 raw JSONL이 모두 생성된 뒤 runner는 `check_repeatability.py`를 실행해
`repeatability-report.json`과 checker stdout/stderr를 보존한다. report가
`passed`가 아니면 runner도 nonzero로 종료한다. 성공 시 `completion.json`에
report SHA-256를 기록한다.

Checker v2는 `throughput_cv_max=0.05`를 warm cell에만 적용한다. Cold는 각
independent run에 첫 request 한 번만 있으므로 throughput CV를 진단 통계로
계속 보고하되 gate로 쓰지 않는다. Cold pass/fail은
`cold_model_load_p50_cv_max=0.10`, peak VRAM 상대 범위, failure count와 token
identity가 결정한다. Runner와 finalizer는
`contract_version=riley.repeatability.v2`인 passing report만 허용한다.

`--finalize-to`는 선택 사항이며 gate가 완전히 통과한 뒤에만 동작한다.
destination은 기존에 없는 `benchmarks/results/<id>` 한 단계 경로여야 한다.
id는 `<YYYYMMDDTHHMMSSZ>-<implementation-id>-repeatability-<run-id>` 형식이고
선택 lane implementation과 일치해야 한다.
runner는 staging tree의 symlink와 비정규 파일을 거부하고, 모든 파일의
크기와 SHA-256를 담은 `finalize-manifest.json`을 만든 뒤 숨은 임시
디렉터리에서 복사본을 재검증하고 destination으로 atomic rename한다. 외부
staging은 삭제하지 않으며 기존 result tree를 덮어쓰지 않는다. 따라서
version-controlled evidence는 이 finalize 결과만 review해서 추가한다.
finalize tree의 top-level `raw.jsonl`은 20개 measured raw만 결정적으로 결합한
파일이며, `metadata.json`과 `README.md`는 preparation/cache evidence, exact
commands, variance/comparability summary를 포함한다.

canonical Gate A는 exact repository `benchmarks/matrix.yaml`,
`benchmarks/prompts.jsonl`, `preflight.sh`, `check_repeatability.py`만 사용하며
subprocess를 만들기 전에 전체 `validate_contract` 검증을 통과해야 한다. runner
plan은 두 스크립트, runner 자신,
Python, uv의 path/SHA-256와 runtime version을 보존한다. canonical 값은
uv 0.12.5 Linux x86_64 binary와 CPython 3.13.15 Linux x86_64 binary로
fail closed한다. offline unit test에서
fake executable을 주입할 때만 `--allow-noncanonical-tools`를 쓰며, 이 mode는
`--finalize-to`와 함께 사용할 수 없다.

checker는 통계 전에 공통 result schema와 cross-file validator를 모든 raw
JSONL에 적용한다. malformed row는 `error`, schema-valid하지만 서로 다른
revision/hash/environment를 가리키는 row는 threshold 실패와 구분해
`incomparable`로 취급한다. 동일 cell의 각 trial/request position은 5개
run에서 input 및 generated token ID SHA-256가 같아야 한다.
checker도 canonical matrix에 `validate_matrix`를 적용한다. synthetic fixture용
`--allow-noncanonical-matrix`는 offline test 전용이며 canonical runner/finalize가
절대 전달하지 않는다.
