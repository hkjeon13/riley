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
