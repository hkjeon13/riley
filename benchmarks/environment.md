# Baseline environment v1

이 문서는 첫 release까지 비교 기준으로 사용하는 primary 환경을 고정한다. 실제 측정 row는 이 문서의 값과 실행 직전 `nvidia-smi` snapshot을 함께 기록한다.

## Gate A 상태 출처

이 문서는 primary 환경과 실행 조건만 고정하며, 실행 후의 pass/fail 상태를
문서 본문에 덮어쓰지 않는다. Gate A의 권위 있는 상태와 raw artifact 위치는
`benchmarks/results/PR01.md`에 별도로 기록한다. 그 인덱스가 없거나 passing
evidence를 가리키지 않으면 PR 01을 `Complete`로 전환하거나 성능 우위를
주장하지 않는다.

## Primary hardware

`environment_id`: `rtx4090-ubuntu22-driver580-v1`

| 항목 | 고정 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Compute capability | 8.9 (Ada) |
| VRAM | 24,564 MiB |
| Primary compute dtype | BF16 |
| Driver | 580.173.02 |
| Driver가 보고한 CUDA API | 13.0 |
| Host CUDA toolkit | 설치되지 않음; lane의 locked wheel/container runtime을 사용 |
| CPU | Intel Core i7-13700K, 16 cores / 24 threads |
| RAM | 67,185,598,464 bytes |
| OS | Ubuntu 22.04, Linux 6.8.0-138-generic, x86_64 |

primary `environment_id`는 persistence mode `Disabled`, 24개 cpufreq policy의
CPU governor `powersave`, VRAM `24,564 MiB`, NVIDIA driver `580.173.02`를 exact
값으로 요구한다. runner는 첫
snapshot의 power limit과 graphics/memory application clock도 고정해 이후
19개 invocation과 비교한다. 설정을 바꾼 결과는 이 환경 ID로 기록하지 않고
새 `environment_id`를 사용한다.

로컬 macOS x86_64 호스트에는 NVIDIA GPU와 `nvcc`가 없으므로 문서 작성, schema 검사, model-download 없는 unit test만 수행한다. GPU correctness와 성능 결과는 위 target host에서만 승인한다.

## Run preflight

각 독립 run 전에 다음 조건을 확인하고 raw artifact의 provenance에 원문을 보존한다.

각 cell의 5개 independent run은 모두 새 process에서 model을 다시 load한다.
`warm` run은 그 process 안에서 warmup 5회와 measured trial 30회 동안만 model
state를 재사용한다. `cold` run은 warmup 없이 1회만 측정하며 process/model을
다음 run이나 cell에 재사용하지 않는다.

여기서 `cold`는 process와 model-state cold만 뜻한다. Gate A 전에 선택 lane을
locked dependency로 offline sync하고, HF와 vLLM 모두 gate의 세 distinct
concurrency / prompt / output compile/model profile을 fresh unmeasured
subprocess로 prime한다. 이후
5개 independent run은 동일한 repository-external `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`를 재사용한다. OS page cache, immutable model/tokenizer
disk cache, wheel cache와 compiled-kernel disk cache를 비우지 않는다.
Python project environment도 repository 내부 `.venv`를 쓰지 않는다. runner가
lane ID, dependency lock SHA-256, gate nonce로
`UV_PYTHON_INSTALL_DIR/project-environments/<lane>-<lock-prefix>-<nonce>`의
fresh `UV_PROJECT_ENVIRONMENT`를 derive하고 offline sync, prime, measured run에
동일하게 주입하며, 그 path와 managed interpreter symlink를 포함한 내용은 plan
및 cache fingerprint에 바인딩한다. runner와 lane interpreter는 Linux x86_64
CPython 3.13.15 및 pinned binary SHA-256를 사용한다. runner가
`UV_PYTHON=3.13.15`, `UV_PYTHON_DOWNLOADS=never`,
`PYTHONDONTWRITEBYTECODE=1`, deterministic thread/hash 설정, telemetry opt-out,
`CUDA_CACHE_MAXSIZE=4294967296`를 exact environment로 주입한다. CUDA driver JIT
cache는 external `CUDA_CACHE_PATH`, Triton JIT cache는 external
`TRITON_CACHE_DIR` 아래에서만 prime/reuse하며, sync 뒤 모든
lane command는 `uv run --frozen --offline --no-sync`이다.
`HF_HUB_OFFLINE=1`과 `TRANSFORMERS_OFFLINE=1`은 exact contract다. runner는
준비 command/log/timestamp/lock·tool hash와 cache inventory fingerprint를 보존하고,
각 measured invocation이 post-prime cache를 변경하면 측정을 중단한다.

두 Python reference lane의 `peak_vram_bytes`와 GPU utilization은 primary GPU
전체를 NVML로 sampling하고, CPU utilization은 frontend와 recursive worker
process tree의 psutil CPU time으로 계산한다. 따라서 preflight의 “다른 CUDA
process 없음” 조건이 계측 정의의 일부다.

- 다른 CUDA compute process가 없음
- idle GPU memory가 256 MiB 이하
- 시작 GPU 온도가 50°C 이하
- persistence mode `Disabled`, 모든 CPU policy governor `powersave`, VRAM
  `24,564 MiB`, driver `580.173.02` (governor policy count는 정확히 24)
- power limit과 graphics/memory application clock이 20개 invocation에서 동일
- model, matrix, prompt, lane manifest, dependency lock의 SHA-256가 동일
- clean Git revision 사용
- `timedatectl`의 `NTPSynchronized=yes`
- 외부 staging output filesystem의 사용 가능 공간이 매 invocation 직전 최소
  20 GiB(21,474,836,480 bytes)
- 시작 온도만 50°C를 초과하면 30초마다 최대 20분까지 bounded stabilization을
  수행하고 모든 attempt를 보존; 다른 preflight 오류는 즉시 중단

조건을 만족하지 못한 run은 실패가 아니라 `incomparable`로 분류하고 repeatability 계산에 넣지 않는다.

## Primary checkpoint

| 항목 | 고정 값 |
|---|---|
| Model ID | `HuggingFaceTB/SmolLM2-135M` |
| Revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Architecture | `LlamaForCausalLM` / `model_type=llama` |
| Weight | `model.safetensors`, 269,060,552 bytes |
| Weight SHA-256 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| Stored dtype | BF16 |
| Max positions | 8,192 |
| License | Apache-2.0 |

고정 revision의 [`config.json`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json), [artifact tree](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2), [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md)을 provenance source로 사용한다. 다운로드는 전체 revision SHA를 요구하고 `trust_remote_code=false`로 실행한 뒤 weight checksum을 다시 검증한다.

이 모델은 영어 중심이다. 한국어 fixture는 tokenizer, tensor, logits parity를 검증하지만 언어 품질 비교 근거로 사용하지 않는다. 135M 모델 선택은 download와 반복 correctness 비용을 작게 유지하고 작은 batch에서 host/runtime overhead를 드러내기 위한 결정이며, 큰 모델 전체의 성능을 대표한다고 주장하지 않는다.

## 격리된 실행 lane

### Transformers reference

- 역할: golden fixture와 명확한 eager reference 생성
- dependency class: `python-reference`
- Python: 3.13.x, lock 생성 기준 patch 3.13.15
- lock generator: uv 0.12.5, `exclude-newer=2026-08-24T23:59:59Z`
- PyTorch: 2.13.0
- Transformers: 5.15.1
- attention implementation: eager
- dependency source: `tools/python/reference/pyproject.toml`과 `uv.lock`
- 금지: `trust_remote_code`, floating revision, production fallback으로 사용

### vLLM baseline

- 역할: serving 성능 비교 대상
- dependency class: `python-reference`
- Python: 3.13.x, lock 생성 기준 patch 3.13.15
- lock generator: uv 0.12.5, `exclude-newer=2026-08-24T23:59:59Z`
- vLLM: 0.27.1
- dependency source: `benchmarks/lanes/vllm/pyproject.toml`과 lane lock
- model, revision, dtype, token IDs, EOS 정책과 output length는 matrix와 같아야 함

SGLang 또는 TensorRT-LLM은 선택적인 세 번째 baseline이다. PR 01의 최소 비교 계약은 Transformers와 vLLM으로 고정하고, 다른 engine은 동일 schema를 만족하는 별도 lane으로만 추가한다.

### rustinfer production

- 역할: PR 11 이후 실제 native benchmark 대상
- dependency class: `native-production`
- 허용: Rust release binary/library, native CUDA library, NVIDIA runtime library, model/tokenizer artifact
- 금지: Python interpreter/subprocess, PyTorch, Transformers, Triton Python JIT

PR 01에서는 argv와 dependency 금지 계약만 정의한다. 아직 존재하지 않는 binary의 측정값을 만들거나 Python reference를 production 결과로 표기하지 않는다.

## 비교 가능성

다음 중 하나라도 다르면 직접적인 before/after 비교가 아니다.

- checkpoint 또는 tokenizer revision
- dtype/quantization
- tokenized prompt IDs 또는 requested output tokens
- sampling/EOS 처리
- cold/warm state
- GPU/driver/environment ID
- dependency lane lock

각 lane의 기계 판독 가능한 manifest와 실행 argv는 `benchmarks/lanes/`에, 공통 workload와 반복성 threshold는 `benchmarks/matrix.yaml`에 둔다.
