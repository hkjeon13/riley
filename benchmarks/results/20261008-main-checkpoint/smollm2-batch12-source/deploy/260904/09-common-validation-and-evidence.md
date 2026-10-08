# 공통 검증과 evidence 계약

모든 260904 카드가 공유하는 최소 검증이다. 개별 카드가 더 엄격하면 개별 카드가 우선한다.

## 시작 전 snapshot

```bash
git status --short --branch
git rev-parse HEAD
git diff --check
```

- unrelated dirty/untracked 파일을 보존한다.
- claim/candidate 작업은 최신 승인 base의 별도 clean worktree에서 한다.
- `--allow-dirty-source` 결과는 개발 진단일 뿐 competitive/release evidence가 아니다.

## 코드 PR 공통 gate

수정 범위에 맞는 최소 집합을 실행한다.

```bash
cargo fmt --all --check
cargo test -p riley-cuda --lib
cargo test -p riley-cuda --test graph_cpu
cargo test -p riley-runtime --lib
cargo test -p riley-runtime --test architecture_boundary
cargo test -p riley-scheduler --lib
cargo test -p riley-server --lib
python3 -m pytest -q benchmarks/competitive/scripts/tests
git diff --check
```

CUDA feature/native build가 필요한 PR은 feature-off test만으로 완료하지 않는다. 로컬에 CUDA/GPU가
없으면 `source-complete / gpu-pending`으로 끝내고 GPU pass를 주장하지 않는다. 기존 unrelated test
failure는 baseline에서도 재현했을 때만 별도 issue로 분리하며 새 threshold 완화로 숨기지 않는다.

Q-track에서 수정한 native tool은 해당 directory의 strict C11 gate를 추가 실행한다.

```bash
make -C tools/native/<changed-tool> test
make -C tools/native/<changed-tool> analyze
python3 -m pytest -q ci/release/test_gate_e_native_guardian_review_contract_v1.py
python3 -m pytest -q ci/release/test_rc3_gate_e_guardian_lease_contract_v1.py
```

`analyze` 지원 여부는 해당 Makefile에서 먼저 확인한다. 없는 target을 통과한 것으로 기록하지 않는다.

## GPU test entry points

승인된 clean GPU environment에서 최소 다음 existing entry point를 사용하고, 카드가 요구하는 새 test
target은 Cargo manifest에 명시적으로 등록한다.

```bash
cargo test -p riley-cuda --features cuda --test graph_gpu
cargo test -p riley-runtime --features cuda --test graph_dispatch_gpu
```

G01~G03이 추가하는 focused GPU target은 broad ignored suite에만 숨기지 않는다. exact test binary와
filter, passed/failed/ignored count를 receipt에 기록한다.

## GPU correctness receipt 최소 필드

- Git revision과 clean state
- source archive SHA-256, release binary/image SHA-256
- model revision, weights/tokenizer hashes, dtype
- GPU UUID/name/compute capability, driver, CUDA runtime/toolkit
- exact command/options/environment allowlist digest
- test name, semantic class, input/output/token/KV hashes
- allocation before/after, completion/close status, failure count

GPU parity는 같은 binary에서 eager/graph 또는 reference/candidate를 실행하고 token/status/KV continuation을
비교한다. HTTP 200, CUDA graph instantiate 성공 또는 kernel launch 성공만으로 parity를 선언하지 않는다.

## 성능 campaign 공통 계약

- independent fresh process 5회 이상, AB와 BA 모두 포함
- prime/warmup은 measured sample과 분리
- failure, timeout, dropped trace, token mismatch를 success percentile에서 숨기지 않음
- engine-only와 HTTP streaming을 별도 series로 보존
- c1 output tok/s와 saturation throughput을 다른 metric으로 표기
- GPU thermal/contention/clock preflight와 model/request/runtime identity를 raw record에 연결

새 코드가 들어간 뒤 과거 campaign은 현재 candidate claim에 재사용하지 않는다. diagnostic Tier D와
competitive Tier C/S report도 서로 대체하지 않는다.

## artifact와 문서 갱신

- raw artifact는 create-only/append-only directory에 기록한다.
- favorable summary만 commit하고 raw failure를 버리지 않는다.
- 각 카드 종료 시 [STATUS.md](STATUS.md)에 revision, decision, 검증, artifact만 기록한다.
- historical 계획의 수치를 덮어쓰지 않고 새 campaign ID를 만든다.

## 공통 decision vocabulary

| Decision | 의미 |
|---|---|
| `source-complete / gpu-pending` | source/CPU 계약만 통과 |
| `promoted` | correctness와 사전 성능 gate를 모두 통과 |
| `experimental/not-promoted` | correctness는 통과했지만 성능/default gate 미달 |
| `rejected` | 가설 또는 correctness가 실패하여 candidate 제거 |
| `blocked` | 권한, artifact, 선행 evidence가 없어 안전하게 진행 불가 |
| `incomparable` | identity/campaign 조건이 달라 비교 불가 |
