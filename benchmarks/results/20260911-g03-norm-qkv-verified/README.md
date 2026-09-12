# G03 input norm → QKV — remote verification

사용자가 바로 앞의 8개 소스 전송·CUDA 검증 승인 요청에 진행하도록 답한 뒤 해당 payload만 기존 `ai-assistant:/tmp/riley-g01-native-260910` scratch에 전송했다. 기존 7파일 해시와 새 audit 부재, 로컬 소스/tar 일치, 전송 후 8개 원격 해시를 확인했다. 추가 수정은 없었다.

## 결과

- CUDA native 및 CUDA 활성화 Rust 빌드와 원격 C11 ABI 검사 성공(exit 0).
- **GPU 21 passed**, 모든 테스트 명령 exit 0: norm-QKV native 1, 기존 tail native 1, C07 14, ledger/transfer 3, SwiGLU 2.
- norm-QKV **640 replay**에서 zero/original 입력 교대 및 eager Q/K/V byte parity 확인. 기존 layer-tail/norm-MLP/MLP도 각각 **640 replay** 통과했다.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시와 continuation이 직전 layer-tail 검증 기준과 동일. 토큰은 `[808,2775,288,536]`, `[2,2,2,2]`, `[5,5,5,0]`.
- Q/KV 폭 차이, 부모/alias/geometry/profile 거절, optional workspace, capture/close/Drop, scratch/pinned 복원과 GEMM plan 불변성 검사 통과.
- 공통 capture lifecycle 추출 후 MLP/tail native 및 실제 모델 회귀를 통과했다.
- 이전 CPU 391개 결과를 유지한다. 소스가 바뀌지 않아 CPU 검사를 재실행하지 않았다. 최종 fmt/diff 성공. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. commit/push/deploy 없음. 성능 측정 미실행.

## 범위와 다음 절차

input norm → Q/K/V를 한 graph로 실행하며 실제 모델의 완료된 scratch snapshot과 마지막 layer weight에 대해 eager parity를 검증했다. 이것은 RoPE, KV write, attention 또는 실제 모든 layer 입력을 연결한 full decode 증거가 아니다. projection bias/FixedContiguous37Balanced, 필수 nonzero workspace 알고리즘 및 CUDA 실패 주입은 이번 통과 범위에 포함하지 않는다.

다음은 RoPE→KV write→attention을 이 앞부분과 연결하고 검증된 layer tail까지 통합하는 작업이다. 이후 모든 layer와 embedding/final norm/head/output, fresh metadata·오류/완료·자원 수명·bucket/fallback qualification이 남아 있다. G02H/G03 완료나 성능 측정 준비 완료로 승격하지 않는다. 사용자의 요청대로 성능 측정은 시작하지 않았다.

## 증거

[승인 및 소스 해시](authorized-transfer.json), [원격 이전 해시](remote-before.txt), [전송 후 해시](remote-hashes.txt), [빌드](build.log), [GPU·바이너리·checkpoint 식별](runtime-identities.txt), [QKV native](gpu-native.log), [tail native](gpu-tail-native.log), [실제 모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log), [모델 parity](model-parity.json), [변경 보존](model-preservation.json), [로컬 검사](local-checks.log).

이전 8파일 전송 승인 차단은 이번 승인과 실행으로 해소됐다.
