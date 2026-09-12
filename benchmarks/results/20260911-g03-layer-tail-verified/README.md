# G03 layer tail — authorized remote verification

사용자가 명시 승인한 소스 7개만 `ai-assistant:/tmp/riley-g01-native-260910`에 전송했다. 로컬 소스/tar와 원격 전후 해시를 확인했으며 검증 중 추가 수정은 없었다.

## 결과

- CUDA native 및 CUDA 활성화 Rust 빌드, 원격 C11 ABI 검사 성공(exit 0).
- **GPU 20 passed**, 모든 테스트 명령 exit 0: native 1, C07 14, ledger/transfer 3, SwiGLU 2.
- layer-tail **640 replay**, norm-MLP **640 replay**, MLP **640 replay**가 eager byte parity와 zero/original 입력 교대를 통과했다.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시와 continuation이 직전 norm-MLP 기준과 동일. 토큰은 각각 `[808,2775,288,536]`, `[2,2,2,2]`, `[5,5,5,0]`.
- projection weight/plan 등록·alias·geometry 거절, close/Drop, scratch/pinned 복원과 선택 GEMM plan 불변성 검사 통과.
- 소스가 바뀌지 않아 앞 단계 CPU 391개 결과를 유지했다. 최종 fmt/diff 검사 통과. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. commit/push/deploy 및 성능 측정 없음.

## 검증 범위

attention context → output projection → attention residual → post-attention norm → MLP → final residual을 단일 graph로 실행했다. 완료된 iteration의 실제 scratch snapshot과 마지막 layer weight로 eager parity를 검사했다. projection/norm/final output scratch 재사용과 in-place attention residual은 이 범위에서 검증됐다.

이는 Q/K/V·RoPE·KV write·attention부터 실행한 전체 layer graph가 아니며, 실제 실행 중 모든 layer 입력을 연결한 full decode도 아니다. FixedContiguous37Balanced, projection bias, 필수 nonzero workspace 알고리즘 및 capture/실행 실패 주입은 이번 통과 범위에 포함하지 않는다.

성능 측정 전까지 남은 것은 input norm/QKV/RoPE/KV/attention 통합, 실제 layer 입력 기반 연결, embedding부터 output/status까지 전체 model DAG, multi-step·오류/완료·bucket/admission/fallback qualification이다. 이 기준을 통과하기 전 성능 준비 완료나 G02H/G03 완료로 승격하지 않는다. 성능 측정은 사용자 요청대로 시작하지 않았다.

## 증거

[승인 및 소스 해시](authorized-transfer.json), [원격 이전 해시](remote-before.txt), [전송 후 해시](remote-hashes.txt), [빌드](build.log), [GPU·바이너리·checkpoint 식별](runtime-identities.txt), [native](gpu-native.log), [모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log), [모델 parity](model-parity.json), [변경 보존](model-preservation.json), [로컬 검사](local-checks.log).

이전 7파일 전송 승인 차단은 이번 명시 승인과 실행으로 해소됐다.
