# G03 post-attention norm + MLP — remote verification

사용자가 명시 승인한 소스 9개만 기존 `ai-assistant:/tmp/riley-g01-native-260910` scratch에 전송했다. 전송 전 원격 기존 해시, 로컬 소스 및 tar 내용, 전송 후 원격 해시를 모두 확인했다. 이번 검증 중 추가 소스 수정은 없었다.

## 검증 결과

- CUDA native 및 CUDA 활성화 Rust 빌드 성공(exit 0). 원격 C11 ABI 구문 검사 성공.
- **GPU 20 passed**, 모든 테스트 명령 exit 0: native MLP/norm-MLP 1, C07 모델 14, ledger/transfer 3, SwiGLU 2.
- 정규화→MLP **640 replay**(5 case × 4 iteration × 32), 기존 MLP **640 replay**에서 eager down/residual 결과의 byte parity 확인. zero/original 입력 교대, norm 재계산, plan 불변성, pinned tail 보존 및 scratch 복원 통과.
- SmolLM2 L30, canonical L2/L3 logits·initialized KV 해시 및 continuation이 직전 MLP 검증 기준과 동일. 토큰은 각각 `[808,2775,288,536]`, `[2,2,2,2]`, `[5,5,5,0]`.
- native fixture에서 등록되지 않은/aliased norm weight, NaN/무한대/0/음수 epsilon, 잘못된 profile 및 HF geometry 거절을 확인했다. 정상 capture의 close/Drop과 domain 해제도 통과했다.
- 로컬 fmt/diff 통과. 이전 단계 CPU 391개 검증을 유지하며 소스가 바뀌지 않아 재실행하지 않았다. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6개 파일 전후 해시 일치. commit/push/deploy 없음.

## 의미와 남은 범위

동일 graph 안에서 residual 입력 → post-attention RMSNorm → gate/up GEMM → SiLU/multiply → down GEMM → residual add의 실제 실행 정확성을 확인했다. canonical 및 HF SmolLM2 norm은 각각 기존 eager kernel/profile을 유지한다.

이 검사는 완료된 iteration의 scratch snapshot과 마지막 layer weight를 사용하는 부분 graph다. 모든 layer의 실행 중 입력 연결이나 전체 decode graph replay를 증명하지 않는다. 첫 staging 영역에 입력된 scratch는 norm 결과로 덮어쓰며 residual 영역을 새로 공급한다. FixedContiguous37Balanced, 필수 nonzero workspace 알고리즘 및 capture/실행 실패 주입은 이번 검증 범위에 포함하지 않는다.

다음 단계는 attention 출력·residual과 이 정규화-MLP 구간을 실제 layer 입력에 연결하고, layer별 상태를 같은 retained model DAG로 통합하는 작업이다. 전체 decode의 fresh input·완료·오류·복원 계약을 검증한 뒤 G02H/G03 qualification과 동일 조건 SmolLM2 vLLM 성능 비교가 필요하다. 성능 향상은 아직 미검증이다.

## 증거

[전송 승인 및 소스 해시](authorized-transfer.json), [전송 전 해시](remote-before.txt), [전송 후 해시](remote-hashes.txt), [바이너리/GPU/checkpoint 식별](runtime-identities.txt), [빌드](build.log), [native](gpu-native.log), [모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log), [모델 parity](model-parity.json), [model-loader 보존](model-preservation.json), [로컬 검사](local-checks.log).

이전 정규화-MLP 9파일 전송 승인 차단은 이번 명시 승인과 실행으로 해소됐다.
