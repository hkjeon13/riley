# G03 norm/QKV → RoPE — remote verification

사용자가 직전 9파일 전송·CUDA 검증·동일 파일 내 수정/재전송 요청에 동의한 뒤 지정된 `ai-assistant:/tmp/riley-g01-native-260910` scratch에 해당 소스만 전송했다. 로컬 소스/tar와 원격 전후 SHA-256이 일치했다. 추가 수정은 없었다.

## 결과

- CUDA native 및 CUDA 활성화 Rust 빌드, 원격 C11 ABI 검사 성공(exit 0).
- **GPU 22 passed**, 모든 테스트 명령 exit 0: QKV-RoPE native 1, norm-QKV native 1, tail native 1, C07 14, ledger/transfer 3, SwiGLU 2.
- 실제 모델 QKV→RoPE **640 replay**에서 eager rotated Q/K 및 V byte parity를 확인했다. zero input/position 0과 원본 input/현재 위치를 교대했다. 현재 step 0~3 범위의 packed metadata와 host 위치 일치도 확인했다.
- 기존 norm-QKV/layer-tail/norm-MLP/MLP도 각각 **640 replay** 통과했다. SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시 및 continuation은 직전 norm-QKV 기준과 동일하다. 토큰은 각각 `[808,2775,288,536]`, `[2,2,2,2]`, `[5,5,5,0]`.
- 테이블 밖 위치·잘못된 offset/alias/payload 거절, 완료 전 read 차단, close/Drop/domain 해제, 실제 packed position parent와 rotated scratch 복원 통과.
- 소스가 바뀌지 않아 앞 단계 CPU 391개 결과를 유지했다. 최종 fmt/diff 통과. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. commit/push/deploy 및 성능 측정 없음.

## 범위와 남은 절차

input norm→Q/K/V→D64 indexed RoPE를 단일 retained graph로 실행했다. 실제 모델의 완료된 scratch snapshot과 마지막 layer weight/table에 대한 부분 graph 진단이며 모든 layer의 실제 실행 중 activation trace나 전체 decode replay 증거가 아니다.

KV write·attention은 아직 이 graph에 연결되지 않았다. 다음으로 실제 packed KV metadata와 layer span을 연결하고 attention 및 layer tail을 통합해야 한다. 이후 모든 layer/embedding/final norm/head/output, metadata freshness·오류/완료·자원 수명·bucket/admission/fallback qualification이 남아 있다. projection bias/FixedContiguous37Balanced, 필수 nonzero workspace 알고리즘, CUDA 실패 주입 및 장문 위치 전 범위는 이번 통과 범위에 포함하지 않는다.

G02H/G03 또는 성능 측정 준비 완료로 승격하지 않는다. 사용자 요청대로 성능 측정은 시작하지 않았다.

## 증거

[승인 및 소스 해시](authorized-transfer.json), [전송 전](remote-before.txt), [전송 후](remote-hashes.txt), [빌드](build.log), [GPU·바이너리·checkpoint](runtime-identities.txt), [RoPE native](gpu-rope-native.log), [QKV native](gpu-qkv-native.log), [tail native](gpu-tail-native.log), [모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log), [parity](model-parity.json), [변경 보존](model-preservation.json), [로컬 검사](local-checks.log).

이전 QKV-RoPE 9파일 전송 승인 차단은 이번 승인과 실행으로 해소됐다.
