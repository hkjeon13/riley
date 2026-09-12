# G03 norm/QKV/RoPE → KV write — remote verification

사용자가 직전 9개 소스 전송·CUDA 검증·동일 파일 내 수정/재전송 요청에 동의한 뒤 지정된 `ai-assistant:/tmp/riley-g01-native-260910` scratch에 해당 소스만 전송했다. 로컬 소스/tar와 원격 전후 SHA-256 일치를 확인했다. 검증 중 추가 수정은 없었다.

## 결과

- CUDA native 및 CUDA 활성화 Rust 빌드, 원격 C11 ABI 검사 성공(exit 0).
- **GPU 22 passed**, 모든 테스트 명령 exit 0: KV/RoPE native 1, norm-QKV native 1, tail native 1, C07 14, ledger/transfer 3, SwiGLU 2.
- 실제 모델 통합 KV write **640 replay**(5 case × 4 iteration × 32). 각 audit 종료 후 key/value cache parent 전체를 독립 CPU scatter oracle과 비교해 지정한 위치 외 다른 token/layer의 보존을 확인했다. zero/original 입력 교대에서 기록된 K/V 결과를 oracle에 순서대로 적용했다. cache·scratch·packed metadata 원상복원도 확인했다.
- 기존 QKV-RoPE/norm-QKV/layer-tail/norm-MLP/MLP 각각 **640 replay** eager parity 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시 및 continuation이 직전 QKV-RoPE 기준과 동일. 토큰은 각각 `[808,2775,288,536]`, `[2,2,2,2]`, `[5,5,5,0]`.
- cache parent size/layer/block/alias/valid prefix 및 replay 위치 거절, close/Drop/domain 해제 확인.
- 소스가 바뀌지 않아 이전 CPU 391개 결과를 유지했다. 최종 fmt/diff 통과. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6파일 해시 보존. commit/push/deploy 및 성능 측정 없음.

## 범위와 남은 절차

norm→Q/K/V→RoPE→KV write를 단일 retained graph에서 실행했다. 실제 모델의 완료된 scratch snapshot과 마지막 layer weight를 쓰는 진단이며 실제 모든 layer의 activation trace가 아니다.

logical/physical block mapping 및 valid prefix는 capture 시 고정된다. replay는 해당 범위의 위치만 허용한다. mapping/prefix 변경에는 재capture가 필요하므로 일반적인 block 경계를 넘는 retained decode 또는 동적 packed admission으로 승격하지 않는다. 실제 모델 검증은 현재 fixture의 step 0~3 범위다.

다음은 attention과 layer tail의 통합, 모든 layer/embedding/final norm/head/output 연결, 동적 metadata·오류/완료·자원 수명·bucket/admission/fallback qualification이다. projection bias/FixedContiguous37Balanced, 필수 nonzero workspace 알고리즘 및 CUDA 실패 주입은 이번 통과 범위에 포함하지 않는다. 전체 decode/G02H/G03 완료나 성능 측정 준비 완료로 승격하지 않는다. 사용자 요청대로 성능 측정은 시작하지 않았다.

## 증거

[승인·소스 해시](authorized-transfer.json), [전송 전](remote-before.txt), [전송 후](remote-hashes.txt), [빌드](build.log), [GPU·바이너리·checkpoint](runtime-identities.txt), [KV native](gpu-kv-native.log), [QKV native](gpu-qkv-native.log), [tail native](gpu-tail-native.log), [모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log), [parity](model-parity.json), [변경 보존](model-preservation.json), [로컬 검사](local-checks.log).

이전 KV-write 9파일 전송 승인 차단은 이번 승인과 실행으로 해소됐다.
