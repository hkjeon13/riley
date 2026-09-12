# G03 attention chain 원격 검증 — 2026-09-11

## 결과

승인된 9개 소스를 `ai-assistant:/tmp/riley-g01-native-260910`에 전송하고 전후 SHA-256을 확인했다. 추가 소스 수정 없이 CUDA/native/Rust 빌드와 C ABI 검증이 통과했다. 이전 payload의 승인 차단은 해소되었다.

GPU 테스트는 총 **22 passed, 0 failed**다. 새 `input norm → Q/K/V → indexed RoPE → KV write → grouped attention` 경로는 **640 replay**에서 eager 결과와 일치했다. 기존 QKV-KV, QKV-RoPE, norm-QKV, layer-tail, norm-MLP, MLP 경로도 각각 640 replay 검증을 통과했다.

SmolLM2 L30 및 canonical L2/L3의 5개 binding parity receipt가 직전 KV-write 검증과 모두 일치한다. logits·initialized KV·continuation 토큰을 유지했다. 전체 cache CPU scatter oracle, scratch·hidden context·packed metadata·pinned staging 복원을 검증했다. metadata 범위·정렬·겹침, logical block 제한 및 잘못된 위치의 replay 거절도 native 테스트에서 통과했다.

## 검증 구성

| 로그 | 필터 | 통과 |
| --- | --- | ---: |
| gpu-attention-native.log | qkv_rope_capture | 1 |
| gpu-model.log | c07_ | 14 |
| gpu-qkv-native.log | norm_qkv_parent | 1 |
| gpu-tail-native.log | mlp_reserved_geometry | 1 |
| gpu-aggregate.log | aggregate_ | 3 |
| gpu-swiglu.log | swiglu | 2 |

모든 GPU 테스트는 CUDA feature test binary에 `--ignored --nocapture --test-threads=1`을 지정해 실행했다. CUDA 빌드는 `cargo test -p riley-cuda -p riley-runtime --features cuda --lib --no-run`으로 수행했다. 경고는 남아 있으나 빌드와 모든 테스트 명령은 exit 0이다. debug 테스트 소요 시간은 성능 수치로 사용하지 않는다.

직전 [로컬 구현 기록](../20260911-g03-attention-chain/README.md)의 CPU 391/ABI/fmt/일반 Clippy 결과를 유지하며, 이번에는 추가 소스 수정 없이 fmt/diff를 재확인했다. 별도 model-loader 6파일의 hash가 보존되었다.

## 증거 파일

- `authorized-transfer.json`, `remote-before.txt`, `remote-hashes.txt`: 승인 범위와 전후 source identity.
- `build.log`, `gpu-*.log`: 실제 CUDA 빌드 및 GPU 실행 결과.
- `model-parity.json`: 직전 로그 hash, 5개 receipt 비교, 경로별 replay 수 및 테스트 수.
- `runtime-identities.txt`: GPU/driver, 테스트 binary 및 checkpoint manifest identity.
- `model-preservation.json`, `local-checks.log`: 별도 변경 보존과 로컬 검사.
- `manifest.json`: 결과와 증거 파일 SHA-256.

## 범위와 남은 절차

이번 결과는 완료된 iteration의 실제 scratch와 마지막 layer weight를 이용하는 진단용 부분 graph 검증이다. M=1, D=64, logical block 0, 고정 physical block 및 valid prefix를 사용한다. mapping/prefix 변경 시 recapture가 필요하며, 일반 동적 metadata나 block 경계를 넘는 decode를 검증한 것은 아니다. 모델 fixture의 step 0–3 검증이며 긴 context 증거가 아니다.

다음은 이 attention 앞부분과 기존 layer-tail을 한 layer graph로 연결하는 단계다. 이후 실제 전체 layer/model decode 연결, embedding/final norm/head/output/status, 동적 metadata, 오류·completion·lifetime, bucket/admission/fallback qualification이 남는다. FixedContiguous37Balanced, projection bias, nonzero required workspace, CUDA failure injection도 미검증이다.

**전체 decode graph 및 G02H/G03 qualification 완료로 승격하지 않는다. 성능 측정은 시작하지 않았다.**
