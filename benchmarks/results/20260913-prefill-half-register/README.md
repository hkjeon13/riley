# BF16 storage / FP16 register prefill diagnostic

FlashInfer 0.6.16.post3의 고정 Q128 prefill 경로에 명시적 register 변환을 구현했다. 기존 16-bit load는 K/V 타입 변환을 하지 않으므로 Q 타입만 half로 변경하면 BF16 비트를 잘못 해석한다. Q/K/V 저장과 출력은 BF16로 유지하고 MMA 입력 fragment와 probability만 FP16으로 변환하며 FP32로 누적한다. 전체 KV repacking allocation/copy는 추가하지 않았다. Python은 offline source 준비에만 사용한다.

범위 밖 또는 nonfinite 입력은 retained device status에 bit32를 OR한다. abs(x)<2^-14는 signed zero로 명시적으로 변환한다. 별도 수치 profile이며 기존 BF16 모델 품질을 보장하지 않는다. 오류 출력은 소비하면 안 된다. 현재 native 진단 전용이고 serving/runtime에 연결하지 않았다.

## 검증

- RTX 4090 SM89, CUDA 13.0.88. GPU compute 경쟁 작업 없음, GUI 유지.
- 일반 868,032개 출력, mixed 850,752개 prefill 출력: 기존 FP64 oracle max-absolute tolerance 0.01 통과. 최대 관측 오차 0.000488714. Decode/inactive 출력 보호, invalid packet 후 recovery 통과.
- Q/K/V 각각 +1e5, -1e5, NaN, +Inf, -Inf를 주입한 15회 실행에서 모든 유효 case가 status32를 검출했다. Invalid packet case는 기존 status4를 유지했다.
- Mixed graph replay memcheck 0 errors, racecheck 0 errors/0 warnings.
- Native workspace 33,996→34,008 bytes. 모델 연결 시 Rust allocation 계약도 변경해야 한다.
- SM90a / SM100a compile 통과. 해당 GPU runtime은 장비 부재로 skip했으며 지원 완료를 뜻하지 않는다.
- 설치 dependency header tree 전후 동일. 생성 header hash `70452db62dc984d5be3155f97f5f7b369e4098e6617be938a0d2b8a705edcbb9` 고정.

[빌드 receipt](native/receipt.json), [검증 및 로그 hash](native/validation.json), [일반 입력](native/all.log), [혼합 입력](native/mixed.log), [memcheck](native/memcheck.log), [racecheck](native/racecheck.log).

첫 두 빌드는 default argument 순서 및 host 함수 내부 Params 타입의 CUDA template 사용 제한으로 실패했다. 인자를 앞으로 이동하고 Params를 namespace로 옮긴 v3가 위 검증 대상이다. 실패 로그는 원격 `/tmp/riley-opt-260912/prefill-half-register-v{1,2}/build.log`, 성공 원본은 `prefill-half-register-v3`에 보존했다.

## 다음 판정

합성 입력의 native correctness/safety 증거다. 변환 경계의 세부 수치 검사, full-model 자유 생성 및 자연어 NLL/KL, 상태 오류의 모델 출력 차단, graph identity와 workspace ownership 연결이 남아 있다. 기존 품질 gate를 유지한다. 고정 SmolLM2 Q128 이외 dispatch 경로는 검증하지 않았다.

모델 단계 후 변환 비용을 포함한 동일 조건 serving 비교로 진행한다. 새 serving 측정이 없어 vLLM 비교표를 갱신하지 않으며 throughput 개선이나 목표 달성을 주장하지 않는다. 기본 backend도 변경하지 않는다.

준비: `benchmarks/analysis/prepare_prefill_half_probe.py --source SOURCE --output FRESH_OUTPUT --headers PINNED_FLASHINFER_DATA --nvcc CUDA13_NVCC`. 실행: 생성 `probe all.bin`, `probe mixed.bin --prefill-only`; fault 예시 `probe fault.bin --all q 1e5`. Fault 모드는 all-query 전용이다.
