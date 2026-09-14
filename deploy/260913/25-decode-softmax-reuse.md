# PR batch: decode softmax를 출력 차원 block 사이에서 재사용

상태: 코드에서 중복 계산을 확인한 설계 후보. 구현·GPU 측정 전이며 승격 근거가 아니다. C64 matrix의 frozen binary와 실행에는 반영하지 않는다.

## 근거와 범위

`kernels/src/decode_gqa_attention_v50.cuh`의 `independent_values`는 grid.x=8이고 각 block은 head의 출력 8차원을 맡는다. maximum, exponential, BF16 probability, denominator 계산은 출력 차원에 의존하지 않지만 각 block에서 반복한다. 이는 소스에서 확인한 중복이며 그 비용이 GPU 시간을 지배한다는 증거는 아직 없다.

QK score 계산은 그대로 두고 다음 세 변경을 하나의 native 실험 batch로 구현한다.

1. 요청/head당 softmax 준비를 한 번 수행한다. 원래 역순 128-token tile, warp max reduction, lane별 denominator 합산 순서와 BF16 probability 변환을 유지한다.
2. 준비 결과로 token별 BF16 probability, tile별 alpha, 최종 lane-class별 inverse denominator를 bounded scratch에 저장한다. Values kernel의 8개 출력 block이 이를 읽고 기존 순서로 accumulator rescaling 및 K16 MMA를 수행한다. Probability를 전역 정규화하거나 prefix/suffix attention state를 합치지 않는다.
3. 기존 두-kernel 경로와 새 세-kernel 경로를 opt-in으로 유지하고 scratch 수명·CUDA graph capture/replay·활성 row mask를 연결한다. 실행 shape에 따른 승격은 benchmark 후 결정한다.

최대 32 rows × 9 heads × 4096 tokens의 BF16 probability는 2,359,296 bytes다. Tile alpha 및 denominator 저장 공간이 추가된다. 추가 launch, probability write/read, 기존 score 읽기의 제거량, L2 hit 영향을 함께 비교해야 한다. 원본 kernel도 같은 값을 여러 번 읽으므로 논리적 byte 수를 HBM traffic으로 단정하지 않는다.

## 수치 계약

Denominator는 warp lane 전체가 같은 값이라고 가정하지 않는다. 원본의 `t=lane%4`별 local sum과 마지막 xor 2/1 reduction을 그대로 재현하고 소비 lane에 맞는 inverse를 제공한다. 첫 tile의 `maximum=-inf`, partial tile의 zero padding, alpha 계산·곱셈, mask를 유지한다. Exponential 근사 교체나 alpha=1 생략은 이 batch에 섞지 않는다.

## 검증과 기각

- Native 원본 oracle과 출력 bitwise 비교: context 1/15/16/17/63/64/65/127/128/129/4095/4096, 활성 rows 1/8/16/32, 비연속 physical pages, shared pages, 극단 finite score 및 tail padding. Sanitizer는 bounded case로 실행한다.
- 실제 모델 greedy/reference 일치, stop/cancel/recovery, graph replay 및 scratch lifetime 확인. 결과 차이가 있으면 기준을 완화하지 않고 원인을 조사한다.
- 대표 shape에서 QK+prepare+PV 전체 시간을 측정한다. Softmax 준비 단독 speedup을 승격 증거로 사용하지 않는다.
- Native 이득이 확인되면 prior/candidate/vLLM 동일 workload의 shared/unique serving 비교를 수행한다. C8의 추가 launch 회귀와 C32/C64 처리량·P95/P99를 포함한다. 이득이 없으면 scratch/launch 비용과 제거된 instruction 비용을 분석하고 기본값은 원본으로 둔다.

Rust → C ABI → CUDA만 사용한다. SM89에서 검증 가능한 batch이며 Hopper/Blackwell 전용 명령을 요구하지 않는다. 이후 하드웨어별 pipeline은 별도 검증한다. Rollback은 새 opt-in을 끄고 원래 enqueue 경로를 선택하는 것으로 가능해야 한다.
