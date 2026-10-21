# Gate/Up·Down·decode KV·attention v16A 구조 후보 실험

2026-10-21 KST. 직전 채택 코드는 v15C(`57502e7a0`)다. 큰 변경 후보를 먼저 실제 GPU에서 선별했다. **이 checkpoint에서는 runtime을 추가 채택하지 않는다.** QK register-only reduction과 PV multi-warp CTA를 전체 모델·정상 C16/C32 비교에 넘긴다. 나머지는 느리거나 정확도에 실패해 제외한다. 구성요소 시간은 서버 처리량이 아니다.

## 변경 후보와 결과

| 후보 | 유지한 계약 / 변경 | 실제 결과와 판정 |
|---|---|---|
| Gate/Up 긴 K mainloop | 원래 두 K32 chain·BF16 SiLU 경계, K128/256 타일로 barrier 상각 | 생성 finite7R×6상태에서 일치. R16 K128 81.754→80.897µs, K256 100.275µs. 작은 이득/악화로 채택 안 함 |
| Down 긴 K chain mainloop | native4 FP32 partial과 BF16 split/reduce 순서, K64/128 pipeline·CTA 분배 재설계 | 바이트 일치. R16 48.529→50.404µs 또는48.527→48.126µs, R32 54.393→78.231µs 또는54.483→58.940µs. 채택 안 함 |
| KV 행 병렬화 | 원래32 FP32 chain을4 registers와8 lanes로 보존 | 일치했지만 R16 20.654→67.379µs, R32 35.203→127.255µs. 제외 |
| KV 가중치 전치·출력 축 coalescing | cold transpose, 원래32-chain tree | 두 타일 모두 일치하나 느림. R32 약35.09→47.18~52.64µs. 제외 |
| KV FP32 exact expansion | BF16 비트를 정확히 FP32로 확장해 내부 반복 변환 제거, 공개 dtype/누적 순서는 유지 | 두 K 타일 모두 일치하나 input expansion 포함 R32 약35.10→42.52~42.89µs. 메모리 증가를 도입하지 않음 |
| KV zero-inserted Tensor Core chain | BF16에 정확히 표현되는 입력으로 한 product씩 native chain을 재현하려 한 연구 | R2/state2에서 BF16 원소1개 불일치. 즉시 탈락. 정확도 허용치 변경 없음 |
| PV GQA4 공유 확대 | 같은2-chain/WMMA 누적,4 query heads가 V를 공유 | 37,056 조합 일치하나 대표 R32/T213 합계 약80.55→103.41µs. 제외 |
| mixed Gate/Up 단일 weight tile | decode와 prefill을 다른 warp/fragment에 배치해 각 native family를 유지 | 수정 후210 조합 일치. 큰 CTA·scratch/등록 pressure 비용으로 대체로 느림. 아래 상세 |
| PV multi-warp CTA | 각 warp가 기존 depth tile·GQA·reduction을 그대로 소유,4 warps/CTA | 37,056 조합 context bytes 일치. 전체 모델 검증 후보 |
| QK register reduction + PV multi-warp | native serial16 또는 tree8 순서 유지, shared staging/barrier 대신 warp shuffle | 37,056 조합 context bytes 일치. 전체 모델 검증 후보 |

첫 Gate/Up·Down probe의 reference는 추출한 native custom kernels다. 실제 채택 경로가 R17 이상에서 사용하는 merged Lt나 작은 R의 모든 fallback을 대신하는 비교가 아니다. 생성 finite weights/activations로 검사했으며 checkpoint activation을 사용한 전체 모델 증명이 아니다. 큰 K tile·weight 전치·더 많은 공유가 이득이라는 가정으로 코드를 채택하지 않았다.

## mixed MLP 결합에서 확인한 수치·실행 제약

cuBLAS 원래 separate Gate/Up과 비교한6종 finite 상태에서 M16은 두 K32 chain, M64/115/128/129/130은 한 chain 구현과 일치했다. 두 체인을 한 체인으로 일괄 바꾸면 BF16 원소가 달라졌다. M144/160은 이 단순 family 후보들과 달랐으므로 지원 범위를 넓히지 않는다. 이 관측은 모든 가중치·shape에 대한 증명이 아니다.

이에 decode prefix를16행 경계로 padding하고 prefill 행들은 원래 한-chain 경로로 계산해, 동일 weight tile을 함께 사용하는 세 가지 큰 구조를 검사했다. 행당 warp pair, 출력 열당 warp, 큰 N/R/K tile의 asynchronous pipeline이다. 처음160 padded-row prototype은 실제 rows만 확인해 padded extent161 이상에서 출력 한 행을 놓쳤다. 이 시도는 실패로 보존했다.192행 capacity의 r2와 static fragment-index r3는 전체210 조합을 다시 통과했다.

그럼에도 prefill128+decode16의 기존 약200µs가 row-warp 약468µs, column-warp 약238µs, 큰 pipeline 약298µs였다. 일부 M129/130에서 column-warp가 빠른 값도 있었지만 전체 범위를 개선한 결과가 아니다. 큰 pipeline의165,376-byte shared allocation은 A100에서 한 CTA/SM 수준의 자원 제한을 만든다. 이 값과 ptxas의 register/spill 기록은 코드·빌드 근거이며, DRAM/L2/SM hardware counter 측정은 아니다. GPU 시간만 보고 메모리 병목을 확정하지 않는다.

## 선택한 attention 후보: 세 교차 반복의 중앙값

QK/softmax/PV 전체를 CUDA Graph64×8 replay로 측정했다. 정상 HTTP 측정과 분리했고 GPU7에서 수행했다. 단위µs. 각1~32 R, T24~216,6종 finite 상태 중3종은 요청별 서로 다른 길이이며 슬롯과 block ID도 순열로 배치했다. 총37,056 조합의 최종 context bytes를 검사했다. QK/probability의 별도 원소별 비교를 추가한 검사는 아니며, softmax 코드는 변경하지 않았다.

| T | R16 기존→후보 | R32 기존→후보 |
|---:|---:|---:|
| 128 | 31.218→27.095 | 47.795→39.982 |
| 129 | 27.868→24.776 | 45.010→38.298 |
| 144 | 33.368→29.685 | 50.650→41.489 |
| 159 | 30.579→27.116 | 54.370→43.946 |
| 192 | 41.493→34.684 | 69.916→53.566 |
| 213 | 38.097→33.534 | 80.560→59.896 |

PV는 별도 V staging을 추가하지 않는다. 기존 warp의 FMA/WMMA와 private scratch를 그대로 두고4개 depth tile을 한 CTA로 묶는다. SIMT QK의 serial16은 여전히 같은 순서의15개 FP32 add를 수행하고 tree8은 원래 tree를 warp shuffle로 전달한다. 입력 dtype·softmax·BF16 store 경계·supported context 범위는 유지한다. [검증 전 재현 패치](../benchmarks/candidates/native-attention-v16a/prototype.patch)는 v15C 기준이다.

## 자원·재현과 다음 gate

모든 finite GPU 작업은 원래 holder controller를 통해 깨끗한 물리 GPU7에서 수행했다.13개 실제 exe/UID/environment/start-tick 관측 receipt를 보존한다. 숫자 실패한 TF32와 초기 padded-row 경계 실패도 성공으로 바꾸지 않는다. GPU6의 허용된 작은 외부 작업은 보존했고 원래 GPU6/7 holder PID2027914/2027915와 context는 해제·재시작하지 않았다. 모델·빌드·스크립트·scratch는 EFS다.

- [압축 원자료](../benchmarks/evidence/20261021-mlp-decode-v16a.zip), [자원·선택·시간 감사](../benchmarks/evidence/20261021-mlp-decode-v16a/component-audit.json), [파일 검증](../benchmarks/evidence/20261021-mlp-decode-v16a/local-archive-verification.json).
- EFS `/mnt/efs/nlp/psyche/riley-mlp-decode-v16a-20261021`: 13개 ELF는 heavy SHA index로 연결하며 raw 로그·실패·컴파일·명령·CUDA source는 압축에 포함한다.
- 외부 설계 참고: [CUTLASS hierarchical tiling/pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html), [PTX fragment/reduction 계약](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1684). 실제 이득은 Riley에서 측정했으며 라이브러리의 발표 성능을 전용하지 않는다.

다음 gate는 v15C 기준 독립 source/빈 target 빌드, 원래 native/HF full-vocabulary BF16·mixed·GPU 계약, GPU7에서 C16/C32 각각5회 before/after/stock 정상 비교다. 이 checkpoint의 구성요소 이득으로 README 최신 서버 수치를 대체하지 않는다. 계획의 큰 처리량 목표가 달성됐다는 판정을 하지 않는다.
