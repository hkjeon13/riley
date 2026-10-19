# A100 Gate/Up 배치 후보 검증

A100-SXM4-40GB의 GPU3에서 별도 검증했다. 환경은 SM80/108SM, CUDA runtime12040, cuBLAS120405, Rust1.85이다. HF 기준은 같은 장치의 torch2.13+cu129 / transformers5.19 eager cached-forward로 새로 생성했다. RTX4090 결과와 합산하지 않는다.

기존 M1 cuBLAS는 Tensor Core의 `ampere_bf16_s16816gemm_bf16_128x64_sliced1x2_ldg8_f2f_stages_64x6_tn`을 선택했다. SM89의 16-lane SIMT 순서를 옮긴 후보는 Gate10개/Up8개 값이 달랐다. CPU FFMA 재현값은 그 후보와 전 값이 일치했고, 새 A100 M1도 이전 A100 M1과 일치했다. 해당 포트는 미채택이다.

새 후보는 각 Gate/Up projection에서 요청 행을 최대16개씩 묶어 같은 장치의 M1 연산 결과를 유지한다. 총 배치32는 projection당 두 번의 GEMM으로 처리한다. Gate와 Up의 출력 열 병합은 Mistral에서 수치가 달라 별도 행렬을 유지했다.

## 구현과 실제 실행 검증

private v225는 actual164의 935개 고정 파일에 7개 파일 차이를 적용한 별도 빌드다. CPU518개 통과, 실패0, 일반 GPU검사40개 ignored이며 별도의 실제 GPU검사를 수행했다. 주 작업 소스935개 해시는 유지했다.

`bounded-gateup-sm80-v1`은 독립된 mode6이다. 기존 SM89 mode5의 보호 조건을 유지한다. 새 모드는 위 하드웨어/라이브러리와 검증한 Q4/Mistral 형상, 최대32개 활성 요청, 기존 workspace/MPS 설정에서만 허용한다. 기존 buffer/plan/stream 소유권과 completion 처리를 유지하고, 요청 행 메타데이터 캐시와 실제 배치1~32의 개별 pre/post Graph를 사용한다. Attention은 기존 경로를 유지한다.

| 모델 | fresh HF logits 행 | 개선 전후 수치 일치 | 검증한 실제 배치 Graph 키 | 배치32의 레이어당 Gate/Up GEMM 제출 | 전체 GPU 메모리 peak MiB 개선 전→후 |
|---|---:|---|---|---:|---:|
| Qwen3-4B | 104 | 전체 BF16 값 일치 | 1~32 | 64→4 | 14643→15901 |
| Mistral-7B-v0.3 | 104 | 전체 BF16 값 일치 | 1~32 | 64→4 | 20029→20925 |

각 모델에서 개선 전 C1 eager, 개선 전 C32 Graph/배치 축소, 개선 후 C1 eager, 개선 후 C32 Graph/배치 축소를 비교했다. 모든 저장 full-vocabulary 행이 일치했고, helper 내부에서 각 replica도 비교했다. 각 배치 키를 최소 세 번 실행해 warm/capture/변경 context의 첫 replay를 검사했다.

실제 캡처 중 호출한 cuBLAS의 N/K/요청 행 수/배치 범위/형식/상태/capture ID를 독립적으로 검사했다. Qwen3는 개선 전432개 대비 개선 후2304개 segment Graph와 동일 수의 첫 replay, Mistral은384개 대비2048개를 확인했다. 개선 후 Gate/Up의 요청 행 수는 최대16이고, 총 요청1~32 각각 레이어 수만큼의 post Graph가 존재했다. 전체 GPU 메모리는22GiB 한도 이내였다. 이 숫자는 GEMM 제출 및 Graph 검증이며 처리량 향상 배수를 뜻하지 않는다.

## 남은 검증

이 후보는 주 작업 소스에 아직 채택하지 않았다. 실제 API의 서로 다른 요청, 취소/재시도, 긴 context, 다른 기존 profile/model 회귀와 같은 장치 vLLM 비교를 완료해야 한다. 성능 측정 구간에는 진단 preload를 사용하지 않는다.

현재까지의 추가 projection 진단은 `v230-projection-eligibility-summary.json`과 `v232b-native-M1-Lt-summary.json`에 있다. 일반 GEMM의 행 수를 늘리면 일부 형상에서 값이 달라졌다. M1 형상을 유지한 Lt strided batch는 두 모델 K/V와 Mistral Down에서 전체1~32가 일치했다. 이 결과는 단일 레이어의 독립 진단이며 실행 경로에는 아직 반영하지 않았다. Qwen3의 현재 실제 QKV 경로는 N6144의 병합 M1이므로 별도 Query N4096의 성공을 현재 QKV 배치 지원 증거로 사용하지 않는다.

원시 증거와 독립 검사 결과는 `artifacts/multimodel-parity-20261017/`의 `v225-independent-build-audit.json`, `v226-controller-result.json`, `v228-controller-result.json`, 모델별 `*-v228-values.json`, `*-v228-independent-audit.json`에 있다. 서버 성능 측정과 배포·commit·push는 수행하지 않았다.
