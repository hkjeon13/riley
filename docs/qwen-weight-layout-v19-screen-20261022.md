# Qwen projection 가중치 배치·backend 독립 실험 v19

2026-10-22. 연산 검증이 끝난 체크포인트이며 HTTP 결과는 아직 없다. 기존 v15C 실행 경로와 원래 native/HF 정확도 기준을 유지한다. 아래 수치를 C16/C32 처리량 개선율로 해석하지 않는다.

## 실제 바꾼 구조

Gate/Up을 원래 `[channel][K]`에서 `[K64 tile][N32 tile][Gate32|Up32][K64]`로 cold permutation했다. 같은 K 단계를 실행하는 CTA들이 인접한 출력 타일을 읽도록 바꾼다. 원래 두 FP32 누산 체인, 합산 순서, projection/SiLU/product의 BF16 경계는 그대로다. tile-major와 K-major를 별도로 비교했다. Down도 원래 split/reduction 순서를 보존한 tile-major/K-major와 W4/W8 backend를 시험했다.

별도 실험은 가중치 전치 후 cuBLASLt NN 경로다. 설치된 Lt `120405`, workspace 4MiB 이하, 반환 heuristic 최대64개와 swizzle0/1, Down splitK2/reduction4를 검사했다. 실제 반환·실행 후보는70개다. 6개 상태에서 BF16가 모두 같은 후보4개는 전부 기존 경로보다 느렸다. 그 외66개는 정확도 기준을 통과하지 못했다. heuristic 순번은 안정적인 backend 식별자가 아니며 서버 설정으로 채택하지 않는다.

이 설계는 [CUTLASS의 TensorOp layout 원칙](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/functionality.md)과 [cuBLAS12.4의 layout/algorithm API](https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html)를 참고했다. CUTLASS를 서버 의존성으로 추가한 실험은 아니다.

## 연산 결과

각 시간은 같은 프로세스에서 대조/후보 순서를 번갈아 실행한 5쌍의 중앙값이다. Graph당32회 연산, 3회 warm replay 후 8회 replay를 측정했다. cold weight permutation은 측정에서 제외했다. 추가 입력 packing은 없다. 가중치 사본·cold 비용은 서버 평가에서 별도로 필요하다.

| 실제 Qwen 레이어0 · 상태0 | 원래 µs | K-major µs | 지연 변화 |
|---|---:|---:|---:|
| Gate/Up R2 | 80.977 | 77.452 | -4.35% |
| Gate/Up R8 | 82.188 | 78.209 | -4.84% |
| Gate/Up R16 | 83.138 | 79.256 | -4.67% |
| Gate/Up R32 | 83.166 | 87.331 | +5.01% |

표의 변화는 중앙값끼리의 비율이다. 전체 pair별 비율 통계는 JSON에 별도로 보존했다. 실제 레이어0/17/35의 모든 R2..16에서 Gate/Up K-major 지연은 paired median 기준4.28~5.07% 감소했다. Down K-major W4는 같은 범위에서3.62~6.23% 느렸다. Gate/Up R17..32 및 Down은 서버 후보에 넣지 않는다.

- 합성 가중치: R2..32 ×6상태 ×6경로 =1,116건, BF16 bit mismatch0.
- 실제 체크포인트 가중치: 레이어0/17/35 ×R2..32 ×6상태 ×6경로 =3,348건, mismatch0. 각 safetensors shard 전체 SHA와 manifest 크기를 검증하고 원본 BF16를 추출했다. 입력 activation은 합성 finite 상태이며 실제 실행 activation이라고 주장하지 않는다.
- 추가 타일·stage 실험: 같은3레이어 ×R2/8/16 ×6상태 ×5경로 =270건, mismatch0. W4/S3는 기존보다 약3.6% 빠르지만 W4/S2보다 느렸고, W8/S1·W8/S2·W4/S1은 모두 원래보다 느렸다. 수치 검사는6상태 전부, 시간은 상태0이다. NN은 수치검사 마지막 상태5에서 paired timing했다.
- 처음 tile-major revision은 R17 Down W8/R32의 shared-memory launch 실패로 중단했다. 이 부분 실행은 통과 건수로 합산하지 않는다. 후속 v2에서 W8/R16과 Y grid로 고쳤고 각 launch 오류를 즉시 검사했다. 원본 실패 로그도 남겼다.

## 서버 통합 후보와 남은 검증

가중치별 packed allocation을 cold upload에서 한 번 만들고 모든 배치 shape가 공유하도록 구현 중이다. 기존 row-major 사본을 보존해야 prefill과 R17..32의 수치 경로가 그대로 유지된다. 파생 사본의 3GiB 예산은 Qwen36레이어 중32레이어를 허용한다. 나머지4레이어는 기존 경로다. packed allocation identity와 command-batch lease가 Graph replay에도 포함돼야 한다. 추가 메모리만큼 빠를 것이라는 가정으로 채택하지 않는다.

독립 source build, 원래 full-vocabulary native/HF 기준, 원래 GPU 계약/120 alignment 사례, 새 packed span/identity 검증을 완료한 뒤 C16/C32의 fresh matched v15C/후보/vLLM 비교를 진행한다. 현재 기본 실행 경로와 README의 기존 HTTP 수치는 바뀌지 않았다.

## GPU 홀드·근거

original holder6/7 PID2027914/2027915를 종료하거나 재시작하지 않았다. 합성 및 NN 연산 화면은 GPU7, 실제 가중치 및 추가 타일 검증은 GPU6에서 실행했다. 완료된4개 observer의 sample에서 외부 GPU PID는0건, whole-device peak는1361MiB 이하다. 이는 샘플 기반 관측이며 샘플 사이의 상태를 증명하지 않는다. 뒤이어 GPU7의 다른 계정 작업이 발견돼 admission guard가 실제 가중치 실행을 차단했고 GPU6에서 새로 실행했다. 그 실패는 성능 표본으로 쓰지 않는다. 다른 계정 작업에는 조작하지 않았다.

[전체 원본·소스 archive](../benchmarks/evidence/20261022-weight-layout-v19-screen.zip) · [전체 연산 통계](../benchmarks/evidence/20261022-weight-layout-v19-screen/component-summary.json) · [shard/weight SHA](../benchmarks/evidence/20261022-weight-layout-v19-screen/weight-extraction.json). Archive181,534B,106entries, SHA256 `74d726a40c82c0bdc734da73ce98fa64cc498c7a785fababd1821ad49b985d02`.
