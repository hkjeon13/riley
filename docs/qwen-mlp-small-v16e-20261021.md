# Mixed Gate/Up small-tile v16E 구조 실험

2026-10-21 KST. **바이트 검사는 통과했으나 성능이 불리해 미채택한다.** v16A의 큰 CTA 후보를 작은 shared-memory pipeline으로 재설계했다. 기존 v15C와 선행 v16C의 최적화된 prefill 경로를 함께 대조했다. 구성요소 시간이며 서버 성능이 아니다.

## 구현과 수치 범위

decode의 두 K32 누적 chain과 prefill의 한 chain을 한 weight-tile pipeline에서 유지했다. K32 double buffering과160/176 padded-row tile을 사용한다. 32-wide K tile에서도 전역 K 위치의 parity로 native decode chain을 선택한다. 기존 큰 후보165,376B 대신35,840/38,400B shared memory다. padding후 전체extent로 tile을 선택해 이전 A의 padded-row 경계 오류를 반복하지 않는다.

생성 finite weights/activations에서 decode R2/4/8/12/16/24/30/32 × prefill64/115/128/129/130 ×6상태=240구성을 검사했다. R2~16 기준은 현재 native fused kernel이며 R17 이상 기준은 이미 채택된 merged Lt31/15/17/0/0/0/0이다. 과거의 느린 R17+ custom kernel을 현재 대조로 사용하지 않았다. 원래경로와240구성 일치 후, 같은240구성을 v16C의 single-prefill merged Lt21 + SiLU와도 대조했다. 두 기준과 gated-product BF16 bytes가 일치했다. 전체 모델/native-HF logits 증명은 아니다.

## 최적화된 v16C 경로와 직접 대조

단위µs. CUDA-event 한 번의 구성요소 screen이며5회 서버 중앙값이 아니다. Down과 prefix MLP Graph 전체 비용은 포함하지 않는다.

| Prefill | Decode | v16C Gate/Up+SiLU 부분 | 새 fused 부분 | 변화 |
|---:|---:|---:|---:|---:|
| 115 | 16 | 196.410 | 204.551 | +4.14% |
| 115 | 32 | 198.665 | 211.652 | +6.54% |
| 128 | 16 | 197.555 | 210.764 | +6.69% |
| 128 | 32 | 200.083 | 219.085 | +9.50% |
| 129 | 16 | 207.818 | 211.340 | +1.69% |
| 129 | 32 | 209.832 | 229.218 | +9.24% |
| 130 | 16 | 208.131 | 211.370 | +1.56% |
| 130 | 32 | 210.003 | 229.117 | +9.10% |

원래 separate prefill의129/130행 대비로는 약14~23% 감소가 보였다. 그러나 그 비용은 v16C의 merged Lt21 경로도 이미 줄인다. 최적화된 대조에서는 일부 R2/4에서 작은 이득만 남고 다른 많은 구성에서 느렸다. 기존 경로를 놓친 큰 개선율을 채택 근거로 사용하지 않는다.

ptxas는160행164 registers,176행166 registers 및 spill0을 기록했다. shared-memory 한도만 낮추는 것으로 충분하지 않았다는 시간 결과다. 이 빌드 정보가 DRAM/L2/SM hardware counter나 특정 원인의 인과 증명은 아니다. 전체 모델 빌드나 C16/C32 서버 비교로 넘길 만큼 넓은 이득이 없어 runtime에는 추가하지 않았다.

## 보존

GPU7의2개 직접 실행 observer receipt, 전체240+240 구성·세 대조 시간·CUDA source·빌드·선행 queue 교정 기록을 보존한다. 모두 original holder controller 안에서 수행했으며 GPU6 외부 작업은 중단하거나 새로 허용하지 않았다. 모든 scratch는 EFS다. [압축 원자료](../benchmarks/evidence/20261021-mlp-small-v16e.zip)는33,412B/33entries, SHA256 `13799b9120f9e30df3e37c2453f861f18941971dbca44b96399e27062b6f151f`다.

[전체 구성·판정](../benchmarks/evidence/20261021-mlp-small-v16e/component-audit.json) · [로컬 CRC/SHA/size 검증](../benchmarks/evidence/20261021-mlp-small-v16e/local-archive-verification.json). ELF는 archived heavy index로 EFS에 연결한다.
