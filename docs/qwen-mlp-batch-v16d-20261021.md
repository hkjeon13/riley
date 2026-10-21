# Coalesced prefill Gate/Up v16D 구성요소 실험

2026-10-21 KST. **여러 prefill 요청의 Gate와 Up을 한 번의 merged GEMM으로 실행하는 경로를 전체 모델 검증 후보로 선택했다.** 구성요소 개선을 서버 처리량으로 환산하거나 runtime 채택으로 간주하지 않는다.

## 구조와 원래 수치 기준

실제 Qwen3-4B layer0 merged Gate/Up checkpoint weight와 finite activation seed의6상태를 사용했다. M131..576 ×6=2,676구성에서 원래65..130행 native 그룹별 Gate/Up 결과와 BF16 bytes가 일치했다. 기존 coalesced dispatcher가 실제 선택하는 경로와도 불일치0이다. N19456/K2560의 merged Lt21/20/11/0/0/0/0 한 호출로 Gate·Up 양쪽을 계산한다. SiLU, Down, Rust dispatch와 Graph/HTTP 비용은 아래 GEMM 시간에 포함하지 않는다.

기존 dispatcher는 M161..576을 이미 합치지만 Gate/Up은161..256 및513..576에서 classic separate GEMM,257..512에서 두 pinned Lt21 호출이다. 초기 screen은 항상 두 Lt21 호출을 대조로 삼았으므로 일부 행의 개선율을 실제 경로 대비로 볼 수 없었다. 원본을 보존하고 actual-route 대조를 추가했다. Down도 기존에 이미 coalesced+Lt6이므로 이 실험을 새로운 Down batching 개선으로 표현하지 않는다.

## 실제 dispatcher 대조

µs. 각 M161..576의state0에서 CUDA-event 한 번씩 총416측정이다. 반복 서버 중앙값이나 hardware counter가 아니다.

| Rows | 기존 실제 경로 | 새 merged GEMM | 시간 변화 |
|---:|---:|---:|---:|
| 161 | 137.979 | 114.674 | -16.89% |
| 192 | 138.258 | 115.268 | -16.63% |
| 256 | 139.021 | 119.981 | -13.70% |
| 257 | 200.241 | 164.498 | -17.85% |
| 384 | 207.754 | 170.430 | -17.97% |
| 390 | 205.024 | 197.694 | -3.58% |
| 512 | 210.898 | 203.900 | -3.32% |
| 513 | 271.916 | 255.987 | -5.86% |
| 576 | 275.167 | 259.418 | -5.72% |

416행 모두 이 screen에서2.34~18.75% 짧았다. M161..256의 중앙 시간 변화는−16.04%,257..384는−17.80%,385..512는−3.43%,513..576는−5.64%다. 단일 구성요소 결과의 범위이며 전체 처리량의 동일 개선을 보장하지 않는다. 후속 v16F/G에서 cold scratch 용량, Graph 수명, 원래 HF/native 정확도와 정상 C16/C32를 독립 빌드로 검증한다.

## 실행·보존

GPU7에서 original holder controller/직접 실행 observer를 통해 실행했다. GPU6 외부 작업과 두 original holder를 유지했다. 모든 입력·스크립트·scratch는 EFS다. 앞선 launcher의 missing observer marker, upload 완료 전 실행 및 missing holder import 실패도 보존했다. 마지막 문제는 compiler 실패가 아니며, 빌드된 ELF를 수정 없이 재사용했다. 입력 proof에는 실제 weight/seed SHA와 사용하지 않은 Down CLI 인자를 구분했다.

[원자료](../benchmarks/evidence/20261021-mlp-batch-v16d.zip):77,986B/61entries, SHA256 `95d59ac012698276cdeedb565b07b0877f9ff9a863268ae6f1c0f36ca8e8f4f1`. 초기 비교와 actual-route 대조를 모두 포함한다. [전체 행별 결과](../benchmarks/evidence/20261021-mlp-batch-v16d/routes-analysis.json) · [판정](../benchmarks/evidence/20261021-mlp-batch-v16d/component-audit.json) · [로컬 CRC/SHA 검증](../benchmarks/evidence/20261021-mlp-batch-v16d/local-archive-verification.json). ELF는 heavy index로 EFS에 연결한다.
