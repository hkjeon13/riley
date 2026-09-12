# 기본 vLLM compiled 출력 차이 분리 검증

상태: **원인 범위 축소, 출력 정합성 미해결, 성능 측정 0회.**
정식 서버 후보 `9b53ffa14cea7c066fda8eff65b977c860b12afb`는 변경하지 않았다.

## 같은 입력에서 재현한 결과

SmolLM2 동일 checkpoint/tokenizer, `Hello` 128회, greedy 출력 32개를 유지했다.
각 vLLM 설정은 새 프로세스로 실행하고 logprobs 없음/5 두 정확성 요청을 보냈다.

| 변경한 설정 | 기본 compiled 출력과 첫 차이 (0-based) | eager 출력과 일치 |
|---|---:|---|
| 기본 설정 재실행 | 없음 | 아니오 |
| CUDA graph만 NONE | 없음 | 아니오 |
| native custom-ops 전체 | 29 | 예 |
| native SwiGLU만 선택 | 29 | 예 |
| native RoPE만 선택 | 없음 | 아니오 |
| native RMSNorm만 선택 | 없음 | 아니오 |

모든 설정에서 logprobs 요청 여부는 결과를 바꾸지 않았다. 따라서 이 입력의
compiled/eager 차이는 CUDA graph 자체나 logprobs 요청으로 설명되지 않는다.
컴파일을 유지한 상태에서 SwiGLU 경로 하나의 변경으로 eager 결과를 재현했다.
이는 해당 설정 변경의 인과 증거이며, 다른 연산의 수치 차이가 없다는 증거는 아니다.

## 실제 텐서의 반올림 경계

- HF layer0 MLP의 gate/up projection 196608개 원소를 사용했다.
  compiled SwiGLU와 eager는 51712개 원소에서 달랐다(최대 절댓값 0.00390625).
  compiled는 FP32 SiLU와 곱셈 후 한 번 BF16 변환한 계산과 정확히 같았고,
  vLLM native SiLU/product는 BF16 중간 결과를 사용하는 eager와 정확히 같았다.
- embedding 입력 73728개 원소에 대해 compiled RMSNorm과 HF eager는
  19072개에서 달랐다(최대 절댓값 0.00390625). compiled와 vLLM native
  RMSNorm은 이 텐서에서 정확히 같았다.
- 텐서 검사는 실제 모델 입력을 사용한 개별 표현식 비교다. 전체 vLLM compiled
  그래프의 모든 레이어 중간값을 추출해 대조한 결과는 아니다.

## 채택하지 않은 수정

격리된 Riley 사본에서 norm 단독, SwiGLU 단독, 두 변경 조합을 대조했다.
어느 조합도 기본 vLLM의 32개 출력 전체를 재현하지 못했다.
별도 HF 진단에서도 FP32 residual 유지 및 norm/SwiGLU 조합을 확인했으나
기본 vLLM 출력과의 완전 일치는 얻지 못했다.

`riley-*.log`의 graph/eager 일치는 **동일하게 바꾼 진단 사본 내부의 일치**다.
원래 HF 계약 또는 기본 vLLM과의 일치를 뜻하지 않는다. 진단용 SwiGLU 변경은
SiLU 단계를 raw gate 전달로 바꾸고 product 단계에서 activation을 계산하므로
독립 SiLU primitive 계약도 보존하지 않는다. 해당 패치는 실험 기록이며, 원래
작업 트리와 정식 후보에는 반영하지 않았다. 일반 연산 검증으로 사용할 수 없다.

## 남은 작업

1. 동일한 고정 prefix에서 Riley와 기본 vLLM의 레이어/연산 중간값을 대조해
   남은 차이를 특정해야 한다. 여러 반올림 변경을 조합해 출력 토큰만 맞추는
   방식으로는 정식 수정의 타당성을 입증할 수 없다.
2. 해당 차이를 수정하더라도 기존 HF 정합성을 보존하거나 명시적인 별도 수치
   프로파일로 구분하고, 전체 decode와 HTTP lifecycle을 다시 검증해야 한다.
3. Blender PID 4007728, 201778, 501891이 여전히 GPU를 사용한다.
   v2 preflight는 GPU memory 743 MiB > 256 MiB에서 실패한다. 해당 세션을
   종료하지 않았으며 독점 검사는 미통과다.

`summary.json`은 원본 배열, 비교 결과와 증거 해시를 포함한다.
`provenance.json`은 설치된 vLLM 소스 파일 해시와 정식 후보 보존 상태를 기록한다.
측정 준비 gate나 기본 vLLM 비교 기준을 변경하지 않았고 성능 캠페인은 실행하지 않았다.
