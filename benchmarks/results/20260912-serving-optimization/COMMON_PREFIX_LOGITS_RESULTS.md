# 공통 prefix의 HF FP32/BF16 logits 진단

**독립 HF 실행에서는 출력 index 8의 BF16 최고값이 동률이지만, index 29에서는 FP32와 BF16 모두 token 638이 단독 최고값이었다.** 따라서 이번 결과를 모든 serving 출력 차이가 동률 때문이라는 설명으로 일반화할 수 없다. 실제 vLLM graph logits는 아직 캡처하지 않았고 strict serving 정답 기준은 유지한다.

같은 모델 파일과 고정된 실제 input ID history를 새 FP32/BF16 프로세스에 전달했다. 실행은 **HF eager, cache off, teacher-forced B1**이다. 재토큰화나 자유 생성을 하지 않았으며, P128 뒤에 동일한 앞선 출력 8개 또는 29개를 붙인 길이 **136 / 157**의 입력을 사용했다. 출력 index는 0부터 센다. 각 경우 전체 **49,152개 logits**, log-probabilities와 첫 layer hidden tensor를 저장했다. 이것은 vLLM의 batching·KV cache·실제 graph 실행을 재현한 결과가 아니다.

| Output index / 입력 길이 | 실행 | 최고 ID | 최고 logit | Top1−Top2 gap | 이전 C1 reference ID |
| --- | --- | --- | ---: | ---: | ---: |
| 8 / 136 | HF FP32 | 2341 | 18.512554169 | 0.036720276 | 1443 |
| 8 / 136 | HF BF16 | 1443, 2341 동률 | 18.625 | 0 | 1443 |
| 29 / 157 | HF FP32 | 638 | 19.806793213 | 0.149744034 | 253 |
| 29 / 157 | HF BF16 | 638 | 20.125 | 0.25 | 253 |

Index 8에서 FP32의 다음 후보는 1838(`18.475833893`), C1 reference 1443은 `18.464542389`였다. BF16에서는 1443과 2341이 `18.625`로 동률이며, 분석기의 동률 시 낮은 ID 규칙은 **1443**을 선택한다. 이것은 새 HF 배열에서 확인한 동률이지 이전 vLLM 응답 내부의 동률을 확인한 것이 아니다.

Index 29에서는 이전 C1 reference **253**보다 **638**이 두 dtype 모두 높다. 253의 logit은 FP32 `19.657049179`, BF16 `19.875`다. 같은 history에서도 독립 HF 실행과 이전 C1 reference의 선택이 다르므로, 새 FP32 결과를 과거 canonical oracle로 소급하거나 이 결과만으로 높은 concurrency의 vLLM 응답이 잘못됐다고 판정하지 않는다.

## 증빙과 남은 확인

[분석 JSON](common-prefix-logits-analysis/analysis.json)의 `analysis_complete=true`, `diagnostic_only=true`, `actual_vllm_graph_logits_captured=false`, `acceptance_gate_changed=false`를 확인했다. 네 개 전체 logit 배열의 해시·49,152개 finite 값·고정 input history·argmax·gap도 로컬에서 재확인했다. 개별 dtype의 새 worker는 종료·backend 정리 receipt를 남겼다. 설치된 dependency/native library 정보는 이번 capture의 증빙이며 과거 기준 환경과 동일하다는 주장이 아니다. 과거 serving·reference 파일은 분석의 명시된 provenance 경계를 따른다.

- [FP32 capture](raw/common-prefix-fp32/completion.json), SHA256 `861f733b800c6bbbef97943b614e5f6b008bb909da31d999ddcc8c0a7e92423b`.
- [BF16 capture](raw/common-prefix-bf16/completion.json), SHA256 `22ed4625534d6c9029cc0059d3bf1e24775eabf9ff580e7fb6acf428f19ed64a`.
- [전체 분석](common-prefix-logits-analysis/analysis.json), SHA256 `795f2678f014224dc90be0b7cc0af72a693e0dc5b624110bf588d095536d7225`.
- 각 49,152행의 [index 8 vocabulary](common-prefix-logits-analysis/common-prefix-136-output-08.vocabulary.tsv), [index 29 vocabulary](common-prefix-logits-analysis/common-prefix-157-output-29.vocabulary.tsv).

다음 확인은 같은 고정 history와 실제 serving 조건에서 **vLLM graph의 전체 logits를 직접 수집**하고 source/runtime/batching/KV 조건에 묶어 비교하는 것이다. 이전 singleton logprob만으로 전체 분포나 동률을 추정하지 않는다. 아직 accuracy tolerance를 정하거나 비교 정답을 변경하지 않았으며, 이 진단에는 성능·kernel 원인·높은 concurrency 정확성 판정이 없다.
