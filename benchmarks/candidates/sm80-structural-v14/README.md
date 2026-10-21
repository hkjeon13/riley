# SM80 Qwen structural v14 — 미채택 실험

**D 후보를 기본 코드에 적용하지 않았다.** 정상5회 비교의 처리량 중앙값은 v13B 대비 C16 +0.76%, C32 -0.27%이며, 반복별2% 초과 tail 플래그도3개였다. 기본 코드는 검증된 v13B snapshot을 유지한다. [prototype.patch](prototype.patch)는 이 snapshot에 적용 가능한 실험 패치로만 보존한다.

실험 범위: AWS A100/SM80/108SM, CUDA12.4, cuBLAS/Lt120405, Qwen3-4B BF16, 기존 `owned-hf-eager-bf16-v1` + `native-prefill-sm80-v5` 경로. 정확도·성능·미채택 판정은 [결과 보고서](../../../docs/qwen-mixed-attention-structural-v14-20261021.md)에 기록한다.

- mixed iteration을 두 번의 물리 pass로 나누는 A/B 후보는 C16 성능 악화로 제외했다.
- 실험 D는 단일 batch를 유지한다. embedding report를 iteration 완료 때 검증하고, R1 지원 context는 전체 decode Graph를 사용한다. 지원 범위 밖에서는 기존 MLP piece fallback을 유지한다.
- BF16 Gate/Up·SiLU·multiply 경계를 유지하는 merged Lt recipe31을 R2~16에도 적용한다. weight 중복 비용은 v13부터 존재하며 이번 변경으로 새로운 dtype이나 tolerance를 도입하지 않는다.
- R24~32 dynamic attention은 QK를 독립 실행하고 softmax/PV를 결합한다. context24~216의 기존 native family에 한정한다. probability의 BF16 경계는 shared memory에서 유지한다. PV family에 따라4/2 query heads를 공유한다. R1~23과 적격성 밖 경로는 기존 attention을 유지한다.
- CUDA Graph는 서빙 전에 준비한다. 추가 runtime capture, GPU allocation, stream reset 및 fallback은 정상 arm의 시작/종료 증거에서 검사한다.

출력 logits의 tolerance를 완화하거나 토큰 일치만으로 정확도 gate를 대체하지 않는다. original-native/HF full-vocabulary BF16, 실제 HTTP/SSE 계약, 동시성별 새 프로세스 비교와 자원 회수는 별도의 gate다. 정상 비교는 GPU7에서 기준선v13B·최종D·stock vLLM을 순차 실행한다. GPU6의 허용된 작은 외부 예측 작업은 보존하고 정확도/구성요소 검사만 실행한다.

원격 최종 root: `/mnt/efs/nlp/psyche/riley-structural-v14d-20261021`.
구조 실험·탈락 후보와 구성요소 연구: 같은 EFS의 `riley-structural-v14a/b/c-20261021`.
원래 GPU6/7 holder PID2027914/2027915, start ticks6083215119/6083215121 및 idle20GiB 예약은 유지한다.

D는 full-vocabulary BF16 2,464개와 mixed 출력 2,016개의 original-native/HF 바이트 일치 및 CPU525개 검사를 통과했다. 정확도 통과를 처리량 개선으로 간주하지 않는다. 현재 로컬 기본 소스935개는 원래 v13B manifest와 모두 일치하고, 별도 디렉토리에서 패치로 재구성한 D의935개도 실험 manifest와 모두 일치했다.

실험 패치 SHA256: `4491e3a49ce20f7440c780938fb9980713cc12587ede7022056a6733075061ee`.
