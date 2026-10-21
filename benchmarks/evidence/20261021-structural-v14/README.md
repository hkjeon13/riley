# v14 구조 실험 — 미채택

최종 D의 정상5회 중앙값은 v13B 대비 C16 +0.76%, C32 -0.27%였고2% 초과 tail 플래그3개가 남았다. 기본 코드는 v13B로 유지한다. [결과 보고서](../../../docs/qwen-mixed-attention-structural-v14-20261021.md), [미적용 재현 패치](../../candidates/sm80-structural-v14/prototype.patch).

- [5회 통계](candidate/final-statistics.json), [정확도·자원·실행 경로 감사](candidate/final-integrity-audit.json), [미채택 결정](candidate/adoption-decision.json)
- [기본 v13의 새 CPU 재검사](candidate/retained-v13-cpu-recheck.json), [기본 source935 manifest](retained-v13-source-manifest.json)
- [실험 D source935 manifest](candidate/source-manifest.json), [v13→D4파일 lineage](candidate/source-lineage.json), [기본 소스와 D 패치 재구성 검사](local-source-and-patch-audit.json)
- [정상·진단 SSE 독립 감사](candidate/independent-audit.json), [tail 요청과 shape](candidate/tail-request-and-shape-audit.json)
- [실제 Lt recipe](candidate/merged-dispatch-proof.json), [실제 fused kernel grid](candidate/fused-attention-dispatch-proof.json), [decode chunk 노출 비중](candidate/attention-profile-exposure.json)
- [원래 holder3회 검증](candidate/final-holder-audit.json), [로컬 ZIP 검증](local-archive-verification.json), [stock 실제 backend](stock-startup-policy-verification.json)

[전체 압축 증거](../20261021-structural-v14.zip)는11,119,006bytes/1,920entries다. SHA256: `62edbb7f085b6131b2d3953b8bcdeb58123e39700880f8f50565fee958c65a48`. 모든 entry의 크기·SHA256을 다운로드 후 로컬에서 검증했다. Git에는 요약 index와 ZIP을 보존하고 확장된 raw 디렉토리 전체를 추가하지 않는다.

D의525개 CPU/2,464개 BF16/2,016개 mixed 출력 일치 결과는 실험 코드의 검사다. 기본 v13는 source935 일치, 새524개 CPU 검사 및 이번10개 fresh before API arm으로 다시 확인했다. 정확도 통과를 성능 개선으로 계산하지 않는다.

모델·full-vocabulary BF16·원래 Nsight trace·ELF는 EFS에 유지한다. ZIP의 `candidate/heavy-evidence-index.json`은4,498개 큰 원본의 경로·크기·SHA256을 기록한다. 최종 원격 root: `/mnt/efs/nlp/psyche/riley-structural-v14d-20261021`. A/B/C 탈락 코드·연구도 ZIP에 포함한다. 최초 route r2는 preload 누락으로 호출 증거가 없었으며, r3가 실제 recipe proof다. 두 기록을 구분해 보존했다.
