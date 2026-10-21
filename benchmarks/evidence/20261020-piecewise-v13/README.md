# 유지한 기본 v13B의 검증 snapshot

[결과 보고서](../../../docs/qwen-piecewise-graph-merged-gateup-v13-20261020.md), [compact ZIP](../20261020-piecewise-v13.zip).

ZIP SHA256: `4df7f88552b077d01105f169cb0a8bc6a53c4258bcb32783825675dc11ad5945` (6,338,800bytes/714entries). source935 manifest SHA256: `474ebd145037fc91a023dcc8a4adc996be1d20eaead886a5058d9086e59a6fbe`.

[당시 최종 감사](candidate/final-integrity-audit.json), [당시5회 성능](candidate/final-statistics.json), [source manifest](candidate/source-manifest.json). 무거운 BF16/trace/ELF는 EFS 원본과 hash index로 연결한다. Git에는 compact ZIP과 핵심 index만 보존한다.

2026-10-21 v14 후보는 전체 처리량 개선을 입증하지 못해 채택하지 않았다. 기본 코드는 이 v13B935개 소스와 일치하며 CPU524개와 fresh10개의 before API arm을 다시 검사했다. [v14 판정·재검사](../20261021-structural-v14/README.md). 두 날짜의 성능 표본을 합쳐 개선율을 만들지 않는다.
