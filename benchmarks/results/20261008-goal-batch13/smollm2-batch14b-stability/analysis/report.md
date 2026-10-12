# Batch14 안정성 검증

지속 부하 300s, idle 취소 20×2(early/decode, 매회 전체 corpus exact 재요청), 부하 중 취소 40회. 제외 없음.

| 조건 | 엔진 | 결과 | 지속 요청 | 오류 | exact | 처리량 첫/끝 round | TPOT p50 첫/끝 | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | RSS 기울기 MiB/min | drain 횟수(최대 ms) | 취소(idle/부하) | 종료 회수 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-fixed | candidate | 통과 | 16896 | 0 | 16896 | 1110.4/1109.2 | 0.794/0.796 | 1158/1158 | 762.3/762.4 | 0.025 | 84 (20.7) | 40/40 | True |
| c1-natural | candidate | 통과 | 4608 | 0 | 4608 | 1115.7/1116.9 | 0.800/0.799 | 1178/1178 | 762.2/762.3 | 0.046 | 84 (20.8) | 40/40 | True |
| c8-fixed | candidate | 통과 | 87552 | 0 | 87552 | 5821.0/5774.3 | 1.184/1.207 | 1186/1186 | 762.5/762.8 | 0.074 | 84 (20.7) | 40/40 | True |
| c8-natural | candidate | 통과 | 25344 | 0 | 25344 | 6228.8/6315.2 | 1.191/1.174 | 1334/1334 | 762.4/762.6 | 0.057 | 84 (21.0) | 40/40 | True |
| c16-fixed | candidate | 통과 | 129408 | 0 | 129408 | 8454.6/8877.2 | 1.638/1.525 | 1214/1214 | 764.3/764.9 | 0.152 | 84 (20.7) | 40/40 | True |
| c16-natural | candidate | 통과 | 40320 | 0 | 40320 | 9843.4/9944.3 | 1.506/1.489 | 1514/1514 | 764.2/764.7 | 0.123 | 84 (20.7) | 40/40 | True |
| c32-fixed | candidate | 통과 | 171648 | 0 | 171648 | 11583.6/11556.0 | 2.402/2.434 | 1270/1270 | 768.0/768.8 | 0.209 | 84 (20.7) | 40/40 | True |
| c32-natural | candidate | 통과 | 54912 | 0 | 54912 | 13661.8/13673.4 | 2.168/2.170 | 1874/1874 | 768.3/769.2 | 0.248 | 84 (20.7) | 40/40 | True |
| c32-natural | vllm | 실패: host gate timeout cancel-under-load | — | 0 | — | —/— | —/— | —/— | —/— | — | — (—) | —/— | None |
| c32-fixed | vllm | 미실행 | | | | | | | | | | | |

전체 lane 통과: False

한계: early cancellation closes right after the request write; the server-side phase (queued/prefill) is not observable from the client; memory is sampled once per second; peaks between samples are not proven; no performance qualification
