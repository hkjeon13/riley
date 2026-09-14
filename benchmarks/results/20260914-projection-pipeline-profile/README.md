# Projection pipeline profile

고정 control/candidate, shared/unique C32의 bounded Nsight 진단이다. 같은 binary SHA, rolling·FFN·adaptive decode와720MiB KV payload를 사용하고 projection flag만 바꿨다. Warmup64+retained64, 네 lane의512 응답을 실제 SSE frame에서 재구성해 frozen prior fixture와 일치함을 검증했다. Python 클라이언트는 각 요청 phase에서 GC를 비활성화하고 이후 복구했다. Rust serving runtime은 변경하지 않았다.

| Lane | Prefill/mixed graphs | Graph span ms | Attention ms | Projection ms | FFN ms |
|---|---:|---:|---:|---:|---:|
| control-shared | 27 | 101.126 | 44.844 | 24.556 | 15.856 |
| projection-shared | 27 | 93.692 | 45.098 | 16.830 | 15.908 |
| projection-unique | 133 | 648.940 | 258.188 | 114.032 | 198.779 |
| control-unique | 133 | 719.472 | 257.594 | 184.812 | 199.379 |

Unique 두 trace는 prefill/mixed graph 수133개로 동일하다. Projection kernel 합은 184.812→114.032ms이고 prefill/mixed span은 719.472→648.940ms다. 후보의 attention은 span 대비 39.79%, FFN은 30.63%다. 이는 단일 trace의 시간 구성이고 반복 serving 성능 효과는 [별도 concurrency matrix](../20260914-projection-serving-matrix/README.md)로 판단한다.

Graph count 중간80%를 선택했으므로 warmup과 retained가 섞이고 두 phase 사이 준비 구간도 포함될 수 있다. 약55–59ms 최대 gap에는 클라이언트의 phase 전환·full GC 및 profiler overhead가 포함될 가능성이 있다. Gap 합계나 optimistic speedup 필드를 서버 최적화 예상치로 사용하지 않는다. 개별 graph 내부의 kernel 합도 실제 HBM bandwidth나 CPU stall 원인에 대한 증거는 아니다. Control/candidate의 전체 선택 graph 수도 unique215/216으로 다르므로 전체 trace span 비율을 정확한 동일 작업 speedup으로 해석하지 않는다.

네 profiler/server exit0, owned process 잔존0, RSS guard 통과 및 Blender3개 복구 receipt를 검증했다. Raw Nsight/SQLite는 원격에만 남겼다. Curated archive에는 숫자 분석·fixture·응답·실행/복구 receipt 및 source hash만 포함한다. [DONE] 전송 종료는 저장된 frames 밖의 hash-bound client 검사에 의존한다.

다음 batch 선택에서는 남은 attention/FFN 비중과 실제 mixed scheduling 비용을 함께 검토한다. 이미 serving에서 회귀한 GQA staging/softmax 공유 및 monolithic persistent 확대를 반복하지 않는다. Graph gap 개선을 선택하려면 먼저 phase 경계와 요청이 준비되지 않은 idle을 분리해야 한다. 수치 계약을 바꾸는 split reduction은 별도 gate를 통과해야 하며 이 프로파일만으로 선택하거나 기존 허용오차를 완화하지 않는다.
