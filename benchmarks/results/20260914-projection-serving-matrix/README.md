# Projection serving concurrency matrix

동일 모델·바이너리·KV budget과 공통 Python 측정 클라이언트 GC 통제/tmpfs 조건으로 C8·C32·C64 비교를 완료했다. Rust 서버에 GC를 도입한 것이 아니다. 후보는 opt-in 상태이며 전체 목표는 미달이다.

각 값은 역순 두 실행의 run-level estimate 중앙값이다. Concurrency 간 throughput이나 percentile을 합산하지 않는다. C64는 client64/active32이며 GPU active64 결과가 아니다. 모델 경로는 각 launch 기록에 있고 checkpoint 내용 해시는 이 준비 기록에서 검증하지 않았다.

| Client / active | Workload | Riley tok/s | vLLM tok/s | 차이 | TTFT P50 Riley / vLLM ms | TPOT P50 Riley / vLLM ms | E2E P99 Riley / vLLM ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 / 8 | shared | 5141.1 | 4512.6 | +13.93% | 11.76 / 14.29 | 1.223 / 1.324 | 52.92 / 71.78 |
| 8 / 8 | unique | 2762.3 | 2888.3 | -4.36% | 50.22 / 29.41 | 1.364 / 1.861 | 95.44 / 115.54 |
| 32 / 32 | shared | 10339.9 | 11886.6 | -13.01% | 13.89 / 28.62 | 2.723 / 1.798 | 104.89 / 111.05 |
| 32 / 32 | unique | 3811.2 | 5001.0 | -23.79% | 62.18 / 46.98 | 6.616 / 4.991 | 304.00 / 245.47 |
| 64 / 32 | shared | 10286.6 | 13097.8 | -21.46% | 112.69 / 84.37 | 2.708 / 2.205 | 218.70 / 223.33 |
| 64 / 32 | unique | 3780.2 | 4568.8 | -17.26% | 333.64 / 270.08 | 6.642 / 5.405 | 599.06 / 567.43 |

C8 shared만 제한된 기술 통계상 최소 지표를 만족한다. 이 조건도 throughput +15% 및 TPOT −10% 목표폭은 미달이다. C8 unique는 TTFT가 더 높고, C32/C64는 두 workload 모두 throughput이 부족하다. C64 shared는 동일 binary control 대비 E2E P95/P99 및 ITL P99 회귀도 있어 기본값으로 승격하지 않는다.

전체 393216 retained 요청/12582912 output tokens와12288 warmup을 검증했다. Riley retained294912개는 frozen prior reference와 일치했다. Stop/cancel/recovery는 각각312건이다. 각 단계의 검증 범위와 vLLM 출력 차이는 개별 보고서에 유지한다. 두 번의 closed-loop 실행은 장기 안정성·open-loop·추가 모델·multi-GPU/Hopper/Blackwell qualification을 대체하지 않는다.

다음 작업은 고정 candidate/control의 shared·unique C32에 bounded Nsight trace를 수집하여 prefill projection·attention·FFN과 decode·graph 사이 대기의 남은 비중을 확인하는 것이다. Native projection 개선을 serving 이득으로 환산하지 않는다. Raw trace/SQLite는 원격에 보관하고 수치만 추출한다. 결과를 바탕으로 연관 개선 2–5개를 영역 단위 batch로 결정한다.

## 원본 결과

- [C8 전체 표·검증·원본](../20260914-projection-gc-controlled-serving-c8/README.md)
- [C32 전체 표·검증·원본](../20260914-projection-gc-controlled-serving/README.md)
- [C64 전체 표·검증·원본](../20260914-projection-gc-controlled-serving-c64/README.md)

세 조건의 공통 preparation 필드가 일치함을 확인했다. 각 fixture 파일 해시와 source report 해시는 matrix.json에 별도로 기록하며, concurrency별 생성 reference 파일이 동일하다고 가정하지 않는다.
