# Serving 최적화 작업 마무리 — 2026-09-15

사용자 요청에 따라 추가 개선·실험을 종료한다. 전체 vLLM 우위 목표는 달성하지 못했다. Projection CTA 실험은 opt-in으로 남기며 기본 경로로 승격하지 않는다. 이 문서는 작업 종료 기록이며 목표 성공 선언이 아니다.

## 최종 비교

동일 SmolLM2-135M BF16 / RTX 4090 / 출력 32 tokens. C8은 2026-09-15, C32는 2026-09-14의 각각 같은 실행 안에서 비교한다. 날짜 간 수치를 합치지 않는다. 각 값은 정순·역순 두 실행 지표의 중앙값이며 pooled percentile이나 신뢰구간이 아니다. TTFT/TPOT는 P50, tail은 요청 E2E P95/P99, 시간 단위는 ms다.

| 조건 | 엔진 | 출력 tok/s ↑ | TTFT ↓ | TPOT ↓ | E2E P95 ↓ | E2E P99 ↓ |
|---|---|---:|---:|---:|---:|---:|
| C8 공유 | Riley 기존판 | 5,183.33 | 12.464 | 1.196 | 49.869 | 52.253 |
| C8 공유 | Riley CTA 실험판 | 5,224.20 | 12.208 | 1.198 | 49.706 | 50.966 |
| C8 공유 | vLLM 0.29.0 | 4,583.28 | 14.353 | 1.298 | 60.700 | 68.028 |
| C32 공유 | Riley 기존판 | 10,212.66 | 14.517 | 2.668 | 117.085 | 132.074 |
| C32 공유 | Riley CTA 실험판 | 10,730.27 | 13.769 | 2.538 | 110.671 | 128.306 |
| C32 공유 | vLLM 0.29.0 | 9,723.89 | 36.144 | 1.712 | 324.196 | 328.079 |
| C32 비공유 | Riley 기존판 | 3,863.98 | 93.645 | 5.591 | 325.113 | 345.337 |
| C32 비공유 | Riley CTA 실험판 | 3,909.66 | 77.028 | 5.949 | 315.420 | 341.853 |
| C32 비공유 | vLLM 0.29.0 | 3,790.14 | 60.439 | 7.456 | 376.275 | 528.891 |

## 판정

- C8 공유: CTA 실험판의 vLLM 대비 throughput +13.98%, TTFT -14.94%, TPOT -7.71%. 기존 Riley 대비 throughput 차이는 두 순서에서 각각 +0.58% / +1.00%로 작다. 이 조건의 관측 우위를 전체 workload 우위로 일반화하지 않는다.
- C32 공유: vLLM 대비 throughput +10.35%지만 TPOT +48.26%로 느리다.
- C32 비공유: vLLM 대비 throughput +3.15%지만 TTFT +27.45%로 느리다. 기존판 대비 개선 방향도 순서에 따라 바뀌어 non-qualified다.
- 따라서 throughput +15%, TTFT/TPOT 각각 −10%, 높은 concurrency에서의 일관된 우위를 달성했다고 주장하지 않는다. C16/C64 추가 검증과 Hopper/Blackwell/multi-GPU 실측은 이번 종료에서 수행하지 않는다.

## 측정 종료와 검증 범위

C8은 총 12개 중 9개 lane을 완료했다. 마지막 vLLM 기동에서 디스크 읽기 대기가 지속돼, 사용자 마무리 요청에 따라 controller를 SIGINT로 종료했다. C8 공유는 두 순서가 모두 완료됐다. C8 비공유는 한 순서만 완료되어 위 표에서 제외했으며 raw와 `unique_single_order_not_qualified`에 보존했다. 미완료를 pass로 바꾸거나 실패 시도를 새 실행과 합치지 않았다.

C8 완료 9개 lane의 warmup 포함 19,008개 응답에 대해 SSE 재구성·usage·토큰·시각 순서·집계를 검산했다. Riley 완료 응답은 고정 기준과 일치한다. vLLM은 프로토콜 검증과 Riley reference 일치율을 분리하며, 다른 엔진과 모든 출력이 bitwise 동일하다는 주장은 하지 않는다. Stop/cancel/recovery는 각 Riley 엔진에서 동시성 8개, vLLM preflight는 32개다. 종료 요청 때문에 lifecycle exit -2 및 미완료 matrix가 명시적으로 기록됐다. C32의 6,912개 응답 검증은 별도 전체 matrix verifier를 사용한다.

C8 도중 호스트 부하 및 메모리 상태가 바뀌었다. PSI는 관측값으로 보존하고 수치를 보정하지 않는다. Python 3.13.15와 고정 revision 모델을 복구한 이후의 실행이며, 복구 전 C32와 절대 수치를 직접 합치지 않는다. Native 개선·모델 correctness·serving 개선은 별개 증거다.

## 보존 및 복구

- C8: `benchmarks/results/20260915-projection-cta-serving-c8-closeout/` — raw archive SHA-256, 54개 source snapshot, 부분 완료 verifier, receipt, host/비교 assessment.
- C32: `benchmarks/results/20260914-projection-cta-serving-c32/` — 전체 matrix와 기존 non-qualified 판정.
- Blender 9876/9911/9887의 get_scene_info가 모두 success. 로컬 뷰어 31840/31970/32010은 HTTP 200. 공개 도메인 직접 요청은 403이므로 공개 UI 동작까지 검증됐다고 주장하지 않는다.
- Riley runtime은 Rust → C ABI → CUDA를 유지한다. Python은 vLLM 및 실험 orchestration에만 사용한다.
- 추가 최적화·GPU 측정은 시작하지 않는다.
