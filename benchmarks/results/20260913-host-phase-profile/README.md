# Host phase profile: duplicate future descriptor preparation

Nsight 실패 이후 opt-in wall-time 계측으로 실제 serving을 비교했다. **Paired 회귀의 대부분은 native 실행·대기 밖의 descriptor/adapter 비용 증가와 함께 나타났다.** 소스에서 동일 future batch를 두 번 생성하고 첫 결과를 버리는 경로도 확인했다. 다음 optimization batch는 이 중복 생성과 중복 structural validation을 제거하되 소유권·검증 계약을 유지하는 것이다.

## 실제 측정

동일 4090/SmolLM2-135M/checkpoint/natural workload, C32, single→paired→paired→single. lane마다 warmup 192 + retained 768개이며 각 lane의 전체 scheduled input token 합계는 244,160으로 동일하다. `RILEY_SERVING_PHASE_TIMING=1`일 때만 시각을 읽고 합계를 종료 시 기록한다. GPU 연산 자체나 serving 동작은 변경하지 않았다. Python은 외부 측정 client다.

아래는 server lifetime의 합계(ms)이며 warmup을 포함한다. `native`는 CUDA transfer/submit/wait 호출의 wall time이므로 순수 GPU 시간이라고 부를 수 없다. `adapter`는 engine 실행 구간에서 runtime의 계측 구간을 뺀 값으로 authority/descriptor 생성·결과 mapping 및 계측되지 않은 host 시간을 포함한다. 시계 호출·descheduling 비용도 포함될 수 있다.

| 반복/경로 | 계측 합계 | Native transfer/submit/wait | Runtime 준비·검증 | Adapter 등 | 바깥 plan/sample/commit |
|---|---:|---:|---:|---:|---:|
| 0 single | 6,338.62 | 5,591.99 | 312.48 | 314.23 | 119.93 |
| 0 paired | 6,690.94 | 5,647.04 | 467.79 | 462.09 | 114.01 |
| 1 paired | 6,703.49 | 5,665.67 | 465.36 | 457.10 | 115.37 |
| 1 single | 6,379.11 | 5,647.30 | 304.89 | 308.81 | 118.12 |

Paired의 추가 합계 352.32 / 324.38ms 중 native 차이는 55.05 / 18.37ms다. 증가분의 약 84% / 94%가 native 호출 밖에서 관찰됐다. 이것은 해당 wall-time 분해의 사실이며, 모든 host 시간을 없앨 수 있다는 성능 예측이 아니다.

Runtime의 `future_prepare`는 첫 반복에서 903회 / 188.05ms다. Scheduler `AuthorizedDecodeWindow::prepare_wire`가 `future_token::prepare`를 호출해 packet/reference를 만든 후, execution adapter가 `(first, second, _)`로 그 값을 버린다. 이어 runtime `submit_decode_window`가 같은 expectation에서 `prepare`를 다시 호출한다. Future prepare 내부에서도 successor structural expectation을 검증한 뒤 `encode_into`가 동일 expectation을 다시 검증한다. 이 명시적인 중복이 제거 대상이다.

Decode 평균 batch 폭은 single 28.07 / 26.25, paired window 26.39 / 26.06이었다. 반복별 편차가 있으며 이 수치만으로 배치 폭을 회귀의 주원인이라고 확정하지 않는다.

## Correctness와 범위

- 두 단계의 계측 구현 모두 실제 CUDA release 빌드 통과. [최종 빌드](evidence/phase-build.log), [exit 0](evidence/phase-build.exit).
- 최종 계측 4개 lane의 warmup+retained 3,840개에서 prompt ID·token ID·text hash·finish reference 일치, HTTP/SSE 오류 0건. 종료 exit 모두 0.
- [최종 분석 JSON](analysis.json), [원본 phase 로그와 launch metadata](evidence/host-phase-c32-v2/completion.json), [계측기](../../analysis/host_phase_analysis.py).
- 첫 단계 `host-phase-c32-v1`은 engine의 4개 구간만 측정했다. 최종 결론은 runtime까지 분해한 `v2`에 근거한다.
- 요청 자료는 frames만 제외하고 token/text/usage/checks/arrival timestamps를 gzip으로 보존했다. [Manifest](evidence/manifest.json)에 원본 remote 경로·원본 hash와 보관 파일 hash가 있다. 보관 hash 및 모든 request의 reference check를 다시 확인했다.
- 같은 시기 호스트 I/O pressure가 남았다. CPU cycles나 kernel별 GPU 비용, 순수 GPU idle을 측정한 결과가 아니다. 이 계측의 HTTP timing을 새 성능 개선 표나 release qualification으로 쓰지 않는다.
- 기본값은 계측 비활성·single이다. [직전 unprofiled vLLM 비교](../20260913-paired-serving-recovery/README.md)의 paired 비승격 결론은 유지한다.

## 다음 optimization batch 계약

1. 두 immutable expectation과 검증된 future packet/reference를 하나의 불투명한 소유 객체로 묶어 scheduler→runtime에 전달한다. 임의의 expectation과 다른 prepared bytes를 섞을 수 없어야 한다. Runtime의 현재 owner·issued cookie·pure-decode eligibility·reservation checks는 유지한다.
2. Rust의 immutable borrow로 검증된 expectation을 묶는 내부 proof 타입을 사용해 structural validation 직후 encoding에서 같은 전체 검증을 반복하지 않는다. 외부 `encode_into`는 여전히 잘못된 입력을 buffer 변경 전에 거절한다.

잘못된 owner/cookie/order/row/page mapping 거절과 바이트 결과 동등성을 CPU 검사로 확인하고, 실제 GPU/HTTP stop·cancel·reference 검사를 수행한다. 이후 계측을 끈 상태에서 직전 paired binary, 새 paired, single, vLLM을 같은 조건의 교차 순서로 비교한다. 이 batch의 예상 절약 시간을 실제 개선으로 계산하거나 전체 성능 목표 달성으로 표시하지 않는다.
