# SmolLM2 GUI 유지 성능 비교 — 2026-09-11

측정 완료. 고정된 Riley 후보는 이번 셀에서 vLLM보다 느리다. 정확성 통과와 성능 개선은 별개이며, 성능 우위는 확인되지 않았다.

| 지표 | Riley | vLLM |
|---|---:|---:|
| 엔진 요청 완료 | 506.709 ms | 35.549 ms |
| 엔진 첫 토큰 | 435.240 ms | 7.005 ms |
| 엔진 이후 토큰당 시간 | 2.305 ms | 0.909 ms |
| HTTP 요청 완료 | 512.912 ms | 35.715 ms |
| HTTP 첫 텍스트 이벤트 | 441.109 ms | 7.787 ms |

각 실행 30회 중앙값을 구한 뒤 5쌍의 중앙값으로 집계했다. 쌍별 Riley/vLLM 요청 완료 시간 비율은 엔진 중앙값 14.254배(범위 13.759–14.440), HTTP 중앙값 14.361배(14.221–14.595)다. 범위는 관측된 5쌍의 최소·최대이며 신뢰구간이 아니다.

## 조건과 검증

- RTX4090, SmolLM2-135M BF16, c1/p128/o32, greedy, Riley commit `59f02242a2993b698d5f1c6b970b96bad66fb0a4`, vLLM 0.27.1.
- GUI 유지 조건 `g04-smol-gui-retained-512mib-v1`. 사용자 요청에 따라 Blender만 종료하고 Xorg/GNOME은 유지했다. 유휴 메모리 상한을 이 진단에만 512MiB로 적용했다. 기존 256MiB 조건 통과 결과가 아니다.
- 각 방식에서 새 프로세스로 AB/BA 5쌍, 실행당 warmup 5회 제외 후 30회. 엔진·HTTP 각각 양쪽 150회, 총 600회 보존.
- 엔진 300회는 입력/생성 토큰 해시를 고정 기준과 대조했다. HTTP 300회는 고정 출력 텍스트, length 종료 및 SSE DONE을 검증했다. 각 HTTP 프로세스는 측정 전 비스트리밍 요청의 128/32 토큰 수도 확인했다.
- 각 실행 전 CUDA compute 프로세스 부재, 환경/소스/바이너리 해시를 검사했다. 기존 시작 온도 50°C 상한을 유지했다. 최종 HTTP 실행은 각 lane 전에 동일하게 48°C 이하 냉각 대기를 적용했다.
- `completion.json`, `raw-sha256.json`과 원본 로그를 보존하고 로컬 복사본의 모든 원본 해시를 재검증했다.

## 해석과 한계

이 결과는 GUI 유지 단일 셀의 진단 비교다. 더 큰 모델, 동시성, 길이 또는 M4/M5 성능 인증으로 확대하지 않는다. 호스트 요청 시간을 측정했으며 CUDA event GPU 구간은 미측정이다. HTTP SSE 이벤트를 토큰별 ITL로 해석하지 않는다. 엔진 간 내부 요청 경계 차이는 가능하며 동일 HTTP 클라이언트에서도 큰 지연 차이가 재현됐다.

vLLM 원본의 legacy `environment_id`는 v1 라벨을 유지한다. 실제 환경은 `condition.json`과 lane별 live preflight의 v2 호스트 정보다. 원본을 사후 수정하지 않았으며 정식 canonical 집계 입력으로 재사용하지 않는다.

Riley 첫 토큰 시간이 전체 요청 시간의 약 86%다. 소스상 owned graph는 M=1로 prompt를 한 토큰씩 처리하며, profile의 prefill 보정은 기본 GEMM 뒤 추가 GEMM을 실행한다. 따라서 다음 최적화 후보는 일괄 prefill과 중복 GEMM 제거다. 각각의 비용 비중은 CUDA 구간 프로파일링으로 확인해야 한다. Decode도 약 2.30ms/token으로 vLLM 약 0.91ms/token보다 느리므로 별도 개선이 필요하다.

## 측정 도구 보정

`http-attempt1`은 종료된 연결의 TIME_WAIT 때문에 bind probe가 실패했고, `http-attempt2`는 시작 온도 51°C로 중단됐다. 둘 다 보존하되 집계에서 제외했다. `measure-gui.py`는 기존 후보·검증·요청·시간 측정 코드를 유지하고 probe에 SO_REUSEADDR 및 lane 전 동일 냉각 대기만 추가했다. 활성 listener는 여전히 거부됨을 `probe-check.json`으로 확인했다. 원본 runner와 후보 소스·바이너리는 수정하지 않았다. 최종 HTTP 5쌍 전체를 새로 측정했다.

집계 재생성: `python3 summarize.py`. 기계 판독 결과는 `summary.json`, 조건 변경은 `condition.json`, 도구 변경은 `harness-change.json`에 있다.

Blender 복구: 측정 종료 후 기존 세 명령으로 재실행했다. PID 2857155/2857156/2857157 생존과 localhost 9876/9911/9887 포트 LISTEN 및 시작 로그를 확인했다. GUI 세션은 종료하지 않았다.
