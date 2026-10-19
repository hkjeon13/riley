# QKV·Gate/Up 및 Attention 후속 검증 — SM89

검증 기록: 2026-10-18T13:56:13.887729+00:00

이번 후속 후보는 모두 미반영입니다. 현재 소스는 검증된 통합164이며, 935개 파일 해시를 다시 확인했습니다. 기존 성능 개선 내용은 [exact-smallrows-graph-sm89.md](exact-smallrows-graph-sm89.md)에 있습니다. 광범위한 모델·배치에서 stock vLLM 이상의 성능이라는 목표는 아직 달성하지 못했습니다.

## 실제 서버 성능

같은 RTX 4090에서 현재164, 비동기 가중치 복사 후보171b, stock vLLM 0.27.1을 각각 새 프로세스로 실행했습니다. 단위는 출력 tok/s입니다. 셀마다 새 비교 1회이며 반복 검증의 중앙값으로 일반화하지 않습니다.

| 모델 / 동시성 | 현재 Riley | 후보(미반영) | 변화 | stock vLLM |
|---|---:|---:|---:|---:|
| Qwen3-4B C16 | 557.6 | 564.4 | +1.22% | 1021.1 |
| Mistral-7B C32 | 480.2 | 481.0 | +0.16% | 867.5 |

원래 EOS·greedy, context1024, 토큰 예산512, KV block16 / 블록 수64×동시성, prefix cache off, 22GiB 예산, 같은 CPU affinity를 사용했습니다. 측정 구간의 신규 Graph 캡처는 0건입니다. Riley는 CUDA12.8, stock 설치본은 CUDA13입니다. 두 셀 모두 세 실행에서 측정 출력 토큰 수는 3,584개로 같았습니다.

처리량 개선이 Qwen +1.22%, Mistral +0.16%로 작았습니다. Qwen TPOT p95는 24.769→24.827ms, Mistral TTFT p95는 1173.537→1179.174ms로 증가했습니다. 후보의 모든 지연 지표 무회귀 조건은 두 셀 모두 통과하지 못했습니다. 동일한 후보의 반복 측정으로 이 결과를 덮어쓰지 않았습니다.

stock vLLM의 HF 일치 횟수는 속도와 별도로 원본에 남겼습니다. 속도 표로 수치 일치나 모델 품질 동등성을 주장하지 않습니다.

## 구조 확인과 실험 결과

설치된 vLLM의 QKVParallelLinear·MergedColumnParallelLinear 및 GPU runner 소스 해시를 재확인했습니다. 현재 Riley의 Qwen HF 경로는 별도 Q/K/V 계획으로 계산한 뒤 비트 패킹하며, 개별 무바이어스 Query 함수의 조건만으로 실제 경로를 판단할 수 없습니다. Driver API에서 후보의 Qwen Query·출력 커널 72개와 Mistral K·V 커널 64개 실행을 확인했습니다. Query가 실행되지 않는 문제를 수정했다는 주장은 하지 않습니다.

| 실험 | 수치 검증 | 성능 및 결정 |
|---|---|---|
| 입력 F32 확장 및 공유 메모리 배치166/167 | 두 모델 모두 일치 | 전반적으로 느려 미반영 |
| 차가운 가중치 비트 재배열168 | 두 모델 각각 14,112조건 일치 | 안정적인 이득이 없어 미반영 |
| 가중치 cp.async 이중 버퍼169/170/171b | 선택136가중치 79,968조건 + CAPI 9,792조건, 변경 Graph 및 모델 출력 일치 | 부품에서는 약11~18% 개선했지만 실제 서버 이득·지연 기준 미달 |
| Attention K·V 직접 읽기172b | 2,879조건 / 변경 Graph35상태 일치 | 시험한 큰 배치의 warm 결과 대부분 악화, 미반영 |
| K만 또는 V만 직접 읽기174 | 변형마다 2,879조건 / Graph35상태 일치 | K는 거의 동일하거나 느림, V는 더 느림, 미반영 |
| 공유 메모리 로드와 register-only FMA asm 분리175 | 모델마다 7,056조건 및 변경 Graph 일치 | 최대 일관 부품 이득 Qwen4.76%, Mistral2.27%, 미반영 |
| PV 페이지 주소를 CTA에서 한 번 계산177 | 2,879조건 / Graph35상태 일치 | 직접 V 접근 일부 비용은 줄었지만 warm 결과 모두 기준보다 느림, 미반영 |

직접 읽기 실험은 합성 BF16 입력에 대한 부품 검증입니다. Softmax를 제외하고 지정한 probability를 사용했으므로 전체 Attention·HF·서버 검증으로 간주하지 않습니다. 부품의 postflush는 512MiB 메모리 쓰기 절차이며 하드웨어 캐시 miss를 입증한 계측은 아닙니다.

171b는 CPU 테스트517개, 일반 GPU 테스트40개 제외, 두 모델의 기존 같은 GPU HF 기준104행씩, C1 eager 및 실제1~32행의 변경-context Graph, 실제 Graph 노드 검증을 통과했습니다. 서버 응답 값1,712개를 토큰·텍스트·프롬프트·종료 사유·사용량까지 검사했고 자원·소유 PID·연결 복구 검사도 통과했습니다. GPU 검사는 CPU 테스트의 제외40개와 별개로 실제 실행했습니다.

이번 결과는 커널 몇 개를 줄이거나 가중치 복사를 숨긴다는 구조만으로 전체 성능 향상을 보장하지 않음을 보여줍니다. 실제 실행의 작은 개선을 전체 병목 해결로 확대 해석하지 않았습니다. 수치 허용오차는 완화하지 않았습니다.

## 남은 작업

후속 가설은 Q32/KV8/D128의 큰 배치에서 기존 QK·HF Softmax·PV의 연산 순서와 BF16 경계를 그대로 유지하며 하나의 CTA가 query/head와 페이지 주소를 재사용하는 방식입니다. 아직 구현·검증하지 않았습니다. 기존 H9/D64 소형 모델에서 느렸던 fusion과 구분하며, 부품·실제 가중치·전체 모델·지연·자원 검증 후에만 반영합니다.

A100 GPU2/3은 마지막 확인에서 wckim의 VLLM::Worker_TP0/1이 점유 중이었습니다. 이번 결과는 RTX4090만의 결과이며 A100과 합산하지 않았습니다. 마지막 RTX 상태는 GPU 계산 프로세스 없음, 사용253MiB, 활용률0%였습니다. 배포·커밋·푸시·장시간 부하 실행은 수행하지 않았습니다.

## 근거

- [실제 Graph 커널](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v171b-driver-controller-result.json)
- [전체 서버 값·자원 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v173-pilot-independent-audit.json)
- [비교 원본 표](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v173-verified-performance-table.json)
- [Attention K/V 분리 결과](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v174-controller-result.json)
- [PV 주소 재사용 결과](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v177-controller-result.json)
- [미구현 후속 가설](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/post-v177-fused-native-wide-attention-hypothesis.json)
