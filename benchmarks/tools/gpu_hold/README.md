# aws-a100 GPU 6/7 지속 홀드

테스트 종료 시 CUDA holder를 종료하지 않는다. 대기 중 각 GPU에 20GiB를 예약하고 테스트 중에는 64MiB로 줄인다. CUDA context와 holder PID는 전환 내내 유지한다. GPU 연산을 반복하는 busy loop는 없다.

원격 root: `/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019`.
Lease directory: 위 root의 `leases-v2`.

- GPU 6 holder PID: `2027914`.
- GPU 7 holder PID: `2027915`.
- 6번의 작은 `fresh-demand` 작업은 사용자 지시로 공존을 허용했다. 고정 부모 PID/start ticks, 동일 UID 및 최대 2GiB 조건으로 해당 작업의 다음 자식과 GPU helper도 식별한다. 공존한 측정은 별도로 표시한다.
- 원격 `holder-transition-proof.json`: 두 GPU 모두 `20GiB → 64MiB → 20GiB`, PID/start ticks 불변을 실제 확인했다.
- 공유 GPU의 CUDA context와 메모리를 유지하는 방식이며, 시스템 차원의 배타적 할당은 아니다.

원격 root에서 실행한다:

```bash
python3 control.py --lease-dir "$PWD/leases-v2" --index 6 status
python3 control.py --lease-dir "$PWD/leases-v2" --index 7 status
python3 control.py --lease-dir "$PWD/leases-v2" --index 6 run -- <실험 명령과 인자>
```

`run`은 파일 lock으로 같은 GPU의 controller 중복을 막고, holder PID/소유자/start ticks/heartbeat를 확인한 뒤 예약을 줄인다. 자식 종료 또는 정상적인 interrupt 시 예약을 복구한다. controller가 사라지면 holder도 대기 예약 복구를 시도한다. 명시적으로 해제하라는 사용자 지시 전까지 holder 자체는 유지한다. OOM 시 다른 작업을 종료하지 않고 작은 context를 유지하며 복구를 재시도한다.

기존 측정 harness는 holder를 외부 작업으로 오인하지 않도록 PID와 start ticks를 검증해 별도로 구분해야 한다. GPU 2/3의 기존 결과와 GPU 6/7의 결과를 같은 표본으로 합치지 않는다. 6번 공존 작업의 자원 사용도 성능 측정 기록에 남긴다.

2026-10-21 검증에서는 GPU6의 공존 예측 작업을 보존하고 정확도·구성요소 검사만 실행했다. 정상 C16·C32 성능 비교는 모두 GPU7에서 순차 실행했다. 새 공존 작업은 살아 있는 고정 부모 PID/start ticks, 같은 UID, 정확한 script directory의 `*_predict.py` 자식 및2GiB 한도로 제한했으며 광범위한 descendant 허용은 하지 않았다. policy의 각 기존 항목별로 명시된 예외와 새 항목을 구분한다. 홀드는 시스템 차원의 배타적 GPU 할당이 아니다.
