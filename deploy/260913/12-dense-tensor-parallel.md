# PR 12 — Dense 모델 tensor parallel 실행

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

단일 GPU 용량을 넘어 dense 모델을 실행하고 compute/communication 중첩의 기반을 만든다.

## 의존성과 변경 위치

선행: 01; 10과 transport 원칙 공유하되 필수 의존 아님.

예상 수정 위치: model weight sharding, runtime rank executor, collective adapter. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 지원 dense 모델 하나의 row/column parallel weight partition을 정의한다.
2. 기본 collective backend로 all-reduce/all-gather와 stream 의존성을 연결한다.
3. rank별 KV/head partition 및 output 결합을 구현한다.
4. rank failure·timeout·group teardown과 topology 식별을 추가한다.

## 범위 경계

pipeline parallel·expert parallel·custom collective kernel을 한 번에 넣지 않는다. 지원 head 분할 조건 밖 조합은 명시적으로 거부한다.

## Correctness·수명 계약

collective 순서·shape·dtype가 rank 사이 동일해야 한다. 일부 rank만 다음 iteration으로 진행하지 않으며 실패한 group의 buffer를 조기 재사용하지 않는다.

## 검증과 하드웨어 skip

CPU partition/reassembly, rank state simulation, 단일 GPU 기존 경로. 실제 multi-GPU parity·collective race·serving은 장비 부재로 skip. 확보 후 같은 GPU 수·연결로 엔진 비교.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

정확한 distributed full-model 실행과 scaling 비용을 제시한다. GPU 수 증가를 엔진 개선으로 계산하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

TP 기능을 비활성화한다. 단일 GPU에 맞지 않는 모델은 잘못된 fallback 대신 명시적 오류를 반환한다.

## 연구 근거

[ParallelKittens](https://arxiv.org/abs/2511.13940), [NanoFlow](https://arxiv.org/abs/2408.12757). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
