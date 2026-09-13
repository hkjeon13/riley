# PR 14 — KV 양자화와 fused attention

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

긴 context·높은 concurrency에서 KV 용량과 읽기량을 줄이면서 압축 해제 비용을 attention 안에 통합한다.

## 의존성과 변경 위치

선행: 01, 03, 10의 layout identity와 호환.

예상 수정 위치: KV page format, attention adapter/kernel, quality benchmark. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. K/V 비대칭 quantization의 포맷·scale·residual 정책 하나를 정한다.
2. append 시 quantization과 page metadata 갱신을 구현한다.
3. 소비 tile에서 dequantization하는 attention 경로를 연결한다.
4. 명시적인 KV dtype 선택과 BF16 fallback을 추가한다.

## 범위 경계

weight quantization과 TurboQuant/Fast-TurboQuant의 여러 알고리즘을 동시에 구현하지 않는다. 최초 후보는 KIVI 원리를 검토하되 측정 전 포맷을 고정한다.

## Correctness·수명 계약

BF16 exact 모드와 분리한다. 결과를 본 뒤 품질 허용치를 완화하지 않는다. scale·padding·고정밀 residual까지 실제 메모리 집계에 포함한다.

## 검증과 하드웨어 skip

page append/잔여 group, 긴 문맥 retrieval·generation 품질, FP32 오차, graph·수명 검사. 동일 KV quantization을 지원하는 비교 엔진 조건과 BF16 대비 tradeoff를 각각 기록.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

사전 품질 기준과 serving 이득을 모두 충족해야 해당 모드를 승격한다. BF16 목표 달성 증거로 대체하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

새 요청부터 BF16 KV로 복귀. 진행 중 cache를 무단 재해석하지 않는다.

## 연구 근거

[KIVI](https://arxiv.org/abs/2402.02750), [QServe](https://arxiv.org/abs/2405.04532), [TurboQuant](https://arxiv.org/abs/2504.19874). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
