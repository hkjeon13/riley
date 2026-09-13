# PR 15 — 저정밀 weight와 fused dequant GEMM

상태: **계획만 작성 / 미구현**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

weight 이동량 감소가 unpack·scale 비용에 상쇄되지 않는 GEMM 경로를 마련한다.

## 의존성과 변경 위치

선행: 01, 05.

예상 수정 위치: model packed-weight loading, GEMM backend, quantization metadata. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. 지원 모델 하나의 weight-only 포맷·group size·packing을 고정한다.
2. offline weight 재배열과 변환 checksum을 기록한다.
3. tile dequantization+matmul을 결합한 backend를 연결한다.
4. shape별 dispatch·workspace와 BF16 fallback을 통합한다.

## 범위 경계

W4A8KV4 전체 QServe 재구현은 제외한다. activation/KV 양자화와 별개로 weight 경로의 비용을 판정한다.

## Correctness·수명 계약

checkpoint 원본을 덮어쓰지 않는다. dtype/포맷 identity를 강제하고 별도 quality 모드로 노출한다. LUT 방식과 uniform INT4를 혼동하지 않는다.

## 검증과 하드웨어 skip

packing roundtrip·scale 경계·full-model 품질, small/large batch·전 layer working set. 압축 해제만 따로 BF16 global tensor에 저장하는 비교안의 비용도 포함한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

품질 기준과 serving throughput·latency 통과 필요. 지원되는 동일 quantization 비교와 BF16 tradeoff를 구분한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

원본 checkpoint와 BF16 backend를 선택한다.

## 연구 근거

[FLUTE](https://arxiv.org/abs/2407.10960), [QServe](https://arxiv.org/abs/2405.04532). 논문 성능 배수는 Riley의 예상 개선율이 아니다.
