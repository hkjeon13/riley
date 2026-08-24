# PR 17 — 후속 확장 진입 Gate

**상태:** Planned  
**선행 조건:** [PR 16](16-reliability-and-release.md)  
**다음:** 없음

[← 이전](16-reliability-and-release.md) | [목차](README.md)

## 목적

첫 release 이후 기능 요청이 core를 무질서하게 확장하지 않도록, 각 기능의 진입 조건과 독립된 작업 축을 정의한다.

## Quantization

진입 조건:

- BF16/FP16 path 안정
- weight packing interface 존재
- dequant/GEMM backend 후보 비교
- 정확도 평가 corpus 존재

분리할 PR:

1. quantization metadata IR
2. checkpoint loader
3. GEMM backend
4. KV quantization은 별도
5. end-to-end 품질·성능 보고

INT8, FP8, AWQ/GPTQ 등을 한 PR에 넣지 않는다.

## Prefix Cache

진입 조건:

- paged KV reference count 안정
- block hash와 token identity 정의
- eviction 정책 benchmark
- multi-tenant privacy 경계

prefix sharing은 단순 성능 기능이 아니라 lifetime과 보안 기능이다.

## KV Offload/Prefetch

진입 조건:

- GPU cache pressure가 실제 bottleneck
- PCIe/NVLink bandwidth 측정
- async copy와 stream ordering 검증
- head-of-line blocking 측정

CPU offload, SSD, remote cache를 단계별로 분리한다.

## MoE

분리할 축:

1. MoE IR/router semantics
2. top-k routing correctness
3. dispatch metadata
4. grouped expert GEMM
5. weighted combine
6. expert parallel communication

Mixtral류와 DeepSeek류 router 차이를 semantic parameter로 보존한다.

## Mamba/SSM

필요 primitive:

- causal depthwise Conv1d
- conv state update
- selective scan
- selective state update
- recurrent cache type

attention의 특수 case로 구현하지 말고 `MixerSpec`의 별도 variant로 둔다.

## Multimodal

순서:

1. modality encoder interface
2. projector/merger
3. token packing
4. multimodal position IDs
5. shared decoder 연결

Qwen-VL 전체를 model-specific monolith로 추가하지 않는다.

## Multi-GPU

진입 조건:

- single GPU 병목과 목표 명확
- tensor/expert/pipeline parallel 중 하나 선택
- NCCL failure와 timeout 정책
- process topology와 rank lifecycle
- 단일 GPU regression 방지

## Speculative Decoding

진입 조건:

- baseline scheduler와 KV rollback 안정
- verify mode attention 지원
- draft/target tokenizer compatibility
- acceptance metric 정의

## 확장 승인 질문

새 기능마다 다음을 답한다.

1. 실제 사용자 workload에서 어떤 병목을 해결하는가?
2. 기존 IR로 표현 가능한가?
3. core에 넣어야 하는가, plugin/backend로 분리 가능한가?
4. correctness reference는 무엇인가?
5. memory와 operational complexity는 얼마인가?
6. 실패 시 기존 stable path로 되돌릴 수 있는가?

## 완료 기준

이 문서는 구현 완료 문서가 아니라 범위 통제 문서다. 각 확장은 별도 deploy 문서와 benchmark contract를 만든 뒤 시작한다.

[← 이전](16-reliability-and-release.md) | [목차](README.md)
