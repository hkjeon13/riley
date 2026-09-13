# 구조 연구와 현재 코드의 접점

2026-09-13 read-only 검토. 원격 `/tmp/riley-opt-260912/prefill-shapes-source-v11`, V56 commit `030c207565eda3ec45533a166e6d9a201c8d1831`, clean checkout에서 확인했다. 구현·빌드·새 GPU 측정은 수행하지 않았다. 아래 파일/행은 이 원격 revision 기준이다.

## 비동기 scheduler는 실제로 새 실행 계약이 필요하다

확인한 serving 경로:

`engine::VariableServingSession::execute → execute_llama_iteration_variable_graph* → VariableSession::execute_rows → replay_transfer → cudaGraphLaunch + cudaStreamSynchronize → 결과 검증 → scheduler settlement → confirm_scheduler_commit`

| 근거 | 확인한 내용 | 설계 영향 |
|---|---|---|
| `crates/riley-server/src/engine.rs:2453` | 8/16/32-row variable session dispatch | 기존 single-token graph 경로만 분석해서 serving 전체를 설명하면 안 됨 |
| `crates/riley-runtime/src/llama/variable_session.rs:55` | retained iteration이 있으면 다음 실행을 거부 | 최대 두 iteration을 겹치려면 identity/replay/issued/retained 상태 전이가 바뀜 |
| 같은 파일 `:66` | replay와 결과 읽기를 연속 수행하고 completion 검증 | enqueue와 collect를 분리하되 결과 buffer 수명을 유지해야 함 |
| 같은 파일 `:76` | 성공한 scheduler settlement 후에만 retained를 해제 | GPU 완료와 논리 commit을 별도 증거로 관리해야 함 |
| `kernels/src/graph_resources.cu:440` | graph launch 직후 stream synchronize; 실패 시에도 동기화 시도 | 호출 삭제만으로 비동기화하면 기존 KV/owner 안전 계약을 깨뜨림 |
| `crates/riley-server/src/engine.rs:3538` | settlement 오류가 없을 때 session commit 확인 | 응답·EOS·cancel 처리와 다음 iteration 예약 관계를 함께 설계해야 함 |

따라서 제안 batch는 enqueue/collect 분리, 두 세트의 metadata/completion 수명, event 기반 완료 증명, 다음 token의 GPU 전달, scheduler settlement를 함께 포함해야 한다. 오류 시 completion이 불명확한 owner를 보존하는 기존 정책도 유지해야 한다. 이것은 구현 제안이며 성능 이득은 측정하지 않았다.

## FlashInfer 연결: KV 전체 재배치가 필요하다고 단정할 근거는 없다

`kernels/src/prefill_shape_attention.cuh:17`의 현재 paged index는 다음과 같다.

```text
((physical_page * 3 + kv_head) * 16 + token_in_page) * 64 + dim
```

즉 이 paged 경로의 물리 순서는 `[page, KV head=3, token=16, dim=64]`다. page 안에서 token-major라는 모호한 이름보다 이 stride를 계약에 써야 한다. page table은 logical→physical mapping이며 head 수, page 길이, head dimension은 현재 고정 모델 값이다.

이 순서는 HND 형태 adapter의 무복사 가능성을 검토할 근거다. 하지만 모든 layer의 K/V allocation, pointer offset, alignment, output stride 및 wrapper 요구 조건을 아직 대조하지 않았으므로 **무복사 통합 가능을 확정하지 않는다.** 초기 설계에서 새 KV pool 또는 매 iteration transpose를 기본값으로 넣을 이유도 없다.

## Graph 재사용과 결정성 사이의 제약

FlashInfer 공식 API는 `plan()`을 graph capture 안에서 호출할 수 없다고 명시한다. auxiliary plan을 재사용하더라도 sequence length 변화에 따른 재계획 비용을 따로 확인해야 한다. `fixed_split_size`는 FA2 merge의 batch-size invariant reduction을 돕지만, sequence length 변화로 CTA 수가 달라질 수 있어 CUDA Graph 호환성을 보장하지 않는다. `disable_split_kv`도 별도 옵션으로 존재한다. [공식 attention API](https://docs.flashinfer.ai/api/attention.html).

이에 따라 adapter 검토는 다음 세 가지를 서로 분리한다.

1. 같은 수학적 attention과 mask/RoPE/KV 범위를 계산하는가.
2. batch 구성이 변해도 결과가 재현되는가.
3. 기존 Riley의 BF16 중간 rounding과 byte-exact한가.

두 번째를 만족해도 세 번째를 증명하지 않는다. graph capture만 성공했다고 동적 길이 전체에서 재사용해도 된다는 뜻도 아니다.

## 비동기 실행 계약 초안과 도입 판정

2026-09-13 원격 HEAD `030c207565eda3ec45533a166e6d9a201c8d1831`에서 `issue_rows`, `execute_rows`, `confirm_scheduler_commit` 및 native graph replay를 다시 읽었다. 현재 `retained`가 남아 있으면 다음 발급/실행을 거부하고, native replay는 stream synchronization이 성공해야 완료를 노출한다. 따라서 event를 추가하는 것만으로는 overlap이 생기지 않는다.

SGLang은 future token을 통해 다음 batch 준비를 진행하며, TensorRT-LLM은 이전 step의 CPU stop 처리 이전에 다음 GPU step을 제출한다. 아래는 그 원리를 현재 Riley의 owner/commit 계약에 적용한 **설계 제안**이며 구현된 동작이 아니다. [SGLang](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html).

### 세 개의 진행 위치

`submitted`, `device_completed`, `scheduler_committed`를 분리한다. 현재 `last_accepted_replay`는 scheduler commit의 증거로 유지하고, 미완료 작업 발급 번호를 대신 나타내도록 의미를 바꾸지 않는다. 초기 설계는 최대 두 iteration의 ticket을 유지하며, CPU에서 아직 반영하지 않은 이전 결과에 의존하는 다음 token은 GPU 경로로 전달한다.

| 상태 또는 사건 | 다음 실행·외부 응답 | buffer·KV 수명 |
|---|---|---|
| Prepared, 제출 전 취소 | 제출 없이 ticket 폐기 가능 | GPU에 전달하지 않았다는 증거가 있으면 예약 해제 |
| Submitted, 완료 event 미확인 | 의존 순서를 보장한 후속 작업만 제출; 결과 노출 금지 | descriptor, output slot, 참조 page generation을 고정 |
| DeviceCompleted, 결과 검증 전 | CPU 결과 검증 수행; 아직 응답하지 않음 | 완료가 확인된 staging slot만 재사용 후보가 됨 |
| Validated, scheduler 반영 전 | 같은 request의 이전 commit 순서를 지켜 반영 | ticket과 결과 소유권 유지 |
| Committed | 유효 token을 한 번만 streaming 전달 | page는 후속 ticket의 참조까지 없어야 회수 |
| EOS·길이 제한·취소를 뒤늦게 발견 | 이미 제출된 추가 step의 출력은 버림; 추가 제출 중단 | 취소 자체는 GPU 완료 증거가 아님. 마지막 접근 완료까지 보유 |
| launch/event 오류로 완료 여부 불명 | 해당 실행 owner를 격리하고 후속 제출 중지 | 완료를 입증하는 drain/close 성공 전까지 page 반환 금지 |

두 ticket의 staging/completion buffer는 구별해야 한다. 그러나 같은 stream에서 GPU graph를 순차 실행하면 모든 activation workspace를 두 벌로 만들 필요는 없다. CPU가 갱신하거나 읽는 buffer와 GPU 실행 간 alias를 먼저 제거한다. 여러 stream에서 graph를 동시에 실행하는 설계는 별도 scratch·KV 의존성 분석이 필요하며, CPU/GPU overlap의 필수 조건은 아니다.

KV append 공간은 제출 전에 예약한다. CPU가 아직 EOS를 보지 못한 다음 step도 이미 제출될 수 있으므로 마지막 page 경계에서 추가 공간 또는 device-side 유효성 제어가 필요하다. 새 request가 같은 slot을 재사용할 때는 request identity와 generation을 검증해 늦은 completion이 새 request에 적용되지 않게 한다. device token 전달은 잘못된 결과를 외부에 확정하는 행위와 구분한다.

### 하나의 구현 batch로 묶을 범위

1. 제출·완료 조회·결과 반영 API와 iteration ticket/진행 위치 분리.
2. GPU token 전달과 descriptor/completion buffer 이중화.
3. event 기반 owner·KV 참조 수명 및 오류 시 drain.
4. EOS/cancel/응답 전달과 다음 batch 준비의 중첩.

채택 증거는 CPU 준비/후처리와 GPU 실행이 실제 겹치는 trace, 기존 exact 결과 유지, 동적 admission·page 경계·취소·오류에서 재사용 안전성, 그리고 같은 workload의 serving 비교다. iteration 사이 대기 시간을 아직 정량적으로 분리하지 않았으므로 예상 개선율은 제시하지 않는다. 이 방식은 attention의 부동소수점 연산 순서를 바꾸지 않고 실행 구조를 바꿀 수 있어, 새로운 attention backend의 수치 계약과 독립적으로 평가할 수 있다.

**우선순위 판단:** 비동기 runtime은 구현 계약을 구체화한 첫 구조 후보로 두고, attention backend는 기존 구현 대비 수치 호환성 검사를 먼저 끝낸다. persistent/tile 실행은 연산 간 메모리 이동을 줄일 더 큰 후보로 유지한다. 하드웨어별 우선순위는 별도이며, GPU별 지원 여부를 전체 연구 후보의 탈락 기준으로 사용하지 않는다. 이는 구현 착수나 성능 개선 확인을 뜻하지 않는다.

## 다음 조사에서 닫아야 할 항목

- FlashInfer의 고정 release에서 SM89/BF16/9 Q heads/3 KV heads/head64/page16 조합 지원 및 실제 backend 선택.
- K/V parent allocation과 layer offset의 HND view 적합성, page table 변환 비용.
- RoPE를 Riley가 이미 적용한 tensor인지 확인하여 backend에서 중복 적용하지 않는 계약.
- 동적 length에서 plan 재사용·graph bucket·determinism의 동시 성립 범위.
- 비동기 scheduler의 상태 전이 표와 cancel/launch-failure/event-failure에서 KV 회수 규칙.

위 항목을 확인한 후 backend adapter와 비동기 execution 중 먼저 구현할 batch를 선택한다. 현재 사실만으로 FlashInfer가 더 빠르다거나 synchronize가 전체 격차의 주원인이라고 결론내리지 않는다. [전체 연구 보고서](RESEARCH_20260913.md)의 우선순위는 이 적용성 검토에 따라 조정할 수 있다.

## 설치된 FlashInfer 소스 대조

원격 vLLM 환경에 이미 설치된 `flashinfer-python 0.6.16.post3`의 소스를 읽었다. 새 설치나 import/JIT/GPU 실행은 하지 않았다. 웹의 main 및 0.6.18 문서와 동일 버전으로 취급하지 않는다. [검토한 소스와 SHA256](raw/flashinfer-source-audit/manifest.json)을 보존했다.

- `data/include/flashinfer/utils.cuh:149`의 `DISPATCH_GQA_GROUP_SIZE`는 **group size 3을 명시적으로 포함**한다. 현재 9 Q heads / 3 KV heads 비율을 CUDA-core decode가 정적으로 거부한다고 볼 근거는 없다. 다만 head64/BF16/page16의 전체 compile·runtime 성공은 아직 검증하지 않았다.
- `decode.py:900`에서 graph wrapper는 고정 batch size를 기록하고 `:1102`에서 다른 batch size를 거부한다. Riley의 32개 슬롯 중 live row 수가 변하는 정책을 그대로 연결할 수 있다고 가정하지 않는다. bucket별 wrapper 또는 안전한 inactive-row 표현을 설계해야 한다.
- `decode.py:1114`는 plan 경로에서 indptr를 CPU로 옮긴다. GPU metadata를 매번 plan에 전달하면 host dependency가 다시 생길 수 있다. 기존 CPU page-table 정보로 plan을 준비하는 경로와 GPU 실행의 중첩을 검토해야 한다.
- tensor-core decode는 prefill module 계열로 연결되고 CUDA-core decode는 별도 module 계열을 쓴다. 따라서 “FlashInfer decode 하나”로 묶지 말고 두 경로의 지원 조건과 graph 계획 비용을 따로 기록해야 한다.

이 확인으로 GQA 비율은 즉시 탈락 사유에서 제외했다. 남은 주요 도입 질문은 **수치 계약, 동적 batch/length의 plan 비용, graph 재사용**이다. 소스에 지원 분기가 있다는 사실을 실제 성능이나 correctness 통과로 해석하지 않는다.

## 격리 호환성 검사에서 확인한 toolchain 문제

후속 검사에서는 실제 RTX 4090과 FlashInfer 0.6.16.post3/PyTorch 2.13.0+cu130을 사용해 C32, context398, BF16, HND page16, Q9/KV3/head64 조합의 eager/graph 실행을 시도했다. 성능 timing은 포함하지 않았다. synthetic paged 입력과 FP32 attention 대조, graph replay 동일성 검사를 준비했지만 **JIT 빌드 단계에서 실패해 attention 수치 결과는 얻지 못했다.** 프로세스가 exit0이어도 probe는 예외를 JSON으로 기록하므로 통과가 아니다.

첫 실행은 PATH에 venv의 ninja가 없어 실패했고, 해당 경로를 명시한 재실행에서는 compiler/header mismatch가 드러났다. `/data/riley-g04-cuda13/bin/nvcc`는 실제로 CUDA 13.3.73이고 같은 prefix의 runtime headers는 13.0이다. venv도 nvcc 13.3.73과 runtime 13.0.96을 함께 포함한다. FlashInfer의 CCCL compatibility check가 이를 거부했다.

이 결과는 모델 shape 미지원 증거가 아니다. 반대로 기존 Riley가 빌드됐다는 사실만으로 외부 backend JIT 환경이 정상이라고 볼 수도 없다. 다음 실행 전에 **검사용 별도 prefix에 일치하는 compiler/runtime/header 조합을 구성**해야 한다. compatibility check를 비활성화하거나 시스템 CUDA/기존 venv를 수정하지 않았다.

[검사 스크립트](raw/flashinfer-source-audit/compatibility_probe.py), [최초 로그](raw/flashinfer-source-audit/result.log), [PATH 수정 후 로그](raw/flashinfer-source-audit/result-path-fixed.log)를 보존했다. 두 프로세스는 종료됐으며 진행 중인 검사로 취급하지 않는다. 앞 절의 “GPU 실행 없음”은 소스 대조 시점 기록이고, 이 후속 단계에서는 GPU tensor 준비와 FP32 reference 계산까지 수행했다. Serving binary와 기존 correctness 검사는 변경하지 않았다.

### Toolchain 정렬 후 실행 결과

별도 `flashinfer-compatibility/toolchain130` prefix에 PyPI의 nvcc/CRT/NVVM 13.0.88 및 runtime 13.0.96 wheel을 SHA256 확인 후 풀었다. pip 환경을 교체하지 않았다. 해당 prefix에 lib64 및 libcudart linker symlink를 구성했다. compiler/header mismatch 검사는 유지했다. [Package receipt](raw/flashinfer-source-audit/toolchain130-receipt.json).

위와 동일한 C32/context398 synthetic paged 입력에서 CUDA-core decode와 FA2 tensor-core decode 모두 실행됐다. 무작위 physical page 순서를 사용했다.

| 경로 | FP32 reference 대비 max absolute error | RMSE | eager/graph exact | Q 수정 후 eager/graph exact |
|---|---:|---:|---|---|
| CUDA-core | 0.00171393 | 0.000196524 | 일치 | 일치 |
| FA2 tensor-core | 0.00172445 | 0.000221221 | 일치 | 일치 |

두 출력 모두 finite지만, FP32 결과를 BF16으로 반올림한 값과 byte-exact하지는 않았다. 이 결과는 **실행 가능성과 고정 shape graph의 입력 갱신 확인**이다. Riley kernel 또는 full-model logits와 대조하지 않았고, 수치 허용치를 사전에 정의한 acceptance test도 아니므로 correctness 승인으로 표현하지 않는다. 동적 KV 길이·batch 변화, RoPE 통합, NaN/Inf 정책도 아직 미검증이다.

환경상 JIT 문제는 해결됐다. 다음 기술적 판정은 Riley attention 출력과 동일 입력 대조 및 동적 metadata graph 계약이다. 성능 수치를 측정하거나 serving에 통합하지 않았다. [최종 결과](raw/flashinfer-source-audit/result-complete-prefix.log). 이전 linker 실패도 [원본 로그](raw/flashinfer-source-audit/result-toolchain130.log)에 보존했다.
