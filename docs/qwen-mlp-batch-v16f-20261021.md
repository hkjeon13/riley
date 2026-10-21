# Coalesced prefill v16F: serving-capacity 검사에서 중단

2026-10-21 KST. **미채택. 정상 C16/C32 비교는 시작하지 않았다.** 원래 정확도 검사는 통과했으나 추가한576-token serving-capacity 검사에서 cold scratch 생성자가 거부됐다. 수치 불일치로 기록하거나 처리량 결과를 부여하지 않는다.

## 구조 변경

v16C의 단일 prefill용 merged Gate/Up을 여러 prefill 그룹의 합65..576행으로 확대했다. 각 원래 그룹은65..130행이며 N19456/K2560, pinned Lt21/20/11/0/0/0/0을 사용한다. decode Lt31 계획과 prefix MLP Graph를 분리하고, 대상 밖 입력은 기존 경로를 사용한다. v16B의 native-order QK shuffle/PV grid도 포함한 combined 후보다.

merged 결과를 위한 raw scratch를 weights upload 후, merged plan/Graph 준비 전에 확보한다. 필요 용량은 기존 intermediate의2배와576×19456×2B 중 작은 값이다. 기존4096-token teacher의 이미 큰 storage는 재사용한다. 정상576-token 경로는 약11.2MB에서22.4MB로 커지므로 cold resize 검증이 별도로 필요했다. allocation을 성공시킨 뒤 교체하고 Graph/total byte report를 갱신하며 hot allocation은 추가하지 않는다.

## 검증과 발견한 오류

원래 native/HF BF16 파일4,480개가 모두 일치했다. CPU524통과/40ignored, GPU contract6개 통과: 기존120 alignment/view 사례와 새 raw merged66사례를 포함한다. 새 사례는11개 행수65/96/115/128/129/130/192/256/384/512/576 ×3입력 ×2실행 방식이다. 큰 행의 원래 기준은65..130행 balanced native 그룹이다.

추가 검사는 같은 원래 natural128 HF/native reference를 사용하며 R1..32, prefill chunks4, metadata budget576으로 실행한다. 원래 teacher가 큰4096-token storage를 재사용해 지나치던 cold resize를 실제로 발생시킨다. tolerance와 reference는 변경하지 않았다.

F의 guard는 `!owned_decode_segments.is_empty()`를 이미 준비된 Graph로 간주했다. 하지만 constructor가 미리 만드는 segment slot의 graph는None이고 warmed/replayed도false다. 이 상태에서 `native_prefill_merged_scratch: raw scratch must be sized before Graph preparation`으로 **logits 파일을 만들기 전에** 실패했다(return101). 정상 측정 queue는 prerequisite 실패로 중단했고 완료 정상 arm은0이다. 실패를 숨기거나 accuracy 기준을 낮추지 않았다.

G는 실제 `graph.is_some() || warmed || replayed` 상태를 검사하도록 수정하고 빈 target에서 독립 재빌드·전체 재검증한다. 실제 captured step/Graph나 이미 사용한 slot은 계속 거부한다. F의 이미 통과한 결과를 수정 후 바이너리의 증명으로 재사용하지 않는다.

## 증거와 환경

GPU7에서10개 observer receipt를 보존한다. 원래9개 검사는 성공, 추가 bound 검사는 실패다. 전체 device peak는각22,528MiB 이하였으며 GPU6 외부 작업과 original GPU6/7 holder PID/context는 유지했다. 모든 실행 파일·scratch는 EFS, 기존4개 C02 unit fixture만 승인된 private/tmp에서 재검사했다.

[판정·resource](../benchmarks/evidence/20261021-mlp-batch-v16f/qualification-decision.json) · [source lineage](../benchmarks/evidence/20261021-mlp-batch-v16f/source-lineage.json) · [원자료 ZIP](../benchmarks/evidence/20261021-mlp-batch-v16f.zip):402,238B/159entries, SHA256 `fedbc0219b7a95e0bad545027cb63324818792542892e85bdbe6de049f1395a5`. 패치, guard 실패 stderr, 원래 수치 audit, queue 실패와937-source manifest를 보존했다. 대형 BF16/ELF는 heavy index로 EFS에 연결한다. [로컬 검증](../benchmarks/evidence/20261021-mlp-batch-v16f/local-archive-verification.json)은 CRC/모든entry SHA·size 및 Git baseline에서 패치 적용 후6개 변경 파일의 hash 재구성을 확인했다.
