# PR-N01 — 수치·모델·메모리 계약과 최소 실험 준비

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. **로컬 계약·validator·lifecycle controller·schema와 unit test는 구현됐지만, 원격 GPU receipt와 materialized model 자산은 아직 없다.** strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

선행 PR 없음. 큰 실행층 구현 전에 무엇을 같은 모델·정확성·20GB 실험으로 인정할지 고정한다. 구현된 controller와 native fixture 도구를 재사용하며, 첫 원격 receipt가 이 준비가 실행 환경에서도 유효한지 판정한다.

## 변경 묶음

1. 모델 manifest: 135M strict 회귀, SmolLM2-1.7B 연결 검증, Qwen2.5-3B 주 평가의 checkpoint SHA, tensor shape/tie, tokenizer, context/RoPE, QKV bias, head 수를 고정한다. Qwen 0.5B tokenizer profile을 3B 지원으로 간주하지 않는다.
2. `strict`와 `native-bf16` 수치 계약: weight/KV/output/accumulator dtype, cast와 reduction, exp/softmax, bias/RoPE 위치를 기록한다. 오프라인 reference와 고정 corpus를 생성할 수 있지만 서버는 Python에 의존하지 않는다.
3. 수치 판정 manifest: tensor의 abs/relative/norm 오차, teacher-forced logit·분포 차이, perplexity 및 업무 평가 허용폭, 자유 생성 평가를 각각 수치화한다. near-zero relative error 처리와 평가 token 수·seed를 명시한다. reference 재실행 변동 및 기존 native library의 수치 특성을 근거로 정하고 후보 속도를 보기 전에 고정한다. 현재 native profile은 의도적으로 `criteria-pending`이며, 기준이 빈칸인 동안 N02 이후 성능 합격 판정은 불가다.
4. 실행 receipt: 자산 준비/초기화/warmup/timed/종료를 분리하고 bytes 단위 sampled GPU 관측값과 CPU RAM을 기록한다. loading·packing·graph·KV·외부 사용량을 포함하며 추정 불확실성 1GB 이상을 확보한다. 구현된 lifecycle은 `leave_stopped_no_restore`, physical GPU 0 한 장, manifest-relative CWD, duplicate-key 거부를 강제한다. 실행 전에 contract source의 실제 SHA-256과 target model/profile linkage를 대조하고 warmup·timed-serving의 실행 중 poll이 없으면 실패한다. receipt는 연속 GPU high-water를 증명하지 않으며 기존 복구 receipt를 위조하지 않는다.

## 대상 영역

`crates/riley-runtime/src/llama/variable_session.rs`, `crates/riley-model`, `benchmarks/analysis`, 기존 native fixture/manifest 도구. 실제 변경 파일은 코드 조사 후 최소로 확정한다. 이 PR에서 모델 전체 실행기를 범용화하지 않는다.

## 검증과 완료

- 불일치 모델·부족한 메모리·누락 자산·기동 실패·사용자 중단을 오류로 기록하고 원본을 보존한다.
- 고정 fixture 재현, manifest schema 및 메모리 payload 계산을 검증한다. 추정치, sampled observed peak, 연속 high-water 미검증 상태를 별도로 표시한다.
- 이전 Blender 복구 assertion을 사용하는 새 controller가 없는지 확인한다. 과거 archive 검증기는 과거 계약을 유지한다.
- 산출물: 고정 모델/수치/실행 manifest, 대표 shape 목록, 전체 모델 미지원 항목. 성능 개선 주장은 하지 않는다.

## 중단·롤백

수치 기준 또는 모델 자산을 고정할 수 없으면 그 모델의 실험만 보류한다. 기존 controller·strict 실행은 그대로 사용 가능하게 유지한다. 기존 evidence는 수정하지 않는다.
