# Adaptive FFN model integration

Rust session → retained C ABI recorder → CUDA mixed model 경로에 adaptive FFN을 opt-in으로 연결했다. `RILEY_PREFILL_FFN_ADAPTIVE=1`은 projection pipeline 활성화를 요구하며 잘못된 값·지원되지 않는 조합은 거부한다. 483개 기존 parent/packed weight 계약을 그대로 사용하고 추가 persistent allocation은 없다. Graph identity에는 별도 version과 adaptive kernel source가 포함된다. Serving Python 호출은 없다.

Full-model 비교는 기존 projection session을 기준으로398/47/512/191/192/193 길이와 반복128-token prefix를 실행했다. Cold7개+cached4개 요청에서 각3회 teacher-forced token7의 BF16 logits 총3,244,032 bytes가 일치했다. 이는 full-model 연산 검증이며 자연어 자유 생성·수치 품질·serving 비교를 대체하지 않는다. 동일 test의 full-model memcheck0 errors, native 경계/sanitizer 결과는 앞선 별도 보고서에 있다.

CUDA release build가 통과했고 candidate SHA는 `3ae029380abc732a96f65e361548daef92a794ea9a74905dd6e245ccaf4786fc`다. 직전 projection binary SHA `009697527d143357c3b3d1a2daa5d8b783298024ac1f8f2aa4ea388bbd3ad929`를 별도 보존했다. 로컬 server feature test는69 passed/1 ignored이며 CUDA 기능 test로 표기하지 않는다. 장비 없는 Hopper/Blackwell runtime과 multi-GPU 검증은 미완료다.

Native128/160행 회귀를 유지한 채 serving 손익을 확인한다. 현재 C32 shared/unique, prior/control/adaptive/vLLM 역순16 lane, 각256 warmup+8192 retained 요청 비교를 시작했다. GC 통제/tmpfs 조건이며 완료 전 throughput·latency 개선을 주장하지 않는다. Backend default는 유지한다.

모델 검증 종료 후 Blender3개 복구 receipt를 보존했다. 이어지는 serving 측정에서만 같은3개를 잠시 중지하고 종료 시 복구한다. 세 정적 웹 뷰어는 유지한다.
