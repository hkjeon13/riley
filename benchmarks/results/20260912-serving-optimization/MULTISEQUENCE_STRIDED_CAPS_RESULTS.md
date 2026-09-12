# M1 strided batch 지원 검사

RTX 4090 UUID `9087e4256acab722b8c9cc0423b39fb0`, cuBLASLt 13.1.1에서 기존 M1 알고리즘의 strided batch 지원을 확인했다. 실제 source `8329c1aeec6e013f581128888c536e15f8bf7300`의 변경되지 않은 `gemm.cu`를 독립 진단 translation unit에 포함해 opaque algorithm을 직접 사용했다. 기존 native archive의 나머지 객체를 연결했다. 생산 소스와 바이너리는 변경하지 않았다.

다섯 projection(QKV, gate/up, O, down, head) × batch 2/4 × compact/padded stride의 **20개 조합 모두 AlgoCheck 성공**이다. 전부 `STRIDED_BATCH_SUPPORT=1`, A/B/C/D 최소 정렬 `[2,2,2,2]`, workspace 0바이트였다. Algo ID 13과 custom option 74/75/89 및 M1 metadata를 확인했고 원래 algorithm/metadata가 바뀌지 않았음을 각 case에서 검사했다. 새 batch descriptor를 만들었으며 기존 descriptor는 수정하지 않았다.

이전 anchored M2/M4 지원 실패와 다른 결과다. 여기서는 각 행의 M=1을 유지하고 batch count만 2/4로 설정한다. Compact stride도 지원 검사에 통과했으므로 정렬 때문에 추가 padding이 반드시 필요하다는 가설은 지지되지 않는다.

GEMM은 실행하지 않았다. Tensor buffer allocation, 수치 동등성, kernel 개수, serving 성능에 관한 증거는 없다. 모든 plan과 context를 닫았고 Riley가 관리하는 live allocation 0을 확인했다. 다음 단계는 실제 121개 가중치 그룹과 다양한 입력에서 batch 실행 결과를 독립 M1 행 실행과 bitwise 비교하는 것이다. 통과 후 retained graph 및 scheduler 통합을 진행한다.

[원본 receipt](raw/multisequence-strided-caps-v1/receipt.json), [원본 출력](raw/multisequence-strided-caps-v1/native.jsonl), [진단 코드](multisequence_strided_caps_v1.cpp), [실행·build 증빙 도구](run_multisequence_strided_caps_v1.py). 로컬 원본 출력 SHA도 receipt와 대조했다.
