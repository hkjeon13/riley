# M1 strided batch retained CUDA Graph 검사

실제 RTX 4090에서 2,178개 case의 그래프를 각각 한 번 캡처·instantiate하고 3번 재실행했다. **총 6,534회 replay**, 최종 M1 기준 BF16 불일치 **0개**다. 121개 가중치 그룹, 세 입력 패턴, batch2/active2·batch4/active4·batch4/active3, compact/padded 간격을 포함한다.

매 case에서 같은 device 주소와 graph exec를 유지하며 입력을 원본→0→원본으로 바꿨다. 출력은 매번 poison으로 덮은 후 graph를 실행했다. 첫/마지막 출력은 바이트 단위로 같아야 하고 마지막 출력은 독립 M1 oracle과 bitwise 비교했다. 가운데 0 입력 출력은 부호 비트를 제외한 수치 0인지 검사했다. 즉 가운데 signed zero까지 독립 oracle과 bitwise 비교한 것은 아니다. Output padding과 외부 guard, 입력·가중치 보존도 통과했다.

모든 graph exec·graph·layout을 정리한 뒤 기존 allocation/plan/context 정리를 통과했다. 실행 전후 기록된 private libcuda 경로와 SHA도 원격에서 재확인했다. 로컬 원본 JSONL SHA는 receipt와 일치한다.

이 검사는 GEMM 구간만 캡처했다. H2D는 그래프 외부에서 수행했고 replay마다 synchronize했다. 활성 행 수에 따라 graph bucket을 바꾸는 scheduler 전이, descriptor packet staging, 전체 decode graph, serving latency·throughput의 증거가 아니다.

다음 통합 batch는 (1) M1 strided plan과 descriptor의 cold owner 수명, (2) 이미 검사한 다중 행 attention·precise 연산과 전체 graph 연결, (3) scheduler reservation 및 전체 결과 검증 후 일괄 commit을 함께 구현한다. 런타임에서 M1 호출을 행별로 반복하는 우회 경로를 성능 개선으로 간주하지 않는다. 기존 C1 후보 및 baseline은 보존한다.

[원본 receipt](raw/multisequence-strided-graph-v1/receipt.json), [CUDA Graph 진단 코드](multisequence_strided_graph_v1.cpp), [build·실행 도구](run_multisequence_strided_graph_v1.py).
