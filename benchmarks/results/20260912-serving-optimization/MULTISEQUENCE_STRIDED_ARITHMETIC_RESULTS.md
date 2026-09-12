# M1 strided batch 실제 가중치 검사

RTX 4090의 실제 실행에서 **2,178개 case, BF16 word 12,280,320개 비교, 불일치 0개**를 확인했다. 다섯 projection의 기존 M1 algorithm 13(custom 74/75/89)을 유지했다. 121개 가중치 그룹 × 세 입력 패턴 × batch2/active2·batch4/active4·batch4/active3 × compact/padded 두 간격을 검사했다.

각 case는 후보 strided cuBLASLt 호출 한 번의 출력을 별도 buffer에서 실행한 독립 M1 행별 결과와 비교한다. M1 plan의 algorithm/metadata는 유지하고 별도의 batch layout을 생성했다. 메모리 간격의 padding, 외부 guard, 입력과 가중치 보존, finite 출력, Riley allocation 통계 및 모든 자원 정리가 통과했다. Inactive 행은 0 입력으로 계산하고 그 결과도 비교했으며, inactive 행 실행을 생략했다는 주장은 하지 않는다.

이전 M2/M4 strict 후보에서는 workspace arm마다 BF16 word 861개가 달랐다. 이번 결과는 M=1을 유지하는 strided batch를 실제 다중 요청 경로로 통합할 근거다. 다만 아직 retained CUDA graph나 scheduler/server에 통합하지 않았고 serving 성능을 측정하지 않았다. 호출 한 번이 GPU kernel 한 개라는 주장도 하지 않는다.

Native 실행 전후 기록된 driver·CUDA runtime·cuBLASLt mapping을 파일 해시와 inode로 재검증했다. 실제 driver는 private 580.173.02였으며, 링크 시 참조한 시스템 driver와 런타임에 로드된 driver를 구분한다. 원본 receipt와 별도 검증 receipt를 보존했다.

첫 v1 build는 존재하지 않는 error-stage enum 때문에 실패했고 GPU를 실행하지 않았다. v2는 2,178개 비교를 통과했으나 기존 harness에서 물려받은 workspace 필드명이 새 호출 방식과 맞지 않아, 해당 기록을 수정한 v3를 새 디렉터리에서 다시 실행했다. 최종 v3도 불일치 0이다. 후보는 workspace가 0바이트라 nullptr를 전달하고, 독립 M1 oracle은 기존 ABI의 guarded workspace span을 사용한다.

다음 단계는 이 방식의 plan/layout 수명과 retained graph replay를 검증하고, 이미 검사한 다중 행 attention·precise 연산 및 scheduler와 묶어 실제 다중 요청 실행을 구현하는 것이다.

[원본 receipt](raw/multisequence-strided-arithmetic-v3/receipt.json), [런타임·원본 재검증](raw/multisequence-strided-arithmetic-v3/validation.json), [진단 코드](multisequence_strided_arithmetic_v3.cpp), [실행 도구](run_multisequence_strided_arithmetic_v3.py).
