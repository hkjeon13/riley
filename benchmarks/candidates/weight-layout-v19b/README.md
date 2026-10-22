# Qualified capacity-aware v19B candidate

The exact eight-file AWS-qualified source patch adds model-owned K-major Gate/Up weights, Graph identity/resource leases and capacity<=16 selection. Capacity32 retains the original projection without packed storage. Numerical kernels preserve original native/HF BF16 boundaries. 기존 tail/처리량 채택 기준을 통과하지 못해 기본 runtime은 v15C를 유지한다. B안의 구현은 검증된 미적용 패치로 보존한다.

Display patch normalizes only empty context lines to Git-accepted empty-line notation. The archive retains the raw unified diff. Both reconstruct identical qualified source hashes, and the reconstructed source diff passes whitespace checks. Source lineage and binary identity are recorded beside the patch. No rejected Down/lossless backend is included.
