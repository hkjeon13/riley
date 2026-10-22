# v19B capacity-aware matched comparison

Archive `20261022-weight-layout-v19b.zip`: 6569950 bytes, 788 entries, SHA256 `2a2d851e0e90a5e643f309c14d7d9cdad4d826667931e713167bc42cb65d9472`. ZIP CRCs, every manifest hash and all eight reconstructed source hashes verified locally. EFS `/mnt/efs/nlp/psyche/riley-weight-layout-v19b-20261022` retains heavy files with SHA/size index.

10096 literal original native/HF BF16 files;524 CPU tests; original5 GPU contracts plus packed span and Graph identity checks;120 alignment/view cases. Original accuracy/tail/resource gates preserved.30 fresh normal arms and4 separate profiles;0 completed normal arms excluded/repeated. Capacity16 gets32 packed layers;capacity32 has no packed copies or kernel launches. See [implementation and all results](../../../docs/qwen-weight-layout-v19b-20261022.md).

기존 tail/처리량 채택 기준을 통과하지 못해 기본 runtime은 v15C를 유지한다. B안의 구현은 검증된 미적용 패치로 보존한다.
