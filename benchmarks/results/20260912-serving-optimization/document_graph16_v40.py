from pathlib import Path
import subprocess
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/tests/README_shared16.md';s=p.read_text();a=s.index('## Remaining serving integration');s=s[:a]+'''## Serving integration completed in V40

Commit `f09ef11281fbbd9a4f341c3d57c9848ecfac4e1b` connects these building blocks through V4 wire validation, graph resource ownership, M16 head, retained Rust session, sixteen-row scheduler selection and explicit `--graph-numerics variable-smol-v4`. V3 remains available and default selection is unchanged. The standalone probe keeps its historical single-reference metadata setup; V4 serving uses canonical token offset26752 and completion magic0x34524d52.

Whole Rust-owned32-request testing matched all2336 full-vocabulary outputs with maximum16 decode rows. V3 regression2336+224 outputs also matched. Final owned and native recorder memchecks reported zero errors. Native recorder covers partial prefill and active1/2/4/8/9/15/16, cold alias/head rejection, stale-output hiding and cleanup. HTTP32 mixed streaming/nonstreaming, invalid bounds and disconnect recovery passed; all allocations and KV reservations returned to zero.

Round48 measured frozen V40 against V37 and vLLM0.27.1 at client/admission4/8/16/32, fixed and natural mixed lengths, two reversed orders, each96 warmup+384 retained requests. All18432 retained requests completed; all12288 Riley references matched. V40 improves high-concurrency throughput but regresses C4/C8; it is an explicit experimental profile, not a general replacement or vLLM performance win. See the campaign status and Round48 analysis for full latency/throughput results. Further result-transfer/validation and prefill profiling is required; long-run stability remains unqualified.
''';p.write_text(s)
p=r/'kernels/tests/README_variable_wire16.md';s=p.read_text();a=s.index('## Pending integration; no serving performance claim');s=s[:a]+'''## V40 integration and remaining performance work

V40 connects the canonical token offset26752, result magic0x34524d52, exact graph extents/aliases, retained transfers, M16 head, source digest, scheduler selection and sixteen output slots. Select it explicitly with `--graph-numerics variable-smol-v4`; V3 remains available. See `README_shared16.md` for GPU/HTTP receipts and the scope of Round48 serving comparisons.

The standalone native16 probe deliberately retains its old reference packet/result format. Its default result template uses the old magic; the compiled V4 serving wrapper explicitly chooses0x34524d52. Do not feed its standalone reference setup into V4 serving.

Round48 found high-concurrency gains but low-concurrency regressions and a remaining vLLM throughput/TPOT gap. V40 is not a completed performance goal. Runtime native intervals include GPU execution, synchronization and host readback; they must not be labeled GPU-only timing. Profile these costs and preserve full correctness/ownership checks in subsequent changes.
''';p.write_text(s)
subprocess.run(['git','diff','--check'],cwd=r,check=True)
subprocess.run(['git','add','kernels/tests/README_shared16.md','kernels/tests/README_variable_wire16.md'],cwd=r,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Document sixteen-row integration evidence and remaining serving gap'],cwd=r,check=True)
