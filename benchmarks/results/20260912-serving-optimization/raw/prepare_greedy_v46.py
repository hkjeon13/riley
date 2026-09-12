from pathlib import Path
import hashlib,json
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';o=r/'greedy-v46';o.mkdir()
source=(s/'kernels/src/batch_primitives.cu').read_text();fragment=source[source.index('__device__ __forceinline__ void select_argmax_candidate('):source.index('__device__ __forceinline__ bool resolve_row_cache_base(')]
header='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <math_constants.h>\n#include <cmath>\n#include "riley_cuda.h"\nconstexpr uint32_t kThreads=256,kWarpSize=32,kFullWarpMask=0xffffffffU;\n'
(o/'probe.cu').write_text(header+fragment+(r/'greedy_probe_v46_body.cu').read_text())
(o/'source.json').write_text(json.dumps({'kernel_fragment_sha256':hashlib.sha256(fragment.encode()).hexdigest(),'source_path':str(s/'kernels/src/batch_primitives.cu'),'unchanged_kernel_extract':True},indent=2))
