"""Split the isolated attention fast-math unit from other model operators."""
from pathlib import Path
r=Path('/tmp/riley-g04-native-profile-source-260911')
p=r/'kernels/src/graph_numerics.cu';s=p.read_text()
a=s.index('__global__ void compiled_norm');b=s.index('cudaError_t enqueue_compiled_attention');c=s.index('namespace riley_cuda_internal',a)
(r/'kernels/src/graph_numerics_precise.cu').write_text('#include "ffi_internal.hpp"\n#include <cuda_bf16.h>\n'+s[a:c]+s[c:b]+'}\n')
p.write_text(s[:a]+'namespace riley_cuda_internal {\n'+s[b:])
p=r/'kernels/CMakeLists.txt';s=p.read_text().replace('    src/graph_numerics.cu','    src/graph_numerics_precise.cu\n    src/graph_numerics.cu');p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('native-trace-injected.bin','native-trace-precise.bin');p.write_text(s)
