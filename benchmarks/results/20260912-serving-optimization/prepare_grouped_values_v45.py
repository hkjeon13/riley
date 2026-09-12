from pathlib import Path
import shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';out=r/'grouped-values-v45';out.mkdir()
shutil.copy2(r/'decode_grouped_values_v45.cuh',out)
s=(src/'kernels/tests/shared16_primitive_probe.cu').read_text().replace('#include "decode_shared16_attention.cuh"','#include "decode_grouped_values_v45.cuh"').replace('riley_shared16_attention::enqueue','riley_shared16_grouped::enqueue')
(out/'probe.cu').write_text(s)
