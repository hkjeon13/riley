from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';out=r/'query-tile-v44';out.mkdir(exist_ok=True)
s=(src/'kernels/tests/prefill_shape_attention_probe.cu').read_text().replace('#include "prefill_shape_attention.cuh"','#include "prefill_query_tile_v44.cuh"').replace('riley_prefill_shape::launch','riley_prefill_query_tile::launch')
s=s.replace('if(a!=b)exit(3);','if(a!=b){size_t diffs=0;for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]){if(diffs<4)fprintf(stderr,"mismatch rows=%d start=%d at=%zu new=%04x old=%04x\\n",rows,start,i,a[i],b[i]);++diffs;}fprintf(stderr,"total mismatches=%zu\\n",diffs);exit(3);}')
s=s.replace('for(int rows:{1,17,127,129,398})','for(int rows:{1,7,8,9,15,16,17,127,128,129,398})')
(out/'probe.cu').write_text(s)
