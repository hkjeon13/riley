from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';k=s/'kernels/src'
h=(r/'prefill-projection-v51-matched/prefill_projection_v51.cuh').read_text();a=h.index('template<int N,int K');b=h.index('template<int Warps,int Tiles>');h=h[:a]+h[b:];h=h.replace('template<int Warps,int Tiles>','template<int Warps>');h=h.replace(' uint32_t rows=*live_rows;',' constexpr int Tiles=1;\n uint32_t rows=*live_rows;');(k/'prefill_fused_gate_v51.cuh').write_text(h)
p=k/'mixed_model_v49.cuh';x=p.read_text().replace('#include "prefill_shape_projection.cuh"','#include "prefill_shape_projection.cuh"\n#include "prefill_fused_gate_v51.cuh"');a=x.index('  if(capacity==1&&tiled)enqueue_tile_projection<1536');b=x.index('  if(capacity==1&&tiled)enqueue_tile_projection<576,1536',a);old=x[a:b];x=x[:a]+'''  if(capacity>1&&tiled){
   riley_prefill51::gate_up<4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),w(base+7),b(11),capacity,shape+2);
  }else{
'''+old+'  }\n'+x[b:];p.write_text(x)
p=s/'crates/riley-cuda/build.rs';x=p.read_text();line='        kernels_dir.join("src/mixed_model_v49.cuh"),';assert line in x;x=x.replace(line,line+'\n        kernels_dir.join("src/prefill_fused_gate_v51.cuh"),');p.write_text(x)
p=s/'crates/riley-runtime/src/llama/graph_decode_full.rs';x=p.read_text();line='include_bytes!("../../../../kernels/src/mixed_model_v49.cuh").as_slice(),';assert line in x;x=x.replace(line,line+'\n            include_bytes!("../../../../kernels/src/prefill_fused_gate_v51.cuh").as_slice(),');p.write_text(x)
