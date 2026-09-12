from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
s=(r/'kernels/src/decode_tiled.cuh').read_text().split('template<int N,int K,int Interval>\n__global__ void tile_projection_merge')[0]
s=s.replace('#include "decode_shape.cuh"','#include "decode_tiled.cuh"').replace('// One-row projection: independent original BF16 rounding chunks run in parallel.\n// Each warp writes eight output columns. The merge retains the original order.','// One MMA shares weights across up to eight independent decode rows.\n// Live rows are supplied by validated metadata; inactive rows never load/store.')
s=s.replace('template<int N,int K,int Interval>','template<int N,int K,int Interval,bool Tiled>').replace('tile_projection_parts','shared_projection_parts')
s=s.replace('float* parts,__nv_bfloat16* out){','float* parts,__nv_bfloat16* out,const uint32_t* live_rows){\n const uint32_t rows=*live_rows;if(rows<1||rows>8)return;')
s=s.replace('uint32_t a=*reinterpret_cast<const uint32_t*>(x+k+2*t),aa=*reinterpret_cast<const uint32_t*>(x+k+2*t+8);','uint32_t a=g<rows?*reinterpret_cast<const uint32_t*>(x+g*K+k+2*t):0,aa=g<rows?*reinterpret_cast<const uint32_t*>(x+g*K+k+2*t+8):0;')
s=s.replace('uint32_t b=*reinterpret_cast<const uint32_t*>(tile+lane*2),bb=*reinterpret_cast<const uint32_t*>(tile+64+lane*2);','uint32_t b,bb;\n  if constexpr(Tiled){b=*reinterpret_cast<const uint32_t*>(tile+lane*2);bb=*reinterpret_cast<const uint32_t*>(tile+64+lane*2);}\n  else {b=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t);bb=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t+8);}')
s=s.replace('mma(d,a,a,aa,aa,b,bb)','mma(d,a,0,aa,0,b,bb)').replace('if(g==0)','if(g<rows)').replace('blockIdx.y*N+base+2*t+j','(blockIdx.y*8+g)*N+base+2*t+j').replace('out[base+2*t+j]','out[g*N+base+2*t+j]')
s+='''
template<int N,int K,int Interval>
__global__ void shared_projection_merge(const float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 uint32_t rows=*live_rows;if(rows<1||rows>8)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*8*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval,bool Tiled>
inline void enqueue_shared_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts,const uint32_t* live_rows){
 constexpr int chunk=Interval>0?Interval:K;
 shared_projection_parts<N,K,Interval,Tiled><<<dim3(N/8,(K+chunk-1)/chunk),32,0,stream>>>(x,w,parts,out,live_rows);
 if constexpr(Interval>0)shared_projection_merge<N,K,Interval><<<(8*N+255)/256,256,0,stream>>>(parts,out,live_rows);
}
'''
(r/'kernels/src/decode_shared.cuh').write_text(s)
