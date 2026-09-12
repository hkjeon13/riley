from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/decode_shared.cuh';s=p.read_text();a='__global__ void shared_projection_parts(';s=s.replace(a,'__device__ __forceinline__ void shared_projection_compute(').replace('const uint32_t* live_rows){\n const uint32_t rows=', 'const uint32_t* live_rows,int column,int chunk_index){\n const uint32_t rows=',1).replace('base=blockIdx.x*8','base=column*8',1).replace('int begin=blockIdx.y*Chunk','int begin=chunk_index*Chunk',1).replace('parts[(blockIdx.y*8+g)*N','parts[(chunk_index*8+g)*N',1)
pos=s.index('\ntemplate<int N,int K,int Interval>');s=s[:pos]+'''
template<int N,int K,int Interval,bool Tiled>
__global__ void shared_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 shared_projection_compute<N,K,Interval,Tiled>(x,w,parts,out,live_rows,blockIdx.x,blockIdx.y);
}
// Independent projections share one dispatch while retaining each MMA order.
__global__ void shared_gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* g,__nv_bfloat16* u,const uint32_t* live){
 shared_projection_compute<1536,576,0,true>(x,blockIdx.y?up:gate,nullptr,blockIdx.y?u:g,live,blockIdx.x,0);
}
__global__ void shared_qkv_parts(const __nv_bfloat16* x,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* parts,const uint32_t* live){
 int col=blockIdx.x;
 if(col<72)shared_projection_compute<576,576,192,false>(x,q,parts,nullptr,live,col,blockIdx.y);
 else shared_projection_compute<192,576,192,false>(x,col<96?k:v,parts+3*8*576+(col<96?0:3*8*192),nullptr,live,col<96?col-72:col-96,blockIdx.y);
}
__global__ void shared_qkv_merge(const float* parts,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,const uint32_t* live){
 uint32_t rows=*live;if(rows<1||rows>8)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*960)return;
 int row=i/960,col=i%960,n=col<576?576:192;
 const float* src=col<576?parts:parts+3*8*576+(col<768?0:3*8*192);
 __nv_bfloat16* out=col<576?q:(col<768?k:v);col=col<576?col:(col<768?col-576:col-768);
 float value=0.;for(int c=0;c<3;++c)value+=src[c*8*n+row*n+col];out[row*n+col]=__float2bfloat16_rn(value);
}
inline void enqueue_shared_qkv(cudaStream_t s,const __nv_bfloat16* x,const __nv_bfloat16* qw,const __nv_bfloat16* kw,const __nv_bfloat16* vw,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,float* parts,const uint32_t* active){
 shared_qkv_parts<<<dim3(120,3),32,0,s>>>(x,qw,kw,vw,parts,active);
 shared_qkv_merge<<<30,256,0,s>>>(parts,q,k,v,active);
}
''' +s[pos:];p.write_text(s)
p=r/'kernels/src/decode_shared_model.cuh';s=p.read_text();a=s.index(' enqueue_shared_projection<576,576,192,false>');b=s.index(' auto* lk=',a);s=s[:a]+''' enqueue_shared_qkv(stream,b(1),w(base+1),w(base+2),w(base+3),b(2),b(5),b(6),static_cast<float*>(scratch[7]),active);
'''+s[b:]
a=s.index(' if(tiled)enqueue_shared_projection<1536');b=s.index(' riley_prefill_pointwise::swiglu',a);s=s[:a]+''' if(tiled)shared_gate_up<<<dim3(192,2),32,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(8),b(9),active);
 else {
 enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]),active);
 enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]),active);
 }
'''+s[b:];p.write_text(s)
