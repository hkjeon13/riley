from pathlib import Path
r=Path('/tmp/riley-opt-260912');old=r/'decode-gate-v56';d=r/'decode-gate-layout-v56';d.mkdir(exist_ok=True)
s=(old/'gate.cuh').read_text().replace('lane=threadIdx.x,','lane=threadIdx.x%32,')
s+='''
namespace riley_gate_v56 {
template<int Steps>
__global__ void split_rows(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* active){
 uint32_t rows=*active;int warp=threadIdx.x/32;if(!rows||rows>32||warp*16>=rows)return;
 compute<Steps,1>(x+warp*16*576,gate,up,out+warp*16*1536,min(rows-uint32_t(warp*16),16u));
}
__global__ void hybrid(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* active){
 uint32_t rows=*active;if(!rows||rows>32)return;
 if(rows<=16){if(threadIdx.x<32)compute<2,1>(x,gate,up,out,rows);return;}
 int warp=threadIdx.x/32;__shared__ __nv_bfloat16 partials[2][256];
 shared32_projection_compute<1536,576,0,true,true>(x,warp?up:gate,nullptr,partials[warp],active,blockIdx.x,0);
 __syncthreads();
 for(int i=threadIdx.x;i<256;i+=blockDim.x){int row=i/8,col=i%8;if(row<rows){float g=__bfloat162float(partials[0][i]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][i]));}}
}
}
'''
(d/'gate.cuh').write_text(s)
s=(old/'probe.cu').read_text()
a=s.index('  if(variant==1)');b=s.index('\n };',a)
s=s[:a]+'''  if(variant==1)riley_gate_v56::gate_up<2,true><<<192,32,0,stream>>>(x,g,u,out,active);
  if(variant==2)riley_gate_v56::hybrid<<<192,64,0,stream>>>(x,g,u,out,active);
  if(variant==3)riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(x,g,u,out,active);
  if(variant==4)riley_gate_v56::split_rows<2><<<192,64,0,stream>>>(x,g,u,out,active);'''+s[b:]
(d/'probe.cu').write_text(s)
s=(r/'qualify_gate_v56.py').read_text().replace("d=r/'decode-gate-v56'","d=r/'decode-gate-layout-v56'")
(r/'qualify_gate_layout_v56.py').write_text(s)
