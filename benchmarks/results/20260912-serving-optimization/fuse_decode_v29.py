from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/decode_shared.cuh';s=p.read_text().replace('template<int N,int K,int Interval,bool Tiled>\n__device__','template<int N,int K,int Interval,bool Tiled,bool Compact=false>\n__device__').replace('const int lane=threadIdx.x,g=','const int lane=threadIdx.x%32,g=',1).replace('else out[g*N+base+2*t+j]=v;','else if constexpr(Compact)out[g*8+2*t+j]=v;\n  else out[g*N+base+2*t+j]=v;');s+='''
// Two warps compute independent gate/up tiles, then share rounded results.
template<bool Tiled>
__global__ void shared_gate_up_swiglu(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* live){
 uint32_t rows=*live;if(rows<1||rows>8)return;
 int warp=threadIdx.x/32;__shared__ __nv_bfloat16 partials[2][64];
 shared_projection_compute<1536,576,0,Tiled,true>(x,warp?up:gate,nullptr,partials[warp],live,blockIdx.x,0);
 __syncthreads();
 int row=threadIdx.x/8,col=threadIdx.x%8;if(row<rows){float g=__bfloat162float(partials[0][threadIdx.x]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][threadIdx.x]));}
}
''';p.write_text(s)
p=r/'kernels/src/decode_shared_model.cuh';s=p.read_text();a=s.index('// Fixed-capacity GEMM');s=s[:a]+'''
__device__ __forceinline__ __nv_bfloat16 qkv_rounded(const float* p,int n,int row,int col){
 float value=0.;for(int chunk=0;chunk<3;++chunk)value+=p[chunk*8*n+row*n+col];return __float2bfloat16_rn(value);
}
__global__ void qkv_merge_rope(const float* parts,__nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,const uint32_t* pages,const uint32_t* shape,const uint32_t* active){
 uint32_t rows=*active,row=blockIdx.y;if(rows<1||rows>8||row>=rows)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=384)return;
 shape+=row*416;pages+=row*416;uint32_t pos=shape[1];int head=i/32,dim=i%32;
 const float* src=head<9?parts:parts+3*8*576;int n=head<9?576:192,base=(head<9?head:head-9)*64;
 float a=__bfloat162float(qkv_rounded(src,n,row,base+dim)),b=__bfloat162float(qkv_rounded(src,n,row,base+dim+32));
 float c=__bfloat162float(__float2bfloat16_rn(cos[pos*32+dim])),sn=__bfloat162float(__float2bfloat16_rn(sin[pos*32+dim]));
 auto first=__float2bfloat16_rn(a*c-b*sn),second=__float2bfloat16_rn(b*c+a*sn);
 if(head<9){qo[row*576+base+dim]=first;qo[row*576+base+dim+32]=second;}
 else {uint32_t dst=((pages[pos/16]*3+head-9)*16+pos%16)*64+dim;keys[dst]=first;keys[dst+32]=second;const float* v=parts+3*8*(576+192);values[dst]=qkv_rounded(v,192,row,base+dim);values[dst+32]=qkv_rounded(v,192,row,base+dim+32);}
}
'''+s[a:];a='enqueue_shared_qkv(stream,b(1),w(base+1),w(base+2),w(base+3),b(2),b(5),b(6),static_cast<float*>(scratch[7]),active);';assert a in s;s=s.replace(a,'shared_qkv_parts<<<dim3(120,3),32,0,stream>>>(b(1),w(base+1),w(base+2),w(base+3),static_cast<float*>(scratch[7]),active);')
a='shared_rope_kv<<<dim3(2,8),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,cos,sin,pages,shape,active);';assert a in s;s=s.replace(a,'qkv_merge_rope<<<dim3(2,8),256,0,stream>>>(static_cast<float*>(scratch[7]),b(3),lk,lv,cos,sin,pages,shape,active);')
a=s.index(' if(tiled)shared_gate_up<<<');b=s.index(' if(tiled)enqueue_shared_projection<576,1536',a);s=s[:a]+''' if(tiled)shared_gate_up_swiglu<true><<<192,64,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(11),active);
 else shared_gate_up_swiglu<false><<<192,64,0,stream>>>(b(1),w(base+6),w(base+7),b(11),active);
'''+s[b:];p.write_text(s)
