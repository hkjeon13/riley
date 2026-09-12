// Internal variable-prefill primitive, not a serving entry point.
// Caller retains all parents and validates a unique logical-to-physical page map,
// its full extent, and allocation sizes before enqueue; no host publication here.
#pragma once
__global__ void prefill_shape_rope_kv(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,
 const uint32_t* pages,uint32_t start,uint32_t rows){
 const uint32_t row=blockIdx.y;if(row>=rows)return;
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;
 const uint32_t pos=start+row;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[pos*32+dim]));
 const __nv_bfloat16* src=head<9?q+row*576:k+row*192;
 int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 const __nv_bfloat16 first=__float2bfloat16_rn(a*c-b*s),second=__float2bfloat16_rn(b*c+a*s);
 if(head<9){qo[row*576+base+dim]=first;qo[row*576+base+dim+32]=second;}
 else{
  uint32_t destination=((pages[pos/16]*3+head-9)*16+pos%16)*64+dim;
  keys[destination]=first;keys[destination+32]=second;
  values[destination]=v[row*192+base+dim];values[destination+32]=v[row*192+base+dim+32];
 }
}
cudaError_t launch_prefill_shape_rope_kv(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,
 const uint32_t* pages,uint32_t start,uint32_t rows,uint32_t context){
 if(!q||!k||!v||!qo||!keys||!values||!cos||!sin||!pages||!rows||rows>1024||!context||context>4096||start>=context||rows>context-start)return cudaErrorInvalidValue;
 prefill_shape_rope_kv<<<dim3(2,rows),256,0,stream>>>(q,k,v,qo,keys,values,cos,sin,pages,start,rows);
 return cudaGetLastError();
}
