#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
__global__ void compiled_rope(const __nv_bfloat16* q,const __nv_bfloat16* k,__nv_bfloat16* qo,__nv_bfloat16* ko,const float* cos,const float* sin,const uint32_t* pos){
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;__nv_bfloat16* dst=head<9?qo:ko;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 dst[base+dim]=__float2bfloat16_rn(a*c-b*s);dst[base+dim+32]=__float2bfloat16_rn(b*c+a*s);
}

#include "prefill_shape_rope_kv.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* upload(const std::vector<T>& a){T* p;ck(cudaMalloc(&p,a.size()*sizeof(T)));ck(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
void run(int rows,int start){
 const int context=1152,blocks=72;std::vector<unsigned short>q(rows*576),k(rows*192),v(rows*192),qo(q.size()),ko(k.size()),actual(q.size()),keys(blocks*3*16*64,0x55aa),values(keys),got(keys),gotv(keys);
 unsigned rng=rows+start+123;for(auto* a:{&q,&k,&v})for(auto& t:*a){rng=rng*1664525+1013904223;t=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(rng%65537)-32768)/1024.f));}
 std::vector<float>cos(context*32),sin(cos.size());for(int i=0;i<int(cos.size());++i){cos[i]=cosf(float(i)/37.f);sin[i]=sinf(float(i)/37.f);}
 std::vector<unsigned>pages(blocks),pos(rows);for(int i=0;i<blocks;++i)pages[i]=(i*5+17)%blocks;for(int i=0;i<rows;++i)pos[i]=start+i;
 auto dq=upload(q),dk=upload(k),dv=upload(v),dqo=upload(qo),dko=upload(ko),dactual=upload(actual),dkeys=upload(keys),dvalues=upload(values);
 auto dc=upload(cos),ds=upload(sin);auto dp=upload(pages),dpos=upload(pos);
 for(int i=0;i<rows;++i)compiled_rope<<<2,256>>>((__nv_bfloat16*)dq+i*576,(__nv_bfloat16*)dk+i*192,(__nv_bfloat16*)dqo+i*576,(__nv_bfloat16*)dko+i*192,dc,ds,dpos+i);
 ck(launch_prefill_shape_rope_kv(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)dactual,(__nv_bfloat16*)dkeys,(__nv_bfloat16*)dvalues,dc,ds,dp,start,rows,context));ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(qo.data(),dqo,qo.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ko.data(),dko,ko.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual.data(),dactual,actual.size()*2,cudaMemcpyDeviceToHost));
 for(int row=0;row<rows;++row)for(int i=0;i<192;++i){int p=start+row;int dst=((pages[p/16]*3+i/64)*16+p%16)*64+i%64;keys[dst]=ko[row*192+i];values[dst]=v[row*192+i];}
 ck(cudaMemcpy(got.data(),dkeys,got.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(gotv.data(),dvalues,gotv.size()*2,cudaMemcpyDeviceToHost));
 if(actual!=qo||got!=keys||gotv!=values){fprintf(stderr,"mismatch rows%d start%d\n",rows,start);exit(3);}
 auto invalid=launch_prefill_shape_rope_kv(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)dactual,(__nv_bfloat16*)dkeys,(__nv_bfloat16*)dvalues,dc,ds,dp,context-1,2,context);if(invalid!=cudaErrorInvalidValue)exit(4);
 for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)dqo,(void*)dko,(void*)dactual,(void*)dkeys,(void*)dvalues,(void*)dc,(void*)ds,(void*)dp,(void*)dpos})ck(cudaFree(p));
 printf("{\"rows\":%d,\"start\":%d,\"exact_query_elements\":%zu,\"full_cache_elements\":%zu,\"untouched_pages_checked\":true}\n",rows,start,qo.size(),keys.size()*2);
}
int main(){for(int start:{0,13,128})for(int rows:{1,15,16,17,127,128,129,398,1024})run(rows,start);}
