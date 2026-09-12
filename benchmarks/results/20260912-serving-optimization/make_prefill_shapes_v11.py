from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');old=(r/'kernels/src/graph_numerics_precise.cu').read_text()
s=old[old.index('template<int N,int K,int Interval,int Warps>\n__global__ void gemm_prefill_m16_vector'):old.index('\nnamespace riley_cuda_internal {\ntemplate<int N,int K,int Interval,int Warps>')]
s=s.replace('gemm_prefill_m16_vector','gemm_prefill_shape_vector').replace('const uint32_t* position){','uint32_t rows){').replace(' if(*position!=127)return;','')
for name,which,offset in [('a0','row','2*t'),('a1','next_row','2*t'),('a2','row','2*t+8'),('a3','next_row','2*t+8')]:
 s=s.replace(f'uint32_t {name}=*reinterpret_cast<const uint32_t*>(x+({which}*K+depth+{offset}));',f'uint32_t {name}={which}<rows?*reinterpret_cast<const uint32_t*>(x+({which}*K+depth+{offset})):0;')
s=s.replace(' y[row*N+', ' if(row<rows)y[row*N+').replace(' y[next_row*N+', ' if(next_row<rows)y[next_row*N+')
s='''// Internal primitive for a future variable-prefill owner. Not wired into serving.
// Rows are validated by the launcher; inactive tail rows do not read or write memory.
// Each active row preserves the fixed kernel MMA and intermediate BF16 recurrence.
#pragma once
'''+s+'''
template<int N,int K,int Interval,int Warps>
cudaError_t launch_prefill_shape(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,uint32_t rows){
 if(rows==0||rows>1024||x==nullptr||w==nullptr||y==nullptr)return cudaErrorInvalidValue;
 gemm_prefill_shape_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),(rows+15)/16),32*Warps,0,stream>>>(x,w,y,rows);
 return cudaGetLastError();
}
'''
(r/'kernels/src/prefill_shape_projection.cuh').write_text(s)
base=old[old.index('template<int N,int K,int Interval,int Warps>\n__global__ void gemm_prefill_m16('):old.index('\nnamespace riley_cuda_internal {\ntemplate<int N,int K,int Interval,int Warps>')]
head='''#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
__device__ uint32_t pack(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a))|(uint32_t(__bfloat16_as_ushort(b))<<16);}
'''
test=r'''
#include "prefill_shape_projection.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<int N,int K,int I,int W>void run(int rows,int pattern){
 int padded=((rows+127)/128)*128;
 std::vector<unsigned short>x(padded*K),w(N*K),a(padded*N),b(rows*N+64,0x55aa);unsigned int rng=17+pattern;
 for(auto* v:{&x,&w})for(auto& q:*v){rng=rng*1664525u+1013904223u;float f=float(int(rng%4097)-2048)/2048.f;
 if(pattern==1)f=ldexpf(f,int((rng>>16)%17)-8);
 q=__bfloat16_as_ushort(__float2bfloat16_rn(f));}
 __nv_bfloat16 *dx,*dw,*da,*db;uint32_t *dp;
 // Input has exactly live rows: sanitizer can catch accidental padded reads.
 ck(cudaMalloc(&dx,rows*K*2));ck(cudaMalloc(&dw,w.size()*2));ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));ck(cudaMalloc(&dp,4));
 ck(cudaMemcpy(dx,x.data(),rows*K*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));uint32_t pos=127;ck(cudaMemcpy(dp,&pos,4,cudaMemcpyHostToDevice));
 __nv_bfloat16* ref;ck(cudaMalloc(&ref,x.size()*2));ck(cudaMemcpy(ref,x.data(),x.size()*2,cudaMemcpyHostToDevice));
 for(int offset=0;offset<padded;offset+=128)gemm_prefill_m16<N,K,I,W><<<dim3(N/(8*W),8),32*W>>>(ref+offset*K,dw,da+offset*N,dp);
 ck(launch_prefill_shape<N,K,I,W>(0,dx,dw,db,rows));ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
 for(int i=0;i<rows*N;++i)if(a[i]!=b[i]){fprintf(stderr,"mismatch rows%d N%d K%d pattern%d index%d\n",rows,N,K,pattern,i);exit(3);}
 for(int i=rows*N;i<int(b.size());++i)if(b[i]!=0x55aa){fprintf(stderr,"tail overwrite\n");exit(4);}
 if(launch_prefill_shape<N,K,I,W>(0,dx,dw,db,0)!=cudaErrorInvalidValue||launch_prefill_shape<N,K,I,W>(0,dx,dw,db,1025)!=cudaErrorInvalidValue)exit(5);
 for(void* p:{(void*)dx,(void*)dw,(void*)da,(void*)db,(void*)dp,(void*)ref})ck(cudaFree(p));
 printf("{\"rows\":%d,\"n\":%d,\"k\":%d,\"pattern\":%d,\"exact_elements\":%d,\"tail_guard\":true}\n",rows,N,K,pattern,rows*N);
}
int main(){for(int p:{0,1})for(int m:{1,7,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,398,511,512,1023,1024}){
run<576,576,192,2>(m,p);run<192,576,192,1>(m,p);run<576,576,128,2>(m,p);run<1536,576,0,4>(m,p);run<576,1536,320,2>(m,p);
}}
'''
(r/'kernels/tests').mkdir(exist_ok=True)
(r/'kernels/tests/prefill_shape_projection_probe.cu').write_text(head+base+test)
print('created variable-row projection primitive and exact/tail test')
