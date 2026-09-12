from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');s=(r/'kernels/src/graph_numerics.cu').read_text()
b=s[s.index('__device__ float exponential'):s.index('\n#if MODE == 6\n// P128 query reuse')]
new=b.replace('__global__ void attention(', '__global__ void attention_shape(').replace('n>160','n>4096')
h='''#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
// Private primitive: caller validates complete cache/map extents and retains parents.
// No serving admission change. Preserve MODE6 accumulation and causal tile order.
namespace riley_prefill_shape {
'''
# Resolve conditional directives by preprocessing only the mode clauses in generated text.
def mode6(text):
 out=[];stack=[];active=True
 for line in text.splitlines():
  t=line.strip()
  if t.startswith('#if '):
   expr=t[4:].replace('MODE','6').replace('||',' or ').replace('&&',' and ')
   val=bool(eval(expr,{'__builtins__':{}},{}));stack.append((active,val));active=active and val
  elif t.startswith('#elif '):
   parent,taken=stack[-1];expr=t[6:].replace('MODE','6').replace('||',' or ').replace('&&',' and ');val=bool(eval(expr,{'__builtins__':{}},{}));active=parent and not taken and val;stack[-1]=(parent,taken or val)
  elif t=='#else':
   parent,val=stack[-1];active=parent and not val
  elif t=='#endif':active=stack.pop()[0]
  elif active:out.append(line)
 assert not stack
 return '\n'.join(out)+'\n'
footer='''
inline cudaError_t launch(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* pages,uint32_t start,uint32_t rows,uint32_t context){
 if(!q||!k||!v||!out||!pages||!rows||rows>1024||!context||context>4096||start>=context||rows>context-start)return cudaErrorInvalidValue;
 attention_shape<<<dim3(rows,3),96,0,stream>>>(q,k,v,out,rows,start+rows,nullptr,pages);
 return cudaGetLastError();
}
} // namespace riley_prefill_shape
'''
(r/'kernels/src/prefill_shape_attention.cuh').write_text(h+mode6(new)+footer)
head='''#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
namespace oracle {
'''+mode6(b)+'}\n'
test=r'''
#include "prefill_shape_attention.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* upload(const std::vector<T>& a){T* p;ck(cudaMalloc(&p,a.size()*sizeof(T)));ck(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
float bf(unsigned short x){return __bfloat162float(__ushort_as_bfloat16(x));}
void run(int rows,int start){
 const int context=4096,blocks=256,end=start+rows;std::vector<unsigned short>q(rows*576),k(context*192),v(k.size()),a(q.size()),b(q.size());std::vector<unsigned>pages(blocks);
 for(int i=0;i<blocks;++i)pages[i]=(i*5+17)%blocks;
 auto at=[&](int p,int h,int d){return ((pages[p/16]*3+h)*16+p%16)*64+d;};
 unsigned rng=start+rows+131;for(auto* xs:{&q,&k,&v})for(auto& x:*xs){rng=rng*1664525+1013904223;x=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(rng%2049)-1024)/2048.f));}
 auto dq=upload(q),dk=upload(k),dv=upload(v),da=upload(a),db=upload(b);auto dp=upload(pages);
 ck(riley_prefill_shape::launch(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)da,dp,start,rows,context));ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));
 if(end<=160){oracle::attention<<<dim3(rows,3),96>>>((__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)db,rows,end,nullptr,dp);ck(cudaDeviceSynchronize());ck(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));if(a!=b)exit(3);}
 // Independent FP64 softmax/dot diagnostic, not full-model numerical qualification.
 // Predeclared absolute bound0.005 for bounded inputs in[-0.5,0.5].
 double maxerr=0.;int sampled=0;
 for(int row:{0,rows/2,rows-1})for(int h=0;h<9;++h){int count=start+row+1;std::vector<double>prob(count);double mx=-1e300,den=0.;
  for(int p=0;p<count;++p){double sum=0;for(int d=0;d<64;++d)sum+=double(bf(q[(row*9+h)*64+d]))*bf(k[at(p,h/3,d)]);prob[p]=sum*.125;mx=std::max(mx,prob[p]);}
  for(double& x:prob){x=exp(x-mx);den+=x;}
  for(int d=0;d<64;++d){double y=0;for(int p=0;p<count;++p)y+=prob[p]*bf(v[at(p,h/3,d)]);double value=bf(a[(row*9+h)*64+d]);if(!std::isfinite(value))exit(4);maxerr=std::max(maxerr,std::abs(value-y/den));++sampled;}
 }
 if(maxerr>0.005){fprintf(stderr,"fp64 error %g\n",maxerr);exit(5);}
 // Poison the suffix for each sampled causal query and run that query alone.
 for(int row:{0,rows/2,rows-1}){auto pk=k,pv=v;int count=start+row+1;for(int p=count;p<context;++p)for(int h=0;h<3;++h)for(int d=0;d<64;++d)pk[at(p,h,d)]=pv[at(p,h,d)]=0x7fc1;
  ck(cudaMemcpy(dk,pk.data(),pk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,pv.data(),pv.size()*2,cudaMemcpyHostToDevice));ck(riley_prefill_shape::launch(0,(__nv_bfloat16*)dq+row*576,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)db,dp,count-1,1,context));ck(cudaDeviceSynchronize());ck(cudaMemcpy(b.data(),db,576*2,cudaMemcpyDeviceToHost));for(int i=0;i<576;++i)if(a[row*576+i]!=b[i])exit(6);
 }
 for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)da,(void*)db,(void*)dp})ck(cudaFree(p));
 printf("{\"rows\":%d,\"start\":%d,\"old_exact_checked\":%s,\"fp64_sampled_elements\":%d,\"max_abs_error\":%.9g,\"causal_suffix_nan_exact\":true}\n",rows,start,end<=160?"true":"false",sampled,maxerr);
}
int main(){for(int start:{0,13,128,1024})for(int rows:{1,17,127,129,398})run(rows,start);run(1,4095);run(1024,3072);}
'''
(r/'kernels/tests/prefill_shape_attention_probe.cu').write_text(head+test)
