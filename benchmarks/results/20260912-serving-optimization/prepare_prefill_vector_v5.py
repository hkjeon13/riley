from pathlib import Path
import re
root=Path('/tmp/riley-multisequence-integration-20260912');p=root/'kernels/src/graph_numerics_precise.cu';s=p.read_text()
a=s.index('template<int N,int K,int Interval,int Warps>\n__global__ void gemm_prefill_m16');b=s.index('\nnamespace riley_cuda_internal {',a)
base=s[a:b];new=base.replace('gemm_prefill_m16(', 'gemm_prefill_m16_vector(')
new=new.replace(' for(int depth=0;depth<K;depth+=16){',' // Keep the depth recurrence compact; MMA and BF16 chunk-round order are unchanged.\n #pragma unroll 1\n for(int depth=0;depth<K;depth+=16){')
pat=r'pack\((x|w)\[([^\]]+)\],\1\[([^\]]+)\]\)'
count=0
def replace(m):
 global count
 count+=1
 return '*reinterpret_cast<const uint32_t*>('+m[1]+'+('+m[2]+'))'
new=re.sub(pat,replace,new);assert count==6
new='// CUDA allocations and fixed even BF16 offsets guarantee four-byte alignment.\n// Packed loads replace paired scalar loads; no floating-point operations change.\n'+new
s=s[:b]+'\n'+new+s[b:]
old=' gemm_prefill_m16<N,K,Interval,Warps><<<dim3(N/(8*Warps),8),32*Warps,0,s>>>'
s=s.replace(old,' if constexpr(N==192){\n'+old,1)
s=s.replace('position);return cudaGetLastError();\n}\ncudaError_t enqueue_compiled_prefill_m16_gemm','position);\n }else{\n gemm_prefill_m16_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),8),32*Warps,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,(const uint32_t*)position);\n }\n return cudaGetLastError();\n}\ncudaError_t enqueue_compiled_prefill_m16_gemm',1)
p.write_text(s)
p=root/'crates/riley-runtime/src/llama/graph_decode_multi_session.rs';s=p.read_text();anchor='        hash.update(include_bytes!("../../../../kernels/src/graph_numerics.cu"));';assert anchor in s
s=s.replace(anchor,anchor+'\n        hash.update(include_bytes!("../../../../kernels/src/graph_numerics_precise.cu"));');p.write_text(s)
