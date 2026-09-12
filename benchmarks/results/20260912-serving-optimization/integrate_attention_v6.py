from pathlib import Path
root=Path('/tmp/riley-multisequence-integration-20260912');p=root/'kernels/src/graph_numerics.cu';s=p.read_text();proto=Path('/tmp/attention_query_reuse_probe_v6.cu').read_text();a=proto.index('__device__ __forceinline__ void attention_reference_row(');b=proto.index('#include <vector>',a);new=proto[a:b]
new=new.replace('const uint32_t* blocks){\n int lane=', 'const uint32_t* blocks,const int* position){\n int lane=',1)
anchor=' int count=row+1,end=(blockIdx.x+1)*R;'
assert anchor in new
new=new.replace(anchor,anchor+'''
 if(*position!=127){
  if(warp<3)for(int local=0;local<R;++local)attention_reference_row(q,k,v,out,128,160,position,blocks,blockIdx.x*R+local);
  return;
 }''')
new='\n#if MODE == 6\n// P128 query reuse: four independent queries, shared softmax results, and four\n// output slices per head. The row oracle retains exceptional-value semantics.\n'+new+'\n#endif\n'
a=s.index('\nnamespace riley_cuda_internal {');s=s[:a]+new+s[a:]
anchor=' auto* m=(const uint8_t*)metadata;attention<<<dim3(rows,3),96,0,s>>>'
assert anchor in s
replacement=''' auto* m=(const uint8_t*)metadata;
#if MODE == 6
 if(rows==128){
  attention_queries<4,4><<<dim3(32,3),384,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(const uint32_t*)(m+16),(const int*)(m+4));
  return cudaGetLastError();
 }
#endif
 attention<<<dim3(rows,3),96,0,s>>>'''
s=s.replace(anchor,replacement,1);p.write_text(s)
p=root/'kernels/src/graph_numerics_precise.cu';s=p.read_text();assert 'if constexpr(N==192){' in s;s=s.replace('if constexpr(N==192){','// Whole-model V5 trace regressed Q/O; keep packed loads for gate/up/down only.\n if constexpr(N==192||(N==576&&K==576)){',1);p.write_text(s)
