from pathlib import Path
s=Path('/tmp/attention_query_reuse_probe_v5.cu').read_text();a=s.index('__global__ void attention(');b=s.index('// Finite-input prototype',a)
ref=s[a:b].strip().replace('__global__ void attention(', '__device__ __forceinline__ void attention_reference_row(').replace('const uint32_t* blocks){','const uint32_t* blocks,int reference_row){',1).replace('int row=blockIdx.x,','int row=reference_row,',1)
a=s.index('template<int R,');s=s[:a]+ref+'\n'+s[a:]
s=s.replace(' int count=row+1,end=(blockIdx.x+1)*R;',''' int count=row+1,end=(blockIdx.x+1)*R;
 // Conservative magnitude bound also rules out overflowing finite QK products.
 // Slow-path preserves per-query masking when future K/V contains NaN or Inf.
 bool bad=false;
 for(int i=threadIdx.x;i<R*3*64;i+=blockDim.x){
  int qi=((blockIdx.x*R+i/(3*64))*9+kvh*3+(i/64)%3)*64+i%64;
  bad|=(__bfloat16_as_ushort(q[qi])&0x7fff)>=0x5d80;
 }
 for(int i=threadIdx.x;i<end*64;i+=blockDim.x){
  int ki=cache_index(i/64,kvh,i%64,blocks);
  bad|=(__bfloat16_as_ushort(k[ki])&0x7fff)>=0x5d80;
  bad|=(__bfloat16_as_ushort(v[ki])&0x7fff)>=0x5d80;
 }
 if(__syncthreads_or(bad)){
  if(warp<3)for(int local=0;local<R;++local)attention_reference_row(q,k,v,out,128,128,nullptr,blocks,blockIdx.x*R+local);
  return;
 }''')
s=s.replace('seed<24','seed<32')
anchor='  check(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));'
special='''  if(seed==24)q[0]=__ushort_as_bfloat16(0x7fc1);
  if(seed==25)v[((blocks[7]*3)*16+15)*64]=__ushort_as_bfloat16(0x7fc1);
  if(seed==26)k[((blocks[0]*3)*16+3)*64]=__ushort_as_bfloat16(0x7f80);
  if(seed==27)v[((blocks[0]*3)*16+3)*64]=__ushort_as_bfloat16(0xff80);
  if(seed==28)for(auto& x:q)x=__ushort_as_bfloat16(0x7f7f);
  if(seed==29)for(auto& x:k)x=__ushort_as_bfloat16(0x7f7f);
  if(seed==30)for(auto& x:v)x=__ushort_as_bfloat16(0x7f7f);
  if(seed==31)v[((blocks[4]*3)*16)*64]=__ushort_as_bfloat16(0x7fff);
'''
assert anchor in s;s=s.replace(anchor,special+anchor,1)
# Restore the final timing data to an ordinary finite case after edge validation.
a=s.index(' cudaEvent_t start,end;')
s=s[:a]+''' for(auto& x:v)x=__float2bfloat16_rn(.25F);
 check(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));
'''+s[a:]
Path('/tmp/attention_query_reuse_probe_v6.cu').write_text(s)
