#include "rope-baseline.cuh"
#include "mixed_rope_v49.cuh"
#include "decode_shared32_model.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int packed(int i){int t=(i%1024)/64,d=i%64;return(i/1024)*1024+(d/8)*128+(t/8)*64+(d%8)*8+t%8;}
int main(){
 constexpr int cap=512,physical=8192,extent=physical*3072;
 auto*q=alloc<__nv_bfloat16>(cap*576);auto*k=alloc<__nv_bfloat16>(cap*192);auto*v=alloc<__nv_bfloat16>(cap*192);
 auto*a=alloc<__nv_bfloat16>(cap*576);auto*b=alloc<__nv_bfloat16>(cap*576);
 auto*ka=alloc<__nv_bfloat16>(extent);auto*kb=alloc<__nv_bfloat16>(extent);auto*va=alloc<__nv_bfloat16>(extent);auto*vb=alloc<__nv_bfloat16>(extent);
 auto*c=alloc<float>(4096*32);auto*sn=alloc<float>(4096*32);auto*parts=alloc<float>(3*32*960);auto*m=alloc<uint32_t>(32+32*416);
 for(int i=0;i<4096*32;++i){c[i]=float(i%127)/128;sn[i]=float(i%97)/128;}
 for(int i=0;i<cap*576;++i)q[i]=__float2bfloat16_rn(float(i%251-125)/256);
 for(int i=0;i<cap*192;++i){k[i]=__float2bfloat16_rn(float(i%127-63)/128);v[i]=__ushort_as_bfloat16(uint16_t(i*17));}
 for(int i=0;i<3*32*960;++i)parts[i]=float(i%997-498)/512;
 int cases=0;
 for(bool decode:{false,true})for(int owners:{0,1,4,16,32,33})for(int start:{0,7,15,16,127,398,3968}){
  memset(m,0,(32+32*416)*4);m[5]=owners;int total=0;
  for(int row=0;row<32;++row){auto*sh=m+32+row*416;int count=decode?1:(row==0?128:1);sh[1]=start+count-1;sh[2]=count;sh[4]=start;sh[16]=total;if(row<owners)total+=count;for(int page=0;page<256;++page)sh[32+page]=row*256+(page*17+3)%256;}
  m[9]=total;
  for(int i=0;i<extent;++i)ka[i]=kb[i]=va[i]=vb[i]=__ushort_as_bfloat16(0x4123);
  for(int i=0;i<cap*576;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
  if(decode){
   riley_shared32_model::qkv_merge_rope<false><<<dim3(2,32),256>>>(parts,a,ka,va,c,sn,m+64,m+32,m+5);
   riley_shared32_model::qkv_merge_rope<true><<<dim3(2,32),256>>>(parts,b,kb,vb,c,sn,m+64,m+32,m+5);
  }else{
   mixed_rope_kv_reference<<<dim3(2,cap),256>>>(q,k,v,a,ka,va,c,sn,m,cap);
   mixed_rope_kv_v7<<<dim3(2,cap),256>>>(q,k,v,b,kb,vb,c,sn,m,cap);
  }
  CK(cudaDeviceSynchronize());
  if(memcmp(a,b,cap*576*2)||memcmp(ka,kb,extent*2)){fprintf(stderr,"Q/K mismatch %d %d %d\n",decode,owners,start);return 3;}
  for(int i=0;i<extent;++i)if(__bfloat16_as_ushort(va[i])!=__bfloat16_as_ushort(vb[packed(i)])){fprintf(stderr,"V mismatch %d %d %d i=%d\n",decode,owners,start,i);return 4;}
  ++cases;
 }
 printf("PASS writer_cases=%d packed_V_semantic_exact=true Q_K_exact=true inactive_and_other_pages_unchanged=true contexts_through_4096=true\n",cases);
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)ka,(void*)kb,(void*)va,(void*)vb,(void*)c,(void*)sn,(void*)parts,(void*)m})CK(cudaFree(p));
}
