#define main primitive_main
#include "shared_projection_v16.cu"
#undef main
void grouped(){
 auto*x=alloc<__nv_bfloat16>(8*576);auto*w=alloc<__nv_bfloat16>(1536*576);auto*tw=alloc<__nv_bfloat16>(1536*576);auto*parts=alloc<float>(3*8*960);auto*live=alloc<uint32_t>(1);
 __nv_bfloat16 *actual[3],*expected[3];for(int i=0;i<3;++i){actual[i]=alloc<__nv_bfloat16>(8*1536+16);expected[i]=alloc<__nv_bfloat16>(8*1536);}
 for(int i=0;i<8*576;++i)x[i]=__float2bfloat16_rn(float((i*17)%127-63)/128);
 for(int i=0;i<1536*576;++i)w[i]=__float2bfloat16_rn(float((i*13)%251-125)/256);
 for(int n=0;n<1536;n+=8)for(int k=0;k<576;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)tw[((n/8)*36+k/16)*128+hi*64+lane*2+z]=w[(n+lane/4)*576+k+hi*8+2*(lane%4)+z];
 for(int rows=0;rows<=9;++rows)for(int mode=0;mode<2;++mode){
  *live=rows;for(int j=0;j<3;++j)for(int i=0;i<8*1536+16;++i)actual[j][i]=__ushort_as_bfloat16(0x4123);
  if(rows>0&&rows<=8)for(int row=0;row<rows;++row){
   if(mode==0){enqueue_decode_projection<576,576,192>(0,x+row*576,w,expected[0]+row*576,parts);for(int j=1;j<3;++j)enqueue_decode_projection<192,576,192>(0,x+row*576,w+j*192*576,expected[j]+row*192,parts);}
   else for(int j=0;j<2;++j)enqueue_tile_projection<1536,576,0>(0,x+row*576,tw,expected[j]+row*1536,parts);
  }
  if(mode==0)enqueue_shared_qkv(0,x,w,w+192*576,w+384*576,actual[0],actual[1],actual[2],parts,live);
  else shared_gate_up<<<dim3(192,2),32>>>(x,tw,tw,actual[0],actual[1],live);
  OK(cudaDeviceSynchronize());
  for(int j=0;j<(mode?2:3);++j){int n=mode?1536:(j?192:576),valid=rows>0&&rows<=8?rows*n:0;
   for(int i=0;i<8*1536+16;++i)if(__bfloat16_as_ushort(actual[j][i])!=(i<valid?__bfloat16_as_ushort(expected[j][i]):0x4123)){fprintf(stderr,"group mismatch rows=%d mode=%d output=%d i=%d\n",rows,mode,j,i);exit(5);}
  }
 }
 for(int j=0;j<3;++j){OK(cudaFree(actual[j]));OK(cudaFree(expected[j]));}for(void*p:{(void*)x,(void*)w,(void*)tw,(void*)parts,(void*)live})OK(cudaFree(p));
 puts("grouped QKV/gate-up active1..8 invalid0,9 exact=true inactive_guard=true");
}
int main(){grouped();return 0;}
