from pathlib import Path
import hashlib,json
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';out=r/'shared32-v47';out.mkdir()
s=(src/'kernels/src/decode_shared16.cuh').read_text().replace('shared16','shared32').replace('sixteen','thirty-two')
a=s.index('template<int N,int K,int Interval,bool Tiled,bool Compact=false>');b=s.index('\ntemplate<int N,int K,int Interval,bool Tiled>\n__global__',a)
compute='''template<int N,int K,int Interval,bool Tiled,bool Compact=false>
__device__ __forceinline__ void shared32_projection_compute(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows,int column,int chunk_index){
 const uint32_t rows=*live_rows;if(rows<1||rows>32)return;
 const int lane=threadIdx.x%32,g=lane/4,t=lane%4,base=column*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=chunk_index*Chunk,end=min(begin+Chunk,K);float d[2][4]={};
 const auto* xp=x+g*K+begin+2*t;
 const auto* wp=Tiled?w+column*(K/16)*128+(begin/16)*128+lane*2:w+(base+g)*K+begin+2*t;
 #pragma unroll 1
 for(int depth=0;depth<Chunk;depth+=64){
  uint32_t a[2][4],aa[2][4],a1[2][4],a3[2][4],b[4],bb[4];
  #pragma unroll
  for(int step=0;step<4;++step){
   int offset=depth+step*16;bool valid=begin+offset<end;
   #pragma unroll
   for(int tile=0;tile<2;++tile){
    const auto* p=xp+tile*16*K;int row=g+tile*16;
    a[tile][step]=valid&&row<rows?*reinterpret_cast<const uint32_t*>(p+offset):0;
    aa[tile][step]=valid&&row<rows?*reinterpret_cast<const uint32_t*>(p+offset+8):0;
    a1[tile][step]=valid&&row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*K+offset):0;
    a3[tile][step]=valid&&row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*K+offset+8):0;
   }
   if constexpr(Tiled){b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128):0;bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128+64):0;}
   else{b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset):0;bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset+8):0;}
  }
  #pragma unroll
  for(int step=0;step<4;++step)if(begin+depth+step*16<end){
   #pragma unroll
   for(int tile=0;tile<2;++tile)riley_prefill_shape::mma(d[tile],a[tile][step],a1[tile][step],aa[tile][step],a3[tile][step],b[step],bb[step]);
  }
 }
 #pragma unroll
 for(int tile=0;tile<2;++tile)for(int half=0;half<2;++half){int row=tile*16+g+half*8;if(row>=rows)continue;
  for(int j=0;j<2;++j){auto value=__float2bfloat16_rn(d[tile][half*2+j]);
   if constexpr(Interval>0)parts[(chunk_index*32+row)*N+base+2*t+j]=__bfloat162float(value);
   else if constexpr(Compact)out[row*8+2*t+j]=value;
   else out[row*N+base+2*t+j]=value;
  }
 }
}
'''
s=s[:a]+compute+s[b:];s=s.replace('rows>16','rows>32').replace('3*16*','3*32*').replace('c*16*n','c*32*n').replace('chunk*16*N','chunk*32*N').replace('(16*N+255)','(32*N+255)').replace('<<<60,256','<<<120,256').replace('partials[2][128]','partials[2][256]').replace('i<128;i+=blockDim.x','i<256;i+=blockDim.x')
(out/'decode_shared32_v47.cuh').write_text(s)
s=(src/'kernels/src/decode_shared16_attention.cuh').read_text().replace('riley_shared16_attention','riley_shared32_attention').replace('active>16','active>32').replace(',144)',',288)');(out/'decode_shared32_attention_v47.cuh').write_text(s)
manifest={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'kernels/src/decode_shared16.cuh',src/'kernels/src/decode_shared16_attention.cuh']};(out/'baseline-source.json').write_text(json.dumps(manifest,indent=2))
