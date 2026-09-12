from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'mixed-attention-v49';d.mkdir()
s=(r/'packed-prefill-loaded-v48/packed_prefill_attention_v48.cuh').read_text().replace('riley_packed_prefill_attention','riley_mixed_attention')
a=s.index('template<int TileRows>');b=s.index('// Each CTA',a);body=s[a:b].replace('const uint32_t* live_rows=nullptr)', 'const uint32_t* live_rows,uint32_t query_block)').replace('blockIdx.x','query_block');s=s[:a]+body+s[b:]
s=s.replace('active>4','active>32').replace('shape+32,shape+2);','shape+32,shape+2,blockIdx.x);')
pos=s.rfind('\n}')
s=s[:pos]+'''
// Compact tile offsets are prototype metadata at row word17 and header word24.
// V7 authority validation will own these fields before production integration.
__global__ void compact_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta){
 const uint32_t active=meta[5],total=meta[9],tiles=meta[24];
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total)return;
 for(uint32_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x){
  const uint32_t* shape=nullptr;uint32_t local=0;
  for(uint32_t owner=0;owner<active;++owner){
   const uint32_t* row=meta+32+owner*416;const uint32_t count=row[2],offset=row[16],start=row[17],nt=count<32?count:(count+7)/8;
   if(!count||offset>total||count>total-offset||start>tiles||nt>tiles-start)return;
   if(tile>=start&&tile-start<nt){shape=row;local=tile-start;break;}
  }
  if(!shape)return;
  attention_body<8>(q+shape[16]*576,k,v,out+shape[16]*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local);
  __syncwarp();
 }
}
'''+s[pos:];(d/'mixed_attention_v49.cuh').write_text(s)
print(d)
