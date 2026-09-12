from pathlib import Path
r=Path('/tmp/riley-opt-260912');old=r/'mixed-attention-v49';d=r/'mixed-attention-map-v49';d.mkdir()
s=(old/'mixed_attention_v49.cuh').read_text();pos=s.rfind('\n}');s=s[:pos]+'''
__global__ void mapped_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta){
 const uint32_t active=meta[5],total=meta[9],tiles=meta[24],tile=blockIdx.x;
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total||tile>=tiles)return;
 const uint32_t entry=meta[32+32*416+tile],owner=entry>>16,local=entry&0xffffU;
 if(owner>=active)return;const uint32_t* shape=meta+32+owner*416;const uint32_t count=shape[2],offset=shape[16],nt=count<32?count:(count+7)/8;
 if(!count||offset>total||count>total-offset||local>=nt)return;
 attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local);
}
'''+s[pos:];(d/'mixed_attention_v49.cuh').write_text(s)
s=(old/'probe-guards.cu').read_text().replace('32+32*416','32+32*416+1024')
s=s.replace('total+=n;tiles+=n<32?n:(n+7)/8;', 'total+=n;unsigned nt=n<32?n:(n+7)/8;for(unsigned j=0;j<nt;++j)m[32+32*416+tiles+j]=(i<<16)|j;tiles+=nt;')
s=s.replace('blocks:{64,128,256}', 'blocks:{64,128,256,1024}').replace('riley_mixed_attention::compact_attention<<<dim3(blocks,9),32>>>', 'if(blocks==1024)riley_mixed_attention::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);else riley_mixed_attention::compact_attention<<<dim3(blocks,9),32>>>')
s=s.replace('graph[4]','graph[5]').replace('exec[4]','exec[5]').replace('variant<4','variant<5').replace('order<4','order<5').replace('3-order','4-order').replace('for(int i=0;i<4;++i){CK(cudaGraphExecDestroy','for(int i=0;i<5;++i){CK(cudaGraphExecDestroy')
s=s.replace('else riley_mixed_attention::compact_attention<<<dim3(32U<<variant,9)', 'else if(variant==4)riley_mixed_attention::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);else riley_mixed_attention::compact_attention<<<dim3(32U<<variant,9)')
s=s.replace('three_geometries_exact','four_geometries_exact');(d/'probe.cu').write_text(s)
print('Prepared direct tile map variant; existing evidence preserved')
