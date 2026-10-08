from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'packed-value-v54';d.mkdir(exist_ok=True)
base=(r/'prefill-shapes-source-v11/kernels/src/mixed_attention_v49.cuh').read_text();(d/'baseline.cuh').write_text(base)
s=base.replace('namespace riley_mixed_attention','namespace riley_attention_v54')
pos=s.index('// Exceptional')
s=s[:pos]+'''__device__ __forceinline__ int packed_index(int pos,int head,int dim,const uint32_t* pages){
 int page=pages?pages[pos/16]:pos/16,t=pos%16;
 return (page*3+head)*1024+(dim/8)*128+(t/8)*64+(dim%8)*8+t%8;
}
__device__ __forceinline__ uint32_t masked_pair(const __nv_bfloat16* p,int pos,int end){
 if(pos>=end)return 0;uint32_t bits=*reinterpret_cast<const uint32_t*>(p);return pos+1<end?bits:bits&0xffffU;
}
'''+s[pos:]
s=s.replace('__device__ void single_query','template<bool Vector>\n__device__ void single_query').replace('template<int TileRows>','template<int TileRows,bool Vector>').replace('__global__ void mapped_attention','template<bool Vector>\n__global__ void mapped_attention').replace('single_query(q,','single_query<Vector>(q,').replace('attention_body<8>(','attention_body<8,Vector>(')
s=s.replace('v[page_base+(pos-token)*(blocks?64:192)+dim]','v[packed_index(pos,kvh,dim,blocks)]').replace('v[base+(pos-token)*(blocks?64:192)+dim]','v[packed_index(pos,kvh,dim,blocks)]')
old='''uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));'''
new='''uint32_t b,bb;
    if constexpr(Vector){int vi=((blocks?blocks[token/16]:token/16)*3+kvh)*1024+block*128;
     b=masked_pair(v+vi+lane*2,token+2*t,end);bb=masked_pair(v+vi+64+lane*2,token+2*t+8,end);
    }else{b=pair(val(token+2*t),val(token+2*t+1));bb=pair(val(token+2*t+8),val(token+2*t+9));}'''
assert old in s;s=s.replace(old,new)
old='uint32_t vb=pair(val(token+2*t),val(token+2*t+1)),vhi=pair(val(token+2*t+8),val(token+2*t+9));'
new='''uint32_t vb,vhi;
    if constexpr(Vector){int vi=((blocks?blocks[token/16]:token/16)*3+kvh)*1024+b*128;
     vb=masked_pair(v+vi+lane*2,token+2*t,end);vhi=masked_pair(v+vi+64+lane*2,token+2*t+8,end);
    }else{vb=pair(val(token+2*t),val(token+2*t+1));vhi=pair(val(token+2*t+8),val(token+2*t+9));}'''
assert old in s;s=s.replace(old,new);(d/'packed.cuh').write_text(s)
p=(r/'attention-state-v52/probe.cu').read_text();a=p.index('#include "variant1.cuh"');b=p.index('#include <cstdio>',a);p=p[:a]+'#include "packed.cuh"\n'+p[b:]
p=p.replace('int main(int argc','int packed_at(int i){int t=(i%1024)/64,d=i%64;return(i/1024)*1024+(d/8)*128+(t/8)*64+(d%8)*8+t%8;}\nint main(int argc')
p=p.replace('auto*v=alloc<__nv_bfloat16>(physical*16*192);','auto*v=alloc<__nv_bfloat16>(physical*16*192);auto*vp=alloc<__nv_bfloat16>(physical*16*192);')
needle='v[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/64);}'
assert needle in p;p=p.replace(needle,needle+'for(int i=0;i<physical*16*192;++i)vp[packed_at(i)]=v[i];')
p=p.replace('if(exceptional)v[at]=__ushort_as_bfloat16(exceptional==2?0x7f80:0x7fc1);','if(exceptional)v[at]=__ushort_as_bfloat16(exceptional==2?0x7f80:0x7fc1);vp[packed_at(at)]=v[at];')
p=p.replace('v[at]=old;','v[at]=old;vp[packed_at(at)]=old;').replace('(void*)v,(void*)a','(void*)v,(void*)vp,(void*)a')
p=p.replace('variant:{0,1,2,3}','variant:{0,1,2}')
p='\n'.join(l for l in p.splitlines() if not l.strip().startswith('if(variant==3)'))+'\n'
for variant,flag in [(1,'false'),(2,'true')]:
 p=p.replace(f'riley_attention_v52_{variant}::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m)',f'riley_attention_v54::mapped_attention<{flag}><<<dim3(capacity,9),32>>>(q,k,vp,b,capacity,m)')
 p=p.replace(f'riley_attention_v52_{variant}::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m)',f'riley_attention_v54::mapped_attention<{flag}><<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,vp,b,benchmark_capacity,m)')
p=p.replace('riley_attention_v52_3::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m)','riley_attention_v54::mapped_attention<true><<<dim3(capacity,9),32>>>(q,k,vp,b,capacity,m)')
p=p.replace('graph[4]','graph[3]').replace('exec[4]','exec[3]').replace('variant<4','variant<3').replace('order<4','order<3').replace('3-order','2-order').replace('i<4;++i){CK(cudaGraphExecDestroy','i<3;++i){CK(cudaGraphExecDestroy').replace('four_geometries_exact=true','three_variants_exact=true packed_value=true')
(d/'probe.cu').write_text(p)
