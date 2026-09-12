from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'shared32_probe_v47.cu';s=p.read_text();a=s.index('int main(')
s=s[:a]+'''void qkv(bool bench){
 auto*x=alloc<__nv_bfloat16>(32*576);auto*wq=alloc<__nv_bfloat16>(576*576);auto*wk=alloc<__nv_bfloat16>(192*576);auto*wv=alloc<__nv_bfloat16>(192*576);
 auto*aq=alloc<__nv_bfloat16>(32*576);auto*ak=alloc<__nv_bfloat16>(32*192);auto*av=alloc<__nv_bfloat16>(32*192);auto*bq=alloc<__nv_bfloat16>(32*576+16);auto*bk=alloc<__nv_bfloat16>(32*192+16);auto*bv=alloc<__nv_bfloat16>(32*192+16);auto*p=alloc<float>(3*32*960);auto*l=alloc<uint32_t>(3);
 for(int i=0;i<32*576;++i)x[i]=__float2bfloat16_rn(float(i*17%127-63)/128);for(int i=0;i<576*576;++i)wq[i]=__float2bfloat16_rn(float(i*13%251-125)/256);for(int i=0;i<192*576;++i){wk[i]=__float2bfloat16_rn(float(i*7%127-63)/128);wv[i]=__float2bfloat16_rn(float(i*11%251-125)/256);}
 auto launch=[&](bool candidate,cudaStream_t stream){if(candidate)enqueue_shared32_qkv(stream,x,wq,wk,wv,bq,bk,bv,p,l);else{enqueue_shared16_qkv(stream,x,wq,wk,wv,aq,ak,av,p,l+1);if(l[0]>16)enqueue_shared16_qkv(stream,x+16*576,wq,wk,wv,aq+16*576,ak+16*192,av+16*192,p,l+2);}};
 for(int rows=0;rows<=33;++rows){live_set(l,rows);for(int i=0;i<32*576+16;++i)bq[i]=__ushort_as_bfloat16(0x4123);for(int i=0;i<32*192+16;++i)bk[i]=bv[i]=__ushort_as_bfloat16(0x4123);if(rows>=1&&rows<=32)launch(false,0);launch(true,0);OK(cudaDeviceSynchronize());int n=rows>=1&&rows<=32?rows:0;exact(aq,bq,n*576,32*576,"qkv-q",rows);exact(ak,bk,n*192,32*192,"qkv-k",rows);exact(av,bv,n*192,32*192,"qkv-v",rows);}
 if(bench)for(int rows:{4,8,16,24,32}){live_set(l,rows);OK(cudaDeviceSynchronize());timing("qkv-fused",rows,0,launch);}else printf("qkv active0..33 exact guards=true\\n");
 for(void*z:{(void*)x,(void*)wq,(void*)wk,(void*)wv,(void*)aq,(void*)ak,(void*)av,(void*)bq,(void*)bk,(void*)bv,(void*)p,(void*)l})OK(cudaFree(z));
}
'''+s[a:];s=s.replace('gate(bench);attention(bench);','gate(bench);attention(bench);qkv(bench);');p.write_text(s)
# Keep original run intact, write extended evidence to a fresh directory.
import shutil
out=r/'shared32-qkv-v47';out.mkdir()
for name in ['decode_shared32_v47.cuh','decode_shared32_attention_v47.cuh','baseline-source.json']:shutil.copy2(r/'shared32-v47'/name,out/name)
s=(r/'check_shared32_v47.py').read_text().replace("out=r/'shared32-v47'","out=r/'shared32-qkv-v47'");(r/'check_shared32_qkv_v47.py').write_text(s)
