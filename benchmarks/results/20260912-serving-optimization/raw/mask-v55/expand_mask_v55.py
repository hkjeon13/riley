from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'decode-mask-v55/probe.cu';s=p.read_text()
s=s.replace('for(int rows:{0,1,4,8,16,24,32,33}){','for(int rows:{0,1,4,8,16,24,32,33})for(int exceptional:{0,1,2,3}){')
s=s.replace('if(timing&&(context<128','if(timing&&(exceptional||context<128')
needle='   for(int i=0;i<32*576+16;++i)a[i]'
insert='''   int pos=shape[1]+(exceptional==3?1:0);bool inject=exceptional&&pos<4096;
   int at=pos<4096?((shape[32+pos/16]*3)*16+pos%16)*64:0;
   auto packed_at=[](int i){int t=(i%1024)/64,dim=i%64;return(i/1024)*1024+(dim/8)*128+(t/8)*64+(dim%8)*8+t%8;};
   auto old=v[at];if(inject)v[at]=__ushort_as_bfloat16(exceptional==2?0x7f80:0x7fc1);vp[packed_at(at)]=v[at];
'''
assert needle in s;s=s.replace(needle,insert+needle)
needle='    for(auto graph_exec:execs)OK(cudaGraphExecDestroy(graph_exec));OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));\n   }'
assert needle in s;s=s.replace(needle,needle+'\n   v[at]=old;vp[packed_at(at)]=old;')
s=s.replace('exact_scores_outputs_padding=true','exact_scores_outputs_padding=true finite_nonfinite_future_masks=true')
p.write_text(s)
