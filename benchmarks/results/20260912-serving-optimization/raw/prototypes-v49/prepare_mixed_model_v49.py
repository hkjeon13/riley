from pathlib import Path
import shutil
r=Path('/tmp/riley-opt-260912');d=r/'mixed-model-v49';d.mkdir();old=r/'packed-prefill-loaded-v48'
shutil.copy2(r/'mixed-attention-v49/mixed_attention_v49.cuh',d/'mixed_attention_v49.cuh')
s=(old/'packed_prefill_rope_v48.cuh').read_text().replace('active>4','active>32');(d/'mixed_rope_v49.cuh').write_text(s)
s=(old/'packed_prefill_model_v48.cuh').read_text().replace('packed_prefill_attention_v48.cuh','mixed_attention_v49.cuh').replace('packed_prefill_rope_v48.cuh','mixed_rope_v49.cuh').replace('active<=4','active<=32').replace('riley_packed_prefill_attention::attention<<<dim3(max((capacity+7)/8,min(capacity,31U)),9,4)','riley_mixed_attention::compact_attention<<<dim3(128,9)');(d/'mixed_model_v49.cuh').write_text(s)
s=(old/'probe.cu').read_text().replace('packed_prefill_model_v48.cuh','mixed_model_v49.cuh').replace('physical=256','physical=1024').replace('old_host,4*old_bytes','old_host,32*old_bytes').replace('i<64;++i)put(p,b+128+i*4,owner*64+(i*5+17)%64)','i<32;++i)put(p,b+128+i*4,owner*32+(i*5+17)%32)')
a=s.index('int lists[8][4]');b=s.index('  unsigned packed_cap',a)
s=s[:a]+'''int lists[4]={1,7,31,33};
 for(int pattern=0;pattern<6;++pattern)for(unsigned owners:{1U,4U,8U,16U,32U}){unsigned counts[32],starts[32],prompts[32],total=0,max_count=0;for(unsigned i=0;i<owners;++i){unsigned n=1;if(pattern==0)n=i==0?128:1;if(pattern==1)n=i<4?128:1;if(pattern==3)n=lists[i%4];if(pattern==4)n=i==0?398:1;if(pattern==5)n=31;counts[i]=n;starts[i]=n==1?128+13*(i%4):std::array<unsigned,4>{0,13,73,129}[i%4];prompts[i]=starts[i]+counts[i]+(n>1&&i%2?7:0);total+=counts[i];max_count=std::max(max_count,counts[i]);}
'''+s[b:]
oldtext='unsigned offset=0;for(unsigned i=0;i<owners;++i){packet(host,true,i,i,starts[i],counts[i],prompts[i],offset);offset+=counts[i];}'
newtext='unsigned offset=0,tiles=0;for(unsigned i=0;i<owners;++i){packet(host,true,i,i,starts[i],counts[i],prompts[i],offset);put(host,128+i*1664+68,tiles);offset+=counts[i];tiles+=counts[i]<32?counts[i]:(counts[i]+7)/8;}put(host,96,tiles);'
assert oldtext in s;s=s.replace(oldtext,newtext)
# Deliberately omit misleading timing against sequential per-owner execution.
a=s.index('  if(timing&&');b=s.index('\n }\n CK(cudaStreamDestroy',a);s=s[:a]+s[b:]
s=s.replace('packed pattern','mixed_geometry pattern').replace('packed full model cases','mixed geometry full model cases')
(d/'probe.cu').write_text(s)
print('Prepared30 full-model geometry cases; not a serving/decoder-route timing claim')
