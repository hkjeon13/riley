from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'attention-columns-v53'
s=(d/'probe.cu').read_text().replace('physical=2048','physical=8192').replace('for(int seed:{1,17})','for(int seed:{99})').replace('for(int pattern=0;pattern<6;++pattern)','for(int pattern:{2,3,4})').replace('for(int owners:{1,4,8,16,32})','for(int owners:{1,4,32})').replace('unsigned prefix=(i%4)*127','unsigned prefix=4096-n').replace('p<64;++p)s[32+p]=i*64+(p*5+17)%64','p<256;++p)s[32+p]=i*256+(p*5+17)%256');(d/'boundary.cu').write_text(s)
s=(r/'analyze_attention_state_v52.py').read_text().replace('attention-state-v52','attention-columns-v53').replace('len(xs)==480','len(xs)==600').replace('range(4)','range(5)').replace('range(1,4)','range(1,5)')
s=s.replace("{'0':'V51 baseline','1':'query tile sized state','2':'score/exponential lifetime alias','3':'both'}","{'0':'V52 baseline','1':'two column groups separate CTAs','2':'four column groups separate CTAs','3':'two column groups co-located warps','4':'four column groups co-located warps'}")
s=s.replace('mapped dispatch identical','mapped input metadata identical; column dispatch variants');(r/'analyze_attention_columns_v53.py').write_text(s)
