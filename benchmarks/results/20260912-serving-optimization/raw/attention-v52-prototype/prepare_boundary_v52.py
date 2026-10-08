from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'attention-state-v52'
s=(d/'probe.cu').read_text().replace('physical=2048','physical=8192').replace('for(int seed:{1,17})','for(int seed:{99})').replace('for(int pattern=0;pattern<6;++pattern)','for(int pattern:{2,3,4})').replace('for(int owners:{1,4,8,16,32})','for(int owners:{1,4,32})').replace('unsigned prefix=(i%4)*127','unsigned prefix=4096-n').replace('p<64;++p)s[32+p]=i*64+(p*5+17)%64','p<256;++p)s[32+p]=i*256+(p*5+17)%256')
s=s.replace('four_geometries_exact=true','four_variants_exact=true context4096=true');(d/'boundary.cu').write_text(s)
