from pathlib import Path
r=Path('/tmp/riley-opt-260912');o=r/'paired-values-v45';o.mkdir()
s=(r/'grouped-values-v45/decode_grouped_values_v45.cuh').read_text().replace('riley_shared16_grouped','riley_shared16_paired')
s=s.replace('const int row=blockIdx.y/3,kvh=blockIdx.y%3;','const int row=blockIdx.y/6,kvh=(blockIdx.y%6)/2,pair=blockIdx.y%2;\n const int heads=pair?1:2,head_base=kvh*3+pair*2,norm_width=pair?32:16;')
s=s.replace('norm_head=lane/8,norm_lane=lane%8','norm_head=lane/norm_width,norm_lane=lane%norm_width').replace('[3][128]','[2][128]')
s=s.replace('norm_head<3','norm_head<heads').replace('g<3','g<heads').replace('(kvh*3+norm_head)','(head_base+norm_head)').replace('(kvh*3+g)','(head_base+g)')
s=s.replace('for(int i=norm_lane;i<end-begin;i+=8)norm_max=fmaxf(norm_max,scores[(head_base+norm_head)*4096+begin+i]);','''for(int i=norm_lane;i<end-begin;i+=32){
   norm_max=fmaxf(norm_max,scores[(head_base+norm_head)*4096+begin+i]);
   if(norm_width==16&&i+16<end-begin)norm_hi=fmaxf(norm_hi,scores[(head_base+norm_head)*4096+begin+i+16]);
  }
  norm_max=fmaxf(norm_max,norm_hi);''')
s=s.replace('float norm_max=__shfl_sync(0xffffffff,maximum,norm_head*4);','float norm_max=__shfl_sync(0xffffffff,maximum,norm_head*4),norm_hi=norm_max;')
s=s.replace('for(int offset=4;offset>0;offset>>=1)','for(int offset=norm_width/2;offset>0;offset>>=1)').replace('norm_max,offset,8)','norm_max,offset,norm_width)').replace('(g<heads?g:0)*8','(g<heads?g:0)*norm_width').replace('i<128;i+=8','i<128;i+=norm_width').replace('dim3(8,48)','dim3(8,96)')
(o/'decode_paired_values_v45.cuh').write_text(s)
for name in ['probe.cu','timing.cu']:
 s=(r/'grouped-values-v45'/name).read_text().replace('decode_grouped_values_v45.cuh','decode_paired_values_v45.cuh').replace('riley_shared16_grouped','riley_shared16_paired').replace('dim3(8,48)','dim3(8,96)');(o/name).write_text(s)
s=(r/'check_grouped_values_v45.py').read_text().replace('grouped-values-v45','paired-values-v45');(r/'check_paired_values_v45.py').write_text(s)
s=(r/'run_grouped_timing_v45.py').read_text().replace('grouped-values-v45','paired-values-v45');s=s.replace("subprocess.run(['python3',str(r/'prepare_grouped_timing_v45.py')],check=True)",'')
(r/'run_paired_timing_v45.py').write_text(s)
