from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'attention-pairs-v57'
s=(d/'variant3.cuh').read_text().replace('riley_mixed_v57_3','riley_mixed_v57_4')
s=s.replace('float (*scores)[128]','float (*scores)[132]').replace('__nv_bfloat16 (*probs)[128]','__nv_bfloat16 (*probs)[136]').replace('float (*exponentials)[128]','float (*exponentials)[132]')
s=s.replace('scores[TileRows][128]','scores[TileRows][132]').replace('float (*exps)[128]','float (*exps)[132]').replace('probs[TileRows][128]','probs[TileRows][136]')
(d/'variant4.cuh').write_text(s)
for name in ['mixed.cu','boundary.cu']:
 p=d/name;s=p.read_text().replace('#include "variant3.cuh"','#include "variant3.cuh"\n#include "variant4.cuh"').replace('variant:{0,1,2,3}','variant:{0,1,2,3,4}')
 lines=[]
 for line in s.splitlines():
  lines.append(line)
  if line.strip().startswith('if(variant==3)'):lines.append(line.replace('variant==3','variant==4').replace('riley_mixed_v57_3','riley_mixed_v57_4'))
 s='\n'.join(lines)+'\n'
 s=s.replace('graph[4]','graph[5]').replace('exec[4]','exec[5]').replace('variant<4','variant<5').replace('order<4','order<5').replace('3-order','4-order').replace('i<4;++i){CK(cudaGraphExecDestroy','i<5;++i){CK(cudaGraphExecDestroy').replace('four_geometries_exact=true','five_variants_exact=true')
 p.write_text(s)
