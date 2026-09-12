from pathlib import Path
import shutil
r=Path('/tmp/riley-opt-260912');o=r/'greedy-split-v46';o.mkdir();shutil.copy2(r/'greedy_split_v46.cuh',o)
s=(r/'greedy-v46/probe.cu').read_text();s=s.replace('#include <vector>','#include "greedy_split_v46.cuh"\n#include <vector>')
s=s.replace('bf16_argmax_kernel<<<std::min(rows,65535),256>>>(d,result,rows,vocab);','GreedyPartial* partial;CK(cudaMalloc(&partial,rows*((vocab+2047)/2048)*sizeof(GreedyPartial)));split_argmax(0,d,result,partial,rows,vocab);')
s=s.replace('CK(cudaFree(result));CK(cudaFree(d));','CK(cudaFree(partial));CK(cudaFree(result));CK(cudaFree(d));')
s=s.replace('  cudaGraph_t graph[2];','  GreedyPartial* partial;CK(cudaMalloc(&partial,rows*((vocab+2047)/2048)*sizeof(GreedyPartial)));\n  cudaGraph_t graph[2];')
s=s.replace('bf16_argmax_kernel<<<rows,256,0,stream>>>(d,result,rows,vocab);','split_argmax(stream,d,result,partial,rows,vocab);')
s=s.replace('CK(cudaStreamDestroy(stream));CK(cudaFree(d));','CK(cudaStreamDestroy(stream));CK(cudaFree(partial));CK(cudaFree(d));')
(o/'probe.cu').write_text(s)
s=(r/'run_greedy_v46.py').read_text().replace("o=r/'greedy-v46'","o=r/'greedy-split-v46'")
s=s.replace("subprocess.run(['python3',str(r/'prepare_greedy_v46.py')],check=True)",'')
(r/'run_greedy_split_v46.py').write_text(s)
