from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'prefill-projection-v51-matched';d.mkdir(exist_ok=True)
(d/'prefill_projection_v51.cuh').write_text((r/'prefill-projection-v51/prefill_projection_v51.cuh').read_text())
s=(r/'prefill-projection-v51/probe.cu').read_text().replace('int cases=0;','int cases=0;int capacity=1024;')
s=s.replace('dim3(N/(8*W),64)','dim3(N/(8*W),(capacity+15)/16)').replace('dim3(N/(8*W),32)','dim3(N/(8*W),(capacity+31)/32)').replace('dim3(N/(8*W),16)','dim3(N/(8*W),(capacity+63)/64)').replace('dim3(6,1024)','dim3(6,capacity)').replace(',1024,meta+2)',',capacity,meta+2)').replace(',meta,1024)',',meta,capacity)')
s=s.replace('for(int rows:{','for(int cap:{512,1024})for(int rows:{')
s=s.replace('   meta[0]=meta[1]=0;meta[2]=rows;','   if(timing&&cap!=(rows<=512?512:1024))continue;capacity=cap;\n   meta[0]=meta[1]=0;meta[2]=rows;')
s=s.replace('\\\"rows\\\":%d,','\\\"capacity\\\":%d,\\\"rows\\\":%d,').replace('N,K,I,Fused,rows,v,pair','N,K,I,Fused,capacity,rows,v,pair')
(d/'probe.cu').write_text(s)
