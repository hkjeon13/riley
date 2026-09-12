from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
for name,marker in [('crates/riley-cuda/src/graph_resources.rs','    pub fn record_v3_shared('),('crates/riley-cuda/src/ffi.rs','    pub(super) fn record_v3_shared(')]:
 p=r/name;s=p.read_text();a=s.index(marker);start=s.index('{',a);depth=1;b=start+1
 while depth:
  depth+=(s[b]=='{')-(s[b]=='}');b+=1
 s=s[:b]+'\n'+s[a:b].replace('record_v3_shared','record_v4_shared').replace('V3 prefill','V4 shared')+s[b:]
 if name.endswith('ffi.rs'):
  line=next(l for l in s.splitlines() if 'fn riley_cuda_graph_resources_record_v3_shared(' in l);s=s.replace(line,line+'\n'+line.replace('record_v3_shared','record_v4_shared'))
 p.write_text(s)
p=r/'kernels/src/graph_resources.cu';p.write_text(p.read_text().rstrip()+'\n')
