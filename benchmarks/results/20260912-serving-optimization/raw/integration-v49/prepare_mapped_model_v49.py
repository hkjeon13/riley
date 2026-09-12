from pathlib import Path
import shutil
r=Path('/tmp/riley-opt-260912');old=r/'mixed-model-v49';d=r/'mixed-model-map-v49';d.mkdir()
s=(r/'mixed-attention-map-v49/mixed_attention_v49.cuh').read_text().replace('entry=meta[32+32*416+tile]','entry=meta[32+32*416+1024+tile]');(d/'mixed_attention_v49.cuh').write_text(s)
shutil.copy2(old/'mixed_rope_v49.cuh',d/'mixed_rope_v49.cuh')
s=(old/'mixed_model_v49.cuh').read_text().replace('riley_mixed_attention::compact_attention<<<dim3(128,9)','riley_mixed_attention::mapped_attention<<<dim3(capacity,9)');(d/'mixed_model_v49.cuh').write_text(s)
s=(old/'probe.cu').read_text().replace('packed_bytes=57472','packed_bytes=61568')
oldtext='offset+=counts[i];tiles+=counts[i]<32?counts[i]:(counts[i]+7)/8;'
new='offset+=counts[i];unsigned nt=counts[i]<32?counts[i]:(counts[i]+7)/8;for(unsigned j=0;j<nt;++j)put(host,57472+(tiles+j)*4,(i<<16)|j);tiles+=nt;'
assert oldtext in s;s=s.replace(oldtext,new);(d/'probe.cu').write_text(s)
s=(r/'check_mixed_model_v49.py').read_text().replace("d=r/'mixed-model-v49'","d=r/'mixed-model-map-v49'");(r/'check_mapped_model_v49.py').write_text(s)
print('Prepared mapped full-model fixture with tile map after token slab, extent61568')
