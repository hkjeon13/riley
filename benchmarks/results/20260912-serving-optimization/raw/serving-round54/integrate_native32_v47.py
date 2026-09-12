from pathlib import Path
import re
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
def edit(n,f):
 p=src/n;p.write_text(f(p.read_text()))
for a,b in [('decode_shared32_v47.cuh','decode_shared32.cuh'),('decode_shared32_attention_v47.cuh','decode_shared32_attention.cuh')]: (src/'kernels/src'/b).write_text((r/'shared32-qkv-v47'/a).read_text())
s=(src/'kernels/src/decode_shared16_model.cuh').read_text().replace('shared16','shared32').replace('rows>16','rows>32').replace('chunk*16*n','chunk*32*n').replace('3*16*','3*32*').replace('<<<16,','<<<32,').replace('dim3(2,16)','dim3(2,32)').replace('pointwise,16)','pointwise,32)');(src/'kernels/src/decode_shared32_model.cuh').write_text(s)
s=(src/'kernels/src/decode_shared16_result.cuh').read_text().replace('shared16','shared32').replace('batch_bytes=16*','batch_bytes=32*').replace('active>16','active>32').replace('<<<16,','<<<32,').replace('Magic=0x33524d52','Magic=0x35524d52');(src/'kernels/src/decode_shared32_result.cuh').write_text(s)
edit('kernels/src/prefill_shape_model.cuh',lambda s:s.replace('WireRows==8||WireRows==16','WireRows==8||WireRows==16||WireRows==32'))
edit('kernels/src/prefill_shape_packet.hpp',lambda s:s.replace('Rows==8||Rows==16','Rows==8||Rows==16||Rows==32').replace('Rows==8?0x33444d52:0x34444d52,version=Rows==8?3:4','Rows==8?0x33444d52:(Rows==16?0x34444d52:0x35444d52),version=Rows==8?3:(Rows==16?4:5)').replace('(Rows==16&&capacity==16)','(Rows>=16&&capacity==16)||(Rows==32&&capacity==32)')+'\ninline bool valid_v5_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {return valid_variable_shape_packet<32>(p,bytes,physical,capacity);}\n')
edit('kernels/src/compact_shared_result.cuh',lambda s:s.replace('Rows==8||Rows==16','Rows==8||Rows==16||Rows==32').replace('Rows==8?0x33524d52U:0x34524d52U','Rows==8?0x33524d52U:(Rows==16?0x34524d52U:0x35524d52U)'))
s=(src/'kernels/src/graph_numerics_precise.cu').read_text();section=s[s.index('#include "decode_shared16_model.cuh"'):].replace('shared16','shared32').replace('_v4_','_v5_').replace('prefill_model<16>','prefill_model<32>').replace('0x34524d52','0x35524d52');edit('kernels/src/graph_numerics_precise.cu',lambda s:s+'\n'+section)
p=src/'kernels/src/ffi_internal.hpp';s=p.read_text();decls=re.findall(r'cudaError_t enqueue_compiled_v4_[\s\S]*?noexcept;',s);assert len(decls)==4;p.write_text(s.replace(decls[-1],decls[-1]+'\n'+'\n'.join(x.replace('_v4_','_v5_') for x in decls)))
edit('kernels/src/gemm.cu',lambda s:s.replace('rows!=2&&rows!=4&&rows!=8&&rows!=16','rows!=2&&rows!=4&&rows!=8&&rows!=16&&rows!=32'))
p=src/'kernels/src/graph_resources.cu';s=p.read_text();s=s.replace('Rows==8||Rows==16','Rows==8||Rows==16||Rows==32').replace('!Compact||Rows==16','!Compact||Rows==16||Rows==32');s=s.replace('r->variable_rows==16?valid_v4_shape_packet(source,bytes,r->v3_prefill_physical,16):valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8)','r->variable_rows==32?valid_v5_shape_packet(source,bytes,r->v3_prefill_physical,32):(r->variable_rows==16?valid_v4_shape_packet(source,bytes,r->v3_prefill_physical,16):valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8))')
for kind in ['shared_model','shared_result','prefill_model','result_header']:
 s=s.replace(f'Rows==8?enqueue_compiled_v3_{kind}:enqueue_compiled_v4_{kind}',f'Rows==8?enqueue_compiled_v3_{kind}:(Rows==16?enqueue_compiled_v4_{kind}:enqueue_compiled_v5_{kind})')
start=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v4_shared(');end=s.index('\n}',s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v4_shared_greedy(',start))+2;wrappers=s[start:end].replace('_v4_','_v5_').replace('record_variable_shared<16','record_variable_shared<32');s=s[:end]+'\n'+wrappers+s[end:];p.write_text(s)
for filename in ['kernels/include/riley_cuda.h','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs']:
 p=src/filename;s=p.read_text()
 if filename.endswith('.h'):
  ds=re.findall(r'RileyCudaStatus riley_cuda_graph_resources_record_v4_[\s\S]*?;',s)
  assert len(ds)==2;s=s.replace(ds[-1],ds[-1]+'\n'+'\n'.join(x.replace('_v4_','_v5_') for x in ds))
 else:
  # Clone each FFI declaration and Rust method with balanced braces where present.
  pattern=r'(?:pub\([^)]*\)\s+|pub\s+)?(?:unsafe\s+)?fn (?:riley_cuda_graph_resources_)?record_v4_shared(?:_greedy)?\('
  matches=list(re.finditer(pattern,s))
  for m in reversed(matches):
   start=m.start();brace=s.find('{',m.end());semi=s.find(';',m.end())
   if semi>=0 and (brace<0 or semi<brace):end=semi+1
   else:
    depth=1;end=brace+1
    while depth:
     if s[end]=='{':depth+=1
     elif s[end]=='}':depth-=1
     end+=1
   s=s[:end]+'\n'+s[start:end].replace('_v4_','_v5_')+s[end:]
 p.write_text(s)
edit('crates/riley-cuda/build.rs',lambda s:s.replace('println!("cargo:rerun-if-changed=../../kernels/src/decode_shared16.cuh");','println!("cargo:rerun-if-changed=../../kernels/src/decode_shared16.cuh");\n'+''.join(f'    println!("cargo:rerun-if-changed=../../kernels/src/{n}");\n' for n in ['decode_shared32.cuh','decode_shared32_attention.cuh','decode_shared32_model.cuh','decode_shared32_result.cuh'])))
print('native32 integrated')
