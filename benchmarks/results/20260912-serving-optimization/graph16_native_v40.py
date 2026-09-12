from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/prefill_shape_model.cuh';s=p.read_text().replace('inline cudaError_t enqueue_v3_prefill_model(', 'template<uint32_t WireRows=8>\ninline cudaError_t enqueue_v3_prefill_model(').replace(' if(!scratch||',' static_assert(WireRows==8||WireRows==16,"wire capacity");\n if(!scratch||',1).replace('+13440)', '+(128+WireRows*1664))');p.write_text(s)
p=r/'kernels/src/decode_shared16_result.cuh';s=p.read_text().replace('__global__ void finish(', 'template<uint32_t Magic>\n__global__ void finish(').replace('result[31]=0x33524d52','result[31]=Magic').replace('inline cudaError_t enqueue(', 'template<uint32_t Magic=0x33524d52>\ninline cudaError_t enqueue(').replace(' finish<<<',' finish<Magic><<<');p.write_text(s)
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text().replace('__global__ void v3_prefill_result_header(', 'template<uint32_t Magic>\n__global__ void v3_prefill_result_header(').replace('result[31]=0x33524d52;', 'result[31]=Magic;').replace(' v3_prefill_result_header<<<',' v3_prefill_result_header<0x33524d52><<<')
a=s.index('cudaError_t enqueue_compiled_v3_prefill_model(');b=s.index('\n}',a)+2;prefill=s[a:b].replace('enqueue_compiled_v3_prefill_model','enqueue_compiled_v4_prefill_model').replace('enqueue_v3_prefill_model(', 'enqueue_v3_prefill_model<16>(')
a=s.index('cudaError_t enqueue_compiled_v3_result_header(');b=s.index('\n}',a)+2;header=s[a:b].replace('enqueue_compiled_v3_result_header','enqueue_compiled_v4_result_header').replace('<0x33524d52>','<0x34524d52>')
a=s.index('cudaError_t enqueue_compiled_v3_shared_model(');b=s.index('\n}',s.index('cudaError_t enqueue_compiled_v3_shared_result(',a))+2;shared=s[a:b].replace('enqueue_compiled_v3_','enqueue_compiled_v4_').replace('riley_shared_model::','riley_shared16_model::').replace('riley_shared_result::enqueue(', 'riley_shared16_result::enqueue<0x34524d52>(')
s+='\n#include "decode_shared16_model.cuh"\n#include "decode_shared16_result.cuh"\nnamespace riley_cuda_internal {\n'+prefill+'\n'+header+'\n'+shared+'\n}\n';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();lines=[line.replace('enqueue_compiled_v3_','enqueue_compiled_v4_') for line in s.splitlines() if 'cudaError_t enqueue_compiled_v3_' in line];s+='\nnamespace riley_cuda_internal {\n'+'\n'.join(lines)+'\n}\n';p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('bool v3_shared=false;', 'bool v3_shared=false;\n  uint32_t variable_rows=8;')
s=s.replace('r->v3_prefill_capacity?17536:r->transfer_bytes','r->v3_prefill_capacity?(128+r->variable_rows*1664+4096):r->transfer_bytes')
s=s.replace('!valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8)', '!(r->variable_rows==16?valid_v4_shape_packet(source,bytes,r->v3_prefill_physical,16):valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8))')
a=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(');b=s.index('\n}',a)+2;fn=s[a:b]
fn=fn.replace('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(', 'template<uint32_t Rows>\nstatic RileyCudaStatus record_variable_shared(')
fn=fn.replace(' clear_error(error);', ' static_assert(Rows==8||Rows==16,"wire capacity");\n constexpr uint64_t request_bytes=128+Rows*1664+4096;\n clear_error(error);',1)
for old,new in [('capacity<8','capacity<Rows'),('staging->byte_len<1574912','staging->byte_len<2*Rows*98432'),('8*9*4096*4','Rows*9*4096*4'),('d[16]->byte_len!=17536','d[16]->byte_len!=request_bytes'),('8*1152','Rows*1152'),('8*98304','Rows*98304'),('d[25]->byte_len!=787456','d[25]->byte_len!=Rows*98432'),('d[24],8,49152','d[24],Rows,49152'),('transfer=787456','transfer=Rows*98432'),('host,17536,','host,request_bytes,'),('r->v3_shared=true;','r->v3_shared=true;r->variable_rows=Rows;')]:fn=fn.replace(old,new)
for name in ['shared_model','shared_result','prefill_model','result_header']:
 fn=fn.replace('enqueue_compiled_v3_'+name+'(', '(Rows==8?enqueue_compiled_v3_'+name+':enqueue_compiled_v4_'+name+')(')
wrappers=''
for version,rows in [(3,8),(4,16)]:
 wrappers+='''
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v%d_shared(
 RileyCudaGraphResources* r,RileyCudaDeviceBuffer*const* d,RileyCudaDeviceBuffer*const* w,
 uint64_t weight_count,RileyCudaGemmPlan* head,RileyCudaGemmPlan* shared_head,RileyCudaPinnedHostBuffer* staging,
 uint32_t capacity,uint32_t physical,RileyCudaErrorInfo* error) noexcept {
 return record_variable_shared<%d>(r,d,w,weight_count,head,shared_head,staging,capacity,physical,error);
}
'''%(version,rows)
s=s[:a]+fn+wrappers+s[b:];p.write_text(s)
p=r/'kernels/src/gemm.cu';s=p.read_text().replace('(rows!=2&&rows!=4&&rows!=8)', '(rows!=2&&rows!=4&&rows!=8&&rows!=16)');p.write_text(s)
p=r/'kernels/include/riley_cuda.h';s=p.read_text();a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(');b=s.index('RILEY_CUDA_NOEXCEPT;',a)+len('RILEY_CUDA_NOEXCEPT;');s=s[:b]+'\n'+s[a:b].replace('_v3_shared','_v4_shared')+s[b:];p.write_text(s)
