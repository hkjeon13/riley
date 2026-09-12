from pathlib import Path
b=Path('/tmp/riley-shared-integration-v10');files=[]
def edit(p,f):
 q=b/p;s=q.read_text();t=f(s);assert s!=t,p;q.write_text(t);files.append(p)
# A separate binder preserves the M1 contract of the old entry point.
q=(b/'kernels/src/gemm.cu').read_text();start=q.index('RileyCudaStatus bind_reserved_gemm_state(');end=q.index('\nRileyCudaStatus bind_reserved_strided',start);fn=q[start:end]
fn=fn.replace('bind_reserved_gemm_state(', 'bind_reserved_shared_row_gemm_state(').replace('RileyCudaDeviceBuffer* output, RileyCudaDeviceBuffer* workspace,','RileyCudaDeviceBuffer* output, uint32_t rows, uint64_t n, uint64_t k,').replace('plan->config.m != 1)', '(rows!=2&&rows!=4&&rows!=8) || plan->config.m != rows || plan->config.n != n || plan->config.k != k || plan->algorithm_info.workspace_bytes != 0 || plan->algorithm_info.algorithm_id != 21 || plan->algorithm_info.split_k != 1)').replace('state->workspace = workspace;', 'state->workspace = nullptr;').replace('state->workspace_lease_held = workspace != nullptr;', 'state->workspace_lease_held = false;').replace('requires selected no-split M=1 plan','requires shared-row algorithm21 with exact M/N/K and no workspace')
edit('kernels/src/gemm.cu',lambda s:s[:end]+ '\n'+fn+s[end:])
proto=fn[:fn.index(' noexcept {')]+' noexcept;\n'
edit('kernels/src/ffi_internal.hpp',lambda s:s.replace('RileyCudaStatus bind_reserved_strided_gemm_state(',proto+'\nRileyCudaStatus bind_reserved_strided_gemm_state('))
# Keep the old ABI; add named shared-row record/append variants.
p='kernels/src/graph_multisequence_record.inc'
def record(s):
 s=s.replace('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_multisequence_decode(', 'static RileyCudaStatus record_multisequence_decode_impl(').replace('uint32_t full, RileyCudaErrorInfo* error)', 'uint32_t full, bool shared_rows, RileyCudaErrorInfo* error)')
 s=s.replace('    return bind_reserved_strided_gemm_state(', '    if(shared_rows && (plan==0||plan==1||plan==4))\n      return bind_reserved_shared_row_gemm_state(plans[plan],r->stream,d[input],w[weight],d[output],bucket,n,k,&states[index],error);\n    return bind_reserved_strided_gemm_state(')
 sig='''extern "C" RileyCudaStatus NAME(RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,uint64_t dc,RileyCudaDeviceBuffer* const* w,uint64_t wc,RileyCudaGemmPlan* const* p,uint64_t pc,RileyCudaPinnedHostBuffer* h,uint32_t b,uint32_t ph,uint32_t f,RileyCudaErrorInfo* e) noexcept { return record_multisequence_decode_impl(r,d,dc,w,wc,p,pc,h,b,ph,f,SHARED,e); }\n'''
 return s+'\n'+sig.replace('NAME','riley_cuda_graph_resources_record_multisequence_decode').replace('SHARED','false')+sig.replace('NAME','riley_cuda_graph_resources_record_shared_multisequence_decode').replace('SHARED','true')
edit(p,record)
def cat(s):
 s=s.replace('extern "C" RileyCudaStatus riley_cuda_graph_resources_append_multisequence_decode(', 'static RileyCudaStatus append_multisequence_decode_impl(').replace('uint32_t full,RileyCudaErrorInfo* error)', 'uint32_t full,bool shared_rows,RileyCudaErrorInfo* error)').replace('riley_cuda_graph_resources_record_multisequence_decode(r,d,device_count,w,weight_count,plans,plan_count,staging,bucket,physical,full,error)', 'record_multisequence_decode_impl(r,d,device_count,w,weight_count,plans,plan_count,staging,bucket,physical,full,shared_rows,error)')
 sig='''extern "C" RileyCudaStatus NAME(RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,uint64_t dc,RileyCudaDeviceBuffer* const* w,uint64_t wc,RileyCudaGemmPlan* const* p,uint64_t pc,RileyCudaPinnedHostBuffer* h,uint32_t b,uint32_t ph,uint32_t f,RileyCudaErrorInfo* e) noexcept { return append_multisequence_decode_impl(r,d,dc,w,wc,p,pc,h,b,ph,f,SHARED,e); }\n'''
 return s+'\n'+sig.replace('NAME','riley_cuda_graph_resources_append_multisequence_decode').replace('SHARED','false')+sig.replace('NAME','riley_cuda_graph_resources_append_shared_multisequence_decode').replace('SHARED','true')
edit('kernels/src/graph_multisequence_catalog.inc',cat)
def header(s):
 a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_multisequence_decode(');z=s.index('RileyCudaStatus riley_cuda_graph_resources_replay_catalog(',a);return s[:z]+s[a:z].replace('_multisequence_decode','_shared_multisequence_decode')+s[z:]
edit('kernels/include/riley_cuda.h',header)
def ffi(s):
 a=s.index('    fn riley_cuda_graph_resources_record_multisequence_decode(');z=s.index('    fn riley_cuda_graph_resources_replay_catalog(',a);s=s[:z]+s[a:z].replace('_multisequence_decode','_shared_multisequence_decode')+s[z:]
 a=s.index('    pub(super) fn record_multisequence_decode(');z=s.index('\nimpl GraphResourcesHandle',a);part=s[a:z].replace('        append: bool,','        append: bool,\n        shared_rows: bool,').replace('let call = if append {','let call = if shared_rows {\n            if append { riley_cuda_graph_resources_append_shared_multisequence_decode } else { riley_cuda_graph_resources_record_shared_multisequence_decode }\n        } else if append {');return s[:a]+part+s[z:]
edit('crates/riley-cuda/src/ffi.rs',ffi)
def rust(s):
 a=s.index('    pub fn append_multisequence_decode(');z=s.index('    fn prepare_multisequence_entry(',a);method=s[a:z].replace('append_multisequence_decode','append_shared_multisequence_decode').replace('full, true,','full, true, true,')
 s=s[:z]+method+s[z:];s=s.replace('physical, full, false,','physical, full, false, false,').replace('physical, full, true,\n','physical, full, true, false,\n')
 a=s.index('    fn prepare_multisequence_entry(');part=s[a:];part=part.replace('        append: bool,','        append: bool,\n        shared_rows: bool,',1).replace('.iter()\n                .map(|i| {\n                    self.strided_plans','.iter()\n                .enumerate()\n                .map(|(role, i)| {\n                    if shared_rows && matches!(role, 0 | 1 | 4) {\n                        return self.parents.plans.get(*i).ok_or_else(bad)?.graph_resource_handle();\n                    }\n                    self.strided_plans',1).replace('                append,\n','                append,\n                shared_rows,\n',1).replace('physical, full, append,','physical, full, append, shared_rows,',1)
 return s[:a]+part
edit('crates/riley-cuda/src/graph_resources.rs',rust)
import tarfile,json
Path('/tmp/shared-v10-files.json').write_text(json.dumps(files))
with tarfile.open('/tmp/shared-v10-native.tar','w') as t:
 for p in files:t.add(b/p,arcname=p)
print(files)
