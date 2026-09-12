from pathlib import Path
r=Path('/tmp/riley-g04-vllm-profile-source-260911')
p=r/'crates/riley-runtime/src/llama/executor/config.rs';s=p.read_text().replace('        metadata_transport: config.metadata_transport,','        metadata_transport: config.metadata_transport,\n        vllm_smol_p128_graph: config.vllm_smol_p128_graph,',1);p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();t=(r/'vllm-test-template.txt').read_text()
t=t.replace('        let mut eager_stream = context.create_stream()?;\n','').replace('        let candidate = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;','        let candidate = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config.with_vllm_smol_p128_graph())?;')
a=t.index('            // Fresh eager owner');b=t.index('            let mapping:',a);t=t[:a]+'            // New requests reuse the same retained parents and fresh scheduler mappings.\n'+t[b:]
a=t.index('                baseline.execute(');b=t.index('                token = candidate.greedy_token()',a);t=t[:a]+'''                let before=context.allocation_stats()?;
                candidate.execute(&rows)?;
                assert_eq!(context.allocation_stats()?,before);
'''+t[b:]
t=t.replace('            baseline.close()?;\n','').replace('        eager_stream.close()?;\n','').replace('[159_usize, 23, 159]','[159_usize, 23, 150, 159]').replace('candidate.replay_count(), 341','candidate.replay_count(), 491')
t=t.replace('        let mut expected_tokens = Vec::new();','''        assert_eq!(candidate.numerical_profile_id(),"vllm-smol-p128-v1");
        assert!(candidate.validate_request_shape(127,32).is_err());
        assert!(candidate.validate_request_shape(129,32).is_err());
        assert!(candidate.validate_request_shape(128,33).is_err());
        candidate.validate_request_shape(128,32)?;
        let mut expected_tokens = Vec::new();''')
t=t.replace('            if limit == 159 {','''            let reference=[28,339,5248,253,1838,3241,282,253,1443,929,3156,28,198,198,504,808,2775,339,1277,288,1643,314,338,339,5248,2045,288,1138,346,253,1443,282];
            assert_eq!(tokens.as_slice(), &reference[..tokens.len()]);
            if limit == 159 {''')
t=t.replace('G04_OWNED p128_o32=true full_logits_exact=true requests=3 cancelled_prefix=23 replays=341 captures=1 zero_allocations=true','G04_VLLM_PROFILE p128_o32=true vllm_tokens_exact=true requests=4 cancelled_prefill=23 cancelled_output=23 replays=491 captures=1 live_allocation_deltas_zero=true')
a=s.rfind('\n}');s=s[:a]+'\n'+t+s[a:];p.write_text(s)
# Remove redundant old attention work; retain only the scheduler-addressed KV write.
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
__global__ void compiled_kv_write(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* keys,__nv_bfloat16* values,const uint32_t* metadata){
 int i=threadIdx.x;if(i>=192)return;uint32_t pos=metadata[1],physical=metadata[4+pos/16];
 int destination=((physical*3+i/64)*16+pos%16)*64+i%64;keys[destination]=k[i];values[destination]=v[i];
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_kv_write(cudaStream_t s,const void* k,const void* v,void* keys,void* values,const void* metadata) noexcept {
 compiled_kv_write<<<1,256,0,s>>>((const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const uint32_t*)metadata);return cudaGetLastError();
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();a=s.rindex('}  // namespace riley_cuda_internal');s=s[:a]+'cudaError_t enqueue_compiled_kv_write(cudaStream_t,const void*,const void*,void*,void*,const void*) noexcept;\n'+s[a:];p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();needle='      if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_decode_kv_attention(';assert needle in s;s=s.replace(needle,'''      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_kv_write(r->stream->stream,d[8]->device_data,d[7]->device_data,static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,d[19]->device_data));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_decode_kv_attention(''',1)
s=s.replace('#include "ffi_internal.hpp"','#include "ffi_internal.hpp"\n#include <cublasLt.h>',1).replace('runtime!=13000)','runtime!=13000||cublasLtGetVersion()!=130101)',1)
p.write_text(s)
# The older convenience generation path prefills eagerly, so reject this profile.
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();a=s.index('    pub fn generate_greedy_with_decode_graph_policy(');a=s.index('{',a)+1;s=s[:a]+'\n        if self.config.vllm_smol_p128_graph() { return Err(rejected("vllm-smol-p128-v1 requires the persistent owned executor")); }'+s[a:];p.write_text(s)
