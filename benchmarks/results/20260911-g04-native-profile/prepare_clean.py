from pathlib import Path
import subprocess
r=Path('/tmp/riley-g04-vllm-profile-source-260911');assert not r.exists()
subprocess.run(['git','clone','--no-hardlinks','/tmp/riley-g04-followup-source-260911',str(r)],check=True)
x=Path('/tmp/riley-g04-native-profile-source-260911')
for name in ['kernels/CMakeLists.txt','kernels/src/ffi_internal.hpp','kernels/src/graph_numerics.cu','kernels/src/graph_numerics_precise.cu']:(r/name).write_bytes((x/name).read_bytes())
s=(x/'kernels/src/graph_resources.cu').read_text().replace('#include <cstdio>\n','').replace('  bool diagnostic_trace = false;','  bool vllm_smol_p128 = false;').replace('  void* diagnostic_keys=nullptr;void* diagnostic_values=nullptr;\n','')
a=s.index('  if(r->diagnostic_trace){');b=s.index('  std::memmove(destination',a);s=s[:a]+s[b:]
s=s.replace('  if(profile==2&&transfer<92416)return reject(error,"trace staging too small",RILEY_CUDA_STATUS_INVALID_ARGUMENT);\n','')
a=s.index('      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2){\n        const size_t ids[4]');b=s.index('      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(3,3);',a);s=s[:a]+s[b:]
s=s.replace('r->diagnostic_trace=profile==2;r->diagnostic_keys=d[15]->device_data;r->diagnostic_values=d[16]->device_data;','r->vllm_smol_p128=profile==2;')
s=s.replace('if(u32(0)>=r->decode_vocab||live>capacity', 'if((r->vllm_smol_p128&&pos>=160)||u32(0)>=r->decode_vocab||live>capacity')
s=s.replace('(profile==2&&(k!=384||inter!=3072||layers!=30))','(profile==2&&(k!=384||inter!=3072||layers!=30||vocab!=49152))')
needle='  for(size_t i=0;i<22;++i){'
a=s.index(needle,s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_decode('))
s=s[:a]+'''  if(profile==2){
    int major=0,minor=0,runtime=0;
    if(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,r->owner->device)!=CUDA_SUCCESS||
       cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,r->owner->device)!=CUDA_SUCCESS||
       cudaRuntimeGetVersion(&runtime)!=cudaSuccess||major!=8||minor!=9||runtime!=13000)
      return reject(error,"vllm-smol-p128-v1 requires SM89 and CUDA runtime 13.0",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
'''+s[a:]
assert 'diagnostic' not in s and 'fopen' not in s and '/tmp/' not in s
(r/'kernels/src/graph_resources.cu').write_text(s)
# Public selector retains the existing bool API as a compatibility wrapper.
p=r/'crates/riley-cuda/src/graph_resources.rs';s=p.read_text();a=s.index('    pub fn record_decode(');b=s.index('\n    }\n}',a)+6;original=s[a:b]
new=original.replace('pub fn record_decode(', 'pub fn record_decode_with_profile(').replace('hf: bool,','profile: DecodeNumericalProfile,').replace('                hf,','                profile as u32,')
# The non-CUDA discard tuple may use the integer as well.
wrapper=original[:original.index('        #[cfg')]+'''        self.record_decode_with_profile(devices,workspace,weights,plans,staging,geometry,eps,
            if hf { DecodeNumericalProfile::HuggingFaceSmolLm2 } else { DecodeNumericalProfile::Canonical },publish_logits)
    }

    /// Records an explicitly selected numerical contract; no cross-profile fallback.
    /// # Errors
    /// Rejects unsupported geometry, environment, parents or capture failure.
    #[allow(clippy::too_many_arguments)]
'''
s=s[:a]+wrapper+new+s[b:]
insert='''
/// Versioned arithmetic for the retained full decode graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DecodeNumericalProfile {
    /// Existing canonical graph arithmetic.
    Canonical = 0,
    /// Existing HF-compatible SmolLM2 graph arithmetic.
    HuggingFaceSmolLm2 = 1,
    /// SM89/CUDA13 SmolLM2-135M, P128 and at most 32 generated tokens.
    /// Preserves vLLM 0.27.1 compiled/FlashAttention2 arithmetic for this bucket.
    VllmSmolP128V1 = 2,
}
'''
a=s.index('\n/// Owns');s=s[:a]+insert+s[a:];p.write_text(s)
p=r/'crates/riley-cuda/src/lib.rs';s=p.read_text().replace('pub use graph_resources::{','pub use graph_resources::{DecodeNumericalProfile, ');p.write_text(s)
p=r/'crates/riley-cuda/src/ffi.rs';s=p.read_text();a=s.index('    pub(super) fn record_decode(');s=s[:a]+s[a:].replace('hf: bool,','profile: u32,',1).replace('u32::from(hf),','profile,',1);p.write_text(s)
# Cold runtime selection, default unchanged.
p=r/'crates/riley-runtime/src/llama/executor/config.rs';s=p.read_text().replace('    metadata_transport: BatchMetadataTransport,','    metadata_transport: BatchMetadataTransport,\n    vllm_smol_p128_graph: bool,',1).replace('            metadata_transport: BatchMetadataTransport::Synchronous,','            metadata_transport: BatchMetadataTransport::Synchronous,\n            vllm_smol_p128_graph: false,',1)
a=s.index('    #[must_use]\n    pub const fn metadata(self)')
s=s[:a]+'''    /// Selects the explicit VllmSmolP128V1 owned graph. Eager fallback is forbidden.
    #[must_use]
    pub const fn with_vllm_smol_p128_graph(mut self) -> Self { self.vllm_smol_p128_graph = true; self }
    /// Whether the bounded vLLM numerical graph is required.
    #[must_use]
    pub const fn vllm_smol_p128_graph(self) -> bool { self.vllm_smol_p128_graph }

'''+s[a:];p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();s=s.replace('        let profile = f.rms_norm_profile();','        let profile = f.rms_norm_profile();\n        let vllm_profile = self.config.vllm_smol_p128_graph();',1)
s=s.replace('            .record_decode(\n','            .record_decode_with_profile(\n',1).replace('                profile == LlamaRmsNormProfile::HuggingFaceSmolLm2,','                if vllm_profile { riley_cuda::DecodeNumericalProfile::VllmSmolP128V1 } else if profile == LlamaRmsNormProfile::HuggingFaceSmolLm2 { riley_cuda::DecodeNumericalProfile::HuggingFaceSmolLm2 } else { riley_cuda::DecodeNumericalProfile::Canonical },',1)
s=s.replace('        f.plan.sequence_length() == 1\n','        (!self.config.vllm_smol_p128_graph() || (f.rms_norm_profile()==LlamaRmsNormProfile::HuggingFaceSmolLm2 && f.plan.layers().len()==30 && f.plan.dimensions().hidden_size()==576 && f.plan.dimensions().intermediate_size()==1536 && f.plan.dimensions().vocabulary_size()==49152 && f.plan.dimensions().query_heads()==9 && f.plan.dimensions().key_value_heads()==3))\n            && f.plan.sequence_length() == 1\n',1)
s=s.replace('        let profile = match f.rms_norm_profile() {','        let profile = if self.config.vllm_smol_p128_graph() { 2_u32 } else { match f.rms_norm_profile() {',1).replace('            _ => unreachable!(),\n        };','            _ => unreachable!(),\n        }};',1)
# Scope identity includes profile; exact cuBLAS version is checked before publication.
s=s.replace('        let device = GraphDeviceSignature::new(', '        if self.config.vllm_smol_p128_graph() && (major!=8 || minor!=9 || runtime!=13000 || cublas!=130101) { return Err(rejected("vllm-smol-p128-v1 environment mismatch")); }\n        let device = GraphDeviceSignature::new(',1)
s=s.replace('GraphImplementationId::new(0xF001)','GraphImplementationId::new(if profile==2 {0xF101} else {0xF001})')
s=s.replace('    replays: u64,\n','    replays: u64,\n    vllm_smol_p128_graph: bool,\n',1)
s=s.replace('        let vocabulary_size = self.vocabulary_size();\n        let maximum_position_count', '        let vllm_smol_p128_graph = self.config.vllm_smol_p128_graph();\n        let vocabulary_size = self.vocabulary_size();\n        let maximum_position_count',1)
s=s.replace('            replays: 0,\n','            replays: 0,\n            vllm_smol_p128_graph,\n',1)
a=s.index('impl OwnedLlamaDecodeExecutor {')+len('impl OwnedLlamaDecodeExecutor {')
s=s[:a]+'''
    /// Versioned request bounds, checked before any output is published.
    /// # Errors
    /// Rejects requests outside the explicitly selected numerical profile.
    pub fn validate_request_shape(&self, prompt_tokens: usize, output_tokens: usize) -> LlamaBatchExecutorResult<()> {
        if self.vllm_smol_p128_graph && (prompt_tokens!=128 || output_tokens==0 || output_tokens>32) { return Err(rejected("vllm-smol-p128-v1 requires 128 prompt tokens and 1..32 output tokens")); }
        Ok(())
    }
    /// Stable arithmetic identity for request admission and evidence.
    #[must_use]
    pub const fn numerical_profile_id(&self) -> &'static str { if self.vllm_smol_p128_graph { "vllm-smol-p128-v1" } else { "existing" } }
'''+s[a:]
s=s.replace('        let pos = packed.position_ids()[0];','        let pos = packed.position_ids()[0];\n        if self.vllm_smol_p128_graph && pos>=160 { return Err(rejected("vllm-smol-p128-v1 position exceeds 159")); }',1)
p.write_text(s)
# Reject alternate public eager/generation paths for the new graph-only profile.
p=r/'crates/riley-runtime/src/llama/batch_executor.rs';s=p.read_text();needle='    pub fn execute(';a=s.index(needle);a=s.index('{',a)+1;s=s[:a]+'\n        if self.config.vllm_smol_p128_graph() { return Err(LlamaBatchExecutorError::InvalidConfiguration { field: "graph numerics", reason: "vllm-smol-p128-v1 requires an owned graph" }); }'+s[a:];p.write_text(s)
# Request admission for server and engine-only harness.
p=r/'crates/riley-server/src/engine.rs';s=p.read_text();needle='            let total_tokens = prompt_token_ids';a=s.index(needle);s=s[:a]+'''            if let Some(graph)=&self.decode_graph { graph.validate_request_shape(prompt_token_ids.len(),request.max_new_tokens).map_err(|_| private_request_error("request is outside graph numerical profile bounds"))?; }
'''+s[a:];p.write_text(s)
p=r/'crates/riley-server/src/benchmark.rs';s=p.read_text();a=s.index('            PreparedNativeBenchmarkTrial::prepare(',s.index('        pub fn prepare_trial('));s=s[:a]+'''            if let Some(graph)=&self.decode_graph { for request in &requests { graph.validate_request_shape(request.prompt_token_ids().len(),output_tokens).map_err(|_| invalid("graph numerics","request outside numerical profile bounds"))?; } }
'''+s[a:];p.write_text(s)
