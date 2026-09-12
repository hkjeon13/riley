from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text();s+='''
/// Cold scratch for the fixed SmolLM2 V3 implementation. KV and weights remain
/// in the loaded model owner and are exclusively borrowed during the session.
pub struct VariableGraphBuffers {
    pub(crate) devices:Vec<riley_cuda::CudaDeviceBuffer>,
    pub(crate) staging:riley_cuda::CudaPinnedHostBuffer,
    pub(crate) head:riley_cuda::CudaPreparedGemm,
    pub(crate) capacity:u32,
}
impl VariableGraphBuffers {
    pub fn prepare(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self> {
        // Invalid capacity is rejected again by the native recorder; allocating
        // zero/oversized geometry is prevented with the GEMM config validator.
        let checked=if (1..=1024).contains(&capacity) {capacity}else{0};
        let config=riley_cuda::CudaGemmConfig::new(checked as usize,49152,576,0)?;
        let _=config;
        let sizes=[1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072];
        let mut devices=Vec::with_capacity(18);
        for bytes in sizes {devices.push(context.allocate_device_buffer(bytes*capacity as u64)?);}
        for bytes in [17536,1152,128,4,98304,8] {devices.push(context.allocate_device_buffer(bytes)?);}
        Ok(Self{devices,staging:context.allocate_pinned_host_buffer(196864)?,
            head:context.prepare_gemm(riley_cuda::CudaGemmConfig::new(1,49152,576,0)?)?,capacity})
    }
    pub fn close(self)->riley_cuda::CudaResult<()> {self.head.close()}
}
''';p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();s+='''
impl PreparedLlamaBatchExecutor {
    /// Captures V3 using this loader's actual immutable weights, RoPE and KV pool.
    /// Unsupported model geometry/numerics are rejected before capture.
    pub fn prepare_variable_session<'a>(
        &'a mut self, stream:&'a mut CudaStream,
        scratch:&'a mut crate::llama::variable_session::VariableGraphBuffers,
    )->LlamaBatchExecutorResult<crate::llama::variable_session::BorrowedVariableSession<'a>> {
        use sha2::{Digest,Sha256};
        let context=self.maximum_position_count()?;
        let physical=self.owner.layout.physical_block_count();
        let f=&mut self.owner.forward;
        let d=f.plan.dimensions();
        if self.owner.poisoned || f.is_poisoned() || f.plan.sequence_length()!=1
            || f.rms_norm_profile()!=LlamaRmsNormProfile::HuggingFaceSmolLm2
            || f.plan.layers().len()!=30 || d.hidden_size()!=576 || d.intermediate_size()!=1536
            || d.vocabulary_size()!=49152 || d.query_heads()!=9 || d.key_value_heads()!=3
            || self.owner.layout.head_dimension()!=64 || context==0 || context>4096 || physical==0 || physical>4096
            || f.plan.rope_theta()!=100000.0 || f.plan.final_norm_epsilon()!=1e-5 {
            return Err(rejected("V3 requires fixed SmolLM2 geometry and numerics"));
        }
        let mut weights=vec![f.plan.embedding_weight().index(),f.plan.final_norm_weight().index(),f.plan.lm_head_weight().index()];
        for l in f.plan.layers() {
            if l.query_bias().is_some() || l.key_bias().is_some() || l.value_bias().is_some() || l.output_bias().is_some()
                || l.input_norm_epsilon()!=1e-5 || l.post_attention_norm_epsilon()!=1e-5 {return Err(rejected("V3 unsupported bias or norm epsilon"));}
            weights.extend([l.input_norm_weight(),l.query_weight(),l.key_weight(),l.value_weight(),l.output_weight(),
                l.post_attention_norm_weight(),l.gate_weight(),l.up_weight(),l.down_weight()].map(|w|w.index()));
        }
        let cuda=|e|cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion),e);
        let mut hash=Sha256::new();hash.update(b"riley.v3.loaded-smol.variable.v1");
        for source in [include_bytes!("../../../../kernels/src/prefill_shape_model.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_projection.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_rope_kv.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_attention.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_pointwise.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_packet.hpp").as_slice(),
            include_bytes!("../../../../kernels/src/graph_resources.cu").as_slice(),
            include_bytes!("../../../../kernels/src/graph_numerics_precise.cu").as_slice(),
            include_bytes!("multi_descriptor/variable_wire.rs").as_slice()] {hash.update((source.len() as u64).to_le_bytes());hash.update(source);}
        hash.update(scratch.capacity.to_le_bytes());hash.update((physical as u64).to_le_bytes());hash.update((context as u64).to_le_bytes());
        for &w in &weights {hash.update((w as u64).to_le_bytes());}
        let mut chunk=vec![0;f.io_staging.byte_len().min(1024*1024) as usize];
        if chunk.is_empty(){return Err(rejected("V3 needs model staging for content identity"));}
        for buffer in f.weights.borrow_graph_weight_parents().chain([&mut self.owner.absolute_rope_cos,&mut self.owner.absolute_rope_sin]) {
            hash.update(buffer.byte_len().to_le_bytes());let mut offset=0;
            while offset<buffer.byte_len(){let n=(buffer.byte_len()-offset).min(chunk.len() as u64) as usize;
                buffer.download_to_slice(offset,&mut chunk[..n],&mut f.io_staging,stream).map_err(cuda)?;
                hash.update(&chunk[..n]);offset+=n as u64;}
        }
        let mut devices:Vec<_>=f.weights.borrow_graph_weight_parents().collect();let base=devices.len();
        let (rows,tail)=scratch.devices.split_at_mut(12);devices.extend(rows);
        devices.extend([&mut self.owner.key_cache,&mut self.owner.value_cache,&mut self.owner.absolute_rope_cos,&mut self.owner.absolute_rope_sin]);
        devices.extend(tail);
        let mut graph=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream,devices,
            pinned:vec![&mut scratch.staging],plans:vec![&mut scratch.head]}).map_err(cuda)?;
        graph.record_v3_prefill(&std::array::from_fn(|i|base+i),None,&weights,0,0,scratch.capacity,physical as u32).map_err(cuda)?;
        crate::llama::variable_session::BorrowedVariableSession::new(graph,hash.finalize().into(),physical as u32,context as u32)
            .map_err(|_|rejected("V3 session identity rejected"))
    }
}
''';s=s.replace('../../../../kernels/','../../../../../kernels/') # llama file -> src -> crate -> crates -> repo: 4? check below
p.write_text(s)
