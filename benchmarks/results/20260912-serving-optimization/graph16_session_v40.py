from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text().replace('namespace riley_cuda_internal { cudaError_t enqueue_compiled_v4_result_header(cudaStream_t,const void*,void*,const void*,const void*) noexcept; }','cudaError_t enqueue_compiled_v4_result_header(cudaStream_t,const void*,void*,const void*,const void*) noexcept;');p.write_text(s)
p=r/'crates/riley-cuda/build.rs';s=p.read_text();line='        kernels_dir.join("src/decode_shared_result.cuh"),';s=s.replace(line,line+'\n'+ '\n'.join('        kernels_dir.join("src/'+n+'"),' for n in ['decode_shared16.cuh','decode_shared16_attention.cuh','decode_shared16_model.cuh','decode_shared16_result.cuh']));p.write_text(s)
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text()
s=s.replace('pub struct VariableSession<G: VariableGraph>','pub struct VariableSession<G: VariableGraph,const ROWS:usize=8>').replace('impl<G: VariableGraph> VariableSession<G>', 'impl<G: VariableGraph,const ROWS:usize> VariableSession<G,ROWS>')
s=s.replace('Option<wire::Expectation>', 'Option<wire::Expectation<ROWS>>').replace('e:wire::Expectation)', 'e:wire::Expectation<ROWS>)')
s=s.replace('if catalog_digest==', 'if !matches!(ROWS,8|16) || catalog_digest==',1)
s=s.replace('wire::REQUEST_BYTES','wire::Layout::<ROWS>::REQUEST_BYTES').replace('wire::BATCH_RESULT_BYTES','wire::Layout::<ROWS>::BATCH_RESULT_BYTES')
s=s.replace('if self.shared{8}else{1}','if self.shared{ROWS}else{1}').replace('e.max_active_rows!=8','e.max_active_rows!=ROWS as u32')
s=s.replace('pub(crate) capacity:u32,','pub(crate) capacity:u32,\n    pub(crate) wire_rows:usize,')
s=s.replace('?)?,capacity})','?)?,capacity,wire_rows:8})')
a=s.index('    pub fn prepare_shared(');b=s.index('    pub fn close(self)',a)
s=s[:a]+'''    pub fn prepare_shared(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{Self::prepare_shared_rows::<8>(context,capacity)}
    pub fn prepare_shared16(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{Self::prepare_shared_rows::<16>(context,capacity)}
    fn prepare_shared_rows<const ROWS:usize>(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{
        let mut s=Self::prepare_base(context,if capacity==0{0}else{capacity.max(ROWS as u32)},true)?;
        s.wire_rows=ROWS;
        s.devices[7]=context.allocate_device_buffer((s.capacity as u64*384).max(ROWS as u64*9*4096*4))?;
        s.devices[12]=context.allocate_device_buffer(wire::Layout::<ROWS>::REQUEST_BYTES as u64)?;
        s.staging=context.allocate_pinned_host_buffer(wire::Layout::<ROWS>::STAGING_BYTES as u64)?;
        for bytes in [ROWS as u64*1152,ROWS as u64*98304,wire::Layout::<ROWS>::BATCH_RESULT_BYTES as u64]{s.shared_devices.push(context.allocate_device_buffer(bytes)?);}
        s.shared_head=Some(context.prepare_gemm(riley_cuda::CudaGemmConfig::new(ROWS as u64,49152,576,0)?)?);Ok(s)
    }
'''+s[b:]
s=s.replace("pub type BorrowedVariableSession<'a> = VariableSession<BorrowedGraphResourceReservation<'a>>;", "pub type BorrowedVariableSession<'a,const ROWS:usize=8> = VariableSession<BorrowedGraphResourceReservation<'a>,ROWS>;")
s=s.replace('pub type OwnedVariableSession = VariableSession<riley_cuda::OwnedGraphResourceReservation<VariableModelParents>>;', 'pub type OwnedVariableSession<const ROWS:usize=8> = VariableSession<riley_cuda::OwnedGraphResourceReservation<VariableModelParents>,ROWS>;')
s=s.replace("impl VariableSession<BorrowedGraphResourceReservation<'_>>", "impl<const ROWS:usize> VariableSession<BorrowedGraphResourceReservation<'_>,ROWS>").replace('impl OwnedVariableSession {', 'impl<const ROWS:usize> OwnedVariableSession<ROWS> {')
s=s.replace('self.into_variable_session(context,capacity,false)', 'self.into_variable_session::<8>(context,capacity,false)').replace('self.into_variable_session(context,capacity,true)', 'self.into_variable_session::<8>(context,capacity,true)')
s=s.replace('    fn into_variable_session(', '''    pub fn into_owned_variable_shared16_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<16>>{self.into_variable_session::<16>(context,capacity,true)}
    fn into_variable_session<const ROWS:usize>(''')
s=s.replace('shared:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession>', 'shared:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>')
s=s.replace('VariableGraphBuffers::prepare_shared(context,capacity)', 'VariableGraphBuffers::prepare_shared_rows::<ROWS>(context,capacity)')
s=s.replace('p.executor.prepare_variable_session(&mut p.stream,&mut p.scratch)', 'p.executor.prepare_variable_session_rows::<ROWS>(&mut p.stream,&mut p.scratch)')
p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();s=s.replace("    pub fn prepare_variable_session<'a>(", """    pub fn prepare_variable_session<'a>(&'a mut self,stream:&'a mut CudaStream,scratch:&'a mut crate::llama::variable_session::VariableGraphBuffers)->LlamaBatchExecutorResult<crate::llama::variable_session::BorrowedVariableSession<'a>>{self.prepare_variable_session_rows::<8>(stream,scratch)}
    pub fn prepare_variable_session_rows<'a,const ROWS:usize>(""")
s=s.replace("BorrowedVariableSession<'a>> {", "BorrowedVariableSession<'a,ROWS>> {")
needle='        let context=self.maximum_position_count()?.min(4096);'
s=s.replace(needle,'''        if !matches!(ROWS,8|16) || scratch.wire_rows!=ROWS || (ROWS==16 && scratch.shared_head.is_none()) {return Err(rejected("wire capacity differs from prepared buffers"));}
'''+needle)
needle='include_bytes!("../../../../kernels/src/decode_shared_result.cuh").as_slice(),';s=s.replace(needle,needle+'\n'+'\n'.join('include_bytes!("../../../../kernels/src/'+n+'").as_slice(),' for n in ['decode_shared16.cuh','decode_shared16_attention.cuh','decode_shared16_model.cuh','decode_shared16_result.cuh']))
s=s.replace('hash.update([u8::from(scratch.shared_head.is_some())]);','hash.update([u8::from(scratch.shared_head.is_some())]);hash.update((ROWS as u32).to_le_bytes());')
old='if shared {graph.record_v3_shared(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}'
new='if shared && ROWS==16 {graph.record_v4_shared(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else '+old
assert old in s;s=s.replace(old,new);p.write_text(s)
