from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text()
s=s.replace("pub struct BorrowedVariableSession<'a> {\n    graph:BorrowedGraphResourceReservation<'a>,", "pub struct VariableSession<G: VariableGraph> {\n    graph:G,")
s=s.replace("impl<'a> BorrowedVariableSession<'a> {", "impl<G: VariableGraph> VariableSession<G> {").replace("pub fn new(graph:BorrowedGraphResourceReservation<'a>,", "pub fn new(graph:G,")
s=s.replace('''    /// Native destruction remains the authority for GPU parent release.
    pub fn close(self)->riley_cuda::CudaResult<()> {self.graph.close()}
''','''    pub(crate) fn into_recorded_parts(self)->(G,VariableSessionIdentity) {(self.graph,self.identity)}
''')
s+='''
mod sealed {
    pub trait Sealed {}
    impl Sealed for riley_cuda::BorrowedGraphResourceReservation<'_> {}
    impl Sealed for riley_cuda::OwnedGraphResourceReservation<super::VariableModelParents> {}
}
/// Native reservation operations; sealed so a caller cannot fabricate completion.
pub trait VariableGraph: sealed::Sealed {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()>;
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()>;
}
impl VariableGraph for BorrowedGraphResourceReservation<'_> {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::read_transfer(self,output)}
}
impl VariableGraph for riley_cuda::OwnedGraphResourceReservation<VariableModelParents> {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::read_transfer(self,output)}
}
pub type BorrowedVariableSession<'a> = VariableSession<BorrowedGraphResourceReservation<'a>>;
pub type OwnedVariableSession = VariableSession<riley_cuda::OwnedGraphResourceReservation<VariableModelParents>>;
/// No fields are exposed while the native graph owns parent leases.
pub struct VariableModelParents {
    executor:super::PreparedLlamaBatchExecutor,
    stream:riley_cuda::CudaStream,
    scratch:VariableGraphBuffers,
}
impl VariableSession<BorrowedGraphResourceReservation<'_>> {
    pub fn close(self)->riley_cuda::CudaResult<()> {self.graph.close()}
}
impl OwnedVariableSession {
    pub fn close(self)->super::LlamaBatchExecutorResult<()> {
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        let parents=self.graph.close().map_err(cuda)?;
        let scratch=parents.scratch.close().map_err(cuda);
        let executor=parents.executor.close();
        let stream=parents.stream.close().map_err(cuda);
        scratch.and(executor).and(stream)
    }
}
impl super::PreparedLlamaBatchExecutor {
    /// Moves the loaded model and every graph parent into an owned session.
    /// The native reservation is destroyed before model/stream/scratch release.
    pub fn into_owned_variable_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession> {
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        let parents=VariableModelParents{executor:self,stream:context.create_stream().map_err(cuda)?,scratch:VariableGraphBuffers::prepare(context,capacity).map_err(cuda)?};
        let mut identity=None;
        let graph=riley_cuda::OwnedGraphResourceReservation::prepare(parents,|p| {
            let session=p.executor.prepare_variable_session(&mut p.stream,&mut p.scratch)?;
            let (graph,i)=session.into_recorded_parts();identity=Some(i);Ok::<_,super::LlamaBatchExecutorError>(graph)
        })?;
        let i=identity.expect("successful recording provides identity");
        OwnedVariableSession::new(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)
            .map_err(|_|super::LlamaBatchExecutorError::InvalidConfiguration{field:"V3 owned session",reason:"identity creation failed"})
    }
}
'''
p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text().replace('pub fn execute_llama_iteration_variable_graph(','pub fn execute_llama_iteration_variable_graph<G:riley_runtime::llama::variable_session::VariableGraph>(').replace("executor:&mut riley_runtime::llama::variable_session::BorrowedVariableSession<'_>,","executor:&mut riley_runtime::llama::variable_session::VariableSession<G>,");p.write_text(s)
p=r/'crates/riley-scheduler/tests/v3_owned_gpu.rs';s=(r/'crates/riley-scheduler/tests/v3_loaded_gpu.rs').read_text().replace('loaded_variable_scheduler_gpu_completion_and_settlement','owned_variable_scheduler_gpu_completion_and_settlement')
s=s.replace('let mut scratch=VariableGraphBuffers::prepare(&context,1024)?;\n let mut session=executor.prepare_variable_session(&mut stream,&mut scratch)?;', 'let mut session=executor.into_owned_variable_session(&context,1024)?;')
s=s.replace('session.close()?;scratch.close()?;executor.close()?;', 'session.close()?;').replace('V3_LOADED_SCHEDULER','V3_OWNED_SCHEDULER');p.write_text(s)
