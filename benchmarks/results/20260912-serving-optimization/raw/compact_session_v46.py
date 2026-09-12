from pathlib import Path
root=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
def change(path,fn):
 p=root/path;s=p.read_text();p.write_text(fn(s))
def rep(s,a,b):
 assert a in s,a[:120]
 return s.replace(a,b,1)
def session(s):
 s=rep(s,'shared:bool,started:bool','compact:bool,shared:bool,started:bool')
 s=rep(s,'shared:false,issued:None','compact:false,shared:false,issued:None')
 at='    pub fn issue(&mut self)'
 s=rep(s,at,'    pub(crate) fn new_shared_compact(graph:G,digest:[u8;32],physical:u32,context:u32)->Result<Self>{if ROWS!=16{return Err(bad("compact requires sixteen rows"));}let mut s=Self::new_shared(graph,digest,physical,context)?;s.compact=true;Ok(s)}\n    pub fn supports_compact_greedy(&self)->bool{self.compact}\n'+at)
 s=rep(s,'        wire::encode_into(&mut self.input,&e)?;','        let compact=self.compact && e.mode==wire::ResultMode::Greedy;\n        let output_bytes=if compact{wire::Layout::<ROWS>::COMPACT_RESULT_BYTES}else{self.output.len()};\n        wire::encode_into(&mut self.input,&e)?;')
 s=rep(s,'self.graph.read_transfer(&mut self.output)','self.graph.read_transfer(&mut self.output[..output_bytes])')
 s=rep(s,'match if self.shared {wire::validate_batch_result','match if compact {wire::validate_compact_result(&self.output[..output_bytes],e)}else if self.shared {wire::validate_batch_result')
 s=rep(s,'    pub(crate) wire_rows:usize,','    pub(crate) wire_rows:usize,\n    pub(crate) compact:bool,')
 s=rep(s,'capacity,wire_rows:8}','capacity,wire_rows:8,compact:false}')
 # Preserve public factory APIs and add explicit compact factory.
 s=s.replace('context,capacity,false)}','context,capacity,false,false)}').replace('context,capacity,true)}','context,capacity,true,false)}')
 s=rep(s,'    fn into_variable_session<const ROWS:usize>','    pub fn into_owned_variable_shared16_greedy_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<16>>{self.into_variable_session::<16>(context,capacity,true,true)}\n    fn into_variable_session<const ROWS:usize>')
 s=rep(s,'capacity:u32,shared:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>','capacity:u32,shared:bool,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>')
 s=rep(s,'        let parents=VariableModelParents','        let mut parents=VariableModelParents')
 s=rep(s,'        let mut identity=None;','        parents.scratch.compact=compact;\n        let mut identity=None;')
 s=rep(s,'(if shared {OwnedVariableSession::new_shared','(if compact {OwnedVariableSession::new_shared_compact(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)}else if shared {OwnedVariableSession::new_shared')
 return s
def graph(s):
 s=rep(s,'        let context=self.maximum_position_count()?.min(4096);','        if scratch.compact && (ROWS!=16 || scratch.shared_head.is_none()){return Err(rejected("compact requires shared sixteen-row buffers"));}\n        let context=self.maximum_position_count()?.min(4096);')
 s=rep(s,'            include_bytes!("../../../../kernels/src/prefill_query_tile_attention.cuh").as_slice(),','            include_bytes!("../../../../kernels/src/prefill_query_tile_attention.cuh").as_slice(),\n            include_bytes!("../../../../kernels/src/compact_shared_result.cuh").as_slice(),')
 s=rep(s,'        for m in std::iter::once(scratch.head.algorithm_metadata())','        hash.update([u8::from(scratch.compact)]);\n        for m in std::iter::once(scratch.head.algorithm_metadata())')
 s=rep(s,'        if shared && ROWS==16 {graph.record_v4_shared','        if scratch.compact {graph.record_v4_shared_greedy(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else if shared && ROWS==16 {graph.record_v4_shared')
 s=rep(s,'(if shared {crate::llama::variable_session::BorrowedVariableSession::new_shared','(if scratch.compact {crate::llama::variable_session::BorrowedVariableSession::new_shared_compact(graph,hash.finalize().into(),physical as u32,context as u32)}else if shared {crate::llama::variable_session::BorrowedVariableSession::new_shared')
 return s
def execution(s):
 at='    let id=authority.plan().iteration_id();'
 start=s.index('pub fn execute_llama_iteration_variable_graph<')
 pos=s.index(at,start)
 s=s[:pos]+'''    execute_llama_iteration_variable_graph_mode(authority,executor,false)
}

/// Selects the completion format before dispatch; no model replay is used to fall back.
#[cfg(feature = "cuda")]
pub fn execute_llama_iteration_variable_graph_mode<G:riley_runtime::llama::variable_session::VariableGraph,const ROWS:usize>(
    authority:&crate::AuthorizedExecution<'_>,
    executor:&mut riley_runtime::llama::variable_session::VariableSession<G,ROWS>,
    compact_greedy:bool,
) -> Result<DownloadedLlamaIteration,IterationExecutionFailure> {
'''+s[pos:]
 start=s.index('pub fn execute_llama_iteration_variable_graph_mode<');a=s[:start];s=s[start:]
 s=rep(s,'    let mut logits=zeroed_vec(prepared.output_count*49152*2,','    if compact_greedy && !executor.supports_compact_greedy(){return Err(fail(crate::descriptor::Error{field:"V4 compact",reason:"session lacks compact graph"},Some(ExecutionAbort::NotDispatched)));}\n    let mut tokens=Vec::new();\n    if compact_greedy {tokens.try_reserve_exact(prepared.output_count).map_err(|_|fail(crate::descriptor::Error{field:"V4 compact",reason:"token allocation failed"},Some(ExecutionAbort::NotDispatched)))?;tokens.resize(prepared.output_count,0);}\n    let mut logits=zeroed_vec(if compact_greedy{0}else{prepared.output_count*49152*2},')
 s=rep(s,'&cookies,crate::descriptor::ResultMode::FullLogits)','&cookies,if compact_greedy{crate::descriptor::ResultMode::Greedy}else{crate::descriptor::ResultMode::FullLogits})')
 s=rep(s,'argmax[row.output_slot as usize]=token;let start=row.output_slot as usize*98304;logits[start..start+98304].copy_from_slice(row.logits);','if compact_greedy {tokens[row.output_slot as usize]=token;}else{argmax[row.output_slot as usize]=token;let start=row.output_slot as usize*98304;logits[start..start+98304].copy_from_slice(row.logits);}')
 s=rep(s,'output:DownloadedLlamaOutput::ValidatedLogits{logits,argmax},','output:if compact_greedy{DownloadedLlamaOutput::GreedyTokens(tokens)}else{DownloadedLlamaOutput::ValidatedLogits{logits,argmax}},')
 return a+s
def engine(s):
 s=rep(s,'|| config.executor.metadata().max_input_tokens()!=1 || config.gpu_greedy)','|| config.executor.metadata().max_input_tokens()!=1 || (config.gpu_greedy && config.executor.variable_graph_rows()!=16))')
 s=rep(s,"fn execute(&mut self,authority:&riley_scheduler::AuthorizedExecution<'_>)","fn execute(&mut self,authority:&riley_scheduler::AuthorizedExecution<'_>,gpu_greedy:bool)")
 s=rep(s,'Self::Sixteen(g)=>riley_scheduler::execution::execute_llama_iteration_variable_graph(authority,g)','Self::Sixteen(g)=>riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(authority,g,gpu_greedy)')
 s=rep(s,'|| !matches!(resources.scheduler.config().max_active_sequences,1|2|4|8|16|32) || resources.gpu_greedy)','|| !matches!(resources.scheduler.config().max_active_sequences,1|2|4|8|16|32) || (resources.gpu_greedy && resources.executor.config().variable_graph_rows()!=16))')
 s=rep(s,'V3 requires graph policy require, capacity1/2/4/8/16/32 and CPU sampling','V3 requires graph policy require, capacity1/2/4/8/16/32; GPU greedy requires sixteen-row graphs')
 s=rep(s,'let graph=if resources.executor.config().variable_graph_rows()==16 {','let graph=if resources.gpu_greedy {resources.executor.into_owned_variable_shared16_greedy_session(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32).map(VariableServingSession::Sixteen)}else if resources.executor.config().variable_graph_rows()==16 {')
 s=rep(s,'let downloaded=graph.execute(&authority)','let downloaded=graph.execute(&authority,gpu_greedy)')
 return s
change('crates/riley-runtime/src/llama/variable_session.rs',session)
change('crates/riley-runtime/src/llama/graph_decode_full.rs',graph)
change('crates/riley-scheduler/src/execution.rs',execution)
change('crates/riley-server/src/engine.rs',engine)
