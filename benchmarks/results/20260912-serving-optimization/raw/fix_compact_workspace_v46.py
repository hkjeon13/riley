from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
p=src/'crates/riley-scheduler/src/execution.rs';s=p.read_text();start=s.index('pub fn execute_llama_iteration_variable_graph_mode<');pos=s.index('    let id=authority.plan().iteration_id();',start)
s=s[:pos]+'''    execute_variable_graph_impl(authority,executor,compact_greedy,None)
}
/// Uses the caller-owned token allocation and leaves the standard empty placeholder.
#[cfg(feature = "cuda")]
pub fn execute_llama_iteration_variable_graph_greedy_workspace<G:riley_runtime::llama::variable_session::VariableGraph,const ROWS:usize>(
    authority:&crate::AuthorizedExecution<'_>,
    executor:&mut riley_runtime::llama::variable_session::VariableSession<G,ROWS>,
    workspace:&mut Vec<u32>,
) -> Result<DownloadedLlamaIteration,IterationExecutionFailure> {
    execute_variable_graph_impl(authority,executor,true,Some(workspace))
}
#[cfg(feature = "cuda")]
fn execute_variable_graph_impl<G:riley_runtime::llama::variable_session::VariableGraph,const ROWS:usize>(
    authority:&crate::AuthorizedExecution<'_>,
    executor:&mut riley_runtime::llama::variable_session::VariableSession<G,ROWS>,
    compact_greedy:bool,workspace:Option<&mut Vec<u32>>,
) -> Result<DownloadedLlamaIteration,IterationExecutionFailure> {
'''+s[pos:]
a='''    let mut tokens=Vec::new();
    if compact_greedy {tokens.try_reserve_exact(prepared.output_count).map_err(|_|fail(crate::descriptor::Error{field:"V4 compact",reason:"token allocation failed"},Some(ExecutionAbort::NotDispatched)))?;tokens.resize(prepared.output_count,0);}'''
b='''    let mut local_tokens=Vec::new();
    let tokens=workspace.unwrap_or(&mut local_tokens);
    if compact_greedy {prepare_greedy_token_workspace(tokens,prepared.output_count).map_err(|e|IterationExecutionFailure::new(id,Some(ExecutionAbort::NotDispatched),e))?;}'''
assert a in s;s=s.replace(a,b,1);s=s.replace('DownloadedLlamaOutput::GreedyTokens(tokens)}else{DownloadedLlamaOutput::ValidatedLogits','DownloadedLlamaOutput::GreedyTokens(std::mem::take(tokens))}else{DownloadedLlamaOutput::ValidatedLogits',1);p.write_text(s)
p=src/'crates/riley-server/src/engine.rs';s=p.read_text().replace("authority:&riley_scheduler::AuthorizedExecution<'_>,gpu_greedy:bool)","authority:&riley_scheduler::AuthorizedExecution<'_>,gpu_greedy:bool,workspace:&mut Vec<u32>)",1);s=s.replace('Self::Sixteen(g)=>riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(authority,g,gpu_greedy)','Self::Sixteen(g)=>if gpu_greedy{riley_scheduler::execution::execute_llama_iteration_variable_graph_greedy_workspace(authority,g,workspace)}else{riley_scheduler::execution::execute_llama_iteration_variable_graph(authority,g)}',1);s=s.replace('graph.execute(&authority,gpu_greedy)','graph.execute(&authority,gpu_greedy,&mut self.greedy_token_workspace)',1);p.write_text(s)
# Save retry1, use a fresh result directory for retry2.
p=r/'run_v4_http_v46_c32.py';p.write_text(p.read_text().replace('v4-http-v46-c32-retry1','v4-http-v46-c32-retry2'))
p=r/'fix_compact_cli_v46.py';s=p.read_text();s=s[s.index('env=os.environ.copy();'):];s='from pathlib import Path\nimport os,subprocess\nr=Path("/tmp/riley-opt-260912")\n'+s;s=s.replace("('final-build',","('workspace-build',").replace("('http-retry1',","('http-retry2',");(r/'retry_compact_workspace_v46.py').write_text(s)
