from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
def edit(n,f):
 p=src/n;p.write_text(f(p.read_text()))
edit('crates/riley-cuda/build.rs',lambda s:s.replace('        kernels_dir.join("src/decode_shared16.cuh"),','        kernels_dir.join("src/decode_shared16.cuh"),\n'+''.join(f'        kernels_dir.join("src/{n}"),\n' for n in ['decode_shared32.cuh','decode_shared32_attention.cuh','decode_shared32_model.cuh','decode_shared32_result.cuh'])))
def wire(s):
 s=s.replace('matches!(ROWS,8|16)','matches!(ROWS,8|16|32)').replace('1|2|4|8|16)','1|2|4|8|16|32)')
 s=s.replace('if ROWS==8 {3} else {4}','if ROWS==8 {3} else if ROWS==16 {4} else {5}').replace('if ROWS==8 {0x33444d52} else {0x34444d52}','if ROWS==8 {0x33444d52} else if ROWS==16 {0x34444d52} else {0x35444d52}').replace('if ROWS==8 {0x33524d52} else {0x34524d52}','if ROWS==8 {0x33524d52} else if ROWS==16 {0x34524d52} else {0x35524d52}')
 s=s.replace('    #[test] fn compact_sixteen_identity_status_and_publication(){compact_contract::<16>();}','    #[test] fn compact_sixteen_identity_status_and_publication(){compact_contract::<16>();}\n    #[test] fn compact_thirtytwo_identity_status_and_publication(){compact_contract::<32>();}')
 return s
edit('crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs',wire)
edit('crates/riley-runtime/src/llama/variable_session.rs',lambda s:s.replace('matches!(ROWS,8|16)','matches!(ROWS,8|16|32)').replace('if ROWS!=16','if !matches!(ROWS,16|32)').replace('!shared || ROWS!=16','!shared || !matches!(ROWS,16|32)').replace('compact requires sixteen rows','compact requires sixteen or thirty-two rows'))
def graph(s):
 s=s.replace('matches!(ROWS,8|16)','matches!(ROWS,8|16|32)').replace('(ROWS==16 && scratch.shared_head.is_none())','(ROWS>=16 && scratch.shared_head.is_none())').replace('(ROWS!=16 || scratch.shared_head.is_none())','(!matches!(ROWS,16|32) || scratch.shared_head.is_none())')
 s=s.replace('include_bytes!("../../../../kernels/src/decode_shared16.cuh").as_slice(),','include_bytes!("../../../../kernels/src/decode_shared16.cuh").as_slice(),\n'+''.join(f'include_bytes!("../../../../kernels/src/{n}").as_slice(),\n' for n in ['decode_shared32.cuh','decode_shared32_attention.cuh','decode_shared32_model.cuh','decode_shared32_result.cuh']))
 s=s.replace('if scratch.compact {graph.record_v4_shared_greedy','if scratch.compact && ROWS==32 {graph.record_v5_shared_greedy(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else if shared && ROWS==32 {graph.record_v5_shared(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else if scratch.compact {graph.record_v4_shared_greedy',1)
 return s
edit('crates/riley-runtime/src/llama/graph_decode_full.rs',graph)
edit('crates/riley-runtime/src/llama/executor/config.rs',lambda s:s.replace('    pub const fn variable_graph_rows','    pub const fn with_variable_graph32(self)->Self {let mut s=self.with_variable_graph();s.variable_graph_rows=32;s}\n    pub const fn variable_graph_rows',1))
edit('crates/riley-scheduler/src/config.rs',lambda s:s.replace('    VariablePrefillDecode16,','    VariablePrefillDecode16,\n    /// Variable prefill and up to thirty-two decode rows on a prepared V5 session.\n    VariablePrefillDecode32,'))
def scheduler(s):
 s=s.replace('ExecutionShapePolicy::VariablePrefillDecode16)', 'ExecutionShapePolicy::VariablePrefillDecode16 | ExecutionShapePolicy::VariablePrefillDecode32)')
 s=s.replace('ExecutionShapePolicy::VariablePrefillDecode16=>16,','ExecutionShapePolicy::VariablePrefillDecode16=>16,ExecutionShapePolicy::VariablePrefillDecode32=>32,')
 return s
edit('crates/riley-scheduler/src/scheduler.rs',scheduler)
# Bounded array avoids per-iteration allocation while supporting every retained width.
edit('crates/riley-scheduler/src/execution.rs',lambda s:s.replace('argmax:[u32;16]','argmax:[u32;32]').replace('ROWS>16','ROWS>32').replace('let mut argmax=[0u32;16]','let mut argmax=[0u32;32]').replace('argmax:[3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]','argmax:[3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'))
def engine(s):
 s=s.replace('config.gpu_greedy && config.executor.variable_graph_rows()!=16','config.gpu_greedy && !matches!(config.executor.variable_graph_rows(),16|32)').replace('resources.gpu_greedy && resources.executor.config().variable_graph_rows()!=16','resources.gpu_greedy && !matches!(resources.executor.config().variable_graph_rows(),16|32)')
 s=s.replace('let shape_policy = if config.executor.variable_graph() && config.executor.variable_graph_rows()==16 {','let shape_policy = if config.executor.variable_graph() && config.executor.variable_graph_rows()==32 {riley_scheduler::ExecutionShapePolicy::VariablePrefillDecode32} else if config.executor.variable_graph() && config.executor.variable_graph_rows()==16 {')
 s=s.replace('        Sixteen(riley_runtime::llama::variable_session::OwnedVariableSession<16>),','        Sixteen(riley_runtime::llama::variable_session::OwnedVariableSession<16>),\n        ThirtyTwo(riley_runtime::llama::variable_session::OwnedVariableSession<32>),')
 # Clone independent enum arms, whose generic G differs between widths.
 a='Self::Sixteen(g)=>if gpu_greedy{riley_scheduler::execution::execute_llama_iteration_variable_graph_greedy_workspace(authority,g,workspace)}else{riley_scheduler::execution::execute_llama_iteration_variable_graph(authority,g)}'
 assert a in s;s=s.replace(a,a+','+a.replace('Self::Sixteen','Self::ThirtyTwo'))
 s=s.replace('Self::Sixteen(g)=>g.confirm_scheduler_commit(id)','Self::Sixteen(g)=>g.confirm_scheduler_commit(id),Self::ThirtyTwo(g)=>g.confirm_scheduler_commit(id)').replace('Self::Sixteen(g)=>g.close()','Self::Sixteen(g)=>g.close(),Self::ThirtyTwo(g)=>g.close()')
 s=s.replace('let graph=if resources.gpu_greedy {','let graph=if resources.executor.config().variable_graph_rows()==32 {if resources.gpu_greedy{resources.executor.into_owned_variable_shared_greedy_session_rows::<32>(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32).map(VariableServingSession::ThirtyTwo)}else{resources.executor.into_owned_variable_shared_session_rows::<32>(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32).map(VariableServingSession::ThirtyTwo)}}else if resources.gpu_greedy {',1)
 return s
edit('crates/riley-server/src/engine.rs',engine)
def main(s):
 s=s.replace('    variable_graph16: bool,','    variable_graph16: bool,\n    variable_graph32: bool,')
 for a in ['(false, false, false, false)','(true, false, false, false)','(true, true, false, false)','(false, false, true, false)','(false, false, true, true)']:s=s.replace(a,a[:-1]+', false)')
 s=s.replace('Some("variable-smol-v4") => (false, false, true, true, false),','Some("variable-smol-v4") => (false, false, true, true, false),\n                    Some("variable-smol-v5") => (false, false, true, false, true),')
 s=s.replace('shared_rows_graph, variable_graph, variable_graph16) =','shared_rows_graph, variable_graph, variable_graph16, variable_graph32) =')
 s=s.replace('        variable_graph16,','        variable_graph16,\n        variable_graph32,').replace('                variable_graph16: false,','                variable_graph16: false,\n                variable_graph32: false,')
 s=s.replace('&& !options.variable_graph16)','&& !options.variable_graph16 && !options.variable_graph32)')
 s=s.replace('let executor = if options.variable_graph16 {','let executor = if options.variable_graph32 {executor.with_variable_graph32()} else if options.variable_graph16 {')
 s=s.replace('GPU greedy requires variable-smol-v4','GPU greedy requires variable-smol-v4 or variable-smol-v5')
 s=s.replace('variable-smol-v3, variable-smol-v4\\n','variable-smol-v3, variable-smol-v4, variable-smol-v5\\n')
 return s
edit('crates/riley-server/src/main.rs',main)
print('runtime32 integrated')
