from pathlib import Path
import re
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text()
marker='    pub fn supports_compact_greedy'
s=s.replace(marker,'''    pub(crate) fn new_shared_packed(graph:G,digest:[u8;32],physical:u32,context:u32,compact:bool)->Result<Self>{
        if ROWS!=32{return Err(bad("packed requires32 rows"));}
        let mut s=if compact{Self::new_shared_compact(graph,digest,physical,context)?}else{Self::new_shared(graph,digest,physical,context)?};s.identity.packed_prefill=true;Ok(s)
    }
'''+marker)
s=s.replace('    pub(crate) compact:bool,','    pub(crate) compact:bool,\n    pub(crate) packed_prefill:bool,')
s=s.replace('capacity,wire_rows:8,compact:false}', 'capacity,wire_rows:8,compact:false,packed_prefill:false}')
s=re.sub(r'(self.into_variable_session::<(?:8|16|ROWS)>\(context,capacity,(?:true|false),(?:true|false))\)',r'\1,false)',s)
s=s.replace('    fn into_variable_session<const ROWS:usize>', '''    pub fn into_owned_variable_packed_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>>{self.into_variable_session::<32>(context,capacity,true,compact,true)}
    fn into_variable_session<const ROWS:usize>''')
s=s.replace('capacity:u32,shared:bool,compact:bool)->','capacity:u32,shared:bool,compact:bool,packed:bool)->')
s=s.replace('if !matches!(ROWS,8|16|32) || (compact', 'if (packed && (ROWS!=32||!shared)) || !matches!(ROWS,8|16|32) || (compact')
s=s.replace('parents.scratch.compact=compact;', 'parents.scratch.compact=compact;parents.scratch.packed_prefill=packed;')
s=s.replace('(if compact {OwnedVariableSession::', '(if packed {OwnedVariableSession::new_shared_packed(graph,i.catalog_digest,i.physical_block_count,i.context_tokens,compact)}else if compact {OwnedVariableSession::')
p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text()
marker='            include_bytes!("../../../../kernels/src/prefill_shape_model.cuh").as_slice(),';s=s.replace(marker,marker+'\n'+''.join('            include_bytes!("../../../../kernels/src/'+n+'").as_slice(),\n' for n in ['packed_prefill_attention_v48.cuh','packed_prefill_rope_v48.cuh','packed_prefill_model_v48.cuh']))
s=s.replace('        if scratch.compact && ROWS==32 {', '        if scratch.packed_prefill && scratch.compact {graph.record_v6_shared_greedy(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else if scratch.packed_prefill {graph.record_v6_shared(&std::array::from_fn(|i|base+i),None,&weights,0,1,0,scratch.capacity,physical as u32)}else if scratch.compact && ROWS==32 {')
s=s.replace('(if scratch.compact {crate::llama::variable_session::BorrowedVariableSession::', '(if scratch.packed_prefill {crate::llama::variable_session::BorrowedVariableSession::new_shared_packed(graph,hash.finalize().into(),physical as u32,context as u32,scratch.compact)}else if scratch.compact {crate::llama::variable_session::BorrowedVariableSession::')
p.write_text(s)
p=r/'crates/riley-runtime/src/llama/executor/config.rs';s=p.read_text().replace('    variable_graph_rows: usize,','    variable_graph_rows: usize,\n    packed_prefill: bool,').replace('            variable_graph_rows: 8,','            variable_graph_rows: 8,\n            packed_prefill: false,').replace('self.variable_graph=true;self.variable_graph_rows=8;', 'self.variable_graph=true;self.variable_graph_rows=8;self.packed_prefill=false;').replace('    pub const fn variable_graph_rows(self)', '    pub const fn with_packed_prefill(self)->Self {let mut s=self.with_variable_graph32();s.packed_prefill=true;s}\n    pub const fn packed_prefill(self)->bool {self.packed_prefill}\n    pub const fn variable_graph_rows(self)').replace('        variable_graph_rows: config.variable_graph_rows,', '        variable_graph_rows: config.variable_graph_rows,\n        packed_prefill: config.packed_prefill,');p.write_text(s)
p=r/'crates/riley-server/src/main.rs';s=p.read_text().replace('    variable_graph32: bool,','    variable_graph32: bool,\n    packed_prefill: bool,').replace('                variable_graph32: false,','                variable_graph32: false,\n                packed_prefill: false,')
# Expand only graph numerical tuple, not unrelated tuples.
a=s.index('let enabled = match value.to_str()');b=s.index('set_once(&mut graph_numerics',a);part=s[a:b];part=re.sub(r'\((true|false), (true|false), (true|false), (true|false), (true|false)\)',r'(\1, \2, \3, \4, \5, false)',part);part=part.replace('                    _ => {','                    Some("variable-smol-v6") => (false, false, true, false, true, true),\n                    _ => {');s=s[:a]+part+s[b:]
s=s.replace('variable_graph16, variable_graph32) = graph_numerics.unwrap_or((false, false, false, false, false))','variable_graph16, variable_graph32, packed_prefill) = graph_numerics.unwrap_or((false, false, false, false, false, false))')
s=s.replace('        variable_graph32,','        variable_graph32,\n        packed_prefill,')
s=s.replace('let executor = if options.variable_graph32', 'let executor = if options.packed_prefill {executor.with_packed_prefill()} else if options.variable_graph32')
s=s.replace('variable-smol-v4 or variable-smol-v5', 'variable-smol-v4, variable-smol-v5 or variable-smol-v6').replace('variable-smol-v4, variable-smol-v5\n','variable-smol-v4, variable-smol-v5, variable-smol-v6\n')
p.write_text(s)
p=r/'crates/riley-server/src/engine.rs';s=p.read_text().replace('let shape_policy = if config.executor.variable_graph()', 'let shape_policy = if config.executor.packed_prefill() {riley_scheduler::ExecutionShapePolicy::PackedPrefillDecode32} else if config.executor.variable_graph()',1)
s=s.replace('let graph=if resources.executor.config().variable_graph_rows()==32', 'let graph=if resources.executor.config().packed_prefill() {resources.executor.into_owned_variable_packed_session(&resources.context,resources.scheduler.config().iteration_token_budget as u32,resources.gpu_greedy).map(VariableServingSession::ThirtyTwo)}else if resources.executor.config().variable_graph_rows()==32',1)
s=s.replace('Some(VariableServingSession::ThirtyTwo(_))=>"variable-smol-v5"', 'Some(VariableServingSession::ThirtyTwo(g))=>if g.supports_packed_prefill(){"variable-smol-v6"}else{"variable-smol-v5"}')
p.write_text(s)
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('    pub fn supports_compact_greedy', '    pub fn supports_packed_prefill(&self)->bool{self.identity.packed_prefill}\n    pub fn supports_compact_greedy');p.write_text(s)
print('Runtime and explicit variable-smol-v6 serving profile connected')
