from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/executor/config.rs';s=p.read_text().replace('    shared_rows_graph: bool,','    shared_rows_graph: bool,\n    variable_graph: bool,').replace('            shared_rows_graph: false,','            shared_rows_graph: false,\n            variable_graph: false,').replace('        shared_rows_graph: config.shared_rows_graph,','        shared_rows_graph: config.shared_rows_graph,\n        variable_graph: config.variable_graph,')
pos=s.index('    pub const fn with_shared_rows_graph(');s=s[:pos]+'''    pub const fn with_variable_graph(mut self)->Self {self.variable_graph=true;self.vllm_smol_p128_graph=false;self.shared_rows_graph=false;self}
    pub const fn variable_graph(self)->bool {self.variable_graph}
'''+s[pos:];p.write_text(s)
p=r/'crates/riley-server/src/main.rs';s=p.read_text().replace('    shared_rows_graph: bool,','    shared_rows_graph: bool,\n    variable_graph: bool,').replace('                shared_rows_graph: false,','                shared_rows_graph: false,\n                variable_graph: false,')
s=s.replace('Some("existing") => (false, false),','Some("existing") => (false, false, false),').replace('Some("vllm-smol-p128-v1") => (true, false),','Some("vllm-smol-p128-v1") => (true, false, false),').replace('Some("shared-smol-p128-v1") => (true, true),','Some("shared-smol-p128-v1") => (true, true, false),\n                    Some("variable-smol-v3") => (false, false, true),')
s=s.replace('let (vllm_smol_p128_graph, shared_rows_graph) = graph_numerics.unwrap_or((false, false));','let (vllm_smol_p128_graph, shared_rows_graph, variable_graph) = graph_numerics.unwrap_or((false, false, false));')
s=s.replace('    if vllm_smol_p128_graph\n','    if (vllm_smol_p128_graph || variable_graph)\n').replace('if vllm_smol_p128_graph && c02_runtime_config.is_some()', 'if (vllm_smol_p128_graph || variable_graph) && c02_runtime_config.is_some()')
s=s.replace('        shared_rows_graph,','        shared_rows_graph,\n        variable_graph,')
needle='    if options.vllm_smol_p128_graph\n';pos=s.index(needle);s=s[:pos]+'''    if options.variable_graph && (options.max_active_sequences!=1 || options.prefill_chunk_tokens>1024
        || options.batch_token_budget>1024 || options.batch_shape_policy!=BatchShapePolicyMode::FixedMaximum
        || options.sampling_backend!=SamplingBackendMode::Cpu) {
        return Err("variable-smol-v3 currently requires one active sequence, CPU sampling, fixed-max shape and token/chunk budgets at most1024".to_owned());
    }
'''+s[pos:]
s=s.replace('    validate_reduction_profile_context(options.reduction_profile, max_sequence_tokens)?;','    if options.variable_graph && max_sequence_tokens>4096 {return Err("variable-smol-v3 context must not exceed4096".to_owned());}\n    validate_reduction_profile_context(options.reduction_profile, max_sequence_tokens)?;')
s=s.replace('        options.batch_token_budget,\n        if multi_graph', '        if options.variable_graph {1} else {options.batch_token_budget},\n        if multi_graph')
s=s.replace('    let executor = if options.shared_rows_graph {','    let executor = if options.variable_graph {\n        executor.with_variable_graph()\n    } else if options.shared_rows_graph {')
p.write_text(s)
p=r/'crates/riley-server/src/engine.rs';s=p.read_text().replace('        multi_graph: Option<riley_runtime::llama::OwnedLlamaMultiDecodeExecutor>,','        multi_graph: Option<riley_runtime::llama::OwnedLlamaMultiDecodeExecutor>,\n        variable_graph: Option<riley_runtime::llama::variable_session::OwnedVariableSession>,')
s=s.replace('            let supported = resources.executor.supports_owned_decode_graph();','''            let use_variable=resources.executor.config().variable_graph();
            if use_variable && (resources.execution_graph_policy!=ExecutionGraphPolicy::Require
                || resources.scheduler.config().max_active_sequences!=1 || resources.gpu_greedy) {
                return Err(internal("V3 requires graph policy require, one active request and CPU sampling"));
            }
            let supported = use_variable || resources.executor.supports_owned_decode_graph();''')
s=s.replace('            let (executor, decode_graph, multi_graph) = if use_multi {','''            let (executor, decode_graph, multi_graph, variable_graph) = if use_variable {
                let graph=resources.executor.into_owned_variable_session(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32)
                    .map_err(|e|internal(format!("V3 preparation failed: {e}")))?;
                (None,None,None,Some(graph))
            } else if use_multi {''')
s=s.replace('(None, None, Some(graph))','(None, None, Some(graph), None)').replace('(None, Some(graph), None)','(None, Some(graph), None, None)').replace('(Some(resources.executor), None, None)','(Some(resources.executor), None, None, None)')
s=s.replace('                    if use_multi {\n                        "vllm-smol-p128-multi-v1"','                    if use_variable { "variable-smol-v3" } else if use_multi {\n                        "vllm-smol-p128-multi-v1"')
s=s.replace('                multi_graph,\n','                multi_graph,\n                variable_graph,\n')
pos=s.index('            if let Some(graph) = self.multi_graph.take() {');s=s[:pos]+'''            if let Some(graph)=self.variable_graph.take() {
                graph.close().map_err(|e|internal(format!("V3 close failed; KV remains retained: {e}")))?;
            }
'''+s[pos:]
s=s.replace('                if let Some(graph) = self.multi_graph.as_mut() {','''                if let Some(graph)=self.variable_graph.as_mut() {
                    let authority=self.scheduler.as_ref().ok_or_else(||internal("scheduler closed"))?.authorize_execution(&plan)
                        .map_err(|e|internal(format!("V3 authority failed: {e}")))?;
                    let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,graph)
                        .map_err(|e|internal(format!("V3 iteration failed: {}",e.error())))?;
                    (downloaded,riley_scheduler::IterationTiming::default(),None)
                } else if let Some(graph) = self.multi_graph.as_mut() {''')
needle='            if let Some(graph) = self.multi_graph.as_mut() {';pos=s.index(needle);s=s[:pos]+'''            if let Some(graph)=self.variable_graph.as_mut() {
                if !updates.settlement_failures().is_empty(){return Err(internal("V3 scheduler settlement failed"));}
                graph.confirm_scheduler_commit(result.iteration_id().get()).map_err(|e|internal(format!("V3 commit failed: {e}")))?;
            }
'''+s[pos:]
s=s.replace('                && self.multi_graph.is_none()','                && self.multi_graph.is_none()\n                && self.variable_graph.is_none()')
p.write_text(s)
