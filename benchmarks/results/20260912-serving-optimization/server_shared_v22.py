from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-server/src/engine.rs';s=p.read_text().replace('config.scheduler.max_active_sequences!=1','!matches!(config.scheduler.max_active_sequences,1|2|4|8)').replace('resources.scheduler.config().max_active_sequences!=1','!matches!(resources.scheduler.config().max_active_sequences,1|2|4|8)').replace('V3 requires graph policy require, one active request and CPU sampling','V3 requires graph policy require, capacity1/2/4/8 and CPU sampling')
s=s.replace('let shape_policy = if config.executor.vllm_smol_p128_batched_prefill()', 'let shape_policy = if config.executor.variable_graph() && config.scheduler.max_active_sequences>1 {\n                riley_scheduler::ExecutionShapePolicy::VariablePrefillDecodeN\n            } else if config.executor.vllm_smol_p128_batched_prefill()')
a='let graph=resources.executor.into_owned_variable_session(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32)';b='let graph=if resources.scheduler.config().max_active_sequences>1 {resources.executor.into_owned_variable_shared_session(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32)}else{resources.executor.into_owned_variable_session(&resources.context,resources.scheduler.config().max_prefill_chunk_tokens as u32)}';assert a in s;s=s.replace(a,b);p.write_text(s)
p=r/'crates/riley-server/src/main.rs';s=p.read_text().replace('options.max_active_sequences!=1 || options.prefill_chunk_tokens>1024','!matches!(options.max_active_sequences,1|2|4|8) || options.batch_token_budget<options.max_active_sequences || options.prefill_chunk_tokens>1024').replace('variable-smol-v3 currently requires one active sequence, CPU sampling, fixed-max shape and token/chunk budgets at most1024','variable-smol-v3 requires capacity1/2/4/8, CPU sampling, fixed-max shape and a token budget covering all active rows, at most1024')
a='    fn variable_profile_is_explicit_and_requires_graph_policy() {';pos=s.index(a);# add separate test before existing annotation
pos=s.rfind('    #[test]',0,pos)
s=s[:pos]+'''    #[test]
    fn variable_profile_accepts_supported_shared_capacities() {
        for capacity in ["1","2","3","4","8","16"] {
            let result=super::parse_arguments(["serve","--model","/tmp/model","--graph-numerics","variable-smol-v3","--execution-graph-policy","require","--max-active-sequences",capacity,"--batch-token-budget","128","--prefill-chunk-tokens","128"].map(std::ffi::OsString::from));
            assert_eq!(result.is_ok(),matches!(capacity,"1"|"2"|"4"|"8"));
        }
    }
'''+s[pos:];p.write_text(s)
