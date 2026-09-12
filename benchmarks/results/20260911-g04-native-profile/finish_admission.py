from pathlib import Path
r=Path('/tmp/riley-g04-vllm-profile-source-260911')
p=r/'crates/riley-server/src/main.rs';s=p.read_text();needle='    Ok(CliCommand::Serve(ServeOptions {';s=s.replace(needle,'''    if vllm_smol_p128_graph && c02_runtime_config.is_some() { return Err("vllm-smol-p128-v1 uses separate numerical qualification, not C02 exact-change artifacts".to_owned()); }
'''+needle,1)
s+='''
#[cfg(test)]
mod graph_numerics_cli_tests {
    use super::*;
    #[test]
    fn numerical_profile_requires_explicit_required_graph() {
        for policy in ["auto","disabled"] {
            assert!(parse_arguments(["serve","--model","/tmp/model","--graph-numerics","vllm-smol-p128-v1","--execution-graph-policy",policy].map(OsString::from)).is_err());
        }
        let CliCommand::Serve(options)=parse_arguments(["serve","--model","/tmp/model","--graph-numerics","vllm-smol-p128-v1","--execution-graph-policy","require"].map(OsString::from)).expect("explicit profile") else { panic!("serve") };
        assert!(options.vllm_smol_p128_graph);
        let CliCommand::Serve(default)=parse_arguments(["serve","--model","/tmp/model"].map(OsString::from)).expect("default") else { panic!("serve") };
        assert!(!default.vllm_smol_p128_graph);
        assert!(parse_arguments(["serve","--model","/tmp/model","--graph-numerics","unknown"].map(OsString::from)).is_err());
        assert!(parse_arguments(["serve","--model","/tmp/model","--graph-numerics","existing","--graph-numerics","existing"].map(OsString::from)).is_err());
    }
}
''';p.write_text(s)
p=r/'crates/riley-server/src/engine.rs';s=p.read_text();needle='            let supported = resources.executor.supports_owned_decode_graph();';s=s.replace(needle,'''            if resources.executor.config().vllm_smol_p128_graph() && resources.execution_graph_policy!=ExecutionGraphPolicy::Require { return Err(internal("vllm-smol-p128-v1 requires an explicitly required graph")); }
'''+needle,1)
s=s.replace('"RILEY_GRAPH prepared={} policy={:?} fallback={} gpu_iteration_timing={}",\n                use_graph, resources.execution_graph_policy, !use_graph, !use_graph','"RILEY_GRAPH prepared={} policy={:?} fallback={} gpu_iteration_timing={} numerics={}",\n                use_graph, resources.execution_graph_policy, !use_graph, !use_graph, decode_graph.as_ref().map_or("existing", |graph| graph.numerical_profile_id())')
p.write_text(s)
p=r/'crates/riley-server/src/benchmark.rs';s=p.read_text();needle='            if policy == ExecutionGraphPolicy::Disabled {';s=s.replace(needle,'''            if self.executor.as_ref().is_some_and(|executor| executor.config().vllm_smol_p128_graph()) && policy!=ExecutionGraphPolicy::Require { return Err(invalid("graph numerics","vllm-smol-p128-v1 requires an explicitly required graph")); }
'''+needle,1)
# Trial preparation must not accept graph-only config before conversion.
needle='            if let Some(graph) = &self.decode_graph {';a=s.index(needle,s.index('        pub fn prepare_trial('));s=s[:a]+'''            if self.executor.as_ref().is_some_and(|executor| executor.config().vllm_smol_p128_graph()) { return Err(invalid("graph numerics","owned graph preparation is required before trials")); }
'''+s[a:];p.write_text(s)
p=r/'crates/riley-runtime/src/llama/executor/config.rs';s=p.read_text();s+='''
#[cfg(test)]
mod graph_numerical_profile_tests {
    use super::*;
    #[test]
    fn explicit_profile_survives_normalization_without_changing_default() {
        let c=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        assert!(!normalize_prepared_config(c).vllm_smol_p128_graph());
        assert!(normalize_prepared_config(c.with_vllm_smol_p128_graph()).vllm_smol_p128_graph());
    }
}
''';p.write_text(s)
