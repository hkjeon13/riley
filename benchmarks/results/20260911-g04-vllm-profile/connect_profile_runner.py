from pathlib import Path
import json
r=Path('/tmp/riley-g04-vllm-profile-source-260911');p=r/'crates/riley-server/src/bin/riley-profile.rs';s=p.read_text()
s=s.replace('    ExecutionGraph(bool),','    ExecutionGraph(bool),\n    VllmSmolP128,',1).replace('struct Options {','struct Options {\n    prepare_only: bool,',1)
s=s.replace('const KNOWN_FLAGS: &[&str] = &[','const KNOWN_FLAGS: &[&str] = &[\n    "--prepare-only",',1)
s=s.replace('    let options = Options {','    let options = Options {\n        prepare_only: match values.remove("--prepare-only").as_deref() { None|Some("false")=>false,Some("true")=>true,_=>return Err("--prepare-only requires true or false".to_owned()) },',1)
s=s.replace('            | "require"\n','            | "require"\n            | "vllm-smol-p128-v1"\n',1)
s=s.replace('            ("execution_graph_policy", "disabled")', '            ("graph_numerics", "vllm-smol-p128-v1") => Ok(RuntimeSelection::VllmSmolP128),\n            ("execution_graph_policy", "disabled")',1)
s=s.replace('            RuntimeSelection::ExecutionGraph(_) => "g04-full-decode-p128-o32",','            RuntimeSelection::ExecutionGraph(_) => "g04-full-decode-p128-o32",\n            RuntimeSelection::VllmSmolP128 => "g04-vllm-smol-p128-v1",',1)
s=s.replace('                | RuntimeSelection::ExecutionGraph(true),','                | RuntimeSelection::ExecutionGraph(true)\n                | RuntimeSelection::VllmSmolP128,',1)
s=s.replace('        if self.source.semantic_class != "E0" {\n            return Err("--semantic-class must be E0".to_owned());\n        }','''        let expected_class=if matches!(self.runtime_selection()?,RuntimeSelection::VllmSmolP128) { "VLLM_REFERENCE" } else { "E0" };
        if self.source.semantic_class != expected_class { return Err(format!("--semantic-class must be {expected_class} for this profile")); }
        if expected_class=="VLLM_REFERENCE" && (self.workload.concurrency!=1 || self.workload.prompt_tokens!=128 || self.workload.output_tokens!=32) { return Err("vllm-smol-p128-v1 evidence requires c1/p128/o32".to_owned()); }''',1)
s=s.replace('        RuntimeSelection::ExecutionGraph(_)\n','        RuntimeSelection::ExecutionGraph(_) | RuntimeSelection::VllmSmolP128\n')
s=s.replace('        RuntimeSelection::ExecutionGraph(_) => executor','        RuntimeSelection::VllmSmolP128 => executor.with_vllm_smol_p128_graph().with_separate_residual_norm().with_iteration_batch_completion().with_packed_async_metadata().with_grouped_ragged_attention_heads(),\n        RuntimeSelection::ExecutionGraph(_) => executor',1)
s=s.replace('            | RuntimeSelection::ExecutionGraph(_)\n','            | RuntimeSelection::ExecutionGraph(_) | RuntimeSelection::VllmSmolP128\n')
s=s.replace('                RuntimeSelection::ExecutionGraph(true)\n','                RuntimeSelection::ExecutionGraph(true) | RuntimeSelection::VllmSmolP128\n',1)
needle='    let measured = run_trials(&mut executor, &options, &token_rows);';s=s.replace(needle,'''    if options.prepare_only {
        let requests=token_rows.iter().cloned().map(PretokenizedBenchmarkRequest::new).collect();
        let prepared=executor.prepare_trial(requests,options.workload.output_tokens).map_err(|e| format!("trial preparation failed: {e}"))?;
        drop(prepared);
        let report=executor.close().map_err(|e| format!("preparation cleanup failed: {e}"))?;
        if !report.allocations_are_zero() || report.scheduler_completions()!=0 { return Err("preparation cleanup was not idle".to_owned()); }
        let receipt=serde_json::json!({"schema_version":"riley.native-profile-preparation.v1","prepared":true,"performance_trials":0,"source":options.source,"environment":options.environment,"workload":options.workload,"cleanup_allocations_zero":true});
        let mut bytes=serde_json::to_vec_pretty(&receipt).map_err(|_|"cannot serialize preparation receipt".to_owned())?;bytes.push(b'\\n');
        return write_output(options.output_path.as_deref(),&bytes);
    }
'''+needle,1)
s=s.replace('    let output_path = options.output_path;\n    let evidence = Evidence {\n        schema_version: SCHEMA_VERSION,','    let matched_vllm=matches!(options.runtime_selection()?,RuntimeSelection::VllmSmolP128);\n    let output_path = options.output_path;\n    let evidence = Evidence {\n        schema_version: if matched_vllm { "riley.vllm-profile-run.v1" } else { SCHEMA_VERSION },',1)
s=s.replace('  --semantic-class E0','  --semantic-class E0|VLLM_REFERENCE  VLLM_REFERENCE only for graph_numerics=vllm-smol-p128-v1\n  --prepare-only true|false          prepare/validate/close without any performance trial',1)
s=s.replace('decode_fast_path|execution_graph_policy\n','decode_fast_path|execution_graph_policy|graph_numerics\n',1)
# Add a contract test to the existing test module, which owns valid_arguments().
a=s.index('    #[test]\n    fn graph_profile_binds_single_row')
s=s[:a]+'''    #[test]
    fn vllm_profile_has_distinct_semantics_and_prepare_only_mode() {
        let mut args=valid_arguments("candidate","execution_graph_policy","require");
        for (flag,value) in [("--runtime-flag-name","graph_numerics"),("--runtime-flag-value","vllm-smol-p128-v1"),("--semantic-class","VLLM_REFERENCE"),("--correctness-gate-id","g04-vllm-smol-p128-v1"),("--prompt-tokens","128"),("--output-tokens","32"),("--concurrency","1")] {
            let i=args.iter().position(|x|x==flag).unwrap();args[i+1]=OsString::from(value);
        }
        args.extend([OsString::from("--prepare-only"),OsString::from("true")]);
        let Command::Run(options)=parse_arguments(args.clone()).expect("matched profile") else {panic!("run")};
        assert!(options.prepare_only);
        assert!(benchmark_config(&options).unwrap().executor.vllm_smol_p128_graph());
        let i=args.iter().position(|x|x=="--semantic-class").unwrap();args[i+1]=OsString::from("E0");assert!(parse_arguments(args.clone()).is_err());
        args[i+1]=OsString::from("VLLM_REFERENCE");let i=args.iter().position(|x|x=="--prompt-tokens").unwrap();args[i+1]=OsString::from("127");assert!(parse_arguments(args).is_err());
    }

'''+s[a:];p.write_text(s)
# Separate closed schema; the existing E0 schema and pair checker remain unchanged.
p=r/'benchmarks/schemas/native-profile-run.schema.json';d=json.loads(p.read_text());d['$id']='https://riley.dev/schemas/vllm-profile-run-v1.schema.json';d['title']='Riley candidate run qualified against a pinned vLLM reference';d['description']='Distinct numerical contract; not an E0 bitwise comparison with the old Riley eager profile.'
d['properties']['schema_version']={'const':'riley.vllm-profile-run.v1'};d['properties']['role']={'const':'candidate'};d['$defs']['source']['properties']['semantic_class']={'const':'VLLM_REFERENCE'};d['$defs']['source']['properties']['correctness_gate_id']={'const':'g04-vllm-smol-p128-v1'};d['$defs']['runtimeFlag']['properties']={'name':{'const':'graph_numerics'},'value':{'const':'vllm-smol-p128-v1'}}
(r/'benchmarks/schemas/vllm-profile-run.schema.json').write_text(json.dumps(d,indent=2)+'\n')
