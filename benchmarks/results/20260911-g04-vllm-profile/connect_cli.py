from pathlib import Path
import re
r=Path('/tmp/riley-g04-vllm-profile-source-260911')
p=r/'crates/riley-server/src/main.rs';s=p.read_text();s=s.replace('    reduction_profile: ReductionProfileMode,','    reduction_profile: ReductionProfileMode,\n    vllm_smol_p128_graph: bool,',1).replace('    let mut execution_graph_policy = None;','    let mut execution_graph_policy = None;\n    let mut graph_numerics = None;',1)
needle='            "--execution-graph-policy" => {';s=s.replace(needle,'''            "--graph-numerics" => {
                let value=next_value(&mut arguments,"--graph-numerics")?;
                let enabled=match value.to_str() { Some("existing")=>false, Some("vllm-smol-p128-v1")=>true, _=>return Err("--graph-numerics requires existing or vllm-smol-p128-v1".to_owned()) };
                set_once(&mut graph_numerics,enabled,"--graph-numerics")?;
            }
'''+needle,1)
needle='    Ok(CliCommand::Serve(ServeOptions {';s=s.replace(needle,'''    let vllm_smol_p128_graph=graph_numerics.unwrap_or(false);
    if vllm_smol_p128_graph && execution_graph_policy!=Some(riley_runtime::llama::ExecutionGraphPolicy::Require) { return Err("vllm-smol-p128-v1 requires --execution-graph-policy require".to_owned()); }
'''+needle,1)
s=s.replace('        reduction_profile: reduction_profile.unwrap_or(ReductionProfileMode::CanonicalV1),','        reduction_profile: reduction_profile.unwrap_or(ReductionProfileMode::CanonicalV1),\n        vllm_smol_p128_graph,',1)
s=re.sub(r'(?m)^(\s*)reduction_profile: ReductionProfileMode::([^\n]+),$',r'\1reduction_profile: ReductionProfileMode::\2,\n\1vllm_smol_p128_graph: false,',s)
needle='    // Full graph kernels use the reviewed exact grouped-head implementation.';s=s.replace(needle,'    let executor=if options.vllm_smol_p128_graph { executor.with_vllm_smol_p128_graph() } else { executor };\n'+needle,1)
# Add usage beside the existing graph flag, without broad replacement of parser strings.
a=s.index('const USAGE:');b=s.index('";',a)
u=s[a:b];lines=u.splitlines();lines=[line+'\n  --graph-numerics existing|vllm-smol-p128-v1  explicit bounded arithmetic (default: existing)' if line.lstrip().startswith('--execution-graph-policy') else line for line in lines];s=s[:a]+'\n'.join(lines)+s[b:]
p.write_text(s)
# Runtime test: preserve the old test unchanged and add the new contract case.
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();a=s.index('    fn owned_graph_smol_p128_o32_reuses_scheduler_block_mappings');a=s.rfind('    #[test]',0,a);b=s.rfind('\n}')
t=s[a:b];assert t.count('fn owned_graph_')==1
t=t.replace('owned_graph_smol_p128_o32_reuses_scheduler_block_mappings','owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings')
# The test uses one shared config for baseline and candidate: select only candidate.
needle='PreparedLlamaBatchExecutor::prepare';idx=t.index(needle);print(t[max(0,idx-100):idx+330])
# save intermediate for exact constructor adjustment after inspecting it
(r/'vllm-test-template.txt').write_text(t)
