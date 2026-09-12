from pathlib import Path
import os,subprocess,json
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
p=src/'crates/riley-server/src/main.rs';t=p.read_text();pos=t.index('    #[test]',t.index('mod tests'));t=t[:pos]+'''    #[test]
    fn mixed_v7_profile_preserves_aggregate_budget(){
        let result=super::parse_arguments(["serve","--model","/tmp/model","--graph-numerics","variable-smol-v7","--execution-graph-policy","require","--max-active-sequences","32","--batch-token-budget","1024","--prefill-chunk-tokens","512"].map(std::ffi::OsString::from)).unwrap();
        let super::CliCommand::Serve(options)=result else{panic!("serve")};assert!(options.variable_graph&&options.variable_graph32&&options.packed_prefill&&options.mixed_execution&&!options.variable_graph16);assert_eq!(options.prefill_chunk_tokens,512);
    }
'''+t[pos:];p.write_text(t)
for old,new in [('run_v6_http_v48.py','run_v7_http_v49.py'),('run_v6_fallback_v48.py','run_v7_fallback_v49.py')]:
 t=(r/old).read_text().replace('v6','v7').replace('v48','v49');(r/new).write_text(t)
s=(r/'qualify_packed_v48.py').read_text();s=s[s.index('env=os.environ.copy();'):].replace('v48','v49').replace('v6','v7').replace('packed-v49','mixed-v49').replace('packed_v7_profile','mixed_v7_profile');s='from pathlib import Path\nimport os,subprocess,json\nr=Path("/tmp/riley-opt-260912");s=r/"prefill-shapes-source-v11"\n'+s
(r/'run_mixed_qualification_v49.py').write_text(s)
subprocess.run(['python3',str(r/'run_mixed_qualification_v49.py')],check=True)
