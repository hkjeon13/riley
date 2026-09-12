from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('wire::ResultMode::Greedy','super::multi_descriptor::ResultMode::Greedy').replace('Self::prepare_base(context,capacity,true,false)','Self::prepare_base(context,capacity,true)');s=s.replace('    pub fn into_owned_variable_shared16_greedy_session','    pub fn into_owned_variable_shared_greedy_session_rows<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{self.into_variable_session::<ROWS>(context,capacity,true,true)}\n    pub fn into_owned_variable_shared16_greedy_session');s=s.replace('if !matches!(ROWS,8|16) {return Err(super::LlamaBatchExecutorError','if !matches!(ROWS,8|16) || (compact && (!shared || ROWS!=16)) {return Err(super::LlamaBatchExecutorError');p.write_text(s)
p=r/'crates/riley-scheduler/tests/v3_shared_owned_gpu.rs';s=p.read_text();s=s.replace('let root=PathBuf', '''run_shared_profile_mode::<ROWS>(active,request_count,physical,capacity,chunk,false)
}
#[test]
#[ignore="requires pinned checkpoint and GPU; alternating compact and full completions"]
fn loaded_compact_sixteen_alternates_full_and_greedy()->Result<(),Box<dyn std::error::Error>>{run_shared_profile_mode::<16>(32,32,2048,128,73,true)}
#[test]
#[ignore="requires pinned checkpoint and GPU; compact partial and wide prefill"]
fn loaded_compact_sixteen_prefill_shapes()->Result<(),Box<dyn std::error::Error>>{run_shared_profile_mode::<16>(4,3,64,512,129,true)}
fn run_shared_profile_mode<const ROWS:usize>(active:usize,request_count:usize,physical:usize,capacity:u32,chunk:usize,compact:bool)->Result<(),Box<dyn std::error::Error>>{
let root=PathBuf''',1)
s=s.replace('let mut session=executor.into_owned_variable_shared_session_rows::<ROWS>(&context,capacity)?;','let mut session=if compact{executor.into_owned_variable_shared_greedy_session_rows::<ROWS>(&context,capacity)?}else{executor.into_owned_variable_shared_session_rows::<ROWS>(&context,capacity)?};assert_eq!(session.supports_compact_greedy(),compact);')
s=s.replace('let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session).unwrap();','let greedy=compact && iteration%3!=0;let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,greedy).unwrap();',1)
a='let logits=&downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304];assert_eq!(logits,&reference[*generated*98304..(*generated+1)*98304],"request {:?} generated {}",id,*generated);'
b='let logits=&reference[*generated*98304..(*generated+1)*98304];if !greedy{assert_eq!(&downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304],logits,"request {:?} generated {}",id,*generated);}else{assert!(downloaded.logits_bf16_native().is_empty());}'
assert a in s;s=s.replace(a,b)
s=s.replace(' samples.push(SampledIterationToken::new(token,false));',' if greedy{assert_eq!(downloaded.greedy_token_ids()[slot],token,"compact request {:?} generated {}",id,*generated);}\n samples.push(SampledIterationToken::new(token,false));')
s=s.replace('let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session).unwrap();assert_eq!(downloaded.output_count(),0);','let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,compact).unwrap();assert_eq!(downloaded.output_count(),0);')
p.write_text(s)
