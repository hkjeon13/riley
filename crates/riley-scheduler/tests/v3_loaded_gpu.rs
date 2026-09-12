#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig,variable_session::VariableGraphBuffers};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::{fs::File,io::Read,path::PathBuf};
fn read32(f:&mut File)->std::io::Result<u32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(u32::from_le_bytes(b))}
fn read64(f:&mut File)->std::io::Result<u64>{let mut b=[0;8];f.read_exact(&mut b)?;Ok(u64::from_le_bytes(b))}
#[test]
#[ignore="requires actual pinned checkpoint fixture and GPU"]
fn loaded_variable_scheduler_gpu_completion_and_settlement()->Result<(),Box<dyn std::error::Error>>{
let root=PathBuf::from(std::env::var_os("RILEY_V3_MODEL_FIXTURE").ok_or("fixture directory missing")?);let model=LoadedModel::load(PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?).as_path(),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
let mut f=File::open(root.join("requests.bin"))?;let count=read32(&mut f)?;let mut requests=vec![];for _ in 0..count{let n=read32(&mut f)?;let mut tokens=vec![];for _ in 0..n{tokens.push(read32(&mut f)?);}requests.push(tokens);}
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut replay_count=0;

for prompt in requests {
 let limit=match prompt.len(){16=>32,128=>64,_=>128};
let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,64)?,PreparedLlamaForwardConfig::default());
 let mut executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
 assert_eq!(executor.maximum_position_count()?,8192);
 let mut scratch=VariableGraphBuffers::prepare(&context,1024)?;
 let mut session=executor.prepare_variable_session(&mut stream,&mut scratch)?;
 let mut scheduler=Scheduler::new(SchedulerConfig{max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:1,
 max_sequence_tokens:1024,iteration_token_budget:73,max_prefill_chunk_tokens:73,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,
 admission_timeout_ns:None,max_promised_kv_blocks:64,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,64,3,64)?)?;
 scheduler.submit(RequestDescriptor::new(prompt.clone(),limit),0)?;
 let reference=std::fs::read(root.join(format!("decode-logits-{}.bf16",prompt.len())))?;
 let mut generated=0;let mut iteration=0;
 while generated<limit {
  iteration+=1;let plan=scheduler.plan_iteration(iteration*2)?.into_parts().0.unwrap();
  let authority=scheduler.authorize_execution(&plan)?;
  let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session).unwrap();
  assert!(session.issue().is_err(),"cannot issue before settlement");
  assert!(session.confirm_scheduler_commit(plan.iteration_id().get()+1).is_err());
  let samples=if downloaded.output_count()==1 {
    let logits=downloaded.logits_bf16_native();let expected=&reference[generated*98304..(generated+1)*98304]; if logits!=expected {std::fs::write(root.join("loaded-mismatch.bf16"),logits)?; panic!("loaded/fixture mismatch prompt={} generated={} differing_bf16={}",prompt.len(),generated,logits.chunks_exact(2).zip(expected.chunks_exact(2)).filter(|(a,b)|a!=b).count());}
    let token=logits.chunks_exact(2).enumerate().map(|(i,b)|(i,f32::from_bits((u16::from_le_bytes([b[0],b[1]])as u32)<<16)))
       .max_by(|a,b|a.1.total_cmp(&b.1).then_with(||b.0.cmp(&a.0))).unwrap().0 as u32;
    generated+=1;vec![SampledIterationToken::new(token,false)]
  }else{assert!(downloaded.logits_bf16_native().is_empty());vec![]};
  drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0)).unwrap();
  let updates=scheduler.complete_iteration(&result,iteration*2+1)?;assert!(updates.settlement_failures().is_empty());
  session.confirm_scheduler_commit(plan.iteration_id().get()).unwrap();replay_count+=1;
 }
 scheduler.close(iteration*2+2,None)?;session.close()?;scratch.close()?;executor.close()?;
 eprintln!("V3_LOADED_SCHEDULER prompt={} output={} iterations={} logits_reference_exact=true",prompt.len(),limit,iteration);
}
stream.close()?;assert!(context.allocation_stats()?.is_zero());eprintln!("V3_LOADED_SCHEDULER replays={} allocation_zero=true",replay_count);Ok(())}
