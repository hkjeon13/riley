#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::collections::BTreeMap;

use std::io::{Seek,SeekFrom,Write};
#[derive(serde::Deserialize)]
struct Case {prompt:Vec<u32>,targets:Vec<u32>}
// Offline fixture producer; both backends consume the same human continuation.
fn generate(inputs:&[Case],steps:usize,name:&str)->Result<Vec<Vec<u32>>,Box<dyn std::error::Error>> {
 let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
 let context=CudaRuntime::initialize()?.device(0)?.create_context()?;
 let mut stream=context.create_stream()?;
 let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,2048)?,PreparedLlamaForwardConfig::default());
 let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
 let greedy=name=="greedy";
 let multi=name!="serial";
 let mut session=if greedy {executor.into_owned_variable_verification_greedy_session(&context,1024)?}else if multi {executor.into_owned_variable_verification_session(&context,1024)?}else{executor.into_owned_variable_projection_pipeline_session(&context,1024,false,false,false)?};
 assert!(session.read_verification_logits().is_err(),"read before completion");
 assert!(session.read_verification_tokens().is_err(),"tokens before completion");
 let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:128,max_waiting_prompt_tokens:32768,max_active_sequences:32,max_sequence_tokens:1024,iteration_token_budget:1024,max_prefill_chunk_tokens:if multi{8}else{128},aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:2048,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,2048,3,64)?,riley_scheduler::ExecutionShapePolicy::MixedPrefillDecode32)?;
 let directory=std::path::PathBuf::from(std::env::var("RILEY_VERIFICATION_HEAD_DIR")?);
 let mut dump=std::fs::File::create(directory.join(format!("{name}.bf16")))?;
 dump.set_len((inputs.len()*9*if greedy {4}else{98304})as u64)?;
 let mut indices=BTreeMap::new();
 let mut outputs=BTreeMap::new();
 for(index,case)in inputs.iter().enumerate() {let id=scheduler.submit(RequestDescriptor::new(case.prompt.clone(),steps),0)?.request_id();outputs.insert(id,Vec::new());indices.insert(id,index);}
 let mut seen=vec![0u32;inputs.len()*9];
 let mut iteration=0;let mut multi_endpoint_calls=0;
 while let Some(plan)=scheduler.plan_iteration(iteration+1)?.into_parts().0 {
  iteration+=1;
  let slots:BTreeMap<_,_>=plan.prefill_items().iter().chain(plan.decode_items()).filter_map(|w|w.output_slot().map(|s|(s.get(),w.request_id()))).collect();
  let authority=scheduler.authorize_execution(&plan)?;
  let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,false)?;
  if greedy {
   let verification=session.read_verification_tokens()?;
   assert_eq!(verification.iteration_id(),plan.iteration_id().get());
   assert!(verification.replay_id()>0 && verification.owner_generation()>0 && verification.catalog_digest()!=[0;32]);
   assert!(session.read_verification_logits().is_err(),"wrong result mode");
   let mut endpoints=0;
   for (i,row) in verification.rows().iter().enumerate(){
    assert_ne!(row.cookie,0);
    if (31..40).contains(&row.position){
     let index=indices.iter().find_map(|(id,&index)|(id.get()==row.sequence_tag).then_some(index)).unwrap();
     let slot=index*9+(row.position-31)as usize;seen[slot]+=1;
     dump.seek(SeekFrom::Start((slot*4)as u64))?;dump.write_all(&verification.token(i).unwrap().to_le_bytes())?;endpoints+=1;
    }
   }
   if endpoints>4 {multi_endpoint_calls+=1;}
  } else if multi {
   let verification=session.read_verification_logits()?;
   assert_eq!(verification.iteration_id(),plan.iteration_id().get());
   assert!(verification.replay_id()>0 && verification.owner_generation()>0 && verification.catalog_digest()!=[0;32]);
   assert!(verification.rows().iter().all(|r|r.cookie!=0));
   let mut endpoints=0;
   for (i,row) in verification.rows().iter().enumerate(){
    if (31..40).contains(&row.position){
     let index=indices.iter().find_map(|(id,&index)|(id.get()==row.sequence_tag).then_some(index)).unwrap();
     let slot=index*9+(row.position-31)as usize;seen[slot]+=1;
     dump.seek(SeekFrom::Start((slot*98304)as u64))?;dump.write_all(verification.logits(i).unwrap())?;endpoints+=1;
    }
   }
   if endpoints>4 {multi_endpoint_calls+=1;}
  }
  let mut samples=Vec::new();
  for slot in 0..downloaded.output_count() {
   let id=slots[&(slot as u32)];let index=indices[&id];let generated=outputs[&id].len();let token=inputs[index].targets[generated];
   if !multi {dump.seek(SeekFrom::Start(((index*steps+generated)*98304)as u64))?;
   dump.write_all(&downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304])?;}
   outputs.get_mut(&id).unwrap().push(token);samples.push(SampledIterationToken::new(token,false));
  }
  drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0))?;
  assert!(scheduler.complete_iteration(&result,iteration+1)?.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get())?;
  assert!(session.read_verification_logits().is_err(),"read after settlement");
  assert!(session.read_verification_tokens().is_err(),"tokens after settlement");
 }
 session.close()?;scheduler.close(iteration+2,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
 if multi {assert!(seen.iter().all(|&n|n==1),"all verification positions exactly once");assert!(multi_endpoint_calls>0);}
 eprintln!("VERIFY_HEAD_RUN lane={} model_iterations={} multi_endpoint_calls={} allocation_zero=true",name,iteration,multi_endpoint_calls);
 dump.sync_all()?;
 let result:Vec<_>=outputs.into_values().collect();assert!(result.iter().all(|v|v.len()==steps));Ok(result)
}
#[test]
#[ignore="requires SM89 pinned checkpoint and fixed natural cases"]
fn multi_position_head_matches_serial_target_logits()->Result<(),Box<dyn std::error::Error>> {
 let directory=std::path::PathBuf::from(std::env::var("RILEY_VERIFICATION_HEAD_DIR")?);
 let cases:Vec<Case>=serde_json::from_slice(&std::fs::read(directory.join("cases.json"))?)?;
 assert_eq!(cases.len(),8);assert!(cases.iter().all(|c|c.prompt.len()==32&&c.targets.len()==32));
 let expanded:Vec<_>=cases.iter().map(|c|{let mut prompt=c.prompt.clone();prompt.extend_from_slice(&c.targets[..8]);Case{prompt,targets:vec![c.targets[8]]}}).collect();
 generate(&cases,9,"serial")?;generate(&expanded,1,"multi")?;generate(&expanded,1,"greedy")?;
 let a=std::fs::read(directory.join("serial.bf16"))?;let b=std::fs::read(directory.join("multi.bf16"))?;
 assert_eq!(a.len(),8*9*98304);assert_eq!(a.len(),b.len());
 let differences=a.chunks_exact(2).zip(b.chunks_exact(2)).filter(|(x,y)|x!=y).count();
 eprintln!("VERIFICATION_HEAD_GATE output_rows=72 bytes_per_lane={} differing_bf16={} allocation_zero=true speculative_scheduler_implemented=false serving_measured=false",a.len(),differences);
 assert_eq!(differences,0,"multi-position head must preserve serial target logits");
 let tokens=std::fs::read(directory.join("greedy.bf16"))?;
 assert_eq!(tokens.len(),72*4);
 for (row,token) in a.chunks_exact(98304).zip(tokens.chunks_exact(4)) {
  let mut best=f32::NEG_INFINITY;let mut expected=0u32;
  for (id,bits) in row.chunks_exact(2).enumerate() {
   let value=f32::from_bits((u16::from_le_bytes([bits[0],bits[1]])as u32)<<16);
   assert!(value.is_finite());if value>best {best=value;expected=id as u32;}
  }
  assert_eq!(u32::from_le_bytes(token.try_into().unwrap()),expected);
 }
 eprintln!("VERIFICATION_GREEDY_GATE rows=72 exact_tokens=true auxiliary_bytes=512");Ok(())
}
