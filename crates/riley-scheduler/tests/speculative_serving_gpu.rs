#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::collections::BTreeMap;
#[derive(serde::Deserialize)]struct Case{prompt:Vec<u32>}
fn generate(cases:&[Case],speculative:bool)->Result<(Vec<Vec<u32>>,u64,u64,u64),Box<dyn std::error::Error>>{
 let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
 let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;
 let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,2048)?,PreparedLlamaForwardConfig::default()))?;
 let wide=speculative && std::env::var_os("RILEY_VERIFY_WIDE").is_some();
 let mut session=if wide{executor.into_owned_variable_verification_wide_session(&context,1024)?}else if speculative{executor.into_owned_variable_verification_greedy_session(&context,1024)?}else{executor.into_owned_variable_projection_pipeline_session(&context,1024,false,false,false)?};
 let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:128,max_waiting_prompt_tokens:32768,max_active_sequences:32,max_sequence_tokens:1024,iteration_token_budget:1024,max_prefill_chunk_tokens:128,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:2048,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,2048,3,64)?,riley_scheduler::ExecutionShapePolicy::MixedPrefillDecode32)?;
 let mut outputs=BTreeMap::new();for c in cases{let id=scheduler.submit(RequestDescriptor::new(c.prompt.clone(),32),0)?.request_id();outputs.insert(id,Vec::new());}
 let(mut now,mut calls,mut verify_calls,mut accepted)=(0u64,0u64,0u64,0u64);
 loop{
  now+=2;
  if speculative {if let Some(plan)=if wide{scheduler.plan_wide_speculative_iteration(now,None)?}else{scheduler.plan_speculative_iteration(now,None)?}{
   let authority=scheduler.authorize_speculative_execution(&plan)?;
   let targets=riley_scheduler::execution::execute_speculative_variable_graph(&authority,&mut session)?;drop(authority);
   let updates=scheduler.complete_speculative_iteration(&plan,&targets,now+1,IterationTiming::new(0,0))?;
   assert!(updates.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get())?;
   for e in updates.token_events(){outputs.get_mut(&e.request_id()).unwrap().push(e.token_id());}
   accepted+=updates.token_events().len().saturating_sub(plan.rows().len()) as u64;calls+=1;verify_calls+=1;continue;
  }}
  let Some(plan)=scheduler.plan_iteration(now)?.into_parts().0 else{break};
  let authority=scheduler.authorize_execution(&plan)?;
  let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,wide)?;
  let samples:Vec<_>=if wide{downloaded.greedy_token_ids().iter().map(|&t|SampledIterationToken::new(t,false)).collect()}else{downloaded.logits_bf16_native().chunks_exact(98304).map(|row|{
   let(mut best,mut token)=(f32::NEG_INFINITY,0u32);
   for(i,b)in row.chunks_exact(2).enumerate(){let v=f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16);assert!(v.is_finite());if v>best{best=v;token=i as u32;}}
   SampledIterationToken::new(token,false)
  }).collect()};drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0))?;
  let updates=scheduler.complete_iteration(&result,now+1)?;assert!(updates.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get())?;
  for e in updates.token_events(){outputs.get_mut(&e.request_id()).unwrap().push(e.token_id());}calls+=1;
 }
 session.close()?;scheduler.close(now+2,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
 let output:Vec<_>=outputs.into_values().collect();assert!(output.iter().all(|v|v.len()==32));Ok((output,calls,verify_calls,accepted))
}
#[test]
#[ignore="requires SM89 checkpoint and fixed natural cases"]
fn speculative_scheduler_gpu_matches_serial_generation()->Result<(),Box<dyn std::error::Error>>{
 let d=std::path::PathBuf::from(std::env::var("RILEY_SPECULATIVE_GPU_DIR")?);
 let cases:Vec<Case>=serde_json::from_slice(&std::fs::read(d.join("cases.json"))?)?;
 let(serial,serial_calls,_,_)=generate(&cases,false)?;let(speculative,spec_calls,verify_calls,accepted)=generate(&cases,true)?;
 let differences=serial.iter().flatten().zip(speculative.iter().flatten()).filter(|(a,b)|a!=b).count();
 std::fs::write(d.join("generation.json"),serde_json::to_vec_pretty(&serde_json::json!({"serial":serial,"speculative":speculative,"serial_calls":serial_calls,"speculative_calls":spec_calls,"verification_calls":verify_calls,"differing_tokens":differences,"accepted_draft_tokens":accepted}))?)?;
 eprintln!("SPECULATIVE_GPU_GATE requests={} differing_tokens={} verification_calls={} serial_calls={} speculative_calls={} serving_measured=false",cases.len(),differences,verify_calls,serial_calls,spec_calls);
 assert!(verify_calls>0);if std::env::var_os("RILEY_REQUIRE_SPECULATIVE_ACCEPTANCE").is_some(){assert!(accepted>0,"acceptance control must commit multiple verified inputs");}assert_eq!(differences,0,"strict serial generation equivalence");Ok(())
}
