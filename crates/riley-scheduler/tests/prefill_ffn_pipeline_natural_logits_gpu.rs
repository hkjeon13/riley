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
fn generate(experimental:bool,inputs:&[Case])->Result<Vec<Vec<u32>>,Box<dyn std::error::Error>> {
 let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
 let context=CudaRuntime::initialize()?.device(0)?.create_context()?;
 let mut stream=context.create_stream()?;
 let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,2048)?,PreparedLlamaForwardConfig::default());
 let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
 let mut session=if experimental {executor.into_owned_variable_prefill_ffn_pipeline_session(&context,1024,false,false)?}else{executor.into_owned_variable_mixed_session(&context,1024,false)?};
 let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:64,max_waiting_prompt_tokens:32768,max_active_sequences:32,max_sequence_tokens:1024,iteration_token_budget:1024,max_prefill_chunk_tokens:128,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:2048,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,2048,3,64)?,riley_scheduler::ExecutionShapePolicy::MixedPrefillDecode32)?;
 let directory=std::path::PathBuf::from(std::env::var("RILEY_FFN_NATURAL_SCREEN_DIR")?);
 let mut dump=std::fs::File::create(directory.join(if experimental {"candidate.bf16"}else{"baseline.bf16"}))?;
 dump.set_len((inputs.len()*32*98304)as u64)?;
 let mut indices=BTreeMap::new();
 let mut outputs=BTreeMap::new();
 for(index,case)in inputs.iter().enumerate() {let id=scheduler.submit(RequestDescriptor::new(case.prompt.clone(),32),0)?.request_id();outputs.insert(id,Vec::new());indices.insert(id,index);}
 let mut iteration=0;
 while let Some(plan)=scheduler.plan_iteration(iteration+1)?.into_parts().0 {
  iteration+=1;
  let slots:BTreeMap<_,_>=plan.prefill_items().iter().chain(plan.decode_items()).filter_map(|w|w.output_slot().map(|s|(s.get(),w.request_id()))).collect();
  let authority=scheduler.authorize_execution(&plan)?;
  let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,false)?;
  let mut samples=Vec::new();
  for slot in 0..downloaded.output_count() {
   let id=slots[&(slot as u32)];let index=indices[&id];let generated=outputs[&id].len();let token=inputs[index].targets[generated];
   dump.seek(SeekFrom::Start(((index*32+generated)*98304)as u64))?;
   dump.write_all(&downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304])?;
   outputs.get_mut(&id).unwrap().push(token);samples.push(SampledIterationToken::new(token,false));
  }
  drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0))?;
  assert!(scheduler.complete_iteration(&result,iteration+1)?.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get())?;
 }
 session.close()?;scheduler.close(iteration+2,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
 dump.sync_all()?;
 let result:Vec<_>=outputs.into_values().collect();assert!(result.iter().all(|v|v.len()==32));Ok(result)
}
#[test]
#[ignore="requires pinned model, prefill FFN pipeline and prepared natural screen"]
fn dump_natural_logits()->Result<(),Box<dyn std::error::Error>> {
 let path=std::path::PathBuf::from(std::env::var("RILEY_FFN_NATURAL_SCREEN_DIR")?).join("cases.json");
 let cases:Vec<Case>=serde_json::from_slice(&std::fs::read(path)?)?;
 assert_eq!(cases.len(),8);assert!(cases.iter().all(|c|c.prompt.len()==32&&c.targets.len()==32));
 assert_eq!(generate(false,&cases)?.len(),8);assert_eq!(generate(true,&cases)?.len(),8);
 let directory=std::path::PathBuf::from(std::env::var("RILEY_FFN_NATURAL_SCREEN_DIR")?);
 assert!(std::fs::read(directory.join("baseline.bf16"))? == std::fs::read(directory.join("candidate.bf16"))?,"prefill FFN pipeline full logits must be bitwise equal");
 eprintln!("NATURAL_DUMP passages=8 targets_per_passage=32 backends=2 allocation_zero=true serving_measured=false");Ok(())
}
