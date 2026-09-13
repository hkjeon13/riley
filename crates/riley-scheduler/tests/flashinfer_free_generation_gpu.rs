#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::collections::BTreeMap;

// Independent synthetic boundary probes, not a natural-language quality corpus.
fn prompts()->Vec<Vec<u32>> {
 let mut state=9131701u64;
 [1,15,16,17,127,128,129,511].into_iter().map(|length| (0..length).map(|_| {
  state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  32+((state>>32)%49000)as u32
 }).collect()).collect()
}
fn generate(experimental:bool,inputs:&[Vec<u32>])->Result<Vec<Vec<u32>>,Box<dyn std::error::Error>> {
 let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
 let context=CudaRuntime::initialize()?.device(0)?.create_context()?;
 let mut stream=context.create_stream()?;
 let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,2048)?,PreparedLlamaForwardConfig::default());
 let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
 let mut session=if experimental {executor.into_owned_variable_flashinfer_experimental_session(&context,1024,true)?}else{executor.into_owned_variable_mixed_session(&context,1024,true)?};
 let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:64,max_waiting_prompt_tokens:32768,max_active_sequences:32,max_sequence_tokens:1024,iteration_token_budget:1024,max_prefill_chunk_tokens:128,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:2048,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,2048,3,64)?,riley_scheduler::ExecutionShapePolicy::MixedPrefillDecode32)?;
 let mut outputs=BTreeMap::new();
 for prompt in inputs {let id=scheduler.submit(RequestDescriptor::new(prompt.clone(),32),0)?.request_id();outputs.insert(id,Vec::new());}
 let mut iteration=0;
 while let Some(plan)=scheduler.plan_iteration(iteration+1)?.into_parts().0 {
  iteration+=1;
  let slots:BTreeMap<_,_>=plan.prefill_items().iter().chain(plan.decode_items()).filter_map(|w|w.output_slot().map(|s|(s.get(),w.request_id()))).collect();
  let authority=scheduler.authorize_execution(&plan)?;
  let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,true)?;
  let mut samples=Vec::new();
  for(slot,&token)in downloaded.greedy_token_ids().iter().enumerate(){outputs.get_mut(&slots[&(slot as u32)]).unwrap().push(token);samples.push(SampledIterationToken::new(token,false));}
  drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0))?;
  assert!(scheduler.complete_iteration(&result,iteration+1)?.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get())?;
 }
 session.close()?;scheduler.close(iteration+2,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
 let result:Vec<_>=outputs.into_values().collect();assert!(result.iter().all(|v|v.len()==32));Ok(result)
}
#[test]
#[ignore="requires optional FlashInfer, pinned checkpoint and GPU; independent free-running probe"]
fn independent_free_generation()->Result<(),Box<dyn std::error::Error>> {
 let unique=prompts();let inputs:Vec<_>=unique.iter().cycle().take(32).cloned().collect();
 let baseline=generate(false,&inputs)?;let candidate=generate(true,&inputs)?;
 eprintln!("FREE_TOKENS baseline={:?}",baseline);eprintln!("FREE_TOKENS candidate={:?}",candidate);
 let mut differences=0;let mut sequence_differences=0;
 for(i,(a,b))in baseline.iter().zip(&candidate).enumerate(){
  differences+=a.iter().zip(b).filter(|(x,y)|x!=y).count();
  if let Some(first)=a.iter().zip(b).position(|(x,y)|x!=y){sequence_differences+=1;eprintln!("FREE_DIVERGENCE request={} prompt_tokens={} first_index={} baseline={} candidate={}",i,inputs[i].len(),first,a[first],b[first]);}
 }
 for i in 8..32 {assert_eq!(baseline[i],baseline[i%8],"baseline batch invariance");assert_eq!(candidate[i],candidate[i%8],"candidate batch invariance");}
 eprintln!("FREE_RESULT requests=32 tokens=1024 differing_sequences={} differing_tokens={} repeated_prompt_invariance=true natural_language_quality_tested=false",sequence_differences,differences);
 assert_eq!(differences,0,"independent free-running greedy equivalence");Ok(())
}
