#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::{fs::File,io::Read,path::PathBuf};
fn read32(f:&mut File)->std::io::Result<u32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(u32::from_le_bytes(b))}
#[test]
#[ignore="requires pinned checkpoint and GPU; owned shared scheduler"]
fn loaded_shared_scheduler_matches_every_logit()->Result<(),Box<dyn std::error::Error>>{
let root=PathBuf::from(std::env::var_os("RILEY_V3_MODEL_FIXTURE").ok_or("fixture missing")?);let model=LoadedModel::load(PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("model missing")?).as_path(),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
let mut f=File::open(root.join("requests.bin"))?;let count=read32(&mut f)?;let mut requests=vec![];for _ in 0..count{let n=read32(&mut f)?;let mut tokens=vec![];for _ in 0..n{tokens.push(read32(&mut f)?);}requests.push(tokens);}
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;
let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,64)?,PreparedLlamaForwardConfig::default());
let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;let mut session=executor.into_owned_variable_shared_session(&context,128)?;
let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:4,max_sequence_tokens:1024,iteration_token_budget:73,max_prefill_chunk_tokens:73,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:64,metrics_window_samples:16},riley_runtime::paged_kv::KvLayout::checked(30,64,3,64)?,riley_scheduler::ExecutionShapePolicy::VariablePrefillDecodeN)?;
let mut refs=std::collections::BTreeMap::new();for prompt in requests {let limit=match prompt.len(){16=>32,128=>64,_=>128};let reference=std::fs::read(root.join(format!("decode-logits-{}.bf16",prompt.len())))?;let id=scheduler.submit(RequestDescriptor::new(prompt,limit),0)?.request_id();refs.insert(id,(reference,0usize,limit));}
let(mut iteration,mut widest,mut checked)=(0u64,0usize,0usize);
loop {
 iteration+=1;let Some(plan)=scheduler.plan_iteration(iteration*2)?.into_parts().0 else{break};widest=widest.max(plan.decode_items().len());
 let slots:std::collections::BTreeMap<_,_>=plan.prefill_items().iter().chain(plan.decode_items()).filter_map(|w|w.output_slot().map(|slot|(slot.get(),w.request_id()))).collect();
 let authority=scheduler.authorize_execution(&plan)?;let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session).unwrap();
 assert!(session.issue_rows(1).is_err());assert!(session.confirm_scheduler_commit(plan.iteration_id().get()+1).is_err());
 let mut samples=vec![];
 for slot in 0..downloaded.output_count(){let id=slots[&(slot as u32)];let(reference,generated,_)=refs.get_mut(&id).unwrap();let logits=&downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304];assert_eq!(logits,&reference[*generated*98304..(*generated+1)*98304],"request {:?} generated {}",id,*generated);
 let token=logits.chunks_exact(2).enumerate().map(|(i,b)|(i,f32::from_bits((u16::from_le_bytes([b[0],b[1]])as u32)<<16))).max_by(|a,b|a.1.total_cmp(&b.1).then_with(||b.0.cmp(&a.0))).unwrap().0 as u32;
 samples.push(SampledIterationToken::new(token,false));*generated+=1;checked+=1;
 }
 drop(authority);let result=downloaded.into_result(&samples,IterationTiming::new(0,0)).unwrap();assert!(scheduler.complete_iteration(&result,iteration*2+1)?.settlement_failures().is_empty());session.confirm_scheduler_commit(plan.iteration_id().get()).unwrap();
}
assert!(widest>=3);for(_,(_,generated,limit))in refs{assert_eq!(generated,limit);}
scheduler.submit(RequestDescriptor::new(vec![17;128],4),iteration*2+1)?;
let plan=scheduler.plan_iteration(iteration*2+2)?.into_parts().0.unwrap();let authority=scheduler.authorize_execution(&plan)?;
let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session).unwrap();assert_eq!(downloaded.output_count(),0);drop(authority);drop(downloaded);assert!(session.issue_rows(2).is_err());
session.close()?;assert!(scheduler.abort_iteration(plan.iteration_id(),riley_scheduler::ExecutionAbort::DeviceQuiescedMutationUnknown,iteration*2+3)?.settlement_failures().is_empty());scheduler.close(iteration*2+4,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());eprintln!("LOADED_SHARED rows_max={} logits_checked={} iterations={} pending_close_abort=true allocation_zero=true",widest,checked,iteration-1);Ok(())}
