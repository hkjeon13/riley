#![cfg(feature="cuda")]
use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::{fs::File,io::Read,path::PathBuf};
fn read32(f:&mut File)->std::io::Result<u32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(u32::from_le_bytes(b))}
fn read64(f:&mut File)->std::io::Result<u64>{let mut b=[0;8];f.read_exact(&mut b)?;Ok(u64::from_le_bytes(b))}
#[test]
#[ignore="requires actual pinned checkpoint fixture and GPU"]
fn variable_scheduler_gpu_completion_and_settlement()->Result<(),Box<dyn std::error::Error>>{
let root=PathBuf::from(std::env::var_os("RILEY_V3_MODEL_FIXTURE").ok_or("fixture directory missing")?);let mut f=File::open(root.join("weights.bin"))?;let mut weight_bytes=vec![];for _ in 0..273{let size=read64(&mut f)?;let mut b=vec![0;size as usize];f.read_exact(&mut b)?;weight_bytes.push(b);}
let rope=std::fs::read(root.join("rope.bin"))?;assert_eq!(rope.len(),1024*32*4*2);
let mut f=File::open(root.join("requests.bin"))?;let count=read32(&mut f)?;let mut requests=vec![];for _ in 0..count{let n=read32(&mut f)?;let mut tokens=vec![];for _ in 0..n{tokens.push(read32(&mut f)?);}requests.push(tokens);}
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut upload=context.allocate_pinned_host_buffer(1<<20)?;let mut replay_count=0;

for prompt in requests {
 let limit=match prompt.len(){16=>32,128=>64,_=>128};
let mut sizes=vec![1152*1024;5];sizes.extend([384*1024,384*1024,384*1024,3072*1024,3072*1024,2304*1024,3072*1024]);sizes.extend([30*64*16*384,30*64*16*384,1024*128,1024*128,17536,1152,128,4,98304,8]);let mut buffers=vec![];
for size in sizes{let mut b=context.allocate_device_buffer(size)?;b.upload_from_slice(0,&vec![0;size as usize],&mut upload,&mut stream)?;buffers.push(b);}
buffers[14].upload_from_slice(0,&rope[..1024*128],&mut upload,&mut stream)?;buffers[15].upload_from_slice(0,&rope[1024*128..],&mut upload,&mut stream)?;
let mut weights=vec![];for data in &weight_bytes{if data.is_empty(){weights.push(weights[0]);continue;}let mut b=context.allocate_device_buffer(data.len()as u64)?;b.upload_from_slice(0,data,&mut upload,&mut stream)?;weights.push(buffers.len());buffers.push(b);}
let mut head=context.prepare_gemm(CudaGemmConfig::new(1,49152,576,0)?)?;let mut staging=context.allocate_pinned_host_buffer(196864)?;
let mut owner=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream:&mut stream,devices:buffers.iter_mut().collect(),pinned:vec![&mut staging],plans:vec![&mut head]})?;owner.record_v3_prefill(&std::array::from_fn(|i|i),None,&weights,0,0,1024,64)?;
 let mut session=riley_runtime::llama::variable_session::BorrowedVariableSession::new(owner,[11;32],64,1024).unwrap();
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
    let logits=downloaded.logits_bf16_native();assert_eq!(logits,&reference[generated*98304..(generated+1)*98304]);
    let token=logits.chunks_exact(2).enumerate().map(|(i,b)|(i,f32::from_bits((u16::from_le_bytes([b[0],b[1]])as u32)<<16)))
       .max_by(|a,b|a.1.total_cmp(&b.1).then_with(||b.0.cmp(&a.0))).unwrap().0 as u32;
    generated+=1;vec![SampledIterationToken::new(token,false)]
  }else{assert!(downloaded.logits_bf16_native().is_empty());vec![]};
  drop(authority);
  let result=downloaded.into_result(&samples,IterationTiming::new(0,0)).unwrap();
  let updates=scheduler.complete_iteration(&result,iteration*2+1)?;assert!(updates.settlement_failures().is_empty());
  session.confirm_scheduler_commit(plan.iteration_id().get()).unwrap();replay_count+=1;
 }
 scheduler.close(iteration*2+2,None)?;session.close()?;head.close()?;drop(buffers);drop(staging);
 eprintln!("V3_SCHEDULER prompt={} output={} iterations={} logits_reference_exact=true",prompt.len(),limit,iteration);
}
drop(upload);stream.close()?;assert!(context.allocation_stats()?.is_zero());eprintln!("V3_SCHEDULER replays={} allocation_zero=true",replay_count);Ok(())}
