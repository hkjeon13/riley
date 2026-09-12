from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('''        if wire::validate_result(&self.output,e).is_err() {self.poisoned=true;return Err(bad("invalid GPU completion; close required"));}
        self.completed=true;
        wire::validate_result(&self.output,e)''','''        match wire::validate_result(&self.output,e) {
            Ok(result)=>{self.completed=true;Ok(result)},
            Err(_)=>{self.poisoned=true;Err(bad("invalid GPU completion; close required"))}
        }''');p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text();line='    let mut logits=zeroed_vec(prepared.output_count*49152*2,"V3 logits").map_err(|e|IterationExecutionFailure::new(id,Some(ExecutionAbort::NotDispatched),e))?;';s=s.replace(line+'\n','');pos=s.index('    let (identity,replay,cookie)=executor.issue()');s=s[:pos]+line+'\n'+s[pos:];p.write_text(s)
p=r/'crates/riley-scheduler/Cargo.toml';s=p.read_text().replace('[dev-dependencies]','[dev-dependencies]\nriley-cuda = { path = "../riley-cuda" }');p.write_text(s)
source=(r/'crates/riley-cuda/tests/v3_recorder_decode_gpu.rs').read_text()
setup=source[source.index('let root=PathBuf'):source.index('for prompt in requests')]
alloc=source[source.index('let mut sizes='):source.index('let mut out=vec![0;98432];')]
# allocation segment only depends on existing context/weights/rope
body='''
for prompt in requests {
 let limit=match prompt.len(){16=>32,128=>64,_=>128};
'''+alloc+'''
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
'''
header='''#![cfg(feature="cuda")]
use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming};
use std::{fs::File,io::Read,path::PathBuf};
fn read32(f:&mut File)->std::io::Result<u32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(u32::from_le_bytes(b))}
fn read64(f:&mut File)->std::io::Result<u64>{let mut b=[0;8];f.read_exact(&mut b)?;Ok(u64::from_le_bytes(b))}
#[test]
#[ignore="requires actual pinned checkpoint fixture and GPU"]
fn variable_scheduler_gpu_completion_and_settlement()->Result<(),Box<dyn std::error::Error>>{
'''
(r/'crates/riley-scheduler/tests/v3_variable_gpu.rs').write_text(header+setup+body)
