//! Finite hash-exact native batch-executor generation, not a performance lane.
#![cfg(feature="cuda")]
use std::{error::Error,fs,path::PathBuf};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use riley_model::{LoadedModel,LoadLimits};
use riley_scheduler::{SchedulerConfig,OverloadPolicy};
use riley_runtime::llama::{LlamaBatchMetadataConfig,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig};
use riley_server::benchmark::{NativeBenchmarkConfig,NativeBenchmarkExecutor,PretokenizedBenchmarkRequest};
type Result<T=()> = std::result::Result<T,Box<dyn Error>>;
fn ids(v:&Value)->Result<Vec<u32>> {v.as_array().ok_or("IDs missing")?.iter().map(|v|Ok(u32::try_from(v.as_u64().ok_or("ID not unsigned")?)?)).collect()}
#[test]
#[ignore="pinned single GPU and fresh HF fixtures"]
fn batch_executor_owns_all_model_greedy_prefixes_and_resources() -> Result {
 let root=PathBuf::from(std::env::var("RILEY_FRESH_HF_ROOT")?);let output=PathBuf::from(std::env::var("RILEY_BATCH_NATIVE_OUTPUT")?);fs::create_dir(&output)?;let mut models=Vec::new();
 let selected=std::env::var("RILEY_BATCH_MODEL_KEYS").unwrap_or_else(|_|"smol,qwen25,qwen17,qwen4".to_owned());
 for key in selected.split(',') {
  let f:Value=serde_json::from_slice(&fs::read(root.join(key).join("fixture.json"))?)?;
  let limit=f["weight_limit_bytes"].as_u64().ok_or("limit missing")?;
  let model=LoadedModel::load(&PathBuf::from(f["model_path"].as_str().ok_or("path missing")?),LoadLimits::default().with_weight_byte_limits(limit,limit)?)?;
  let scheduler=SchedulerConfig {max_waiting_requests:1,max_waiting_prompt_tokens:256,max_active_sequences:1,max_sequence_tokens:256,iteration_token_budget:128,max_prefill_chunk_tokens:128,aging_threshold_ns:10_000_000,overload_policy:OverloadPolicy::RejectImmediately,admission_timeout_ns:None,max_promised_kv_blocks:16,metrics_window_samples:32};
  let attention=model.spec().blocks().first().ok_or("model layers missing")?.attention();
  let head_dimension=attention.head_dimension();
  let wide_query=attention.query_heads()*head_dimension>attention.hidden_size();
  let forward=if wide_query {PreparedLlamaForwardConfig::default().with_merged_qkv_projection()} else {PreparedLlamaForwardConfig::default()};
  let executor=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,128,16,1,16)?,forward);
  let executor=if wide_query {executor.with_fused_qkv_rope()} else {executor};
  let executor=if head_dimension==128 {executor.with_native_bf16_paged_split_gqa_d128_two_stage().with_serving_sequence_token_bound(256)} else {executor};
  let mut engine=match NativeBenchmarkExecutor::prepare(model,NativeBenchmarkConfig {device_ordinal:0,scheduler,executor,gpu_greedy:false}) {Ok(x)=>x,Err(e)=>{let r=json!({"model":f["model_id"],"passed":false,"preparation_error":e.to_string()});fs::write(output.join(format!("{key}.json")),serde_json::to_vec_pretty(&r)?)?;models.push(r);continue;}};
  let operation=(|| -> Result<Vec<Value>> {
   let mut cases=Vec::new();
   for c in f["cases"].as_array().ok_or("cases missing")? {
    let expected=ids(&c["expected_token_ids"])?;let mut h=Sha256::new();for t in &expected {h.update(t.to_le_bytes());}let expected_hash=h.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>();
    let trial=engine.prepare_trial(vec![PretokenizedBenchmarkRequest::new(ids(&c["prompt_token_ids"])?)],expected.len())?;
    let trace=engine.run_trial(trial)?;let req=trace.requests().first().ok_or("no completed request")?;
    let got=req.generated_token_ids_sha256().to_hex();cases.push(json!({"case":c["id"],"expected_token_ids":expected,"expected_u32le_sha256":expected_hash,"actual_u32le_sha256":got,"exact":got==expected_hash,"generated_tokens":req.generated_tokens(),"mode":"fixed-prefix including EOS as ordinary; normal HTTP termination remains separately required"}));
   }Ok(cases)
  })();
  let cleanup=engine.close()?;let zero=cleanup.allocations_are_zero();let (cases,error)=match operation {Ok(c)=>(c,None),Err(e)=>(Vec::new(),Some(e.to_string()))};
  let passed=error.is_none() && zero && cases.len()==3 && cases.iter().all(|c|c["exact"]==true);
  let report=json!({"model":f["model_id"],"revision":f["revision"],"passed":passed,"cases":cases,"execution_error":error,"close_allocations_zero":zero,"device_live_bytes":cleanup.device_live_bytes(),"pinned_live_bytes":cleanup.pinned_host_live_bytes()});
  fs::write(output.join(format!("{key}.json")),serde_json::to_vec_pretty(&report)?)?;println!("BATCH_FRESH_MODEL {key} exact={passed} allocations_zero={zero}");models.push(report);
 }
 let passed=models.iter().all(|r|r["passed"]==true);fs::write(output.join("result.json"),serde_json::to_vec_pretty(&json!({"passed":passed,"models":models,"scope":"fixed native prefixes only; no benchmark or HTTP termination claim"}))?)?;assert!(passed,"batch exact token gate failed; reports retained");Ok(())
}
