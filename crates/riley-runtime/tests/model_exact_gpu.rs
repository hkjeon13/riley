//! Finite comparison against fresh source-bound HF BF16 eager GPU rows.
#![cfg(feature = "cuda")]
use std::{convert::Infallible, error::Error, fs, path::PathBuf};
use riley_cuda::{CudaContext, CudaStream, CudaRuntime};
use riley_model::{LoadedModel, LoadLimits};
use riley_runtime::{generation::{GenerationRequest, GenerationState}, sampling::SamplingParams};
use riley_runtime::llama::{PreparedLlamaDecode, PreparedLlamaDecodeConfig, PreparedLlamaGeneration, LlamaGenerationEvent};
use serde_json::{Value, json};
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
fn ids(v: &Value) -> Result<Vec<u32>> { v.as_array().ok_or("token IDs not an array")?.iter().map(|v| Ok(u32::try_from(v.as_u64().ok_or("token ID not unsigned")?)?)).collect() }
fn config() -> PreparedLlamaDecodeConfig { PreparedLlamaDecodeConfig::default().with_paged_kv_cache().with_optimized_decode_attention() }
fn numeric(actual: &[u8], expected: &[u8]) -> Result<Value> {
 if actual.len()!=expected.len() || actual.len()%2!=0 { return Err("BF16 row dimensions differ".into()); }
 let mut dot=0.0_f64; let mut aa=0.0_f64; let mut bb=0.0_f64; let mut sum=0.0_f64; let mut max=0.0_f64; let mut bits=0;
 let mut top=0; let mut best=f32::NEG_INFINITY;
 for (i,(a,b)) in actual.chunks_exact(2).zip(expected.chunks_exact(2)).enumerate() {
  let x=f32::from_bits(u32::from(u16::from_le_bytes([a[0],a[1]]))<<16); let y=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16);
  if !x.is_finite() || !y.is_finite() {return Err("nonfinite BF16 logits".into());}
  if x>best {best=x;top=i;}
  if a!=b {bits+=1;}
  let x=f64::from(x);let y=f64::from(y);let e=(x-y).abs();sum+=e;max=max.max(e);dot+=x*y;aa+=x*x;bb+=y*y;
 }
 Ok(json!({"top1":top,"cosine":dot/(aa*bb).sqrt(),"mean_abs":sum/(actual.len()/2) as f64,"max_abs":max,"unequal_elements":bits,"threshold_verdict":"ungraded; exact token gate assessed separately"}))
}
fn teacher(model: &LoadedModel, context:&CudaContext, stream:&mut CudaStream, prompt:&[u32], expected:&[u32], case:&Value, root:&PathBuf, out:&PathBuf) -> Result<Value> {
 let mut owner=PreparedLlamaDecode::prepare(model,context,stream,prompt.len(),prompt.len()+expected.len(),config())?;
 let operation=(|| -> Result<Value> {
  owner.prefill(prompt,stream)?;
  let rows=case["logit_rows"].as_array().ok_or("logit rows missing")?;
  if rows.len()!=expected.len() {return Err("HF decode row count differs".into());}
  let mut metrics=Vec::new(); let mut bytes=vec![0;model.spec().embedding().vocabulary_size()*2];
  for (step,row) in rows.iter().enumerate() {
   owner.download_last_logits(&mut bytes,stream)?;
   let filename=row["file"].as_str().ok_or("row filename missing")?;
   if filename.contains('/') || filename.contains("..") {return Err("row path must be a direct leaf".into());}
   let hf=fs::read(root.join(filename))?;
   fs::write(out.join(format!("{}-step{step}-native.bf16",case["id"].as_str().ok_or("case ID missing")?)),&bytes)?;
   let mut m=numeric(&bytes,&hf)?;m["step"]=json!(step);m["stage"]=json!(if step==0 {"prefill"} else {"cache-on-decode"});m["hf_top1"]=row["top1"].clone();m["top1_exact"]=json!(m["top1"]==row["top1"]);metrics.push(m);
   if step+1<expected.len() {owner.decode(expected[step],stream)?;}
  }
  Ok(json!({"rows":metrics,"all_top1_exact":metrics.iter().all(|m|m["top1_exact"]==true)}))
 })();
 let close=owner.close();
 if let Err(e)=close {return Err(format!("decode close: {e}").into());}
 if !context.allocation_stats()?.is_zero() {return Err("teacher owner allocations remain after close".into());}
 operation
}
fn greedy(model:&LoadedModel,context:&CudaContext,stream:&mut CudaStream,prompt:Vec<u32>,fixture:&Value,expected:&[u32]) -> Result<Value> {
 let mut owner=PreparedLlamaGeneration::prepare(model,context,stream,prompt.len(),16,config())?;
 let operation=(|| -> Result<Value> {
  let request=GenerationRequest {request_id:b"fresh-hf-finite-probe".to_vec(),seed:0,prompt_token_ids:prompt,sampling_params:SamplingParams {temperature:0.0,top_k:None,top_p:None,repetition_penalty:1.0},min_new_tokens:0,max_new_tokens:16,eos_token_ids:ids(&fixture["eos_token_ids"])?,stop_token_ids:vec![],stop_strings:vec![]};
  let mut state=GenerationState::new(request,model.spec().embedding().vocabulary_size(),model.tokenizer().maximum_decoded_token_bytes())?;let mut observed=Vec::new();
  let summary=owner.generate(&mut state,stream,||false,|event|{if let LlamaGenerationEvent::Token{token,..}=event {observed.push(token.token_id());} Ok::<(),Infallible>(())})?;
  let actual=state.generated_token_ids();let first=(0..actual.len().max(expected.len())).find(|&i|actual.get(i)!=expected.get(i));
  Ok(json!({"expected_token_ids":expected,"actual_token_ids":actual,"callback_tokens":observed,"exact":actual==expected,"first_difference":first,"prefill_calls":summary.prefill_tokens(),"decode_calls":summary.decode_tokens(),"rng_draws":state.rng_draws()}))
 })();
 let close=owner.close();if let Err(e)=close {return Err(format!("generation close: {e}").into());}
 if !context.allocation_stats()?.is_zero() {return Err("generation allocations remain after close".into());} operation
}
#[test]
#[ignore="requires fresh HF fixtures and pinned single CUDA GPU"]
fn fresh_model_prefill_decode_and_own_greedy_match_hf() -> Result {
 let root=PathBuf::from(std::env::var("RILEY_FRESH_HF_ROOT")?);let output=PathBuf::from(std::env::var("RILEY_FRESH_NATIVE_OUTPUT")?);fs::create_dir(&output)?;
 let runtime=CudaRuntime::initialize()?;assert_eq!(runtime.device_count(),1);
 let context=runtime.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut reports=Vec::new();
 for key in ["smol","qwen25","qwen17","qwen4"] {
  let reference=root.join(key);let fixture:Value=serde_json::from_slice(&fs::read(reference.join("fixture.json"))?)?;let dir=output.join(key);fs::create_dir(&dir)?;
  let path=fixture["model_path"].as_str().ok_or("model path missing")?;let limit=fixture["weight_limit_bytes"].as_u64().ok_or("weight limit missing")?;
  let model=LoadedModel::load(&PathBuf::from(path),LoadLimits::default().with_weight_byte_limits(limit,limit)?)?;let mut cases=Vec::new();
  for case in fixture["cases"].as_array().ok_or("cases missing")? {
   let prompt=ids(&case["prompt_token_ids"])?;let expected=ids(&case["expected_token_ids"])?;
   let numerical=teacher(&model,&context,&mut stream,&prompt,&expected,case,&reference,&dir).unwrap_or_else(|e|json!({"error":e.to_string(),"all_top1_exact":false}));
   let generated=greedy(&model,&context,&mut stream,prompt,&fixture,&expected).unwrap_or_else(|e|json!({"error":e.to_string(),"exact":false}));
   cases.push(json!({"id":case["id"],"numerical":numerical,"generation":generated,"allocation_zero_after_case":context.allocation_stats()?.is_zero()}));
  }
  let passed=cases.iter().all(|c|c["generation"]["exact"]==true && c["numerical"]["all_top1_exact"]==true && c["allocation_zero_after_case"]==true);
  let report=json!({"model":fixture["model_id"],"revision":fixture["revision"],"passed":passed,"cases":cases});fs::write(dir.join("result.json"),serde_json::to_vec_pretty(&report)?)?;println!("FRESH_MODEL {key} exact={passed}");reports.push(report);
 }
 stream.close()?;let zero=context.allocation_stats()?.is_zero();context.close()?;
 let passed=zero && reports.iter().all(|r|r["passed"]==true);fs::write(output.join("result.json"),serde_json::to_vec_pretty(&json!({"passed":passed,"context_allocation_zero":zero,"models":reports}))?)?;
 assert!(passed,"fresh exact semantic gate failed; full rows and reports preserved");Ok(())
}
