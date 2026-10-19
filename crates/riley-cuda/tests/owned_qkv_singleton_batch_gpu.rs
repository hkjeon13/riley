//! Before-change M=1 bias arithmetic versus qualified singleton batch layouts.
#![cfg(feature="cuda-cublas-gemm-probe")]
use riley_cuda::{CudaRuntime,CudaGemmConfig,CudaGemmReductionPolicy,CudaBufferSpan,CudaBufferSpanMut,CudaDType,BiasGemmParams,begin_step_capture,end_step_capture,launch_step_graph};
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{fs,path::PathBuf};
#[test]
#[ignore="Requires an admitted SM89/cuBLAS120804 GPU and pinned Qwen QKV activation fixture"]
fn singleton_qkv_batches_preserve_bias_rounding_and_replay_inputs()->Result<(),Box<dyn std::error::Error>>{
 let root=PathBuf::from(std::env::var("RILEY_COMPONENT_ORACLE_ROOT")?);let fixture:Value=serde_json::from_slice(&fs::read(root.join("fixture.json"))?)?;
 let read=|c:&Value,key:&str|->Result<Vec<u8>,Box<dyn std::error::Error>>{
  let name=c[key].as_str().ok_or("missing filename")?;assert!(!name.contains("..") && name.bytes().all(|b|b.is_ascii_alphanumeric()||b"-_.".contains(&b)));
  let bytes=fs::read(root.join(name))?;let sha=Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>();assert_eq!(Some(sha.as_str()),fixture["files"][name]["sha256"].as_str());assert_eq!(Some(bytes.len()as u64),fixture["files"][name]["bytes"].as_u64());Ok(bytes)
 };
 let rt=CudaRuntime::initialize()?;assert_eq!(rt.device_count(),1);let c=rt.device(0)?.create_context()?;let mut s=c.create_stream()?;let mut host=c.allocate_pinned_host_buffer(64<<20)?;let mut cases=0;
 for case in fixture["cases"].as_array().ok_or("cases")?{
  if case["m"].as_u64()!=Some(1)||case.get("bias").is_none()||!case["id"].as_str().ok_or("id")?.starts_with("qwen2p5-3b-decode-"){continue;}
  let n=case["n"].as_u64().ok_or("n")?;let k=case["k"].as_u64().ok_or("k")?;let seed=read(case,"input")?;let w=read(case,"weight")?;let b=read(case,"bias")?;
  let mut x=c.allocate_device_buffer(32*k*2)?;let mut weight=c.allocate_device_buffer(w.len()as u64)?;let mut bias=c.allocate_device_buffer(b.len()as u64)?;let mut before=c.allocate_device_buffer(32*n*2)?;let mut after=c.allocate_device_buffer(32*n*2)?;
  weight.upload_from_slice(0,&w,&mut host,&mut s)?;bias.upload_from_slice(0,&b,&mut host,&mut s)?;
  let config=|m|CudaGemmConfig::new(m,n,k,1<<20).map(|v|v.with_reduction_policy(CudaGemmReductionPolicy::HfCompatibleBiasSplitKV1));
  let mut one=c.prepare_hf_compatible_bias_epilogue_probe(config(1)?,CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2)?)?;assert_eq!(one.algorithm_metadata().workspace_bytes(),0);
  assert!(c.prepare_hf_bias_singleton_batch(config(33)?,CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2)?).is_err());
  let mut input=Vec::new();for row in 0..32{let mut v=seed.clone();v.rotate_left(row*2%seed.len());input.extend(v);}
  for count in 2..=32u64{
   x.upload_from_slice(0,&input,&mut host,&mut s)?;
   for row in 0..count{one.execute(&mut BiasGemmParams{input:CudaBufferSpan::new(&x,CudaDType::BF16,row*k*2,k*2)?,weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len())?,bias:CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2)?,output:CudaBufferSpanMut::new(&mut before,CudaDType::BF16,row*n*2,n*2)?,workspace:None},&mut s)?;}
   let mut expected=vec![0;(count*n*2)as usize];before.download_to_slice(0,&mut expected,&mut host,&mut s)?;
   let mut plan=c.prepare_hf_bias_singleton_batch(config(count)?,CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2)?)?;
   macro_rules! make {($out:expr)=>{BiasGemmParams{input:CudaBufferSpan::new(&x,CudaDType::BF16,0,count*k*2).unwrap(),weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len()).unwrap(),bias:CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2).unwrap(),output:CudaBufferSpanMut::new($out,CudaDType::BF16,0,count*n*2).unwrap(),workspace:None}};}
   plan.execute(&mut make!(&mut after),&mut s)?;
   let graph={let mut batch=s.begin_command_batch()?;begin_step_capture(&mut batch.commands())?;plan.execute(&mut make!(&mut after),&mut batch.commands())?;let g=end_step_capture(&mut batch.commands())?;launch_step_graph(&mut batch.commands(),&g)?;batch.finish()?;g};
   let mut actual=vec![0;expected.len()];after.download_to_slice(0,&mut actual,&mut host,&mut s)?;assert_eq!(actual,expected,"QKV batch C{count} {}",case["id"]);
   let zero=vec![0;input.len()];x.upload_from_slice(0,&zero,&mut host,&mut s)?;
   for row in 0..count{one.execute(&mut BiasGemmParams{input:CudaBufferSpan::new(&x,CudaDType::BF16,row*k*2,k*2)?,weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len())?,bias:CudaBufferSpan::new(&bias,CudaDType::BF16,0,n*2)?,output:CudaBufferSpanMut::new(&mut before,CudaDType::BF16,row*n*2,n*2)?,workspace:None},&mut s)?;}
   before.download_to_slice(0,&mut expected,&mut host,&mut s)?;let stable=c.allocation_stats()?;
   {let mut batch=s.begin_command_batch()?;launch_step_graph(&mut batch.commands(),&graph)?;batch.finish()?;}
   after.download_to_slice(0,&mut actual,&mut host,&mut s)?;assert_eq!(actual,expected,"replayed QKV ignored changed input");assert_eq!(c.allocation_stats()?,stable);graph.close()?;plan.close()?;cases+=1;
  }
  one.close()?;after.close()?;before.close()?;bias.close()?;weight.close()?;x.close()?;
 }
 assert_eq!(cases,93);println!("QKV_SINGLETON_BATCH_CONDITIONS {cases}");host.close()?;s.close()?;assert!(c.allocation_stats()?.is_zero());c.close()?;Ok(())
}
