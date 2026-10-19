#![cfg(feature = "cuda-cublas-gemm-probe")]

use std::error::Error;
use riley_cuda::{CudaBufferSpan,CudaBufferSpanMut,CudaContext,CudaDType,CudaDeviceBuffer,CudaGemmConfig,CudaPinnedHostBuffer,CudaRuntime,CudaStream};
type Result<T=()> = std::result::Result<T,Box<dyn Error>>;

fn upload(c:&CudaContext,s:&mut CudaStream,h:&mut CudaPinnedHostBuffer,b:&[u8])->Result<CudaDeviceBuffer>{
 let mut d=c.allocate_device_buffer(b.len() as u64)?;d.upload_from_slice(0,b,h,s)?;Ok(d)
}
fn bf16(x:f32)->u16{let b=x.to_bits();((b+0x7fff+((b>>16)&1))>>16) as u16}

// Zero Q/K gives a uniform distribution over each causal prefix. Probability
// rounds to BF16 before AV, exactly as the independent HF eager contract.
// V is +/-1: every F32 partial sum is exact (<=9216 terms, <=8+14 bits),
// so the oracle does not depend on the GPU GEMM's reduction order.
#[test]
#[ignore="remote GPU; fixed six shapes, two executions each; up to 6GiB scratch"]
fn owned_inplace_scores_match_exact_causal_bf16_oracle_and_reuse_allocations()->Result{
 run_attention_oracle(false)
}

#[test]
#[ignore="remote GPU; command-batch attention, six shapes, up to 6GiB scratch"]
fn batched_owned_attention_matches_exact_causal_bf16_oracle()->Result{
 run_attention_oracle(true)
}

fn run_attention_oracle(batched:bool)->Result{
 let rt=CudaRuntime::initialize()?;assert!(rt.device_count()>0);let c=rt.device(0)?.create_context()?;let mut stream=c.create_stream()?;assert!(c.allocation_stats()?.is_zero());
 let shapes=[(1,149,16,2,128),(129,257,9,3,64),(513,1025,16,8,128),(128,2049,32,8,128),(129,9216,32,8,128),(8192,8704,32,8,128)];
 for (m,t,h,k,d) in shapes {
  let (m,t,h,k,d)=(m as usize,t as usize,h as usize,k as usize,d as usize);let pages=t.div_ceil(16);let physical=pages+3;let qbytes=m*h*d*2;let pool=physical*k*16*d*2;let score=h*m*t*2;let tiled=h*m.min(128)*t*4;
  let mut staging=c.allocate_pinned_host_buffer(4*1024*1024)?;let query=upload(&c,&mut stream,&mut staging,&vec![0;qbytes])?;let key=upload(&c,&mut stream,&mut staging,&vec![0;pool])?;let mut host_value=vec![0xa5;pool];let ids:Vec<u32>=(0..pages).map(|p|(physical-1-p) as u32).collect();
  for token in 0..t {for head in 0..k {for depth in 0..d {
   let sign=if (token%4<3)^((head+depth)%2==1) {0x3f80u16}else{0xbf80u16};let ix=(((ids[token/16] as usize*k+head)*16+token%16)*d+depth)*2;host_value[ix..ix+2].copy_from_slice(&sign.to_le_bytes());
  }}}
  let value=upload(&c,&mut stream,&mut staging,&host_value)?;drop(host_value);let id_bytes:Vec<u8>=ids.iter().flat_map(|id|id.to_le_bytes()).collect();let table=upload(&c,&mut stream,&mut staging,&id_bytes)?;let mut output=upload(&c,&mut stream,&mut staging,&vec![0xa5;qbytes])?;
  let lengths=[qbytes,t*h*d*2,t*h*d*2,score,tiled,tiled,qbytes,16*1024*1024];let mut buffers=Vec::new();for len in lengths {buffers.push(c.allocate_device_buffer(len as u64)?);}let mut scratch:[CudaDeviceBuffer;8]=buffers.try_into().map_err(|_|"eight buffers required")?;
  let cfg=CudaGemmConfig::new(m as u64,t as u64,d as u64,16*1024*1024)?;let mut plan=if batched {c.prepare_owned_hf_gemm(cfg)?} else {c.prepare_cublas_gemm_probe(cfg)?};let before=c.allocation_stats()?;
  for pass in 0..2 {
   if batched {
    let mut batch=stream.begin_command_batch()?;
    {let mut commands=batch.commands();
     plan.execute_hf_attention(CudaBufferSpan::new(&query,CudaDType::BF16,0,qbytes as u64)?,CudaBufferSpan::new(&key,CudaDType::BF16,0,pool as u64)?,CudaBufferSpan::new(&value,CudaDType::BF16,0,pool as u64)?,CudaBufferSpan::new(&table,CudaDType::U32,0,id_bytes.len() as u64)?,CudaBufferSpanMut::new(&mut output,CudaDType::BF16,0,qbytes as u64)?,&mut scratch,m as u64,t as u64,physical as u64,h as u64,k as u64,d as u64,&mut commands)?;
    }
    batch.finish()?;
   } else {
    plan.execute_hf_attention(CudaBufferSpan::new(&query,CudaDType::BF16,0,qbytes as u64)?,CudaBufferSpan::new(&key,CudaDType::BF16,0,pool as u64)?,CudaBufferSpan::new(&value,CudaDType::BF16,0,pool as u64)?,CudaBufferSpan::new(&table,CudaDType::U32,0,id_bytes.len() as u64)?,CudaBufferSpanMut::new(&mut output,CudaDType::BF16,0,qbytes as u64)?,&mut scratch,m as u64,t as u64,physical as u64,h as u64,k as u64,d as u64,&mut stream)?;
   }
   let mut actual=vec![0;qbytes];output.download_to_slice(0,&mut actual,&mut staging,&mut stream)?;
   for row in 0..m {let visible=t-m+row+1;let probability=f32::from_bits(u32::from(bf16(1.0/(visible as f32)))<<16);let sum=(visible/4*2+visible%4) as f32;
    for head in 0..h {for depth in 0..d {let sign=if (head/(h/k)+depth)%2==0 {1.0}else{-1.0};let expected=bf16(probability*sum*sign);let ix=((row*h+head)*d+depth)*2;let got=u16::from_le_bytes([actual[ix],actual[ix+1]]);assert_eq!(got,expected,"M{m} T{t} H{h} pass{pass} row{row} head{head} depth{depth}");}}
   }
   assert_eq!(c.allocation_stats()?,before,"attention must not allocate in execution");
  }
  eprintln!("inplace-attention exact M={m} T={t} H={h} score_elements={} saved_bytes={score}",m*t*h);
  plan.close()?;for b in scratch {b.close()?;}output.close()?;table.close()?;value.close()?;key.close()?;query.close()?;staging.close()?;assert!(c.allocation_stats()?.is_zero());
 }
 stream.close()?;c.close()?;Ok(())
}
