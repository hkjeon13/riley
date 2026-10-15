#![cfg(feature="cuda")]
use std::{error::Error,fs,path::PathBuf};use serde_json::{Value,json};
use riley_cuda::{CudaRuntime,CudaContext,CudaStream,CudaDeviceBuffer,CudaPinnedHostBuffer,CudaBufferSpan,CudaDType,PackedBatchHostV1,PackedBatchV1,PreparedPagedDecodeAttention,PagedDecodeAttentionRequest,DecodeAttentionPreference,DecodeAttentionBackendAvailability,prepared_materialized_ragged_attention,prepare_materialized_ragged_attention_metadata};
type R<T=()> = Result<T,Box<dyn Error>>;
fn upload(cx:&CudaContext,s:&mut CudaStream,p:&mut CudaPinnedHostBuffer,b:&[u8])->R<CudaDeviceBuffer>{let mut x=cx.allocate_device_buffer(b.len() as u64)?;x.upload_from_slice(0,b,p,s)?;Ok(x)}
fn span(x:&CudaDeviceBuffer,t:CudaDType)->riley_cuda::CudaResult<CudaBufferSpan<'_>>{CudaBufferSpan::new(x,t,0,x.byte_len())}
#[test]#[ignore="two source-bound native GPU attention contexts"]
fn qwen_prepared_attention_reference()->R {
 let root=PathBuf::from(std::env::var("RILEY_ATTN_HF")?);let out=PathBuf::from(std::env::var("RILEY_ATTN_NATIVE")?);fs::create_dir(&out)?;let f:Value=serde_json::from_slice(&fs::read(root.join("fixture.json"))?)?;let rt=CudaRuntime::initialize()?;let cx=rt.device(0)?.create_context()?;let mut stream=cx.create_stream()?;let mut reports=Vec::new();
 for c in f["cases"].as_array().ok_or("cases")? {
  let read=|field:&str|->R<Vec<u8>>{Ok(fs::read(root.join(c[field]["file"].as_str().ok_or("file")?))?)};let tokens=c["tokens"].as_u64().ok_or("tokens")?;let blocks=c["blocks"].as_u64().ok_or("blocks")?;let qh=c["query_heads"].as_u64().ok_or("qh")?;let kvh=c["kv_heads"].as_u64().ok_or("kvh")?;assert_eq!(c["head_dim"],128);
  let offsets=[0_u32,blocks as u32];let ids=(0..blocks as u32).collect::<Vec<_>>();let valid=(0..blocks).map(|b|(tokens-b*16).min(16) as u16).collect::<Vec<_>>();let slots=[0_u32];let positions=[(tokens-1) as u32];let host=PackedBatchHostV1::new(&offsets,&ids,&valid,&slots,&positions,blocks)?;
  let mut staging=cx.allocate_pinned_host_buffer(4*1024*1024)?;let query=upload(&cx,&mut stream,&mut staging,&read("query")?)?;let kp=upload(&cx,&mut stream,&mut staging,&read("key_pool")?)?;let vp=upload(&cx,&mut stream,&mut staging,&read("value_pool")?)?;let offsetsd=upload(&cx,&mut stream,&mut staging,&offsets.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;let idsd=upload(&cx,&mut stream,&mut staging,&ids.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;let validd=upload(&cx,&mut stream,&mut staging,&valid.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;let slotsd=upload(&cx,&mut stream,&mut staging,&slots.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;let posd=upload(&cx,&mut stream,&mut staging,&positions.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;
  let batch=PackedBatchV1::new(host,span(&offsetsd,CudaDType::U32)?,span(&idsd,CudaDType::U32)?,span(&validd,CudaDType::U16)?,span(&slotsd,CudaDType::U32)?,span(&posd,CudaDType::U32)?)?;
  let reference=PreparedPagedDecodeAttention::select(&cx,PagedDecodeAttentionRequest::new(tokens,blocks,qh,kvh,128,1.0_f32/(128.0_f32).sqrt()),DecodeAttentionPreference::Reference,DecodeAttentionBackendAvailability::linked())?;
  let mut workspace=cx.allocate_device_buffer(reference.workspace_bytes())?;
  let mut valid_stage=cx.allocate_device_buffer(blocks*2)?;
  let mut output=cx.allocate_device_buffer(qh*128*2*2)?;
  let mut valid_host=vec![0u16;blocks as usize];let mut encoded_valid=vec![0u8;blocks as usize*2];let mut duplicate_scratch=vec![0u8;blocks as usize];let zero_host=vec![0u8;output.byte_len() as usize];
  prepare_materialized_ragged_attention_metadata(host,&reference,&mut output,&mut valid_stage,&mut staging,&mut valid_host,&mut encoded_valid,&zero_host,&mut stream)?;
  let mut transport_slab=cx.allocate_device_buffer(4)?;
  let before=cx.allocation_stats()?;
  let invalid=CudaBufferSpan::new(&query,CudaDType::BF16,0,2)?;
  assert!(prepared_materialized_ragged_attention(invalid,span(&kp,CudaDType::BF16)?,span(&vp,CudaDType::BF16)?,&mut output,batch,&reference,&mut workspace,&valid_stage,&mut valid_host,&mut duplicate_scratch,&mut stream).is_err());

  let mut command_batch=stream.begin_command_batch()?;
  {let mut commands=command_batch.commands();
   transport_slab.copy_from_pinned_in_command_batch(0,&staging,0,4,&mut commands)?;
   prepared_materialized_ragged_attention(span(&query,CudaDType::BF16)?,span(&kp,CudaDType::BF16)?,span(&vp,CudaDType::BF16)?,&mut output,batch,&reference,&mut workspace,&valid_stage,&mut valid_host,&mut duplicate_scratch,&mut commands)?;
   prepared_materialized_ragged_attention(span(&query,CudaDType::BF16)?,span(&kp,CudaDType::BF16)?,span(&vp,CudaDType::BF16)?,&mut output,batch,&reference,&mut workspace,&valid_stage,&mut valid_host,&mut duplicate_scratch,&mut commands)?;
  }
  command_batch.finish()?;

  let after=cx.allocation_stats()?;assert_eq!(before,after,"prepared attention changed owned GPU/pinned allocations");
  let expected=read("expected")?;let mut actual=vec![0;expected.len()];output.download_to_slice(0,&mut actual,&mut staging,&mut stream)?;let mut max=0.0_f32;let mut unequal=0;for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)){if a!=b{unequal+=1;}let v=|x:&[u8]|f32::from_bits(u32::from(u16::from_le_bytes([x[0],x[1]]))<<16);max=max.max((v(a)-v(b)).abs());}let name=c["name"].as_str().ok_or("name")?;fs::write(out.join(format!("{name}.bf16")),&actual)?;
  let mut inactive=vec![1u8;expected.len()];output.download_to_slice(expected.len() as u64,&mut inactive,&mut staging,&mut stream)?;assert!(inactive.iter().all(|b|*b==0),"inactive row must be zero");
  transport_slab.close()?;output.close()?;valid_stage.close()?;workspace.close()?;drop(reference);posd.close()?;slotsd.close()?;validd.close()?;idsd.close()?;offsetsd.close()?;vp.close()?;kp.close()?;query.close()?;staging.close()?;assert!(cx.allocation_stats()?.is_zero());reports.push(json!({"case":name,"tokens":tokens,"unequal_elements":unequal,"max_abs":max,"close_allocations_zero":true,"execute_allocation_counts_unchanged":true,"short_query_rejected":true,"inactive_row_zero":true,"command_batch_pending_copy_and_repeated_reference":true}));
 }
 stream.close()?;cx.close()?;fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"cases":reports,"scope":"cold-prepared reference with unchanged GPU/pinned allocation counts against captured HF contexts"}))?)?;assert!(reports.iter().all(|r|r["unequal_elements"]==0),"attention context differs; raw preserved");Ok(())
}
