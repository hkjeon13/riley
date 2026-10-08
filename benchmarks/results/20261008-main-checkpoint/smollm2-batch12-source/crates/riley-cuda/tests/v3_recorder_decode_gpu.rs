#![cfg(feature="cuda")]
use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};
use std::{fs::File,io::Read,path::PathBuf};
fn read32(f:&mut File)->std::io::Result<u32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(u32::from_le_bytes(b))}
fn read64(f:&mut File)->std::io::Result<u64>{let mut b=[0;8];f.read_exact(&mut b)?;Ok(u64::from_le_bytes(b))}
fn p32(p:&mut[u8],i:usize,v:u32){p[i..i+4].copy_from_slice(&v.to_le_bytes());}
fn p64(p:&mut[u8],i:usize,v:u64){p[i..i+8].copy_from_slice(&v.to_le_bytes());}
fn packet(tokens:&[u32],start:usize,count:usize,replay:u64,limit:u32)->Vec<u8>{let mut p=vec![0;17536];let target=(start+count)as u32;let live=target.div_ceil(16);
for(i,v)in[(0,0x33444d52),(4,3),(8,17536),(12,1664),(20,1),(24,8),(28,64),(128,tokens[start]),(132,target-1),(136,count as u32),(140,live),(144,start as u32),(148,target),(156,limit),(160,tokens.len()as u32),(164,1024),(172,if start+count==tokens.len(){count as u32-1}else{u32::MAX})]{p32(&mut p,i,v);}
for(i,v)in[(40,1),(48,replay),(56,replay),(176,1),(184,replay)]{p64(&mut p,i,v);}p[64]=1;
for i in 0..live{p32(&mut p,256+i as usize*4,(i*5+17)%64);let valid=if i+1<live{16u16}else{((target-1)%16+1)as u16};p[1280+i as usize*2..1282+i as usize*2].copy_from_slice(&valid.to_le_bytes());}
for (i,&t)in tokens[start..start+count].iter().enumerate(){p32(&mut p,13440+i*4,t);}p}
#[test]
#[ignore="requires CUDA and exported pinned checkpoint fixtures; actual recorder parity"]
fn checkpoint_variable_decode_matches_full_prefix_recompute()->Result<(),Box<dyn std::error::Error>>{
let root=PathBuf::from(std::env::var_os("RILEY_V3_MODEL_FIXTURE").ok_or("fixture directory missing")?);let mut f=File::open(root.join("weights.bin"))?;let mut weight_bytes=vec![];for _ in 0..273{let size=read64(&mut f)?;let mut b=vec![0;size as usize];f.read_exact(&mut b)?;weight_bytes.push(b);}
let rope=std::fs::read(root.join("rope.bin"))?;assert_eq!(rope.len(),1024*32*4*2);
let mut f=File::open(root.join("requests.bin"))?;let count=read32(&mut f)?;let mut requests=vec![];for _ in 0..count{let n=read32(&mut f)?;let mut tokens=vec![];for _ in 0..n{tokens.push(read32(&mut f)?);}requests.push(tokens);}
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut upload=context.allocate_pinned_host_buffer(1<<20)?;let mut replay_count=0;
for prompt in requests {let output_limit=match prompt.len(){16=>32,128=>64,_=>128};let mut baseline_outputs:Vec<Vec<u8>>=vec![];let mut baseline_tokens:Vec<u32>=vec![];let mut baseline_kv:Option<(Vec<u8>,Vec<u8>)>=None;
for phase in 0..3 {let tokens=&prompt;let chunk=if phase==1 {73}else{tokens.len()};let mut sizes=vec![1152*1024;5];sizes.extend([384*1024,384*1024,384*1024,3072*1024,3072*1024,2304*1024,3072*1024]);sizes.extend([30*64*16*384,30*64*16*384,1024*128,1024*128,17536,1152,128,4,98304,8]);let mut buffers=vec![];
for size in sizes{let mut b=context.allocate_device_buffer(size)?;b.upload_from_slice(0,&vec![0;size as usize],&mut upload,&mut stream)?;buffers.push(b);}
buffers[14].upload_from_slice(0,&rope[..1024*128],&mut upload,&mut stream)?;buffers[15].upload_from_slice(0,&rope[1024*128..],&mut upload,&mut stream)?;
let mut weights=vec![];for data in &weight_bytes{if data.is_empty(){weights.push(weights[0]);continue;}let mut b=context.allocate_device_buffer(data.len()as u64)?;b.upload_from_slice(0,data,&mut upload,&mut stream)?;weights.push(buffers.len());buffers.push(b);}
let mut head=context.prepare_gemm(CudaGemmConfig::new(1,49152,576,0)?)?;let mut staging=context.allocate_pinned_host_buffer(196864)?;
let mut owner=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream:&mut stream,devices:buffers.iter_mut().collect(),pinned:vec![&mut staging],plans:vec![&mut head]})?;owner.record_v3_prefill(&std::array::from_fn(|i|i),None,&weights,0,0,1024,64)?;let mut out=vec![0;98432];
for start in (0..tokens.len()).step_by(chunk){let n=chunk.min(tokens.len()-start);replay_count+=1;owner.replay_transfer(&packet(&tokens,start,n,replay_count,output_limit))?;owner.read_transfer(&mut out)?;assert_eq!(&out[..4],&0u32.to_le_bytes());assert_eq!(&out[4..8],&u32::from(start+n==tokens.len()).to_le_bytes());assert_eq!(&out[12..16],&0u32.to_le_bytes());assert_eq!(&out[24..32],&replay_count.to_le_bytes());assert_eq!(&out[56..60],&((start+n)as u32).to_le_bytes());assert_eq!(&out[60..64],&(tokens.len()as u32).to_le_bytes());assert_eq!(&out[124..128],&0x33524d52u32.to_le_bytes());if start+n<tokens.len(){assert!(out[128..].iter().all(|&x|x==0));}}
// Compare a full-prefill cached chain, a chunked-prefill cached chain, and
// independently recomputed complete prefixes on the same generated history.
if phase==0 {baseline_tokens.push(u32::from_le_bytes(out[8..12].try_into()?));baseline_outputs.push(out.clone());}
else {assert_eq!(&out[..16],&baseline_outputs[0][..16]);assert_eq!(&out[128..],&baseline_outputs[0][128..]);}
for generated in 1..output_limit as usize {
 let mut history=prompt.clone();history.extend_from_slice(&baseline_tokens[..generated]);replay_count+=1;
 let mut p=if phase==2 {packet(&history,0,history.len(),replay_count,output_limit)} else {
  let mut p=packet(&history,history.len()-1,1,replay_count,output_limit);p32(&mut p,16,1);p32(&mut p,152,generated as u32);p32(&mut p,160,prompt.len() as u32);p[13440..].fill(0);p
 };
 owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;
 assert_eq!(&out[..8],&[0,0,0,0,1,0,0,0]);assert_eq!(&out[12..16],&0u32.to_le_bytes());assert_eq!(&out[24..32],&replay_count.to_le_bytes());assert_eq!(&out[76..80],&p[16..20]);
 if phase==0 {baseline_tokens.push(u32::from_le_bytes(out[8..12].try_into()?));baseline_outputs.push(out.clone());}
 else {assert_eq!(&out[..16],&baseline_outputs[generated][..16],"generation argmax {generated}");assert_eq!(&out[128..],&baseline_outputs[generated][128..],"generation logits {generated}");}
 for bytes in out[128..].chunks_exact(2){let v=u16::from_le_bytes(bytes.try_into()?);assert_ne!(v&0x7f80,0x7f80);}
 // Replaying a logically exhausted decode is rejected before GPU work and
 // invalidates the previous read; the next valid generation can still execute.
 if phase!=2 {p32(&mut p,152,output_limit);assert!(owner.replay_transfer(&p).is_err());assert!(owner.read_transfer(&mut out).is_err());}
}
for bytes in out[128..].chunks_exact(2){let v=u16::from_le_bytes(bytes.try_into()?);assert_ne!(v&0x7f80,0x7f80,"nonfinite GPU logit");}
owner.close()?;let mut keys=vec![0;30*64*16*384];let mut values=keys.clone();buffers[12].download_to_slice(0,&mut keys,&mut upload,&mut stream)?;buffers[13].download_to_slice(0,&mut values,&mut upload,&mut stream)?;
if let Some((ek,ev))=&baseline_kv {assert_eq!(keys,*ek,"cached/recomputed keys");assert_eq!(values,*ev,"cached/recomputed values");}else{baseline_kv=Some((keys,values));}
eprintln!("V3_DECODE prompt={} phase={} generated={} logits_exact=true full_kv_exact=true",tokens.len(),phase,output_limit);

head.close()?;drop(buffers);drop(staging);
}
std::fs::write(root.join(format!("decode-tokens-{}.json",prompt.len())),format!("{:?}",baseline_tokens))?;
let mut file=File::create(root.join(format!("decode-logits-{}.bf16",prompt.len())))?;for result in &baseline_outputs {std::io::Write::write_all(&mut file,&result[128..])?;}
}
drop(upload);stream.close()?;assert!(context.allocation_stats()?.is_zero());eprintln!("V3_DECODE cases={} replays={} allocation_zero=true",count,replay_count);Ok(())}
