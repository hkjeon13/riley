from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-cuda/tests/v3_recorder_checkpoint_gpu.rs';s=p.read_text()
s=s.replace('fn checkpoint_prefill_recorder_full_and_chunked','fn checkpoint_variable_decode_matches_full_prefix_recompute')
s=s.replace('for tokens in requests {let mut baseline:Option<(Vec<u8>,Vec<u8>,Vec<u8>)>=None;\nfor chunk in [tokens.len(),73]{','for prompt in requests {let mut baseline_outputs:Vec<Vec<u8>>=vec![];let mut baseline_tokens:Vec<u32>=vec![];let mut baseline_kv:Option<(Vec<u8>,Vec<u8>)>=None;\nfor phase in 0..3 {let tokens=&prompt;let chunk=if phase==1 {73}else{tokens.len()};')
needle='for bytes in out[128..].chunks_exact(2)'
at=s.index(needle)
s=s[:at]+'''// Compare a full-prefill cached chain, a chunked-prefill cached chain, and
// independently recomputed complete prefixes on the same generated history.
if phase==0 {baseline_tokens.push(u32::from_le_bytes(out[8..12].try_into()?));baseline_outputs.push(out.clone());}
else {assert_eq!(&out[..16],&baseline_outputs[0][..16]);assert_eq!(&out[128..],&baseline_outputs[0][128..]);}
for generated in 1..8 {
 let mut history=prompt.clone();history.extend_from_slice(&baseline_tokens[..generated]);replay_count+=1;
 let mut p=if phase==2 {packet(&history,0,history.len(),replay_count)} else {
  let mut p=packet(&history,history.len()-1,1,replay_count);p32(&mut p,16,1);p32(&mut p,152,generated as u32);p32(&mut p,160,prompt.len() as u32);p[13440..].fill(0);p
 };
 owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;
 assert_eq!(&out[..8],&[0,0,0,0,1,0,0,0]);assert_eq!(&out[12..16],&0u32.to_le_bytes());assert_eq!(&out[24..32],&replay_count.to_le_bytes());assert_eq!(&out[76..80],&p[16..20]);
 if phase==0 {baseline_tokens.push(u32::from_le_bytes(out[8..12].try_into()?));baseline_outputs.push(out.clone());}
 else {assert_eq!(&out[..16],&baseline_outputs[generated][..16],"generation argmax {generated}");assert_eq!(&out[128..],&baseline_outputs[generated][128..],"generation logits {generated}");}
 for bytes in out[128..].chunks_exact(2){let v=u16::from_le_bytes(bytes.try_into()?);assert_ne!(v&0x7f80,0x7f80);}
 // Replaying a logically exhausted decode is rejected before GPU work and
 // invalidates the previous read; the next valid generation can still execute.
 if phase!=2 {p32(&mut p,152,32);assert!(owner.replay_transfer(&p).is_err());assert!(owner.read_transfer(&mut out).is_err());}
}
'''+s[at:]
a=s.index('if let Some((expected,ek,ev))=&baseline');b=s.index('\nhead.close()?',a)
s=s[:a]+'''if let Some((ek,ev))=&baseline_kv {assert_eq!(keys,*ek,"cached/recomputed keys");assert_eq!(values,*ev,"cached/recomputed values");}else{baseline_kv=Some((keys,values));}
eprintln!("V3_DECODE prompt={} phase={} generated=8 logits_exact=true full_kv_exact=true",tokens.len(),phase);
'''+s[b:]
s=s.replace('V3_CHECKPOINT cases={} replays={} allocation_zero=true','V3_DECODE cases={} replays={} allocation_zero=true')
(r/'crates/riley-cuda/tests/v3_recorder_decode_gpu.rs').write_text(s)
