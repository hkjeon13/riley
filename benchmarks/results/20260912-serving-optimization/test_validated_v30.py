from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();s+='''
#[cfg(test)]
mod lane_argmax_tests {
 use super::*;
 #[test]
 fn independent_lanes_match_scalar_and_reject_nonfinite(){
  for size in [1,7,8,9,16,257,49152] {for seed in 0..8 {
   let words:Vec<u16>=(0..size).map(|i|match (i+seed)%13 {0=>0,1=>0x8000,_=>((i*139+seed*73)%0x7f80)as u16 | if i%2==0{0x8000}else{0}}).collect();let mut bytes:Vec<_>=words.iter().flat_map(|x|x.to_le_bytes()).collect();
   let mut maximum=f32::NEG_INFINITY;let mut expected=0;
   for(i,&word)in words.iter().enumerate(){let value=f32::from_bits(u32::from(word)<<16);if value>maximum{maximum=value;expected=i as u32;}}
   assert_eq!(finite_argmax(&bytes).unwrap(),expected);
   for bad in [0x7f80u16,0xff80,0x7fc0,0xff81] {for at in [0,size/2,size-1]{let old=[bytes[at*2],bytes[at*2+1]];bytes[at*2..at*2+2].copy_from_slice(&bad.to_le_bytes());assert!(finite_argmax(&bytes).is_err());bytes[at*2..at*2+2].copy_from_slice(&old);}}
  }}
  for winner in 0..32 {let mut values=vec![0xbf80u16;32];values[winner]=0;values[(winner+8).min(31)]=0x8000;let bytes:Vec<_>=values.iter().flat_map(|x|x.to_le_bytes()).collect();assert_eq!(finite_argmax(&bytes).unwrap(),winner as u32);}
 }
}
''';p.write_text(s)
p=r/'crates/riley-server/src/engine.rs';s=p.read_text().replace('                    (token,Some(0.0))','                    request.state.sampling_rng().map_err(|source|internal(format!("sampling RNG failed: {source}")))?;\n                    (token,Some(0.0))',1);s+='''
#[cfg(test)]
mod validated_argmax_tests {
 use super::*;
 #[test]
 fn reuse_requires_eligible_validated_candidate(){
  let p=riley_runtime::sampling::SamplingParams{temperature:0.0,..Default::default()};
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[7],&[1,1],p),Some(3));
  assert_eq!(reuse_validated_argmax(None,8,8,&[],&[],p),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[3],&[],p),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,3,&[],&[],p),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[8],&[],p),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[],&[8],p),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[],&[],riley_runtime::sampling::SamplingParams{temperature:0.5,..p}),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[],&[],riley_runtime::sampling::SamplingParams{repetition_penalty:1.1,..p}),None);
  assert_eq!(reuse_validated_argmax(Some(3),8,8,&[],&[],riley_runtime::sampling::SamplingParams{top_k:Some(0),..p}),None);
 }
}
''';p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text();s+='''
#[cfg(test)]
mod validated_argmax_routing_tests {
 use super::*;
 #[test]
 fn validated_logits_keep_distinct_dense_argmax_and_full_rows(){
  let mut d=DownloadedLlamaIteration{iteration_id:IterationId::new(1),vocabulary_size:4,output_count:2,output:DownloadedLlamaOutput::ValidatedLogits{logits:vec![0;16],argmax:[3,1,0,0,0,0,0,0]},commit_outputs:vec![]};
  assert_eq!(d.validated_argmax_for_slot(OutputSlot::new(0)),Some(3));assert_eq!(d.validated_argmax_for_slot(OutputSlot::new(1)),Some(1));assert_eq!(d.validated_argmax_for_slot(OutputSlot::new(2)),None);
  assert_eq!(d.logits_for_slot(OutputSlot::new(1)).unwrap().len(),8);assert!(d.greedy_token_ids().is_empty());
  d.output=DownloadedLlamaOutput::Logits(vec![0;16]);assert_eq!(d.validated_argmax_for_slot(OutputSlot::new(0)),None);
 }
}
''';p.write_text(s)
