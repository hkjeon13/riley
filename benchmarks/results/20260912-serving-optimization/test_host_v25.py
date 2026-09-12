from pathlib import Path
import subprocess
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/sampling.rs';old=subprocess.check_output(['git','show','HEAD:crates/riley-runtime/src/sampling.rs'],cwd=r,text=True);a=old.index('    pub fn process_bf16_native(');b=old.index('    /// Processes logits and immediately',a);reference=old[a:b].replace('pub fn process_bf16_native','fn process_bf16_reference');s=p.read_text();s+='\n#[cfg(test)]\nimpl SamplingWorkspace {\n'+reference+'}\n';s+='''
#[cfg(test)]
mod greedy_fast_comparison {
 use super::*;
 #[test]
 fn greedy_fast_preserves_full_distribution_and_errors(){
  for size in [1,2,17,257,49152] {
   let mut fast=SamplingWorkspace::new(size).unwrap();let mut reference=SamplingWorkspace::new(size).unwrap();
   for seed in 0..8 {
    let mut bytes=Vec::new();for i in 0..size {let bits=match i%11 {0=>0u16,1=>0x8000,_=>((i*139+seed*73)%0x7f80)as u16 | if i%2==0{0x8000}else{0}};bytes.extend_from_slice(&bits.to_ne_bytes());}
    let mask:Vec<_>=(0..size).map(|i|seed!=7 && (i+seed)%3!=0).collect();
    let params=SamplingParams{temperature:0.0,..SamplingParams::default()};
    let history=[0,0];
    let a=fast.process_bf16_native(&bytes,TokenConstraints::AllowedMask(&mask),&history,params);
    let b=reference.process_bf16_reference(&bytes,TokenConstraints::AllowedMask(&mask),&history,params);
    match(a,b){(Ok(a),Ok(b))=>{assert_eq!(a.greedy_token,b.greedy_token);for (x,y) in a.processed_logits().iter().zip(b.processed_logits()){assert_eq!(x.to_bits(),y.to_bits());}assert_eq!(a.probabilities(),b.probabilities());assert_eq!(a.log_probabilities(),b.log_probabilities());},(Err(a),Err(b))=>assert_eq!(a,b),_=>panic!("fast/reference result differs")}
   }
  }
 }
}
''';p.write_text(s)
