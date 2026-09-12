from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();a=s.index('    let mut maximum=f32::NEG_INFINITY;let mut host_token=0;');b=s.index('    check(token==host_token',a);s=s[:a]+'    let host_token=finite_argmax(logits)?;\n'+s[b:];pos=s.index('pub fn validate_result<');s=s[:pos]+'''// Independent accumulators shorten the maximum dependency chain. All lanes
// reject nonfinite values; final ties still choose the lowest vocabulary ID.
fn finite_argmax(logits:&[u8])->Result<u32>{
 let mut maxima=[f32::NEG_INFINITY;8];let mut indices=[0u32;8];
 let mut chunks=logits.chunks_exact(16);
 for (block,chunk) in chunks.by_ref().enumerate(){
  let mut nonfinite=false;
  for lane in 0..8 {let bits=u16::from_le_bytes([chunk[2*lane],chunk[2*lane+1]]);nonfinite|=bits&0x7f80==0x7f80;let value=f32::from_bits(u32::from(bits)<<16);
   if value>maxima[lane]{maxima[lane]=value;indices[lane]=(block*8+lane)as u32;}}
  check(!nonfinite,"logits","nonfinite logit")?;
 }
 let mut maximum=f32::NEG_INFINITY;let mut token=0;
 for lane in 0..8 {if maxima[lane]>maximum || (maxima[lane]==maximum && indices[lane]<token){maximum=maxima[lane];token=indices[lane];}}
 let base=logits.len()/16*8;
 for (i,word) in chunks.remainder().chunks_exact(2).enumerate(){let value=f32::from_bits(u32::from(u16::from_le_bytes([word[0],word[1]]))<<16);check(value.is_finite(),"logits","nonfinite logit")?;if value>maximum{maximum=value;token=(base+i)as u32;}}
 Ok(token)
}
'''+s[pos:];p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text().replace('    Logits(Vec<u8>),','    Logits(Vec<u8>),\n    ValidatedLogits { logits:Vec<u8>, argmax:[u32;8] },').replace('DownloadedLlamaOutput::Logits(logits) => logits,','DownloadedLlamaOutput::Logits(logits) | DownloadedLlamaOutput::ValidatedLogits{logits,..} => logits,').replace('DownloadedLlamaOutput::Logits(_) => &[],','DownloadedLlamaOutput::Logits(_) | DownloadedLlamaOutput::ValidatedLogits{..} => &[],')
pos=s.index('    /// Returns one device-selected greedy token');s=s[:pos]+'''    /// Raw argmax independently checked against every finite logit by V3
    /// completion validation. Sampling constraints must still be applied.
    #[must_use]
    pub fn validated_argmax_for_slot(&self,slot:OutputSlot)->Option<u32>{
        let index=slot.get()as usize;if index>=self.output_count{return None;}
        match &self.output {DownloadedLlamaOutput::ValidatedLogits{argmax,..}=>argmax.get(index).copied(),_=>None}
    }

'''+s[pos:];a=s.index('pub fn execute_llama_iteration_variable_graph<');s=s[:a]+s[a:].replace('    let (identity,replay,cookies)=executor.issue_rows','    if prepared.output_count>8{return Err(fail(crate::descriptor::Error{field:"V3 output",reason:"too many output slots"},Some(ExecutionAbort::NotDispatched)));}\n    let mut argmax=[0u32;8];\n    let (identity,replay,cookies)=executor.issue_rows',1).replace('for row in rows {if row.token.is_some(){','for row in rows {if let Some(token)=row.token {argmax[row.output_slot as usize]=token;',1).replace('output:DownloadedLlamaOutput::Logits(logits),','output:DownloadedLlamaOutput::ValidatedLogits{logits,argmax},',1);p.write_text(s)
p=r/'crates/riley-server/src/engine.rs';s=p.read_text();needle='''                        Some(0.0),
                    )
                } else {
                    let logits = downloaded''';assert needle in s;s=s.replace(needle,'''                        Some(0.0),
                    )
                } else if let Some(token)=super::reuse_validated_argmax(
                    downloaded.validated_argmax_for_slot(slot),downloaded.vocabulary_size(),self.addressable_tokens,
                    request.state.masked_finish_token_ids(),request.state.history_token_ids(),request.state.request().sampling_params) {
                    (token,Some(0.0))
                } else {
                    let logits = downloaded''',1)
s+='''
#[cfg(any(feature="cuda",test))]
fn reuse_validated_argmax(candidate:Option<u32>,vocabulary:usize,addressable:usize,masked:&[u32],history:&[u32],params:riley_runtime::sampling::SamplingParams)->Option<u32>{
 let token=candidate?;
 if params.temperature!=0.0 || params.repetition_penalty!=1.0 || params.validate(vocabulary).is_err()
  || addressable>vocabulary || token as usize>=addressable
  || masked.iter().any(|&id|id as usize>=vocabulary || id==token)
  || history.iter().any(|&id|id as usize>=vocabulary){return None;}
 Some(token)
}
''';p.write_text(s)
