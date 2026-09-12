from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/llama/multi_descriptor/shape_progress.rs'
p.write_text('''//! Request progress checked before packing. This is not GPU admission or ownership proof.
use super::{check, overflow, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputStage { Prefill, Decode }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub prompt_tokens: u32,
    pub output_limit: u32,
    pub context_tokens: u32,
    pub committed_tokens: u32,
    pub input_tokens: u32,
    /// Index of the output that this step would publish, zero-based.
    pub generated_index: u32,
    pub stage: InputStage,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedProgress {
    pub target_tokens: u32,
    pub last_position: u32,
    pub live_pages: u32,
    pub last_page_tokens: u16,
    /// None for an unfinished prompt: no token/logits may be published yet.
    pub logits_input_row: Option<u32>,
}
impl Progress {
    pub fn validate(self) -> Result<ValidatedProgress> {
        check(self.prompt_tokens>0 && self.output_limit>0 && self.context_tokens>0 && self.context_tokens<=4096,
            "request_shape", "empty or unsupported request geometry")?;
        let last_target=self.prompt_tokens.checked_add(self.output_limit-1).ok_or_else(||overflow("request_shape"))?;
        check(last_target<=self.context_tokens,"request_shape","requested generation exceeds context")?;
        check(self.input_tokens>0 && self.input_tokens<=1024,"input_count","empty or unsupported input chunk")?;
        check(self.generated_index<self.output_limit,"generation_bound","output limit exhausted")?;
        let target=self.committed_tokens.checked_add(self.input_tokens).ok_or_else(||overflow("target_length"))?;
        check(target<=self.context_tokens,"target_length","context exceeded")?;
        let publish=match self.stage {
            InputStage::Prefill=>{
                check(self.generated_index==0 && self.committed_tokens<self.prompt_tokens && target<=self.prompt_tokens,
                    "prefill", "prefill must extend only the unfinished prompt")?;
                target==self.prompt_tokens
            },
            InputStage::Decode=>{
                let committed=self.prompt_tokens.checked_add(self.generated_index).and_then(|v|v.checked_sub(1)).ok_or_else(||overflow("decode"))?;
                check(self.generated_index>0 && self.input_tokens==1 && self.committed_tokens==committed,
                    "decode", "decode position does not match published output count")?;
                true
            }
        };
        Ok(ValidatedProgress { target_tokens:target,last_position:target-1,live_pages:target.div_ceil(16),
            last_page_tokens:((target-1)%16+1) as u16,logits_input_row:publish.then_some(self.input_tokens-1) })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunked_requests_publish_only_after_complete_prompt_then_exact_output_budget() {
        for prompt in [1,16,17,127,128,129,255,256,257,398,1024] {
            for limit in [1,32,64,128] {
                let mut committed=0;
                while committed<prompt {
                    let count=(prompt-committed).min(73);
                    let v=Progress{prompt_tokens:prompt,output_limit:limit,context_tokens:prompt+limit-1,
                        committed_tokens:committed,input_tokens:count,generated_index:0,stage:InputStage::Prefill}.validate().unwrap();
                    committed+=count;
                    assert_eq!(v.logits_input_row.is_some(),committed==prompt);
                    assert_eq!(v.live_pages,(committed+15)/16);
                    assert_eq!(v.last_page_tokens,((committed-1)%16+1) as u16);
                }
                for generated in 1..limit {
                    let x=Progress{prompt_tokens:prompt,output_limit:limit,context_tokens:prompt+limit-1,
                        committed_tokens:committed,input_tokens:1,generated_index:generated,stage:InputStage::Decode};
                    let v=x.validate().unwrap();assert_eq!(v.logits_input_row,Some(0));committed+=1;
                    assert!(Progress{generated_index:generated+1,..x}.validate().is_err());
                }
                assert_eq!(committed,prompt+limit-1);
            }
        }
    }
    #[test]
    fn rejects_cross_stage_overlap_overflow_and_context_exhaustion() {
        let x=Progress{prompt_tokens:129,output_limit:32,context_tokens:160,committed_tokens:128,input_tokens:1,generated_index:0,stage:InputStage::Prefill};
        assert_eq!(x.validate().unwrap().logits_input_row,Some(0));
        for bad in [Progress{input_tokens:2,..x},Progress{generated_index:1,..x},Progress{stage:InputStage::Decode,..x},
            Progress{context_tokens:159,..x},Progress{prompt_tokens:u32::MAX,..x},Progress{committed_tokens:u32::MAX,..x},Progress{input_tokens:0,..x}] {
            assert!(bad.validate().is_err(),"{bad:?}");
        }
    }
}
''')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/mod.rs';s=p.read_text();s=s.replace('use std::collections::BTreeSet;','pub mod shape_progress;\n\nuse std::collections::BTreeSet;')
start=s.index('        let target = row\n            .committed_length')
end=s.index('        let live = target.div_ceil(16) as usize;',start)
old=s[start:end]
new='''        // The v2 wire remains deliberately bounded; the logical request-progress
        // calculation is shared with the variable-shape path being integrated.
        let progress = shape_progress::Progress {
            prompt_tokens: 128, output_limit: row.max_output_tokens, context_tokens: 160,
            committed_tokens: row.committed_length, input_tokens: input_count as u32,
            generated_index: row.generated_index,
            stage: if e.stage==Stage::Prefill128 { shape_progress::InputStage::Prefill } else { shape_progress::InputStage::Decode },
        }.validate()?;
        let target=progress.target_tokens;
        check(row.target_length==target,"target_length","target differs from committed prefix plus inputs")?;
        let pos=progress.last_position;
'''
s=s[:start]+new+s[end:];p.write_text(s)
print('integrated request-progress validation; existing wire bounds preserved')
