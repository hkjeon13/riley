from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-scheduler/src/authority.rs';s=p.read_text().replace('    pub(crate) committed_length: usize,','    pub(crate) committed_length: usize,\n    pub(crate) prompt_tokens: usize,');s=s.replace('    pub fn committed_length(&self) -> usize {','    pub fn prompt_tokens(&self) -> usize { self.prompt_tokens }\n    pub fn committed_length(&self) -> usize {')
needle='''            rows.push(ReservationExpectation {'''
assert needle in s;s=s.replace(needle,'''            let progress=crate::descriptor::shape_progress::Progress {
                prompt_tokens:to32(row.prompt_tokens)?, output_limit:to32(row.max_output_tokens)?, context_tokens:160,
                committed_tokens:to32(row.committed_length)?, input_tokens:to32(work.input_tokens().len())?, generated_index:to32(row.generated_index)?,
                stage:if stage==Stage::Prefill128 { crate::descriptor::shape_progress::InputStage::Prefill } else { crate::descriptor::shape_progress::InputStage::Decode },
            }.validate()?;
            // v2 has no prompt-length field: keep it restricted until the new wire
            // and native executor carry this scheduler-owned value end to end.
            if row.prompt_tokens!=128 || progress.target_tokens!=to32(work.target_logical_length())? || progress.logits_input_row.is_none() { return Err(bad()); }
            rows.push(ReservationExpectation {''');p.write_text(s)
p=r/'crates/riley-scheduler/src/scheduler.rs';s=p.read_text();needle='''                committed_length: sequence.logical_length() as usize,''';assert s.count(needle)==1;s=s.replace(needle,needle+'\n                prompt_tokens: record.descriptor.prompt_token_ids.len(),');p.write_text(s)
