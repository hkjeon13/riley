from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-scheduler/src/config.rs';s=p.read_text().replace('    CompletePrefill128DecodeN,','''    CompletePrefill128DecodeN,
    /// One variable-length prefill chunk or up to eight independent decode rows.
    /// Alternates ready classes after dispatch; never mixes them in one plan.
    /// This is a host contract only; GPU support must be checked separately.
    VariablePrefillDecodeN,''');p.write_text(s)
p=r/'crates/riley-scheduler/src/scheduler.rs';s=p.read_text();s=s.replace('self.execution_shape_policy == ExecutionShapePolicy::CompletePrefill128DecodeN','matches!(self.execution_shape_policy, ExecutionShapePolicy::CompletePrefill128DecodeN | ExecutionShapePolicy::VariablePrefillDecodeN)')
s=s.replace('if item.remaining_tokens != 128 {','if self.execution_shape_policy == ExecutionShapePolicy::CompletePrefill128DecodeN && item.remaining_tokens != 128 {')
s=s.replace('                    token_count: 128,','                    token_count: item.remaining_tokens.min(self.config.max_prefill_chunk_tokens).min(self.config.iteration_token_budget),')
s=s.replace('// nor decode subtraction may split a prompt into 127 input tokens.','// nor decode subtraction changes the independently bounded prefill chunk.')
a='    if policy == ExecutionShapePolicy::CompletePrefill128DecodeN {'
b='''    if policy == ExecutionShapePolicy::VariablePrefillDecodeN
        && descriptor.prompt_token_ids.iter().any(|&token|token>=49_152) {
        return Err(SchedulerError::InvalidConfiguration {field:"prompt_token_ids",reason:"V3 vocabulary is 49152"});
    }
'''+a
assert a in s;s=s.replace(a,b)
a='    for (valid, field, reason) in [\n'
start=s.index('fn validate_execution_shape_config(');pos=s.index(a,start)
b='''    if policy == ExecutionShapePolicy::VariablePrefillDecodeN {
        for (valid,field,reason) in [
            (matches!(config.max_active_sequences,1|2|4|8),"max_active_sequences","V3 supports capacities 1, 2, 4 or 8"),
            (config.iteration_token_budget>=config.max_active_sequences && config.iteration_token_budget<=1024,"iteration_token_budget","V3 budget must cover decode capacity and fit 1024 tokens"),
            ((1..=1024).contains(&config.max_prefill_chunk_tokens),"max_prefill_chunk_tokens","V3 chunks must fit 1024 tokens"),
            (config.max_sequence_tokens<=4096,"max_sequence_tokens","V3 context limit is 4096"),
            (layout.layer_count()==30 && layout.key_value_head_count()==3 && layout.head_dimension()==64 && layout.physical_block_count()<=4096,"kv_layout","V3 requires Smol geometry and at most4096 physical pages"),
        ] {
            if !valid {return Err(SchedulerError::InvalidConfiguration{field,reason});}
        }
        return Ok(());
    }
'''
s=s[:pos]+b+s[pos:];p.write_text(s)
