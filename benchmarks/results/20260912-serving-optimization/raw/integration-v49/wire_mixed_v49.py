from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text().replace('pub packed_prefill:bool,','pub packed_prefill:bool,\n    pub mixed_execution:bool,')
s=s.replace('if e.packed_prefill{0x36524d52}', 'if e.mixed_execution{0x37524d52}else if e.packed_prefill{0x36524d52}')
pos=s.index('pub fn validate<const ROWS:usize>');s=s[:pos]+'''pub const MIXED_REQUEST_BYTES:usize=Layout::<32>::REQUEST_BYTES+4096;
pub fn request_bytes<const ROWS:usize>(e:&Expectation<ROWS>)->usize{if e.mixed_execution{MIXED_REQUEST_BYTES}else{Layout::<ROWS>::REQUEST_BYTES}}
fn stage_number(stage:InputStage)->u32{if stage==InputStage::Prefill{0}else{1}}
fn packet_stage<const ROWS:usize>(e:&Expectation<ROWS>)->u32{if e.mixed_execution && e.rows.iter().any(|r|r.progress.stage==InputStage::Prefill) && e.rows.iter().any(|r|r.progress.stage==InputStage::Decode){2}else{stage_number(e.stage)}}
'''+s[pos:]
s=s.replace('    check(!e.packed_prefill || ROWS==32', '    check(!e.mixed_execution || (e.packed_prefill&&ROWS==32),"capacity","mixed execution requires packed32 capability")?;\n    check(!e.packed_prefill || ROWS==32',1)
s=s.replace('check(e.stage!=InputStage::Prefill || e.rows.len()<=if e.packed_prefill{4}else{1},"stage","unsupported prefill owner count")?;', '''let prefills=e.rows.iter().filter(|r|r.progress.stage==InputStage::Prefill).count();
    check(e.stage==if prefills>0{InputStage::Prefill}else{InputStage::Decode},"stage","aggregate stage differs from rows")?;
    check(prefills<=if e.packed_prefill{4}else{1},"stage","unsupported prefill owner count")?;''')
s=s.replace('|| e.stage==InputStage::Decode).count()', '|| r.progress.stage==InputStage::Decode).count()')
s=s.replace('check(row.progress.stage==e.stage,', 'check(e.mixed_execution || row.progress.stage==e.stage,')
s=s.replace('packet.len()==Layout::<ROWS>::REQUEST_BYTES', 'packet.len()==request_bytes(e)')
s=s.replace('(0,if e.packed_prefill{0x36444d52}', '(0,if e.mixed_execution{0x37444d52}else if e.packed_prefill{0x36444d52}').replace('(4,if e.packed_prefill{6}', '(4,if e.mixed_execution{7}else if e.packed_prefill{6}')
s=s.replace('(8,Layout::<ROWS>::REQUEST_BYTES as u32)', '(8,request_bytes(e) as u32)').replace('(16,if e.stage==InputStage::Prefill{0}else{1})','(16,packet_stage(e))')
s=s.replace('let mut token_offset=0usize;', 'let mut token_offset=0usize;let mut tile_offset=0usize;')
s=s.replace('        if e.stage==InputStage::Prefill {\n            if e.packed_prefill', '''        if e.mixed_execution {u32_at(packet,b+72,stage_number(p.stage));}
        if e.stage==InputStage::Prefill {
            if e.mixed_execution {
                u32_at(packet,b+68,tile_offset as u32);
                let tiles=if p.input_tokens<32{p.input_tokens}else{p.input_tokens.div_ceil(8)};
                for local in 0..tiles {u32_at(packet,Layout::<32>::REQUEST_BYTES+tile_offset*4,((i as u32)<<16)|local);tile_offset+=1;}
            }
            if e.packed_prefill''')
s=s.replace('    if e.packed_prefill && e.stage==InputStage::Prefill {u32_at(packet,36,token_offset as u32);}', '    if e.packed_prefill && e.stage==InputStage::Prefill {u32_at(packet,36,token_offset as u32);}\n    if e.mixed_execution {u32_at(packet,96,tile_offset as u32);}')
s=s.replace('(76,if e.stage==InputStage::Prefill{0}else{1})', '(76,stage_number(p.stage))')
s=s.replace('packed_prefill:false}', 'packed_prefill:false,mixed_execution:false}')
p.write_text(s)
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('pub packed_prefill:bool,','pub packed_prefill:bool, pub mixed_execution:bool,',1).replace('context_tokens,packed_prefill:false}', 'context_tokens,packed_prefill:false,mixed_execution:false}').replace('if e.packed_prefill!=i.packed_prefill', 'if e.mixed_execution!=i.mixed_execution || e.packed_prefill!=i.packed_prefill');p.write_text(s)
p=r/'crates/riley-scheduler/src/authority.rs';s=p.read_text().replace('pub packed_prefill: bool,','pub packed_prefill: bool,\n    pub mixed_execution: bool,',1)
s=s.replace('|| (!prefill.is_empty() && !decode.is_empty())', '|| (!owner.mixed_execution && !prefill.is_empty() && !decode.is_empty())')
s=s.replace('input_tokens:to32(work.input_tokens().len())?, generated_index:to32(row.generated_index)?, stage,', 'input_tokens:to32(work.input_tokens().len())?, generated_index:to32(row.generated_index)?, stage:if work.kind()==crate::WorkKind::Prefill{InputStage::Prefill}else{InputStage::Decode},')
s=s.replace('Expectation { packed_prefill:owner.packed_prefill,', 'Expectation { mixed_execution:owner.mixed_execution, packed_prefill:owner.packed_prefill,')
s=s.replace('VariableOwnerGeometry{packed_prefill:', 'VariableOwnerGeometry{mixed_execution:false,packed_prefill:').replace('VariableOwnerGeometry { packed_prefill:', 'VariableOwnerGeometry { mixed_execution:false, packed_prefill:')
p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text().replace('VariableOwnerGeometry {packed_prefill:identity.packed_prefill,', 'VariableOwnerGeometry {mixed_execution:identity.mixed_execution,packed_prefill:identity.packed_prefill,');p.write_text(s)
p=r/'crates/riley-scheduler/src/config.rs';s=p.read_text().replace('    PackedPrefillDecode32,','    PackedPrefillDecode32,\n    /// Ready decode owners and up to four prefill owners share one token budget.\n    MixedPrefillDecode32,');p.write_text(s)
p=r/'crates/riley-scheduler/src/scheduler.rs';s=p.read_text().replace(' | ExecutionShapePolicy::PackedPrefillDecode32',' | ExecutionShapePolicy::PackedPrefillDecode32 | ExecutionShapePolicy::MixedPrefillDecode32')
marker='        if matches!(self.execution_shape_policy, ExecutionShapePolicy::CompletePrefill128DecodeN'
a=s.index(marker,s.index('prefill.sort_unstable_by_key'))
s=s[:a]+'''        if self.execution_shape_policy==ExecutionShapePolicy::MixedPrefillDecode32 {
            let mut budget=self.config.iteration_token_budget;
            for item in decode.iter().take(32) {
                if budget==0 {break;}
                selected.push(Candidate{request_id:item.request_id,kind:WorkKind::Decode,token_count:1});budget-=1;
            }
            for item in prefill.iter().take(4) {
                if budget==0 || selected.len()==32 {break;}
                let token_count=item.remaining_tokens.min(self.config.max_prefill_chunk_tokens).min(budget);
                selected.push(Candidate{request_id:item.request_id,kind:WorkKind::Prefill,token_count});budget-=token_count;
            }
            return Ok((selected,false));
        }
'''+s[a:];p.write_text(s)
print('V7 CPU wire, retained-capability rejection and mixed scheduler policy implemented')
