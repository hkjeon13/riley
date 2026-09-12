from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-scheduler/src/config.rs';s=p.read_text().replace('    VariablePrefillDecode32,','    VariablePrefillDecode32,\n    /// Up to four ready prefill owners share a token-packed V6 graph.\n    PackedPrefillDecode32,');p.write_text(s)
p=r/'crates/riley-scheduler/src/scheduler.rs';s=p.read_text().replace('ExecutionShapePolicy::VariablePrefillDecode32', 'ExecutionShapePolicy::VariablePrefillDecode32 | ExecutionShapePolicy::PackedPrefillDecode32')
s=s.replace('                let item = &prefill[0];','''                let limit=if self.execution_shape_policy==ExecutionShapePolicy::PackedPrefillDecode32{4}else{1};
                let mut budget=self.config.iteration_token_budget;
                for item in prefill.iter().take(limit) {
                if budget==0 {break;}''',1)
s=s.replace('token_count: item.remaining_tokens.min(self.config.max_prefill_chunk_tokens).min(self.config.iteration_token_budget),\n                });', '''token_count: item.remaining_tokens.min(self.config.max_prefill_chunk_tokens).min(budget),
                });
                budget-=selected.last().unwrap().token_count;
                }''',1)
s=s.replace('inflight.prefill_count == 1', 'inflight.prefill_count > 0');p.write_text(s)
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('pub physical_block_count:u32, pub context_tokens:u32,','pub physical_block_count:u32, pub context_tokens:u32, pub packed_prefill:bool,',1).replace('catalog_digest,physical_block_count,context_tokens},','catalog_digest,physical_block_count,context_tokens,packed_prefill:false},',1).replace('if e.packed_prefill || e.rows.is_empty()', 'if e.packed_prefill!=i.packed_prefill || e.rows.is_empty()');p.write_text(s)
p=r/'crates/riley-scheduler/src/authority.rs';s=p.read_text().replace('    pub context_tokens: u32,\n}', '    pub context_tokens: u32,\n    pub packed_prefill: bool,\n}',1)
s=s.replace('        for ((work,row),cookie) in prefill.iter()', '        let mut partial_slot=self.plan.output_slots().len() as u32;\n        for ((work,row),cookie) in prefill.iter()',1)
s=s.replace('let output_slot = work.output_slot().map(|s|s.get()).unwrap_or(0);','let output_slot = work.output_slot().map(|s|s.get()).unwrap_or_else(||{let slot=partial_slot;partial_slot+=1;slot});',1)
s=s.replace('packed_prefill:false, owner_generation:', 'packed_prefill:owner.packed_prefill, owner_generation:')
s=s.replace('super::VariableOwnerGeometry{generation:', 'super::VariableOwnerGeometry{packed_prefill:false,generation:').replace('super::VariableOwnerGeometry { generation:', 'super::VariableOwnerGeometry { packed_prefill:false, generation:')
# Reuse independently asserted lifecycle test with packed policy and expanded chunks.
a=s.index('    #[test]\n    fn variable_policy_batches_live_decode_and_retries_without_false_progress()')
b=s.index('    #[test]',a+15);t=s[a:b]
t=t.replace('variable_policy_batches_live_decode_and_retries_without_false_progress','packed_policy_batches_prefill_and_preserves_retry_partial_cancel')
t=t.replace('iteration_token_budget:73,max_prefill_chunk_tokens:73','iteration_token_budget:256,max_prefill_chunk_tokens:128').replace('ExecutionShapePolicy::VariablePrefillDecodeN','ExecutionShapePolicy::PackedPrefillDecode32')
t=t.replace('packed_prefill:false','packed_prefill:true')
t=t.replace('let mut last=None;', 'let mut last=None;let mut prefill_width=0;')
t=t.replace('if prefill {assert_eq!(plan.prefill_items().len(),1);assert!(plan.total_tokens()<=73);}', 'if prefill {prefill_width=prefill_width.max(plan.prefill_items().len());assert!(plan.prefill_items().len()<=4);assert!(plan.total_tokens()<=256);}')
t=t.replace('authority.variable_descriptor_expectation(&owner','authority.variable_descriptor_expectation_rows::<32>(&owner')
t=t.replace('variable_wire::REQUEST_BYTES','variable_wire::Layout::<32>::REQUEST_BYTES')
t=t.replace('assert!(partial && retried && widest>=3);','assert!(partial && retried && widest>=3 && prefill_width>=3);')
# Partial flag should include mixed output/partial iterations, not only all-partial.
t=t.replace('partial |= prefill && plan.output_slots().is_empty();','partial |= prefill && plan.output_slots().len()<plan.prefill_items().len();')
s=s[:b]+t+s[b:];p.write_text(s)
p=r/'crates/riley-scheduler/src/execution.rs';s=p.read_text().replace('VariableOwnerGeometry {generation:identity.generation', 'VariableOwnerGeometry {packed_prefill:identity.packed_prefill,generation:identity.generation');p.write_text(s)
print('Packed scheduler policy, partial slots and retained capability boundary implemented')
