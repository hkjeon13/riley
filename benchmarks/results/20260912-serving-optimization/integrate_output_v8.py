from pathlib import Path
b=Path('/tmp/riley-output-integration-v8');changed=[]
def edit(path,fn):
 p=b/path;s=p.read_text();t=fn(s);assert t!=s,path;p.write_text(t);changed.append(path)
edit('kernels/src/gemm.cu',lambda s:s.replace('const uint64_t width=config->n/16','const uint64_t width=config->n').replace('plan->column_chunks=16','plan->column_chunks=1'))
edit('kernels/src/graph_resources.cu',lambda s:s.replace('CatalogEntry catalog[3]','CatalogEntry catalog[6]').replace('(*resources)->catalog[2].graph || (*resources)->catalog[2].exec)', '(*resources)->catalog[2].graph || (*resources)->catalog[2].exec ||\n      (*resources)->catalog[3].graph || (*resources)->catalog[3].exec ||\n      (*resources)->catalog[4].graph || (*resources)->catalog[4].exec ||\n      (*resources)->catalog[5].graph || (*resources)->catalog[5].exec)'))
edit('kernels/src/graph_multisequence_catalog.inc',lambda s:s.replace('(bucket!=2&&bucket!=4&&bucket!=8)||!r->exec','(bucket!=2&&bucket!=4&&bucket!=8)||full>1||!r->exec').replace('r->catalog[bucket==2?0:bucket==4?1:2]','r->catalog[(full?0:3)+(bucket==2?0:bucket==4?1:2)]').replace('index>3','index>6'))
edit('kernels/src/graph_multisequence_record.inc',lambda s:s.replace('i<16?d[i]->byte_len!=sizes[i]', 'i<16?(i==12?d[i]->byte_len!=1152+uint64_t(bucket)*98304:d[i]->byte_len!=sizes[i])'))
edit('crates/riley-runtime/src/llama/graph_decode_multi_parents.rs',lambda s:s.replace('        Ok(Self {','        for _ in 0..3 {\n            staging.push(context.allocate_pinned_host_buffer(2 * 1792).map_err(cuda)?);\n        }\n        Ok(Self {'))
def full(s):
 needle='''                    .map_err(cuda)?;
            }
        }
        Ok(graph)'''
 return s.replace(needle,'''                    .map_err(cuda)?;
                graph.append_multisequence_decode(
                    indices, &multi_weights, &std::array::from_fn(|i| index * 5 + i),
                    index + 4, [2, 4, 8][index], geometry[1] as u32, false,
                ).map_err(cuda)?;
            }
        }
        Ok(graph)''')
edit('crates/riley-runtime/src/llama/graph_decode_full.rs',full)
def session(s):
 s=s.replace('full-logits.v2','dual-output.v2').replace('        catalog.retain', '        let greedy: Vec<_> = catalog.iter().map(|e| CatalogEntry { mode: ResultMode::Greedy, ..*e }).collect();\n        catalog.extend(greedy);\n        catalog.retain')
 s=s.replace('wire::result_bytes(decoded.bucket, ResultMode::FullLogits)','wire::result_bytes(decoded.bucket, e.mode)')
 s=s.replace('                put32(&mut self.output, 224, 1152);','                if e.mode == ResultMode::FullLogits {\n                put32(&mut self.output, 224, 1152);').replace('self.output[1152..bytes].copy_from_slice(&self.inner.output[..98304]);','self.output[1152..bytes].copy_from_slice(&self.inner.output[..98304]);\n                }')
 s=s.replace('            self.input[..bytes].fill(0);','            let transfer = bytes.max(wire::REQUEST_BYTES);\n            self.input[..transfer].fill(0);').replace('            let index = if decoded.bucket', '            let mut index = if decoded.bucket').replace('            self.inner\n                .graph\n                .replay_catalog(index, &self.input[..bytes])','            if e.mode == ResultMode::Greedy { index += 3; }\n            self.inner\n                .graph\n                .replay_catalog(index, &self.input[..transfer])').replace('.read_catalog(index, &mut self.output[..bytes])','.read_catalog(index, &mut self.output[..transfer])')
 return s
edit('crates/riley-runtime/src/llama/graph_decode_multi_session.rs',session)
# Preserve default full-logit authority API for existing contract tests, add explicit mode helper.
edit('crates/riley-scheduler/src/authority.rs',lambda s:s.replace('''        use crate::descriptor::{
            BlockOwnership''','''        self.descriptor_expectation_with_mode(owner, replay, cookies, crate::descriptor::ResultMode::FullLogits)
    }
    pub(crate) fn descriptor_expectation_with_mode(
        &self, owner: crate::descriptor::OwnerExpectation, replay: u64, cookies: &[u64],
        mode: crate::descriptor::ResultMode,
    ) -> crate::descriptor::Result<crate::descriptor::SubmissionExpectation> {
        use crate::descriptor::{
            BlockOwnership''',1).replace('            mode: ResultMode::FullLogits,\n            rows,','            mode,\n            rows,',1))
edit('crates/riley-scheduler/src/execution.rs',lambda s:s.replace('authority.descriptor_expectation(owner, replay, &cookies)', 'authority.descriptor_expectation_with_mode(owner, replay, &cookies, if greedy_workspace.is_some() { crate::descriptor::ResultMode::Greedy } else { crate::descriptor::ResultMode::FullLogits })'))
import json,tarfile
Path('/tmp/output-v8-files.json').write_text(json.dumps(changed))
with tarfile.open('/tmp/output-v8.tar','w') as t:
 for p in changed:t.add(b/p,arcname=p)
print(changed)
