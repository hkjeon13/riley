from pathlib import Path
b=Path('/tmp/riley-shared-integration-v10');files=[]
def edit(p,f):
 q=b/p;s=q.read_text();t=f(s);assert s!=t,p;q.write_text(t);files.append(p)
edit('crates/riley-runtime/src/llama/graph_decode_multi_parents.rs',lambda s:s.replace('CudaContext, CudaPreparedStridedGemm, CudaStridedGemmConfig','CudaContext, CudaPreparedStridedGemm, CudaStridedGemmConfig, CudaPreparedGemm, CudaGemmConfig').replace('    pub(super) staging:', '    pub(super) shared: Vec<CudaPreparedGemm>,\n    pub(super) staging:').replace('prepare(context: &CudaContext)', 'prepare(context: &CudaContext, shared_rows: bool)').replace('        let mut staging = Vec::new();','        let mut staging = Vec::new();\n        let mut shared = Vec::new();').replace('            let b = u64::from(bucket);','''            let b = u64::from(bucket);
            if shared_rows {
                for n in [960, 3072, 49152] {
                    shared.push(context.prepare_gemm(CudaGemmConfig::new(b, n, 576, 0).map_err(cuda)?).map_err(cuda)?);
                }
            }''').replace('            scratch,','            scratch,\n            shared,'))
def full(s):
 s=s.replace('        let mut multi_indices = Vec::new();','        let mut multi_indices = Vec::new();\n        let mut shared_indices = Vec::new();')
 s=s.replace('        if let Some(multi) = multi {\n            for scratch', '        if let Some(multi) = multi {\n            let first_shared = plans.len();\n            shared_indices.extend((0..multi.shared.len()).map(|i| first_shared + i));\n            plans.extend(multi.shared.iter_mut());\n            for scratch')
 a=s.index('            for (index, indices) in multi_indices.iter().enumerate() {');z=s.index('\n        Ok(graph)',a)
 old=s[a:z]
 new='''            for (index, indices) in multi_indices.iter().enumerate() {
                let shared = !shared_indices.is_empty();
                let p = if shared { [shared_indices[index*3], shared_indices[index*3+1], index*5+2, index*5+3, shared_indices[index*3+2]] } else { std::array::from_fn(|i| index*5+i) };
                for full in [true, false] {
                    let staging = index + if full {1} else {4};
                    if shared { graph.append_shared_multisequence_decode(indices, &multi_weights, &p, staging, [2,4,8][index], geometry[1] as u32, full) }
                    else { graph.append_multisequence_decode(indices, &multi_weights, &p, staging, [2,4,8][index], geometry[1] as u32, full) }.map_err(cuda)?;
                }
            }
        }'''
 s=s[:a]+new+s[z:]
 s=s.replace('pub struct OwnedLlamaDecodeExecutor {','pub struct OwnedLlamaDecodeExecutor {\n    multi_plan_identity: Vec<u8>,')
 marker='''        mut self,
        context: &riley_cuda::CudaContext,
        catalog: bool,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {'''
 repl='''        self,
        context: &riley_cuda::CudaContext,
        catalog: bool,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {
        self.into_owned_decode_graph_with_catalog_profile(context, catalog, false)
    }
    pub(crate) fn into_owned_decode_graph_with_catalog_profile(
        mut self, context: &riley_cuda::CudaContext, catalog: bool, shared_rows: bool,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {'''
 assert marker in s;s=s.replace(marker,repl,1).replace('MultiDecodeParents::prepare(context)?','MultiDecodeParents::prepare(context, shared_rows)?')
 s=s.replace('        let vocabulary_size = self.vocabulary_size();\n        let maximum_position_count = self.maximum_position_count()?;\n        let config = self.config.metadata();', '''        let multi_plan_identity = format!("shared-rows-v1:{}:{:?}", shared_rows, multi.as_ref().map(|m| m.shared.iter().map(|p| p.algorithm_metadata()).collect::<Vec<_>>())).into_bytes();
        let vocabulary_size = self.vocabulary_size();
        let maximum_position_count = self.maximum_position_count()?;
        let config = self.config.metadata();''',1)
 s=s.replace('        Ok(OwnedLlamaDecodeExecutor {\n            graph,','        Ok(OwnedLlamaDecodeExecutor {\n            multi_plan_identity,\n            graph,')
 return s
edit('crates/riley-runtime/src/llama/graph_decode_full.rs',full)
def session(s):
 marker='''    ) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        let physical'''
 repl='''    ) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        self.into_owned_multi_decode_graph_profile(context, false)
    }
    /// Experimental arithmetic-changing QKV/gate-up/head profile; not model-quality qualified.
    pub fn into_owned_shared_multi_decode_graph(self, context: &riley_cuda::CudaContext) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        self.into_owned_multi_decode_graph_profile(context, true)
    }
    fn into_owned_multi_decode_graph_profile(self, context: &riley_cuda::CudaContext, shared_rows: bool) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        let physical'''
 assert marker in s;s=s.replace(marker,repl,1).replace('self.into_owned_decode_graph_with_catalog(context, true)?','self.into_owned_decode_graph_with_catalog_profile(context, true, shared_rows)?').replace('        hash.update(wire::CONTRACT_SHA256.as_bytes());','        hash.update(wire::CONTRACT_SHA256.as_bytes());\n        hash.update(&inner.multi_plan_identity);');return s
edit('crates/riley-runtime/src/llama/graph_decode_multi_session.rs',session)
import tarfile,json
with tarfile.open('/tmp/shared-v10-runtime.tar','w') as t:
 for p in files:t.add(b/p,arcname=p)
Path('/tmp/shared-v10-runtime-files.json').write_text(json.dumps(files))
