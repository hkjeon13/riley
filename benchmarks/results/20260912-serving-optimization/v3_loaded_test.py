from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();needle='        hash.update(scratch.capacity.to_le_bytes());';at=s.index(needle);s=s[:at]+'''        let m=scratch.head.algorithm_metadata();
        let (major,minor)=m.compute_capability();hash.update(major.to_le_bytes());hash.update(minor.to_le_bytes());
        hash.update(m.runtime_version().to_le_bytes());hash.update(m.cublaslt_version().to_le_bytes());
        for v in [m.backend_id(),m.algorithm_id() as u32,m.tile_id(),m.stages_id(),m.split_k(),m.reduction_scheme(),m.cta_swizzling(),m.custom_option(),u32::from(m.deterministic())] {hash.update(v.to_le_bytes());}
        let (mr,n,k)=m.dimensions();for v in [mr,n,k,m.workspace_bytes(),m.numerical_implementation_flags()] {hash.update(v.to_le_bytes());}
'''+s[at:];p.write_text(s)
s=(r/'crates/riley-scheduler/tests/v3_variable_gpu.rs').read_text()
s=s.replace('use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};','use riley_cuda::CudaRuntime;\nuse riley_model::{LoadedModel,LoadLimits};\nuse riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig,variable_session::VariableGraphBuffers};')
s=s.replace('variable_scheduler_gpu_completion_and_settlement','loaded_variable_scheduler_gpu_completion_and_settlement')
a=s.index('let mut f=File::open(root.join("weights.bin"))');b=s.index('let mut f=File::open(root.join("requests.bin"))')
s=s[:a]+'''let model=LoadedModel::load(PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?).as_path(),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
'''+s[b:]
s=s.replace('let mut upload=context.allocate_pinned_host_buffer(1<<20)?;','')
a=s.index('let mut sizes=');b=s.index(' let mut scheduler=')
s=s[:a]+'''let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,64)?,PreparedLlamaForwardConfig::default());
 let mut executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
 assert_eq!(executor.maximum_position_count()?,8192);
 let mut scratch=VariableGraphBuffers::prepare(&context,1024)?;
 let mut session=executor.prepare_variable_session(&mut stream,&mut scratch)?;
'''+s[b:]
s=s.replace('session.close()?;head.close()?;drop(buffers);drop(staging);','session.close()?;scratch.close()?;executor.close()?;')
s=s.replace('drop(upload);','').replace('V3_SCHEDULER','V3_LOADED_SCHEDULER')
(r/'crates/riley-scheduler/tests/v3_loaded_gpu.rs').write_text(s)
