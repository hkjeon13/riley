#![cfg(feature = "cuda")]
use riley_cuda::{CudaRuntime, FA3_COMPILED};
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{LlamaBatchMetadataConfig, PreparedLlamaBatchExecutor,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig};

#[test]
#[ignore = "requires optional FA3 build, real SmolLM2 checkpoint and non-Hopper GPU"]
fn model_recorder_rejects_non_hopper_and_releases_resources() -> Result<(),Box<dyn std::error::Error>> {
    assert!(FA3_COMPILED,"enable RILEY_FA3_SOURCE for this test");
    let runtime=CudaRuntime::initialize()?;let device=runtime.device(0)?;
    assert_ne!(device.properties().compute_capability(),(9,0),"this test exercises device rejection");
    let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),
        LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
    let context=device.create_context()?;let mut stream=context.create_stream()?;
    let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,64)?,PreparedLlamaForwardConfig::default());
    let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
    match executor.into_owned_variable_fa3_session(&context,32,true) {
        Ok(_)=>panic!("FA3 model recorder accepted non-Hopper"),
        Err(error)=>{
            let message=error.to_string();
            assert!(message.contains("FA3 cold preparation"),"unexpected rejection: {message}");
            println!("expected model rejection: {message}");
        }
    }
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero(),"model graph rejection leaked retained allocations");
    context.close()?;
    println!("FA3 model recorder non-Hopper rejection and allocation rollback passed; attention not executed");
    Ok(())
}
