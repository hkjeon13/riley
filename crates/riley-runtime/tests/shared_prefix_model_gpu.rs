#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_runtime::llama::variable_session::OwnedVariableSession;
use riley_runtime::llama::multi_descriptor::{BlockOwnership,ResultMode,variable_wire::{Expectation,Row},shape_progress::{Progress,InputStage}};
use riley_runtime::paged_kv::{KvBlockPool,KvLayout,SequenceState,PrefixDescriptor,KvIdentity};
type TestResult<T> = Result<T,Box<dyn std::error::Error>>;

fn execute(session:&mut OwnedVariableSession<32>,pool:&mut KvBlockPool,sequences:&mut[SequenceState;2],iteration:u64,stage:InputStage)->TestResult<Vec<Vec<u8>>> {
    let prefill=stage==InputStage::Prefill;
    let target=if prefill{32}else{33};
    let committed=[sequences[0].logical_length(),sequences[1].logical_length()];
    let reservations=[sequences[0].reserve_to(pool,target)?,sequences[1].reserve_to(pool,target)?];
    let (identity,replay,cookies)=session.issue_rows(2)?;
    assert!(identity.shared_prefixes);
    let mut rows=Vec::new();let mut owners=Vec::new();
    for index in 0..2 {
        let sequence=&sequences[index];
        let table=sequence.execution_block_table(pool,Some(&reservations[index]))?;
        let tag=sequence.sequence_id().value();
        owners.extend(table.physical_block_ids().iter().map(|&physical_id|BlockOwnership{physical_id,sequence_tag:tag}));
        rows.push(Row{sequence_tag:tag,cookie:cookies[index],output_slot:index as u32,
            progress:Progress{prompt_tokens:32,output_limit:4,context_tokens:identity.context_tokens,
                committed_tokens:committed[index],input_tokens:if prefill{32}else{1},generated_index:if prefill{0}else{1},stage},
            input_tokens:if prefill{vec![17;32]}else{vec![7]},physical_ids:table.physical_block_ids().to_vec(),valid_tokens:table.valid_tokens().to_vec()});
    }
    let expectation=Expectation{owner_generation:identity.generation,last_accepted_replay:identity.last_accepted_replay,
        replay_id:replay,iteration_id:iteration,catalog_digest:identity.catalog_digest,
        physical_block_count:identity.physical_block_count,max_active_rows:32,stage,mode:ResultMode::FullLogits,
        rows,block_ownership:owners,packed_prefill:true,mixed_execution:true,shared_prefixes:true};
    let output=session.execute_rows(expectation)?.iter().map(|r|r.logits.to_vec()).collect();
    for (sequence,reservation) in sequences.iter_mut().zip(reservations) {sequence.commit(pool,reservation)?;}
    session.confirm_scheduler_commit(iteration)?;
    Ok(output)
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn shared_prefix_matches_exclusive_full_model_logits()->TestResult<()> {
    for buffered in [false,true] {
        let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
        let context=CudaRuntime::initialize()?.device(0)?.create_context()?;
        let mut stream=context.create_stream()?;
        let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,4,1,8)?,PreparedLlamaForwardConfig::default());
        let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
        let mut session=executor.into_owned_variable_prefix_session(&context,64,false,buffered,false,false)?;
        let layout=KvLayout::checked(30,8,3,64)?;
        let mut pool=KvBlockPool::new(layout)?;
        let mut sequences=[pool.create_sequence(64)?,pool.create_sequence(64)?];
        let prefill=execute(&mut session,&mut pool,&mut sequences,1,InputStage::Prefill)?;
        assert_eq!(prefill[0],prefill[1]);
        let exclusive=execute(&mut session,&mut pool,&mut sequences,2,InputStage::Decode)?;
        assert_eq!(exclusive[0].len(),49152*2);
        for sequence in &mut sequences {sequence.truncate_to(&mut pool,32)?;}
        // The source pages contain the actual model's completed prefix KV.
        let descriptor=PrefixDescriptor::new(KvIdentity{model_revision:[1;32],numerical_profile:[2;32],
            position_encoding:[3;32],partition:[4;32],layout:layout.into()},0,&[17;32])?;
        let mut export=pool.export_prefix(&sequences[0],descriptor.clone())?;
        sequences[1].reset(&mut pool)?;
        sequences[1].import_prefix(&mut pool,&export,&descriptor)?;
        pool.release_prefix(&mut export)?;
        assert_eq!(pool.stats().allocated_block_count(),2);
        let shared=execute(&mut session,&mut pool,&mut sequences,3,InputStage::Decode)?;
        assert_eq!(exclusive,shared,"shared KV changed full-vocabulary BF16 logits");
        session.close()?;
        for sequence in &mut sequences {sequence.close(&mut pool)?;}
        assert_eq!(pool.stats().allocated_block_count(),0);
        stream.close()?;assert!(context.allocation_stats()?.is_zero());context.close()?;
        println!("shared-prefix full-model buffered={buffered} exact_logits_bytes=196608 host_pages=0 cuda_allocations=0");
    }
    Ok(())
}
