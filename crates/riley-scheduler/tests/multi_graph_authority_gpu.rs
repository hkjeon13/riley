//! Actual model, live scheduler authority, common owner and commit integration.
#![cfg(feature = "cuda")]
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::*;
use riley_runtime::paged_kv::{BLOCK_TABLE_V1_VERSION, KvLayout};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_scheduler::*;
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
fn prepare(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
) -> Result<PreparedLlamaBatchExecutor> {
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 128, 10, 1, 40)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_grouped_ragged_attention_heads()
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_packed_async_metadata()
    .with_vllm_smol_p128_graph();
    Ok(PreparedLlamaBatchExecutor::prepare(
        model, context, stream, config,
    )?)
}
#[test]
#[ignore = "requires qualified CUDA runtime and real SmolLM2 checkpoint"]
fn scheduler_authorized_multi_graph_matches_m1_and_commits() -> Result {
    let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    let model = LoadedModel::load(std::path::Path::new(&path), LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut owner =
        prepare(&model, &context, &mut stream)?.into_owned_multi_decode_graph(&context)?;
    let mut oracle = prepare(&model, &context, &mut stream)?.into_owned_decode_graph(&context)?;
    let mut scheduler = Scheduler::new_with_execution_shape(
        SchedulerConfig {
            max_waiting_requests: 8,
            max_waiting_prompt_tokens: 1024,
            max_active_sequences: 4,
            max_sequence_tokens: 160,
            iteration_token_budget: 128,
            max_prefill_chunk_tokens: 128,
            aging_threshold_ns: 1,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: None,
            max_promised_kv_blocks: 40,
            metrics_window_samples: 16,
        },
        KvLayout::checked(30, 40, 3, 64)?,
        ExecutionShapePolicy::CompletePrefill128DecodeN,
    )?;
    for r in 0..4u32 {
        let prompt = (0..128).map(|i| (i * 311 + r * 977 + 13) % 49152).collect();
        scheduler.submit(RequestDescriptor::new(prompt, 32 - r as usize * 3), 0)?;
    }
    let mut rows = 0;
    let mut shapes = std::collections::BTreeSet::new();
    let mut finished = 0;
    for now in 1..200 {
        let Some(plan) = scheduler.plan_iteration(now)?.into_parts().0 else {
            break;
        };
        let authority = scheduler.authorize_execution(&plan)?;
        let downloaded = execute_llama_iteration_multi_graph(&authority, &mut owner, None)
            .map_err(|e| format!("execution failed: {e:?}"))?;
        let mut samples = vec![SampledIterationToken::new(0, false); downloaded.output_count()];
        for work in plan.prefill_items().iter().chain(plan.decode_items()) {
            let table = &plan.block_tables()[work.block_table_index()];
            let row = LlamaBatchRow::new(
                work.request_id().get(),
                if work.input_tokens().len() == 128 {
                    LlamaBatchRowKind::Prefill
                } else {
                    LlamaBatchRowKind::Decode
                },
                work.input_tokens(),
                work.target_logical_length() as u32,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    table.physical_block_ids(),
                    table.valid_tokens(),
                    table.logical_length(),
                ),
                Some(0),
            );
            let expected = oracle.execute(&[row])?;
            let slot = work.output_slot().unwrap().get() as usize;
            assert_eq!(
                &downloaded.logits_bf16_native()[slot * 98304..(slot + 1) * 98304],
                expected,
                "iteration {now}, request {}",
                work.request_id().get()
            );
            samples[slot] = SampledIterationToken::new(oracle.greedy_token()?, false);
            rows += 1;
        }
        if plan.prefill_items().is_empty() {
            shapes.insert(plan.batch_size());
        }
        drop(authority);
        let id = downloaded.iteration_id().get();
        let result = downloaded
            .into_result(&samples, IterationTiming::new(0, 0))
            .map_err(|e| format!("result failed: {e:?}"))?;
        let settled = scheduler.complete_iteration(&result, now)?;
        assert!(settled.settlement_failures().is_empty());
        finished += settled.completions().len();
        owner.confirm_scheduler_commit(id)?;
    }
    assert_eq!(rows, 110);
    assert_eq!(finished, 4);
    assert_eq!(shapes, [1, 2, 3, 4].into_iter().collect());
    owner.close()?;
    oracle.close()?;
    assert!(scheduler.close(200, None)?.settlement_failures().is_empty());
    stream.close()?;
    eprintln!(
        "live scheduler owner parity: {rows} full-logit rows, {finished} completed requests, decode shapes {shapes:?}"
    );
    Ok(())
}
