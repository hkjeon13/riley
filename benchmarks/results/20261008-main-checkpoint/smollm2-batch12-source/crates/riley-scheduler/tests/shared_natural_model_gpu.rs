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
    physical: usize,
) -> Result<PreparedLlamaBatchExecutor> {
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 128, 10, 1, physical)?,
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
#[ignore = "requires CUDA, checkpoint and predeclared natural corpus"]
fn shared_natural_teacher_forced() -> Result {
    let path=std::env::var_os("RILEY_NATURAL_CORPUS").ok_or("corpus missing")?;
    let data: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    assert_eq!(data.len(),64);
    let base=std::env::var_os("RILEY_SHARED_DUMP").ok_or("dump missing")?;
    std::fs::create_dir(base)?;
    for (case,chunk) in data.chunks_exact(8).enumerate() {
        let corpus: Vec<Vec<u32>>=chunk.iter().map(|x|x["tokens"].as_array().unwrap().iter().map(|t|t.as_u64().unwrap() as u32).collect()).collect();
        assert!(corpus.iter().all(|x|x.len()==160));run_mode(&corpus,case)?;
    }
    Ok(())
}
fn run_mode(corpus: &[Vec<u32>], case: usize) -> Result {
    let capacity=8;let cancel_and_replace=false;let alternate=false;
    let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    let model = LoadedModel::load(std::path::Path::new(&path), LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut owner =
        prepare(&model, &context, &mut stream, capacity * 10)?.into_owned_shared_multi_decode_graph(&context)?;
    let mut oracle = prepare(&model, &context, &mut stream, capacity * 10)?.into_owned_decode_graph(&context)?;
    let mut scheduler = Scheduler::new_with_execution_shape(
        SchedulerConfig {
            max_waiting_requests: 8,
            max_waiting_prompt_tokens: 1024,
            max_active_sequences: capacity,
            max_sequence_tokens: 160,
            iteration_token_budget: 128,
            max_prefill_chunk_tokens: 128,
            aging_threshold_ns: 1,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: None,
            max_promised_kv_blocks: capacity * 10,
            metrics_window_samples: 16,
        },
        KvLayout::checked(30, capacity * 10, 3, 64)?,
        ExecutionShapePolicy::CompletePrefill128DecodeN,
    )?;
    for r in 0..capacity as u32 {
        let prompt = corpus[r as usize][..128].to_vec();
        scheduler.submit(RequestDescriptor::new(prompt, 32), 0)?;
    }
    let dump = std::path::PathBuf::from(std::env::var_os("RILEY_SHARED_DUMP").ok_or("dump path missing")?).join(format!("batch{case}"));
    std::fs::create_dir(&dump)?;
    use std::io::Write;
    let mut records = std::fs::OpenOptions::new().create_new(true).write(true).open(dump.join("rows.jsonl"))?;
    let mut cancelled_blocks = std::collections::BTreeSet::<u32>::new();
    let mut replacement = None;
    let mut reused = false;
    let mut rows = 0;
    let mut shapes = std::collections::BTreeSet::new();
    let mut finished = 0;
    for now in 1..200 {
        let Some(plan) = scheduler.plan_iteration(now)?.into_parts().0 else {
            break;
        };
        let authority = scheduler.authorize_execution(&plan)?;
        let greedy = alternate && now % 2 == 0;
        let mut workspace = Vec::with_capacity(capacity);
        let downloaded = execute_llama_iteration_multi_graph(&authority, &mut owner, greedy.then_some(&mut workspace))
            .map_err(|e| format!("execution failed: {e:?}"))?;
        let mut samples = vec![SampledIterationToken::new(0, false); downloaded.output_count()];
        for work in plan.prefill_items().iter().chain(plan.decode_items()) {
            let table = &plan.block_tables()[work.block_table_index()];
            if Some(work.request_id()) == replacement {
                reused |= table
                    .physical_block_ids()
                    .iter()
                    .any(|id| cancelled_blocks.contains(id));
            }
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
            assert!(!greedy);
            let actual = &downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304];
            let name = format!("r{}-i{}", work.request_id().get(), now);
            for (suffix,bytes) in [("shared",actual),("original",expected)] {
                let mut f = std::fs::OpenOptions::new().create_new(true).write(true).open(dump.join(format!("{name}-{suffix}.bin")))?;
                f.write_all(bytes)?;
            }
            let mut se=0.0f64;let mut norm=0.0f64;let mut max=0.0f64;let mut changed=0usize;
            for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let af=f32::from_bits(u32::from(u16::from_le_bytes([a[0],a[1]]))<<16) as f64;
                let bf=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16) as f64;
                assert!(af.is_finite()&&bf.is_finite());let d=(af-bf).abs();se+=d*d;norm+=bf*bf;max=max.max(d);changed+=usize::from(a!=b);
            }
            writeln!(records,"{{\"name\":\"{name}\",\"request\":{},\"input_tokens\":{:?},\"target_length\":{},\"changed_logits\":{changed},\"relative_l2_to_original\":{},\"max_abs_to_original\":{max},\"original_token\":{}}}",work.request_id().get(),work.input_tokens(),work.target_logical_length(),(se/norm.max(1e-30)).sqrt(),oracle.greedy_token()?)?;
            samples[slot] = SampledIterationToken::new(corpus[work.request_id().get() as usize - 1][work.target_logical_length()], false);
            rows += 1;
        }
        if plan.prefill_items().is_empty() {
            shapes.insert(plan.batch_size());
        }
        drop(authority);
        let cancelling = cancel_and_replace && now == if capacity == 8 { 20 } else { 12 };
        if cancelling {
            let work = &plan.decode_items()[0];
            cancelled_blocks
                .extend(plan.block_tables()[work.block_table_index()].physical_block_ids());
            assert!(
                scheduler
                    .cancel(work.request_id(), now)?
                    .deferred_until_iteration_settles()
            );
        }
        let id = downloaded.iteration_id().get();
        let result = downloaded
            .into_result(&samples, IterationTiming::new(0, 0))
            .map_err(|e| format!("result failed: {e:?}"))?;
        let settled = scheduler.complete_iteration(&result, now)?;
        assert!(settled.settlement_failures().is_empty());
        finished += settled.completions().len();
        owner.confirm_scheduler_commit(id)?;
        if cancelling {
            assert!(
                settled
                    .completions()
                    .iter()
                    .any(|c| c.reason() == RequestFinishReason::Cancelled)
            );
            replacement = Some(
                scheduler
                    .submit(
                        RequestDescriptor::new(
                            (0..128).map(|i| (i * 177 + 999) % 49152).collect(),
                            32,
                        ),
                        now,
                    )?
                    .request_id(),
            );
        }
    }
    if cancel_and_replace {
        assert!(
            reused,
            "replacement must reuse at least one cancelled KV page"
        );
        assert_eq!(finished, capacity + 1);
    } else {
        assert_eq!(rows, capacity * 32);
        assert_eq!(finished, capacity);
    }
    assert_eq!(shapes, (1..=capacity).collect());
    owner.close()?;
    oracle.close()?;
    assert!(scheduler.close(200, None)?.settlement_failures().is_empty());
    stream.close()?;
    eprintln!(
        "shared model diagnostic (not numerical acceptance): {rows} full-logit rows, {finished} completed requests, decode shapes {shapes:?}"
    );
    Ok(())
}
