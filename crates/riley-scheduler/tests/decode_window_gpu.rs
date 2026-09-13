#![cfg(feature = "cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig,
    PreparedLlamaForwardConfig,
};
use riley_scheduler::{
    ExecutionShapePolicy, IterationTiming, OverloadPolicy, RequestDescriptor,
    SampledIterationToken, Scheduler, SchedulerConfig,
};

#[test]
#[ignore = "requires checkpoint and CUDA GPU; two-graph dependent decode"]
fn paired_decode_matches_serial_model_and_reclaims_pages() -> Result<(), Box<dyn std::error::Error>>
{
    let path =
        std::path::PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("model missing")?);
    let model = LoadedModel::load(
        &path,
        LoadLimits::default().with_weight_byte_limits(1 << 30, 1 << 30)?,
    )?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut reference: Option<Vec<Vec<u32>>> = None;
    for (window_mode, terminal_mode) in [(false, false), (true, false), (true, true)] {
        let mut stream = context.create_stream()?;
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 64, 1, 256)?,
            PreparedLlamaForwardConfig::default(),
        );
        let executor = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        let mut session =
            executor.into_owned_buffered_variable_mixed_session(&context, 128, true)?;
        let mut scheduler = Scheduler::new_with_execution_shape(
            SchedulerConfig {
                max_waiting_requests: 32,
                max_waiting_prompt_tokens: 4096,
                max_active_sequences: 32,
                max_sequence_tokens: 128,
                iteration_token_budget: 128,
                max_prefill_chunk_tokens: 32,
                aging_threshold_ns: 1,
                overload_policy: OverloadPolicy::Wait,
                admission_timeout_ns: None,
                max_promised_kv_blocks: 256,
                metrics_window_samples: 64,
            },
            riley_runtime::paged_kv::KvLayout::checked(30, 256, 3, 64)?,
            ExecutionShapePolicy::MixedPrefillDecode32,
        )?;
        let mut ids = vec![];
        for row in 0..32 {
            let prompt = (0..15 + row % 9)
                .map(|i| 17 + ((i * 37 + row * 73) % 900) as u32)
                .collect();
            ids.push(
                scheduler
                    .submit(RequestDescriptor::new(prompt, 32 + row % 3), 0)?
                    .request_id(),
            );
        }
        let mut expected_lengths: Vec<_> = (0..32).map(|r| 32 + r % 3).collect();
        let mut outputs = vec![vec![]; 32];
        let mut complete = 0;
        let mut clock = 1;
        let mut windows = 0;
        while complete < 32 {
            let updates = if let Some(window) = if window_mode {
                scheduler.plan_decode_window(clock)?
            } else {
                None
            } {
                let authority = scheduler.authorize_decode_window(&window)?;
                let downloaded = riley_scheduler::execution::execute_llama_decode_window(
                    &authority,
                    &mut session,
                )
                .unwrap();
                assert!(session.issue_rows(1).is_err());
                assert!(session
                    .confirm_scheduler_commit(window.first().iteration_id().get())
                    .is_err());
                let terminal = terminal_mode && windows == 0;
                let mut step = 0;
                let results = downloaded.map(|d| {
                    let samples = d
                        .greedy_token_ids()
                        .iter()
                        .enumerate()
                        .map(|(slot, &t)| {
                            SampledIterationToken::new(t, terminal && step == 0 && slot == 0)
                        })
                        .collect::<Vec<_>>();
                    step += 1;
                    d.into_result(&samples, IterationTiming::new(0, 0)).unwrap()
                });
                drop(authority);
                if terminal {
                    let stopped = window.first().decode_items()[0].request_id();
                    let cancelled = window.first().decode_items()[1].request_id();
                    let stop_row = ids.iter().position(|&id| id == stopped).unwrap();
                    let cancel_row = ids.iter().position(|&id| id == cancelled).unwrap();
                    expected_lengths[stop_row] = outputs[stop_row].len() + 1;
                    expected_lengths[cancel_row] = outputs[cancel_row].len();
                    scheduler.cancel(cancelled, clock + 1)?;
                }
                assert!(session
                    .confirm_decode_window_commit(
                        window.first().iteration_id().get(),
                        window.second().iteration_id().get() + 1
                    )
                    .is_err());
                let updates = scheduler.complete_decode_window_after_drain(
                    &results[0],
                    &results[1],
                    clock + 1,
                )?;
                assert!(updates.settlement_failures().is_empty());
                session.confirm_decode_window_commit(
                    window.first().iteration_id().get(),
                    window.second().iteration_id().get(),
                )?;
                windows += 1;
                updates
            } else {
                let plan = scheduler
                    .plan_iteration(clock)?
                    .into_parts()
                    .0
                    .ok_or("no plan")?;
                let authority = scheduler.authorize_execution(&plan)?;
                let d = riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(
                    &authority,
                    &mut session,
                    true,
                )
                .unwrap();
                let samples = d
                    .greedy_token_ids()
                    .iter()
                    .map(|&t| SampledIterationToken::new(t, false))
                    .collect::<Vec<_>>();
                let result = d.into_result(&samples, IterationTiming::new(0, 0)).unwrap();
                drop(authority);
                let updates = scheduler.complete_iteration(&result, clock + 1)?;
                session.confirm_scheduler_commit(plan.iteration_id().get())?;
                updates
            };
            assert!(updates.settlement_failures().is_empty());
            for event in updates.token_events() {
                outputs[ids.iter().position(|id| *id == event.request_id()).unwrap()]
                    .push(event.token_id());
            }
            complete += updates.completions().len();
            clock += 2;
            assert!(clock < 512);
        }
        for (row, tokens) in outputs.iter().enumerate() {
            assert_eq!(tokens.len(), expected_lengths[row]);
        }
        if let Some(expected) = &reference {
            for (actual, expected) in outputs.iter().zip(expected.iter()) {
                assert_eq!(
                    actual,
                    &expected[..actual.len()],
                    "paired generation must equal serial V7 prefix"
                );
            }
            assert!(windows >= 8);
        } else {
            reference = Some(outputs);
        }
        scheduler.close(clock, None)?;
        session.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        eprintln!(
            "DECODE_WINDOW mode={} terminal={} requests=32 windows={} allocation_zero=true",
            window_mode, terminal_mode, windows
        );
    }
    Ok(())
}
