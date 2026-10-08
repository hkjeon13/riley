//! Diagnostic HTTP harness using the matched production engine constructor.
//! No arithmetic/ownership changes. GPU execution is deferred until benchmark receipts are terminal.
#![cfg(feature = "cuda")]
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{
    ExecutionGraphPolicy, LlamaBatchMetadataConfig, LlamaReductionProfile,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_scheduler::{OverloadPolicy, SchedulerConfig};
use riley_server::domain::{ModelMetadata, RequestLimits};
use riley_server::engine::{
    C02CaptureMetrics, CudaBackendConfig, CudaEngineResources, EngineConfig, InferenceEngine,
};
use riley_server::service::{CompletionBackend, ServerConfig, start_server};
use serde_json::{Value, json};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn create(path: &Path, value: &Value) -> TestResult {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(&mut f, value)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}
fn owned(m: C02CaptureMetrics) -> Value {
    json!({
        "request_states": {"active":m.request_states.active, "pending_requests":m.request_states.pending_requests,
            "completed":m.request_states.completed, "failed":m.request_states.failed,
            "cancelled":m.request_states.cancelled, "capacity_rejections":m.request_states.capacity_rejections},
        "kv_blocks": {"free":m.kv_blocks.free, "reserved":m.kv_blocks.reserved, "active":m.kv_blocks.active},
        "allocation": {"device_live_count":m.allocation.device_live_count,"device_live_bytes":m.allocation.device_live_bytes,
            "pinned_live_count":m.allocation.pinned_live_count,"pinned_live_bytes":m.allocation.pinned_live_bytes},
        "quiescence": {"completion_outbox":m.quiescence.completion_outbox,
            "outstanding_iterations":m.quiescence.outstanding_iterations,
            "riley_owned_live_allocations":m.quiescence.riley_owned_live_allocations,
            "worker_accepting":m.quiescence.worker_accepting,"scheduler_accepting":m.quiescence.scheduler_accepting},
        "shutdown_quiescent":m.is_quiescent()
    })
}
#[test]
#[ignore = "source-bound GPU diagnostic; requires matched96 terminal and pinned checkpoint"]
fn matched_all8_source_owned_http() -> TestResult {
    let case = std::env::var("RILEY_STABILITY_CASE")?;
    let (capacity, workload) = case.split_once('-').ok_or("invalid case")?;
    let concurrency: usize = capacity
        .strip_prefix('c')
        .ok_or("invalid concurrency")?
        .parse()?;
    assert!([1, 8, 16, 32].contains(&concurrency));
    assert!(["fixed", "natural"].contains(&workload));
    let natural = workload == "natural";
    let max_sequence_tokens = if natural { 1024 } else { 160 };
    let max_output_tokens = if natural { 128 } else { 32 };
    let chunk = if natural { 512 } else { 128 };
    let blocks = (if natural { 64 } else { 10 }) * concurrency;
    let output = std::path::PathBuf::from(std::env::var("RILEY_STABILITY_EVIDENCE")?);
    fs::create_dir(&output)?;
    let model = LoadedModel::load(
        &std::path::PathBuf::from(std::env::var("RILEY_REAL_CHECKPOINT")?),
        LoadLimits::default().with_weight_byte_limits(2 << 30, 2 << 30)?,
    )?;
    let scheduler = SchedulerConfig {
        max_waiting_requests: 64,
        max_waiting_prompt_tokens: 64 * max_sequence_tokens,
        max_active_sequences: concurrency,
        max_sequence_tokens,
        iteration_token_budget: 512,
        max_prefill_chunk_tokens: chunk,
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(30_000_000_000),
        max_promised_kv_blocks: blocks,
        metrics_window_samples: 1024,
    };
    let executor = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 1, blocks, 1, blocks)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_synchronous_metadata()
    .with_fixed_maximum_shape()
    .with_reduction_profile(LlamaReductionProfile::CanonicalV1)
    .with_mixed_execution()
    .with_grouped_ragged_attention_heads();
    let resources = CudaEngineResources::prepare(
        ModelMetadata {
            model_id: "g04-smol".to_owned(),
            created_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            owned_by: "riley".to_owned(),
            context_window_tokens: max_sequence_tokens,
            max_output_tokens,
        },
        model,
        CudaBackendConfig {
            device_ordinal: 0,
            scheduler,
            executor,
            gpu_greedy: true,
        },
    )?
    .with_execution_graph_policy(ExecutionGraphPolicy::Require);
    let engine = Arc::new(InferenceEngine::start_cuda(
        resources,
        EngineConfig {
            command_queue_capacity: 64,
            event_channel_capacity: 32,
            max_inflight_requests: concurrency + 64,
            admission_timeout: Duration::from_secs(30),
            idle_poll_interval: Duration::from_millis(1),
        },
    )?);
    let backend: Arc<dyn CompletionBackend> = engine.clone();
    let server = start_server(
        ServerConfig {
            bind_address: SocketAddr::from(([127, 0, 0, 1], 0)),
            worker_threads: concurrency.max(8),
            request_limits: RequestLimits {
                max_output_tokens,
                ..RequestLimits::default()
            },
            ..ServerConfig::default()
        },
        backend,
    )?;
    create(
        &output.join("ready.json"),
        &json!({"case":case,"address":server.local_address().to_string(),
        "numerics":"variable-smol-v7","concurrency":concurrency,"token_budget":512,"chunk_tokens":chunk,
        "context_tokens":max_sequence_tokens,"max_output_tokens":max_output_tokens,"physical_KV_blocks":blocks,
        "performance_qualification":false}),
    )?;
    // Explicit external commands avoid periodic observer interference with the HTTP stream.
    // Command files must be atomically renamed into place by the driver.
    let deadline = Instant::now() + Duration::from_secs(3600);
    let mut index = 0_u64;
    while Instant::now() < deadline {
        let command = output.join(format!("command-{index:06}.json"));
        if !command.exists() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }
        let request: Value = serde_json::from_slice(&fs::read(&command)?)?;
        match request["operation"].as_str().ok_or("missing operation")? {
            "snapshot" => {
                let observations = server.observations();
                let metrics = engine.c02_metrics_snapshot()?;
                let records = observations.observations.into_iter().map(|v|json!({
                    "request_id":v.request_id,"model_id":v.model_id,"status":format!("{:?}",v.status),
                    "error_class":format!("{:?}",v.error_class),"tokens_generated":v.tokens_generated,
                    "first_token_ns":v.time_to_first_token.map(|d|d.as_nanos().to_string()),
                    "active_requests":v.active_requests,"waiting_requests":v.waiting_requests
                })).collect::<Vec<_>>();
                create(
                    &output.join(format!("response-{index:06}.json")),
                    &json!({
                        "metrics":owned(metrics),"observations":records,"dropped_observations":observations.dropped_observations,
                        "note":"No inference of prefill scheduler phase from absence of visible first token"
                    }),
                )?;
            }
            "shutdown" => {
                let evidence = server.shutdown_with_c02_evidence()?;
                assert!(evidence.final_metrics.is_quiescent());
                create(
                    &output.join("final-source-owned.json"),
                    &owned(evidence.final_metrics),
                )?;
                return Ok(());
            }
            _ => return Err("unsupported command".into()),
        }
        index += 1;
    }
    Err("source-owned harness command deadline exceeded; stability unproven".into())
}
