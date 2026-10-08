//! Durable diagnostic-only receipts for the bounded C03-A V1 routing trace.
//!
//! This test support module is deliberately outside the production scheduler
//! crate. A receipt records a failed host-side replay for CI retrieval; it does
//! not replay the scheduler, identify the original panic, qualify a candidate,
//! or establish GPU evidence.

use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use riley_runtime::paged_kv::{KV_BLOCK_SIZE, KvLayout};
use riley_scheduler::{OverloadPolicy, SchedulerConfig};
use serde::Serialize;

use crate::general_mixed_operation_trace::{
    GeneralMixedOperationTrace, parse_general_mixed_operation_trace_descriptor,
    serialize_general_mixed_operation_trace_descriptor,
};

const RECEIPT_FORMAT: &str = "riley.scheduler.routing-fuzz-receipt";
const RECEIPT_FORMAT_VERSION: u8 = 1;
const RECEIPT_SCOPE: &str = "diagnostic-only";
const RECEIPT_TRACE_KIND: &str = "general-mixed-operation-v1";
const RECEIPT_TEST_TARGET: &str = "riley-scheduler::general_mixed_operation_routing";
const FAILURE_PREDICATE: &str = "inner-replayer-panicked-only";
const REDUCER_SCOPE: &str = "v1-selector-local";
const MINIMIZED_CASE_ID: &str = "failing-minimized";
const RECEIPT_DIRECTORY_ENV: &str = "RILEY_ROUTING_FUZZ_RECEIPT_DIR";
const SOURCE_REVISION_ENV: &str = "RILEY_ROUTING_FUZZ_SOURCE_REVISION";
const NOT_ESTABLISHED: [&str; 5] = [
    "c02_qualification",
    "c03_b_gpu_evidence",
    "general_or_global_minimum",
    "panic_site_payload_signature_root_cause",
    "scheduler_reexecution",
];

const SYMBOLIC_KV_LAYER_COUNT: usize = 1;
const SYMBOLIC_KV_PHYSICAL_BLOCK_COUNT: usize = 64;
const SYMBOLIC_KV_HEAD_COUNT: usize = 1;
const SYMBOLIC_KV_HEAD_DIMENSION: usize = 8;
const SYMBOLIC_KV_BLOCK_SIZE_TOKENS: usize = KV_BLOCK_SIZE;
const DECODER_SUBMIT_AND_PRIME_NS: u64 = 0;
const FINAL_PREFILL_AND_MIXED_NS: u64 = 1;
const CLOSE_NS: u64 = 2;

/// Exact test-only scheduler configuration derived from a V1 trace.
///
/// Keeping this projection beside the receipt prevents the recorder from
/// independently reconstructing the harness configuration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GeneralMixedOperationSchedulerConfig {
    max_waiting_requests: usize,
    max_waiting_prompt_tokens: usize,
    max_active_sequences: usize,
    max_sequence_tokens: usize,
    iteration_token_budget: usize,
    max_prefill_chunk_tokens: usize,
    aging_threshold_ns: u64,
    overload_policy: &'static str,
    admission_timeout_ns: Option<u64>,
    max_promised_kv_blocks: usize,
    metrics_window_samples: usize,
}

impl GeneralMixedOperationSchedulerConfig {
    /// Derives the bounded test scheduler settings from one V1 trace.
    #[must_use]
    pub fn for_trace(trace: &GeneralMixedOperationTrace) -> Self {
        let width = trace
            .decoder_count
            .checked_add(trace.final_prefill_count)
            .expect("bounded general mixed request width");
        Self {
            max_waiting_requests: width,
            max_waiting_prompt_tokens: width,
            max_active_sequences: width,
            max_sequence_tokens: 3,
            iteration_token_budget: width,
            max_prefill_chunk_tokens: 1,
            aging_threshold_ns: 2,
            overload_policy: "wait",
            admission_timeout_ns: None,
            max_promised_kv_blocks: width,
            metrics_window_samples: 8,
        }
    }

    /// Materializes the scheduler configuration consumed by the host replay.
    #[must_use]
    pub fn scheduler_config(&self) -> SchedulerConfig {
        SchedulerConfig {
            max_waiting_requests: self.max_waiting_requests,
            max_waiting_prompt_tokens: self.max_waiting_prompt_tokens,
            max_active_sequences: self.max_active_sequences,
            max_sequence_tokens: self.max_sequence_tokens,
            iteration_token_budget: self.iteration_token_budget,
            max_prefill_chunk_tokens: self.max_prefill_chunk_tokens,
            aging_threshold_ns: self.aging_threshold_ns,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: self.admission_timeout_ns,
            max_promised_kv_blocks: self.max_promised_kv_blocks,
            metrics_window_samples: self.metrics_window_samples,
        }
    }
}

/// Builds the fixed symbolic KV layout used by the host replay.
#[must_use]
pub fn symbolic_kv_layout() -> KvLayout {
    KvLayout::checked(
        SYMBOLIC_KV_LAYER_COUNT,
        SYMBOLIC_KV_PHYSICAL_BLOCK_COUNT,
        SYMBOLIC_KV_HEAD_COUNT,
        SYMBOLIC_KV_HEAD_DIMENSION,
    )
    .expect("valid C03-A symbolic KV layout")
}

#[derive(Clone, Copy, Serialize)]
struct SymbolicKvLayoutReceipt {
    layer_count: usize,
    physical_block_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    block_size_tokens: usize,
}

impl SymbolicKvLayoutReceipt {
    const fn fixed() -> Self {
        Self {
            layer_count: SYMBOLIC_KV_LAYER_COUNT,
            physical_block_count: SYMBOLIC_KV_PHYSICAL_BLOCK_COUNT,
            key_value_head_count: SYMBOLIC_KV_HEAD_COUNT,
            head_dimension: SYMBOLIC_KV_HEAD_DIMENSION,
            block_size_tokens: SYMBOLIC_KV_BLOCK_SIZE_TOKENS,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct ReplayTimelineReceipt {
    #[serde(rename = "decoder_submit_and_prime_ns")]
    decoder_submit_and_prime: u64,
    #[serde(rename = "final_prefill_submit_and_mixed_ns")]
    final_prefill_submit_and_mixed: u64,
    #[serde(rename = "close_ns")]
    close: u64,
}

impl ReplayTimelineReceipt {
    const fn fixed() -> Self {
        Self {
            decoder_submit_and_prime: DECODER_SUBMIT_AND_PRIME_NS,
            final_prefill_submit_and_mixed: FINAL_PREFILL_AND_MIXED_NS,
            close: CLOSE_NS,
        }
    }
}

#[derive(Serialize)]
struct RoutingFuzzReceiptV1 {
    format: &'static str,
    format_version: u8,
    scope: &'static str,
    trace_kind: &'static str,
    test_target: &'static str,
    source_revision: String,
    source_case_id: String,
    failure_predicate: &'static str,
    reducer_scope: &'static str,
    source_descriptor_json: String,
    minimized_descriptor_json: String,
    source_operations: String,
    minimized_operations: String,
    source_scheduler_config: GeneralMixedOperationSchedulerConfig,
    minimized_scheduler_config: GeneralMixedOperationSchedulerConfig,
    symbolic_kv_layout: SymbolicKvLayoutReceipt,
    replay_timeline_ns: ReplayTimelineReceipt,
    not_established: [&'static str; 5],
}

/// Builds the canonical diagnostic receipt document without writing it.
///
/// The checker can validate descriptor/config bindings structurally, but it
/// cannot establish that this exact panic still reproduces or that the result
/// is globally minimal.
pub fn general_mixed_operation_receipt_document(
    source_revision: &str,
    source_case_id: &str,
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<String, String> {
    validate_source_revision(source_revision)?;
    validate_source_case_id(source_case_id)?;
    validate_minimized_binding(source, minimized)?;
    let source_descriptor_json =
        serialize_general_mixed_operation_trace_descriptor(source_case_id, source);
    let minimized_descriptor_json =
        serialize_general_mixed_operation_trace_descriptor(MINIMIZED_CASE_ID, minimized);
    let parsed_source = parse_general_mixed_operation_trace_descriptor(&source_descriptor_json)
        .map_err(|error| format!("source descriptor is not strict-canonical: {error}"))?;
    let parsed_minimized =
        parse_general_mixed_operation_trace_descriptor(&minimized_descriptor_json)
            .map_err(|error| format!("minimized descriptor is not strict-canonical: {error}"))?;
    if parsed_source.case_id != source_case_id || parsed_source.trace != *source {
        return Err("source descriptor binding drifted before receipt serialization".to_owned());
    }
    if parsed_minimized.case_id != MINIMIZED_CASE_ID || parsed_minimized.trace != *minimized {
        return Err("minimized descriptor binding drifted before receipt serialization".to_owned());
    }
    let receipt = RoutingFuzzReceiptV1 {
        format: RECEIPT_FORMAT,
        format_version: RECEIPT_FORMAT_VERSION,
        scope: RECEIPT_SCOPE,
        trace_kind: RECEIPT_TRACE_KIND,
        test_target: RECEIPT_TEST_TARGET,
        source_revision: source_revision.to_owned(),
        source_case_id: source_case_id.to_owned(),
        failure_predicate: FAILURE_PREDICATE,
        reducer_scope: REDUCER_SCOPE,
        source_descriptor_json,
        minimized_descriptor_json,
        source_operations: source.describe_operations(),
        minimized_operations: minimized.describe_operations(),
        source_scheduler_config: GeneralMixedOperationSchedulerConfig::for_trace(source),
        minimized_scheduler_config: GeneralMixedOperationSchedulerConfig::for_trace(minimized),
        symbolic_kv_layout: SymbolicKvLayoutReceipt::fixed(),
        replay_timeline_ns: ReplayTimelineReceipt::fixed(),
        not_established: NOT_ESTABLISHED,
    };
    let mut document = serde_json::to_string(&receipt)
        .map_err(|error| format!("receipt JSON serialization failed: {error}"))?;
    document.push('\n');
    Ok(document)
}

/// Writes one receipt with create-new semantics at an explicit test-owned path.
///
/// Existing files are never replaced. Callers should retain the primary replay
/// failure even if this diagnostic write returns an error.
pub fn write_general_mixed_operation_receipt(
    destination: &Path,
    source_revision: &str,
    source_case_id: &str,
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<(), String> {
    if destination.as_os_str().is_empty() {
        return Err("routing fuzz receipt path must not be empty".to_owned());
    }
    let document = general_mixed_operation_receipt_document(
        source_revision,
        source_case_id,
        source,
        minimized,
    )?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| {
            format!(
                "could not create routing fuzz receipt {}: {error}",
                destination.display()
            )
        })?;
    if let Err(error) = file.write_all(document.as_bytes()) {
        drop(file);
        return Err(cleanup_partial_receipt(destination, "write", &error));
    }
    if let Err(error) = file.sync_all() {
        drop(file);
        return Err(cleanup_partial_receipt(destination, "synchronize", &error));
    }
    Ok(())
}

/// Writes one receipt under a pre-created absolute directory using a safe case leaf.
///
/// This is the pure filesystem portion of the environment-backed sink. It is
/// separately callable by tests so process environment mutation is unnecessary.
pub fn write_general_mixed_operation_receipt_in_directory(
    directory: &Path,
    source_revision: &str,
    source_case_id: &str,
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<PathBuf, String> {
    validate_source_case_id(source_case_id)?;
    validate_receipt_directory(directory)?;
    let destination = directory.join(format!(
        "{RECEIPT_TRACE_KIND}-{source_case_id}-{seed:016x}.json",
        seed = source.seed
    ));
    write_general_mixed_operation_receipt(
        &destination,
        source_revision,
        source_case_id,
        source,
        minimized,
    )?;
    Ok(destination)
}

/// Resolves an optional receipt sink from explicit environment-like values.
///
/// Both values must be absent to disable recording. Supplying only one is an
/// error rather than a silent loss of a requested diagnostic. This pure entry
/// point keeps the process-environment wrapper trivial and testable without
/// mutating global environment state.
pub fn write_general_mixed_operation_receipt_from_values(
    receipt_directory: Option<OsString>,
    source_revision: Option<OsString>,
    source_case_id: &str,
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<Option<PathBuf>, String> {
    match (receipt_directory, source_revision) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(format!(
            "{SOURCE_REVISION_ENV} must be unset when {RECEIPT_DIRECTORY_ENV} is unset"
        )),
        (Some(_), None) => Err(format!(
            "{SOURCE_REVISION_ENV} is required when {RECEIPT_DIRECTORY_ENV} is set"
        )),
        (Some(directory), Some(source_revision)) => {
            let source_revision = source_revision.into_string().map_err(|_| {
                format!(
                    "{SOURCE_REVISION_ENV} must be valid UTF-8 when receipt recording is enabled"
                )
            })?;
            let destination = write_general_mixed_operation_receipt_in_directory(
                Path::new(&directory),
                &source_revision,
                source_case_id,
                source,
                minimized,
            )?;
            Ok(Some(destination))
        }
    }
}

/// Writes a receipt only when the CI output-directory environment variable is set.
///
/// Local passing test runs normally leave that variable absent and create no
/// file. CI provides one pre-created absolute directory per run; the safe
/// source case ID becomes a create-new leaf name so concurrent failures never
/// overwrite one another. A validated source revision is mandatory so the
/// resulting diagnostic cannot be mistaken for another tree.
pub fn write_general_mixed_operation_receipt_from_environment(
    source_case_id: &str,
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<Option<PathBuf>, String> {
    write_general_mixed_operation_receipt_from_values(
        env::var_os(RECEIPT_DIRECTORY_ENV),
        env::var_os(SOURCE_REVISION_ENV),
        source_case_id,
        source,
        minimized,
    )
}

fn validate_source_revision(source_revision: &str) -> Result<(), String> {
    if source_revision.len() != 40
        || !source_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        || source_revision.bytes().all(|byte| byte == b'0')
    {
        return Err(
            "routing fuzz receipt source_revision must be 40 lowercase nonzero hexadecimal digits"
                .to_owned(),
        );
    }
    Ok(())
}

fn cleanup_partial_receipt(destination: &Path, operation: &str, error: &std::io::Error) -> String {
    match fs::remove_file(destination) {
        Ok(()) => format!(
            "could not {operation} routing fuzz receipt {}; removed the partial receipt: {error}",
            destination.display()
        ),
        Err(cleanup_error) => format!(
            "could not {operation} routing fuzz receipt {}: {error}; additionally could not remove the partial receipt: {cleanup_error}",
            destination.display()
        ),
    }
}

fn validate_receipt_directory(directory: &Path) -> Result<(), String> {
    if !directory.is_absolute() {
        return Err("routing fuzz receipt directory must be absolute".to_owned());
    }
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        format!(
            "could not inspect routing fuzz receipt directory {}: {error}",
            directory.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "routing fuzz receipt directory must be pre-created and non-symlink: {}",
            directory.display()
        ));
    }
    Ok(())
}

fn validate_source_case_id(source_case_id: &str) -> Result<(), String> {
    let bytes = source_case_id.as_bytes();
    if !(1..=96).contains(&bytes.len())
        || bytes.first() == Some(&b'-')
        || bytes.last() == Some(&b'-')
        || !bytes
            .iter()
            .copied()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(
            "routing fuzz receipt source_case_id must be a bounded lowercase identifier".to_owned(),
        );
    }
    Ok(())
}

fn validate_minimized_binding(
    source: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> Result<(), String> {
    if source.seed != minimized.seed {
        return Err("routing fuzz receipt minimized trace changed source seed".to_owned());
    }
    if source.settlement != minimized.settlement {
        return Err("routing fuzz receipt minimized trace changed settlement".to_owned());
    }
    if minimized.shrink_rank() > source.shrink_rank() {
        return Err("routing fuzz receipt minimized trace increased reducer rank".to_owned());
    }
    Ok(())
}
