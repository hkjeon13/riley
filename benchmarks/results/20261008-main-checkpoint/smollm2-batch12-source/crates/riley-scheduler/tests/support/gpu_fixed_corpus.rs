//! Canonical fixed C03-B GPU-routing corpus descriptors.
//!
//! This module is intentionally GPU- and scheduler-free.  It fixes the small
//! set of source-controlled CUDA fixture topologies before the remote CUDA
//! target adapts them through public scheduler and executor APIs.  The
//! CPU-only contract test consumes the same descriptors to validate their
//! plan, KV-boundary, cancellation, and abort expectations without claiming a
//! device execution result.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

const DESCRIPTOR_FORMAT: &str = "riley.scheduler.gpu-fixed-corpus";
const DESCRIPTOR_FORMAT_VERSION: u8 = 1;
const DESCRIPTOR_TRACE_KIND: &str = "gpu-fixed-v1";
const KV_BLOCK_TOKEN_COUNT: usize = 16;

/// The fixed work category expected for one CUDA fixture route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFixedWorkKind {
    /// Final prompt tokens whose last token produces one output row.
    Prefill,
    /// One previously generated token fed back through a decode row.
    Decode,
}

/// One named phase in a fixed GPU fixture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFixedPhase {
    /// Primes the future decoder requests with their final prompt chunk.
    Prime,
    /// Executes the required mixed prefill/decode output plan.
    Mixed,
    /// Executes C=5's second decoder step at the 16 -> 17 KV boundary.
    BoundaryDecode,
}

/// Descriptor-selected terminal handling for the mixed plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GpuFixedSettlement {
    /// Build and commit complete dense sampled feedback.
    Commit,
    /// Cancel one selected request after device download and before commit.
    DeferredCancel {
        /// Logical label selected by the mixed plan.
        label: u8,
    },
    /// Deliberately fail public commit-data assembly after device download,
    /// then poison the in-flight scheduler plan with its returned disposition.
    AbortAfterInvalidSampleCount,
}

/// Expected terminal reason for a normal fixed-corpus replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFixedTerminalReason {
    /// The selected request was cancelled after the device had finished.
    Cancelled,
    /// The request reached its descriptor generation limit.
    Length,
}

/// Pure, descriptor-derived output routing expectation for one plan phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuFixedRoute {
    /// Source-controlled logical request label.
    pub label: u8,
    /// Zero-based generated-token index expected for this output.
    pub generation_step: usize,
    /// Dense CUDA output slot owned by this route.
    pub output_slot: usize,
    /// Public work kind expected in the iteration plan.
    pub work_kind: GpuFixedWorkKind,
    /// Number of model input tokens in this work item.
    pub input_token_count: usize,
    /// Exact logical KV length reserved by the plan.
    pub target_logical_length: usize,
}

/// On-disk canonical descriptor before its values are validated against the
/// fixed C03-B corpus grammar.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GpuFixedCorpusDescriptorV1 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    concurrency: usize,
    decoder_prompt_token_count: usize,
    decoder_max_new_tokens: usize,
    final_prefill_prompt_token_count: usize,
    final_prefill_max_new_tokens: usize,
    primed_labels: Vec<u8>,
    final_prefill_labels: Vec<u8>,
    settlement: GpuFixedSettlement,
}

/// One fully validated, source-controlled C03-B GPU fixture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuFixedCorpusCase {
    case_id: String,
    concurrency: usize,
    decoder_prompt_token_count: usize,
    decoder_max_new_tokens: usize,
    final_prefill_prompt_token_count: usize,
    final_prefill_max_new_tokens: usize,
    primed_labels: Vec<u8>,
    final_prefill_labels: Vec<u8>,
    settlement: GpuFixedSettlement,
}

impl GpuFixedCorpusCase {
    /// Stable source-controlled case identifier.
    #[must_use]
    pub fn case_id(&self) -> &str {
        &self.case_id
    }

    /// Maximum simultaneous scheduler sequences for this fixture.
    #[must_use]
    pub const fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Prompt length used to prime every future decoder request.
    #[must_use]
    pub const fn decoder_prompt_token_count(&self) -> usize {
        self.decoder_prompt_token_count
    }

    /// Generation capacity of every future decoder request.
    #[must_use]
    pub const fn decoder_max_new_tokens(&self) -> usize {
        self.decoder_max_new_tokens
    }

    /// Prompt length for a request first admitted in the mixed plan.
    #[must_use]
    pub const fn final_prefill_prompt_token_count(&self) -> usize {
        self.final_prefill_prompt_token_count
    }

    /// Generation capacity of a request first admitted in the mixed plan.
    #[must_use]
    pub const fn final_prefill_max_new_tokens(&self) -> usize {
        self.final_prefill_max_new_tokens
    }

    /// Labels that become decoders after the prime plan commits.
    #[must_use]
    pub fn primed_labels(&self) -> &[u8] {
        &self.primed_labels
    }

    /// Labels whose final prompt prefill participates in the mixed plan.
    #[must_use]
    pub fn final_prefill_labels(&self) -> &[u8] {
        &self.final_prefill_labels
    }

    /// Fixed post-download settlement action for the mixed plan.
    #[must_use]
    pub const fn settlement(&self) -> &GpuFixedSettlement {
        &self.settlement
    }

    /// Fixed Paged-KV block width used by this workspace ABI.
    #[must_use]
    pub const fn kv_block_token_count() -> usize {
        KV_BLOCK_TOKEN_COUNT
    }

    /// Maximum logical sequence length promised by the longest decoder.
    #[must_use]
    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.decoder_prompt_token_count + self.decoder_max_new_tokens - 1
    }

    /// Maximum flattened input-token count in any one fixed plan.
    #[must_use]
    pub fn maximum_iteration_input_tokens(&self) -> usize {
        let prime = self.primed_labels.len() * self.decoder_prompt_token_count;
        let mixed = self.primed_labels.len()
            + self.final_prefill_labels.len() * self.final_prefill_prompt_token_count;
        prime.max(mixed)
    }

    /// Maximum physical block-table entries required in any one fixed plan.
    #[must_use]
    pub fn maximum_plan_block_entries(&self) -> usize {
        let mixed = self.primed_labels.len() + self.final_prefill_labels.len();
        let boundary = if self.requires_boundary_decode() {
            self.primed_labels.len() * 2
        } else {
            0
        };
        mixed.max(boundary)
    }

    /// Maximum aggregate promised KV blocks across all admitted requests.
    #[must_use]
    pub fn promised_kv_blocks(&self) -> usize {
        let decoder_blocks = self
            .maximum_sequence_tokens()
            .div_ceil(KV_BLOCK_TOKEN_COUNT);
        let final_prefill_blocks = self
            .final_prefill_prompt_token_count
            .div_ceil(KV_BLOCK_TOKEN_COUNT);
        self.primed_labels.len() * decoder_blocks
            + self.final_prefill_labels.len() * final_prefill_blocks
    }

    /// Whether normal completion requires the second C=5 decoder plan.
    #[must_use]
    pub const fn requires_boundary_decode(&self) -> bool {
        self.decoder_max_new_tokens == 3
    }

    /// Whether this case intentionally reaches public commit-data assembly
    /// failure instead of committing its mixed plan.
    #[must_use]
    pub fn is_commit_data_assembly_failure(&self) -> bool {
        matches!(
            self.settlement,
            GpuFixedSettlement::AbortAfterInvalidSampleCount
        )
    }

    /// Returns every logical label in source-controlled submission order.
    #[must_use]
    pub fn all_labels(&self) -> Vec<u8> {
        self.primed_labels
            .iter()
            .chain(&self.final_prefill_labels)
            .copied()
            .collect()
    }

    /// Derives the exact dense routes expected for one fixture phase without
    /// consulting a scheduler plan or CUDA result.
    #[must_use]
    pub fn routes_for_phase(&self, phase: GpuFixedPhase) -> Vec<GpuFixedRoute> {
        match phase {
            GpuFixedPhase::Prime => self
                .primed_labels
                .iter()
                .enumerate()
                .map(|(output_slot, &label)| GpuFixedRoute {
                    label,
                    generation_step: 0,
                    output_slot,
                    work_kind: GpuFixedWorkKind::Prefill,
                    input_token_count: self.decoder_prompt_token_count,
                    target_logical_length: self.decoder_prompt_token_count,
                })
                .collect(),
            GpuFixedPhase::Mixed => {
                let mut routes = Vec::with_capacity(self.concurrency);
                for (output_slot, &label) in self.primed_labels.iter().enumerate() {
                    routes.push(GpuFixedRoute {
                        label,
                        generation_step: 1,
                        output_slot,
                        work_kind: GpuFixedWorkKind::Decode,
                        input_token_count: 1,
                        target_logical_length: self.decoder_prompt_token_count + 1,
                    });
                }
                let first_prefill_slot = routes.len();
                for (index, &label) in self.final_prefill_labels.iter().enumerate() {
                    routes.push(GpuFixedRoute {
                        label,
                        generation_step: 0,
                        output_slot: first_prefill_slot + index,
                        work_kind: GpuFixedWorkKind::Prefill,
                        input_token_count: self.final_prefill_prompt_token_count,
                        target_logical_length: self.final_prefill_prompt_token_count,
                    });
                }
                routes
            }
            GpuFixedPhase::BoundaryDecode => {
                assert!(
                    self.requires_boundary_decode(),
                    "only the C=5 KV-boundary corpus has a boundary decode phase"
                );
                self.primed_labels
                    .iter()
                    .enumerate()
                    .map(|(output_slot, &label)| GpuFixedRoute {
                        label,
                        generation_step: 2,
                        output_slot,
                        work_kind: GpuFixedWorkKind::Decode,
                        input_token_count: 1,
                        target_logical_length: self.decoder_prompt_token_count + 2,
                    })
                    .collect()
            }
        }
    }

    /// Returns the canonical per-block valid-token vector for a route target.
    #[must_use]
    pub fn valid_tokens_for(target_logical_length: usize) -> Vec<u16> {
        assert!(target_logical_length != 0, "fixed route target is nonzero");
        (0..target_logical_length.div_ceil(KV_BLOCK_TOKEN_COUNT))
            .map(|block_index| {
                let consumed = block_index * KV_BLOCK_TOKEN_COUNT;
                u16::try_from((target_logical_length - consumed).min(KV_BLOCK_TOKEN_COUNT))
                    .expect("fixed C03-B KV tail fits u16")
            })
            .collect()
    }

    /// Expected terminal reason after a normal committed replay.
    #[must_use]
    pub fn normal_terminal_reason(&self, label: u8) -> GpuFixedTerminalReason {
        match self.settlement {
            GpuFixedSettlement::DeferredCancel {
                label: cancelled_label,
            } if label == cancelled_label => GpuFixedTerminalReason::Cancelled,
            GpuFixedSettlement::Commit | GpuFixedSettlement::DeferredCancel { .. } => {
                GpuFixedTerminalReason::Length
            }
            GpuFixedSettlement::AbortAfterInvalidSampleCount => {
                panic!("commit-data assembly failure does not have a normal terminal ledger")
            }
        }
    }
}

impl GpuFixedCorpusDescriptorV1 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("GPU fixed corpus descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err("GPU fixed corpus descriptor format_version is unsupported".to_owned());
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND {
            return Err("GPU fixed corpus descriptor trace_kind is unsupported".to_owned());
        }
        if self.decoder_prompt_token_count == 0
            || self.decoder_max_new_tokens == 0
            || self.final_prefill_prompt_token_count == 0
            || self.final_prefill_max_new_tokens == 0
        {
            return Err("GPU fixed corpus token bounds must be nonzero".to_owned());
        }
        let labels = self
            .primed_labels
            .iter()
            .chain(&self.final_prefill_labels)
            .copied()
            .collect::<Vec<_>>();
        if labels.len() != self.concurrency || self.concurrency == 0 {
            return Err("GPU fixed corpus concurrency must equal its label count".to_owned());
        }
        if labels.len() > usize::from(u8::MAX) {
            return Err("GPU fixed corpus label domain exceeds u8".to_owned());
        }
        let unique = labels.iter().copied().collect::<BTreeSet<_>>();
        let contiguous = labels.iter().enumerate().all(|(index, label)| {
            u8::try_from(index + 1)
                .ok()
                .is_some_and(|expected| *label == expected)
        });
        if unique.len() != labels.len() || !contiguous {
            return Err(
                "GPU fixed corpus labels must be one contiguous ascending domain".to_owned(),
            );
        }
        match self.case_id.as_str() {
            "c5-kv15-to17-mixed-greedy" => {
                expect_case(
                    self,
                    5,
                    15,
                    3,
                    1,
                    1,
                    &[1, 2],
                    &[3, 4, 5],
                    &GpuFixedSettlement::Commit,
                )?;
            }
            "c8-mixed-deferred-cancel" => {
                expect_case(
                    self,
                    8,
                    3,
                    2,
                    1,
                    1,
                    &[1, 2, 3],
                    &[4, 5, 6, 7, 8],
                    &GpuFixedSettlement::DeferredCancel { label: 2 },
                )?;
            }
            "c8-mixed-greedy-commit-assembly-failure" => {
                expect_case(
                    self,
                    8,
                    3,
                    2,
                    1,
                    1,
                    &[1, 2, 3],
                    &[4, 5, 6, 7, 8],
                    &GpuFixedSettlement::AbortAfterInvalidSampleCount,
                )?;
            }
            _ => return Err("GPU fixed corpus case_id is unsupported".to_owned()),
        }
        Ok(())
    }

    fn into_case(self) -> Result<GpuFixedCorpusCase, String> {
        self.validate()?;
        Ok(GpuFixedCorpusCase {
            case_id: self.case_id,
            concurrency: self.concurrency,
            decoder_prompt_token_count: self.decoder_prompt_token_count,
            decoder_max_new_tokens: self.decoder_max_new_tokens,
            final_prefill_prompt_token_count: self.final_prefill_prompt_token_count,
            final_prefill_max_new_tokens: self.final_prefill_max_new_tokens,
            primed_labels: self.primed_labels,
            final_prefill_labels: self.final_prefill_labels,
            settlement: self.settlement,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn expect_case(
    actual: &GpuFixedCorpusDescriptorV1,
    concurrency: usize,
    decoder_prompt_token_count: usize,
    decoder_max_new_tokens: usize,
    final_prefill_prompt_token_count: usize,
    final_prefill_max_new_tokens: usize,
    primed_labels: &[u8],
    final_prefill_labels: &[u8],
    settlement: &GpuFixedSettlement,
) -> Result<(), String> {
    if actual.concurrency != concurrency
        || actual.decoder_prompt_token_count != decoder_prompt_token_count
        || actual.decoder_max_new_tokens != decoder_max_new_tokens
        || actual.final_prefill_prompt_token_count != final_prefill_prompt_token_count
        || actual.final_prefill_max_new_tokens != final_prefill_max_new_tokens
        || actual.primed_labels != primed_labels
        || actual.final_prefill_labels != final_prefill_labels
        || &actual.settlement != settlement
    {
        return Err("GPU fixed corpus case body drifted from its reviewed topology".to_owned());
    }
    Ok(())
}

fn descriptor_document(descriptor: &GpuFixedCorpusDescriptorV1) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("GPU fixed corpus descriptor serializes");
    document.push('\n');
    document
}

/// Parses exactly one canonical GPU fixed-corpus V1 descriptor.
pub fn parse_gpu_fixed_corpus_descriptor(document: &str) -> Result<GpuFixedCorpusCase, String> {
    let descriptor = serde_json::from_str::<GpuFixedCorpusDescriptorV1>(document)
        .map_err(|error| format!("GPU fixed corpus descriptor JSON is invalid: {error}"))?;
    if document != descriptor_document(&descriptor) {
        return Err("GPU fixed corpus descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_case()
}

/// Serializes a previously validated fixed corpus case as canonical JSON.
#[must_use]
pub fn serialize_gpu_fixed_corpus_descriptor(case: &GpuFixedCorpusCase) -> String {
    descriptor_document(&GpuFixedCorpusDescriptorV1 {
        format: DESCRIPTOR_FORMAT.to_owned(),
        format_version: DESCRIPTOR_FORMAT_VERSION,
        trace_kind: DESCRIPTOR_TRACE_KIND.to_owned(),
        case_id: case.case_id.clone(),
        concurrency: case.concurrency,
        decoder_prompt_token_count: case.decoder_prompt_token_count,
        decoder_max_new_tokens: case.decoder_max_new_tokens,
        final_prefill_prompt_token_count: case.final_prefill_prompt_token_count,
        final_prefill_max_new_tokens: case.final_prefill_max_new_tokens,
        primed_labels: case.primed_labels.clone(),
        final_prefill_labels: case.final_prefill_labels.clone(),
        settlement: case.settlement.clone(),
    })
}

const CORPUS_DOCUMENTS_V1: [(&str, &str); 3] = [
    (
        "gpu-fixed-v1/c5-kv15-to17-mixed-greedy.json",
        include_str!("../corpus/output-routing/gpu-fixed-v1/c5-kv15-to17-mixed-greedy.json"),
    ),
    (
        "gpu-fixed-v1/c8-mixed-deferred-cancel.json",
        include_str!("../corpus/output-routing/gpu-fixed-v1/c8-mixed-deferred-cancel.json"),
    ),
    (
        "gpu-fixed-v1/c8-mixed-greedy-commit-assembly-failure.json",
        include_str!(
            "../corpus/output-routing/gpu-fixed-v1/c8-mixed-greedy-commit-assembly-failure.json"
        ),
    ),
];

/// Loads the complete, duplicate-free committed C03-B source corpus.
#[must_use]
pub fn gpu_fixed_corpus() -> Vec<GpuFixedCorpusCase> {
    let mut case_ids = BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V1.len());
    for (path, document) in CORPUS_DOCUMENTS_V1 {
        let case = parse_gpu_fixed_corpus_descriptor(document)
            .unwrap_or_else(|error| panic!("{path}: GPU fixed corpus is invalid: {error}"));
        assert!(
            case_ids.insert(case.case_id.clone()),
            "{path}: GPU fixed corpus repeats case_id {:?}",
            case.case_id
        );
        corpus.push(case);
    }
    corpus
}
