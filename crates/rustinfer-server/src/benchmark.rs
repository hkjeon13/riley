//! Allocation-bounded, pretokenized native benchmark execution.
//!
//! The public trace deliberately contains token hashes rather than token IDs.
//! Token IDs remain process-local long enough to validate scheduler commit
//! ordering and are hashed only after the measured batch interval ends.

use std::error;
use std::fmt;

use rustinfer_scheduler::IterationMetricSample;
#[cfg(any(feature = "cuda", test))]
use rustinfer_scheduler::{IterationUpdates, RequestFinishReason, RequestId as SchedulerRequestId};

/// SHA-256 of consecutive unsigned token IDs encoded as little-endian `u32`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TokenIdSha256([u8; 32]);

impl TokenIdSha256 {
    /// Returns the fixed-width digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encodes the digest as 64 lowercase hexadecimal characters.
    #[must_use]
    pub fn to_hex(self) -> String {
        use fmt::Write as _;

        let mut output = String::with_capacity(64);
        for byte in self.0 {
            // Writing two characters into a pre-sized String is infallible.
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}

/// One caller-owned, already-tokenized benchmark prompt.
///
/// This type intentionally has no `Debug` implementation because token IDs
/// can reconstruct prompt content.
pub struct PretokenizedBenchmarkRequest {
    prompt_token_ids: Vec<u32>,
}

impl PretokenizedBenchmarkRequest {
    /// Creates one request whose stable identity is its position in the batch.
    #[must_use]
    pub fn new(prompt_token_ids: Vec<u32>) -> Self {
        Self { prompt_token_ids }
    }

    /// Borrows the exact model-token input.
    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }
}

/// Sanitized benchmark failure with no prompt or generated-token diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeBenchmarkError {
    /// A public request or executor bound is inconsistent.
    InvalidConfiguration {
        /// Stable field name.
        field: &'static str,
        /// Non-sensitive invariant description.
        reason: &'static str,
    },
    /// A bounded host allocation could not be reserved.
    HostAllocation {
        /// Non-sensitive buffer name.
        resource: &'static str,
        /// Exact requested element capacity.
        requested_elements: usize,
    },
    /// A monotonic timestamp could not be represented or regressed.
    InvalidClock,
    /// CUDA or fixed executor preparation failed.
    Preparation,
    /// Scheduler admission or planning failed.
    Scheduler,
    /// Runtime execution failed.
    Execution,
    /// Greedy logits processing failed.
    Sampling,
    /// Scheduler commit or post-commit observation failed.
    Commit,
    /// The executor is terminal and accepts no further trials.
    Terminal,
    /// Explicit cleanup did not prove complete ownership release.
    Cleanup {
        /// Number of independently observed cleanup failures.
        failure_count: usize,
    },
}

impl fmt::Display for NativeBenchmarkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(formatter, "invalid native benchmark {field}: {reason}")
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "cannot reserve native benchmark {resource} capacity {requested_elements}"
            ),
            Self::InvalidClock => formatter.write_str("native benchmark monotonic clock failed"),
            Self::Preparation => formatter.write_str("native benchmark preparation failed"),
            Self::Scheduler => formatter.write_str("native benchmark scheduler failed"),
            Self::Execution => formatter.write_str("native benchmark execution failed"),
            Self::Sampling => formatter.write_str("native benchmark sampling failed"),
            Self::Commit => formatter.write_str("native benchmark commit failed"),
            Self::Terminal => formatter.write_str("native benchmark executor is terminal"),
            Self::Cleanup { failure_count } => write!(
                formatter,
                "native benchmark cleanup failed in {failure_count} operation(s)"
            ),
        }
    }
}

impl error::Error for NativeBenchmarkError {}

/// Result returned by native benchmark preparation, execution, and cleanup.
pub type NativeBenchmarkResult<T> = Result<T, NativeBenchmarkError>;

/// Hashed, timing-only observation for one successful request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBenchmarkRequestTrace {
    input_index: usize,
    prompt_token_ids_sha256: TokenIdSha256,
    generated_token_ids_sha256: TokenIdSha256,
    prompt_tokens: usize,
    requested_output_tokens: usize,
    arrival_ns: u64,
    committed_token_ns: Vec<u64>,
    terminal_observed_ns: u64,
}

impl NativeBenchmarkRequestTrace {
    /// Zero-based position in the submitted pretokenized batch.
    #[must_use]
    pub const fn input_index(&self) -> usize {
        self.input_index
    }

    /// Hash of the exact input token IDs.
    #[must_use]
    pub const fn prompt_token_ids_sha256(&self) -> TokenIdSha256 {
        self.prompt_token_ids_sha256
    }

    /// Hash of token IDs observed after authoritative scheduler commits.
    #[must_use]
    pub const fn generated_token_ids_sha256(&self) -> TokenIdSha256 {
        self.generated_token_ids_sha256
    }

    /// Exact pretokenized input length.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Canonical fixed output length requested for the batch.
    #[must_use]
    pub const fn requested_output_tokens(&self) -> usize {
        self.requested_output_tokens
    }

    /// Number of committed generated tokens.
    #[must_use]
    pub fn generated_tokens(&self) -> usize {
        self.committed_token_ns.len()
    }

    /// Request arrival offset from the executor's monotonic epoch.
    #[must_use]
    pub const fn arrival_ns(&self) -> u64 {
        self.arrival_ns
    }

    /// Post-commit observation offsets in generation order.
    #[must_use]
    pub fn committed_token_ns(&self) -> &[u64] {
        &self.committed_token_ns
    }

    /// Offset at which the terminal completion became observable.
    #[must_use]
    pub const fn terminal_observed_ns(&self) -> u64 {
        self.terminal_observed_ns
    }
}

/// Timing and bounded-work facts for one successfully committed iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeBenchmarkIterationTrace {
    iteration_id: u64,
    plan_started_ns: u64,
    plan_finished_ns: u64,
    execution_started_ns: u64,
    execution_finished_ns: u64,
    commit_started_ns: u64,
    commit_observed_ns: u64,
    metric: IterationMetricSample,
    gpu_timing_validity: NativeGpuTimingValidity,
}

/// Whether scheduler GPU fields came from valid device-side timing events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeGpuTimingValidity {
    /// Same-stream timing events bracketed dispatch through logits D2H.
    MeasuredCudaEvents,
    /// No dispatch-boundary CUDA events were available in the runtime bridge.
    ///
    /// `IterationMetricSample::{gpu_execution_ns,gpu_idle_gap_ns}` are zero
    /// placeholders and must not be reported as measurements. Host monotonic
    /// execution boundaries remain valid.
    UnavailableExecutionBridgeBoundary,
}

impl NativeBenchmarkIterationTrace {
    /// Scheduler iteration identity.
    #[must_use]
    pub const fn iteration_id(self) -> u64 {
        self.iteration_id
    }

    /// Monotonic planning start offset.
    #[must_use]
    pub const fn plan_started_ns(self) -> u64 {
        self.plan_started_ns
    }

    /// Monotonic planning return offset.
    #[must_use]
    pub const fn plan_finished_ns(self) -> u64 {
        self.plan_finished_ns
    }

    /// Monotonic runtime bridge start offset.
    #[must_use]
    pub const fn execution_started_ns(self) -> u64 {
        self.execution_started_ns
    }

    /// Monotonic runtime bridge return offset.
    #[must_use]
    pub const fn execution_finished_ns(self) -> u64 {
        self.execution_finished_ns
    }

    /// Monotonic scheduler commit-call offset.
    #[must_use]
    pub const fn commit_started_ns(self) -> u64 {
        self.commit_started_ns
    }

    /// Monotonic offset captured only after commit returned successfully.
    #[must_use]
    pub const fn commit_observed_ns(self) -> u64 {
        self.commit_observed_ns
    }

    /// Exact scheduler metric sample for this committed iteration.
    ///
    /// Consult [`Self::gpu_timing_validity`] before consuming its GPU fields.
    #[must_use]
    pub const fn metric(self) -> IterationMetricSample {
        self.metric
    }

    /// Validity of the GPU fields contained in [`Self::metric`].
    #[must_use]
    pub const fn gpu_timing_validity(self) -> NativeGpuTimingValidity {
        self.gpu_timing_validity
    }
}

/// Complete timing-only trace for one fixed-concurrency batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBenchmarkTrialTrace {
    batch_started_ns: u64,
    batch_finished_ns: u64,
    requests: Vec<NativeBenchmarkRequestTrace>,
    iterations: Vec<NativeBenchmarkIterationTrace>,
}

impl NativeBenchmarkTrialTrace {
    /// Monotonic batch start immediately before the first admission.
    #[must_use]
    pub const fn batch_started_ns(&self) -> u64 {
        self.batch_started_ns
    }

    /// Monotonic batch end after every terminal observation was processed.
    #[must_use]
    pub const fn batch_finished_ns(&self) -> u64 {
        self.batch_finished_ns
    }

    /// Request observations in input order.
    #[must_use]
    pub fn requests(&self) -> &[NativeBenchmarkRequestTrace] {
        &self.requests
    }

    /// Successfully committed iterations in execution order.
    #[must_use]
    pub fn iterations(&self) -> &[NativeBenchmarkIterationTrace] {
        &self.iterations
    }
}

/// Opaque, capacity-checked trial ready for timed admission.
///
/// This type has no `Debug` implementation because it still owns prompt IDs.
#[cfg(any(feature = "cuda", test))]
pub struct PreparedNativeBenchmarkTrial {
    output_tokens: usize,
    requests: Vec<PreparedRequest>,
    iterations: Vec<NativeBenchmarkIterationTrace>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    iteration_limit: usize,
}

#[cfg(any(feature = "cuda", test))]
struct PreparedRequest {
    input_index: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    prompt_token_ids: Vec<u32>,
    prompt_hash: TokenIdSha256,
    prompt_tokens: usize,
    scheduler_id: Option<SchedulerRequestId>,
    arrival_ns: Option<u64>,
    generated_token_ids: Vec<u32>,
    committed_token_ns: Vec<u64>,
    terminal_observed_ns: Option<u64>,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct TrialBounds {
    max_requests: usize,
    max_sequence_tokens: usize,
    addressable_tokens: usize,
}

#[cfg(any(feature = "cuda", test))]
impl PreparedNativeBenchmarkTrial {
    #[allow(clippy::too_many_lines)]
    fn prepare(
        requests: Vec<PretokenizedBenchmarkRequest>,
        output_tokens: usize,
        bounds: TrialBounds,
    ) -> NativeBenchmarkResult<Self> {
        if requests.is_empty() {
            return Err(invalid("requests", "must not be empty"));
        }
        if requests.len() > bounds.max_requests {
            return Err(invalid("requests", "exceed prepared concurrency"));
        }
        if output_tokens == 0 {
            return Err(invalid("output_tokens", "must be greater than zero"));
        }
        if bounds.addressable_tokens == 0 {
            return Err(invalid("addressable_tokens", "must be greater than zero"));
        }

        let mut iteration_capacity = 0_usize;
        let mut prepared = reserve_vec(requests.len(), "request observations")?;
        for (input_index, request) in requests.into_iter().enumerate() {
            if request.prompt_token_ids.is_empty() {
                return Err(invalid("prompt_token_ids", "must not be empty"));
            }
            if request.prompt_token_ids.iter().any(|&token_id| {
                usize::try_from(token_id)
                    .ok()
                    .is_none_or(|index| index >= bounds.addressable_tokens)
            }) {
                return Err(invalid(
                    "prompt_token_ids",
                    "contain a token outside the addressable vocabulary",
                ));
            }
            let sequence_tokens = request
                .prompt_token_ids
                .len()
                .checked_add(output_tokens)
                .ok_or_else(|| invalid("sequence_tokens", "overflow usize"))?;
            if sequence_tokens > bounds.max_sequence_tokens {
                return Err(invalid(
                    "sequence_tokens",
                    "exceed the prepared context bound",
                ));
            }
            iteration_capacity = iteration_capacity
                .checked_add(request.prompt_token_ids.len())
                .and_then(|value| value.checked_add(output_tokens - 1))
                .ok_or_else(|| invalid("iteration_capacity", "overflow usize"))?;
            let prompt_hash = hash_token_ids(&request.prompt_token_ids)?;
            prepared.push(PreparedRequest {
                input_index,
                prompt_tokens: request.prompt_token_ids.len(),
                prompt_token_ids: request.prompt_token_ids,
                prompt_hash,
                scheduler_id: None,
                arrival_ns: None,
                generated_token_ids: reserve_vec(output_tokens, "generated token IDs")?,
                committed_token_ns: reserve_vec(output_tokens, "token timestamps")?,
                terminal_observed_ns: None,
            });
        }
        let iterations = reserve_vec(iteration_capacity, "iteration observations")?;
        Ok(Self {
            output_tokens,
            requests: prepared,
            iterations,
            iteration_limit: iteration_capacity,
        })
    }

    fn bind_submission(
        &mut self,
        input_index: usize,
        scheduler_id: SchedulerRequestId,
        arrival_ns: u64,
    ) -> NativeBenchmarkResult<()> {
        let request = self
            .requests
            .get_mut(input_index)
            .ok_or(NativeBenchmarkError::Commit)?;
        if request.scheduler_id.is_some() || request.arrival_ns.is_some() {
            return Err(NativeBenchmarkError::Commit);
        }
        request.scheduler_id = Some(scheduler_id);
        request.arrival_ns = Some(arrival_ns);
        Ok(())
    }

    fn request_mut(
        &mut self,
        scheduler_id: SchedulerRequestId,
    ) -> NativeBenchmarkResult<&mut PreparedRequest> {
        self.requests
            .iter_mut()
            .find(|request| request.scheduler_id == Some(scheduler_id))
            .ok_or(NativeBenchmarkError::Commit)
    }

    fn observe_updates(
        &mut self,
        updates: &IterationUpdates,
        observed_ns: u64,
    ) -> NativeBenchmarkResult<()> {
        if !updates.settlement_failures().is_empty() || updates.iteration_metric().is_none() {
            return Err(NativeBenchmarkError::Commit);
        }
        let output_tokens = self.output_tokens;
        for token in updates.token_events() {
            let request = self.request_mut(token.request_id())?;
            let arrival_ns = request.arrival_ns.ok_or(NativeBenchmarkError::Commit)?;
            if request.terminal_observed_ns.is_some()
                || token.generated_index() != request.generated_token_ids.len()
                || request.generated_token_ids.len() >= output_tokens
                || observed_ns < arrival_ns
                || request
                    .committed_token_ns
                    .last()
                    .is_some_and(|&previous| observed_ns < previous)
            {
                return Err(NativeBenchmarkError::Commit);
            }
            request.generated_token_ids.push(token.token_id());
            request.committed_token_ns.push(observed_ns);
        }
        for completion in updates.completions() {
            let request = self.request_mut(completion.request_id())?;
            let arrival_ns = request.arrival_ns.ok_or(NativeBenchmarkError::Commit)?;
            if completion.reason() != RequestFinishReason::Length
                || request.terminal_observed_ns.is_some()
                || request.generated_token_ids.len() != output_tokens
                || request.generated_token_ids != completion.generated_token_ids()
                || request.committed_token_ns.len() != output_tokens
                || observed_ns < arrival_ns
                || completion.completed_at_ns() > observed_ns
            {
                return Err(NativeBenchmarkError::Commit);
            }
            request.terminal_observed_ns = Some(observed_ns);
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.requests
            .iter()
            .all(|request| request.terminal_observed_ns.is_some())
    }

    fn finish(
        self,
        batch_started_ns: u64,
        batch_finished_ns: u64,
    ) -> NativeBenchmarkResult<NativeBenchmarkTrialTrace> {
        if !self.is_complete() || batch_finished_ns < batch_started_ns {
            return Err(NativeBenchmarkError::Commit);
        }
        let mut traces = reserve_vec(self.requests.len(), "public request traces")?;
        for request in self.requests {
            let arrival_ns = request.arrival_ns.ok_or(NativeBenchmarkError::Commit)?;
            let terminal_observed_ns = request
                .terminal_observed_ns
                .ok_or(NativeBenchmarkError::Commit)?;
            if request.generated_token_ids.len() != self.output_tokens
                || request.committed_token_ns.len() != self.output_tokens
                || arrival_ns < batch_started_ns
                || terminal_observed_ns > batch_finished_ns
            {
                return Err(NativeBenchmarkError::Commit);
            }
            // Hashing intentionally occurs after `batch_finished_ns` was
            // captured, matching the canonical lane's timing boundary.
            let generated_hash = hash_token_ids(&request.generated_token_ids)?;
            traces.push(NativeBenchmarkRequestTrace {
                input_index: request.input_index,
                prompt_token_ids_sha256: request.prompt_hash,
                generated_token_ids_sha256: generated_hash,
                prompt_tokens: request.prompt_tokens,
                requested_output_tokens: self.output_tokens,
                arrival_ns,
                committed_token_ns: request.committed_token_ns,
                terminal_observed_ns,
            });
        }
        Ok(NativeBenchmarkTrialTrace {
            batch_started_ns,
            batch_finished_ns,
            requests: traces,
            iterations: self.iterations,
        })
    }
}

#[cfg(any(feature = "cuda", test))]
fn hash_token_ids(token_ids: &[u32]) -> NativeBenchmarkResult<TokenIdSha256> {
    let mut hasher = FixedSha256::new();
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes())?;
    }
    Ok(TokenIdSha256(hasher.finalize()?))
}

#[cfg(any(feature = "cuda", test))]
struct FixedSha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    message_bytes: u64,
}

#[cfg(any(feature = "cuda", test))]
impl FixedSha256 {
    const INITIAL_STATE: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    const fn new() -> Self {
        Self {
            state: Self::INITIAL_STATE,
            block: [0; 64],
            block_len: 0,
            message_bytes: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) -> NativeBenchmarkResult<()> {
        self.message_bytes = self
            .message_bytes
            .checked_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| invalid("token_hash", "input exceeds SHA-256 byte counter"))?;
        for &byte in bytes {
            self.block[self.block_len] = byte;
            self.block_len += 1;
            if self.block_len == self.block.len() {
                self.compress();
                self.block_len = 0;
            }
        }
        Ok(())
    }

    fn finalize(mut self) -> NativeBenchmarkResult<[u8; 32]> {
        let message_bits = self
            .message_bytes
            .checked_mul(8)
            .ok_or_else(|| invalid("token_hash", "input exceeds SHA-256 bit counter"))?;
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            self.compress();
            self.block_len = 0;
        }
        self.block[self.block_len..56].fill(0);
        self.block[56..].copy_from_slice(&message_bits.to_be_bytes());
        self.compress();

        let mut digest = [0_u8; 32];
        for (destination, word) in digest.chunks_exact_mut(4).zip(self.state) {
            destination.copy_from_slice(&word.to_be_bytes());
        }
        Ok(digest)
    }

    #[allow(clippy::many_single_char_names)]
    fn compress(&mut self) {
        let mut words = [0_u32; 64];
        for (word, bytes) in words[..16].iter_mut().zip(self.block.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (&constant, &word) in Self::ROUND_CONSTANTS.iter().zip(&words) {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(constant)
                .wrapping_add(word);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn invalid(field: &'static str, reason: &'static str) -> NativeBenchmarkError {
    NativeBenchmarkError::InvalidConfiguration { field, reason }
}

#[cfg(any(feature = "cuda", test))]
fn reserve_vec<T>(capacity: usize, resource: &'static str) -> NativeBenchmarkResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| NativeBenchmarkError::HostAllocation {
            resource,
            requested_elements: capacity,
        })?;
    Ok(values)
}

#[cfg(feature = "cuda")]
mod cuda_executor {
    use std::convert::Infallible;
    use std::time::Instant;

    use rustinfer_model::LoadedModel;
    use rustinfer_runtime::llama::{PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig};
    use rustinfer_runtime::sampling::{
        SamplingParams, SamplingRng, SamplingWorkspace, TokenConstraints,
    };
    use rustinfer_runtime::{CudaContext, CudaRuntime, CudaStream};
    use rustinfer_scheduler::{
        ExecutionAbort, IterationPlan, LlamaIterationCudaTimer, OverloadPolicy, RequestDescriptor,
        RequestState, SampledIterationToken, Scheduler, SchedulerConfig,
        execute_llama_iteration_timed,
    };

    use super::{
        NativeBenchmarkError, NativeBenchmarkIterationTrace, NativeBenchmarkResult,
        NativeBenchmarkTrialTrace, NativeGpuTimingValidity, PreparedNativeBenchmarkTrial,
        PretokenizedBenchmarkRequest, TrialBounds, invalid, reserve_vec,
    };

    const GREEDY: SamplingParams = SamplingParams {
        temperature: 0.0,
        top_k: None,
        top_p: None,
        repetition_penalty: 1.0,
    };

    /// Cold CUDA, scheduler, and fixed-batch preparation settings.
    #[derive(Clone, Debug)]
    pub struct NativeBenchmarkConfig {
        /// Visible CUDA device ordinal.
        pub device_ordinal: u32,
        /// Bounded continuous scheduler policy.
        pub scheduler: SchedulerConfig,
        /// Fixed-M runtime metadata and forward policy.
        pub executor: PreparedLlamaBatchExecutorConfig,
    }

    /// Proof that explicit close released scheduler and CUDA allocations.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct NativeBenchmarkCleanupReport {
        scheduler_completions: usize,
        device_live_bytes: u64,
        device_live_allocations: u64,
        pinned_host_live_bytes: u64,
        pinned_host_live_allocations: u64,
    }

    impl NativeBenchmarkCleanupReport {
        /// Terminal notifications reclaimed during final scheduler close.
        #[must_use]
        pub const fn scheduler_completions(self) -> usize {
            self.scheduler_completions
        }

        /// Device bytes remaining after executor close.
        #[must_use]
        pub const fn device_live_bytes(self) -> u64 {
            self.device_live_bytes
        }

        /// Device allocations remaining after executor close.
        #[must_use]
        pub const fn device_live_allocations(self) -> u64 {
            self.device_live_allocations
        }

        /// Pinned-host bytes remaining after executor close.
        #[must_use]
        pub const fn pinned_host_live_bytes(self) -> u64 {
            self.pinned_host_live_bytes
        }

        /// Pinned-host allocations remaining after executor close.
        #[must_use]
        pub const fn pinned_host_live_allocations(self) -> u64 {
            self.pinned_host_live_allocations
        }

        /// Whether every tracked CUDA allocation was released.
        #[must_use]
        pub const fn allocations_are_zero(self) -> bool {
            self.device_live_bytes == 0
                && self.device_live_allocations == 0
                && self.pinned_host_live_bytes == 0
                && self.pinned_host_live_allocations == 0
        }
    }

    /// Synchronous owner of one native continuous-batching benchmark lane.
    ///
    /// The owner is intentionally neither an HTTP backend nor an asynchronous
    /// service. Pretokenized requests are admitted together, then the caller's
    /// thread exclusively drives Scheduler -> CUDA -> commit until completion.
    pub struct NativeBenchmarkExecutor {
        _model: LoadedModel,
        scheduler: Option<Scheduler>,
        context: Option<CudaContext>,
        stream: Option<CudaStream>,
        executor: Option<PreparedLlamaBatchExecutor>,
        timer: Option<LlamaIterationCudaTimer>,
        sampling: SamplingWorkspace,
        allowed_tokens: Vec<bool>,
        addressable_tokens: usize,
        model_sequence_tokens: usize,
        samples: Vec<SampledIterationToken>,
        clock: Instant,
        terminal: bool,
    }

    impl NativeBenchmarkExecutor {
        /// Initializes CUDA and prepares every persistent execution scratch.
        ///
        /// The loaded model remains owned by this executor so a complete lane
        /// has one explicit lifetime and cleanup boundary.
        ///
        /// # Errors
        ///
        /// Returns a sanitized configuration, allocation, CUDA, runtime, or
        /// scheduler preparation error. No model token is included in it.
        #[allow(clippy::too_many_lines)]
        pub fn prepare(
            model: LoadedModel,
            config: NativeBenchmarkConfig,
        ) -> NativeBenchmarkResult<Self> {
            validate_composition(&model, &config)?;
            let vocabulary_size = model.spec().embedding().vocabulary_size();
            let addressable_tokens = model.tokenizer().addressable_token_count();
            if addressable_tokens == 0 || addressable_tokens > vocabulary_size {
                return Err(invalid(
                    "addressable_tokens",
                    "must be within the model vocabulary",
                ));
            }
            let sampling = SamplingWorkspace::new(vocabulary_size)
                .map_err(|_| NativeBenchmarkError::Preparation)?;
            let mut allowed_tokens = reserve_vec(vocabulary_size, "allowed-token mask")?;
            allowed_tokens.resize(vocabulary_size, false);
            allowed_tokens[..addressable_tokens].fill(true);
            let output_capacity = config.executor.metadata().max_output_slots();
            let samples = reserve_vec(output_capacity, "sample staging")?;

            let runtime =
                CudaRuntime::initialize().map_err(|_| NativeBenchmarkError::Preparation)?;
            let device = runtime
                .device(config.device_ordinal)
                .map_err(|_| NativeBenchmarkError::Preparation)?;
            let context = device
                .create_context()
                .map_err(|_| NativeBenchmarkError::Preparation)?;
            let mut stream = match context.create_stream() {
                Ok(stream) => stream,
                Err(_) => {
                    let _ = context.close();
                    return Err(NativeBenchmarkError::Preparation);
                }
            };
            let executor = match PreparedLlamaBatchExecutor::prepare(
                &model,
                &context,
                &mut stream,
                config.executor,
            ) {
                Ok(executor) => executor,
                Err(_) => {
                    let _ = stream.synchronize();
                    let _ = stream.close();
                    let _ = context.close();
                    return Err(NativeBenchmarkError::Preparation);
                }
            };
            let scheduler = match Scheduler::new(config.scheduler, executor.kv_layout()) {
                Ok(scheduler) => scheduler,
                Err(_) => {
                    let _ = executor.close();
                    let _ = stream.synchronize();
                    let _ = stream.close();
                    let _ = context.close();
                    return Err(NativeBenchmarkError::Preparation);
                }
            };
            let timer = match LlamaIterationCudaTimer::prepare(&context) {
                Ok(timer) => timer,
                Err(_) => {
                    let _ = scheduler.close(0, None);
                    let _ = executor.close();
                    let _ = stream.synchronize();
                    let _ = stream.close();
                    let _ = context.close();
                    return Err(NativeBenchmarkError::Preparation);
                }
            };
            let model_sequence_tokens = model.spec().max_sequence_length();
            Ok(Self {
                _model: model,
                scheduler: Some(scheduler),
                context: Some(context),
                stream: Some(stream),
                executor: Some(executor),
                timer: Some(timer),
                sampling,
                allowed_tokens,
                addressable_tokens,
                model_sequence_tokens,
                samples,
                clock: Instant::now(),
                terminal: false,
            })
        }

        /// Validates inputs, hashes prompts, and reserves complete trace storage.
        ///
        /// This method performs no CUDA model execution. Call it before the
        /// surrounding observability sampler and timed [`Self::run_trial`].
        ///
        /// # Errors
        ///
        /// Returns for a terminal/busy executor, invalid pretokenized input,
        /// context overflow, or a bounded host reservation failure.
        pub fn prepare_trial(
            &self,
            requests: Vec<PretokenizedBenchmarkRequest>,
            output_tokens: usize,
        ) -> NativeBenchmarkResult<PreparedNativeBenchmarkTrial> {
            if self.terminal {
                return Err(NativeBenchmarkError::Terminal);
            }
            let scheduler = self
                .scheduler
                .as_ref()
                .ok_or(NativeBenchmarkError::Terminal)?;
            ensure_scheduler_idle(scheduler)?;
            PreparedNativeBenchmarkTrial::prepare(
                requests,
                output_tokens,
                TrialBounds {
                    max_requests: scheduler.config().max_active_sequences,
                    max_sequence_tokens: scheduler
                        .config()
                        .max_sequence_tokens
                        .min(self.model_sequence_tokens),
                    addressable_tokens: self.addressable_tokens,
                },
            )
        }

        /// Runs one prepared fixed-concurrency trial on the caller's thread.
        ///
        /// Every sampled token is treated as ordinary, including model EOS
        /// IDs. The scheduler receives `stop=false` and terminates requests
        /// only after the exact prepared output length commits.
        ///
        /// # Errors
        ///
        /// Returns a sanitized admission, execution, sampling, commit, clock,
        /// or cleanup error. Any failure makes this executor terminal; callers
        /// must still invoke [`Self::close`].
        pub fn run_trial(
            &mut self,
            mut trial: PreparedNativeBenchmarkTrial,
        ) -> NativeBenchmarkResult<NativeBenchmarkTrialTrace> {
            if self.terminal {
                return Err(NativeBenchmarkError::Terminal);
            }
            self.timer
                .as_mut()
                .ok_or(NativeBenchmarkError::Terminal)?
                .reset_window();
            let measured = self.run_trial_inner(&mut trial);
            match measured {
                Ok((batch_started_ns, batch_finished_ns)) => {
                    let trace = trial.finish(batch_started_ns, batch_finished_ns);
                    if trace.is_err() {
                        self.terminal = true;
                        self.contain_failure();
                    }
                    trace
                }
                Err(error) => {
                    self.terminal = true;
                    self.contain_failure();
                    Err(error)
                }
            }
        }

        #[allow(clippy::too_many_lines)]
        fn run_trial_inner(
            &mut self,
            trial: &mut PreparedNativeBenchmarkTrial,
        ) -> NativeBenchmarkResult<(u64, u64)> {
            {
                let scheduler = self
                    .scheduler
                    .as_ref()
                    .ok_or(NativeBenchmarkError::Terminal)?;
                ensure_scheduler_idle(scheduler)?;
            }
            let batch_started_ns = self.now_ns()?;
            for input_index in 0..trial.requests.len() {
                let arrival_ns = self.now_ns()?;
                let prompt_token_ids =
                    std::mem::take(&mut trial.requests[input_index].prompt_token_ids);
                let output_tokens = trial.output_tokens;
                let submission = self
                    .scheduler_mut()?
                    .submit(
                        RequestDescriptor::new(prompt_token_ids, output_tokens),
                        arrival_ns,
                    )
                    .map_err(|_| NativeBenchmarkError::Scheduler)?;
                if submission.state() != RequestState::Admitted {
                    return Err(NativeBenchmarkError::Scheduler);
                }
                trial.bind_submission(input_index, submission.request_id(), arrival_ns)?;
            }

            while !trial.is_complete() {
                let plan_started_ns = self.now_ns()?;
                let planning = self
                    .scheduler_mut()?
                    .plan_iteration(plan_started_ns)
                    .map_err(|_| NativeBenchmarkError::Scheduler)?;
                let plan_finished_ns = self.now_ns()?;
                let (plan, planning_completions) = planning.into_parts();
                if !planning_completions.is_empty() {
                    return Err(NativeBenchmarkError::Scheduler);
                }
                let plan = plan.ok_or(NativeBenchmarkError::Scheduler)?;
                let iteration_id = plan.iteration_id();

                let execution_started_ns = self.now_ns()?;
                let (downloaded, timing) = {
                    let executor = self
                        .executor
                        .as_mut()
                        .ok_or(NativeBenchmarkError::Terminal)?;
                    let stream = self.stream.as_mut().ok_or(NativeBenchmarkError::Terminal)?;
                    let timer = self.timer.as_mut().ok_or(NativeBenchmarkError::Terminal)?;
                    match execute_llama_iteration_timed(&plan, executor, stream, timer) {
                        Ok(measured) => measured,
                        Err(failure) => {
                            if let Some((failed_iteration, abort)) = failure.abort_data() {
                                let now_ns = self.now_ns()?;
                                self.scheduler_mut()?
                                    .abort_iteration(failed_iteration, abort, now_ns)
                                    .map_err(|_| NativeBenchmarkError::Execution)?;
                            }
                            return Err(NativeBenchmarkError::Execution);
                        }
                    }
                };
                let execution_finished_ns = self.now_ns()?;
                if let Err(error) = self.sample_iteration(&plan, &downloaded) {
                    let (failed_iteration, abort) = downloaded.abort_data();
                    let now_ns = self.now_ns()?;
                    self.scheduler_mut()?
                        .abort_iteration(failed_iteration, abort, now_ns)
                        .map_err(|_| NativeBenchmarkError::Execution)?;
                    return Err(error);
                }
                let result = match downloaded.into_result(&self.samples, timing) {
                    Ok(result) => result,
                    Err(failure) => {
                        let (failed_iteration, abort) = failure.abort_data();
                        let now_ns = self.now_ns()?;
                        self.scheduler_mut()?
                            .abort_iteration(failed_iteration, abort, now_ns)
                            .map_err(|_| NativeBenchmarkError::Commit)?;
                        return Err(NativeBenchmarkError::Commit);
                    }
                };
                let commit_started_ns = self.now_ns()?;
                let updates = match self
                    .scheduler_mut()?
                    .complete_iteration(&result, commit_started_ns)
                {
                    Ok(updates) => updates,
                    Err(_) => {
                        if self
                            .scheduler
                            .as_ref()
                            .and_then(Scheduler::inflight_iteration_id)
                            == Some(result.iteration_id())
                        {
                            let now_ns = self.now_ns()?;
                            self.scheduler_mut()?
                                .abort_iteration(
                                    result.iteration_id(),
                                    ExecutionAbort::DeviceQuiescedMutationUnknown,
                                    now_ns,
                                )
                                .map_err(|_| NativeBenchmarkError::Commit)?;
                        }
                        return Err(NativeBenchmarkError::Commit);
                    }
                };
                let commit_observed_ns = self.now_ns()?;
                let metric = updates
                    .iteration_metric()
                    .ok_or(NativeBenchmarkError::Commit)?;
                trial.observe_updates(&updates, commit_observed_ns)?;
                if trial.iterations.len() >= trial.iteration_limit {
                    return Err(NativeBenchmarkError::Commit);
                }
                trial.iterations.push(NativeBenchmarkIterationTrace {
                    iteration_id: iteration_id.get(),
                    plan_started_ns,
                    plan_finished_ns,
                    execution_started_ns,
                    execution_finished_ns,
                    commit_started_ns,
                    commit_observed_ns,
                    metric,
                    gpu_timing_validity: NativeGpuTimingValidity::MeasuredCudaEvents,
                });
            }
            let batch_finished_ns = self.now_ns()?;
            ensure_scheduler_idle(
                self.scheduler
                    .as_ref()
                    .ok_or(NativeBenchmarkError::Terminal)?,
            )?;
            Ok((batch_started_ns, batch_finished_ns))
        }

        fn sample_iteration(
            &mut self,
            plan: &IterationPlan,
            downloaded: &rustinfer_scheduler::DownloadedLlamaIteration,
        ) -> NativeBenchmarkResult<()> {
            self.samples.clear();
            for &slot in plan.output_slots() {
                let logits = downloaded
                    .logits_for_slot(slot)
                    .map_err(|_| NativeBenchmarkError::Sampling)?;
                let constraints = if self.addressable_tokens == self.allowed_tokens.len() {
                    TokenConstraints::AllowAll
                } else {
                    TokenConstraints::AllowedMask(&self.allowed_tokens)
                };
                let distribution = self
                    .sampling
                    .process_bf16_native(logits, constraints, &[], GREEDY)
                    .map_err(|_| NativeBenchmarkError::Sampling)?;
                let sample = distribution
                    .sample(&mut GreedyMustNotDraw)
                    .map_err(|never| match never {})?;
                // Canonical ignore_eos=true: no EOS or stop token is masked,
                // and no sampled ID can terminate before the fixed length.
                self.samples
                    .push(SampledIterationToken::new(sample.token_id(), false));
            }
            Ok(())
        }

        fn scheduler_mut(&mut self) -> NativeBenchmarkResult<&mut Scheduler> {
            self.scheduler
                .as_mut()
                .ok_or(NativeBenchmarkError::Terminal)
        }

        fn now_ns(&self) -> NativeBenchmarkResult<u64> {
            u64::try_from(self.clock.elapsed().as_nanos())
                .map_err(|_| NativeBenchmarkError::InvalidClock)
        }

        fn cleanup_now_ns(&self) -> u64 {
            u64::try_from(self.clock.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }

        fn contain_failure(&mut self) {
            let now_ns = self.cleanup_now_ns();
            let inflight = self
                .scheduler
                .as_ref()
                .and_then(Scheduler::inflight_iteration_id);
            if let Some(iteration_id) = inflight {
                let quiesced = self
                    .stream
                    .as_mut()
                    .is_some_and(|stream| stream.synchronize().is_ok());
                if quiesced {
                    if let Some(scheduler) = self.scheduler.as_mut() {
                        let _ = scheduler.abort_iteration(
                            iteration_id,
                            ExecutionAbort::DeviceQuiescedMutationUnknown,
                            now_ns,
                        );
                    }
                }
            }
            if let Some(scheduler) = self.scheduler.as_mut() {
                scheduler.begin_shutdown();
                if scheduler.inflight_iteration_id().is_none() {
                    let _ = scheduler.shutdown(now_ns);
                }
            }
        }

        /// Explicitly settles the scheduler and releases all CUDA ownership.
        ///
        /// Every independent cleanup operation is attempted. Success proves
        /// the context's tracked device and pinned-host allocations reached
        /// zero before the stream and context were closed.
        ///
        /// # Errors
        ///
        /// Returns the number of cleanup failures if complete release cannot
        /// be proven. Drop remains best-effort and is not benchmark evidence.
        pub fn close(mut self) -> NativeBenchmarkResult<NativeBenchmarkCleanupReport> {
            let (report, failure_count) = self.close_resources();
            if failure_count == 0 {
                Ok(report)
            } else {
                Err(NativeBenchmarkError::Cleanup { failure_count })
            }
        }

        fn close_resources(&mut self) -> (NativeBenchmarkCleanupReport, usize) {
            let mut report = NativeBenchmarkCleanupReport::default();
            let mut failure_count = 0_usize;
            let now_ns = self.cleanup_now_ns();
            let mut inflight_abort = None;
            if self
                .scheduler
                .as_ref()
                .and_then(Scheduler::inflight_iteration_id)
                .is_some()
            {
                match self.stream.as_mut().map(CudaStream::synchronize) {
                    Some(Ok(())) => {
                        inflight_abort = Some(ExecutionAbort::DeviceQuiescedMutationUnknown);
                    }
                    Some(Err(_)) | None => failure_count += 1,
                }
            }
            if let Some(scheduler) = self.scheduler.take() {
                match scheduler.close(now_ns, inflight_abort) {
                    Ok(output) => {
                        report.scheduler_completions = output.completions().len();
                        if !output.settlement_failures().is_empty() {
                            failure_count += 1;
                        }
                    }
                    Err(failure) => {
                        failure_count += 1;
                        let (_, scheduler, _) = failure.into_parts();
                        self.scheduler = Some(scheduler);
                    }
                }
            }
            if let Some(executor) = self.executor.take() {
                if executor.close().is_err() {
                    failure_count += 1;
                }
            }
            if let Some(timer) = self.timer.take() {
                if timer.close().is_err() {
                    failure_count += 1;
                }
            }
            if let Some(context) = self.context.as_ref() {
                match context.allocation_stats() {
                    Ok(stats) => {
                        report.device_live_bytes = stats.device_live_bytes();
                        report.device_live_allocations = stats.device_live_allocations();
                        report.pinned_host_live_bytes = stats.pinned_host_live_bytes();
                        report.pinned_host_live_allocations = stats.pinned_host_live_allocations();
                        if !stats.is_zero() {
                            failure_count += 1;
                        }
                    }
                    Err(_) => failure_count += 1,
                }
            }
            if let Some(mut stream) = self.stream.take() {
                if stream.synchronize().is_err() {
                    failure_count += 1;
                }
                if stream.close().is_err() {
                    failure_count += 1;
                }
            }
            if let Some(context) = self.context.take() {
                if context.close().is_err() {
                    failure_count += 1;
                }
            }
            (report, failure_count)
        }
    }

    impl Drop for NativeBenchmarkExecutor {
        fn drop(&mut self) {
            let _ = self.close_resources();
        }
    }

    struct GreedyMustNotDraw;

    impl SamplingRng for GreedyMustNotDraw {
        type Error = Infallible;

        fn next_u32(&mut self) -> Result<u32, Self::Error> {
            unreachable!("temperature-zero benchmark sampling must not draw RNG")
        }
    }

    fn validate_composition(
        model: &LoadedModel,
        config: &NativeBenchmarkConfig,
    ) -> NativeBenchmarkResult<()> {
        let scheduler = config.scheduler.clone().validate().map_err(|_| {
            invalid(
                "scheduler",
                "configuration violates bounded scheduler invariants",
            )
        })?;
        let metadata = config.executor.metadata();
        if scheduler.overload_policy != OverloadPolicy::RejectImmediately {
            return Err(invalid(
                "overload_policy",
                "must reject rather than queue canonical concurrency",
            ));
        }
        if scheduler.max_active_sequences > metadata.max_rows()
            || scheduler.max_active_sequences > metadata.max_output_slots()
        {
            return Err(invalid(
                "max_active_sequences",
                "exceed fixed executor row or output capacity",
            ));
        }
        if scheduler.iteration_token_budget > metadata.max_input_tokens()
            || scheduler.max_prefill_chunk_tokens > metadata.max_input_tokens()
        {
            return Err(invalid(
                "iteration_token_budget",
                "exceeds fixed executor input capacity",
            ));
        }
        if scheduler.max_sequence_tokens > model.spec().max_sequence_length() {
            return Err(invalid(
                "max_sequence_tokens",
                "exceeds model context capacity",
            ));
        }
        Ok(())
    }

    fn ensure_scheduler_idle(scheduler: &Scheduler) -> NativeBenchmarkResult<()> {
        let pool = scheduler.pool_stats();
        if !scheduler.is_accepting()
            || scheduler.active_sequence_count() != 0
            || scheduler.waiting_request_count() != 0
            || scheduler.promised_kv_blocks() != 0
            || scheduler.inflight_iteration_id().is_some()
            || scheduler.pending_completion_count() != 0
            || pool.allocated_block_count() != 0
        {
            return Err(NativeBenchmarkError::Terminal);
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use cuda_executor::{
    NativeBenchmarkCleanupReport, NativeBenchmarkConfig, NativeBenchmarkExecutor,
};

#[cfg(test)]
mod tests {
    use super::{
        NativeBenchmarkError, PreparedNativeBenchmarkTrial, PretokenizedBenchmarkRequest,
        TrialBounds, hash_token_ids,
    };
    use rustinfer_runtime::paged_kv::KvLayout;
    use rustinfer_scheduler::{
        IterationOutput, IterationResult, OutputSlot, OverloadPolicy, RequestDescriptor, Scheduler,
        SchedulerConfig,
    };

    fn bounds() -> TrialBounds {
        TrialBounds {
            max_requests: 2,
            max_sequence_tokens: 16,
            addressable_tokens: 8,
        }
    }

    fn scheduler() -> Scheduler {
        Scheduler::new(
            SchedulerConfig {
                max_waiting_requests: 2,
                max_waiting_prompt_tokens: 16,
                max_active_sequences: 2,
                max_sequence_tokens: 16,
                iteration_token_budget: 4,
                max_prefill_chunk_tokens: 4,
                aging_threshold_ns: 100,
                overload_policy: OverloadPolicy::RejectImmediately,
                admission_timeout_ns: None,
                max_promised_kv_blocks: 8,
                metrics_window_samples: 8,
            },
            KvLayout::checked(1, 8, 1, 64).expect("CPU test KV layout"),
        )
        .expect("CPU test scheduler")
    }

    #[test]
    fn token_hash_uses_exact_little_endian_u32_bytes() {
        let actual = hash_token_ids(&[1, 0x0102_0304]).expect("bounded token hash");
        assert_eq!(
            actual.to_hex(),
            "cd48bd37d30164fffc22b2f6db0a7f71d0997f5eda1950950efbb3c554684576"
        );
        assert_eq!(
            hash_token_ids(&[]).expect("empty token hash").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hash_token_ids(&(0..32).collect::<Vec<_>>())
                .expect("multi-block token hash")
                .to_hex(),
            "afbc67011b6f94a508935ad8edcbdd3c9b56c4db336f8d3847a8a1815183828f"
        );
        assert_eq!(actual.to_hex().len(), 64);
        assert!(
            actual
                .to_hex()
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn preparation_rejects_unbounded_or_unaddressable_inputs() {
        assert!(matches!(
            PreparedNativeBenchmarkTrial::prepare(Vec::new(), 1, bounds()),
            Err(NativeBenchmarkError::InvalidConfiguration {
                field: "requests",
                ..
            })
        ));
        assert!(matches!(
            PreparedNativeBenchmarkTrial::prepare(
                vec![PretokenizedBenchmarkRequest::new(vec![8])],
                1,
                bounds(),
            ),
            Err(NativeBenchmarkError::InvalidConfiguration {
                field: "prompt_token_ids",
                ..
            })
        ));
        assert!(matches!(
            PreparedNativeBenchmarkTrial::prepare(
                vec![PretokenizedBenchmarkRequest::new(vec![1; 16])],
                1,
                bounds(),
            ),
            Err(NativeBenchmarkError::InvalidConfiguration {
                field: "sequence_tokens",
                ..
            })
        ));
    }

    #[test]
    fn only_post_commit_tokens_are_observed_and_eos_is_ordinary() {
        let mut trial = PreparedNativeBenchmarkTrial::prepare(
            vec![PretokenizedBenchmarkRequest::new(vec![1])],
            2,
            bounds(),
        )
        .expect("prepared CPU observer");
        let mut scheduler = scheduler();
        let submission = scheduler
            .submit(RequestDescriptor::new(vec![1], 2), 0)
            .expect("admit pretokenized request");
        trial
            .bind_submission(0, submission.request_id(), 0)
            .expect("bind request identity");

        let first_plan = scheduler
            .plan_iteration(1)
            .expect("first plan")
            .into_parts()
            .0
            .expect("prefill plan");
        let first_result = IterationResult::new(
            first_plan.iteration_id(),
            vec![IterationOutput::new(OutputSlot::new(0), 2, false)],
            11,
            12,
        )
        .expect("first result");
        assert!(trial.requests[0].generated_token_ids.is_empty());
        let first_updates = scheduler
            .complete_iteration(&first_result, 2)
            .expect("first authoritative commit");
        assert!(first_updates.iteration_metric().is_some());
        assert!(trial.requests[0].generated_token_ids.is_empty());
        trial
            .observe_updates(&first_updates, 3)
            .expect("observe only after commit");
        assert_eq!(trial.requests[0].generated_token_ids, [2]);
        assert!(!trial.is_complete());

        let second_plan = scheduler
            .plan_iteration(4)
            .expect("second plan")
            .into_parts()
            .0
            .expect("decode plan");
        let second_result = IterationResult::new(
            second_plan.iteration_id(),
            vec![IterationOutput::new(OutputSlot::new(0), 2, false)],
            13,
            14,
        )
        .expect("second result");
        let second_updates = scheduler
            .complete_iteration(&second_result, 5)
            .expect("second authoritative commit");
        trial
            .observe_updates(&second_updates, 6)
            .expect("observe terminal commit");

        // Token 2 stands in for a model EOS ID. Because every scheduler output
        // carried stop=false, it remains an ordinary committed ID twice and
        // termination is solely the exact requested length.
        let trace = trial.finish(0, 7).expect("hashed public trace");
        assert_eq!(trace.requests().len(), 1);
        assert_eq!(trace.requests()[0].generated_tokens(), 2);
        assert_eq!(trace.requests()[0].committed_token_ns(), [3, 6]);
        assert_eq!(
            trace.requests()[0].generated_token_ids_sha256(),
            hash_token_ids(&[2, 2]).expect("generated token hash")
        );
        assert_eq!(
            scheduler
                .close(8, None)
                .expect("exact CPU scheduler cleanup")
                .completions()
                .len(),
            0
        );
    }

    #[test]
    fn observer_rejects_regressing_monotonic_time() {
        let mut trial = PreparedNativeBenchmarkTrial::prepare(
            vec![PretokenizedBenchmarkRequest::new(vec![1])],
            1,
            bounds(),
        )
        .expect("prepared CPU observer");
        let mut scheduler = scheduler();
        let submission = scheduler
            .submit(RequestDescriptor::new(vec![1], 1), 10)
            .expect("admit request");
        trial
            .bind_submission(0, submission.request_id(), 10)
            .expect("bind identity");
        let plan = scheduler
            .plan_iteration(11)
            .expect("plan")
            .into_parts()
            .0
            .expect("work");
        let result = IterationResult::new(
            plan.iteration_id(),
            vec![IterationOutput::new(OutputSlot::new(0), 3, false)],
            0,
            0,
        )
        .expect("result");
        let updates = scheduler.complete_iteration(&result, 12).expect("commit");
        assert_eq!(
            trial.observe_updates(&updates, 9),
            Err(NativeBenchmarkError::Commit)
        );
        scheduler.close(13, None).expect("scheduler cleanup");
    }
}
