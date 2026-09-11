//! Allocation-free CPU logits processing and categorical sampling.
//!
//! The sampler owns vocabulary-sized scratch and performs no resizing after
//! construction. Its normative processing order is constraints, one
//! repetition-penalty application per unique history token, temperature,
//! top-k, top-p, F64 normalization, then token-ID-ascending categorical
//! traversal. BF16 input bytes use native-endian storage, matching the CUDA
//! download boundary.

use std::cmp::Ordering;
use std::error;
use std::fmt;

/// Stable identifier for the normative PR11 logits-processing pipeline.
pub const SAMPLING_PIPELINE_ID: &str =
    "bf16ne-constraints-unique-repetition-temperature-top-k-top-p-f64-v1";
/// Version of [`SAMPLING_PIPELINE_ID`].
pub const SAMPLING_PIPELINE_VERSION: u32 = 1;
/// Stable identifier for categorical traversal and U32-to-uniform conversion.
pub const SAMPLING_ALGORITHM_ID: &str = "u32-midpoint-token-id-ascending-categorical-v1";

const BF16_BYTES: usize = 2;
const U32_UNIFORM_SCALE: f64 = 1.0 / 4_294_967_296.0;

/// Request-local random-word source used by stochastic categorical sampling.
///
/// A successful stochastic sample invokes [`Self::next_u32`] exactly once.
/// Temperature-zero greedy sampling never invokes it. The request RNG owns
/// snapshot, restore, fork, and algorithm metadata; this narrow trait keeps
/// those policies separate from logits processing.
pub trait SamplingRng {
    /// Error returned by the request-local RNG.
    type Error;

    /// Produces the next word from the request-local stream.
    ///
    /// # Errors
    ///
    /// Returns the implementation's request-local RNG failure without
    /// substituting or drawing an additional word.
    fn next_u32(&mut self) -> Result<u32, Self::Error>;
}

/// Token mask applied before every other logits transform.
#[derive(Clone, Copy, Debug, Default)]
pub enum TokenConstraints<'a> {
    /// Every vocabulary entry is eligible.
    #[default]
    AllowAll,
    /// A vocabulary-sized mask in which `true` means eligible.
    AllowedMask(&'a [bool]),
}

/// Parameters for the normative PR11 sampling pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// Zero selects deterministic greedy decoding; positive values scale logits.
    pub temperature: f32,
    /// Maximum number of highest-logit tokens retained, or no limit.
    pub top_k: Option<usize>,
    /// Smallest descending-probability prefix whose mass reaches this value.
    pub top_p: Option<f64>,
    /// Standard sign-aware penalty applied once per unique history token.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    /// Validates request parameters against a fixed vocabulary size.
    ///
    /// This cold-path check lets generation reject a request before model
    /// execution or RNG mutation. [`SamplingWorkspace`] repeats the check at
    /// its public processing boundary so direct callers receive the same
    /// contract.
    ///
    /// # Errors
    ///
    /// Returns for an empty or non-U32-addressable vocabulary, a negative or
    /// non-finite temperature, top-k outside `1..=vocabulary_size`, top-p
    /// outside `(0, 1]`, or a non-positive/non-finite repetition penalty.
    pub fn validate(self, vocabulary_size: usize) -> Result<(), SamplingError> {
        if vocabulary_size == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        if u32::try_from(vocabulary_size - 1).is_err() {
            return Err(SamplingError::VocabularyTooLarge { vocabulary_size });
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature {
                value: self.temperature,
            });
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 || top_k > vocabulary_size {
                return Err(SamplingError::InvalidTopK {
                    value: top_k,
                    vocabulary_size,
                });
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(SamplingError::InvalidTopP { value: top_p });
            }
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(SamplingError::InvalidRepetitionPenalty {
                value: self.repetition_penalty,
            });
        }
        Ok(())
    }
}

/// Checked configuration, input, or degenerate-distribution failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SamplingError {
    /// A sampler cannot be built for an empty vocabulary.
    EmptyVocabulary,
    /// The last vocabulary index cannot be represented as a U32 token ID.
    VocabularyTooLarge {
        /// Requested vocabulary size.
        vocabulary_size: usize,
    },
    /// A vocabulary-sized host scratch allocation failed.
    HostAllocation {
        /// Scratch buffer being allocated.
        resource: &'static str,
        /// Requested element count.
        requested_elements: usize,
    },
    /// BF16 byte length does not match the workspace vocabulary.
    LogitsLength {
        /// Required byte length.
        expected_bytes: usize,
        /// Supplied byte length.
        actual_bytes: usize,
    },
    /// The allowed-token mask does not match the workspace vocabulary.
    ConstraintLength {
        /// Required mask length.
        expected_tokens: usize,
        /// Supplied mask length.
        actual_tokens: usize,
    },
    /// A downloaded raw logit was NaN or infinite.
    NonFiniteLogit {
        /// Token owning the invalid value.
        token_id: u32,
        /// Raw BF16 bits, before F32 expansion.
        bf16_bits: u16,
    },
    /// A history token is outside the vocabulary.
    HistoryTokenOutOfRange {
        /// Invalid token ID.
        token_id: u32,
        /// Workspace vocabulary size.
        vocabulary_size: usize,
    },
    /// Temperature was negative, NaN, or infinite.
    InvalidTemperature {
        /// Invalid value.
        value: f32,
    },
    /// Top-k was zero or larger than the vocabulary.
    InvalidTopK {
        /// Invalid value.
        value: usize,
        /// Workspace vocabulary size.
        vocabulary_size: usize,
    },
    /// Top-p was outside `(0, 1]` or was not finite.
    InvalidTopP {
        /// Invalid value.
        value: f64,
    },
    /// Repetition penalty was non-positive, NaN, or infinite.
    InvalidRepetitionPenalty {
        /// Invalid value.
        value: f32,
    },
    /// Token constraints removed the complete vocabulary.
    AllMasked,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVocabulary => formatter.write_str("sampling vocabulary must not be empty"),
            Self::VocabularyTooLarge { vocabulary_size } => write!(
                formatter,
                "sampling vocabulary size {vocabulary_size} cannot use U32 token IDs"
            ),
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} elements for sampling {resource}"
            ),
            Self::LogitsLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "sampling logits have {actual_bytes} bytes, expected {expected_bytes} native-endian BF16 bytes"
            ),
            Self::ConstraintLength {
                expected_tokens,
                actual_tokens,
            } => write!(
                formatter,
                "sampling constraint mask has {actual_tokens} tokens, expected {expected_tokens}"
            ),
            Self::NonFiniteLogit {
                token_id,
                bf16_bits,
            } => write!(
                formatter,
                "sampling logit for token {token_id} is non-finite BF16 0x{bf16_bits:04x}"
            ),
            Self::HistoryTokenOutOfRange {
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "history token {token_id} is outside sampling vocabulary {vocabulary_size}"
            ),
            Self::InvalidTemperature { value } => {
                write!(
                    formatter,
                    "sampling temperature must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidTopK {
                value,
                vocabulary_size,
            } => write!(
                formatter,
                "sampling top-k must be in 1..={vocabulary_size}, got {value}"
            ),
            Self::InvalidTopP { value } => {
                write!(
                    formatter,
                    "sampling top-p must be finite and in (0, 1], got {value}"
                )
            }
            Self::InvalidRepetitionPenalty { value } => write!(
                formatter,
                "sampling repetition penalty must be finite and positive, got {value}"
            ),
            Self::AllMasked => formatter.write_str("sampling constraints masked every token"),
        }
    }
}

impl error::Error for SamplingError {}

/// Failure from the one-call processing-and-sampling convenience API.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingFailure<RngError> {
    /// Logits processing failed before an RNG word was requested.
    Processing(SamplingError),
    /// Request-local random-word generation failed.
    Rng(RngError),
}

impl<RngError: fmt::Display> fmt::Display for SamplingFailure<RngError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Processing(source) => write!(formatter, "sampling processing failed: {source}"),
            Self::Rng(source) => write!(formatter, "sampling RNG failed: {source}"),
        }
    }
}

impl<RngError> error::Error for SamplingFailure<RngError>
where
    RngError: error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Processing(source) => Some(source),
            Self::Rng(source) => Some(source),
        }
    }
}

/// One categorical selection and its normalized probability information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingResult {
    token_id: u32,
    probability: f64,
    log_probability: f64,
}

impl SamplingResult {
    /// Selected vocabulary token ID.
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    /// Selected token probability from stable F64 normalization.
    #[must_use]
    pub const fn probability(self) -> f64 {
        self.probability
    }

    /// Selected token log-probability from stable F64 log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> f64 {
        self.log_probability
    }

    /// Plan-level optional F32 token log-probability contract.
    ///
    /// The CPU sampler always computes this value, so it is always `Some`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn token_logprob(self) -> Option<f32> {
        Some(self.log_probability as f32)
    }
}

/// Vocabulary-sized reusable scratch for CPU logits processing.
#[derive(Debug)]
pub struct SamplingWorkspace {
    processed_logits: Vec<f64>,
    probabilities: Vec<f64>,
    log_probabilities: Vec<f64>,
    candidate_ids: Vec<usize>,
    history_seen: Vec<bool>,
}

impl SamplingWorkspace {
    /// Allocates every vocabulary-sized buffer required by the hot path.
    ///
    /// # Errors
    ///
    /// Returns for an empty or non-U32-addressable vocabulary, or when a host
    /// scratch allocation cannot be reserved.
    pub fn new(vocabulary_size: usize) -> Result<Self, SamplingError> {
        if vocabulary_size == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        if u32::try_from(vocabulary_size - 1).is_err() {
            return Err(SamplingError::VocabularyTooLarge { vocabulary_size });
        }

        Ok(Self {
            processed_logits: allocate_filled(
                vocabulary_size,
                f64::NEG_INFINITY,
                "processed logits",
            )?,
            probabilities: allocate_filled(vocabulary_size, 0.0, "probabilities")?,
            log_probabilities: allocate_filled(
                vocabulary_size,
                f64::NEG_INFINITY,
                "log probabilities",
            )?,
            candidate_ids: allocate_filled(vocabulary_size, 0, "candidate token IDs")?,
            history_seen: allocate_filled(vocabulary_size, false, "history markers")?,
        })
    }

    /// Fixed vocabulary size shared by input and every scratch buffer.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.processed_logits.len()
    }

    /// Processes one native-endian BF16 vocabulary row without drawing RNG.
    ///
    /// The returned distribution borrows the workspace, preventing another
    /// processing pass from overwriting its probability and log-probability
    /// views. All input and parameter errors are reported here, before a
    /// caller can enter categorical sampling.
    ///
    /// # Errors
    ///
    /// Returns for invalid parameters, malformed or non-finite logits, a mask
    /// length mismatch, out-of-vocabulary history, or an all-masked row.
    pub fn process_bf16_native(
        &mut self,
        logits_bf16_native: &[u8],
        constraints: TokenConstraints<'_>,
        unique_history: &[u32],
        params: SamplingParams,
    ) -> Result<ProcessedDistribution<'_>, SamplingError> {
        self.validate_inputs(logits_bf16_native, constraints, unique_history, params)?;

        self.probabilities.fill(0.0);
        self.log_probabilities.fill(f64::NEG_INFINITY);
        self.processed_logits.fill(f64::NEG_INFINITY);
        self.history_seen.fill(false);

        let allowed_mask = match constraints {
            TokenConstraints::AllowAll => None,
            TokenConstraints::AllowedMask(mask) => Some(mask),
        };
        let mut candidate_count = 0;
        for (token_id, bytes) in logits_bf16_native.chunks_exact(BF16_BYTES).enumerate() {
            if allowed_mask.is_none_or(|mask| mask[token_id]) {
                let value = decode_finite_bf16(bytes);
                self.processed_logits[token_id] = f64::from(value);
                self.candidate_ids[candidate_count] = token_id;
                candidate_count += 1;
            }
        }
        if candidate_count == 0 {
            return Err(SamplingError::AllMasked);
        }

        let penalty = f64::from(params.repetition_penalty);
        for &token_id in unique_history {
            let index =
                usize::try_from(token_id).map_err(|_| SamplingError::HistoryTokenOutOfRange {
                    token_id,
                    vocabulary_size: self.vocabulary_size(),
                })?;
            if !self.history_seen[index] {
                self.history_seen[index] = true;
                let logit = &mut self.processed_logits[index];
                if logit.is_finite() {
                    if *logit > 0.0 {
                        *logit /= penalty;
                    } else if *logit < 0.0 {
                        *logit *= penalty;
                    }
                }
            }
        }

        if params.temperature == 0.0 {
            return Ok(self.prepare_greedy(candidate_count));
        }

        let inverse_temperature = 1.0 / f64::from(params.temperature);
        for &token_id in &self.candidate_ids[..candidate_count] {
            self.processed_logits[token_id] *= inverse_temperature;
        }

        let logits = &self.processed_logits;
        self.candidate_ids[..candidate_count].sort_unstable_by(|left, right| {
            finite_cmp(logits[*right], logits[*left]).then_with(|| left.cmp(right))
        });

        let mut support_size = params.top_k.unwrap_or(candidate_count).min(candidate_count);
        for &token_id in &self.candidate_ids[support_size..candidate_count] {
            self.processed_logits[token_id] = f64::NEG_INFINITY;
        }

        if let Some(top_p) = params.top_p {
            support_size = self.apply_top_p(support_size, top_p);
        }
        self.normalize(support_size);

        Ok(self.distribution(false, support_size, None))
    }

    /// Processes logits and immediately selects one token.
    ///
    /// This wrapper preserves the pre-draw error rule: processing completes
    /// successfully before [`SamplingRng::next_u32`] can be invoked.
    ///
    /// # Errors
    ///
    /// Returns [`SamplingFailure::Processing`] for any logits-processing error
    /// and [`SamplingFailure::Rng`] when stochastic word generation fails.
    pub fn sample_bf16_native<R: SamplingRng>(
        &mut self,
        logits_bf16_native: &[u8],
        constraints: TokenConstraints<'_>,
        unique_history: &[u32],
        params: SamplingParams,
        rng: &mut R,
    ) -> Result<SamplingResult, SamplingFailure<R::Error>> {
        let distribution = self
            .process_bf16_native(logits_bf16_native, constraints, unique_history, params)
            .map_err(SamplingFailure::Processing)?;
        distribution.sample(rng).map_err(SamplingFailure::Rng)
    }

    fn validate_inputs(
        &self,
        logits_bf16_native: &[u8],
        constraints: TokenConstraints<'_>,
        unique_history: &[u32],
        params: SamplingParams,
    ) -> Result<(), SamplingError> {
        params.validate(self.vocabulary_size())?;

        let expected_bytes = self.vocabulary_size().checked_mul(BF16_BYTES).ok_or(
            SamplingError::VocabularyTooLarge {
                vocabulary_size: self.vocabulary_size(),
            },
        )?;
        if logits_bf16_native.len() != expected_bytes {
            return Err(SamplingError::LogitsLength {
                expected_bytes,
                actual_bytes: logits_bf16_native.len(),
            });
        }
        if let TokenConstraints::AllowedMask(mask) = constraints {
            if mask.len() != self.vocabulary_size() {
                return Err(SamplingError::ConstraintLength {
                    expected_tokens: self.vocabulary_size(),
                    actual_tokens: mask.len(),
                });
            }
        }
        for &token_id in unique_history {
            if usize::try_from(token_id).map_or(true, |index| index >= self.vocabulary_size()) {
                return Err(SamplingError::HistoryTokenOutOfRange {
                    token_id,
                    vocabulary_size: self.vocabulary_size(),
                });
            }
        }
        for (token_id, bytes) in logits_bf16_native.chunks_exact(BF16_BYTES).enumerate() {
            let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            if !value.is_finite() {
                return Err(SamplingError::NonFiniteLogit {
                    token_id: u32::try_from(token_id).map_err(|_| {
                        SamplingError::VocabularyTooLarge {
                            vocabulary_size: self.vocabulary_size(),
                        }
                    })?,
                    bf16_bits: bits,
                });
            }
        }
        Ok(())
    }

    fn prepare_greedy(&mut self, candidate_count: usize) -> ProcessedDistribution<'_> {
        let mut winner = self.candidate_ids[0];
        for &token_id in &self.candidate_ids[1..candidate_count] {
            let candidate = self.processed_logits[token_id];
            let current = self.processed_logits[winner];
            let ordering = finite_cmp(candidate, current);
            if ordering.is_gt() || (ordering.is_eq() && token_id < winner) {
                winner = token_id;
            }
        }
        for &token_id in &self.candidate_ids[..candidate_count] {
            if token_id != winner {
                self.processed_logits[token_id] = f64::NEG_INFINITY;
            }
        }
        self.probabilities[winner] = 1.0;
        self.log_probabilities[winner] = 0.0;
        let winner = u32::try_from(winner).unwrap_or(u32::MAX);
        self.distribution(true, 1, Some(winner))
    }

    fn apply_top_p(&mut self, support_size: usize, top_p: f64) -> usize {
        if top_p >= 1.0 {
            return support_size;
        }
        let candidates = &self.candidate_ids[..support_size];
        let maximum = self.processed_logits[candidates[0]];
        let total = candidates.iter().fold(0.0, |sum, &token_id| {
            sum + (self.processed_logits[token_id] - maximum).exp()
        });
        let threshold = top_p * total;
        let mut cumulative = 0.0;
        let mut retained = 0;
        for &token_id in candidates {
            cumulative += (self.processed_logits[token_id] - maximum).exp();
            retained += 1;
            if cumulative >= threshold {
                break;
            }
        }
        for &token_id in &self.candidate_ids[retained..support_size] {
            self.processed_logits[token_id] = f64::NEG_INFINITY;
        }
        retained
    }

    fn normalize(&mut self, support_size: usize) {
        let candidates = &self.candidate_ids[..support_size];
        let maximum = candidates
            .iter()
            .map(|&token_id| self.processed_logits[token_id])
            .fold(f64::NEG_INFINITY, f64::max);
        let normalization_sum = candidates.iter().fold(0.0, |sum, &token_id| {
            sum + (self.processed_logits[token_id] - maximum).exp()
        });
        let log_shifted_sum = normalization_sum.ln();
        for &token_id in candidates {
            let log_probability = (self.processed_logits[token_id] - maximum) - log_shifted_sum;
            self.log_probabilities[token_id] = log_probability;
            self.probabilities[token_id] = log_probability.exp();
        }
    }

    fn distribution(
        &self,
        greedy: bool,
        support_size: usize,
        greedy_token: Option<u32>,
    ) -> ProcessedDistribution<'_> {
        ProcessedDistribution {
            processed_logits: &self.processed_logits,
            probabilities: &self.probabilities,
            log_probabilities: &self.log_probabilities,
            greedy,
            support_size,
            greedy_token,
        }
    }
}

/// Borrowed, fully validated distribution produced by [`SamplingWorkspace`].
#[derive(Clone, Copy, Debug)]
pub struct ProcessedDistribution<'a> {
    processed_logits: &'a [f64],
    probabilities: &'a [f64],
    log_probabilities: &'a [f64],
    greedy: bool,
    support_size: usize,
    greedy_token: Option<u32>,
}

impl ProcessedDistribution<'_> {
    /// Final temperature-scaled logits, using negative infinity for masks.
    #[must_use]
    pub const fn processed_logits(&self) -> &[f64] {
        self.processed_logits
    }

    /// F64 normalized probabilities in ascending token-ID layout.
    #[must_use]
    pub const fn probabilities(&self) -> &[f64] {
        self.probabilities
    }

    /// Stable F64 log-probabilities; masked entries are negative infinity.
    #[must_use]
    pub const fn log_probabilities(&self) -> &[f64] {
        self.log_probabilities
    }

    /// Probability for one token, or `None` when it is out of vocabulary.
    #[must_use]
    pub fn probability(&self, token_id: u32) -> Option<f64> {
        usize::try_from(token_id)
            .ok()
            .and_then(|index| self.probabilities.get(index).copied())
    }

    /// Log-probability for one token, or `None` when it is out of vocabulary.
    #[must_use]
    pub fn log_probability(&self, token_id: u32) -> Option<f64> {
        usize::try_from(token_id)
            .ok()
            .and_then(|index| self.log_probabilities.get(index).copied())
    }

    /// Number of tokens retained by greedy, top-k, and top-p filtering.
    #[must_use]
    pub const fn support_size(&self) -> usize {
        self.support_size
    }

    /// Whether temperature zero selected deterministic greedy decoding.
    #[must_use]
    pub const fn is_greedy(&self) -> bool {
        self.greedy
    }

    /// Selects one token from this already validated distribution.
    ///
    /// Stochastic sampling converts its single U32 word to the open-interval
    /// midpoint `(word + 0.5) * 2^-32` and traverses tokens in ascending ID
    /// order. Greedy sampling consumes no word.
    ///
    /// # Errors
    ///
    /// Returns the request RNG error when its sole stochastic draw fails.
    pub fn sample<R: SamplingRng>(&self, rng: &mut R) -> Result<SamplingResult, R::Error> {
        if let Some(token_id) = self.greedy_token {
            return Ok(self.result(token_id));
        }

        let word = rng.next_u32()?;
        let uniform = (f64::from(word) + 0.5) * U32_UNIFORM_SCALE;
        let mut cumulative = 0.0;
        let mut last_supported = 0_u32;
        for (token_id, &probability) in self.probabilities.iter().enumerate() {
            if probability > 0.0 {
                last_supported = u32::try_from(token_id).unwrap_or(u32::MAX);
                cumulative += probability;
                if uniform < cumulative {
                    return Ok(self.result(last_supported));
                }
            }
        }
        Ok(self.result(last_supported))
    }

    fn result(&self, token_id: u32) -> SamplingResult {
        let index = usize::try_from(token_id).unwrap_or(0);
        SamplingResult {
            token_id,
            probability: self.probabilities[index],
            log_probability: self.log_probabilities[index],
        }
    }
}

fn allocate_filled<T: Clone>(
    element_count: usize,
    value: T,
    resource: &'static str,
) -> Result<Vec<T>, SamplingError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(element_count)
        .map_err(|_| SamplingError::HostAllocation {
            resource,
            requested_elements: element_count,
        })?;
    output.resize(element_count, value);
    Ok(output)
}

fn decode_finite_bf16(bytes: &[u8]) -> f32 {
    let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn finite_cmp(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{
        ProcessedDistribution, SAMPLING_ALGORITHM_ID, SAMPLING_PIPELINE_ID,
        SAMPLING_PIPELINE_VERSION, SamplingError, SamplingFailure, SamplingParams, SamplingRng,
        SamplingWorkspace, TokenConstraints,
    };
    use std::convert::Infallible;

    #[derive(Debug, Eq, PartialEq)]
    struct ScriptedRng<'a> {
        words: &'a [u32],
        cursor: usize,
        calls: usize,
        fail: bool,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ScriptedError;

    impl SamplingRng for ScriptedRng<'_> {
        type Error = ScriptedError;

        fn next_u32(&mut self) -> Result<u32, Self::Error> {
            self.calls += 1;
            if self.fail {
                return Err(ScriptedError);
            }
            let word = self.words.get(self.cursor).copied().unwrap_or(0);
            self.cursor += 1;
            Ok(word)
        }
    }

    struct NeverRng {
        calls: usize,
    }

    impl SamplingRng for NeverRng {
        type Error = Infallible;

        fn next_u32(&mut self) -> Result<u32, Self::Error> {
            self.calls += 1;
            unreachable!("greedy sampling must not request RNG")
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let high = value.to_bits().to_ne_bytes();
            let bits = if cfg!(target_endian = "little") {
                [high[2], high[3]]
            } else {
                [high[0], high[1]]
            };
            bytes.extend_from_slice(&u16::from_ne_bytes(bits).to_ne_bytes());
        }
        bytes
    }

    fn process<'a>(
        workspace: &'a mut SamplingWorkspace,
        values: &[f32],
        params: SamplingParams,
    ) -> ProcessedDistribution<'a> {
        workspace
            .process_bf16_native(&bf16_bytes(values), TokenConstraints::AllowAll, &[], params)
            .expect("test distribution must process")
    }

    #[test]
    fn stable_algorithm_identifiers_are_explicit() {
        assert_eq!(SAMPLING_PIPELINE_VERSION, 1);
        assert!(SAMPLING_PIPELINE_ID.contains("unique-repetition"));
        assert!(SAMPLING_ALGORITHM_ID.contains("u32-midpoint"));
        assert!(SAMPLING_ALGORITHM_ID.contains("token-id-ascending"));
    }

    #[test]
    fn empty_vocabulary_and_input_shapes_are_rejected() {
        assert_eq!(
            SamplingWorkspace::new(0).unwrap_err(),
            SamplingError::EmptyVocabulary
        );
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        assert_eq!(
            workspace
                .process_bf16_native(
                    &[],
                    TokenConstraints::AllowAll,
                    &[],
                    SamplingParams::default()
                )
                .unwrap_err(),
            SamplingError::LogitsLength {
                expected_bytes: 4,
                actual_bytes: 0,
            }
        );
        assert_eq!(
            workspace
                .process_bf16_native(
                    &bf16_bytes(&[0.0, 1.0]),
                    TokenConstraints::AllowedMask(&[true]),
                    &[],
                    SamplingParams::default(),
                )
                .unwrap_err(),
            SamplingError::ConstraintLength {
                expected_tokens: 2,
                actual_tokens: 1,
            }
        );
    }

    #[test]
    fn nonfinite_bf16_and_history_bounds_are_rejected() {
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(matches!(
                workspace.process_bf16_native(
                    &bf16_bytes(&[0.0, value]),
                    TokenConstraints::AllowAll,
                    &[],
                    SamplingParams::default(),
                ),
                Err(SamplingError::NonFiniteLogit { token_id: 1, .. })
            ));
        }
        assert_eq!(
            workspace
                .process_bf16_native(
                    &bf16_bytes(&[0.0, 1.0]),
                    TokenConstraints::AllowAll,
                    &[2],
                    SamplingParams::default(),
                )
                .unwrap_err(),
            SamplingError::HistoryTokenOutOfRange {
                token_id: 2,
                vocabulary_size: 2,
            }
        );
    }

    #[test]
    fn constraints_precede_one_penalty_per_unique_history_token() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let distribution = workspace
            .process_bf16_native(
                &bf16_bytes(&[4.0, 3.0, -2.0, 100.0]),
                TokenConstraints::AllowedMask(&[true, true, true, false]),
                &[0, 0, 2, 2],
                SamplingParams {
                    repetition_penalty: 2.0,
                    ..SamplingParams::default()
                },
            )
            .unwrap();
        assert_eq!(distribution.processed_logits()[0], 2.0);
        assert_eq!(distribution.processed_logits()[1], 3.0);
        assert_eq!(distribution.processed_logits()[2], -4.0);
        assert_eq!(distribution.processed_logits()[3], f64::NEG_INFINITY);
    }

    #[test]
    fn all_masked_is_an_error() {
        let mut workspace = SamplingWorkspace::new(3).unwrap();
        assert_eq!(
            workspace
                .process_bf16_native(
                    &bf16_bytes(&[1.0, 2.0, 3.0]),
                    TokenConstraints::AllowedMask(&[false, false, false]),
                    &[],
                    SamplingParams::default(),
                )
                .unwrap_err(),
            SamplingError::AllMasked
        );
    }

    #[test]
    fn greedy_uses_lower_token_tie_and_zero_rng_words() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let distribution = process(
            &mut workspace,
            &[1.0, 5.0, 5.0, 2.0],
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
        );
        assert!(distribution.is_greedy());
        assert_eq!(distribution.support_size(), 1);
        let mut rng = NeverRng { calls: 0 };
        let result = distribution.sample(&mut rng).unwrap();
        assert_eq!(result.token_id(), 1);
        assert_eq!(result.probability(), 1.0);
        assert_eq!(result.token_logprob(), Some(0.0));
        assert_eq!(rng.calls, 0);
    }

    #[test]
    fn signed_zero_logits_use_the_numeric_tie_break() {
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        let greedy = process(
            &mut workspace,
            &[-0.0, 0.0],
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
        );
        let mut rng = NeverRng { calls: 0 };
        assert_eq!(greedy.sample(&mut rng).unwrap().token_id(), 0);
        assert_eq!(rng.calls, 0);

        let top_k = process(
            &mut workspace,
            &[-0.0, 0.0],
            SamplingParams {
                top_k: Some(1),
                ..SamplingParams::default()
            },
        );
        assert_eq!(top_k.probability(0), Some(1.0));
        assert_eq!(top_k.probability(1), Some(0.0));

        let top_p = process(
            &mut workspace,
            &[-0.0, 0.0],
            SamplingParams {
                top_p: Some(0.5),
                ..SamplingParams::default()
            },
        );
        assert_eq!(top_p.support_size(), 1);
        assert_eq!(top_p.probability(0), Some(1.0));
        assert_eq!(top_p.probability(1), Some(0.0));
    }

    #[test]
    fn top_k_boundaries_and_ties_are_deterministic() {
        let mut workspace = SamplingWorkspace::new(3).unwrap();
        for top_k in [Some(0), Some(4)] {
            assert!(matches!(
                process_error(
                    &mut workspace,
                    &[1.0, 1.0, 0.0],
                    SamplingParams {
                        top_k,
                        ..SamplingParams::default()
                    }
                ),
                SamplingError::InvalidTopK { .. }
            ));
        }

        let distribution = process(
            &mut workspace,
            &[1.0, 1.0, 0.0],
            SamplingParams {
                top_k: Some(1),
                ..SamplingParams::default()
            },
        );
        assert_eq!(distribution.support_size(), 1);
        assert_eq!(distribution.probability(0), Some(1.0));
        assert_eq!(distribution.probability(1), Some(0.0));

        let distribution = process(
            &mut workspace,
            &[1.0, 1.0, 0.0],
            SamplingParams {
                top_k: Some(3),
                ..SamplingParams::default()
            },
        );
        assert_eq!(distribution.support_size(), 3);
    }

    #[test]
    fn top_p_keeps_the_minimal_sorted_prefix_and_accepts_one() {
        let mut workspace = SamplingWorkspace::new(3).unwrap();
        let distribution = process(
            &mut workspace,
            &[2.0, 1.0, 0.0],
            SamplingParams {
                top_p: Some(0.75),
                ..SamplingParams::default()
            },
        );
        assert_eq!(distribution.support_size(), 2);
        assert!(distribution.probability(0).unwrap() > 0.0);
        assert!(distribution.probability(1).unwrap() > 0.0);
        assert_eq!(distribution.probability(2), Some(0.0));

        let distribution = process(
            &mut workspace,
            &[2.0, 1.0, 0.0],
            SamplingParams {
                top_p: Some(1.0),
                ..SamplingParams::default()
            },
        );
        assert_eq!(distribution.support_size(), 3);
    }

    #[test]
    fn invalid_float_parameters_are_rejected() {
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        for temperature in [-1.0, f32::INFINITY, f32::NAN] {
            assert!(matches!(
                process_error(
                    &mut workspace,
                    &[0.0, 1.0],
                    SamplingParams {
                        temperature,
                        ..SamplingParams::default()
                    }
                ),
                SamplingError::InvalidTemperature { .. }
            ));
        }
        for top_p in [0.0, -0.1, 1.1, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                process_error(
                    &mut workspace,
                    &[0.0, 1.0],
                    SamplingParams {
                        top_p: Some(top_p),
                        ..SamplingParams::default()
                    }
                ),
                SamplingError::InvalidTopP { .. }
            ));
        }
        for repetition_penalty in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(matches!(
                process_error(
                    &mut workspace,
                    &[0.0, 1.0],
                    SamplingParams {
                        repetition_penalty,
                        ..SamplingParams::default()
                    }
                ),
                SamplingError::InvalidRepetitionPenalty { .. }
            ));
        }
    }

    #[test]
    fn probability_and_log_probability_views_use_stable_f64_normalization() {
        let mut workspace = SamplingWorkspace::new(3).unwrap();
        let distribution = process(
            &mut workspace,
            &[10_000.0, 9_984.0, -10_000.0],
            SamplingParams::default(),
        );
        let sum: f64 = distribution.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-15);
        assert!(distribution.probability(0).unwrap().is_finite());
        assert!(distribution.log_probability(0).unwrap().is_finite());
        assert_eq!(distribution.probability(99), None);
        assert_eq!(distribution.log_probability(99), None);
        for token_id in [0, 1] {
            let probability = distribution.probability(token_id).unwrap();
            let log_probability = distribution.log_probability(token_id).unwrap();
            assert!((probability.ln() - log_probability).abs() < 1.0e-15);
        }
    }

    #[test]
    fn categorical_walks_ascending_ids_and_consumes_exactly_one_word() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let distribution = process(&mut workspace, &[0.0; 4], SamplingParams::default());

        let cases = [
            (0, 0),
            (u32::MAX / 4, 0),
            (u32::MAX / 2, 1),
            (u32::MAX / 2 + 1, 2),
            (u32::MAX, 3),
        ];
        for (word, expected_token) in cases {
            let words = [word];
            let mut rng = ScriptedRng {
                words: &words,
                cursor: 0,
                calls: 0,
                fail: false,
            };
            let result = distribution.sample(&mut rng).unwrap();
            assert_eq!(result.token_id(), expected_token);
            assert_eq!(rng.calls, 1);
            assert_eq!(rng.cursor, 1);
        }
    }

    #[test]
    fn errors_happen_before_draw_and_rng_errors_propagate() {
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        let mut rng = ScriptedRng {
            words: &[],
            cursor: 0,
            calls: 0,
            fail: false,
        };
        let failure = workspace
            .sample_bf16_native(
                &[],
                TokenConstraints::AllowAll,
                &[],
                SamplingParams::default(),
                &mut rng,
            )
            .unwrap_err();
        assert!(matches!(failure, SamplingFailure::Processing(_)));
        assert_eq!(rng.calls, 0);

        rng.fail = true;
        let failure = workspace
            .sample_bf16_native(
                &bf16_bytes(&[0.0, 1.0]),
                TokenConstraints::AllowAll,
                &[],
                SamplingParams::default(),
                &mut rng,
            )
            .unwrap_err();
        assert_eq!(failure, SamplingFailure::Rng(ScriptedError));
        assert_eq!(rng.calls, 1);
    }

    #[test]
    fn repeated_processing_preserves_all_preallocated_capacities() {
        let mut workspace = SamplingWorkspace::new(8).unwrap();
        let capacities = (
            workspace.processed_logits.capacity(),
            workspace.probabilities.capacity(),
            workspace.log_probabilities.capacity(),
            workspace.candidate_ids.capacity(),
            workspace.history_seen.capacity(),
        );
        for top_k in 1..=8 {
            let distribution = process(
                &mut workspace,
                &[0.0; 8],
                SamplingParams {
                    top_k: Some(top_k),
                    top_p: Some(0.9),
                    ..SamplingParams::default()
                },
            );
            assert!(distribution.support_size() <= top_k);
        }
        assert_eq!(
            capacities,
            (
                workspace.processed_logits.capacity(),
                workspace.probabilities.capacity(),
                workspace.log_probabilities.capacity(),
                workspace.candidate_ids.capacity(),
                workspace.history_seen.capacity(),
            )
        );
    }

    fn process_error(
        workspace: &mut SamplingWorkspace,
        values: &[f32],
        params: SamplingParams,
    ) -> SamplingError {
        workspace
            .process_bf16_native(&bf16_bytes(values), TokenConstraints::AllowAll, &[], params)
            .unwrap_err()
    }
}
