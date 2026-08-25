//! Portable request-local random-number generation.
//!
//! [`Philox4x32Rng`] is the native implementation of the normative
//! `rustinfer.philox4x32-10.v1` contract in `benchmarks/RNG.md`. Its complete
//! derivation identity and logical next-word position can be checkpointed with
//! [`RngSnapshot`]. No process-global generator is used.

use std::error;
use std::fmt;

use sha2::{Digest, Sha256};

/// Stable algorithm identifier recorded in snapshots and result metadata.
pub const PHILOX4X32_10_ALGORITHM_ID: &str = "rustinfer.philox4x32-10.v1";

const SEED_TAG: &[u8] = b"rustinfer.philox4x32-10.v1/seed\0";
const STREAM_TAG: &[u8] = b"rustinfer.philox4x32-10.v1/stream\0";
const DOMAIN_TAG: &[u8] = b"rustinfer.philox4x32-10.v1/domain\0";
const FORK_TAG: &[u8] = b"rustinfer.philox4x32-10.v1/fork\0";
const DERIVATION_DIGEST_BYTES: usize = 32;
const DERIVATION_DIGEST_HEX_BYTES: usize = DERIVATION_DIGEST_BYTES * 2;
const WORDS_PER_BLOCK: u128 = 4;
const M0: u32 = 0xd251_1f53;
const M1: u32 = 0xcd9e_8d57;
const W0: u32 = 0x9e37_79b9;
const W1: u32 = 0xbb67_ae85;
const ROUNDS: usize = 10;
const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
const TWO_NEGATIVE_32: f64 = 1.0 / 4_294_967_296.0;

type DerivationDigest = [u8; DERIVATION_DIGEST_BYTES];

/// Result type for portable request-local RNG operations.
pub type RngResult<T> = Result<T, RngError>;

/// Length-prefixed input whose byte length violated the version-1 contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RngInput {
    /// Logical request or stream identifier.
    Stream,
    /// Consumer domain within one request.
    Domain,
    /// Child-stream label supplied to [`Philox4x32Rng::fork`].
    ForkLabel,
}

impl fmt::Display for RngInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream => formatter.write_str("stream"),
            Self::Domain => formatter.write_str("domain"),
            Self::ForkLabel => formatter.write_str("fork label"),
        }
    }
}

/// Checked derivation, snapshot, or draw failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RngError {
    /// A length-prefixed byte input cannot be represented by the `LE32` format.
    InputTooLong {
        /// Input role whose length was invalid.
        input: RngInput,
        /// Supplied byte length.
        byte_len: usize,
    },
    /// A draw was requested after all `2^66` version-1 words were consumed.
    Exhausted,
    /// A snapshot names an algorithm other than the version implemented here.
    UnsupportedAlgorithmId {
        /// Algorithm identifier found in the snapshot.
        algorithm_id: String,
    },
    /// The derivation digest is not exactly 64 lowercase hexadecimal bytes.
    InvalidDerivationDigestHex,
    /// The block is not the canonical unsigned decimal representation of a `u64`.
    InvalidBlockDecimal {
        /// Invalid snapshot field value.
        block: String,
    },
    /// The next-word offset is outside `0..=3`.
    WordOffsetOutOfRange {
        /// Invalid snapshot field value.
        word_offset: u8,
    },
    /// An exhausted snapshot did not use the sole canonical terminal position.
    InconsistentExhaustedState {
        /// Parsed block field.
        block: u64,
        /// Parsed word offset field.
        word_offset: u8,
    },
}

impl fmt::Display for RngError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLong { input, byte_len } => write!(
                formatter,
                "RNG {input} is {byte_len} bytes, exceeding the LE32 length limit"
            ),
            Self::Exhausted => formatter.write_str("the portable RNG stream is exhausted"),
            Self::UnsupportedAlgorithmId { algorithm_id } => write!(
                formatter,
                "unsupported RNG algorithm identifier {algorithm_id:?}"
            ),
            Self::InvalidDerivationDigestHex => formatter.write_str(
                "RNG snapshot derivation_digest_hex must be 64 lowercase hexadecimal bytes",
            ),
            Self::InvalidBlockDecimal { block } => write!(
                formatter,
                "RNG snapshot block {block:?} is not a canonical unsigned decimal u64"
            ),
            Self::WordOffsetOutOfRange { word_offset } => write!(
                formatter,
                "RNG snapshot word_offset {word_offset} is outside 0..=3"
            ),
            Self::InconsistentExhaustedState { block, word_offset } => write!(
                formatter,
                "exhausted RNG snapshot must use block {} and word_offset 3, found block {block} and word_offset {word_offset}",
                u64::MAX
            ),
        }
    }
}

impl error::Error for RngError {}

/// Canonical, portable representation of the next RNG word.
///
/// The field names and types match the canonical JSON object in
/// `benchmarks/RNG.md`. In particular, [`Self::block`] is decimal text rather
/// than a JSON number. Fields are public so a serialization layer can map the
/// object without introducing a production serialization dependency; restore
/// always validates all fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RngSnapshot {
    /// Exact versioned algorithm identifier.
    pub algorithm_id: String,
    /// Full 32-byte derivation digest as 64 lowercase hexadecimal characters.
    pub derivation_digest_hex: String,
    /// Unsigned `u64` block number encoded as canonical decimal text.
    pub block: String,
    /// Word within the block, in `0..=3`.
    pub word_offset: u8,
    /// Whether the terminal word has already been consumed.
    pub exhausted: bool,
}

/// Request-local implementation of `rustinfer.philox4x32-10.v1`.
///
/// The generator is intentionally not `Clone`: callers that need a branch
/// must name it explicitly with [`Self::fork`]. Recreating the same
/// `(master_seed, stream, domain)` tuple deliberately recreates the stream.
#[derive(Debug, Eq, PartialEq)]
pub struct Philox4x32Rng {
    derivation_digest: DerivationDigest,
    key: [u32; 2],
    nonce: [u32; 2],
    block: u64,
    word_offset: u8,
    exhausted: bool,
}

impl Philox4x32Rng {
    /// Derives a request-local stream from a master seed and arbitrary bytes.
    ///
    /// Text identifiers must be supplied as their strict UTF-8 bytes. The
    /// convenience [`Self::from_text`] constructor performs that conversion.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::InputTooLong`] if `stream` or `domain` is longer
    /// than `u32::MAX` bytes and therefore cannot use the required `LE32`
    /// length prefix.
    pub fn new(master_seed: u64, stream: &[u8], domain: &[u8]) -> RngResult<Self> {
        let seed_digest = derive_seed(master_seed);
        let stream_digest = derive_child(STREAM_TAG, &seed_digest, stream, RngInput::Stream)?;
        let derivation_digest = derive_child(DOMAIN_TAG, &stream_digest, domain, RngInput::Domain)?;
        Ok(Self::from_validated_state(derivation_digest, 0, 0, false))
    }

    /// Derives a request-local stream from strict UTF-8 text identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::InputTooLong`] if either encoded string is longer
    /// than `u32::MAX` bytes.
    pub fn from_text(master_seed: u64, stream: &str, domain: &str) -> RngResult<Self> {
        Self::new(master_seed, stream.as_bytes(), domain.as_bytes())
    }

    /// Constructs a generator from a fully validated portable snapshot.
    ///
    /// Snapshot parsing is strict: the algorithm ID must match exactly, the
    /// digest must be lowercase hexadecimal, the block must be canonical
    /// unsigned decimal text, and an exhausted state must use the one terminal
    /// representation defined by version 1.
    ///
    /// # Errors
    ///
    /// Returns a structured [`RngError`] for the first invalid field.
    pub fn from_snapshot(snapshot: &RngSnapshot) -> RngResult<Self> {
        if snapshot.algorithm_id != PHILOX4X32_10_ALGORITHM_ID {
            return Err(RngError::UnsupportedAlgorithmId {
                algorithm_id: snapshot.algorithm_id.clone(),
            });
        }
        let derivation_digest = parse_digest_hex(&snapshot.derivation_digest_hex)?;
        let block = parse_block_decimal(&snapshot.block)?;
        if snapshot.word_offset > 3 {
            return Err(RngError::WordOffsetOutOfRange {
                word_offset: snapshot.word_offset,
            });
        }
        if snapshot.exhausted && (block != u64::MAX || snapshot.word_offset != 3) {
            return Err(RngError::InconsistentExhaustedState {
                block,
                word_offset: snapshot.word_offset,
            });
        }
        Ok(Self::from_validated_state(
            derivation_digest,
            block,
            snapshot.word_offset,
            snapshot.exhausted,
        ))
    }

    /// Replaces this generator with a fully validated portable snapshot.
    ///
    /// Validation is transactional: on error, `self` is unchanged. Restoring
    /// does not consume a word.
    ///
    /// # Errors
    ///
    /// Returns a structured [`RngError`] for the first invalid field.
    pub fn restore(&mut self, snapshot: &RngSnapshot) -> RngResult<()> {
        let restored = Self::from_snapshot(snapshot)?;
        *self = restored;
        Ok(())
    }

    /// Captures the canonical portable position of the next word.
    ///
    /// Snapshotting does not consume a word.
    #[must_use]
    pub fn snapshot(&self) -> RngSnapshot {
        RngSnapshot {
            algorithm_id: PHILOX4X32_10_ALGORITHM_ID.to_owned(),
            derivation_digest_hex: digest_hex(&self.derivation_digest),
            block: self.block.to_string(),
            word_offset: self.word_offset,
            exhausted: self.exhausted,
        }
    }

    /// Creates a deterministic child identity at block zero and word zero.
    ///
    /// Forking depends on the complete parent derivation digest, not its
    /// current draw position, and does not mutate or consume the parent.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::InputTooLong`] if `label` cannot use the required
    /// `LE32` byte-length prefix.
    pub fn fork(&self, label: &[u8]) -> RngResult<Self> {
        let child_digest = derive_child(
            FORK_TAG,
            &self.derivation_digest,
            label,
            RngInput::ForkLabel,
        )?;
        Ok(Self::from_validated_state(child_digest, 0, 0, false))
    }

    /// Creates a deterministic child identity from a strict UTF-8 label.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::InputTooLong`] if the encoded label cannot use the
    /// required `LE32` byte-length prefix.
    pub fn fork_text(&self, label: &str) -> RngResult<Self> {
        self.fork(label.as_bytes())
    }

    /// Returns the next portable 32-bit word and advances exactly one draw.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::Exhausted`] after the fourth word of block
    /// `u64::MAX` was consumed. The terminal state never wraps to block zero.
    pub fn next_u32(&mut self) -> RngResult<u32> {
        if self.exhausted {
            return Err(RngError::Exhausted);
        }

        let counter = block_counter(self.block, self.nonce);
        let words = philox4x32_10(counter, self.key);
        let word = words[usize::from(self.word_offset)];

        if self.word_offset < 3 {
            self.word_offset += 1;
        } else if self.block == u64::MAX {
            self.exhausted = true;
        } else {
            self.block += 1;
            self.word_offset = 0;
        }

        Ok(word)
    }

    /// Returns the next word mapped exactly to a binary64 value in `(0, 1)`.
    ///
    /// The mapping is `(f64::from(word) + 0.5) * 2^-32` and consumes exactly
    /// one word.
    ///
    /// # Errors
    ///
    /// Returns [`RngError::Exhausted`] when no word remains.
    pub fn next_f64_open01(&mut self) -> RngResult<f64> {
        self.next_u32().map(uniform_f64_open01)
    }

    /// Returns the exact number of words consumed by this state.
    ///
    /// The `u128` result represents the terminal value `2^66` without
    /// overflow. Greedy decoding can assert that this remains zero.
    #[must_use]
    pub fn draws(&self) -> u128 {
        if self.exhausted {
            (u128::from(u64::MAX) + 1) * WORDS_PER_BLOCK
        } else {
            u128::from(self.block) * WORDS_PER_BLOCK + u128::from(self.word_offset)
        }
    }

    /// Returns whether the terminal word has already been consumed.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Returns the stable versioned algorithm identifier.
    #[must_use]
    pub const fn algorithm_id(&self) -> &'static str {
        PHILOX4X32_10_ALGORITHM_ID
    }

    /// Returns the complete derivation identity retained by the generator.
    #[must_use]
    pub const fn derivation_digest(&self) -> &[u8; DERIVATION_DIGEST_BYTES] {
        &self.derivation_digest
    }

    fn from_validated_state(
        derivation_digest: DerivationDigest,
        block: u64,
        word_offset: u8,
        exhausted: bool,
    ) -> Self {
        let key = [
            digest_word(&derivation_digest, 0),
            digest_word(&derivation_digest, 4),
        ];
        let nonce = [
            digest_word(&derivation_digest, 8),
            digest_word(&derivation_digest, 12),
        ];
        Self {
            derivation_digest,
            key,
            nonce,
            block,
            word_offset,
            exhausted,
        }
    }
}

impl crate::sampling::SamplingRng for Philox4x32Rng {
    type Error = RngError;

    fn next_u32(&mut self) -> Result<u32, Self::Error> {
        Philox4x32Rng::next_u32(self)
    }
}

fn derive_seed(master_seed: u64) -> DerivationDigest {
    let mut hasher = Sha256::new();
    hasher.update(SEED_TAG);
    hasher.update(master_seed.to_le_bytes());
    hasher.finalize().into()
}

fn derive_child(
    tag: &[u8],
    parent_digest: &DerivationDigest,
    value: &[u8],
    input: RngInput,
) -> RngResult<DerivationDigest> {
    let byte_len = u32::try_from(value.len()).map_err(|_| RngError::InputTooLong {
        input,
        byte_len: value.len(),
    })?;
    let mut hasher = Sha256::new();
    hasher.update(tag);
    hasher.update(parent_digest);
    hasher.update(byte_len.to_le_bytes());
    hasher.update(value);
    Ok(hasher.finalize().into())
}

fn digest_word(digest: &DerivationDigest, start: usize) -> u32 {
    u32::from_le_bytes([
        digest[start],
        digest[start + 1],
        digest[start + 2],
        digest[start + 3],
    ])
}

fn block_counter(block: u64, nonce: [u32; 2]) -> [u32; 4] {
    let bytes = block.to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        nonce[0],
        nonce[1],
    ]
}

fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..ROUNDS {
        let (hi0, lo0) = mul_hi_lo(M0, counter[0]);
        let (hi1, lo1) = mul_hi_lo(M1, counter[2]);
        counter = [
            hi1 ^ counter[1] ^ key[0],
            lo1,
            hi0 ^ counter[3] ^ key[1],
            lo0,
        ];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    counter
}

fn mul_hi_lo(lhs: u32, rhs: u32) -> (u32, u32) {
    let bytes = (u64::from(lhs) * u64::from(rhs)).to_le_bytes();
    (
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    )
}

fn uniform_f64_open01(word: u32) -> f64 {
    (f64::from(word) + 0.5) * TWO_NEGATIVE_32
}

fn digest_hex(digest: &DerivationDigest) -> String {
    let mut output = String::with_capacity(DERIVATION_DIGEST_HEX_BYTES);
    for byte in digest {
        output.push(char::from(LOWER_HEX[usize::from(byte >> 4)]));
        output.push(char::from(LOWER_HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn parse_digest_hex(value: &str) -> RngResult<DerivationDigest> {
    let bytes = value.as_bytes();
    if bytes.len() != DERIVATION_DIGEST_HEX_BYTES
        || !bytes
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(RngError::InvalidDerivationDigestHex);
    }
    let mut digest = [0_u8; DERIVATION_DIGEST_BYTES];
    for (output, pair) in digest.iter_mut().zip(bytes.chunks_exact(2)) {
        *output = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(digest)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("digest hex is validated before decoding"),
    }
}

fn parse_block_decimal(value: &str) -> RngResult<u64> {
    let bytes = value.as_bytes();
    let canonical = !bytes.is_empty()
        && (bytes.len() == 1 || bytes[0] != b'0')
        && bytes.iter().all(u8::is_ascii_digit);
    if !canonical {
        return Err(RngError::InvalidBlockDecimal {
            block: value.to_owned(),
        });
    }
    value
        .parse::<u64>()
        .map_err(|_| RngError::InvalidBlockDecimal {
            block: value.to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use crate::sampling::{SamplingParams, SamplingWorkspace, TokenConstraints};

    use super::{
        PHILOX4X32_10_ALGORITHM_ID, Philox4x32Rng, RngError, RngInput, RngSnapshot, derive_child,
        derive_seed, digest_hex, philox4x32_10, uniform_f64_open01,
    };

    const DIGEST: &str = "71135575b8f1ec48b51e72910fd23520a573d330c60c2b0b3a0fa8e33944e75a";

    fn known_rng() -> Philox4x32Rng {
        Philox4x32Rng::from_text(42, "request-0001", "token-sampling")
            .expect("known inputs are representable")
    }

    #[test]
    fn raw_philox_known_answers_match_random123() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn derivation_and_first_block_match_known_answers() {
        let seed_digest = derive_seed(42);
        assert_eq!(
            digest_hex(&seed_digest),
            "27ba307ffd89d3dbb7ffd3d3d9b25ada9a275b78dae269838a636c8fc900db85"
        );
        let stream_digest = derive_child(
            super::STREAM_TAG,
            &seed_digest,
            b"request-0001",
            RngInput::Stream,
        )
        .expect("known stream is representable");
        assert_eq!(
            digest_hex(&stream_digest),
            "e9e49a0403fd05d078f7528cacf57fa24c3edb759363db4b8cefc994ca38048e"
        );

        let mut rng = known_rng();
        assert_eq!(digest_hex(rng.derivation_digest()), DIGEST);
        assert_eq!(
            [
                rng.next_u32().expect("word 0"),
                rng.next_u32().expect("word 1"),
                rng.next_u32().expect("word 2"),
                rng.next_u32().expect("word 3"),
            ],
            [0xa8da_52b2, 0x9d74_b2a4, 0xd4cd_2d7a, 0x658c_3d44]
        );
        assert_eq!(rng.draws(), 4);
        let snapshot = rng.snapshot();
        assert_eq!(snapshot.block, "1");
        assert_eq!(snapshot.word_offset, 0);
        assert!(!snapshot.exhausted);
    }

    #[test]
    fn f64_open_interval_mapping_is_exact() {
        assert_eq!(uniform_f64_open01(0).to_bits(), 2.0_f64.powi(-33).to_bits());
        assert_eq!(
            uniform_f64_open01(u32::MAX).to_bits(),
            (1.0 - 2.0_f64.powi(-33)).to_bits()
        );

        let mut rng = known_rng();
        for expected_bits in [
            0x3fe5_1b4a_5650_0000,
            0x3fe3_ae96_5490_0000,
            0x3fea_99a5_af50_0000,
            0x3fd9_630f_5120_0000,
        ] {
            let value = rng.next_f64_open01().expect("known block has words");
            assert_eq!(value.to_bits(), expected_bits);
            assert!(value > 0.0 && value < 1.0);
        }
    }

    #[test]
    fn snapshot_restore_replays_suffix_without_consumption() {
        let mut rng = known_rng();
        assert_eq!(rng.next_u32().expect("word 0"), 0xa8da_52b2);
        assert_eq!(rng.next_u32().expect("word 1"), 0x9d74_b2a4);
        let snapshot = rng.snapshot();
        assert_eq!(
            snapshot,
            RngSnapshot {
                algorithm_id: PHILOX4X32_10_ALGORITHM_ID.to_owned(),
                derivation_digest_hex: DIGEST.to_owned(),
                block: "0".to_owned(),
                word_offset: 2,
                exhausted: false,
            }
        );
        assert_eq!(rng.draws(), 2);

        let mut restored = Philox4x32Rng::from_snapshot(&snapshot).expect("valid snapshot");
        assert_eq!(restored.draws(), 2);
        assert_eq!(restored.next_u32().expect("restored word"), 0xd4cd_2d7a);
        assert_eq!(rng.next_u32().expect("original word"), 0xd4cd_2d7a);

        let before_invalid_restore = rng.snapshot();
        let invalid = RngSnapshot {
            algorithm_id: "unknown".to_owned(),
            ..snapshot
        };
        assert!(matches!(
            rng.restore(&invalid),
            Err(RngError::UnsupportedAlgorithmId { .. })
        ));
        assert_eq!(rng.snapshot(), before_invalid_restore);
    }

    #[test]
    fn restore_rejects_every_noncanonical_snapshot_field() {
        let canonical = known_rng().snapshot();
        let mut cases = Vec::new();

        let mut uppercase_digest = canonical.clone();
        uppercase_digest
            .derivation_digest_hex
            .replace_range(0..1, "A");
        cases.push(uppercase_digest);

        let mut short_digest = canonical.clone();
        short_digest.derivation_digest_hex.pop();
        cases.push(short_digest);

        for block in ["", "00", "01", "+1", "-1", " 1", "18446744073709551616"] {
            let mut invalid = canonical.clone();
            invalid.block = block.to_owned();
            cases.push(invalid);
        }

        for invalid in cases {
            assert!(
                Philox4x32Rng::from_snapshot(&invalid).is_err(),
                "{invalid:?}"
            );
        }

        let mut invalid_offset = canonical.clone();
        invalid_offset.word_offset = 4;
        assert_eq!(
            Philox4x32Rng::from_snapshot(&invalid_offset),
            Err(RngError::WordOffsetOutOfRange { word_offset: 4 })
        );

        let mut invalid_exhausted = canonical;
        invalid_exhausted.exhausted = true;
        assert_eq!(
            Philox4x32Rng::from_snapshot(&invalid_exhausted),
            Err(RngError::InconsistentExhaustedState {
                block: 0,
                word_offset: 0,
            })
        );
    }

    #[test]
    fn fork_is_position_independent_and_matches_known_answer() {
        let parent_before_draws = known_rng();
        let mut parent_after_draws = known_rng();
        for _ in 0..7 {
            parent_after_draws
                .next_u32()
                .expect("stream is not exhausted");
        }

        let mut before = parent_before_draws.fork_text("draft").expect("known label");
        let mut after = parent_after_draws.fork_text("draft").expect("known label");
        assert_eq!(before.snapshot(), after.snapshot());
        assert_eq!(parent_after_draws.draws(), 7);
        assert_eq!(
            before.snapshot().derivation_digest_hex,
            "f5f67e3f27ba60ea8e8ac56c614d162ff11b5133486e7c463ca03a6ba9b512c1"
        );

        let expected = [0x4a38_999f, 0xd695_c269, 0x4def_e354, 0xe0d2_c8f5];
        for word in expected {
            assert_eq!(before.next_u32().expect("known child word"), word);
            assert_eq!(after.next_u32().expect("known child word"), word);
        }

        let nested_a = parent_before_draws
            .fork_text("draft")
            .expect("first fork")
            .fork_text("nested")
            .expect("nested fork");
        let nested_b = parent_after_draws
            .fork_text("draft")
            .expect("first fork")
            .fork_text("nested")
            .expect("nested fork");
        assert_eq!(nested_a.snapshot(), nested_b.snapshot());
    }

    #[test]
    fn terminal_draw_exhausts_without_counter_wrap() {
        let base = known_rng().snapshot();
        let terminal = RngSnapshot {
            block: u64::MAX.to_string(),
            word_offset: 3,
            ..base
        };
        let mut rng = Philox4x32Rng::from_snapshot(&terminal).expect("last word is valid");
        assert_eq!(
            rng.draws(),
            u128::from(u64::MAX) * 4 + 3,
            "state points at the final unconsumed word"
        );
        rng.next_u32().expect("terminal word is available");
        assert!(rng.is_exhausted());
        assert_eq!(rng.draws(), (u128::from(u64::MAX) + 1) * 4);
        assert_eq!(
            rng.snapshot().block,
            "18446744073709551615",
            "terminal state preserves the maximum block"
        );
        assert_eq!(rng.snapshot().word_offset, 3);
        assert!(rng.snapshot().exhausted);

        let terminal_snapshot = rng.snapshot();
        assert_eq!(rng.next_u32(), Err(RngError::Exhausted));
        assert_eq!(rng.snapshot(), terminal_snapshot);
        let mut restored = Philox4x32Rng::from_snapshot(&terminal_snapshot)
            .expect("canonical terminal state restores");
        assert_eq!(restored.next_u32(), Err(RngError::Exhausted));
    }

    #[test]
    fn request_streams_are_isolated_from_batch_order() {
        let mut request_a =
            Philox4x32Rng::from_text(7, "request-a", "token-sampling").expect("request A");
        let mut request_b =
            Philox4x32Rng::from_text(7, "request-b", "token-sampling").expect("request B");
        let a_words = [
            request_a.next_u32().expect("A0"),
            request_a.next_u32().expect("A1"),
            request_a.next_u32().expect("A2"),
        ];
        let b_words = [
            request_b.next_u32().expect("B0"),
            request_b.next_u32().expect("B1"),
            request_b.next_u32().expect("B2"),
        ];
        assert_ne!(a_words, b_words);

        let mut reordered_a =
            Philox4x32Rng::from_text(7, "request-a", "token-sampling").expect("request A");
        let mut reordered_b =
            Philox4x32Rng::from_text(7, "request-b", "token-sampling").expect("request B");
        let mut observed_a = [0; 3];
        let mut observed_b = [0; 3];
        for index in 0..3 {
            observed_b[index] = reordered_b.next_u32().expect("reordered B");
            observed_a[index] = reordered_a.next_u32().expect("reordered A");
        }
        assert_eq!(observed_a, a_words);
        assert_eq!(observed_b, b_words);
    }

    #[test]
    fn construction_snapshot_and_fork_consume_zero_draws() {
        let rng = known_rng();
        assert_eq!(rng.draws(), 0, "greedy sampling starts at zero draws");
        let snapshot = rng.snapshot();
        assert_eq!(rng.draws(), 0, "snapshot does not consume a word");
        let child = rng.fork_text("branch").expect("known label");
        assert_eq!(rng.draws(), 0, "fork does not consume the parent");
        assert_eq!(child.draws(), 0, "a child begins at block and word zero");
        let restored = Philox4x32Rng::from_snapshot(&snapshot).expect("valid snapshot");
        assert_eq!(restored.draws(), 0, "restore does not consume a word");
    }

    #[test]
    fn sampling_greedy_consumes_zero_and_stochastic_consumes_one_draw() {
        let mut logits = [0_u8; 4];
        logits[..2].copy_from_slice(&0x3f80_u16.to_ne_bytes());
        logits[2..].copy_from_slice(&0x4000_u16.to_ne_bytes());
        let mut workspace = SamplingWorkspace::new(2).expect("two-token sampling workspace");
        let mut rng = known_rng();

        workspace
            .sample_bf16_native(
                &logits,
                TokenConstraints::AllowAll,
                &[],
                SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                &mut rng,
            )
            .expect("greedy distribution is valid");
        assert_eq!(rng.draws(), 0);

        workspace
            .sample_bf16_native(
                &logits,
                TokenConstraints::AllowAll,
                &[],
                SamplingParams::default(),
                &mut rng,
            )
            .expect("stochastic distribution is valid");
        assert_eq!(rng.draws(), 1);
    }
}
