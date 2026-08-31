//! Pure host-side decoding for fixed-width Llama greedy output records.
//!
//! The batch owner retains all device downloads, workspace storage, output
//! readiness, and poison decisions. This component validates the complete
//! host record sequence before publishing its canonical token map.

use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
};

use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult};

const BF16_BYTES: u64 = 2;
const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Native `{token_id,status}` result bytes produced per greedy output row.
pub(in crate::llama) const GREEDY_RESULT_BYTES: usize = 2 * U32_BYTES;

/// Exact BF16 byte length for one dense `[output_count, vocabulary_size]` map.
///
/// The batch owner uses this checked scalar result when binding an existing
/// gathered-logits buffer to CUDA output primitives.
pub(in crate::llama) fn output_logits_bytes(
    output_count: usize,
    vocabulary_size: usize,
) -> LlamaBatchExecutorResult<u64> {
    let output_count =
        u64::try_from(output_count).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let vocabulary_size = u64::try_from(vocabulary_size).map_err(|_| {
        LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        }
    })?;
    output_count
        .checked_mul(vocabulary_size)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })
}

/// Validates fixed-width greedy records before filling dense output slots.
///
/// A non-finite row remains a non-poisoning typed result. Other invalid native
/// records are returned for the enclosing owner to poison before exposing a
/// failed iteration. No destination token is written until every record has
/// passed validation.
pub(in crate::llama) fn decode_greedy_tokens(
    records: &[u8],
    vocabulary_size: usize,
    destination: &mut [u32],
) -> LlamaBatchExecutorResult<()> {
    for (output_index, record) in records.chunks_exact(GREEDY_RESULT_BYTES).enumerate() {
        let token_id = u32::from_ne_bytes(record[..U32_BYTES].try_into().map_err(|_| {
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_result_record",
                reason: "token word has an invalid native layout",
            }
        })?);
        let status = u32::from_ne_bytes(record[U32_BYTES..].try_into().map_err(|_| {
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_result_record",
                reason: "status word has an invalid native layout",
            }
        })?);
        match status {
            BF16_ARGMAX_STATUS_SUCCESS
                if token_id != BF16_ARGMAX_INVALID_TOKEN_ID
                    && usize::try_from(token_id)
                        .ok()
                        .is_some_and(|token| token < vocabulary_size) => {}
            BF16_ARGMAX_STATUS_NON_FINITE if token_id == BF16_ARGMAX_INVALID_TOKEN_ID => {
                return Err(LlamaBatchExecutorError::GreedyLogitsNonFinite { output_index });
            }
            _ => {
                return Err(LlamaBatchExecutorError::InvalidGreedyResult {
                    output_index,
                    status,
                    token_id,
                });
            }
        }
    }
    for (output, record) in destination
        .iter_mut()
        .zip(records.chunks_exact(GREEDY_RESULT_BYTES))
    {
        *output = u32::from_ne_bytes(record[..U32_BYTES].try_into().map_err(|_| {
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_result_record",
                reason: "token word has an invalid native layout",
            }
        })?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(token_id: u32, status: u32) -> [u8; GREEDY_RESULT_BYTES] {
        let mut bytes = [0; GREEDY_RESULT_BYTES];
        bytes[..U32_BYTES].copy_from_slice(&token_id.to_ne_bytes());
        bytes[U32_BYTES..].copy_from_slice(&status.to_ne_bytes());
        bytes
    }

    #[test]
    fn greedy_records_publish_dense_tokens_only_after_full_validation() {
        let first = record(3, BF16_ARGMAX_STATUS_SUCCESS);
        let second = record(u32::MAX, BF16_ARGMAX_STATUS_SUCCESS);
        let mut records = [0; 2 * GREEDY_RESULT_BYTES];
        records[..GREEDY_RESULT_BYTES].copy_from_slice(&first);
        records[GREEDY_RESULT_BYTES..].copy_from_slice(&second);
        let mut destination = [41, 42];

        let error = decode_greedy_tokens(&records, 8, &mut destination).expect_err("bad token");

        assert!(matches!(
            error,
            LlamaBatchExecutorError::InvalidGreedyResult {
                output_index: 1,
                status: BF16_ARGMAX_STATUS_SUCCESS,
                token_id: u32::MAX,
            }
        ));
        assert_eq!(destination, [41, 42]);
    }

    #[test]
    fn greedy_records_publish_each_valid_dense_token() {
        let first = record(3, BF16_ARGMAX_STATUS_SUCCESS);
        let second = record(7, BF16_ARGMAX_STATUS_SUCCESS);
        let mut records = [0; 2 * GREEDY_RESULT_BYTES];
        records[..GREEDY_RESULT_BYTES].copy_from_slice(&first);
        records[GREEDY_RESULT_BYTES..].copy_from_slice(&second);
        let mut destination = [41, 42];

        decode_greedy_tokens(&records, 8, &mut destination).expect("valid records");

        assert_eq!(destination, [3, 7]);
    }

    #[test]
    fn greedy_records_preserve_nonfinite_status_without_partial_publication() {
        let record = record(BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE);
        let mut destination = [41];

        let error = decode_greedy_tokens(&record, 8, &mut destination).expect_err("non-finite");

        assert!(matches!(
            error,
            LlamaBatchExecutorError::GreedyLogitsNonFinite { output_index: 0 }
        ));
        assert_eq!(destination, [41]);
    }

    #[test]
    fn output_logits_bytes_is_exact_and_fails_closed_on_overflow() {
        assert_eq!(output_logits_bytes(3, 5).expect("representable logits"), 30);
        assert!(matches!(
            output_logits_bytes(usize::MAX, usize::MAX),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::GatheredLogits,
            })
        ));
    }
}
