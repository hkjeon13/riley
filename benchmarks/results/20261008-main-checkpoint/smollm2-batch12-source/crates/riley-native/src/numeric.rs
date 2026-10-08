#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use crate::contract::{CALIBRATION_TOP_K, CROSS_CACHE_EXACT_WINDOW};

#[derive(Debug)]
pub(crate) enum NumericError {
    InvalidBf16Length,
    EmptyValues,
    NonFinite { index: usize },
    DestinationLength { expected: usize, actual: usize },
    ArithmeticOverflow,
}

impl fmt::Display for NumericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBf16Length => formatter.write_str("BF16 byte length must be even"),
            Self::EmptyValues => formatter.write_str("numeric vector must be non-empty"),
            Self::NonFinite { index } => write!(formatter, "numeric value {index} is non-finite"),
            Self::DestinationLength { expected, actual } => {
                write!(
                    formatter,
                    "destination has {actual} bytes, expected {expected}"
                )
            }
            Self::ArithmeticOverflow => formatter.write_str("numeric byte arithmetic overflow"),
        }
    }
}

impl Error for NumericError {}

pub(crate) fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>, NumericError> {
    if bytes.len() % 2 != 0 {
        return Err(NumericError::InvalidBf16Length);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|_| NumericError::ArithmeticOverflow)?;
    for chunk in bytes.chunks_exact(2) {
        values.push(f32::from_bits(
            u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
        ));
    }
    Ok(values)
}

pub(crate) fn ranked_top_k_bf16(bytes: &[u8]) -> Result<(u32, Vec<u32>), NumericError> {
    let values = decode_bf16(bytes)?;
    if values.len() < CALIBRATION_TOP_K {
        return Err(NumericError::EmptyValues);
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(NumericError::NonFinite { index });
        }
    }
    let mut indexes: Vec<usize> = (0..values.len()).collect();
    if indexes.len() > CALIBRATION_TOP_K {
        indexes.select_nth_unstable_by(CALIBRATION_TOP_K, |&left, &right| {
            descending_value_then_id(&values, left, right)
        });
    }
    indexes.truncate(CALIBRATION_TOP_K);
    indexes.sort_unstable_by(|&left, &right| descending_value_then_id(&values, left, right));
    let top_1 = u32::try_from(indexes[0]).map_err(|_| NumericError::ArithmeticOverflow)?;
    let mut token_set = indexes
        .into_iter()
        .map(|index| u32::try_from(index).map_err(|_| NumericError::ArithmeticOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    token_set.sort_unstable();
    Ok((top_1, token_set))
}

fn descending_value_then_id(values: &[f32], left: usize, right: usize) -> Ordering {
    values[right]
        .partial_cmp(&values[left])
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.cmp(&right))
}

pub(crate) fn canonical_log_softmax_bf16(
    logits: &[u8],
    destination: &mut [u8],
) -> Result<(), NumericError> {
    let values = decode_bf16(logits)?;
    if values.is_empty() {
        return Err(NumericError::EmptyValues);
    }
    let expected = values
        .len()
        .checked_mul(4)
        .ok_or(NumericError::ArithmeticOverflow)?;
    if destination.len() != expected {
        return Err(NumericError::DestinationLength {
            expected,
            actual: destination.len(),
        });
    }
    let mut maximum = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(NumericError::NonFinite { index });
        }
        maximum = maximum.max(value);
    }
    let mut sum = 0.0_f32;
    for &value in &values {
        sum += (value - maximum).exp();
    }
    let log_sum = sum.ln();
    if !log_sum.is_finite() {
        return Err(NumericError::NonFinite {
            index: values.len(),
        });
    }
    for (&value, output) in values.iter().zip(destination.chunks_exact_mut(4)) {
        output.copy_from_slice(&(value - maximum - log_sum).to_le_bytes());
    }
    Ok(())
}

pub(crate) fn first_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        if left != right {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(left.len().min(right.len()))
}

pub(crate) fn exact_window_match(left: &[u32], right: &[u32]) -> bool {
    first_divergence(left, right).is_none_or(|step| step >= CROSS_CACHE_EXACT_WINDOW)
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_log_softmax_bf16, exact_window_match, first_divergence, ranked_top_k_bf16,
    };

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| {
                let bits = (value.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect()
    }

    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn ranking_uses_lower_token_id_for_ties_and_sorted_set() {
        let logits = bf16(&[1.0, 4.0, 4.0, 3.0, 2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
        let (top, token_set) = ranked_top_k_bf16(&logits).expect("ranking");
        assert_eq!(top, 1);
        assert_eq!(token_set, (0_u32..10).collect::<Vec<_>>());

        let exactly_ten = bf16(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
        let (top, token_set) = ranked_top_k_bf16(&exactly_ten).expect("exact-length ranking");
        assert_eq!(top, 0);
        assert_eq!(token_set, (0_u32..10).collect::<Vec<_>>());
    }

    #[test]
    fn canonical_log_softmax_is_normalized() {
        let logits = bf16(&[0.0, 1.0, 2.0]);
        let mut output = vec![0_u8; 12];
        canonical_log_softmax_bf16(&logits, &mut output).expect("log-softmax");
        let values = f32_values(&output);
        let probability_sum: f32 = values.iter().map(|value| value.exp()).sum();
        assert!((probability_sum - 1.0).abs() < 1.0e-6);
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    #[test]
    fn divergence_includes_length_mismatch_and_exact_window() {
        assert_eq!(first_divergence(&[1, 2], &[1, 3]), Some(1));
        assert_eq!(first_divergence(&[1], &[1, 2]), Some(1));
        assert_eq!(first_divergence(&[1, 2], &[1, 2]), None);
        assert!(exact_window_match(&[0; 16], &[0; 16]));
        assert!(!exact_window_match(&[0; 15], &[0; 16]));
    }
}
