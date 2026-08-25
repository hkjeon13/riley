use std::error;
use std::fmt;

/// Result type returned by allocation-free CPU reference primitives.
pub type ReferenceResult<T> = Result<T, ReferenceError>;

/// A stable shape, parameter, or index failure from a CPU reference primitive.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReferenceError {
    /// A named slice did not have the exact required number of elements.
    LengthMismatch {
        /// Stable input or output name.
        name: &'static str,
        /// Required element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// Dimension multiplication overflowed `usize`.
    DimensionOverflow {
        /// Stable operation name.
        operation: &'static str,
    },
    /// A dimension or scalar parameter violated the operation contract.
    InvalidParameter {
        /// Stable parameter name.
        name: &'static str,
        /// Actionable expected-value description.
        expected: &'static str,
    },
    /// An embedding token ID did not address the supplied vocabulary.
    TokenOutOfRange {
        /// Position in the token ID input.
        position: usize,
        /// Invalid token ID.
        token_id: u32,
        /// Number of embedding rows.
        vocabulary_size: usize,
    },
}

impl fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} requires {expected} elements, but received {actual}"
            ),
            Self::DimensionOverflow { operation } => {
                write!(formatter, "{operation} dimensions overflow usize")
            }
            Self::InvalidParameter { name, expected } => {
                write!(formatter, "invalid {name}; expected {expected}")
            }
            Self::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "token ID {token_id} at position {position} exceeds vocabulary size {vocabulary_size}"
            ),
        }
    }
}

impl error::Error for ReferenceError {}

/// Gathers token rows from a row-major `f32` embedding table.
///
/// The table shape is `[vocabulary_size, hidden_size]` and the output shape is
/// `[token_ids.len(), hidden_size]`. Validation completes before output writes.
///
/// # Errors
///
/// Returns an error for zero vocabulary/hidden size, dimension overflow,
/// mismatched slice lengths, or a token ID outside the vocabulary.
pub fn embedding(
    table: &[f32],
    vocabulary_size: usize,
    hidden_size: usize,
    token_ids: &[u32],
    output: &mut [f32],
) -> ReferenceResult<()> {
    require_nonzero("vocabulary_size", vocabulary_size)?;
    require_nonzero("hidden_size", hidden_size)?;
    let table_len = checked_product("embedding", vocabulary_size, hidden_size)?;
    require_len("table", table_len, table.len())?;
    let output_len = checked_product("embedding", token_ids.len(), hidden_size)?;
    require_len("output", output_len, output.len())?;

    for (position, &token_id) in token_ids.iter().enumerate() {
        let token = usize::try_from(token_id).map_err(|_| ReferenceError::TokenOutOfRange {
            position,
            token_id,
            vocabulary_size,
        })?;
        if token >= vocabulary_size {
            return Err(ReferenceError::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            });
        }
    }

    for (destination, (position, &token_id)) in output
        .chunks_exact_mut(hidden_size)
        .zip(token_ids.iter().enumerate())
    {
        let token = usize::try_from(token_id).map_err(|_| ReferenceError::TokenOutOfRange {
            position,
            token_id,
            vocabulary_size,
        })?;
        let start = token * hidden_size;
        destination.copy_from_slice(&table[start..start + hidden_size]);
    }
    Ok(())
}

/// Applies row-wise Llama `RMSNorm` with deterministic `f32` accumulation.
///
/// `weight.len()` defines the hidden size and `input` may contain any number
/// of complete rows. The output never aliases an input through this safe API.
///
/// # Errors
///
/// Returns an error for empty weights, a non-positive/non-finite epsilon, or
/// incompatible input/output lengths.
pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> ReferenceResult<()> {
    require_nonzero("weight hidden size", weight.len())?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(ReferenceError::InvalidParameter {
            name: "epsilon",
            expected: "a finite value greater than zero",
        });
    }
    if input.len() % weight.len() != 0 {
        return Err(ReferenceError::InvalidParameter {
            name: "input length",
            expected: "a multiple of the RMSNorm hidden size",
        });
    }
    require_len("output", input.len(), output.len())?;

    for (input_row, output_row) in input
        .chunks_exact(weight.len())
        .zip(output.chunks_exact_mut(weight.len()))
    {
        let square_sum = input_row
            .iter()
            .fold(0.0_f32, |sum, &value| sum + value * value);
        #[allow(clippy::cast_precision_loss)]
        let mean_square = square_sum / weight.len() as f32;
        let inverse_rms = (mean_square + epsilon).sqrt().recip();
        for ((destination, &value), &scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            *destination = value * inverse_rms * scale;
        }
    }
    Ok(())
}

/// Adds two same-length `f32` slices elementwise.
///
/// # Errors
///
/// Returns an error when either input length differs from the output length.
pub fn residual_add(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> ReferenceResult<()> {
    require_elementwise_lengths(lhs, rhs, output)?;
    for ((destination, &left), &right) in output.iter_mut().zip(lhs).zip(rhs) {
        *destination = left + right;
    }
    Ok(())
}

/// Applies the `SiLU` activation `x / (1 + exp(-x))` elementwise.
///
/// # Errors
///
/// Returns an error when the input and output lengths differ.
pub fn silu(input: &[f32], output: &mut [f32]) -> ReferenceResult<()> {
    require_len("output", input.len(), output.len())?;
    for (destination, &value) in output.iter_mut().zip(input) {
        *destination = value / (1.0 + (-value).exp());
    }
    Ok(())
}

/// Multiplies an already activated gate by an up-projection elementwise.
///
/// # Errors
///
/// Returns an error when either input length differs from the output length.
pub fn gated_multiply(
    activated_gate: &[f32],
    up: &[f32],
    output: &mut [f32],
) -> ReferenceResult<()> {
    require_elementwise_lengths(activated_gate, up, output)?;
    for ((destination, &gate), &value) in output.iter_mut().zip(activated_gate).zip(up) {
        *destination = gate * value;
    }
    Ok(())
}

/// Applies standard non-interleaved Llama `RoPE` to `[token, head, head_dim]`.
///
/// The first and second half of each head are paired, matching Llama's
/// `rotate_half` convention. `positions.len()` is the token count.
///
/// # Errors
///
/// Returns an error for zero heads, an odd/zero head dimension, invalid theta,
/// dimension overflow, or mismatched input/output lengths.
pub fn llama_rope(
    input: &[f32],
    positions: &[u32],
    head_count: usize,
    head_dimension: usize,
    theta: f32,
    output: &mut [f32],
) -> ReferenceResult<()> {
    require_nonzero("head_count", head_count)?;
    if head_dimension == 0 || head_dimension % 2 != 0 {
        return Err(ReferenceError::InvalidParameter {
            name: "head_dimension",
            expected: "a non-zero even value",
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(ReferenceError::InvalidParameter {
            name: "theta",
            expected: "a finite value greater than zero",
        });
    }
    let per_token = checked_product("llama_rope", head_count, head_dimension)?;
    let element_count = checked_product("llama_rope", positions.len(), per_token)?;
    require_len("input", element_count, input.len())?;
    require_len("output", element_count, output.len())?;

    let half = head_dimension / 2;
    #[allow(clippy::cast_precision_loss)]
    let dimension = head_dimension as f32;
    for (token_index, &position) in positions.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let position = position as f32;
        for head_index in 0..head_count {
            let head_start = (token_index * head_count + head_index) * head_dimension;
            for pair_index in 0..half {
                #[allow(clippy::cast_precision_loss)]
                let exponent = (2 * pair_index) as f32 / dimension;
                let angle = position / theta.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let first_index = head_start + pair_index;
                let second_index = first_index + half;
                let first = input[first_index];
                let second = input[second_index];
                output[first_index] = first * cos - second * sin;
                output[second_index] = second * cos + first * sin;
            }
        }
    }
    Ok(())
}

/// Casts `f32` values to bfloat16 with ties-to-even rounding.
///
/// NaNs use CUDA's canonical bfloat16 representation `0x7fff`, matching the
/// production `__float2bfloat16_rn` conversion rather than preserving payload
/// bits from the wider input.
///
/// # Errors
///
/// Returns an error when the input and output lengths differ.
pub fn cast_f32_to_bf16(input: &[f32], output: &mut [u16]) -> ReferenceResult<()> {
    require_len("output", input.len(), output.len())?;
    for (destination, &value) in output.iter_mut().zip(input) {
        *destination = f32_to_bf16_bits(value);
    }
    Ok(())
}

/// Expands IEEE bfloat16 storage bits to exactly representable `f32` values.
///
/// # Errors
///
/// Returns an error when the input and output lengths differ.
pub fn cast_bf16_to_f32(input: &[u16], output: &mut [f32]) -> ReferenceResult<()> {
    require_len("output", input.len(), output.len())?;
    for (destination, &bits) in output.iter_mut().zip(input) {
        *destination = f32::from_bits(u32::from(bits) << 16);
    }
    Ok(())
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let is_nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
    let rounded = if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        bits.wrapping_add(0x7fff + tie) >> 16
    };
    u16::try_from(rounded).unwrap_or(0x7fff)
}

/// Row-major dimensions and transpose flags for deterministic reference GEMM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmSpec {
    /// Output row count.
    pub m: usize,
    /// Output column count.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Interpret physical lhs storage as `[k, m]` instead of `[m, k]`.
    pub transpose_lhs: bool,
    /// Interpret physical rhs storage as `[n, k]` instead of `[k, n]`.
    pub transpose_rhs: bool,
}

/// Computes deterministic row-major `output = lhs * rhs + bias` in `f32`.
///
/// Bias is optional and, when present, contains one value per output column.
/// Reduction always proceeds in increasing `k` order.
///
/// # Errors
///
/// Returns an error for dimension overflow or an incompatible slice length.
pub fn gemm(
    spec: GemmSpec,
    lhs: &[f32],
    rhs: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> ReferenceResult<()> {
    let lhs_len = checked_product("gemm", spec.m, spec.k)?;
    let rhs_len = checked_product("gemm", spec.k, spec.n)?;
    let output_len = checked_product("gemm", spec.m, spec.n)?;
    require_len("lhs", lhs_len, lhs.len())?;
    require_len("rhs", rhs_len, rhs.len())?;
    require_len("output", output_len, output.len())?;
    if let Some(bias) = bias {
        require_len("bias", spec.n, bias.len())?;
    }

    for row in 0..spec.m {
        for column in 0..spec.n {
            let mut accumulator = bias.map_or(0.0, |values| values[column]);
            for depth in 0..spec.k {
                let lhs_index = if spec.transpose_lhs {
                    depth * spec.m + row
                } else {
                    row * spec.k + depth
                };
                let rhs_index = if spec.transpose_rhs {
                    column * spec.k + depth
                } else {
                    depth * spec.n + column
                };
                accumulator += lhs[lhs_index] * rhs[rhs_index];
            }
            output[row * spec.n + column] = accumulator;
        }
    }
    Ok(())
}

fn checked_product(operation: &'static str, lhs: usize, rhs: usize) -> ReferenceResult<usize> {
    lhs.checked_mul(rhs)
        .ok_or(ReferenceError::DimensionOverflow { operation })
}

fn require_len(name: &'static str, expected: usize, actual: usize) -> ReferenceResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(ReferenceError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}

fn require_nonzero(name: &'static str, value: usize) -> ReferenceResult<()> {
    if value == 0 {
        Err(ReferenceError::InvalidParameter {
            name,
            expected: "a value greater than zero",
        })
    } else {
        Ok(())
    }
}

fn require_elementwise_lengths(lhs: &[f32], rhs: &[f32], output: &[f32]) -> ReferenceResult<()> {
    require_len("rhs", lhs.len(), rhs.len())?;
    require_len("output", lhs.len(), output.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn embedding_gathers_odd_rows_and_fails_before_writing() {
        let table: Vec<f32> = (0_u16..15).map(f32::from).collect();
        let mut output = [-1.0; 9];
        embedding(&table, 5, 3, &[4, 0, 2], &mut output).unwrap();
        assert_close(
            &output,
            &[12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0],
            0.0,
        );

        output.fill(-1.0);
        assert!(matches!(
            embedding(&table, 5, 3, &[0, 5, 1], &mut output),
            Err(ReferenceError::TokenOutOfRange {
                position: 1,
                token_id: 5,
                vocabulary_size: 5
            })
        ));
        assert_close(&output, &[-1.0; 9], 0.0);

        assert!(matches!(
            embedding(&[], 0, 3, &[], &mut []),
            Err(ReferenceError::InvalidParameter {
                name: "vocabulary_size",
                ..
            })
        ));
    }

    #[test]
    fn rms_norm_uses_rowwise_f32_accumulation() {
        let input = [3.0, 4.0, 0.0, 0.0];
        let weight = [2.0, 0.5];
        let mut output = [0.0; 4];
        rms_norm(&input, &weight, 1.0e-5, &mut output).unwrap();
        let inverse = (12.5_f32 + 1.0e-5).sqrt().recip();
        assert_close(
            &output,
            &[3.0 * inverse * 2.0, 4.0 * inverse * 0.5, 0.0, 0.0],
            1.0e-6,
        );

        assert!(matches!(
            rms_norm(&input, &weight, 0.0, &mut output),
            Err(ReferenceError::InvalidParameter {
                name: "epsilon",
                ..
            })
        ));

        let extremes = [1.0e-20, -1.0e-20, 1.0e10, -1.0e10];
        rms_norm(&extremes, &[1.0, 1.0], 1.0e-5, &mut output).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert_close(&output[2..], &[1.0, -1.0], 1.0e-6);
    }

    #[test]
    fn elementwise_primitives_match_scalar_formulas() {
        let lhs = [-2.0, 0.0, 3.0];
        let rhs = [0.5, 2.0, -4.0];
        let mut output = [0.0; 3];
        residual_add(&lhs, &rhs, &mut output).unwrap();
        assert_close(&output, &[-1.5, 2.0, -1.0], 0.0);

        silu(&lhs, &mut output).unwrap();
        let expected = lhs.map(|value| value / (1.0 + (-value).exp()));
        assert_close(&output, &expected, 1.0e-7);

        gated_multiply(&output, &rhs, &mut [0.0; 2]).unwrap_err();
        let mut product = [0.0; 3];
        gated_multiply(&output, &rhs, &mut product).unwrap();
        assert_close(
            &product,
            &[
                expected[0] * rhs[0],
                expected[1] * rhs[1],
                expected[2] * rhs[2],
            ],
            1.0e-7,
        );
    }

    #[test]
    fn llama_rope_pairs_vector_halves() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        llama_rope(&input, &[1], 1, 4, 10_000.0, &mut output).unwrap();
        let (sin_0, cos_0) = 1.0_f32.sin_cos();
        let (sin_1, cos_1) = 0.01_f32.sin_cos();
        assert_close(
            &output,
            &[
                cos_0 - 3.0 * sin_0,
                2.0 * cos_1 - 4.0 * sin_1,
                3.0 * cos_0 + sin_0,
                4.0 * cos_1 + 2.0 * sin_1,
            ],
            1.0e-6,
        );

        assert!(matches!(
            llama_rope(&input, &[0], 1, 3, 10_000.0, &mut output),
            Err(ReferenceError::InvalidParameter {
                name: "head_dimension",
                ..
            })
        ));
    }

    #[test]
    fn bfloat16_cast_rounds_to_even_and_preserves_special_values() {
        let tie_down = f32::from_bits(0x3f80_8000);
        let tie_up = f32::from_bits(0x3f81_8000);
        let input = [1.0, -2.5, tie_down, tie_up, f32::INFINITY, f32::NAN];
        let mut bits = [0_u16; 6];
        cast_f32_to_bf16(&input, &mut bits).unwrap();
        assert_eq!(&bits[..5], &[0x3f80, 0xc020, 0x3f80, 0x3f82, 0x7f80]);
        assert_eq!(bits[5], 0x7fff);

        let mut expanded = [0.0; 6];
        cast_bf16_to_f32(&bits, &mut expanded).unwrap();
        assert_eq!(expanded[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(expanded[1].to_bits(), (-2.5_f32).to_bits());
        assert!(expanded[4].is_infinite());
        assert!(expanded[5].is_nan());
    }

    #[test]
    fn gemm_supports_bias_odd_shapes_and_transposed_storage() {
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = [0.0; 4];
        gemm(
            GemmSpec {
                m: 2,
                n: 2,
                k: 3,
                transpose_lhs: false,
                transpose_rhs: false,
            },
            &lhs,
            &rhs,
            Some(&[0.5, -0.5]),
            &mut output,
        )
        .unwrap();
        assert_close(&output, &[58.5, 63.5, 139.5, 153.5], 0.0);

        let lhs_transposed = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let rhs_transposed = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        gemm(
            GemmSpec {
                m: 2,
                n: 2,
                k: 3,
                transpose_lhs: true,
                transpose_rhs: true,
            },
            &lhs_transposed,
            &rhs_transposed,
            None,
            &mut output,
        )
        .unwrap();
        assert_close(&output, &[58.0, 64.0, 139.0, 154.0], 0.0);
    }

    #[test]
    fn gemm_rejects_overflow_and_length_mismatch() {
        let mut output = [];
        assert!(matches!(
            gemm(
                GemmSpec {
                    m: usize::MAX,
                    n: 0,
                    k: 2,
                    transpose_lhs: false,
                    transpose_rhs: false,
                },
                &[],
                &[],
                None,
                &mut output,
            ),
            Err(ReferenceError::DimensionOverflow { operation: "gemm" })
        ));
        assert!(matches!(
            gemm(
                GemmSpec {
                    m: 1,
                    n: 1,
                    k: 2,
                    transpose_lhs: false,
                    transpose_rhs: false,
                },
                &[1.0],
                &[1.0, 2.0],
                None,
                &mut [0.0],
            ),
            Err(ReferenceError::LengthMismatch { name: "lhs", .. })
        ));
    }
}
