//! Cold host-side byte builders for absolute Llama `RoPE` tables.
//!
//! The enclosing batch owner retains profile selection, device allocation,
//! upload, kernel launch, and lifetime management. This component only
//! materializes native-endian host bytes for the selected cold path.

use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult, usize_u64,
};
use super::host::allocate_zeroed_host_bytes;

const F32_BYTES_USIZE: usize = std::mem::size_of::<f32>();
const F32_BYTES: u64 = 4;

pub(in crate::llama) type RopeTableBytes = (Box<[u8]>, Box<[u8]>);

/// Derives the absolute `RoPE` position capacity from its preallocated table bytes.
///
/// The enclosing owner reads the device buffer length. This value-only helper
/// preserves the established row-width overflow and floor-division semantics.
pub(in crate::llama) fn absolute_rope_position_count(
    table_byte_len: u64,
    head_dimension: usize,
) -> LlamaBatchExecutorResult<u64> {
    let row_bytes = usize_u64(head_dimension / 2, LlamaBatchExecutorResource::RopeCos)?
        .checked_mul(F32_BYTES)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::RopeCos,
        })?;
    Ok(table_byte_len / row_bytes)
}

/// Returns the absolute `RoPE` half-width and checked cold table element count.
fn absolute_rope_table_shape(
    position_count: usize,
    head_dimension: usize,
) -> LlamaBatchExecutorResult<(usize, usize)> {
    let half = head_dimension / 2;
    let elements =
        position_count
            .checked_mul(half)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeCos,
            })?;
    Ok((half, elements))
}

#[allow(clippy::cast_precision_loss)]
pub(in crate::llama) fn build_absolute_rope_angles(
    position_count: usize,
    head_dimension: usize,
    theta: f32,
) -> LlamaBatchExecutorResult<Box<[u8]>> {
    let (half, elements) = absolute_rope_table_shape(position_count, head_dimension)?;
    let mut angles = allocate_zeroed_host_bytes(elements, F32_BYTES_USIZE)?;
    for position in 0..position_count {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let byte_offset = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(F32_BYTES_USIZE))
                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::RopeCos,
                })?;
            angles[byte_offset..byte_offset + F32_BYTES_USIZE]
                .copy_from_slice(&angle.to_ne_bytes());
        }
    }
    Ok(angles)
}

#[allow(clippy::cast_precision_loss)]
pub(in crate::llama) fn build_absolute_cpu_rope_tables(
    position_count: usize,
    head_dimension: usize,
    theta: f32,
) -> LlamaBatchExecutorResult<RopeTableBytes> {
    let (half, elements) = absolute_rope_table_shape(position_count, head_dimension)?;
    let mut cos = allocate_zeroed_host_bytes(elements, F32_BYTES_USIZE)?;
    let mut sin = allocate_zeroed_host_bytes(elements, F32_BYTES_USIZE)?;
    for position in 0..position_count {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            let byte_offset = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(F32_BYTES_USIZE))
                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::RopeCos,
                })?;
            cos[byte_offset..byte_offset + F32_BYTES_USIZE].copy_from_slice(&cosine.to_ne_bytes());
            sin[byte_offset..byte_offset + F32_BYTES_USIZE].copy_from_slice(&sine.to_ne_bytes());
        }
    }
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], index: usize) -> f32 {
        let offset = index * F32_BYTES_USIZE;
        f32::from_ne_bytes(
            bytes[offset..offset + F32_BYTES_USIZE]
                .try_into()
                .expect("fixed-width f32 bytes"),
        )
    }

    #[test]
    fn absolute_rope_builders_preserve_position_major_native_bytes() {
        let angles = build_absolute_rope_angles(2, 4, 4.0).expect("angle bytes");
        assert_eq!(angles.len(), 4 * F32_BYTES_USIZE);
        assert_eq!(f32_at(&angles, 0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f32_at(&angles, 1).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f32_at(&angles, 2).to_bits(), 1.0_f32.to_bits());
        assert_eq!(f32_at(&angles, 3).to_bits(), 0.5_f32.to_bits());

        let (cos, sin) = build_absolute_cpu_rope_tables(2, 4, 4.0).expect("table bytes");
        let (one_sine, one_cosine) = 1.0_f32.sin_cos();
        let (half_sine, half_cosine) = 0.5_f32.sin_cos();
        assert_eq!(f32_at(&cos, 0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(f32_at(&cos, 1).to_bits(), 1.0_f32.to_bits());
        assert_eq!(f32_at(&cos, 2).to_bits(), one_cosine.to_bits());
        assert_eq!(f32_at(&cos, 3).to_bits(), half_cosine.to_bits());
        assert_eq!(f32_at(&sin, 0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f32_at(&sin, 1).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f32_at(&sin, 2).to_bits(), one_sine.to_bits());
        assert_eq!(f32_at(&sin, 3).to_bits(), half_sine.to_bits());
    }

    #[test]
    fn rope_builders_preserve_overflow_resource_precedence() {
        for error in [
            build_absolute_rope_angles(usize::MAX, 4, 4.0).expect_err("element overflow"),
            build_absolute_cpu_rope_tables(usize::MAX, 4, 4.0).expect_err("table element overflow"),
        ] {
            assert!(matches!(
                error,
                LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::RopeCos,
                }
            ));
        }

        for error in [
            build_absolute_rope_angles(usize::MAX, 2, 4.0).expect_err("byte overflow"),
            build_absolute_cpu_rope_tables(usize::MAX, 2, 4.0).expect_err("table byte overflow"),
        ] {
            assert!(matches!(
                error,
                LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::HostWorkspace,
                }
            ));
        }
    }

    #[test]
    fn absolute_rope_table_shape_preserves_floor_and_overflow_semantics() {
        assert_eq!(
            absolute_rope_table_shape(3, 5).expect("odd head dimension keeps floor half-width"),
            (2, 6)
        );
        assert_eq!(
            absolute_rope_table_shape(3, 1).expect("sub-pair head dimension has zero elements"),
            (0, 0)
        );
        assert!(matches!(
            absolute_rope_table_shape(usize::MAX, 4),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeCos,
            })
        ));
    }

    #[test]
    fn absolute_rope_builders_preserve_empty_floor_shape() {
        for head_dimension in [0, 1] {
            assert!(
                build_absolute_rope_angles(3, head_dimension, 4.0)
                    .expect("sub-pair head dimension keeps an empty angle table")
                    .is_empty()
            );
            let (cos, sin) = build_absolute_cpu_rope_tables(3, head_dimension, 4.0)
                .expect("sub-pair head dimension keeps empty CPU tables");
            assert!(cos.is_empty());
            assert!(sin.is_empty());
        }
        assert!(
            build_absolute_rope_angles(0, usize::MAX, 4.0)
                .expect("zero positions do not precompute a row width")
                .is_empty()
        );
    }

    #[test]
    fn absolute_rope_position_count_preserves_floor_and_overflow_semantics() {
        assert_eq!(
            absolute_rope_position_count(32, 4).expect("exact table rows"),
            4
        );
        assert_eq!(
            absolute_rope_position_count(31, 4).expect("trailing bytes are truncated"),
            3
        );

        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            absolute_rope_position_count(0, usize::MAX),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeCos,
            })
        ));
    }
}
