//! Correctness-first, materialized BF16 GQA attention primitives.

use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::gemm::FIXED37_MAX_REDUCTION_ELEMENTS;
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;

/// Inputs and output for raw grouped-query QK dot products.
#[derive(Debug)]
pub struct QkGqaParams<'a> {
    /// BF16 query tensor with layout `[S, QH, D]`.
    pub query: CudaBufferSpan<'a>,
    /// BF16 key tensor with layout `[S, KVH, D]`.
    pub key: CudaBufferSpan<'a>,
    /// BF16 raw score tensor with layout `[QH, S, S]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Sequence length `S`.
    pub token_count: u64,
    /// Query-head count `QH`.
    pub query_head_count: u64,
    /// Key/value-head count `KVH`; it must divide `QH`.
    pub key_value_head_count: u64,
    /// Elements per query/key head `D`.
    pub head_size: u64,
}

/// Computes raw BF16 QK scores with F32 dot-product accumulation.
///
/// Query head `q` reads key head `q / (QH / KVH)`. The call performs no
/// allocation and completes synchronously on `stream`.
///
/// # Errors
///
/// Returns a dtype, shape, span-capacity, context, launch, or synchronization
/// error.
pub fn qk_gqa(params: &mut QkGqaParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "qk_gqa";
    let shape = validate_gqa_shape(
        OPERATION,
        params.token_count,
        params.query_head_count,
        params.key_value_head_count,
        params.head_size,
    )?;
    require_bf16(OPERATION, "query", params.query.dtype())?;
    require_bf16(OPERATION, "key", params.key.dtype())?;
    require_bf16(OPERATION, "output", params.output.dtype())?;
    require_capacity(OPERATION, "query", params.query.byte_len(), shape.query)?;
    require_capacity(OPERATION, "key", params.key.byte_len(), shape.key_value)?;
    require_capacity(OPERATION, "output", params.output.byte_len(), shape.scores)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.query.buffer(),
            params.key.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::qk_gqa_execute(
            params.query.raw(),
            params.key.raw(),
            params.output.raw(),
            params.token_count,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Computes QK with ascending 37-element F32 left folds and an adjacent
/// balanced tree over the depth-axis partials.
///
/// # Errors
///
/// Returns [`CudaErrorKind::NotSupported`] when `D` exceeds the fixed37
/// partial capacity, in addition to the errors documented by [`qk_gqa`].
pub fn fixed37_qk_gqa(params: &mut QkGqaParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_qk_gqa";
    validate_fixed37_axis(OPERATION, params.head_size)?;
    let shape = validate_gqa_shape(
        OPERATION,
        params.token_count,
        params.query_head_count,
        params.key_value_head_count,
        params.head_size,
    )?;
    require_bf16(OPERATION, "query", params.query.dtype())?;
    require_bf16(OPERATION, "key", params.key.dtype())?;
    require_bf16(OPERATION, "output", params.output.dtype())?;
    require_capacity(OPERATION, "query", params.query.byte_len(), shape.query)?;
    require_capacity(OPERATION, "key", params.key.byte_len(), shape.key_value)?;
    require_capacity(OPERATION, "output", params.output.byte_len(), shape.scores)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.query.buffer(),
            params.key.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_qk_gqa_execute(
            params.query.raw(),
            params.key.raw(),
            params.output.raw(),
            params.token_count,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Parameters for BF16 scale and additive causal masking in place.
#[derive(Debug)]
pub struct ScaleCausalMaskInPlaceParams<'a> {
    /// Mutable BF16 tensor with layout `[QH, S, S]`.
    pub scores: CudaBufferSpanMut<'a>,
    /// Sequence length `S`.
    pub token_count: u64,
    /// Query-head count `QH`.
    pub query_head_count: u64,
    /// Finite positive attention scale, normally `1 / sqrt(D)`.
    pub scale: f32,
}

/// Scales BF16 scores and adds the BF16 causal mask in place.
///
/// Each scaled value is rounded to BF16 before adding the mask. Strictly
/// future entries add the finite BF16 minimum value with bits `0xff7f`; every
/// other entry adds positive zero. The addition result is rounded to BF16.
///
/// # Errors
///
/// Returns a scale, dtype, shape, span-capacity, context, launch, or
/// synchronization error.
pub fn scale_causal_mask_in_place(
    params: &mut ScaleCausalMaskInPlaceParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "scale_causal_mask_in_place";
    let score_bytes = validate_score_shape(OPERATION, params.token_count, params.query_head_count)?;
    validate_scale(OPERATION, params.scale)?;
    require_bf16(OPERATION, "scores", params.scores.dtype())?;
    require_capacity(OPERATION, "scores", params.scores.byte_len(), score_bytes)?;
    validate_resources(OPERATION, stream, &[params.scores.buffer()])?;

    #[cfg(feature = "cuda")]
    {
        ffi::scale_causal_mask_in_place_execute(
            params.scores.raw(),
            params.token_count,
            params.query_head_count,
            params.scale,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Parameters for stable row-wise causal softmax in place.
#[derive(Debug)]
pub struct CausalSoftmaxInPlaceParams<'a> {
    /// Mutable BF16 score/probability tensor with layout `[QH, S, S]`.
    pub scores: CudaBufferSpanMut<'a>,
    /// Sequence length `S`.
    pub token_count: u64,
    /// Query-head count `QH`.
    pub query_head_count: u64,
}

/// Applies stable F32-reduction softmax and writes BF16 probabilities in place.
///
/// The complete final score axis participates in the reduction; causal future
/// values are already finite-minimum mask results from
/// [`scale_causal_mask_in_place`].
///
/// # Errors
///
/// Returns a dtype, shape, span-capacity, context, launch, or synchronization
/// error.
pub fn causal_softmax_in_place(
    params: &mut CausalSoftmaxInPlaceParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "causal_softmax_in_place";
    let score_bytes = validate_score_shape(OPERATION, params.token_count, params.query_head_count)?;
    require_bf16(OPERATION, "scores", params.scores.dtype())?;
    require_capacity(OPERATION, "scores", params.scores.byte_len(), score_bytes)?;
    validate_resources(OPERATION, stream, &[params.scores.buffer()])?;

    #[cfg(feature = "cuda")]
    {
        ffi::causal_softmax_in_place_execute(
            params.scores.raw(),
            params.token_count,
            params.query_head_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Applies softmax with fixed37 maximum and denominator reductions over the
/// complete logical `S` axis, then rounds every probability to BF16.
///
/// NaN input, a `+Inf` maximum, or an all-`-Inf` row produces a complete BF16
/// NaN row. Finite canonical causal-mask values participate normally.
///
/// # Errors
///
/// Returns [`CudaErrorKind::NotSupported`] when `S` exceeds the fixed37
/// partial capacity, in addition to ordinary shape/resource failures.
pub fn fixed37_causal_softmax_in_place(
    params: &mut CausalSoftmaxInPlaceParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_causal_softmax_in_place";
    validate_fixed37_axis(OPERATION, params.token_count)?;
    let score_bytes = validate_score_shape(OPERATION, params.token_count, params.query_head_count)?;
    require_bf16(OPERATION, "scores", params.scores.dtype())?;
    require_capacity(OPERATION, "scores", params.scores.byte_len(), score_bytes)?;
    validate_resources(OPERATION, stream, &[params.scores.buffer()])?;

    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_causal_softmax_in_place_execute(
            params.scores.raw(),
            params.token_count,
            params.query_head_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs and output for grouped-query probability/value dot products.
#[derive(Debug)]
pub struct AvGqaParams<'a> {
    /// BF16 probability tensor with layout `[QH, S, S]`.
    pub probabilities: CudaBufferSpan<'a>,
    /// BF16 value tensor with layout `[S, KVH, D]`.
    pub value: CudaBufferSpan<'a>,
    /// BF16 attention output tensor with layout `[S, QH, D]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Sequence length `S`.
    pub token_count: u64,
    /// Query-head count `QH`.
    pub query_head_count: u64,
    /// Key/value-head count `KVH`; it must divide `QH`.
    pub key_value_head_count: u64,
    /// Elements per value/output head `D`.
    pub head_size: u64,
}

/// Computes BF16 attention output with F32 probability/value accumulation.
///
/// Query head `q` reads value head `q / (QH / KVH)`. The call performs no
/// allocation and completes synchronously on `stream`.
///
/// # Errors
///
/// Returns a dtype, shape, span-capacity, context, launch, or synchronization
/// error.
pub fn av_gqa(params: &mut AvGqaParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "av_gqa";
    let shape = validate_gqa_shape(
        OPERATION,
        params.token_count,
        params.query_head_count,
        params.key_value_head_count,
        params.head_size,
    )?;
    require_bf16(OPERATION, "probabilities", params.probabilities.dtype())?;
    require_bf16(OPERATION, "value", params.value.dtype())?;
    require_bf16(OPERATION, "output", params.output.dtype())?;
    require_capacity(
        OPERATION,
        "probabilities",
        params.probabilities.byte_len(),
        shape.scores,
    )?;
    require_capacity(OPERATION, "value", params.value.byte_len(), shape.key_value)?;
    require_capacity(OPERATION, "output", params.output.byte_len(), shape.query)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.probabilities.buffer(),
            params.value.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::av_gqa_execute(
            params.probabilities.raw(),
            params.value.raw(),
            params.output.raw(),
            params.token_count,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Computes AV with fixed37 reductions over the complete logical `S` axis.
/// The input probabilities are BF16, so the softmax narrowing boundary is
/// preserved before the ascending chunk folds begin.
///
/// # Errors
///
/// Returns [`CudaErrorKind::NotSupported`] when `S` exceeds the fixed37
/// partial capacity, in addition to the errors documented by [`av_gqa`].
pub fn fixed37_av_gqa(params: &mut AvGqaParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_av_gqa";
    validate_fixed37_axis(OPERATION, params.token_count)?;
    let shape = validate_gqa_shape(
        OPERATION,
        params.token_count,
        params.query_head_count,
        params.key_value_head_count,
        params.head_size,
    )?;
    require_bf16(OPERATION, "probabilities", params.probabilities.dtype())?;
    require_bf16(OPERATION, "value", params.value.dtype())?;
    require_bf16(OPERATION, "output", params.output.dtype())?;
    require_capacity(
        OPERATION,
        "probabilities",
        params.probabilities.byte_len(),
        shape.scores,
    )?;
    require_capacity(OPERATION, "value", params.value.byte_len(), shape.key_value)?;
    require_capacity(OPERATION, "output", params.output.byte_len(), shape.query)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.probabilities.buffer(),
            params.value.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_av_gqa_execute(
            params.probabilities.raw(),
            params.value.raw(),
            params.output.raw(),
            params.token_count,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GqaShape {
    query: u64,
    key_value: u64,
    scores: u64,
}

fn validate_gqa_shape(
    operation: &'static str,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
) -> CudaResult<GqaShape> {
    require_nonzero(operation, "token_count", token_count)?;
    require_nonzero(operation, "query_head_count", query_head_count)?;
    require_nonzero(operation, "key_value_head_count", key_value_head_count)?;
    require_nonzero(operation, "head_size", head_size)?;
    if query_head_count % key_value_head_count != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "key_value_head_count must divide query_head_count",
        ));
    }
    Ok(GqaShape {
        query: bf16_bytes(
            operation,
            checked_product3(operation, token_count, query_head_count, head_size)?,
        )?,
        key_value: bf16_bytes(
            operation,
            checked_product3(operation, token_count, key_value_head_count, head_size)?,
        )?,
        scores: validate_score_shape(operation, token_count, query_head_count)?,
    })
}

fn validate_score_shape(
    operation: &'static str,
    token_count: u64,
    query_head_count: u64,
) -> CudaResult<u64> {
    require_nonzero(operation, "token_count", token_count)?;
    require_nonzero(operation, "query_head_count", query_head_count)?;
    bf16_bytes(
        operation,
        checked_product3(operation, query_head_count, token_count, token_count)?,
    )
}

fn checked_product3(
    operation: &'static str,
    first: u64,
    second: u64,
    third: u64,
) -> CudaResult<u64> {
    first
        .checked_mul(second)
        .and_then(|value| value.checked_mul(third))
        .ok_or_else(|| CudaError::out_of_range(operation, "attention shape arithmetic overflow"))
}

fn bf16_bytes(operation: &'static str, element_count: u64) -> CudaResult<u64> {
    element_count
        .checked_mul(CudaDType::BF16.size_bytes())
        .ok_or_else(|| {
            CudaError::out_of_range(operation, "attention byte-length arithmetic overflow")
        })
}

fn require_nonzero(operation: &'static str, field: &'static str, value: u64) -> CudaResult<()> {
    if value == 0 {
        Err(CudaError::invalid_argument(
            operation,
            format!("{field} must be greater than zero"),
        ))
    } else {
        Ok(())
    }
}

fn require_bf16(operation: &'static str, field: &'static str, actual: CudaDType) -> CudaResult<()> {
    if actual == CudaDType::BF16 {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{field} must be bf16, got {actual}"),
        ))
    }
}

fn require_capacity(
    operation: &'static str,
    field: &'static str,
    actual: u64,
    required: u64,
) -> CudaResult<()> {
    if actual >= required {
        Ok(())
    } else {
        Err(CudaError::out_of_range(
            operation,
            format!("{field} requires {required} bytes, span exposes {actual}"),
        ))
    }
}

fn validate_scale(operation: &'static str, scale: f32) -> CudaResult<()> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            "scale must be finite and greater than zero",
        ))
    }
}

fn validate_fixed37_axis(operation: &'static str, element_count: u64) -> CudaResult<()> {
    require_nonzero(operation, "reduction_axis", element_count)?;
    if element_count > FIXED37_MAX_REDUCTION_ELEMENTS {
        return Err(CudaError::new(
            CudaErrorKind::NotSupported,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            format!(
                "reduction axis {element_count} exceeds the fixed37 limit {FIXED37_MAX_REDUCTION_ELEMENTS}"
            ),
        ));
    }
    Ok(())
}

fn validate_resources(
    operation: &'static str,
    stream: &CudaStream,
    buffers: &[&CudaDeviceBuffer],
) -> CudaResult<()> {
    for buffer in buffers {
        ensure_same_context(buffer.context_owner(), &stream.context, operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_fixed37_axis, validate_gqa_shape, validate_scale, validate_score_shape};
    use crate::CudaErrorKind;
    use crate::FIXED37_MAX_REDUCTION_ELEMENTS;

    #[test]
    fn gqa_layout_byte_counts_are_checked() {
        let shape = validate_gqa_shape("test", 5, 6, 2, 4).expect("shape must be valid");
        assert_eq!(shape.query, 5 * 6 * 4 * 2);
        assert_eq!(shape.key_value, 5 * 2 * 4 * 2);
        assert_eq!(shape.scores, 6 * 5 * 5 * 2);

        let error =
            validate_gqa_shape("test", 5, 6, 4, 4).expect_err("KV heads must divide query heads");
        assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    }

    #[test]
    fn attention_rejects_empty_and_overflowing_shapes() {
        let empty = validate_score_shape("test", 0, 1).expect_err("empty S must fail");
        assert_eq!(empty.kind(), CudaErrorKind::InvalidArgument);

        let overflow =
            validate_score_shape("test", u64::MAX, 2).expect_err("score shape overflow must fail");
        assert_eq!(overflow.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn attention_scale_must_be_positive_and_finite() {
        validate_scale("test", 0.125).expect("ordinary scale must pass");
        for invalid in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let error = validate_scale("test", invalid).expect_err("invalid scale must fail");
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }
    }

    #[test]
    fn fixed37_attention_axes_fail_closed_at_the_shared_partial_limit() {
        validate_fixed37_axis("test", FIXED37_MAX_REDUCTION_ELEMENTS)
            .expect("the maximum fixed37 axis must be supported");
        let error = validate_fixed37_axis("test", FIXED37_MAX_REDUCTION_ELEMENTS + 1)
            .expect_err("one element beyond the fixed37 axis must fail");
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
    }
}
