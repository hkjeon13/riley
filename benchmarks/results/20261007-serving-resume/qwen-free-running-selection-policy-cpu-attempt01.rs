use std::error::Error;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;
const QWEN3B_VOCABULARY_SIZE: usize = 151936;
const BF16_BYTES: usize = 2;
fn decode_bf16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

// Frozen original HF oracle policy: scan all logits for finiteness, select only
// tokenizer-addressable IDs; numerical ties (including signed zero) use lower ID.
fn qwen_addressable_greedy_token(row: &[u8]) -> TestResult<u32> {
    if row.len() != QWEN3B_VOCABULARY_SIZE * BF16_BYTES {
        return Err("greedy BF16 logit row extent differs".into());
    }
    let mut maximum = f32::NEG_INFINITY;
    let mut selected = None;
    for (token, bytes) in row.chunks_exact(BF16_BYTES).enumerate() {
        let value = decode_bf16_le(bytes);
        if !value.is_finite() {
            return Err(format!("non-finite greedy logit at token {token}").into());
        }
        if token < 151665 && value > maximum {
            maximum = value;
            selected = Some(u32::try_from(token)?);
        }
    }
    selected.ok_or_else(|| "greedy addressable vocabulary is empty".into())
}

#[test]
fn qwen_addressable_greedy_selection_policy() -> TestResult {
    let mut row = vec![0; QWEN3B_VOCABULARY_SIZE * BF16_BYTES];
    // Equal numerical zeros preserve the lower index despite different sign bits.
    row[..2].copy_from_slice(&0x8000_u16.to_le_bytes());
    assert_eq!(qwen_addressable_greedy_token(&row)?, 0);
    row[7 * 2..8 * 2].copy_from_slice(&0x3f80_u16.to_le_bytes());
    row[11 * 2..12 * 2].copy_from_slice(&0x3f80_u16.to_le_bytes());
    row[151665 * 2..151666 * 2].copy_from_slice(&0x7f7f_u16.to_le_bytes());
    assert_eq!(qwen_addressable_greedy_token(&row)?, 7);
    // Invalid non-addressable values still invalidate the numerical observation.
    row[151665 * 2..151666 * 2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(qwen_addressable_greedy_token(&row).is_err());
    assert!(qwen_addressable_greedy_token(&row[..row.len() - 2]).is_err());
    Ok(())
}

