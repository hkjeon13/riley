//! Bounded greedy speculation policy and completed-append KV settlement.
//!
//! Target verification must consume `[pending_token, draft...]` with the same
//! numerical and sampling policy as ordinary target decode. Its K+1 argmax rows
//! predict the K drafts and one bonus token. This module neither runs that GPU
//! verification nor proves its equivalence. No Python or draft model is called.
use crate::paged_kv::{
    KvBlockPool, PagedKvError, PagedKvResult, SequenceReservation, SequenceState,
};

pub const MAX_DRAFT_TOKENS: usize = 8;
const MAX_LOOKUP_NGRAM: usize = 8;
const MAX_LOOKUP_HISTORY: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stop {
    Eos,
    Length,
    Cancelled,
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyError {
    DraftTooLong,
    TargetRowCount,
    InvalidVocabulary,
    TokenOutOfRange,
    InvalidStopPosition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GreedyDecision {
    output: [u32; MAX_DRAFT_TOKENS + 1],
    length: usize,
    draft_length: usize,
    accepted: usize,
    stop: Option<Stop>,
}
impl GreedyDecision {
    pub fn tokens(&self) -> &[u32] {
        &self.output[..self.length]
    }
    pub fn accepted_draft_tokens(&self) -> usize {
        self.accepted
    }
    pub fn stop(&self) -> Option<Stop> {
        self.stop
    }
    /// Stop-token/string processing may shorten only the verified output prefix.
    pub fn stop_after(&mut self, count: usize) -> Result<(), VerifyError> {
        if count == 0 || count > self.length {return Err(VerifyError::InvalidStopPosition);}
        self.output[count..].fill(0);self.length=count;
        self.accepted=self.accepted.min(count);self.stop=Some(Stop::External);Ok(())
    }
    /// The final emitted token is not in the retained KV; decode it next.
    pub fn pending_token(&self) -> Option<u32> {
        if self.stop.is_some() {
            None
        } else {
            self.tokens().last().copied()
        }
    }
    /// Includes the old pending input and excludes the last emitted output.
    pub fn committed_input_tokens(&self) -> usize {
        self.length
    }
}

/// Greedy only: target argmax token IDs must already include request processors.
/// All rows are validated even when a mismatch/EOS would stop earlier.
/// Storage is fixed and this function performs no allocation.
pub fn verify_greedy(
    draft: &[u32],
    target: &[u32],
    vocabulary: u32,
    remaining: usize,
    eos: Option<u32>,
    cancelled: bool,
) -> Result<GreedyDecision, VerifyError> {
    if draft.len() > MAX_DRAFT_TOKENS {
        return Err(VerifyError::DraftTooLong);
    }
    if target.len() != draft.len() + 1 {
        return Err(VerifyError::TargetRowCount);
    }
    if vocabulary == 0 {
        return Err(VerifyError::InvalidVocabulary);
    }
    if draft.iter().chain(target).any(|&t| t >= vocabulary) || eos.is_some_and(|t| t >= vocabulary)
    {
        return Err(VerifyError::TokenOutOfRange);
    }
    let mut result = GreedyDecision {
        output: [0; MAX_DRAFT_TOKENS + 1],
        length: 0,
        draft_length: draft.len(),
        accepted: 0,
        stop: None,
    };
    if cancelled {
        result.stop = Some(Stop::Cancelled);
        return Ok(result);
    }
    if remaining == 0 {
        result.stop = Some(Stop::Length);
        return Ok(result);
    }
    for (index, &token) in target.iter().enumerate() {
        let matched = index < draft.len() && draft[index] == token;
        result.output[result.length] = token;
        result.length += 1;
        result.accepted += usize::from(matched);
        if eos == Some(token) {
            result.stop = Some(Stop::Eos);
            break;
        }
        if result.length == remaining {
            result.stop = Some(Stop::Length);
            break;
        }
        if !matched {
            break;
        }
    }
    Ok(result)
}

/// Settle only after GPU quiescence and proof that writes were confined to the
/// reserved append. Cancellation alone is not that proof; unknown writes must
/// use `SequenceState::poison`. This function is not a device completion fence.
///
/// A partial result retains `[old pending, accepted draft inputs...]` and
/// discards rejected suffix pages. EOS/length truncation also excludes the last
/// emitted token's KV. Same-page suffix bytes may remain physically present but
/// are outside the published logical length and must be overwritten on append.
/// Errors consume the detached reservation, like the underlying KV API; the
/// sequence retains authoritative pending state for explicit recovery/poison.
pub fn settle_completed_greedy(
    sequence: &mut SequenceState,
    pool: &mut KvBlockPool,
    mut reservation: SequenceReservation,
    round_start: u32,
    decision: &GreedyDecision,
) -> PagedKvResult<()> {
    let target = usize::try_from(round_start)
        .ok()
        .and_then(|n| n.checked_add(decision.draft_length + 1));
    if sequence.logical_length() != round_start
        || target != Some(reservation.target_logical_length() as usize)
    {
        return Err(PagedKvError::ReservationMismatch);
    }
    sequence.execution_block_table(pool, Some(&reservation))?;
    if decision.length == decision.draft_length + 1 {
        sequence.commit(pool, reservation)?;
    } else {
        if decision.length != 0 {
            sequence.commit_prefix(
                pool,
                &mut reservation,
                round_start as usize + decision.length,
            )?;
        }
        sequence.discard_completed_append(pool, reservation)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LookupDraft {
    tokens: [u32; MAX_DRAFT_TOKENS],
    length: usize,
}
impl LookupDraft {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens[..self.length]
    }
}

/// Additional-model-free proposal source: longest suffix match first, newest
/// completed occurrence on ties. Search is bounded to the last `lookback`
/// tokens (capped at 4096), with n-gram width capped at eight; proposals
/// never include unseen tokens or exceed eight tokens.
/// Acceptance still requires target verification. No heap allocation occurs.
pub fn prompt_lookup(history: &[u32], max_ngram: usize, lookback: usize) -> LookupDraft {
    let mut result = LookupDraft {
        tokens: [0; MAX_DRAFT_TOKENS],
        length: 0,
    };
    let lookback = lookback.min(MAX_LOOKUP_HISTORY);
    let lower = history.len().saturating_sub(lookback);
    for width in (1..=max_ngram
        .min(MAX_LOOKUP_NGRAM)
        .min(history.len())
        .min(lookback))
        .rev()
    {
        let suffix = history.len() - width;
        for start in (lower..suffix).rev() {
            let end = start + width;
            if end > suffix || history[start..end] != history[suffix..] {
                continue;
            }
            let length = (history.len() - end).min(MAX_DRAFT_TOKENS);
            result.tokens[..length].copy_from_slice(&history[end..end + length]);
            result.length = length;
            return result;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::KvLayout;
    #[test]
    fn mismatch_bonus_eos_length_cancel_and_invalid_rows() {
        let all = verify_greedy(&[2, 3], &[2, 3, 4], 10, 9, None, false).unwrap();
        assert_eq!(all.tokens(), &[2, 3, 4]);
        assert_eq!(all.pending_token(), Some(4));
        assert_eq!(all.accepted_draft_tokens(), 2);
        let partial = verify_greedy(&[2, 3], &[2, 7, 4], 10, 9, None, false).unwrap();
        assert_eq!(partial.tokens(), &[2, 7]);
        assert_eq!(partial.accepted_draft_tokens(), 1);
        assert_eq!(
            verify_greedy(&[2], &[7, 4], 10, 9, None, false)
                .unwrap()
                .tokens(),
            &[7]
        );
        let eos = verify_greedy(&[2, 3], &[2, 3, 4], 10, 2, Some(3), false).unwrap();
        assert_eq!(eos.tokens(), &[2, 3]);
        assert_eq!(eos.stop(), Some(Stop::Eos));
        assert_eq!(eos.committed_input_tokens(), 2);
        assert_eq!(eos.pending_token(), None);
        assert_eq!(
            verify_greedy(&[2, 3], &[2, 3, 4], 10, 1, None, false)
                .unwrap()
                .tokens(),
            &[2]
        );
        assert!(verify_greedy(&[2], &[2, 4], 10, 9, None, true)
            .unwrap()
            .tokens()
            .is_empty());
        assert!(verify_greedy(&[], &[4], 10, 0, None, false)
            .unwrap()
            .tokens()
            .is_empty());
        assert_eq!(
            verify_greedy(&[2], &[2], 10, 9, None, false),
            Err(VerifyError::TargetRowCount)
        );
        assert_eq!(
            verify_greedy(&[2], &[7, 99], 10, 9, None, false),
            Err(VerifyError::TokenOutOfRange)
        );
    }
    #[test]
    fn real_kv_pool_settlement_across_page_boundaries() {
        for start in [0, 1, 15, 16, 17, 31, 32] {
            for accepted in 0..=8 {
                for limit in [0, 1, 4, 9] {
                    let mut pool =
                        KvBlockPool::new(KvLayout::checked(1, 8, 1, 64).unwrap()).unwrap();
                    let mut sequence = pool.create_sequence(64).unwrap();
                    let initial = sequence.reserve_to(&mut pool, start).unwrap();
                    sequence.commit(&mut pool, initial).unwrap();
                    let reservation = sequence.reserve_to(&mut pool, start + 9).unwrap();
                    let mut target = [2; 9];
                    if accepted < 8 {
                        target[accepted] = 3;
                    }
                    let decision = verify_greedy(&[2; 8], &target, 10, limit, None, false).unwrap();
                    settle_completed_greedy(
                        &mut sequence,
                        &mut pool,
                        reservation,
                        start as u32,
                        &decision,
                    )
                    .unwrap();
                    let expected = start + (accepted + 1).min(limit);
                    assert_eq!(sequence.logical_length() as usize, expected);
                    assert_eq!(
                        sequence.block_table().unwrap().logical_length() as usize,
                        expected
                    );
                    assert_eq!(pool.stats().allocated_block_count(), expected.div_ceil(16));
                    let append = sequence.reserve_to(&mut pool, expected + 1).unwrap();
                    sequence
                        .execution_block_table(&pool, Some(&append))
                        .unwrap();
                    sequence
                        .discard_completed_append(&mut pool, append)
                        .unwrap();
                    sequence.close(&mut pool).unwrap();
                    assert_eq!(pool.stats().allocated_block_count(), 0);
                }
            }
        }
    }
    #[test]
    fn stale_start_rejected_without_publishing_reserved_suffix() {
        let mut pool = KvBlockPool::new(KvLayout::checked(1, 8, 1, 64).unwrap()).unwrap();
        let mut sequence = pool.create_sequence(64).unwrap();
        let reservation = sequence.reserve_to(&mut pool, 3).unwrap();
        let decision = verify_greedy(&[2], &[2, 3], 10, 9, None, false).unwrap();
        assert_eq!(
            settle_completed_greedy(&mut sequence, &mut pool, reservation, 1, &decision),
            Err(PagedKvError::ReservationMismatch)
        );
        assert_eq!(sequence.logical_length(), 0);
        sequence.rollback_abandoned_reservation(&mut pool).unwrap();
        assert_eq!(pool.stats().allocated_block_count(), 0);
    }
    #[test]
    fn verification_matches_independent_autoregressive_target() {
        fn next(history: &[u32]) -> u32 {
            history
                .iter()
                .enumerate()
                .fold(1, |sum, (i, token)| (sum + (i as u32 + 1) * token) % 5)
        }
        // Enumerate every length-four proposal over five tokens. Target rows
        // after a wrong proposal have the wrong history and MUST be ignored.
        for encoded in 0..625 {
            let mut draft = [0; 4];
            let mut value = encoded;
            for token in &mut draft {
                *token = value % 5;
                value /= 5;
            }
            let mut history = vec![1, 2, 4];
            let mut rows = [0; 5];
            for i in 0..5 {
                rows[i] = next(&history);
                if i < 4 {
                    history.push(draft[i]);
                }
            }
            for remaining in 0..=5 {
                for eos in [None, Some(0), Some(4)] {
                    let decision = verify_greedy(&draft, &rows, 5, remaining, eos, false).unwrap();
                    let mut serial = vec![1, 2, 4];
                    for (i, &actual) in decision.tokens().iter().enumerate() {
                        assert_eq!(actual, next(&serial));
                        serial.push(actual);
                        if eos == Some(actual) {
                            assert_eq!(i + 1, decision.tokens().len());
                        }
                    }
                    assert!(decision.tokens().len() <= remaining);
                }
            }
        }
    }
    #[test]
    fn lookup_is_bounded_and_never_makes_tokens_up() {
        assert!(prompt_lookup(&[1, 2, 3], 4, 32).tokens().is_empty());
        assert_eq!(
            prompt_lookup(&[1, 2, 7, 8, 1, 2], 4, 32).tokens(),
            &[7, 8, 1, 2]
        );
        assert!(prompt_lookup(&[1, 2, 7, 8, 1, 2], 4, 2).tokens().is_empty());
        assert!(prompt_lookup(&[1, 1, 1], 0, 32).tokens().is_empty());
        assert!(prompt_lookup(&[], usize::MAX, usize::MAX)
            .tokens()
            .is_empty());
    }
}

/// Fixed CUDA verification record format. Every inactive record is checked too.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationRecordError { Shape, Slot, Status, Token, Padding }
pub fn parse_verification_tokens(bytes:&[u8],active:usize)->Result<[u32;32],VerificationRecordError>{parse_verification_tokens_capacity::<32>(bytes,active)}
pub fn parse_verification_tokens_capacity<const N:usize>(bytes: &[u8], active: usize) -> Result<[u32;N], VerificationRecordError> {
    use VerificationRecordError::*;
    if !matches!(N,32|256) || bytes.len()!=N*16 || !(1..=N).contains(&active) { return Err(Shape); }
    let mut tokens=[0;N];
    for (slot, record) in bytes.chunks_exact(16).enumerate() {
        let word=|i| u32::from_le_bytes(record[i..i+4].try_into().unwrap());
        let (token,error,index,valid)=(word(0),word(4),word(8),word(12));
        if index!=slot as u32 {return Err(Slot);}
        if slot<active {
            if error!=0 || valid!=1 {return Err(Status);}
            if token>=49152 {return Err(Token);}
            tokens[slot]=token;
        } else if token!=0 || error!=0 || valid!=0 {return Err(Padding);}
    }
    Ok(tokens)
}
#[cfg(test)]
mod verification_record_tests {
    use super::*;
    fn records(active:usize)->Vec<u8> {
        (0..32u32).flat_map(|i| [if (i as usize)<active {49151-i}else{0},0,i,u32::from((i as usize)<active)]).flat_map(u32::to_le_bytes).collect()
    }
    #[test]
    fn verification_records_validate_every_field_and_padding() {
        for active in 1..=32 {
            let bytes=records(active);
            let parsed=parse_verification_tokens(&bytes,active).unwrap();
            for slot in 0..32 {
                assert_eq!(parsed[slot],if slot<active {49151-slot as u32}else{0});
                for (offset,value) in [(0,49152u32),(4,1),(8,99),(12,2)] {
                    let mut corrupt=bytes.clone();
                    corrupt[slot*16+offset..slot*16+offset+4].copy_from_slice(&value.to_le_bytes());
                    assert!(parse_verification_tokens(&corrupt,active).is_err(),"active={active} slot={slot} field={offset}");
                }
            }
        }
        assert!(parse_verification_tokens(&records(1)[..511],1).is_err());
        assert!(parse_verification_tokens(&records(1),0).is_err());
        assert!(parse_verification_tokens(&records(32),33).is_err());
    }
}

#[cfg(test)]
mod wide_verification_record_tests {
    use super::*;
    #[test]
    fn wide_records_validate_upper_slots_and_inactive_tail(){
        for active in [1,31,32,33,127,255,256] {
            let bytes:Vec<_>=(0..256u32).flat_map(|i|[if (i as usize)<active{i}else{0},0,i,u32::from((i as usize)<active)]).flat_map(u32::to_le_bytes).collect();
            let parsed=parse_verification_tokens_capacity::<256>(&bytes,active).unwrap();assert_eq!(parsed[active-1],active as u32-1);
            for slot in [0,31,32,127,255]{for field in [0,4,8,12]{let mut corrupt=bytes.clone();corrupt[slot*16+field..slot*16+field+4].copy_from_slice(&999999u32.to_le_bytes());assert!(parse_verification_tokens_capacity::<256>(&corrupt,active).is_err());}}
        }
    }
}
