use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use serde::Deserialize;

use crate::{ArtifactKind, LoadLimits, ModelError, ModelResult, strict_json};

const TOKENIZER_VERSION: &str = "1.0";
const MAX_ADDED_TOKEN_BYTES: usize = 4096;
const MAX_TOTAL_ADDED_TOKEN_BYTES: usize = 1024 * 1024;

/// Options controlling tokenizer encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeOptions {
    /// Requests configured post-processing special tokens.
    ///
    /// The supported `SmolLM2` artifact has no post-processor, so this flag does
    /// not add BOS or EOS. Special-token text already present in the input is
    /// still recognized exactly.
    pub add_special_tokens: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
        }
    }
}

/// Options controlling tokenizer decoding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeOptions {
    /// Omits IDs declared as special added tokens.
    pub skip_special_tokens: bool,
}

/// Python-free tokenizer interface consumed by model loading and generation.
///
/// Artifact parsing is resource-bounded. Request-level input and output limits
/// remain the caller's responsibility until the serving API applies its model
/// context and admission-control policy.
pub trait Tokenizer {
    /// Encodes UTF-8 text into model token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when a pre-token cannot be represented by the strict
    /// vocabulary and merge table.
    fn encode(&self, input: &str, options: EncodeOptions) -> ModelResult<Vec<u32>>;

    /// Decodes model token IDs into UTF-8 text.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown IDs or invalid byte-alphabet tokens.
    fn decode(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<String>;
}

/// Strict first-party backend for the `SmolLM2` `ByteLevel` BPE artifact shape.
pub struct SmolLm2Tokenizer {
    vocab: BTreeMap<String, u32>,
    vocab_by_id: Box<[String]>,
    merge_ranks: BTreeMap<String, BTreeMap<String, usize>>,
    special_ids: BTreeSet<u32>,
    added_trie: AddedTrie,
    byte_encoder: [char; 256],
    byte_decoder: BTreeMap<char, u8>,
}

impl SmolLm2Tokenizer {
    /// Parses a tokenizer artifact with production limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, duplicate, oversized, or unsupported
    /// tokenizer JSON and for inconsistent vocabularies or merge tables.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::default())
    }

    /// Parses a tokenizer artifact with explicit untrusted-input limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, duplicate, oversized, or unsupported
    /// tokenizer JSON and for inconsistent vocabularies or merge tables.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        enforce_count_limit("tokenizer JSON", limits.tokenizer_bytes(), input.len())?;
        let raw: RawTokenizer = strict_json::from_slice(input, ArtifactKind::Tokenizer)?;
        validate_format(&raw)?;
        enforce_count_limit(
            "tokenizer vocabulary",
            usize_to_u64(limits.vocabulary_entries()),
            raw.model.vocab.len(),
        )?;
        enforce_count_limit(
            "tokenizer merges",
            usize_to_u64(limits.merges()),
            raw.model.merges.len(),
        )?;
        enforce_count_limit(
            "tokenizer added tokens",
            usize_to_u64(limits.added_tokens()),
            raw.added_tokens.len(),
        )?;

        let (byte_encoder, byte_decoder) = byte_maps();
        let (vocab, vocab_by_id) = validate_vocab(raw.model.vocab)?;
        let (special_ids, added_trie) =
            validate_added_tokens(raw.added_tokens, &vocab, vocab_by_id.len())?;
        validate_byte_alphabet(&vocab, &special_ids, &byte_decoder)?;
        let merge_ranks = validate_merges(raw.model.merges, &vocab)?;

        Ok(Self {
            vocab,
            vocab_by_id: vocab_by_id.into_boxed_slice(),
            merge_ranks,
            special_ids,
            added_trie,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Returns the dense vocabulary size.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.vocab_by_id.len()
    }

    /// Returns whether the vocabulary contains an ID.
    #[must_use]
    pub fn contains_id(&self, id: u32) -> bool {
        usize::try_from(id)
            .ok()
            .is_some_and(|index| index < self.vocab_by_id.len())
    }

    /// Returns the serialized vocabulary token for an ID.
    #[must_use]
    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.vocab_by_id
            .get(usize::try_from(id).ok()?)
            .map(String::as_str)
    }

    /// Returns whether an ID is a declared special added token.
    #[must_use]
    pub fn is_special_id(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    fn encode_plain(&self, input: &str, output: &mut Vec<u32>) -> ModelResult<()> {
        for digit_segment in split_individual_decimal_digits(input) {
            for piece in gpt2_pieces(digit_segment) {
                let mapped = piece
                    .as_bytes()
                    .iter()
                    .map(|byte| self.byte_encoder[usize::from(*byte)])
                    .collect::<String>();
                output.extend(self.bpe(&mapped)?);
            }
        }
        Ok(())
    }

    fn bpe(&self, mapped: &str) -> ModelResult<Vec<u32>> {
        let mut nodes = mapped
            .chars()
            .map(|character| BpeNode::new(character.to_string()))
            .collect::<Vec<_>>();
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        for index in 0..nodes.len() {
            nodes[index].previous = index.checked_sub(1);
            nodes[index].next = (index + 1 < nodes.len()).then_some(index + 1);
        }

        let mut candidates = BinaryHeap::new();
        for left in 0..nodes.len().saturating_sub(1) {
            push_candidate(&nodes, &self.merge_ranks, left, left + 1, &mut candidates);
        }
        while let Some(Reverse((rank, position, left, right))) = candidates.pop() {
            if !candidate_is_current(&nodes, &self.merge_ranks, rank, left, right) {
                continue;
            }
            merge_nodes(
                &mut nodes,
                &self.merge_ranks,
                left,
                right,
                position,
                &mut candidates,
            );
        }

        let mut ids = Vec::new();
        let mut current = Some(0_usize);
        while let Some(index) = current {
            let node = &nodes[index];
            if node.alive {
                let id = self.vocab.get(&node.text).copied().ok_or_else(|| {
                    invalid(format!(
                        "BPE produced token absent from vocabulary: {:?}",
                        node.text
                    ))
                })?;
                ids.push(id);
            }
            current = node.next;
        }
        Ok(ids)
    }
}

impl Tokenizer for SmolLm2Tokenizer {
    fn encode(&self, input: &str, _options: EncodeOptions) -> ModelResult<Vec<u32>> {
        let mut output = Vec::new();
        let mut cursor = 0_usize;
        while cursor < input.len() {
            let Some((start, end, id)) = self.added_trie.next_match(input, cursor) else {
                self.encode_plain(&input[cursor..], &mut output)?;
                break;
            };
            self.encode_plain(&input[cursor..start], &mut output)?;
            output.push(id);
            cursor = end;
        }
        Ok(output)
    }

    fn decode(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<String> {
        let mut decoded_bytes = Vec::new();
        for &id in ids {
            let token = self
                .token_for_id(id)
                .ok_or(ModelError::InvalidTokenId { id })?;
            if self.is_special_id(id) && options.skip_special_tokens {
                continue;
            }
            let mapped = token
                .chars()
                .map(|character| self.byte_decoder.get(&character).copied())
                .collect::<Option<Vec<_>>>();
            if let Some(mapped) = mapped {
                decoded_bytes.extend(mapped);
            } else {
                decoded_bytes.extend_from_slice(token.as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&decoded_bytes).into_owned())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTokenizer {
    version: String,
    truncation: serde_json::Value,
    padding: serde_json::Value,
    added_tokens: Vec<RawAddedToken>,
    normalizer: serde_json::Value,
    pre_tokenizer: RawSequence,
    post_processor: serde_json::Value,
    decoder: RawByteLevel,
    model: RawBpe,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAddedToken {
    id: u32,
    content: String,
    single_word: JsonBool,
    lstrip: JsonBool,
    rstrip: JsonBool,
    normalized: JsonBool,
    special: JsonBool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSequence {
    #[serde(rename = "type")]
    kind: String,
    pretokenizers: Vec<RawPreTokenizer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, tag = "type")]
enum RawPreTokenizer {
    Digits {
        individual_digits: bool,
    },
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawByteLevel {
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "add_prefix_space")]
    _add_prefix_space: bool,
    #[serde(rename = "trim_offsets")]
    _trim_offsets: bool,
    #[serde(rename = "use_regex")]
    _use_regex: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBpe {
    #[serde(rename = "type")]
    kind: String,
    dropout: serde_json::Value,
    unk_token: serde_json::Value,
    continuing_subword_prefix: serde_json::Value,
    end_of_word_suffix: serde_json::Value,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: BTreeMap<String, u32>,
    merges: Vec<RawMerge>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawMerge {
    Text(String),
    Pair([String; 2]),
}

#[derive(Deserialize)]
#[serde(transparent)]
struct JsonBool(bool);

fn validate_format(raw: &RawTokenizer) -> ModelResult<()> {
    if raw.version != TOKENIZER_VERSION {
        return Err(invalid(format!(
            "unsupported tokenizer version {:?}",
            raw.version
        )));
    }
    for (name, value) in [
        ("truncation", &raw.truncation),
        ("padding", &raw.padding),
        ("normalizer", &raw.normalizer),
        ("post_processor", &raw.post_processor),
        ("model.dropout", &raw.model.dropout),
        ("model.unk_token", &raw.model.unk_token),
        (
            "model.continuing_subword_prefix",
            &raw.model.continuing_subword_prefix,
        ),
        ("model.end_of_word_suffix", &raw.model.end_of_word_suffix),
    ] {
        if !value.is_null() {
            return Err(invalid(format!("{name} must be null")));
        }
    }
    validate_pipeline(raw)?;
    if raw.model.kind != "BPE"
        || raw.model.fuse_unk
        || raw.model.byte_fallback
        || raw.model.ignore_merges
    {
        return Err(invalid(
            "only BPE with fuse_unk=false, byte_fallback=false, and ignore_merges=false is supported",
        ));
    }
    Ok(())
}

fn validate_pipeline(raw: &RawTokenizer) -> ModelResult<()> {
    if raw.pre_tokenizer.kind != "Sequence" || raw.pre_tokenizer.pretokenizers.len() != 2 {
        return Err(invalid("pre_tokenizer must be Sequence(Digits, ByteLevel)"));
    }
    match (
        &raw.pre_tokenizer.pretokenizers[0],
        &raw.pre_tokenizer.pretokenizers[1],
    ) {
        (
            RawPreTokenizer::Digits {
                individual_digits: true,
            },
            RawPreTokenizer::ByteLevel {
                add_prefix_space: false,
                trim_offsets: true,
                use_regex: true,
            },
        ) => {}
        _ => {
            return Err(invalid(
                "pre_tokenizer must use individual digits and ByteLevel(false,true,true)",
            ));
        }
    }
    if raw.decoder.kind != "ByteLevel" {
        return Err(invalid("decoder must be ByteLevel"));
    }
    Ok(())
}

fn validate_vocab(
    vocab: BTreeMap<String, u32>,
) -> ModelResult<(BTreeMap<String, u32>, Vec<String>)> {
    if vocab.is_empty() {
        return Err(invalid("vocabulary must not be empty"));
    }
    let mut by_id = vec![None; vocab.len()];
    for (token, &id) in &vocab {
        if token.is_empty() {
            return Err(invalid("vocabulary tokens must not be empty"));
        }
        let index = usize::try_from(id).map_err(|_| invalid("vocabulary ID does not fit usize"))?;
        let slot = by_id
            .get_mut(index)
            .ok_or_else(|| invalid(format!("vocabulary IDs must be dense; found {id}")))?;
        if slot.replace(token.clone()).is_some() {
            return Err(invalid(format!("duplicate vocabulary ID {id}")));
        }
    }
    let by_id = by_id
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.ok_or_else(|| invalid(format!("missing vocabulary ID {id}"))))
        .collect::<ModelResult<Vec<_>>>()?;
    Ok((vocab, by_id))
}

fn validate_byte_alphabet(
    vocab: &BTreeMap<String, u32>,
    special_ids: &BTreeSet<u32>,
    byte_decoder: &BTreeMap<char, u8>,
) -> ModelResult<()> {
    for (token, id) in vocab {
        if !special_ids.contains(id)
            && !token
                .chars()
                .all(|character| byte_decoder.contains_key(&character))
        {
            return Err(invalid(format!(
                "non-special token ID {id} contains a character outside the ByteLevel alphabet"
            )));
        }
    }
    Ok(())
}

fn validate_added_tokens(
    added: Vec<RawAddedToken>,
    vocab: &BTreeMap<String, u32>,
    vocabulary_size: usize,
) -> ModelResult<(BTreeSet<u32>, AddedTrie)> {
    let mut total_bytes = 0_usize;
    for token in &added {
        if token.content.is_empty() || token.content.len() > MAX_ADDED_TOKEN_BYTES {
            return Err(invalid(format!(
                "added token {} has invalid byte length",
                token.id
            )));
        }
        total_bytes = total_bytes
            .checked_add(token.content.len())
            .ok_or_else(|| ModelError::NumericOverflow {
                field: "tokenizer added-token bytes".to_owned(),
            })?;
    }
    enforce_count_limit(
        "tokenizer added-token bytes",
        usize_to_u64(MAX_TOTAL_ADDED_TOKEN_BYTES),
        total_bytes,
    )?;

    let mut ids = BTreeSet::new();
    let mut contents = BTreeSet::new();
    let mut trie = AddedTrie::default();
    for token in added {
        if token.single_word.0
            || token.lstrip.0
            || token.rstrip.0
            || token.normalized.0
            || !token.special.0
        {
            return Err(invalid(format!(
                "added token {} uses unsupported matching semantics",
                token.id
            )));
        }
        if usize::try_from(token.id)
            .ok()
            .is_none_or(|id| id >= vocabulary_size)
            || vocab.get(&token.content) != Some(&token.id)
        {
            return Err(invalid(format!(
                "added token {} must match the base vocabulary",
                token.id
            )));
        }
        if !ids.insert(token.id) || !contents.insert(token.content.clone()) {
            return Err(invalid("duplicate added-token ID or content"));
        }
        trie.insert(token.content.as_bytes(), token.id)?;
    }
    Ok((ids, trie))
}

fn validate_merges(
    merges: Vec<RawMerge>,
    vocab: &BTreeMap<String, u32>,
) -> ModelResult<BTreeMap<String, BTreeMap<String, usize>>> {
    let mut ranks = BTreeMap::<String, BTreeMap<String, usize>>::new();
    for (rank, merge) in merges.into_iter().enumerate() {
        let (left, right) = match merge {
            RawMerge::Pair([left, right]) => (left, right),
            RawMerge::Text(text) => parse_merge_text(&text)?,
        };
        if left.is_empty() || right.is_empty() {
            return Err(invalid(format!(
                "merge rank {rank} contains an empty symbol"
            )));
        }
        let combined = format!("{left}{right}");
        if !vocab.contains_key(&left)
            || !vocab.contains_key(&right)
            || !vocab.contains_key(&combined)
        {
            return Err(invalid(format!(
                "merge rank {rank} references a token absent from vocabulary"
            )));
        }
        if ranks.entry(left).or_default().insert(right, rank).is_some() {
            return Err(invalid(format!("duplicate merge pair at rank {rank}")));
        }
    }
    Ok(ranks)
}

fn parse_merge_text(text: &str) -> ModelResult<(String, String)> {
    let mut symbols = text.split(' ');
    let left = symbols.next().unwrap_or_default();
    let right = symbols.next().unwrap_or_default();
    if left.is_empty() || right.is_empty() || symbols.next().is_some() {
        return Err(invalid(format!("invalid textual BPE merge {text:?}")));
    }
    Ok((left.to_owned(), right.to_owned()))
}

#[derive(Default)]
struct AddedTrie {
    nodes: Vec<AddedTrieNode>,
}

#[derive(Default)]
struct AddedTrieNode {
    children: BTreeMap<u8, usize>,
    terminal: Option<u32>,
}

impl AddedTrie {
    fn insert(&mut self, bytes: &[u8], id: u32) -> ModelResult<()> {
        if self.nodes.is_empty() {
            self.nodes.push(AddedTrieNode::default());
        }
        let mut node = 0_usize;
        for &byte in bytes {
            let child = if let Some(&child) = self.nodes[node].children.get(&byte) {
                child
            } else {
                let child = self.nodes.len();
                self.nodes.push(AddedTrieNode::default());
                self.nodes[node].children.insert(byte, child);
                child
            };
            node = child;
        }
        if self.nodes[node].terminal.replace(id).is_some() {
            return Err(invalid("duplicate added token trie entry"));
        }
        Ok(())
    }

    fn next_match(&self, input: &str, cursor: usize) -> Option<(usize, usize, u32)> {
        if self.nodes.is_empty() {
            return None;
        }
        for (relative, _) in input[cursor..].char_indices() {
            let start = cursor + relative;
            if let Some((end, id)) = self.longest_at(input.as_bytes(), start) {
                return Some((start, end, id));
            }
        }
        None
    }

    fn longest_at(&self, input: &[u8], start: usize) -> Option<(usize, u32)> {
        let mut node = 0_usize;
        let mut best = None;
        for (relative, &byte) in input[start..].iter().enumerate() {
            let Some(&child) = self.nodes[node].children.get(&byte) else {
                break;
            };
            node = child;
            if let Some(id) = self.nodes[node].terminal {
                best = Some((start + relative + 1, id));
            }
        }
        best
    }
}

struct BpeNode {
    text: String,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

impl BpeNode {
    fn new(text: String) -> Self {
        Self {
            text,
            previous: None,
            next: None,
            alive: true,
        }
    }
}

type CandidateHeap = BinaryHeap<Reverse<(usize, usize, usize, usize)>>;

fn pair_rank(
    nodes: &[BpeNode],
    ranks: &BTreeMap<String, BTreeMap<String, usize>>,
    left: usize,
    right: usize,
) -> Option<usize> {
    ranks
        .get(&nodes[left].text)
        .and_then(|right_ranks| right_ranks.get(&nodes[right].text))
        .copied()
}

fn push_candidate(
    nodes: &[BpeNode],
    ranks: &BTreeMap<String, BTreeMap<String, usize>>,
    left: usize,
    right: usize,
    heap: &mut CandidateHeap,
) {
    if let Some(rank) = pair_rank(nodes, ranks, left, right) {
        heap.push(Reverse((rank, left, left, right)));
    }
}

fn candidate_is_current(
    nodes: &[BpeNode],
    ranks: &BTreeMap<String, BTreeMap<String, usize>>,
    rank: usize,
    left: usize,
    right: usize,
) -> bool {
    nodes
        .get(left)
        .is_some_and(|node| node.alive && node.next == Some(right))
        && nodes.get(right).is_some_and(|node| node.alive)
        && pair_rank(nodes, ranks, left, right) == Some(rank)
}

fn merge_nodes(
    nodes: &mut [BpeNode],
    ranks: &BTreeMap<String, BTreeMap<String, usize>>,
    left: usize,
    right: usize,
    _position: usize,
    heap: &mut CandidateHeap,
) {
    let next = nodes[right].next;
    let right_text = std::mem::take(&mut nodes[right].text);
    nodes[right].alive = false;
    nodes[right].previous = None;
    nodes[right].next = None;
    nodes[left].text.push_str(&right_text);
    nodes[left].next = next;
    if let Some(next) = next {
        nodes[next].previous = Some(left);
    }
    if let Some(previous) = nodes[left].previous {
        push_candidate(nodes, ranks, previous, left, heap);
    }
    if let Some(next) = nodes[left].next {
        push_candidate(nodes, ranks, left, next, heap);
    }
}

fn split_individual_decimal_digits(input: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0_usize;
    for (index, character) in input.char_indices() {
        if !character.is_numeric() {
            continue;
        }
        if start < index {
            segments.push(&input[start..index]);
        }
        let end = index + character.len_utf8();
        segments.push(&input[index..end]);
        start = end;
    }
    if start < input.len() {
        segments.push(&input[start..]);
    }
    segments
}

fn gpt2_pieces(input: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut cursor = 0_usize;
    while cursor < input.len() {
        let end = next_gpt2_piece_end(input, cursor);
        pieces.push(&input[cursor..end]);
        cursor = end;
    }
    pieces
}

fn next_gpt2_piece_end(input: &str, start: usize) -> usize {
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if input[start..].starts_with(contraction) {
            return start + contraction.len();
        }
    }
    let Some(first) = input[start..].chars().next() else {
        return input.len();
    };
    if first == ' ' {
        let after_space = start + 1;
        if after_space < input.len() {
            if let Some(next) = input[after_space..].chars().next() {
                if !next.is_whitespace() {
                    return consume_category_run(input, after_space, category(next));
                }
            }
        }
    }
    if first.is_whitespace() {
        return consume_whitespace(input, start);
    }
    consume_category_run(input, start, category(first))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CharacterCategory {
    Letter,
    Number,
    Other,
}

fn category(character: char) -> CharacterCategory {
    if character.is_numeric() {
        CharacterCategory::Number
    } else if is_general_category_letter(character) {
        CharacterCategory::Letter
    } else {
        CharacterCategory::Other
    }
}

fn consume_category_run(input: &str, start: usize, wanted: CharacterCategory) -> usize {
    let mut end = start;
    for (relative, character) in input[start..].char_indices() {
        if character.is_whitespace() || category(character) != wanted {
            break;
        }
        end = start + relative + character.len_utf8();
    }
    end
}

fn consume_whitespace(input: &str, start: usize) -> usize {
    let mut boundaries = Vec::new();
    for (relative, character) in input[start..].char_indices() {
        if !character.is_whitespace() {
            break;
        }
        boundaries.push(start + relative + character.len_utf8());
    }
    let run_end = boundaries.last().copied().unwrap_or(start);
    if run_end < input.len() && boundaries.len() > 1 {
        boundaries[boundaries.len() - 2]
    } else {
        run_end
    }
}

fn byte_maps() -> ([char; 256], BTreeMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = BTreeMap::new();
    let mut extra = 0_u32;
    for byte in u8::MIN..=u8::MAX {
        let codepoint = if (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte)
        {
            u32::from(byte)
        } else {
            let codepoint = 256 + extra;
            extra += 1;
            codepoint
        };
        let character = char::from_u32(codepoint).unwrap_or('\0');
        encoder[usize::from(byte)] = character;
        decoder.insert(character, byte);
    }
    (encoder, decoder)
}

fn is_general_category_letter(character: char) -> bool {
    character.is_alphabetic() && !character.is_numeric() && !in_ranges(character, MARK_RANGES)
}

fn in_ranges(character: char, ranges: &[(char, char)]) -> bool {
    ranges
        .binary_search_by(|(start, end)| {
            if character < *start {
                std::cmp::Ordering::Greater
            } else if character > *end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

// Generated by ucd-generate 0.3.1 from Unicode 16.0.0
// General_Category=Mark data, matching regex-syntax 0.8.5's generated table.
// Kept inline so production tokenization has no regex or Unicode-table crate
// dependency.
//
// UNICODE, INC. LICENSE AGREEMENT - DATA FILES AND SOFTWARE
// Copyright © 1991-2018 Unicode, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of the Unicode data files and any associated documentation (the "Data
// Files"), or Unicode software and any associated documentation (the
// "Software"), to deal in the Data Files or Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, and/or sell copies, and to permit persons to whom the
// Data Files or Software are furnished to do so, provided that either (a)
// this copyright and permission notice appear with all copies of the Data
// Files or Software, or (b) this copyright and permission notice appear in
// associated Documentation.
// THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
// KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
// THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS
// INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
// CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
// DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
// TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
// OF THE DATA FILES OR SOFTWARE. Except as contained in this notice, the name
// of a copyright holder shall not be used in advertising or otherwise to
// promote the sale, use or other dealings in these Data Files or Software
// without prior written authorization of the copyright holder.
const MARK_RANGES: &[(char, char)] = &[
    ('\u{300}', '\u{36f}'),
    ('\u{483}', '\u{489}'),
    ('\u{591}', '\u{5bd}'),
    ('\u{5bf}', '\u{5bf}'),
    ('\u{5c1}', '\u{5c2}'),
    ('\u{5c4}', '\u{5c5}'),
    ('\u{5c7}', '\u{5c7}'),
    ('\u{610}', '\u{61a}'),
    ('\u{64b}', '\u{65f}'),
    ('\u{670}', '\u{670}'),
    ('\u{6d6}', '\u{6dc}'),
    ('\u{6df}', '\u{6e4}'),
    ('\u{6e7}', '\u{6e8}'),
    ('\u{6ea}', '\u{6ed}'),
    ('\u{711}', '\u{711}'),
    ('\u{730}', '\u{74a}'),
    ('\u{7a6}', '\u{7b0}'),
    ('\u{7eb}', '\u{7f3}'),
    ('\u{7fd}', '\u{7fd}'),
    ('\u{816}', '\u{819}'),
    ('\u{81b}', '\u{823}'),
    ('\u{825}', '\u{827}'),
    ('\u{829}', '\u{82d}'),
    ('\u{859}', '\u{85b}'),
    ('\u{897}', '\u{89f}'),
    ('\u{8ca}', '\u{8e1}'),
    ('\u{8e3}', 'ः'),
    ('\u{93a}', '\u{93c}'),
    ('ा', 'ॏ'),
    ('\u{951}', '\u{957}'),
    ('\u{962}', '\u{963}'),
    ('\u{981}', 'ঃ'),
    ('\u{9bc}', '\u{9bc}'),
    ('\u{9be}', '\u{9c4}'),
    ('ে', 'ৈ'),
    ('ো', '\u{9cd}'),
    ('\u{9d7}', '\u{9d7}'),
    ('\u{9e2}', '\u{9e3}'),
    ('\u{9fe}', '\u{9fe}'),
    ('\u{a01}', 'ਃ'),
    ('\u{a3c}', '\u{a3c}'),
    ('ਾ', '\u{a42}'),
    ('\u{a47}', '\u{a48}'),
    ('\u{a4b}', '\u{a4d}'),
    ('\u{a51}', '\u{a51}'),
    ('\u{a70}', '\u{a71}'),
    ('\u{a75}', '\u{a75}'),
    ('\u{a81}', 'ઃ'),
    ('\u{abc}', '\u{abc}'),
    ('ા', '\u{ac5}'),
    ('\u{ac7}', 'ૉ'),
    ('ો', '\u{acd}'),
    ('\u{ae2}', '\u{ae3}'),
    ('\u{afa}', '\u{aff}'),
    ('\u{b01}', 'ଃ'),
    ('\u{b3c}', '\u{b3c}'),
    ('\u{b3e}', '\u{b44}'),
    ('େ', 'ୈ'),
    ('ୋ', '\u{b4d}'),
    ('\u{b55}', '\u{b57}'),
    ('\u{b62}', '\u{b63}'),
    ('\u{b82}', '\u{b82}'),
    ('\u{bbe}', 'ூ'),
    ('ெ', 'ை'),
    ('ொ', '\u{bcd}'),
    ('\u{bd7}', '\u{bd7}'),
    ('\u{c00}', '\u{c04}'),
    ('\u{c3c}', '\u{c3c}'),
    ('\u{c3e}', 'ౄ'),
    ('\u{c46}', '\u{c48}'),
    ('\u{c4a}', '\u{c4d}'),
    ('\u{c55}', '\u{c56}'),
    ('\u{c62}', '\u{c63}'),
    ('\u{c81}', 'ಃ'),
    ('\u{cbc}', '\u{cbc}'),
    ('ಾ', 'ೄ'),
    ('\u{cc6}', '\u{cc8}'),
    ('\u{cca}', '\u{ccd}'),
    ('\u{cd5}', '\u{cd6}'),
    ('\u{ce2}', '\u{ce3}'),
    ('ೳ', 'ೳ'),
    ('\u{d00}', 'ഃ'),
    ('\u{d3b}', '\u{d3c}'),
    ('\u{d3e}', '\u{d44}'),
    ('െ', 'ൈ'),
    ('ൊ', '\u{d4d}'),
    ('\u{d57}', '\u{d57}'),
    ('\u{d62}', '\u{d63}'),
    ('\u{d81}', 'ඃ'),
    ('\u{dca}', '\u{dca}'),
    ('\u{dcf}', '\u{dd4}'),
    ('\u{dd6}', '\u{dd6}'),
    ('ෘ', '\u{ddf}'),
    ('ෲ', 'ෳ'),
    ('\u{e31}', '\u{e31}'),
    ('\u{e34}', '\u{e3a}'),
    ('\u{e47}', '\u{e4e}'),
    ('\u{eb1}', '\u{eb1}'),
    ('\u{eb4}', '\u{ebc}'),
    ('\u{ec8}', '\u{ece}'),
    ('\u{f18}', '\u{f19}'),
    ('\u{f35}', '\u{f35}'),
    ('\u{f37}', '\u{f37}'),
    ('\u{f39}', '\u{f39}'),
    ('༾', '༿'),
    ('\u{f71}', '\u{f84}'),
    ('\u{f86}', '\u{f87}'),
    ('\u{f8d}', '\u{f97}'),
    ('\u{f99}', '\u{fbc}'),
    ('\u{fc6}', '\u{fc6}'),
    ('ါ', '\u{103e}'),
    ('ၖ', '\u{1059}'),
    ('\u{105e}', '\u{1060}'),
    ('ၢ', 'ၤ'),
    ('ၧ', 'ၭ'),
    ('\u{1071}', '\u{1074}'),
    ('\u{1082}', '\u{108d}'),
    ('ႏ', 'ႏ'),
    ('ႚ', '\u{109d}'),
    ('\u{135d}', '\u{135f}'),
    ('\u{1712}', '\u{1715}'),
    ('\u{1732}', '\u{1734}'),
    ('\u{1752}', '\u{1753}'),
    ('\u{1772}', '\u{1773}'),
    ('\u{17b4}', '\u{17d3}'),
    ('\u{17dd}', '\u{17dd}'),
    ('\u{180b}', '\u{180d}'),
    ('\u{180f}', '\u{180f}'),
    ('\u{1885}', '\u{1886}'),
    ('\u{18a9}', '\u{18a9}'),
    ('\u{1920}', 'ᤫ'),
    ('ᤰ', '\u{193b}'),
    ('\u{1a17}', '\u{1a1b}'),
    ('ᩕ', '\u{1a5e}'),
    ('\u{1a60}', '\u{1a7c}'),
    ('\u{1a7f}', '\u{1a7f}'),
    ('\u{1ab0}', '\u{1ace}'),
    ('\u{1b00}', 'ᬄ'),
    ('\u{1b34}', '\u{1b44}'),
    ('\u{1b6b}', '\u{1b73}'),
    ('\u{1b80}', 'ᮂ'),
    ('ᮡ', '\u{1bad}'),
    ('\u{1be6}', '\u{1bf3}'),
    ('ᰤ', '\u{1c37}'),
    ('\u{1cd0}', '\u{1cd2}'),
    ('\u{1cd4}', '\u{1ce8}'),
    ('\u{1ced}', '\u{1ced}'),
    ('\u{1cf4}', '\u{1cf4}'),
    ('᳷', '\u{1cf9}'),
    ('\u{1dc0}', '\u{1dff}'),
    ('\u{20d0}', '\u{20f0}'),
    ('\u{2cef}', '\u{2cf1}'),
    ('\u{2d7f}', '\u{2d7f}'),
    ('\u{2de0}', '\u{2dff}'),
    ('\u{302a}', '\u{302f}'),
    ('\u{3099}', '\u{309a}'),
    ('\u{a66f}', '\u{a672}'),
    ('\u{a674}', '\u{a67d}'),
    ('\u{a69e}', '\u{a69f}'),
    ('\u{a6f0}', '\u{a6f1}'),
    ('\u{a802}', '\u{a802}'),
    ('\u{a806}', '\u{a806}'),
    ('\u{a80b}', '\u{a80b}'),
    ('ꠣ', 'ꠧ'),
    ('\u{a82c}', '\u{a82c}'),
    ('ꢀ', 'ꢁ'),
    ('ꢴ', '\u{a8c5}'),
    ('\u{a8e0}', '\u{a8f1}'),
    ('\u{a8ff}', '\u{a8ff}'),
    ('\u{a926}', '\u{a92d}'),
    ('\u{a947}', '\u{a953}'),
    ('\u{a980}', 'ꦃ'),
    ('\u{a9b3}', '\u{a9c0}'),
    ('\u{a9e5}', '\u{a9e5}'),
    ('\u{aa29}', '\u{aa36}'),
    ('\u{aa43}', '\u{aa43}'),
    ('\u{aa4c}', 'ꩍ'),
    ('ꩻ', 'ꩽ'),
    ('\u{aab0}', '\u{aab0}'),
    ('\u{aab2}', '\u{aab4}'),
    ('\u{aab7}', '\u{aab8}'),
    ('\u{aabe}', '\u{aabf}'),
    ('\u{aac1}', '\u{aac1}'),
    ('ꫫ', 'ꫯ'),
    ('ꫵ', '\u{aaf6}'),
    ('ꯣ', 'ꯪ'),
    ('꯬', '\u{abed}'),
    ('\u{fb1e}', '\u{fb1e}'),
    ('\u{fe00}', '\u{fe0f}'),
    ('\u{fe20}', '\u{fe2f}'),
    ('\u{101fd}', '\u{101fd}'),
    ('\u{102e0}', '\u{102e0}'),
    ('\u{10376}', '\u{1037a}'),
    ('\u{10a01}', '\u{10a03}'),
    ('\u{10a05}', '\u{10a06}'),
    ('\u{10a0c}', '\u{10a0f}'),
    ('\u{10a38}', '\u{10a3a}'),
    ('\u{10a3f}', '\u{10a3f}'),
    ('\u{10ae5}', '\u{10ae6}'),
    ('\u{10d24}', '\u{10d27}'),
    ('\u{10d69}', '\u{10d6d}'),
    ('\u{10eab}', '\u{10eac}'),
    ('\u{10efc}', '\u{10eff}'),
    ('\u{10f46}', '\u{10f50}'),
    ('\u{10f82}', '\u{10f85}'),
    ('𑀀', '𑀂'),
    ('\u{11038}', '\u{11046}'),
    ('\u{11070}', '\u{11070}'),
    ('\u{11073}', '\u{11074}'),
    ('\u{1107f}', '𑂂'),
    ('𑂰', '\u{110ba}'),
    ('\u{110c2}', '\u{110c2}'),
    ('\u{11100}', '\u{11102}'),
    ('\u{11127}', '\u{11134}'),
    ('𑅅', '𑅆'),
    ('\u{11173}', '\u{11173}'),
    ('\u{11180}', '𑆂'),
    ('𑆳', '\u{111c0}'),
    ('\u{111c9}', '\u{111cc}'),
    ('𑇎', '\u{111cf}'),
    ('𑈬', '\u{11237}'),
    ('\u{1123e}', '\u{1123e}'),
    ('\u{11241}', '\u{11241}'),
    ('\u{112df}', '\u{112ea}'),
    ('\u{11300}', '𑌃'),
    ('\u{1133b}', '\u{1133c}'),
    ('\u{1133e}', '𑍄'),
    ('𑍇', '𑍈'),
    ('𑍋', '\u{1134d}'),
    ('\u{11357}', '\u{11357}'),
    ('𑍢', '𑍣'),
    ('\u{11366}', '\u{1136c}'),
    ('\u{11370}', '\u{11374}'),
    ('\u{113b8}', '\u{113c0}'),
    ('\u{113c2}', '\u{113c2}'),
    ('\u{113c5}', '\u{113c5}'),
    ('\u{113c7}', '𑏊'),
    ('𑏌', '\u{113d0}'),
    ('\u{113d2}', '\u{113d2}'),
    ('\u{113e1}', '\u{113e2}'),
    ('𑐵', '\u{11446}'),
    ('\u{1145e}', '\u{1145e}'),
    ('\u{114b0}', '\u{114c3}'),
    ('\u{115af}', '\u{115b5}'),
    ('𑖸', '\u{115c0}'),
    ('\u{115dc}', '\u{115dd}'),
    ('𑘰', '\u{11640}'),
    ('\u{116ab}', '\u{116b7}'),
    ('\u{1171d}', '\u{1172b}'),
    ('𑠬', '\u{1183a}'),
    ('\u{11930}', '𑤵'),
    ('𑤷', '𑤸'),
    ('\u{1193b}', '\u{1193e}'),
    ('𑥀', '𑥀'),
    ('𑥂', '\u{11943}'),
    ('𑧑', '\u{119d7}'),
    ('\u{119da}', '\u{119e0}'),
    ('𑧤', '𑧤'),
    ('\u{11a01}', '\u{11a0a}'),
    ('\u{11a33}', '𑨹'),
    ('\u{11a3b}', '\u{11a3e}'),
    ('\u{11a47}', '\u{11a47}'),
    ('\u{11a51}', '\u{11a5b}'),
    ('\u{11a8a}', '\u{11a99}'),
    ('𑰯', '\u{11c36}'),
    ('\u{11c38}', '\u{11c3f}'),
    ('\u{11c92}', '\u{11ca7}'),
    ('𑲩', '\u{11cb6}'),
    ('\u{11d31}', '\u{11d36}'),
    ('\u{11d3a}', '\u{11d3a}'),
    ('\u{11d3c}', '\u{11d3d}'),
    ('\u{11d3f}', '\u{11d45}'),
    ('\u{11d47}', '\u{11d47}'),
    ('𑶊', '𑶎'),
    ('\u{11d90}', '\u{11d91}'),
    ('𑶓', '\u{11d97}'),
    ('\u{11ef3}', '𑻶'),
    ('\u{11f00}', '\u{11f01}'),
    ('𑼃', '𑼃'),
    ('𑼴', '\u{11f3a}'),
    ('𑼾', '\u{11f42}'),
    ('\u{11f5a}', '\u{11f5a}'),
    ('\u{13440}', '\u{13440}'),
    ('\u{13447}', '\u{13455}'),
    ('\u{1611e}', '\u{1612f}'),
    ('\u{16af0}', '\u{16af4}'),
    ('\u{16b30}', '\u{16b36}'),
    ('\u{16f4f}', '\u{16f4f}'),
    ('𖽑', '𖾇'),
    ('\u{16f8f}', '\u{16f92}'),
    ('\u{16fe4}', '\u{16fe4}'),
    ('\u{16ff0}', '\u{16ff1}'),
    ('\u{1bc9d}', '\u{1bc9e}'),
    ('\u{1cf00}', '\u{1cf2d}'),
    ('\u{1cf30}', '\u{1cf46}'),
    ('\u{1d165}', '\u{1d169}'),
    ('\u{1d16d}', '\u{1d172}'),
    ('\u{1d17b}', '\u{1d182}'),
    ('\u{1d185}', '\u{1d18b}'),
    ('\u{1d1aa}', '\u{1d1ad}'),
    ('\u{1d242}', '\u{1d244}'),
    ('\u{1da00}', '\u{1da36}'),
    ('\u{1da3b}', '\u{1da6c}'),
    ('\u{1da75}', '\u{1da75}'),
    ('\u{1da84}', '\u{1da84}'),
    ('\u{1da9b}', '\u{1da9f}'),
    ('\u{1daa1}', '\u{1daaf}'),
    ('\u{1e000}', '\u{1e006}'),
    ('\u{1e008}', '\u{1e018}'),
    ('\u{1e01b}', '\u{1e021}'),
    ('\u{1e023}', '\u{1e024}'),
    ('\u{1e026}', '\u{1e02a}'),
    ('\u{1e08f}', '\u{1e08f}'),
    ('\u{1e130}', '\u{1e136}'),
    ('\u{1e2ae}', '\u{1e2ae}'),
    ('\u{1e2ec}', '\u{1e2ef}'),
    ('\u{1e4ec}', '\u{1e4ef}'),
    ('\u{1e5ee}', '\u{1e5ef}'),
    ('\u{1e8d0}', '\u{1e8d6}'),
    ('\u{1e944}', '\u{1e94a}'),
    ('\u{e0100}', '\u{e01ef}'),
];

fn enforce_count_limit(resource: &'static str, limit: u64, actual: usize) -> ModelResult<()> {
    let actual = usize_to_u64(actual);
    if actual > limit {
        return Err(ModelError::LimitExceeded {
            resource,
            limit,
            actual: Some(actual),
        });
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn invalid(reason: impl Into<String>) -> ModelError {
    ModelError::InvalidTokenizer {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
      "version":"1.0",
      "truncation":null,
      "padding":null,
      "added_tokens":[{"id":0,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
      "normalizer":null,
      "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]},
      "post_processor":null,
      "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
      "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        "vocab":{"<|end|>":0,"h":1,"e":2,"l":3,"o":4,"Ġ":5,"w":6,"r":7,"d":8,"1":9,"2":10,"!":11,"'":12,"s":13,"'s":14,"he":15,"hel":16,"hell":17,"hello":18,"Ġw":19,"Ġwo":20,"Ġwor":21,"Ġworl":22,"Ġworld":23,"Ã":24,"©":25,"Ã©":26,"Ċ":27},
        "merges":["h e","he l","hel l","hell o","Ġ w","Ġw o","Ġwo r","Ġwor l","Ġworl d","Ã ©","' s"]}
    }"#;

    fn tokenizer() -> SmolLm2Tokenizer {
        SmolLm2Tokenizer::from_json_slice(FIXTURE.as_bytes()).expect("synthetic tokenizer")
    }

    #[test]
    fn synthetic_bytelevel_bpe_round_trip_and_options() {
        let tokenizer = tokenizer();
        let input = "hello world12!\né<|end|>hello";
        let ids = tokenizer
            .encode(input, EncodeOptions::default())
            .expect("encode");
        assert_eq!(ids, [18, 23, 9, 10, 11, 27, 26, 0, 18]);
        assert_eq!(
            tokenizer.decode(&ids, DecodeOptions::default()).unwrap(),
            input
        );
        assert_eq!(
            tokenizer
                .decode(
                    &ids,
                    DecodeOptions {
                        skip_special_tokens: true,
                    },
                )
                .unwrap(),
            "hello world12!\néhello"
        );
        let without_auto_special = tokenizer
            .encode(
                input,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
            .unwrap();
        assert_eq!(without_auto_special, ids);
    }

    #[test]
    fn decoder_matches_bytelevel_all_or_raw_token_fallback() {
        let byte_alphabet_special = FIXTURE.replace("<|end|>", "é");
        let tokenizer = SmolLm2Tokenizer::from_json_slice(byte_alphabet_special.as_bytes())
            .expect("non-ASCII ByteLevel special token");
        assert_eq!(
            tokenizer
                .decode(&[0], DecodeOptions::default())
                .expect("ByteLevel decode"),
            "�"
        );

        let raw_special = FIXTURE.replace("<|end|>", "π");
        let tokenizer = SmolLm2Tokenizer::from_json_slice(raw_special.as_bytes())
            .expect("special token outside the ByteLevel alphabet");
        assert_eq!(
            tokenizer
                .decode(&[0], DecodeOptions::default())
                .expect("raw-token fallback"),
            "π"
        );
    }

    #[test]
    fn accessors_expose_vocab_and_special_identity() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.vocabulary_size(), 28);
        assert!(tokenizer.contains_id(27));
        assert!(!tokenizer.contains_id(28));
        assert_eq!(tokenizer.token_for_id(18), Some("hello"));
        assert!(tokenizer.is_special_id(0));
        assert!(!tokenizer.is_special_id(18));
    }

    #[test]
    fn scanner_matches_gpt2_contractions_whitespace_and_unicode_categories() {
        assert_eq!(
            gpt2_pieces("can't  stop\tnow\n\n"),
            ["can", "'t", " ", " stop", "\t", "now", "\n\n"]
        );
        assert_eq!(gpt2_pieces("A\u{0301}Ⅻ¼三"), ["A", "\u{0301}", "Ⅻ¼", "三"]);
        assert!('\u{0345}'.is_alphabetic());
        assert_eq!(gpt2_pieces("A\u{0345}Ⅻ"), ["A", "\u{0345}", "Ⅻ"]);
        assert_eq!(
            split_individual_decimal_digits("x٣2¼y"),
            ["x", "٣", "2", "¼", "y"]
        );
    }

    #[test]
    fn added_trie_chooses_earliest_then_longest_exact_match() {
        let mut trie = AddedTrie::default();
        trie.insert(b"<|x|>", 1).unwrap();
        trie.insert(b"<|x|>long", 2).unwrap();
        assert_eq!(trie.next_match("a<|x|>long b", 0), Some((1, 10, 2)));
    }

    #[test]
    fn added_token_trie_has_a_cumulative_node_budget() {
        let tokens = (0_u32..=256)
            .map(|id| RawAddedToken {
                id,
                content: format!("{id:04}{}", "x".repeat(MAX_ADDED_TOKEN_BYTES - 4)),
                single_word: JsonBool(false),
                lstrip: JsonBool(false),
                rstrip: JsonBool(false),
                normalized: JsonBool(false),
                special: JsonBool(true),
            })
            .collect();
        let Err(error) = validate_added_tokens(tokens, &BTreeMap::new(), 257) else {
            panic!("oversized added-token trie must fail before allocation");
        };
        assert!(matches!(
            error,
            ModelError::LimitExceeded {
                resource: "tokenizer added-token bytes",
                limit: 1_048_576,
                actual: Some(1_052_672),
            }
        ));
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (encoder, decoder) = byte_maps();
        assert_eq!(decoder.len(), 256);
        for byte in u8::MIN..=u8::MAX {
            assert_eq!(decoder.get(&encoder[usize::from(byte)]), Some(&byte));
        }
    }

    #[test]
    fn malformed_and_unsupported_json_fails_deterministically() {
        let duplicate = FIXTURE.replacen(
            r#""version":"1.0""#,
            r#""version":"1.0","version":"1.0""#,
            1,
        );
        assert!(matches!(
            SmolLm2Tokenizer::from_json_slice(duplicate.as_bytes()),
            Err(ModelError::InvalidJson { .. })
        ));

        for malformed in [
            FIXTURE.replace(r#""use_regex":true}]"#, r#""use_regex":false}]"#),
            FIXTURE.replace(r#""ignore_merges":false"#, r#""ignore_merges":true"#),
            FIXTURE.replace(r#""special":true"#, r#""special":false"#),
            FIXTURE.replace(r#""h":1"#, r#""h":2"#),
            FIXTURE.replace(r#""merges":["h e""#, r#""merges":["missing e""#),
        ] {
            assert!(matches!(
                SmolLm2Tokenizer::from_json_slice(malformed.as_bytes()),
                Err(ModelError::InvalidTokenizer { .. })
            ));
        }

        let unknown = FIXTURE.replace(r#""version":"1.0""#, r#""version":"1.0","unknown":0"#);
        assert!(matches!(
            SmolLm2Tokenizer::from_json_slice(unknown.as_bytes()),
            Err(ModelError::InvalidJson { .. })
        ));

        for length in 0..FIXTURE.len() {
            let outcome = std::panic::catch_unwind(|| {
                SmolLm2Tokenizer::from_json_slice(&FIXTURE.as_bytes()[..length])
            });
            assert!(outcome.is_ok(), "parser panicked at truncation {length}");
            assert!(outcome.unwrap().is_err(), "truncation {length} parsed");
        }
    }

    #[test]
    fn accepts_structured_merge_pairs() {
        let structured = FIXTURE.replacen(r#""merges":["h e""#, r#""merges":[["h","e"]"#, 1);
        let tokenizer = SmolLm2Tokenizer::from_json_slice(structured.as_bytes()).unwrap();
        assert_eq!(
            tokenizer.encode("hello", EncodeOptions::default()).unwrap(),
            [18]
        );
    }

    #[test]
    fn invalid_ids_and_unrepresentable_input_fail() {
        let tokenizer = tokenizer();
        assert!(matches!(
            tokenizer.decode(&[999], DecodeOptions::default()),
            Err(ModelError::InvalidTokenId { id: 999 })
        ));
        assert!(matches!(
            tokenizer.encode("?", EncodeOptions::default()),
            Err(ModelError::InvalidTokenizer { .. })
        ));
        assert_eq!(
            tokenizer
                .decode(&[24], DecodeOptions::default())
                .expect("ByteLevel uses lossy UTF-8 decoding"),
            "�"
        );
    }
}
