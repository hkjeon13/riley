//! Native Metaspace + byte-fallback BPE for the Mistral v0.3 artifact profile.
//! Uses the same ranked merge engine and bounded artifact validators as ByteLevel.
use super::*;
use serde_json::{Value, json};

/// First-party tokenizer for Mistral's full-attention v0.3 checkpoint profile.
pub struct MistralTokenizer {
    vocab: BTreeMap<String, u32>,
    vocab_by_id: Box<[String]>,
    merge_ranks: BTreeMap<String, BTreeMap<String, usize>>,
    special_ids: BTreeSet<u32>,
    added_trie: AddedTrie,
    fallback_ids: [u32; 256],
    decoded: Box<[Box<[u8]>]>,
    fallback_bytes: Box<[Option<u8>]>,
    maximum_decoded_bytes: usize,
    bos: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMistralTokenizer {
    version: String,
    truncation: Value,
    padding: Value,
    added_tokens: Vec<RawAddedToken>,
    normalizer: Value,
    pre_tokenizer: Value,
    post_processor: Value,
    decoder: Value,
    model: RawBpe,
}

impl MistralTokenizer {
    /// Parses the closed Mistral v0.3 tokenizer profile with production limits.
    ///
    /// # Errors
    /// Rejects unsupported pipelines, invalid vocabulary/merges, or oversized artifacts.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::production())
    }

    /// Parses and validates a native Mistral tokenizer with explicit load limits.
    ///
    /// # Errors
    /// Rejects unsupported pipelines, invalid vocabulary/merges, or oversized artifacts.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        enforce_count_limit("tokenizer JSON", limits.tokenizer_bytes(), input.len())?;
        let raw: RawMistralTokenizer = strict_json::from_slice(input, ArtifactKind::Tokenizer)?;
        let model = &raw.model;
        if raw.version != TOKENIZER_VERSION
            || !raw.truncation.is_null()
            || !raw.padding.is_null()
            || !raw.normalizer.is_null()
            || model.kind != "BPE"
            || !model.dropout.is_null()
            || model.unk_token.as_str() != Some("<unk>")
            || !model.continuing_subword_prefix.is_null()
            || !model.end_of_word_suffix.is_null()
            || !model.fuse_unk
            || !model.byte_fallback
            || !matches!(
                model.ignore_merges,
                RawOptionalJson::Missing | RawOptionalJson::Present(Value::Bool(false))
            )
        {
            return Err(invalid("unsupported Mistral byte-fallback BPE metadata"));
        }
        if raw.pre_tokenizer
            != json!({
                "type":"Metaspace", "replacement":"▁", "prepend_scheme":"first", "split":false
            })
            || raw.decoder
                != json!({
                    "type":"Sequence", "decoders":[
                        {"type":"Replace", "pattern":{"String":"▁"}, "content":" "},
                        {"type":"ByteFallback"}, {"type":"Fuse"},
                        {"type":"Strip", "content":" ", "start":1, "stop":0}
                    ]
                })
        {
            return Err(invalid("unsupported Mistral Metaspace/decoder pipeline"));
        }
        // A closed post-processor prevents additional or reordered special tokens.
        if raw.post_processor
            != json!({
                "type":"TemplateProcessing",
                "single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                "pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},
                    {"SpecialToken":{"id":"<s>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],
                "special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}
            })
        {
            return Err(invalid("unsupported Mistral BOS template"));
        }
        enforce_count_limit(
            "tokenizer vocabulary",
            usize_to_u64(limits.vocabulary_entries()),
            model.vocab.len(),
        )?;
        enforce_count_limit(
            "tokenizer merges",
            usize_to_u64(limits.merges()),
            model.merges.len(),
        )?;
        enforce_count_limit(
            "tokenizer added tokens",
            usize_to_u64(limits.added_tokens()),
            raw.added_tokens.len(),
        )?;
        let (vocab, mut vocab_by_id) = validate_vocab(raw.model.vocab)?;
        let added = validate_added_tokens(
            raw.added_tokens,
            &vocab,
            &mut vocab_by_id,
            TokenizerProfile::SmolLm2,
        )?;
        if vocab.get("<s>") != Some(&1)
            || !added.special_ids.contains(&1)
            || vocab.get("<unk>") != Some(&0)
            || !added.special_ids.contains(&0)
            || vocab.get("</s>") != Some(&2)
            || !added.special_ids.contains(&2)
        {
            return Err(invalid(
                "Mistral BOS/EOS/UNK must be declared special tokens 1/2/0",
            ));
        }
        let mut fallback_ids = [0; 256];
        for (byte, id) in fallback_ids.iter_mut().enumerate() {
            *id = *vocab
                .get(&format!("<0x{byte:02X}>"))
                .filter(|id| !added.special_ids.contains(id))
                .ok_or_else(|| {
                    invalid("Mistral requires all 256 non-special byte fallback tokens")
                })?;
        }
        let merge_ranks = validate_merges(raw.model.merges, &vocab)?;
        let mut fallback_bytes = vec![None; vocab_by_id.len()];
        for (byte, &id) in fallback_ids.iter().enumerate() {
            fallback_bytes[id as usize] = Some(u8::try_from(byte).expect("byte domain"));
        }
        let decoded: Vec<Box<[u8]>> = vocab_by_id
            .iter()
            .enumerate()
            .map(|(id, token)| {
                fallback_bytes[id].map_or_else(
                    || token.replace('▁', " ").into_bytes().into_boxed_slice(),
                    |byte| vec![byte].into_boxed_slice(),
                )
            })
            .collect();
        // Invalid byte fallback is U+FFFD in batch decoding, three UTF-8 bytes.
        let maximum_decoded_bytes = decoded
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0)
            .max(3);
        Ok(Self {
            vocab,
            vocab_by_id: vocab_by_id.into_boxed_slice(),
            merge_ranks,
            special_ids: added.special_ids,
            added_trie: added.trie,
            fallback_ids,
            decoded: decoded.into_boxed_slice(),
            fallback_bytes: fallback_bytes.into_boxed_slice(),
            maximum_decoded_bytes,
            bos: 1,
        })
    }

    /// Returns whether the ID was explicitly declared special in the artifact.
    #[must_use]
    pub fn is_special_id(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Returns the BOS inserted by the validated single-input template.
    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos
    }

    fn encode_plain(
        &self,
        input: &str,
        original_offset: usize,
        out: &mut Vec<u32>,
    ) -> ModelResult<()> {
        if input.is_empty() {
            return Ok(());
        }
        let mapped = input.replace(' ', "▁");
        let mut nodes = Vec::new();
        if original_offset == 0 && !mapped.starts_with('▁') {
            nodes.push(BpeNode::new("▁".to_owned()));
        }
        for character in mapped.chars() {
            let token = character.to_string();
            if self.vocab.contains_key(&token) {
                nodes.push(BpeNode::new(token));
            } else {
                let mut bytes = [0_u8; 4];
                for &byte in character.encode_utf8(&mut bytes).as_bytes() {
                    nodes.push(BpeNode::new(
                        self.vocab_by_id[self.fallback_ids[usize::from(byte)] as usize].clone(),
                    ));
                }
            }
        }
        merge_bpe_nodes(&mut nodes, &self.merge_ranks);
        for node in nodes.iter().filter(|node| node.alive) {
            out.push(
                *self
                    .vocab
                    .get(&node.text)
                    .ok_or_else(|| invalid("merged Mistral token is absent from vocabulary"))?,
            );
        }
        Ok(())
    }

    fn token_bytes(&self, id: u32, options: DecodeOptions) -> ModelResult<&[u8]> {
        let bytes = self
            .decoded
            .get(id as usize)
            .ok_or(ModelError::InvalidTokenId { id })?;
        Ok(if options.skip_special_tokens && self.is_special_id(id) {
            &[]
        } else {
            bytes
        })
    }

    /// Visits the exact batch decoder result without allocating request-sized storage.
    fn visit_decoded(
        &self,
        ids: &[u32],
        options: DecodeOptions,
        mut visit: impl FnMut(&[u8]) -> ModelResult<()>,
    ) -> ModelResult<()> {
        // Validate the whole sequence before a visitor may mutate caller storage.
        for &id in ids {
            self.token_bytes(id, options)?;
        }
        let visible = || {
            ids.iter()
                .copied()
                .filter(|id| !(options.skip_special_tokens && self.is_special_id(*id)))
        };
        let mut iter = visible().peekable();
        let mut at_start = true;
        let mut emit = |bytes: &[u8]| {
            let bytes = if at_start && !bytes.is_empty() {
                at_start = false;
                bytes.strip_prefix(b" ").unwrap_or(bytes)
            } else {
                bytes
            };
            visit(bytes)
        };
        while let Some(id) = iter.next() {
            if self.fallback_bytes[id as usize].is_none() {
                emit(&self.decoded[id as usize])?;
                continue;
            }
            // A fallback run is one UTF-8 string. Upstream replaces every byte
            // in the entire run if that string is invalid, including valid prefixes.
            let run = std::iter::once(id)
                .chain(iter.clone())
                .take_while(|id| self.fallback_bytes[*id as usize].is_some());
            let valid = valid_utf8_bytes(
                run.clone()
                    .map(|id| self.fallback_bytes[id as usize].expect("fallback")),
            );
            let first_byte = [self.fallback_bytes[id as usize].expect("fallback")];
            if valid {
                emit(&first_byte)?;
            } else {
                emit("�".as_bytes())?;
            }
            while let Some(&next) = iter.peek() {
                let Some(byte) = self.fallback_bytes[next as usize] else {
                    break;
                };
                if valid {
                    emit(&[byte])?;
                } else {
                    emit("�".as_bytes())?;
                }
                iter.next();
            }
        }
        Ok(())
    }
}

fn valid_utf8_bytes(bytes: impl Iterator<Item = u8>) -> bool {
    let mut pending = [0; 4];
    let mut count = 0;
    let mut width = 0;
    for byte in bytes {
        if count == 0 {
            width = match byte {
                0..=0x7f => continue,
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return false,
            };
        }
        pending[count] = byte;
        count += 1;
        if count == width {
            if std::str::from_utf8(&pending[..width]).is_err() {
                return false;
            }
            count = 0;
        }
    }
    count == 0
}

impl Tokenizer for MistralTokenizer {
    fn encode(&self, input: &str, options: EncodeOptions) -> ModelResult<Vec<u32>> {
        let mut out = Vec::new();
        if options.add_special_tokens {
            out.push(self.bos);
        }
        let mut cursor = 0;
        while cursor < input.len() {
            let Some((start, end, id)) = self.added_trie.next_match(input, cursor) else {
                self.encode_plain(&input[cursor..], cursor, &mut out)?;
                break;
            };
            self.encode_plain(&input[cursor..start], cursor, &mut out)?;
            out.push(id);
            cursor = end;
        }
        Ok(out)
    }
    fn addressable_token_count(&self) -> usize {
        self.vocab_by_id.len()
    }
    fn maximum_decoded_token_bytes(&self) -> usize {
        self.maximum_decoded_bytes
    }
    fn strips_leading_decoder_space(&self) -> bool {
        true
    }
    fn decoded_bytes_len(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<usize> {
        let mut count = 0_usize;
        self.visit_decoded(ids, options, |bytes| {
            count = count
                .checked_add(bytes.len())
                .ok_or_else(|| invalid("decoded byte count overflow"))?;
            Ok(())
        })?;
        Ok(count)
    }
    fn decode_token_bytes_into(
        &self,
        id: u32,
        options: DecodeOptions,
        destination: &mut [u8],
    ) -> ModelResult<usize> {
        let bytes = self.token_bytes(id, options)?;
        ensure_decode_destination(bytes.len(), destination.len())?;
        destination[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
    fn decode_bytes(
        &self,
        ids: &[u32],
        options: DecodeOptions,
        destination: &mut [u8],
    ) -> ModelResult<usize> {
        let count = self.decoded_bytes_len(ids, options)?;
        ensure_decode_destination(count, destination.len())?;
        let mut offset = 0;
        self.visit_decoded(ids, options, |bytes| {
            destination[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
            Ok(())
        })?;
        Ok(offset)
    }
    fn decode(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<String> {
        let mut bytes = vec![0; self.decoded_bytes_len(ids, options)?];
        self.decode_bytes(ids, options, &mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("Mistral decoder produced invalid UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> Value {
        let mut vocab = BTreeMap::from([
            ("<unk>".to_owned(), 0),
            ("<s>".to_owned(), 1),
            ("</s>".to_owned(), 2),
        ]);
        for byte in 0..256 {
            vocab.insert(format!("<0x{byte:02X}>"), byte + 3);
        }
        for (offset, token) in ["▁", "h", "i", "▁h", "▁hi"].iter().enumerate() {
            vocab.insert((*token).to_owned(), u32::try_from(offset).unwrap() + 259);
        }
        let added: Vec<_> = [(0,"<unk>"),(1,"<s>"),(2,"</s>")].into_iter().map(|(id,content)| json!({
            "id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true
        })).collect();
        json!({
            "version":"1.0","truncation":null,"padding":null,"normalizer":null,"added_tokens":added,
            "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false},
            "post_processor":{"type":"TemplateProcessing",
                "single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                "pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},
                    {"SpecialToken":{"id":"<s>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],
                "special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}},
            "decoder":{"type":"Sequence","decoders":[
                {"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},
                {"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
            "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
                "end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false,
                "vocab":vocab,"merges":[["▁","h"],["▁h","i"]]}
        })
    }

    #[test]
    fn merges_unicode_and_byte_fallback_without_changing_special_token_offsets() {
        let tokenizer =
            MistralTokenizer::from_json_slice(&serde_json::to_vec(&artifact()).unwrap()).unwrap();
        assert_eq!(
            tokenizer.encode("hi", EncodeOptions::default()).unwrap(),
            [1, 263]
        );
        let options = DecodeOptions {
            skip_special_tokens: true,
        };
        for text in ["hi hi", "hi<s>hi", "<s>hi", "한😀hi", "  hi", "\t\nhi"] {
            let ids = tokenizer.encode(text, EncodeOptions::default()).unwrap();
            let expected = text.replace("<s>", "");
            let expected = expected.strip_prefix(' ').unwrap_or(&expected);
            // The artificial leading metaspace is removed once; one original
            // leading space is lost by this HF artifact's Strip decoder.
            assert_eq!(
                tokenizer.decode(&ids, options).unwrap(),
                expected,
                "{text:?}"
            );
        }
    }

    #[test]
    fn byte_fallback_replaces_the_whole_invalid_run_and_validates_before_writes() {
        let tokenizer =
            MistralTokenizer::from_json_slice(&serde_json::to_vec(&artifact()).unwrap()).unwrap();
        let options = DecodeOptions::default();
        assert_eq!(
            tokenizer.decode(&[0x61 + 3, 0xff + 3], options).unwrap(),
            "��"
        );
        assert_eq!(
            tokenizer
                .decode(&[0xe5 + 3, 0x8f + 3, 0xab + 3], options)
                .unwrap(),
            "叫"
        );
        let mut bytes = [0xa5; 8];
        assert!(
            tokenizer
                .decode_bytes(&[263, u32::MAX], options, &mut bytes)
                .is_err()
        );
        assert_eq!(bytes, [0xa5; 8]);
        assert!(
            tokenizer
                .decode_bytes(&[263], options, &mut bytes[..1])
                .is_err()
        );
        assert_eq!(bytes, [0xa5; 8]);
    }

    #[test]
    fn decoder_prefix_is_stripped_once_per_request_after_skipped_specials() {
        let tokenizer =
            MistralTokenizer::from_json_slice(&serde_json::to_vec(&artifact()).unwrap()).unwrap();
        let options = DecodeOptions {
            skip_special_tokens: true,
        };
        let mut state = TokenDecodeState::default();
        let mut bytes = [0; 8];
        assert!(
            tokenizer
                .decode_token_bytes_with_state(u32::MAX, options, &mut bytes, &mut state)
                .is_err()
        );
        assert_eq!(
            tokenizer
                .decode_token_bytes_with_state(1, options, &mut bytes, &mut state)
                .unwrap(),
            0
        );
        let first = tokenizer
            .decode_token_bytes_with_state(263, options, &mut bytes, &mut state)
            .unwrap();
        assert_eq!(&bytes[..first], b"hi");
        let second = tokenizer
            .decode_token_bytes_with_state(263, options, &mut bytes, &mut state)
            .unwrap();
        assert_eq!(&bytes[..second], b" hi");
        let mut other = TokenDecodeState::default();
        let next = tokenizer
            .decode_token_bytes_with_state(263, options, &mut bytes, &mut other)
            .unwrap();
        assert_eq!(&bytes[..next], b"hi");
    }
}
