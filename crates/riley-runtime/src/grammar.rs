//! Byte-level grammar constraints for structured output.
//!
//! A [`Grammar`] is a context-free grammar over bytes without left recursion.
//! Its parse state is a set of stacks, as in llama.cpp's GBNF sampler, but
//! each distinct set is interned as a state of a lazily built DFA, so a state
//! seen before costs one table lookup per byte. [`GrammarAutomaton::apply_mask`]
//! walks a [`TokenIndex`] trie from the current state and memoizes the
//! resulting vocabulary mask per state, which keeps a free-text JSON string
//! (where nearly the whole vocabulary is legal) from costing a vocabulary scan
//! on every generated token.

use std::collections::HashMap;
use std::sync::Arc;

const UNKNOWN: u32 = u32::MAX;
const DEAD: u32 = u32::MAX - 1;
/// Soft bound on interned states; 4 MiB of transition tables at this size.
const MAX_STATES: usize = 4_096;
/// Parse stacks deeper than this are abandoned, bounding nested JSON.
const MAX_STACK_DEPTH: usize = 128;
/// Ambiguity bound; generated grammars stay far below it.
const MAX_STACKS_PER_STATE: usize = 1_024;

/// A set of bytes matched by one grammar position.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ByteClass([u64; 4]);

impl ByteClass {
    /// One byte.
    #[must_use]
    pub const fn byte(value: u8) -> Self {
        Self::range(value, value)
    }

    /// An inclusive byte range.
    #[must_use]
    pub const fn range(first: u8, last: u8) -> Self {
        let mut class = Self([0; 4]);
        let mut value = first as usize;
        while value <= last as usize {
            class.0[value / 64] |= 1 << (value % 64);
            value += 1;
        }
        class
    }

    /// Union of two classes.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// Whether `value` belongs to the class.
    #[must_use]
    pub const fn contains(self, value: u8) -> bool {
        self.0[value as usize / 64] & (1 << (value as usize % 64)) != 0
    }
}

/// One position of a rule alternative.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrammarElement {
    /// Consumes one byte from the class.
    Bytes(ByteClass),
    /// Expands another rule.
    Rule(u32),
}

/// Why a [`GrammarBuilder`] could not produce a grammar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrammarError {
    /// A rule has no alternative or refers to a missing rule.
    MalformedRule,
    /// A rule can reach itself without consuming a byte.
    LeftRecursion,
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MalformedRule => "grammar rule is malformed",
            Self::LeftRecursion => "grammar is left-recursive",
        })
    }
}

impl std::error::Error for GrammarError {}

/// Incrementally assembles a [`Grammar`].
#[derive(Clone, Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Vec<Vec<GrammarElement>>>,
}

impl GrammarBuilder {
    /// Reserves a rule identifier; alternatives may be added later, so rules
    /// can refer to each other.
    pub fn rule(&mut self) -> u32 {
        self.rules.push(Vec::new());
        u32::try_from(self.rules.len() - 1).unwrap_or(u32::MAX)
    }

    /// Adds one alternative to `rule`. An empty alternative matches nothing.
    pub fn alternative(&mut self, rule: u32, elements: Vec<GrammarElement>) {
        if let Some(alternatives) = self.rules.get_mut(rule as usize) {
            alternatives.push(elements);
        }
    }

    /// Number of rules reserved so far.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validates the rules and fixes `start` as the entry point.
    ///
    /// # Errors
    ///
    /// Returns [`GrammarError`] for an empty or dangling rule or left recursion.
    pub fn build(self, start: u32) -> Result<Grammar, GrammarError> {
        let rule_count = self.rules.len();
        if start as usize >= rule_count {
            return Err(GrammarError::MalformedRule);
        }
        for alternatives in &self.rules {
            if alternatives.is_empty() {
                return Err(GrammarError::MalformedRule);
            }
            for element in alternatives.iter().flatten() {
                if matches!(element, GrammarElement::Rule(rule) if *rule as usize >= rule_count) {
                    return Err(GrammarError::MalformedRule);
                }
            }
        }
        reject_left_recursion(&self.rules)?;
        Ok(Grammar {
            rules: self.rules,
            start,
        })
    }
}

fn nullable_rules(rules: &[Vec<Vec<GrammarElement>>]) -> Vec<bool> {
    let mut nullable = vec![false; rules.len()];
    loop {
        let mut changed = false;
        for (index, alternatives) in rules.iter().enumerate() {
            let now = alternatives.iter().any(|alternative| {
                alternative.iter().all(|element| match element {
                    GrammarElement::Bytes(_) => false,
                    GrammarElement::Rule(rule) => nullable[*rule as usize],
                })
            });
            if now && !nullable[index] {
                nullable[index] = true;
                changed = true;
            }
        }
        if !changed {
            return nullable;
        }
    }
}

fn reject_left_recursion(rules: &[Vec<Vec<GrammarElement>>]) -> Result<(), GrammarError> {
    let nullable = nullable_rules(rules);
    // 0 unvisited, 1 on the current path, 2 finished.
    let mut marks = vec![0_u8; rules.len()];
    fn visit(
        rule: usize,
        rules: &[Vec<Vec<GrammarElement>>],
        nullable: &[bool],
        marks: &mut [u8],
    ) -> Result<(), GrammarError> {
        match marks[rule] {
            1 => return Err(GrammarError::LeftRecursion),
            2 => return Ok(()),
            _ => marks[rule] = 1,
        }
        for alternative in &rules[rule] {
            for element in alternative {
                match element {
                    GrammarElement::Bytes(_) => break,
                    GrammarElement::Rule(next) => {
                        visit(*next as usize, rules, nullable, marks)?;
                        if !nullable[*next as usize] {
                            break;
                        }
                    }
                }
            }
        }
        marks[rule] = 2;
        Ok(())
    }
    for rule in 0..rules.len() {
        visit(rule, rules, &nullable, &mut marks)?;
    }
    Ok(())
}

/// A validated byte grammar.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Grammar {
    rules: Vec<Vec<Vec<GrammarElement>>>,
    start: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Frame {
    rule: u32,
    alternative: u32,
    position: u32,
}

type Stack = Vec<Frame>;
type StackSet = Arc<Vec<Stack>>;

impl Grammar {
    fn element(&self, frame: Frame) -> Option<GrammarElement> {
        self.rules[frame.rule as usize][frame.alternative as usize]
            .get(frame.position as usize)
            .copied()
    }

    /// Resolves rule references until every stack has a byte class on top or
    /// is empty, which means the input may end here.
    fn expand(&self, mut stack: Stack, output: &mut Vec<Stack>) {
        loop {
            let Some(frame) = stack.pop() else {
                output.push(stack);
                return;
            };
            match self.element(frame) {
                None => {}
                Some(GrammarElement::Bytes(_)) => {
                    stack.push(frame);
                    output.push(stack);
                    return;
                }
                Some(GrammarElement::Rule(rule)) => {
                    let next = Frame {
                        position: frame.position + 1,
                        ..frame
                    };
                    // A rule in tail position leaves no frame behind, so
                    // right recursion runs in constant stack.
                    if self.element(next).is_some() {
                        stack.push(next);
                    }
                    if stack.len() >= MAX_STACK_DEPTH {
                        return;
                    }
                    for alternative in 0..self.rules[rule as usize].len() {
                        let mut branch = stack.clone();
                        branch.push(Frame {
                            rule,
                            alternative: u32::try_from(alternative).unwrap_or(u32::MAX),
                            position: 0,
                        });
                        self.expand(branch, output);
                    }
                    return;
                }
            }
        }
    }

    fn canonical(mut stacks: Vec<Stack>) -> Vec<Stack> {
        stacks.sort_unstable();
        stacks.dedup();
        stacks.truncate(MAX_STACKS_PER_STATE);
        stacks
    }

    fn start_stacks(&self) -> Vec<Stack> {
        let mut output = Vec::new();
        for alternative in 0..self.rules[self.start as usize].len() {
            self.expand(
                vec![Frame {
                    rule: self.start,
                    alternative: u32::try_from(alternative).unwrap_or(u32::MAX),
                    position: 0,
                }],
                &mut output,
            );
        }
        Self::canonical(output)
    }

    fn advance_stacks(&self, stacks: &[Stack], byte: u8) -> Vec<Stack> {
        let mut output = Vec::new();
        for stack in stacks {
            let Some(&top) = stack.last() else {
                continue;
            };
            let Some(GrammarElement::Bytes(class)) = self.element(top) else {
                continue;
            };
            if class.contains(byte) {
                let mut next = stack.clone();
                if let Some(frame) = next.last_mut() {
                    frame.position += 1;
                }
                self.expand(next, &mut output);
            }
        }
        Self::canonical(output)
    }
}

/// Maps every usable token to its bytes as a trie.
///
/// Tokens that decode to nothing (special tokens) are outside the index, so
/// no grammar state ever allows them; end-of-sequence is the caller's choice.
#[derive(Clone, Debug)]
pub struct TokenIndex {
    vocabulary_size: usize,
    /// Byte offsets of each token's bytes in `bytes`; `offsets[id + 1]` ends it.
    offsets: Vec<u32>,
    bytes: Vec<u8>,
    /// Children of node `n` are `edges[edge_starts[n]..edge_starts[n + 1]]`.
    edge_starts: Vec<u32>,
    edges: Vec<(u8, u32)>,
    /// Tokens that end at node `n` are `tokens[token_starts[n]..token_starts[n + 1]]`.
    token_starts: Vec<u32>,
    tokens: Vec<u32>,
}

impl TokenIndex {
    /// Builds the trie from each token's decoded bytes, in token-ID order.
    #[must_use]
    pub fn new(token_bytes: &[Vec<u8>]) -> Self {
        let mut offsets = Vec::with_capacity(token_bytes.len() + 1);
        let mut bytes = Vec::new();
        offsets.push(0_u32);
        for token in token_bytes {
            bytes.extend_from_slice(token);
            offsets.push(u32::try_from(bytes.len()).unwrap_or(u32::MAX));
        }
        let mut order: Vec<u32> = (0..token_bytes.len())
            .filter(|&id| !token_bytes[id].is_empty())
            .map(|id| u32::try_from(id).unwrap_or(u32::MAX))
            .collect();
        order.sort_unstable_by(|left, right| {
            token_bytes[*left as usize].cmp(&token_bytes[*right as usize])
        });

        // Children are appended depth-first, then flattened by node number.
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut terminals: Vec<Vec<u32>> = vec![Vec::new()];
        for id in order {
            let mut node = 0_usize;
            for &byte in &token_bytes[id as usize] {
                node = match children[node].last() {
                    Some(&(last, next)) if last == byte => next as usize,
                    _ => {
                        let next = children.len();
                        children[node].push((byte, u32::try_from(next).unwrap_or(u32::MAX)));
                        children.push(Vec::new());
                        terminals.push(Vec::new());
                        next
                    }
                };
            }
            terminals[node].push(id);
        }
        let mut edge_starts = Vec::with_capacity(children.len() + 1);
        let mut edges = Vec::new();
        let mut token_starts = Vec::with_capacity(children.len() + 1);
        let mut tokens = Vec::new();
        for (node, node_edges) in children.iter().enumerate() {
            edge_starts.push(u32::try_from(edges.len()).unwrap_or(u32::MAX));
            edges.extend_from_slice(node_edges);
            token_starts.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            tokens.extend_from_slice(&terminals[node]);
        }
        edge_starts.push(u32::try_from(edges.len()).unwrap_or(u32::MAX));
        token_starts.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        Self {
            vocabulary_size: token_bytes.len(),
            offsets,
            bytes,
            edge_starts,
            edges,
            token_starts,
            tokens,
        }
    }

    /// Size of the indexed vocabulary, usable tokens or not.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Bytes `id` decodes to; empty for special or out-of-range tokens.
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let index = id as usize;
        if index + 1 >= self.offsets.len() {
            return &[];
        }
        &self.bytes[self.offsets[index] as usize..self.offsets[index + 1] as usize]
    }
}

/// A request's position in a grammar.
#[derive(Clone, Debug)]
pub struct GrammarCursor {
    stacks: StackSet,
    state: u32,
    generation: u64,
}

struct DfaState {
    stacks: StackSet,
    accepting: bool,
    mask: Option<Box<[u64]>>,
}

/// Lazily determinized grammar shared by every request that uses it.
pub struct GrammarAutomaton {
    grammar: Arc<Grammar>,
    states: Vec<DfaState>,
    transitions: Vec<[u32; 256]>,
    ids: HashMap<StackSet, u32>,
    generation: u64,
}

impl std::fmt::Debug for GrammarAutomaton {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GrammarAutomaton")
            .field("states", &self.states.len())
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl GrammarAutomaton {
    /// Starts with no state interned.
    #[must_use]
    pub fn new(grammar: Arc<Grammar>) -> Self {
        Self {
            grammar,
            states: Vec::new(),
            transitions: Vec::new(),
            ids: HashMap::new(),
            generation: 0,
        }
    }

    /// Interned states, for tests and diagnostics.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    fn intern(&mut self, stacks: StackSet) -> u32 {
        if let Some(&id) = self.ids.get(&stacks) {
            return id;
        }
        let id = u32::try_from(self.states.len()).unwrap_or(DEAD - 1);
        let accepting = stacks.iter().any(Vec::is_empty);
        self.states.push(DfaState {
            stacks: Arc::clone(&stacks),
            accepting,
            mask: None,
        });
        self.transitions.push([UNKNOWN; 256]);
        self.ids.insert(stacks, id);
        id
    }

    /// A cursor before the first byte.
    pub fn start(&mut self) -> GrammarCursor {
        let stacks = Arc::new(self.grammar.start_stacks());
        let state = self.intern(Arc::clone(&stacks));
        GrammarCursor {
            stacks,
            state,
            generation: self.generation,
        }
    }

    fn step(&mut self, state: u32, byte: u8) -> u32 {
        let cached = self.transitions[state as usize][byte as usize];
        if cached != UNKNOWN {
            return cached;
        }
        let next_stacks = self
            .grammar
            .advance_stacks(&self.states[state as usize].stacks, byte);
        let next = if next_stacks.is_empty() {
            DEAD
        } else {
            self.intern(Arc::new(next_stacks))
        };
        self.transitions[state as usize][byte as usize] = next;
        next
    }

    /// Drops every interned state once the soft bound is exceeded. Cursors
    /// keep their own stacks, so they re-intern on next use.
    fn trim(&mut self) {
        if self.states.len() > MAX_STATES {
            self.states.clear();
            self.transitions.clear();
            self.ids.clear();
            self.generation += 1;
        }
    }

    fn resolve(&mut self, cursor: &mut GrammarCursor) {
        if cursor.generation != self.generation {
            cursor.state = self.intern(Arc::clone(&cursor.stacks));
            cursor.generation = self.generation;
        }
    }

    /// Whether the input so far is a complete sentence.
    pub fn is_accepting(&mut self, cursor: &mut GrammarCursor) -> bool {
        self.resolve(cursor);
        self.states[cursor.state as usize].accepting
    }

    /// Consumes `bytes`. On `false` the bytes left the grammar and the cursor
    /// is unchanged.
    pub fn advance(&mut self, cursor: &mut GrammarCursor, bytes: &[u8]) -> bool {
        self.trim();
        self.resolve(cursor);
        let mut state = cursor.state;
        for &byte in bytes {
            state = self.step(state, byte);
            if state == DEAD {
                return false;
            }
        }
        cursor.state = state;
        cursor.stacks = Arc::clone(&self.states[state as usize].stacks);
        true
    }

    /// Clears every entry of `allowed` whose token cannot continue from
    /// `cursor`. Returns whether any token remains.
    ///
    /// # Panics
    ///
    /// Panics when `allowed` is shorter than the indexed vocabulary.
    pub fn apply_mask(
        &mut self,
        cursor: &mut GrammarCursor,
        index: &TokenIndex,
        allowed: &mut [bool],
    ) -> bool {
        self.trim();
        self.resolve(cursor);
        let state = cursor.state as usize;
        if self.states[state].mask.is_none() {
            let mask = self.compute_mask(cursor.state, index);
            self.states[state].mask = Some(mask);
        }
        let mask = self.states[state].mask.as_deref().unwrap_or(&[]);
        let mut any = false;
        for (id, entry) in allowed[..index.vocabulary_size].iter_mut().enumerate() {
            *entry = *entry && mask[id / 64] & (1 << (id % 64)) != 0;
            any |= *entry;
        }
        any
    }

    /// Whether the token with this ID can continue from `cursor`.
    pub fn allows(
        &mut self,
        cursor: &mut GrammarCursor,
        index: &TokenIndex,
        token_id: u32,
    ) -> bool {
        self.trim();
        self.resolve(cursor);
        let state = cursor.state as usize;
        if self.states[state].mask.is_none() {
            let mask = self.compute_mask(cursor.state, index);
            self.states[state].mask = Some(mask);
        }
        let mask = self.states[state].mask.as_deref().unwrap_or(&[]);
        mask.get(token_id as usize / 64)
            .is_some_and(|word| word & (1 << (token_id % 64)) != 0)
    }

    fn compute_mask(&mut self, state: u32, index: &TokenIndex) -> Box<[u64]> {
        let mut mask = vec![0_u64; index.vocabulary_size.div_ceil(64)].into_boxed_slice();
        let mut pending = vec![(0_u32, state)];
        while let Some((node, node_state)) = pending.pop() {
            let node = node as usize;
            let tokens = index.token_starts[node] as usize..index.token_starts[node + 1] as usize;
            for &token in &index.tokens[tokens] {
                mask[token as usize / 64] |= 1 << (token % 64);
            }
            let edges = index.edge_starts[node] as usize..index.edge_starts[node + 1] as usize;
            for edge in edges {
                let (byte, child) = index.edges[edge];
                let next = self.step(node_state, byte);
                if next != DEAD {
                    pending.push((child, next));
                }
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(text: &str) -> Vec<GrammarElement> {
        text.bytes()
            .map(|byte| GrammarElement::Bytes(ByteClass::byte(byte)))
            .collect()
    }

    /// `a(b|c)*d`, written with right recursion.
    fn looping() -> Arc<Grammar> {
        let mut builder = GrammarBuilder::default();
        let start = builder.rule();
        let tail = builder.rule();
        let mut first = literal("a");
        first.push(GrammarElement::Rule(tail));
        builder.alternative(start, first);
        builder.alternative(
            tail,
            vec![
                GrammarElement::Bytes(ByteClass::range(b'b', b'c')),
                GrammarElement::Rule(tail),
            ],
        );
        builder.alternative(tail, literal("d"));
        Arc::new(builder.build(start).unwrap())
    }

    #[test]
    fn accepts_exactly_the_language() {
        let mut automaton = GrammarAutomaton::new(looping());
        let mut cursor = automaton.start();
        assert!(!automaton.is_accepting(&mut cursor));
        assert!(automaton.advance(&mut cursor, b"abcb"));
        assert!(!automaton.is_accepting(&mut cursor));
        assert!(!automaton.advance(&mut cursor, b"x"));
        assert!(automaton.advance(&mut cursor, b"d"));
        assert!(automaton.is_accepting(&mut cursor));
        assert!(!automaton.advance(&mut cursor, b"d"));
    }

    #[test]
    fn right_recursion_keeps_a_constant_stack_and_few_states() {
        let mut automaton = GrammarAutomaton::new(looping());
        let mut cursor = automaton.start();
        assert!(automaton.advance(&mut cursor, b"a"));
        for _ in 0..10_000 {
            assert!(automaton.advance(&mut cursor, b"b"));
        }
        assert!(automaton.state_count() <= 3);
    }

    #[test]
    fn mask_allows_tokens_whose_whole_bytes_fit() {
        let tokens: Vec<Vec<u8>> = [&b"a"[..], b"ab", b"abd", b"bd", b"x", b"", b"ax"]
            .iter()
            .map(|token| token.to_vec())
            .collect();
        let index = TokenIndex::new(&tokens);
        let mut automaton = GrammarAutomaton::new(looping());
        let mut cursor = automaton.start();
        let mut allowed = vec![true; tokens.len()];
        assert!(automaton.apply_mask(&mut cursor, &index, &mut allowed));
        assert_eq!(allowed, [true, true, true, false, false, false, false]);
        assert!(automaton.advance(&mut cursor, b"a"));
        let mut allowed = vec![true; tokens.len()];
        automaton.apply_mask(&mut cursor, &index, &mut allowed);
        assert_eq!(allowed, [false, false, false, true, false, false, false]);
        // The mask only ever narrows what the caller allowed.
        let mut allowed = vec![false; tokens.len()];
        assert!(!automaton.apply_mask(&mut cursor, &index, &mut allowed));
        // A single token's answer is that of the mask.
        for (id, expected) in [false, false, false, true, false, false, false]
            .into_iter()
            .enumerate()
        {
            assert_eq!(automaton.allows(&mut cursor, &index, id as u32), expected);
        }
        assert!(!automaton.allows(&mut cursor, &index, 4096));
    }

    #[test]
    fn cursors_survive_a_state_table_reset() {
        let mut automaton = GrammarAutomaton::new(looping());
        let mut cursor = automaton.start();
        assert!(automaton.advance(&mut cursor, b"ab"));
        automaton.states.clear();
        automaton.transitions.clear();
        automaton.ids.clear();
        automaton.generation += 1;
        assert!(automaton.advance(&mut cursor, b"cd"));
        assert!(automaton.is_accepting(&mut cursor));
    }

    #[test]
    fn left_recursion_and_dangling_rules_are_rejected() {
        let mut builder = GrammarBuilder::default();
        let start = builder.rule();
        let empty = builder.rule();
        builder.alternative(empty, Vec::new());
        builder.alternative(
            start,
            vec![GrammarElement::Rule(empty), GrammarElement::Rule(start)],
        );
        assert_eq!(builder.build(start), Err(GrammarError::LeftRecursion));

        let mut builder = GrammarBuilder::default();
        let start = builder.rule();
        builder.alternative(start, vec![GrammarElement::Rule(9)]);
        assert_eq!(builder.build(start), Err(GrammarError::MalformedRule));
    }
}
