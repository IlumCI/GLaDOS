//! Making invalid output unrepresentable.
//!
//! The usual way to get structured output from a language model is to ask for
//! it, parse what comes back, and retry when it is malformed. That is the only
//! option available when the model is behind an API and all you control is the
//! text going in and out. It is also why tool calling is usually said to need a
//! large model: most of the capability is spent on producing well-formed
//! syntax rather than on choosing correctly.
//!
//! We own the sampler, so none of that applies. At every step the set of tokens
//! that could keep the output valid is computable, and sampling can be
//! restricted to exactly that set. A malformed applet name is not unlikely
//! here, it is unreachable -- there is no sequence of sampling outcomes that
//! produces one. What is left for the model to be wrong about is *which* applet
//! to pick, which is the part that actually needs intelligence.
//!
//! The practical consequence is that model size stops gating correctness of
//! form. A 260K-parameter story model driven through this will choose
//! nonsensically and still never emit anything but a real applet name.

use super::tokenizer::Tokenizer;
use alloc::vec::Vec;

/// The decoded bytes of every token, precomputed once.
///
/// This cannot use the raw vocabulary strings. Byte-fallback pieces are stored
/// as the literal text `<0x0A>`, six bytes, not as the byte they denote -- so
/// matching a grammar against raw vocabulary entries would compare against the
/// escape rather than against a newline.
pub struct Alphabet {
    pieces: Vec<Vec<u8>>,
}

impl Alphabet {
    pub fn new(tok: &Tokenizer) -> Self {
        let mut pieces = Vec::with_capacity(tok.vocab_size());
        for id in 0..tok.vocab_size() {
            let mut buf = Vec::new();
            // prev = 0 rather than BOS: this is the token's own text, with no
            // dummy-prefix stripping applied.
            tok.append_piece(0, id, &mut buf);
            pieces.push(buf);
        }
        Self { pieces }
    }

    pub fn piece(&self, id: usize) -> &[u8] {
        self.pieces.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }
}

/// A set of complete strings the decoder is permitted to produce.
///
/// Every alternative carries a terminator. Without one, `snap` and `snaps`
/// are indistinguishable at the moment `snap` completes -- stopping at the
/// first complete match would make `snaps` unreachable, and not stopping would
/// make `snap` unreachable. The terminator makes the choice explicit and puts
/// it where it belongs, in the model's hands.
pub struct Grammar {
    alternatives: Vec<Vec<u8>>,
}

pub const TERMINATOR: u8 = b'\n';

impl Grammar {
    /// Each entry gains a trailing terminator.
    pub fn new<'a>(words: impl Iterator<Item = &'a str>) -> Self {
        let mut alternatives = Vec::new();
        for w in words {
            let mut v = w.as_bytes().to_vec();
            v.push(TERMINATOR);
            alternatives.push(v);
        }
        Self { alternatives }
    }

    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }
}

/// Position within a partially decoded string.
pub struct Cursor<'a> {
    grammar: &'a Grammar,
    produced: Vec<u8>,
    /// Leading whitespace is tolerated and not counted as content, because
    /// sentencepiece tokens routinely carry a leading space and forbidding
    /// them would leave the model choosing between mangled alternatives.
    started: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(grammar: &'a Grammar) -> Self {
        Self { grammar, produced: Vec::new(), started: false }
    }

    /// The content decoded so far, with leading whitespace removed.
    pub fn text(&self) -> &[u8] {
        &self.produced
    }

    /// Would appending `piece` keep at least one alternative reachable?
    fn admits(&self, piece: &[u8]) -> bool {
        if piece.is_empty() {
            return false;
        }
        // Before any content, a whitespace-only token is a no-op we allow.
        if !self.started && piece.iter().all(|b| *b == b' ') {
            return true;
        }
        let mut candidate = self.produced.clone();
        candidate.extend_from_slice(piece);
        self.grammar
            .alternatives
            .iter()
            .any(|alt| alt.len() >= candidate.len() && alt.starts_with(&candidate))
    }

    /// Token ids that may be sampled next. Empty means the decode is stuck,
    /// which the caller must treat as failure rather than sampling freely.
    pub fn candidates(&self, alphabet: &Alphabet) -> Vec<u32> {
        let mut out = Vec::new();
        for id in 0..alphabet.len() {
            if self.admits(alphabet.piece(id)) {
                out.push(id as u32);
            }
        }
        out
    }

    /// Returns false if the token was a leading-whitespace no-op that left the
    /// string exactly where it was.
    ///
    /// The caller has to know, because such a token consumes a decode step
    /// without making progress. Treating it as progress means the step budget
    /// can be spent entirely on spaces and the decode fails having produced
    /// nothing -- which is precisely what it did.
    pub fn push(&mut self, alphabet: &Alphabet, id: usize) -> bool {
        let piece = alphabet.piece(id);
        if !self.started {
            if piece.iter().all(|b| *b == b' ') {
                return false;
            }
            self.started = true;
        }
        self.produced.extend_from_slice(piece);
        true
    }

    /// Index of the completed alternative, if the string is now whole.
    pub fn finished(&self) -> Option<usize> {
        self.grammar.alternatives.iter().position(|alt| *alt == self.produced)
    }
}

/// How many *advancing* steps a decode may take before it is certainly stuck.
///
/// Every advancing step adds at least one byte, so the longest alternative
/// bounds it. Non-advancing steps are counted separately by the caller; the
/// first version of this folded them together, and a decode that opened with a
/// run of space tokens exhausted its budget before producing a character.
pub fn step_bound(g: &Grammar) -> usize {
    g.alternatives.iter().map(|a| a.len()).max().unwrap_or(0) + 1
}

/// Leading whitespace tokens tolerated before a decode is called stuck.
pub const MAX_LEADING_SPACES: usize = 4;
