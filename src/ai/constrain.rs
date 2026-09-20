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

    /// Build one from literal pieces, so a check can run with no model.
    ///
    /// Everything in this module is a pure function of (alternatives,
    /// vocabulary), and the vocabulary was reachable only through a loaded
    /// `Tokenizer` -- which put every claim about grammar behaviour behind a
    /// checkpoint and a forward pass. A synthetic vocabulary is how `costs`
    /// below gets asserted at boot the way `update::decide` is.
    pub fn from_pieces<'a>(pieces: impl Iterator<Item = &'a [u8]>) -> Self {
        Self { pieces: pieces.map(|p| p.to_vec()).collect() }
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

    /// Strip one leading space from the first piece of a decode.
    ///
    /// Sentencepiece emits the space as its own token, so tolerating
    /// whitespace-only pieces was enough. GPT-2 style byte-level BPE packs it
    /// *into* the word -- " ls" is a single token -- so without this, " ls"
    /// fails to prefix "ls\n" and the only reachable spellings are whichever
    /// space-less variants happen to exist. That took the constrained decode
    /// from always succeeding to almost never.
    fn trim_lead<'p>(&self, piece: &'p [u8]) -> &'p [u8] {
        if !self.started && piece.first() == Some(&b' ') {
            &piece[1..]
        } else {
            piece
        }
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
        let piece = self.trim_lead(piece);
        let n = self.produced.len();
        // Compared in place rather than by building `produced + piece`. This
        // runs once per vocabulary entry per step, so with a 49k vocabulary
        // the old clone was tens of millions of allocations per decode and
        // took the selftest from instant to minutes.
        self.grammar.alternatives.iter().any(|alt| {
            alt.len() >= n + piece.len()
                && alt.starts_with(&self.produced[..])
                && alt[n..].starts_with(piece)
        })
    }

    /// Does `id` move the string toward alternative `alt` in particular?
    ///
    /// `candidates` asks whether *some* alternative stays reachable, which is
    /// the question sampling needs. Teacher forcing needs the stronger one:
    /// the trainer already knows which applet the example is labelled with,
    /// and has to feed the token that spells it rather than whichever token
    /// the untrained model would have picked.
    ///
    /// Leading-whitespace no-ops answer false. They are legal to sample and
    /// carry no decision, so training on one would teach the model to spend a
    /// step saying nothing.
    pub fn advances_toward(&self, alphabet: &Alphabet, id: usize, alt: usize) -> bool {
        let piece = alphabet.piece(id);
        if piece.is_empty() {
            return false;
        }
        if !self.started && piece.iter().all(|b| *b == b' ') {
            return false;
        }
        let piece = self.trim_lead(piece);
        if piece.is_empty() {
            return false;
        }
        let Some(a) = self.grammar.alternatives.get(alt) else {
            return false;
        };
        let n = self.produced.len();
        a.len() >= n + piece.len()
            && a.starts_with(&self.produced[..])
            && a[n..].starts_with(piece)
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
            let trimmed = self.trim_lead(piece);
            self.started = true;
            self.produced.extend_from_slice(trimmed);
            return true;
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

/// What it costs the decoder to *say* one alternative.
///
/// This exists because of a measurement `repair` made on a two-row table. The
/// model picked `retry` five times out of six wherever `retry` sat in the
/// list, and reversing the order changed the answer once -- which rules out
/// position and leaves the name itself. `retry` is one common English token
/// and `skip-hwp` is several uncommon pieces, so under a constrained grammar
/// the cheapest first-token path wins and **a name carries probability mass
/// that has nothing to do with what it names**.
///
/// That is a claim about constrained decoding rather than about repairs, and
/// the applet grammar is the same machinery over twenty-three rows. Nothing
/// had ever asked whether the same asymmetry sits in the action space, so this
/// measures it instead of assuming either way.
///
/// It is deliberately **structure only**. Every field here is a property of
/// (name, vocabulary) and none of them needs a forward pass, so this says what
/// is *available* to the decoder and never what the decoder does with it. The
/// half that needs a model -- whether cheap names are actually over-picked --
/// is a separate measurement and must not be inferred from these numbers.
pub struct Cost {
    /// Index into the grammar's alternatives.
    pub alt: usize,
    /// Fewest tokens whose bytes concatenate to the whole alternative,
    /// terminator included. `None` when no path spells it at all, which is a
    /// name the decoder can never finish.
    pub min_tokens: Option<usize>,
    /// A single token spells the name body exactly, terminator aside.
    ///
    /// Separate from `min_tokens` because the terminator is a constant every
    /// alternative pays, so it cannot bias one against another -- while "is
    /// this name one lexical unit to the tokenizer" is exactly the property
    /// that separated `retry` from `skip-hwp`.
    pub body_one_token: bool,
    /// Distinct vocabulary entries admitted at step 0 that advance toward this
    /// alternative.
    ///
    /// **This is branching and must not be read as "easier".** It was added
    /// meaning "how much opening mass can flow this way", and the measurement
    /// showed it mostly counts *prefixes*: a longer name has more of its own
    /// prefixes in the vocabulary, so `remember` scores 10 against `mv`'s 3
    /// without being any cheaper to reach -- both cost two tokens. More entry
    /// points spread the same mass over more first choices rather than
    /// attracting more of it. Separating the two needs logits, which this
    /// deliberately does not have.
    pub first_step: usize,
}

/// Fewest tokens spelling `alt` end to end, or `None` if nothing does.
///
/// A forward dynamic program over byte positions: `best[i]` is the cheapest
/// way to have produced the first `i` bytes. Every admitted piece is non-empty
/// so each relaxation strictly advances, which is what makes one forward sweep
/// enough.
///
/// The `i == 0` case mirrors `Cursor::trim_lead` rather than re-deriving it:
/// byte-level BPE packs the leading space *into* the word, so " ls" is a
/// single token that spells `ls` from the start and nowhere else. Missing that
/// would report every name as costing one token more than it does, which is
/// the exact failure mode `trim_lead`'s own doc records.
fn min_path(alt: &[u8], alphabet: &Alphabet) -> Option<usize> {
    let n = alt.len();
    let mut best: Vec<Option<usize>> = alloc::vec![None; n + 1];
    best[0] = Some(0);
    for i in 0..n {
        let Some(cost) = best[i] else { continue };
        for id in 0..alphabet.len() {
            let raw = alphabet.piece(id);
            let piece = if i == 0 && raw.first() == Some(&b' ') { &raw[1..] } else { raw };
            // An empty or whitespace-only piece is legal to sample at step 0
            // and advances nothing, so it can never be on a shortest path.
            if piece.is_empty() || piece.iter().all(|b| *b == b' ') {
                continue;
            }
            if !alt[i..].starts_with(piece) {
                continue;
            }
            let j = i + piece.len();
            if best[j].map_or(true, |b| b > cost + 1) {
                best[j] = Some(cost + 1);
            }
        }
    }
    best[n]
}

/// Measure every alternative in a grammar against a vocabulary.
///
/// `first_step` goes through `Cursor::advances_toward` rather than repeating
/// its prefix arithmetic, because a second implementation of "does this token
/// move toward that alternative" is exactly the pair that stops agreeing.
pub fn costs(grammar: &Grammar, alphabet: &Alphabet) -> Vec<Cost> {
    let mut out = Vec::new();
    for (alt, bytes) in grammar.alternatives.iter().enumerate() {
        let body = &bytes[..bytes.len().saturating_sub(1)];
        let mut body_one_token = false;
        let mut first_step = 0usize;
        let cursor = Cursor::new(grammar);
        for id in 0..alphabet.len() {
            let raw = alphabet.piece(id);
            let trimmed = if raw.first() == Some(&b' ') { &raw[1..] } else { raw };
            if !trimmed.is_empty() && trimmed == body {
                body_one_token = true;
            }
            if cursor.advances_toward(alphabet, id, alt) {
                first_step += 1;
            }
        }
        out.push(Cost {
            alt,
            min_tokens: min_path(bytes, alphabet),
            body_one_token,
            first_step,
        });
    }
    out
}

/// The alternatives, for a caller that wants to name what `costs` measured.
pub fn alternatives(g: &Grammar) -> &[Vec<u8>] {
    &g.alternatives
}
