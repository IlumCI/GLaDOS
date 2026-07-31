//! The llama2.c `tokenizer.bin` format, and byte-pair encoding over it.
//!
//! Layout: an i32 `max_token_length`, then one entry per vocabulary token --
//! an f32 merge score, an i32 length, and that many raw bytes. The vocabulary
//! size is *not* in the file. It comes from the model header, which is what
//! ties a tokenizer to a particular checkpoint: hand it the wrong pair and it
//! will parse happily and produce nonsense.
//!
//! This replaces the byte-level vocabulary `model.rs` was written around. That
//! choice existed to avoid needing a file on a machine that could not read
//! files; now that it can, a real vocabulary costs one 6 KB read and buys
//! roughly four times the effective context per token.

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Sentencepiece is trained here with `unk_id=0, bos_id=1, eos_id=2`.
pub const UNK: usize = 0;
pub const BOS: usize = 1;
pub const EOS: usize = 2;

/// With `byte_fallback=True` the 256 single-byte pieces follow the three
/// control ids, so any byte the vocabulary cannot express becomes `byte + 3`.
const BYTE_FALLBACK_BASE: usize = 3;

const V2_MAGIC: &[u8; 8] = b"GLADOSTK";
const V2_VERSION: u32 = 2;

pub const FLAG_DUMMY_PREFIX: u32 = 1 << 0;
pub const FLAG_INDIVIDUAL_DIGITS: u32 = 1 << 1;

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    /// Token ids ordered by their byte string, so a merge candidate can be
    /// looked up by binary search instead of scanning the vocabulary.
    sorted: Vec<u32>,
    pub max_token_length: usize,

    /// v2 only. The legacy llama2.c format has none of this and infers it,
    /// which is exactly the problem: it assumes sentencepiece.
    v2: bool,
    /// Which token represents each raw byte on its own. Replaces the
    /// `byte + 3` guess, which is right for sentencepiece with byte_fallback
    /// and wrong for everything else.
    byte_table: Vec<u32>,
    /// Added tokens, longest first. Matched literally before any BPE.
    specials: Vec<u32>,
    flags: u32,
    bos_id: usize,
    eos_id: usize,
}

impl Tokenizer {
    /// Parse either format. v2 announces itself; the legacy llama2.c layout
    /// has no magic and starts straight into an i32.
    pub fn from_bytes(data: &[u8], vocab_size: usize) -> Option<Self> {
        if data.len() >= 8 && &data[0..8] == V2_MAGIC {
            Self::from_v2(data)
        } else {
            Self::from_legacy(data, vocab_size)
        }
    }

    fn from_v2(data: &[u8]) -> Option<Self> {
        let u32_at = |o: usize| -> Option<u32> {
            if o + 4 > data.len() {
                return None;
            }
            Some(u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]))
        };
        if u32_at(8)? != V2_VERSION {
            return None;
        }
        let size = u32_at(12)? as usize;
        let max_token_length = u32_at(16)? as usize;
        let flags = u32_at(20)?;
        let bos_id = u32_at(24)? as usize;
        let eos_id = u32_at(28)? as usize;
        let _unk = u32_at(32)? as usize;

        let mut o = 36;
        let mut byte_table = Vec::new();
        byte_table.try_reserve_exact(256).ok()?;
        for i in 0..256 {
            byte_table.push(u32_at(o + i * 4)?);
        }
        o += 256 * 4;

        let n_specials = u32_at(o)? as usize;
        o += 4;
        let mut specials = Vec::new();
        specials.try_reserve_exact(n_specials).ok()?;
        for i in 0..n_specials {
            specials.push(u32_at(o + i * 4)?);
        }
        o += n_specials * 4;

        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        vocab.try_reserve_exact(size).ok()?;
        scores.try_reserve_exact(size).ok()?;
        for _ in 0..size {
            if o + 8 > data.len() {
                return None;
            }
            let score = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            let len = u32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]) as usize;
            o += 8;
            if o + len > data.len() {
                return None;
            }
            vocab.push(data[o..o + len].to_vec());
            scores.push(score);
            o += len;
        }

        let mut sorted: Vec<u32> = (0..size as u32).collect();
        sorted.sort_by(|a, b| vocab[*a as usize].cmp(&vocab[*b as usize]));

        Some(Self {
            vocab,
            scores,
            sorted,
            max_token_length,
            v2: true,
            byte_table,
            specials,
            flags,
            bos_id,
            eos_id,
        })
    }

    fn from_legacy(data: &[u8], vocab_size: usize) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let max_token_length =
            i32::from_le_bytes([data[0], data[1], data[2], data[3]]).max(0) as usize;

        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut o = 4usize;
        for _ in 0..vocab_size {
            if o + 8 > data.len() {
                return None;
            }
            let score = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            let len = i32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]);
            o += 8;
            if len < 0 {
                return None;
            }
            let len = len as usize;
            if o + len > data.len() {
                return None;
            }
            vocab.push(data[o..o + len].to_vec());
            scores.push(score);
            o += len;
        }

        let mut sorted: Vec<u32> = (0..vocab_size as u32).collect();
        sorted.sort_by(|a, b| vocab[*a as usize].cmp(&vocab[*b as usize]));

        Some(Self {
            vocab,
            scores,
            sorted,
            max_token_length,
            v2: false,
            // Sentencepiece with byte_fallback puts the 256 byte pieces right
            // after unk/bos/eos.
            byte_table: (0..256u32).map(|b| b + BYTE_FALLBACK_BASE as u32).collect(),
            specials: Vec::new(),
            flags: FLAG_DUMMY_PREFIX,
            bos_id: BOS,
            eos_id: EOS,
        })
    }

    pub fn bos(&self) -> usize {
        self.bos_id
    }

    pub fn eos(&self) -> usize {
        self.eos_id
    }

    pub fn is_v2(&self) -> bool {
        self.v2
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_bytes(&self, id: usize) -> &[u8] {
        self.vocab.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn lookup(&self, needle: &[u8]) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.sorted.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let id = self.sorted[mid] as usize;
            match self.vocab[id].as_slice().cmp(needle) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(id),
            }
        }
        None
    }

    /// Encode text to token ids.
    ///
    /// Greedy BPE: seed with one token per character, then repeatedly merge
    /// whichever adjacent pair has the highest score in the vocabulary. That is
    /// O(n^2) in the token count per merge and there are up to n merges, which
    /// is fine for prompts and would not be for documents.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        if self.v2 {
            return self.encode_v2(text, bos, eos);
        }
        self.encode_legacy(text, bos, eos)
    }

    /// Byte-level BPE, the GPT-2 lineage.
    ///
    /// Three things differ from the sentencepiece path and all three are
    /// silent when wrong: no dummy space, seeds come from the byte table
    /// rather than from a character lookup, and added tokens are matched
    /// literally before anything else -- otherwise BPE shreds `<|im_start|>`
    /// into twenty tokens and the model never learns who is speaking.
    fn encode_v2(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        if bos {
            out.push(self.bos_id);
        }

        let bytes = text.as_bytes();
        let mut i = 0usize;
        let mut chunk = 0usize;
        while i < bytes.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let mut hit = None;
            for &sid in self.specials.iter() {
                let s = self.token_bytes(sid as usize);
                if !s.is_empty() && bytes[i..].starts_with(s) {
                    hit = Some((sid as usize, s.len()));
                    break;
                }
            }
            match hit {
                Some((sid, len)) => {
                    if i > chunk {
                        self.bpe_chunk(&text[chunk..i], &mut out);
                    }
                    out.push(sid);
                    i += len;
                    chunk = i;
                }
                None => i += 1,
            }
        }
        if chunk < bytes.len() {
            self.bpe_chunk(&text[chunk..], &mut out);
        }

        if eos {
            out.push(self.eos_id);
        }
        out
    }

    fn bpe_chunk(&self, text: &str, out: &mut Vec<usize>) {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        pretokenize(text, self.flags & FLAG_INDIVIDUAL_DIGITS != 0, &mut spans);

        let mut toks: Vec<usize> = Vec::new();
        let mut joined: Vec<u8> = Vec::new();
        for (a, b) in spans {
            toks.clear();
            for &byte in text[a..b].as_bytes() {
                toks.push(self.byte_table[byte as usize] as usize);
            }
            loop {
                let mut best = f32::NEG_INFINITY;
                let mut at = usize::MAX;
                let mut id = 0usize;
                for k in 0..toks.len().saturating_sub(1) {
                    joined.clear();
                    joined.extend_from_slice(self.token_bytes(toks[k]));
                    joined.extend_from_slice(self.token_bytes(toks[k + 1]));
                    if let Some(j) = self.lookup(&joined) {
                        if self.scores[j] > best {
                            best = self.scores[j];
                            at = k;
                            id = j;
                        }
                    }
                }
                if at == usize::MAX {
                    break;
                }
                toks[at] = id;
                toks.remove(at + 1);
            }
            out.extend_from_slice(&toks);
        }
    }

    fn encode_legacy(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        if bos {
            out.push(BOS);
        }

        // Sentencepiece was trained with add_dummy_prefix, so every piece of
        // text it ever saw began with a space. Omitting it here means the first
        // word tokenises differently than it did in training -- "Once" and
        // " Once" are unrelated ids -- and the model produces noticeably worse
        // continuations for no visible reason.
        if !text.is_empty() {
            if let Some(id) = self.lookup(b" ") {
                out.push(id);
            }
        }

        let mut scratch = [0u8; 4];
        for ch in text.chars() {
            let s = ch.encode_utf8(&mut scratch);
            match self.lookup(s.as_bytes()) {
                Some(id) => out.push(id),
                None => {
                    for b in s.as_bytes() {
                        out.push(*b as usize + BYTE_FALLBACK_BASE);
                    }
                }
            }
        }

        let mut joined: Vec<u8> = Vec::new();
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0usize;
            let mut best_at = usize::MAX;

            for i in 0..out.len().saturating_sub(1) {
                joined.clear();
                joined.extend_from_slice(self.token_bytes(out[i]));
                joined.extend_from_slice(self.token_bytes(out[i + 1]));
                if let Some(id) = self.lookup(&joined) {
                    if self.scores[id] > best_score {
                        best_score = self.scores[id];
                        best_id = id;
                        best_at = i;
                    }
                }
            }

            if best_at == usize::MAX {
                break;
            }
            out[best_at] = best_id;
            out.remove(best_at + 1);
        }

        if eos {
            out.push(EOS);
        }
        out
    }

    /// Append the printable form of `token` to `out`.
    ///
    /// `prev` is needed for one reason: the dummy prefix above means the first
    /// token after BOS carries a leading space that was never really there.
    pub fn append_piece(&self, prev: usize, token: usize, out: &mut Vec<u8>) {
        if self.v2 {
            // v2 stores real bytes: the byte-level mapping was undone at
            // conversion time, and there is no dummy prefix to strip.
            out.extend_from_slice(self.token_bytes(token));
            return;
        }
        let mut p = self.token_bytes(token);
        if prev == BOS && p.first() == Some(&b' ') {
            p = &p[1..];
        }
        // Byte-fallback pieces are stored as the literal text "<0x0A>", not as
        // the byte they stand for. Printing them raw is a classic way to end up
        // with escape sequences all over the output.
        if p.len() == 6 && &p[..3] == b"<0x" && p[5] == b'>' {
            if let (Some(hi), Some(lo)) = (hex(p[3]), hex(p[4])) {
                out.push(hi * 16 + lo);
                return;
            }
        }
        out.extend_from_slice(p);
    }
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Split text before BPE, the way ByteLevel(use_regex) does.
///
/// The reference splits on:
///     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
/// Merges never cross these boundaries, so an approximation here changes the
/// tokenisation of ordinary sentences even when every merge rule is correct.
/// Hand-rolled rather than by regex because there is no regex engine, and
/// because the Python verifier implements exactly this and agrees with the
/// reference tokenizer on every case tested.
fn pretokenize(text: &str, individual_digits: bool, out: &mut Vec<(usize, usize)>) {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let end = text.len();
    let byte_at = |i: usize| if i < n { cs[i].0 } else { end };

    let mut i = 0usize;
    while i < n {
        let start = cs[i].0;

        let mut matched = false;
        for c in CONTRACTIONS {
            if text[start..].starts_with(c) {
                out.push((start, start + c.len()));
                i += c.chars().count();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // A single leading space belongs to the word that follows it, which is
        // how " the" becomes one token rather than two.
        let lead = usize::from(cs[i].1 == ' ' && i + 1 < n && !cs[i + 1].1.is_whitespace());
        let mut j = i + lead;

        if j < n && cs[j].1.is_alphabetic() {
            while j < n && cs[j].1.is_alphabetic() {
                j += 1;
            }
        } else if j < n && cs[j].1.is_ascii_digit() {
            j += 1;
            if !individual_digits {
                while j < n && cs[j].1.is_ascii_digit() {
                    j += 1;
                }
            }
        } else if j < n && !cs[j].1.is_whitespace() {
            while j < n && !cs[j].1.is_whitespace() && !cs[j].1.is_alphanumeric() {
                j += 1;
            }
        } else {
            // A run of whitespace. The final space is left for the next word
            // unless the run ends the string.
            while j < n && cs[j].1.is_whitespace() {
                j += 1;
            }
            if j < n {
                j -= 1;
            }
        }

        if j <= i {
            j = i + 1;
        }
        out.push((start, byte_at(j)));
        i = j;
    }
}
