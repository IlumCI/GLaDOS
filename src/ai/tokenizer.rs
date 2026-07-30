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

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    /// Token ids ordered by their byte string, so a merge candidate can be
    /// looked up by binary search instead of scanning the vocabulary.
    sorted: Vec<u32>,
    pub max_token_length: usize,
}

impl Tokenizer {
    pub fn from_bytes(data: &[u8], vocab_size: usize) -> Option<Self> {
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

        Some(Self { vocab, scores, sorted, max_token_length })
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
