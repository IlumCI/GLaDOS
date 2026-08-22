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
/// Unused directly, but it is what `BYTE_FALLBACK_BASE` counts from.
#[allow(dead_code)]
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
/// Which pre-tokenizer regex the checkpoint was trained with. Clear means the
/// GPT-2 pattern; `FLAG_SPLIT_CL100K` means the cl100k one Qwen3 spells out as
/// a Split; both set means the Qwen3.5 variant, which adds `\p{M}` to the
/// letter run and bars it from punctuation runs.
pub const FLAG_SPLIT_CL100K: u32 = 1 << 2;
pub const FLAG_SPLIT_CL100KM: u32 = 1 << 3;

/// `\p{M}` -- general categories Mn, Mc and Me -- as a sorted table of
/// inclusive codepoint ranges. Rust's `core` has no per-character general
/// category, so this carries the one class the marks pre-tokenizer needs.
/// Generated from Unicode 15.1 character data (310 ranges); regenerate from
/// `tools/tokenizer.py`'s ground truth if Unicode ever matters here.
static MARK_RANGES: [(u32, u32); 310] = [
    (0x300, 0x36f), (0x483, 0x489), (0x591, 0x5bd), (0x5bf, 0x5bf),
    (0x5c1, 0x5c2), (0x5c4, 0x5c5), (0x5c7, 0x5c7), (0x610, 0x61a),
    (0x64b, 0x65f), (0x670, 0x670), (0x6d6, 0x6dc), (0x6df, 0x6e4),
    (0x6e7, 0x6e8), (0x6ea, 0x6ed), (0x711, 0x711), (0x730, 0x74a),
    (0x7a6, 0x7b0), (0x7eb, 0x7f3), (0x7fd, 0x7fd), (0x816, 0x819),
    (0x81b, 0x823), (0x825, 0x827), (0x829, 0x82d), (0x859, 0x85b),
    (0x898, 0x89f), (0x8ca, 0x8e1), (0x8e3, 0x903), (0x93a, 0x93c),
    (0x93e, 0x94f), (0x951, 0x957), (0x962, 0x963), (0x981, 0x983),
    (0x9bc, 0x9bc), (0x9be, 0x9c4), (0x9c7, 0x9c8), (0x9cb, 0x9cd),
    (0x9d7, 0x9d7), (0x9e2, 0x9e3), (0x9fe, 0x9fe), (0xa01, 0xa03),
    (0xa3c, 0xa3c), (0xa3e, 0xa42), (0xa47, 0xa48), (0xa4b, 0xa4d),
    (0xa51, 0xa51), (0xa70, 0xa71), (0xa75, 0xa75), (0xa81, 0xa83),
    (0xabc, 0xabc), (0xabe, 0xac5), (0xac7, 0xac9), (0xacb, 0xacd),
    (0xae2, 0xae3), (0xafa, 0xaff), (0xb01, 0xb03), (0xb3c, 0xb3c),
    (0xb3e, 0xb44), (0xb47, 0xb48), (0xb4b, 0xb4d), (0xb55, 0xb57),
    (0xb62, 0xb63), (0xb82, 0xb82), (0xbbe, 0xbc2), (0xbc6, 0xbc8),
    (0xbca, 0xbcd), (0xbd7, 0xbd7), (0xc00, 0xc04), (0xc3c, 0xc3c),
    (0xc3e, 0xc44), (0xc46, 0xc48), (0xc4a, 0xc4d), (0xc55, 0xc56),
    (0xc62, 0xc63), (0xc81, 0xc83), (0xcbc, 0xcbc), (0xcbe, 0xcc4),
    (0xcc6, 0xcc8), (0xcca, 0xccd), (0xcd5, 0xcd6), (0xce2, 0xce3),
    (0xcf3, 0xcf3), (0xd00, 0xd03), (0xd3b, 0xd3c), (0xd3e, 0xd44),
    (0xd46, 0xd48), (0xd4a, 0xd4d), (0xd57, 0xd57), (0xd62, 0xd63),
    (0xd81, 0xd83), (0xdca, 0xdca), (0xdcf, 0xdd4), (0xdd6, 0xdd6),
    (0xdd8, 0xddf), (0xdf2, 0xdf3), (0xe31, 0xe31), (0xe34, 0xe3a),
    (0xe47, 0xe4e), (0xeb1, 0xeb1), (0xeb4, 0xebc), (0xec8, 0xece),
    (0xf18, 0xf19), (0xf35, 0xf35), (0xf37, 0xf37), (0xf39, 0xf39),
    (0xf3e, 0xf3f), (0xf71, 0xf84), (0xf86, 0xf87), (0xf8d, 0xf97),
    (0xf99, 0xfbc), (0xfc6, 0xfc6), (0x102b, 0x103e), (0x1056, 0x1059),
    (0x105e, 0x1060), (0x1062, 0x1064), (0x1067, 0x106d), (0x1071, 0x1074),
    (0x1082, 0x108d), (0x108f, 0x108f), (0x109a, 0x109d), (0x135d, 0x135f),
    (0x1712, 0x1715), (0x1732, 0x1734), (0x1752, 0x1753), (0x1772, 0x1773),
    (0x17b4, 0x17d3), (0x17dd, 0x17dd), (0x180b, 0x180d), (0x180f, 0x180f),
    (0x1885, 0x1886), (0x18a9, 0x18a9), (0x1920, 0x192b), (0x1930, 0x193b),
    (0x1a17, 0x1a1b), (0x1a55, 0x1a5e), (0x1a60, 0x1a7c), (0x1a7f, 0x1a7f),
    (0x1ab0, 0x1ace), (0x1b00, 0x1b04), (0x1b34, 0x1b44), (0x1b6b, 0x1b73),
    (0x1b80, 0x1b82), (0x1ba1, 0x1bad), (0x1be6, 0x1bf3), (0x1c24, 0x1c37),
    (0x1cd0, 0x1cd2), (0x1cd4, 0x1ce8), (0x1ced, 0x1ced), (0x1cf4, 0x1cf4),
    (0x1cf7, 0x1cf9), (0x1dc0, 0x1dff), (0x20d0, 0x20f0), (0x2cef, 0x2cf1),
    (0x2d7f, 0x2d7f), (0x2de0, 0x2dff), (0x302a, 0x302f), (0x3099, 0x309a),
    (0xa66f, 0xa672), (0xa674, 0xa67d), (0xa69e, 0xa69f), (0xa6f0, 0xa6f1),
    (0xa802, 0xa802), (0xa806, 0xa806), (0xa80b, 0xa80b), (0xa823, 0xa827),
    (0xa82c, 0xa82c), (0xa880, 0xa881), (0xa8b4, 0xa8c5), (0xa8e0, 0xa8f1),
    (0xa8ff, 0xa8ff), (0xa926, 0xa92d), (0xa947, 0xa953), (0xa980, 0xa983),
    (0xa9b3, 0xa9c0), (0xa9e5, 0xa9e5), (0xaa29, 0xaa36), (0xaa43, 0xaa43),
    (0xaa4c, 0xaa4d), (0xaa7b, 0xaa7d), (0xaab0, 0xaab0), (0xaab2, 0xaab4),
    (0xaab7, 0xaab8), (0xaabe, 0xaabf), (0xaac1, 0xaac1), (0xaaeb, 0xaaef),
    (0xaaf5, 0xaaf6), (0xabe3, 0xabea), (0xabec, 0xabed), (0xfb1e, 0xfb1e),
    (0xfe00, 0xfe0f), (0xfe20, 0xfe2f), (0x101fd, 0x101fd), (0x102e0, 0x102e0),
    (0x10376, 0x1037a), (0x10a01, 0x10a03), (0x10a05, 0x10a06), (0x10a0c, 0x10a0f),
    (0x10a38, 0x10a3a), (0x10a3f, 0x10a3f), (0x10ae5, 0x10ae6), (0x10d24, 0x10d27),
    (0x10eab, 0x10eac), (0x10efd, 0x10eff), (0x10f46, 0x10f50), (0x10f82, 0x10f85),
    (0x11000, 0x11002), (0x11038, 0x11046), (0x11070, 0x11070), (0x11073, 0x11074),
    (0x1107f, 0x11082), (0x110b0, 0x110ba), (0x110c2, 0x110c2), (0x11100, 0x11102),
    (0x11127, 0x11134), (0x11145, 0x11146), (0x11173, 0x11173), (0x11180, 0x11182),
    (0x111b3, 0x111c0), (0x111c9, 0x111cc), (0x111ce, 0x111cf), (0x1122c, 0x11237),
    (0x1123e, 0x1123e), (0x11241, 0x11241), (0x112df, 0x112ea), (0x11300, 0x11303),
    (0x1133b, 0x1133c), (0x1133e, 0x11344), (0x11347, 0x11348), (0x1134b, 0x1134d),
    (0x11357, 0x11357), (0x11362, 0x11363), (0x11366, 0x1136c), (0x11370, 0x11374),
    (0x11435, 0x11446), (0x1145e, 0x1145e), (0x114b0, 0x114c3), (0x115af, 0x115b5),
    (0x115b8, 0x115c0), (0x115dc, 0x115dd), (0x11630, 0x11640), (0x116ab, 0x116b7),
    (0x1171d, 0x1172b), (0x1182c, 0x1183a), (0x11930, 0x11935), (0x11937, 0x11938),
    (0x1193b, 0x1193e), (0x11940, 0x11940), (0x11942, 0x11943), (0x119d1, 0x119d7),
    (0x119da, 0x119e0), (0x119e4, 0x119e4), (0x11a01, 0x11a0a), (0x11a33, 0x11a39),
    (0x11a3b, 0x11a3e), (0x11a47, 0x11a47), (0x11a51, 0x11a5b), (0x11a8a, 0x11a99),
    (0x11c2f, 0x11c36), (0x11c38, 0x11c3f), (0x11c92, 0x11ca7), (0x11ca9, 0x11cb6),
    (0x11d31, 0x11d36), (0x11d3a, 0x11d3a), (0x11d3c, 0x11d3d), (0x11d3f, 0x11d45),
    (0x11d47, 0x11d47), (0x11d8a, 0x11d8e), (0x11d90, 0x11d91), (0x11d93, 0x11d97),
    (0x11ef3, 0x11ef6), (0x11f00, 0x11f01), (0x11f03, 0x11f03), (0x11f34, 0x11f3a),
    (0x11f3e, 0x11f42), (0x13440, 0x13440), (0x13447, 0x13455), (0x16af0, 0x16af4),
    (0x16b30, 0x16b36), (0x16f4f, 0x16f4f), (0x16f51, 0x16f87), (0x16f8f, 0x16f92),
    (0x16fe4, 0x16fe4), (0x16ff0, 0x16ff1), (0x1bc9d, 0x1bc9e), (0x1cf00, 0x1cf2d),
    (0x1cf30, 0x1cf46), (0x1d165, 0x1d169), (0x1d16d, 0x1d172), (0x1d17b, 0x1d182),
    (0x1d185, 0x1d18b), (0x1d1aa, 0x1d1ad), (0x1d242, 0x1d244), (0x1da00, 0x1da36),
    (0x1da3b, 0x1da6c), (0x1da75, 0x1da75), (0x1da84, 0x1da84), (0x1da9b, 0x1da9f),
    (0x1daa1, 0x1daaf), (0x1e000, 0x1e006), (0x1e008, 0x1e018), (0x1e01b, 0x1e021),
    (0x1e023, 0x1e024), (0x1e026, 0x1e02a), (0x1e08f, 0x1e08f), (0x1e130, 0x1e136),
    (0x1e2ae, 0x1e2ae), (0x1e2ec, 0x1e2ef), (0x1e4ec, 0x1e4ef), (0x1e8d0, 0x1e8d6),
    (0x1e944, 0x1e94a), (0xe0100, 0xe01ef),
];

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
        if self.flags & (FLAG_SPLIT_CL100K | FLAG_SPLIT_CL100KM) != 0 {
            pretokenize_cl100k(
                text,
                self.flags & FLAG_SPLIT_CL100KM != 0,
                &mut spans,
            );
        } else {
            pretokenize(text, self.flags & FLAG_INDIVIDUAL_DIGITS != 0, &mut spans);
        }

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

/// `\p{M}` by binary search over `MARK_RANGES`.
fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                core::cmp::Ordering::Greater
            } else if cp > hi {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Split text before BPE, the way Qwen's explicit Split regex does.
///
///     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
///     | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
///
/// With `marks` set this is the Qwen3.5 variant, which differs in exactly two
/// clauses: the letter run admits `\p{M}` (combining marks), and punctuation
/// may no longer swallow them.
///
/// Four differences from the GPT-2 pattern above, and each one moves real
/// boundaries:
///
///   * a word may be led by *any* non-alphanumeric rather than only a space,
///     so `(x` is one piece where GPT-2 gives `(` then `x`;
///   * `\p{N}` takes one digit at a time, so Qwen3 gets digit splitting from
///     its regex and never sets `FLAG_INDIVIDUAL_DIGITS`;
///   * a punctuation run swallows the newlines after it;
///   * whitespace ending in a newline is its own piece, which is what keeps
///     `<|im_end|>\n<|im_start|>` aligned.
///
/// Alternation is ordered and first-match-wins, so the branches are in the
/// order the pattern lists them. The Python verifier implements this same scan
/// and agrees with the reference tokenizer over the training corpus; the one
/// place the two could still part company is the exact membership of
/// `is_alphabetic` and `is_numeric`, which differ slightly between Rust and
/// Python at the edges of Unicode -- and the mark table here is pinned to
/// Unicode 15.1 while Python's `unicodedata` moves with its release.
pub(super) fn pretokenize_cl100k(text: &str, marks: bool, out: &mut Vec<(usize, usize)>) {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let end = text.len();
    let byte_at = |i: usize| if i < n { cs[i].0 } else { end };
    let nl = |c: char| c == '\r' || c == '\n';
    let letter = |c: char| c.is_alphabetic() || (marks && is_combining_mark(c));
    let digit = |c: char| c.is_numeric();
    let word_lead = |c: char| !nl(c) && !letter(c) && !digit(c);
    let punct =
        |c: char| !c.is_whitespace() && !letter(c) && !digit(c);

    let mut i = 0usize;
    while i < n {
        let start = cs[i].0;
        let here = cs[i].1;

        // 1. Contractions, case-insensitively. All ASCII, so a byte length is
        //    also a character count.
        let rest = text.as_bytes();
        let mut hit = 0usize;
        for c in CONTRACTIONS {
            let b = c.as_bytes();
            if rest.len() >= start + b.len() && rest[start..start + b.len()].eq_ignore_ascii_case(b)
            {
                hit = b.len();
                break;
            }
        }
        if hit > 0 {
            out.push((start, start + hit));
            i += hit;
            continue;
        }

        // 2. An optional lead character, then letters. The regex backtracks:
        //    if the lead is taken but no run follows it is given back and the
        //    run is tried at i itself, which is how a leading combining mark
        //    ends up as a piece of its own under `marks`. In the plain pattern
        //    the lead class and the run class are disjoint, so the second
        //    attempt can only fire when marks are admitted.
        let mut matched = false;
        for take_lead in [true, false] {
            let j0 = i + usize::from(take_lead && word_lead(here));
            if j0 < n && letter(cs[j0].1) {
                let mut j = j0;
                while j < n && letter(cs[j].1) {
                    j += 1;
                }
                out.push((start, byte_at(j)));
                i = j;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // 3. One digit, alone.
        if digit(here) {
            out.push((start, byte_at(i + 1)));
            i += 1;
            continue;
        }

        // 4. An optional space, a punctuation run, then any newlines it drags
        //    along. Under `marks` a combining mark ends the run, because the
        //    Qwen3.5 clause excludes \p{M} from what punctuation may take.
        let mut j = i + usize::from(here == ' ');
        if j < n && punct(cs[j].1) {
            while j < n && punct(cs[j].1) {
                j += 1;
            }
            while j < n && nl(cs[j].1) {
                j += 1;
            }
            out.push((start, byte_at(j)));
            i = j;
            continue;
        }

        // 5/6/7. Whitespace.
        if here.is_whitespace() {
            let mut j = i;
            while j < n && cs[j].1.is_whitespace() {
                j += 1;
            }
            // `\s*[\r\n]+` is greedy on both halves, so it ends at the LAST
            // newline in the run rather than the first.
            let mut last_nl = None;
            for k in i..j {
                if nl(cs[k].1) {
                    last_nl = Some(k);
                }
            }
            if let Some(k) = last_nl {
                out.push((start, byte_at(k + 1)));
                i = k + 1;
                continue;
            }
            // `\s+(?!\S)` takes the whole run only at the end of the string.
            // Otherwise it gives back one character, and that last space goes
            // on to lead the next word.
            let stop = if j == n { j } else { (j - 1).max(i + 1) };
            out.push((start, byte_at(stop)));
            i = stop;
            continue;
        }

        out.push((start, byte_at(i + 1)));
        i += 1;
    }
}
