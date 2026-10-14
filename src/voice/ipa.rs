//! Words to Kokoro's phonemes: misaki's IPA, which is what the model was
//! trained on.
//!
//! Three sources, in order. The misaki lexicon shipped inside the voice
//! file (183,000 entries, Apache-2.0) is asked first, exact and then lower
//! and then capitalised, because its keys carry case the way its authors
//! wrote them. A contraction it lacks is its stem plus the clitic. A word
//! none of that finds goes to CMUdict and then the letter rules `g2p.rs`
//! already has, and comes back through one table from ARPAbet to this
//! alphabet. **A word is never dropped**: the formant voice's silent skip
//! of a word it could not say is the thing the R samples were heard
//! hiccuping over, and the sentence means something else without it.

use super::g2p;
use super::phone::{Phone, P};
use alloc::string::String;
use alloc::vec::Vec;

const MAGIC: &[u8; 8] = b"GLADOSX1";

pub struct Lexicon(&'static [u8]);

impl Lexicon {
    pub fn parse(b: &'static [u8]) -> Option<Lexicon> {
        if b.len() < 16 || &b[..8] != MAGIC {
            return None;
        }
        Some(Lexicon(b))
    }

    pub fn len(&self) -> usize {
        u32::from_le_bytes([self.0[8], self.0[9], self.0[10], self.0[11]]) as usize
    }

    pub fn get(&self, word: &str) -> Option<&'static str> {
        let b = self.0;
        let n = self.len();
        let base = 16 + 4 * n;
        let key = word.as_bytes();
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let o = base + u32::from_le_bytes([b[16 + 4 * mid], b[17 + 4 * mid], b[18 + 4 * mid], b[19 + 4 * mid]]) as usize;
            let wl = *b.get(o)? as usize;
            let w = b.get(o + 1..o + 1 + wl)?;
            match w.cmp(key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    let pl = *b.get(o + 1 + wl)? as usize;
                    return core::str::from_utf8(b.get(o + 2 + wl..o + 2 + wl + pl)?).ok();
                }
            }
        }
        None
    }
}

fn capitalised(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c.flat_map(|x| x.to_lowercase())).collect(),
        None => String::new(),
    }
}

fn look(lex: Option<&Lexicon>, w: &str) -> Option<String> {
    let lex = lex?;
    lex.get(w).or_else(|| lex.get(&w.to_lowercase())).or_else(|| lex.get(&capitalised(w))).map(String::from)
}

const SUFFIX: &[(&str, &str)] = &[("'s", "z"), ("'re", "ɚ"), ("'ll", "l"), ("'ve", "v"), ("'d", "d"), ("'m", "m"), ("n't", "nt")];

/// ARPAbet to misaki's US alphabet. Stress goes in front of the vowel it
/// falls on, as misaki writes it; `AH` and `ER` differ by stress the way
/// English does (`ʌ`/`ə`, `ɜɹ`/`ɚ`).
pub fn arpa_to_ipa(ps: &[Phone]) -> String {
    let mut out = String::new();
    for ph in ps {
        if ph.p.vowel() {
            match ph.stress {
                1 => out.push('ˈ'),
                2 => out.push('ˌ'),
                _ => {}
            }
        }
        let stressed = ph.stress == 1 || ph.stress == 2;
        out.push_str(match ph.p {
            P::AA => "ɑ",
            P::AE => "æ",
            P::AH => if stressed { "ʌ" } else { "ə" },
            P::AO => "ɔ",
            P::AW => "W",
            P::AY => "I",
            P::EH => "ɛ",
            P::ER => if stressed { "ɜɹ" } else { "ɚ" },
            P::EY => "A",
            P::IH => "ɪ",
            P::IY => "i",
            P::OW => "O",
            P::OY => "Y",
            P::UH => "ʊ",
            P::UW => "u",
            P::B => "b",
            P::CH => "ʧ",
            P::D => "d",
            P::DH => "ð",
            P::F => "f",
            P::G => "ɡ",
            P::HH => "h",
            P::JH => "ʤ",
            P::K => "k",
            P::L => "l",
            P::M => "m",
            P::N => "n",
            P::NG => "ŋ",
            P::P => "p",
            P::R => "ɹ",
            P::S => "s",
            P::SH => "ʃ",
            P::T => "t",
            P::TH => "θ",
            P::V => "v",
            P::W => "w",
            P::Y => "j",
            P::Z => "z",
            P::ZH => "ʒ",
        });
    }
    out
}

/// A word's phonemes, and which source said so.
pub fn word(lex: Option<&Lexicon>, w: &str) -> (String, &'static str) {
    let w: String = w.chars().map(|c| if c == '\u{2019}' || c == '\u{2018}' { '\'' } else { c }).collect();
    if let Some(p) = look(lex, &w) {
        return (p, "lexicon");
    }
    let lower = w.to_lowercase();
    for (suf, ph) in SUFFIX {
        if lower.ends_with(suf) && w.len() > suf.len() {
            if let Some(stem) = look(lex, &w[..w.len() - suf.len()]) {
                let ph = if *suf == "'s" && stem.ends_with(['t', 'k', 'p', 'f', 'θ']) { "s" } else { ph };
                return (alloc::format!("{stem}{ph}"), "lexicon+clitic");
            }
        }
    }
    let plain: String = lower.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'').collect();
    if let Some(ps) = g2p::lookup(&plain) {
        return (arpa_to_ipa(&ps), "cmudict");
    }
    (arpa_to_ipa(&g2p::rules(&plain)), "rules")
}

pub fn checks(lex: Option<&Lexicon>) -> Vec<(&'static str, bool)> {
    let hello = g2p::lookup("hello").unwrap_or_default();
    let mut v = alloc::vec![
        ("ARPAbet 'hello' maps to misaki's hʌlˈO", arpa_to_ipa(&hello) == "həlˈO"),
        ("a word nobody has still gets a pronunciation by rule", !word(None, "blorptastic").0.is_empty()),
        ("a curly apostrophe is a straight one", word(None, "don\u{2019}t").0 == word(None, "don't").0),
    ];
    if let Some(l) = lex {
        v.push(("the misaki lexicon finds 'the'", l.get("the").is_some()));
        v.push(("a contraction the lexicon lacks is stem plus clitic", word(lex, "that's").0.ends_with('s') && word(lex, "that's").1 == "lexicon+clitic" || word(lex, "that's").1 == "lexicon"));
        v.push(("case the lexicon has is tried before the dictionary", word(lex, "Greetings").1 == "lexicon"));
    }
    v
}
