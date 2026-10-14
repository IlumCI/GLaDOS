//! Words to phones.
//!
//! The lexicon first -- CMUdict, 124,926 words, `tools/lexicon.py` -- and
//! rules only for a word it does not have. The rules are small on purpose:
//! a few dozen digraphs, the magic e, and one letter at a time otherwise.
//! They mispronounce plenty, and every word they are asked about is one the
//! lexicon could not answer, which is mostly names and invented words; a
//! large rule set would be a second, worse lexicon.

use super::phone::{Phone, P};
use alloc::vec::Vec;

static LEXICON: &[u8] = include_bytes!("../../assets/voice/lexicon.bin");

/// Words this machine says that a 1990s dictionary does not have.
const OVERRIDES: &[(&str, &[(P, u8)])] = &[
    ("glados", &[(P::G, 3), (P::L, 3), (P::AE, 1), (P::D, 3), (P::OW, 0), (P::S, 3)]),
    ("aiksi", &[(P::AY, 1), (P::K, 3), (P::S, 3), (P::IY, 0)]),
    ("qwen", &[(P::K, 3), (P::W, 3), (P::EH, 1), (P::N, 3)]),
];

/// Look a word up. `None` when the table does not have it.
pub fn lookup(word: &str) -> Option<Vec<Phone>> {
    for (w, ps) in OVERRIDES {
        if *w == word {
            return Some(ps.iter().map(|&(p, s)| Phone::new(p, s)).collect());
        }
    }
    let b = LEXICON;
    if b.len() < 16 || &b[..8] != b"GLADOSL1" {
        return None;
    }
    let n = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
    let base = 16 + 4 * n;
    let key = word.as_bytes();
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let off = u32::from_le_bytes(b[16 + 4 * mid..20 + 4 * mid].try_into().ok()?) as usize;
        let o = base + off;
        let wl = b[o] as usize;
        let w = &b[o + 1..o + 1 + wl];
        match w.cmp(key) {
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
            core::cmp::Ordering::Equal => {
                let pl = b[o + 1 + wl] as usize;
                return Some(b[o + 2 + wl..o + 2 + wl + pl].iter().map(|&c| Phone::from_code(c)).collect());
            }
        }
    }
    None
}

/// A letter's own name, for spelled initialisms.
pub fn letter(c: char) -> Vec<Phone> {
    let name = match c {
        'a' => "a", 'b' => "bee", 'c' => "see", 'd' => "dee", 'e' => "e", 'f' => "eff", 'g' => "gee",
        'h' => "aitch", 'i' => "i", 'j' => "jay", 'k' => "kay", 'l' => "el", 'm' => "em", 'n' => "en",
        'o' => "o", 'p' => "pee", 'q' => "cue", 'r' => "are", 's' => "ess", 't' => "tee", 'u' => "you",
        'v' => "vee", 'w' => "double-u", 'x' => "ex", 'y' => "why", 'z' => "zee",
        _ => return Vec::new(),
    };
    if name == "double-u" {
        let mut v = lookup("double").unwrap_or_default();
        v.extend(lookup("you").unwrap_or_default());
        return v;
    }
    lookup(name).unwrap_or_default()
}

/// Letter-to-sound rules, for a word the table does not have.
pub fn rules(word: &str) -> Vec<Phone> {
    let w: Vec<char> = word.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let mut out: Vec<Phone> = Vec::new();
    let mut stressed = false;
    let mut vowel = |p: P, out: &mut Vec<Phone>| {
        out.push(Phone::new(p, if stressed { 0 } else { 1 }));
        stressed = true;
    };
    let at = |i: usize| w.get(i).copied().unwrap_or(' ');
    let is_v = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u');
    let mut i = 0;
    while i < w.len() {
        let c = w[i];
        let two: [char; 2] = [c, at(i + 1)];
        // Magic e: a consonant then a final e lengthens the vowel before it.
        let magic = is_v(c) && !is_v(at(i + 1)) && at(i + 1) != ' ' && at(i + 2) == 'e' && i + 3 == w.len();
        let step = match two {
            ['t', 'h'] => { out.push(Phone::new(P::TH, 3)); 2 }
            ['s', 'h'] => { out.push(Phone::new(P::SH, 3)); 2 }
            ['c', 'h'] => { out.push(Phone::new(P::CH, 3)); 2 }
            ['p', 'h'] => { out.push(Phone::new(P::F, 3)); 2 }
            ['n', 'g'] => { out.push(Phone::new(P::NG, 3)); 2 }
            ['c', 'k'] => { out.push(Phone::new(P::K, 3)); 2 }
            ['q', 'u'] => { out.push(Phone::new(P::K, 3)); out.push(Phone::new(P::W, 3)); 2 }
            ['e', 'e'] | ['e', 'a'] => { vowel(P::IY, &mut out); 2 }
            ['o', 'o'] => { vowel(P::UW, &mut out); 2 }
            ['a', 'i'] | ['a', 'y'] => { vowel(P::EY, &mut out); 2 }
            ['o', 'a'] | ['o', 'w'] => { vowel(P::OW, &mut out); 2 }
            ['o', 'u'] => { vowel(P::AW, &mut out); 2 }
            ['o', 'i'] | ['o', 'y'] => { vowel(P::OY, &mut out); 2 }
            _ => 0,
        };
        if step > 0 {
            i += step;
            continue;
        }
        match c {
            'a' => vowel(if magic { P::EY } else { P::AE }, &mut out),
            'e' => {
                // A final e is silent.
                if i + 1 != w.len() || w.len() <= 2 {
                    vowel(if magic { P::IY } else { P::EH }, &mut out);
                }
            }
            'i' => vowel(if magic { P::AY } else { P::IH }, &mut out),
            'o' => vowel(if magic { P::OW } else { P::AA }, &mut out),
            'u' => vowel(if magic { P::UW } else { P::AH }, &mut out),
            'y' => {
                if i == 0 {
                    out.push(Phone::new(P::Y, 3));
                } else {
                    vowel(if i + 1 == w.len() { P::IY } else { P::IH }, &mut out);
                }
            }
            'c' => out.push(Phone::new(if matches!(at(i + 1), 'e' | 'i' | 'y') { P::S } else { P::K }, 3)),
            'g' => out.push(Phone::new(if matches!(at(i + 1), 'e' | 'i' | 'y') { P::JH } else { P::G }, 3)),
            'x' => {
                out.push(Phone::new(P::K, 3));
                out.push(Phone::new(P::S, 3));
            }
            'b' => out.push(Phone::new(P::B, 3)),
            'd' => out.push(Phone::new(P::D, 3)),
            'f' => out.push(Phone::new(P::F, 3)),
            'h' => out.push(Phone::new(P::HH, 3)),
            'j' => out.push(Phone::new(P::JH, 3)),
            'k' => out.push(Phone::new(P::K, 3)),
            'l' => out.push(Phone::new(P::L, 3)),
            'm' => out.push(Phone::new(P::M, 3)),
            'n' => out.push(Phone::new(P::N, 3)),
            'p' => out.push(Phone::new(P::P, 3)),
            'q' => out.push(Phone::new(P::K, 3)),
            'r' => out.push(Phone::new(P::R, 3)),
            's' => out.push(Phone::new(P::S, 3)),
            't' => out.push(Phone::new(P::T, 3)),
            'v' => out.push(Phone::new(P::V, 3)),
            'w' => out.push(Phone::new(P::W, 3)),
            'z' => out.push(Phone::new(P::Z, 3)),
            _ => {}
        }
        // A doubled consonant is one sound.
        if !is_v(c) && at(i + 1) == c {
            i += 1;
        }
        i += 1;
    }
    out
}

/// A word, by table then by rule.
pub fn word(w: &str) -> (Vec<Phone>, bool) {
    match lookup(w) {
        Some(p) => (p, true),
        None => (rules(w), false),
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let names = |ps: &[Phone]| -> Vec<&'static str> { ps.iter().map(|p| p.p.name()).collect() };
    let hello = lookup("hello").unwrap_or_default();
    let the = lookup("the").unwrap_or_default();
    alloc::vec![
        ("the compiled-in lexicon is CMUdict and finds 'hello'", names(&hello) == ["HH", "AH", "L", "OW"]),
        ("stress survives the packing", hello.get(3).map(|p| p.stress) == Some(1)),
        ("'the' is two phones", names(&the) == ["DH", "AH"]),
        ("a word not in the table is None, not a neighbour", lookup("zxqvb").is_none()),
        ("the machine knows its own name", lookup("glados").map(|p| p.len()) == Some(6)),
        ("rules give an unknown word something to say", !rules("blorptastic").is_empty()),
        ("magic e: 'bike' by rule is B AY K", names(&rules("bike")) == ["B", "AY", "K"]),
        ("a letter is said by its name", names(&letter('s')) == ["EH", "S"]),
    ]
}
