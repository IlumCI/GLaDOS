//! From what the model wrote to what a voice can say.
//!
//! **Never read the markup.** A model writes Markdown -- asterisks, pound
//! signs, backticks, links -- and a voice that reads `asterisk asterisk
//! important asterisk asterisk` is broken in the most audible way available.
//! So the normaliser's first job is subtraction: formatting goes, a link
//! keeps its words and loses its address, a code span is said as "some code"
//! rather than spelled.
//!
//! Its second is that every token that survives is something a lexicon can
//! answer: numbers become words, a short all-capitals word is spelled, and
//! punctuation becomes a pause of a stated length rather than a sound.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Word(String),
    /// Spell this one letter by letter.
    Letters(String),
    /// Silence, in milliseconds.
    Pause(u32),
}

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// A whole number in words, up to the billions -- past which a spoken number
/// stops being information, so it is read digit by digit instead.
pub fn number_words(n: u64, out: &mut Vec<Token>) {
    fn under_1000(n: u64, out: &mut Vec<Token>) {
        let h = n / 100;
        let r = n % 100;
        if h > 0 {
            out.push(Token::Word(String::from(ONES[h as usize])));
            out.push(Token::Word(String::from("hundred")));
        }
        if r >= 20 {
            out.push(Token::Word(String::from(TENS[(r / 10) as usize])));
            if r % 10 > 0 {
                out.push(Token::Word(String::from(ONES[(r % 10) as usize])));
            }
        } else if r > 0 || h == 0 {
            out.push(Token::Word(String::from(ONES[r as usize])));
        }
    }
    if n >= 1_000_000_000_000 {
        for d in alloc::format!("{}", n).chars() {
            out.push(Token::Word(String::from(ONES[d as usize - '0' as usize])));
        }
        return;
    }
    let mut parts = [(1_000_000_000u64, "billion"), (1_000_000, "million"), (1_000, "thousand")].iter();
    let mut rest = n;
    let mut said = false;
    for &(scale, name) in &mut parts {
        if rest >= scale {
            under_1000(rest / scale, out);
            out.push(Token::Word(String::from(name)));
            rest %= scale;
            said = true;
        }
    }
    if rest > 0 || !said {
        under_1000(rest, out);
    }
}

/// Strip Markdown and URLs, leaving the words.
pub fn strip(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // Fenced or inline code: said as "some code", never spelled.
        if c == '`' {
            let fence = b[i..].iter().take_while(|&&x| x == '`').count();
            let mut j = i + fence;
            while j < b.len() && !(b[j] == '`' && b[j..].iter().take_while(|&&x| x == '`').count() >= fence) {
                j += 1;
            }
            out.push_str(" some code ");
            i = (j + fence).min(b.len());
            continue;
        }
        // [text](url) keeps the text.
        if c == '[' {
            if let Some(close) = b[i..].iter().position(|&x| x == ']') {
                let close = i + close;
                if b.get(close + 1) == Some(&'(') {
                    if let Some(end) = b[close..].iter().position(|&x| x == ')') {
                        out.extend(b[i + 1..close].iter());
                        i = close + end + 1;
                        continue;
                    }
                }
            }
        }
        // A URL becomes "a link".
        if b[i..].starts_with(&['h', 't', 't', 'p']) {
            let mut j = i;
            while j < b.len() && !b[j].is_whitespace() {
                j += 1;
            }
            let word: String = b[i..j].iter().collect();
            if word.starts_with("http://") || word.starts_with("https://") {
                out.push_str(" a link ");
                i = j;
                continue;
            }
        }
        match c {
            '*' | '_' | '#' | '>' | '|' | '~' | '\\' => out.push(' '),
            '-' if i == 0 || b[i - 1] == '\n' => out.push(' '), // list bullets
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// What to say, as tokens.
pub fn tokens(text: &str) -> Vec<Token> {
    let s = strip(text);
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let digits: String = chars[i..j].iter().collect();
            number_words(digits.parse().unwrap_or(0), &mut out);
            // 3.5 is "three point five", digit by digit after the point.
            if j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
                out.push(Token::Word(String::from("point")));
                j += 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    out.push(Token::Word(String::from(ONES[chars[j] as usize - '0' as usize])));
                    j += 1;
                }
            }
            if chars.get(j) == Some(&'%') {
                out.push(Token::Word(String::from("percent")));
                j += 1;
            }
            i = j;
            continue;
        }
        if c.is_alphabetic() || c == '\'' {
            let mut j = i;
            while j < chars.len() && (chars[j].is_alphabetic() || chars[j] == '\'') {
                j += 1;
            }
            let w: String = chars[i..j].iter().collect();
            let w = w.trim_matches('\'');
            if !w.is_empty() {
                let caps = w.chars().all(|c| c.is_ascii_uppercase());
                // A short all-capitals word is an initialism: `OS`, `AI`,
                // `CPU`. Spelled. A longer one is shouting, and is a word.
                if caps && w.len() >= 2 && w.len() <= 4 && !matches!(w, "OK" | "GLADOS") {
                    out.push(Token::Letters(w.to_lowercase()));
                } else {
                    out.push(Token::Word(w.to_lowercase()));
                }
            }
            i = j;
            continue;
        }
        match c {
            ',' | ';' | ':' => out.push(Token::Pause(180)),
            '.' | '!' | '?' => out.push(Token::Pause(350)),
            '\n' => out.push(Token::Pause(250)),
            '&' => out.push(Token::Word(String::from("and"))),
            '+' => out.push(Token::Word(String::from("plus"))),
            '=' => out.push(Token::Word(String::from("equals"))),
            '@' => out.push(Token::Word(String::from("at"))),
            _ => {}
        }
        i += 1;
    }
    // No doubled pauses, and none leading.
    let mut clean: Vec<Token> = Vec::with_capacity(out.len());
    for t in out {
        match (&t, clean.last_mut()) {
            (Token::Pause(_), None) => {}
            (Token::Pause(a), Some(Token::Pause(b))) => *b = (*b).max(*a),
            _ => clean.push(t),
        }
    }
    clean
}

/// Split into sentences, so a long reply starts speaking after its first.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | '\n')
            && chars.get(i + 1).map_or(true, |n| n.is_whitespace())
            // Not the point in 3.5.
            && !(c == '.' && i > 0 && chars[i - 1].is_ascii_digit() && chars.get(i + 1).map_or(false, |n| n.is_ascii_digit()));
        if end && cur.trim().len() > 1 {
            out.push(core::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let words = |s: &str| -> Vec<String> {
        tokens(s)
            .into_iter()
            .filter_map(|t| match t {
                Token::Word(w) | Token::Letters(w) => Some(w),
                Token::Pause(_) => None,
            })
            .collect()
    };
    let mut n = Vec::new();
    number_words(1_234_567, &mut n);
    let said: Vec<String> = n
        .iter()
        .filter_map(|t| if let Token::Word(w) = t { Some(w.clone()) } else { None })
        .collect();
    alloc::vec![
        ("markdown emphasis is not read", words("**really** _important_") == ["really", "important"]),
        ("a heading's pound signs are not read", words("## Status") == ["status"]),
        ("a link keeps its words, not its address", words("see [the docs](https://x.io/a)") == ["see", "the", "docs"]),
        ("a bare URL is 'a link'", words("go to https://example.com now") == ["go", "to", "a", "link", "now"]),
        ("code is 'some code', never spelled", words("run `cargo build` first") == ["run", "some", "code", "first"]),
        (
            "numbers become words, past a thousand",
            said == ["one", "million", "two", "hundred", "thirty", "four", "thousand", "five", "hundred", "sixty", "seven"]
        ),
        ("a decimal is read with 'point'", words("3.5 GHz") == ["three", "point", "five", "ghz"]),
        ("a short all-capitals word is spelled", matches!(tokens("the OS").last(), Some(Token::Letters(w)) if w == "os")),
        ("punctuation is a pause, never a word", tokens("Yes. No").iter().filter(|t| matches!(t, Token::Pause(_))).count() == 1),
        ("sentences split at their ends and not inside 3.5", sentences("It is 3.5 GHz. Done!").len() == 2),
    ]
}
