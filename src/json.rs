//! JSON, enough of it.
//!
//! Written because Discord speaks it and nothing else here does. Deliberately
//! small: a parser, a handful of accessors, and a writer that escapes properly.
//!
//! ### Numbers are kept as text
//!
//! There is no float parser in this kernel and writing one to read a heartbeat
//! interval would be absurd. A number is stored as the token that produced it
//! and converted on demand, which costs nothing for the integers that are
//! actually read (opcodes, sequence numbers, intervals) and cannot lose
//! precision on the ones that are not.
//!
//! That matters more than it sounds. Discord's snowflake IDs are 64-bit and it
//! sends them as *strings* precisely because JSON numbers are doubles in most
//! parsers -- a parser that helpfully converted them to f64 would silently
//! corrupt every id above 2^53. Keeping the text sidesteps the whole question.
//!
//! ### Objects are a Vec, not a map
//!
//! Gateway payloads have a handful of keys. A linear scan over eight entries
//! beats hashing them, keeps insertion order for the writer, and needs no map
//! implementation.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The literal token, unparsed. See the module note.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(src: &str) -> Option<Json> {
        let b = src.as_bytes();
        let mut p = Parser { b, i: 0, depth: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        // Trailing garbage means the document was not what it claimed to be,
        // and quietly accepting a prefix is how a truncated frame becomes a
        // plausible-looking message.
        if p.i != b.len() {
            return None;
        }
        Some(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(es) => es.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `a.b.c` in one call, because gateway payloads nest and the alternative
    /// is a staircase of `?`.
    pub fn path(&self, dotted: &str) -> Option<&Json> {
        let mut cur = self;
        for part in dotted.split('.') {
            cur = cur.get(part)?;
        }
        Some(cur)
    }

    pub fn idx(&self, i: usize) -> Option<&Json> {
        match self {
            Json::Arr(v) => v.get(i),
            _ => None,
        }
    }

    pub fn items(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        let t = match self {
            Json::Num(t) => t.as_str(),
            // A snowflake arrives as a string; asking for it as a number is
            // the natural thing to write and should not fail.
            Json::Str(s) => s.as_str(),
            _ => return None,
        };
        let (neg, digits) = match t.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, t),
        };
        // Stop at a decimal point or exponent rather than refusing: an interval
        // of 41250.0 is an integer to every caller here.
        let digits = digits.split(['.', 'e', 'E']).next()?;
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let mut n: i64 = 0;
        for c in digits.bytes() {
            n = n.checked_mul(10)?.checked_add((c - b'0') as i64)?;
        }
        Some(if neg { -n } else { n })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

/// Recursion limit.
///
/// The parser descends for arrays and objects, and a hostile or corrupt frame
/// of ten thousand open brackets would otherwise walk off a 64 KiB task stack.
/// Discord nests perhaps six deep.
const MAX_DEPTH: usize = 32;

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Json> {
        if self.i >= self.b.len() {
            return None;
        }
        match self.b[self.i] {
            b'n' => self.eat("null").then_some(Json::Null),
            b't' => self.eat("true").then_some(Json::Bool(true)),
            b'f' => self.eat("false").then_some(Json::Bool(false)),
            b'"' => self.string().map(Json::Str),
            b'[' => self.array(),
            b'{' => self.object(),
            _ => self.number(),
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        if self.i < self.b.len() && self.b[self.i] == b'-' {
            self.i += 1;
        }
        let digits = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        {
            self.i += 1;
        }
        if self.i == digits {
            return None;
        }
        let t = core::str::from_utf8(&self.b[start..self.i]).ok()?;
        Some(Json::Num(String::from(t)))
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            if self.i >= self.b.len() {
                return None;
            }
            match self.b[self.i] {
                b'"' => {
                    self.i += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.i += 1;
                    if self.i >= self.b.len() {
                        return None;
                    }
                    let c = self.b[self.i];
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // Surrogate pair. Emoji are the common case in a
                            // chat client, and a lone high surrogate is not a
                            // character -- decoding it as one produces a
                            // replacement glyph where a face should be.
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return None;
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                0x1_0000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(ch)?);
                        }
                        _ => return None,
                    }
                }
                _ => {
                    // Copy the UTF-8 sequence whole rather than byte by byte,
                    // so a multi-byte character survives.
                    let s = core::str::from_utf8(&self.b[self.i..]).ok()?;
                    let ch = s.chars().next()?;
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        if self.i + 4 > self.b.len() {
            return None;
        }
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.b[self.i];
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return None,
            };
            v = v * 16 + d as u32;
            self.i += 1;
        }
        Some(v)
    }

    fn array(&mut self) -> Option<Json> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return None;
        }
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            self.depth -= 1;
            return Some(Json::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i)? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    self.depth -= 1;
                    return Some(Json::Arr(out));
                }
                _ => return None,
            }
        }
    }

    fn object(&mut self) -> Option<Json> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return None;
        }
        self.i += 1;
        let mut out: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            self.depth -= 1;
            return Some(Json::Obj(out));
        }
        loop {
            self.ws();
            if *self.b.get(self.i)? != b'"' {
                return None;
            }
            let k = self.string()?;
            self.ws();
            if *self.b.get(self.i)? != b':' {
                return None;
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            out.push((k, v));
            self.ws();
            match self.b.get(self.i)? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    self.depth -= 1;
                    return Some(Json::Obj(out));
                }
                _ => return None,
            }
        }
    }
}

/// Append `s` as a quoted, escaped JSON string.
///
/// The control-character rule is the one people get wrong: anything below 0x20
/// must be escaped, not just the six with short forms. An unescaped newline in
/// a message body is a payload the gateway rejects, and the rejection arrives
/// as a close frame with no explanation of which field was at fault.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let d = ((c as u32) >> shift) & 0xF;
                    out.push(char::from_digit(d, 16).unwrap_or('0'));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn selftest() -> bool {
    let mut ok = true;
    let mut check = |name: &str, pass: bool| {
        if !pass {
            crate::kprintln!("  FAIL  json {}", name);
            ok = false;
        }
    };

    let v = Json::parse(r#"{"op":10,"d":{"heartbeat_interval":41250},"s":null,"t":"READY"}"#);
    let v = match v {
        Some(v) => v,
        None => {
            crate::kprintln!("  FAIL  json parse returned nothing");
            return false;
        }
    };
    check("op", v.get("op").and_then(|x| x.as_i64()) == Some(10));
    check(
        "nested",
        v.path("d.heartbeat_interval").and_then(|x| x.as_i64()) == Some(41250),
    );
    check("null", v.get("s").map(|x| x.is_null()) == Some(true));
    check("string", v.get("t").and_then(|x| x.as_str()) == Some("READY"));
    check("absent", v.get("nope").is_none());

    // A snowflake is a string and is 64-bit. A parser that turned it into a
    // double would return 1071098070310551552 here, off by 8.
    let s = Json::parse(r#"{"id":"1071098070310551555"}"#).unwrap_or(Json::Null);
    check(
        "snowflake",
        s.get("id").and_then(|x| x.as_i64()) == Some(1071098070310551555),
    );

    let e = Json::parse(r#"["a\nb","A","😀","x\"y"]"#).unwrap_or(Json::Null);
    check("escape newline", e.idx(0).and_then(|x| x.as_str()) == Some("a\nb"));
    check("escape \\u", e.idx(1).and_then(|x| x.as_str()) == Some("A"));
    check("surrogate pair", e.idx(2).and_then(|x| x.as_str()) == Some("\u{1F600}"));
    check("escape quote", e.idx(3).and_then(|x| x.as_str()) == Some("x\"y"));

    check("trailing garbage rejected", Json::parse("{} x").is_none());
    check("unterminated rejected", Json::parse(r#"{"a":"#).is_none());
    check("lone surrogate rejected", Json::parse(r#""\ud83d""#).is_none());

    let mut w = String::new();
    write_str(&mut w, "he said \"hi\"\nand\tleft");
    check(
        "writer escapes",
        w == r#""he said \"hi\"\nand\tleft""#,
    );
    // Round-trip, which is the property that actually matters.
    check(
        "writer round-trips",
        Json::parse(&w).and_then(|j| j.as_str().map(String::from))
            == Some(String::from("he said \"hi\"\nand\tleft")),
    );

    let deep = {
        let mut s = String::new();
        for _ in 0..64 {
            s.push('[');
        }
        s
    };
    check("depth bounded", Json::parse(&deep).is_none());

    ok
}
