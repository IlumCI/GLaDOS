//! Plain language at the prompt.
//!
//! A line the shell did not recognise used to be Aiksi, and English therefore
//! answered `undefined variable 'list'` -- so talking to the machine needed a
//! prefix, `ask` or `agent`, and the prefix was the whole of the interface's
//! claim that the model lives here. Now a line that is not a command and is
//! not code is language, and the machine decides what to do with it.
//!
//! **What is code is decided by the parser, not by a guess.** A line that
//! lexes and parses is code and runs exactly as before -- `x = 3`, `fib(10)`.
//! The one exception is a lone identifier the interpreter does not know,
//! because `hello` parses as a variable read and nobody typing it meant one.
//! A line that does not parse is language.
//!
//! **What to do with language is decided by the router, and said.** The
//! council's agreement is the confidence signal the tree already measured --
//! 90.3% right when all three cores agree, 50% when they split -- so:
//!
//! - all three agree, or two agree and the line names a path or an applet:
//!   it is a task, and the agent takes it, read-only;
//! - the task needs a mutating applet: it is refused, offered, or run, by
//!   `intent trust readonly|ask|full`, and `readonly` is the default;
//! - otherwise it is conversation, and it is a chat turn.
//!
//! Every decision prints one line saying which way it went and why. An
//! operator who disagrees types the command they meant, and `learn.rs`
//! records the correction -- so every line typed here feeds the loop that
//! already learns from being corrected.
//!
//! **What this does not do is grant anything.** Operator-only verbs --
//! `store unlock`, `update stage`, the trust commands -- are shell verbs the
//! grammar cannot spell, and nothing here composes a command line out of the
//! model's output: the decision is one of three fixed strings with the
//! operator's own words appended.

use crate::sync::Racy;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicU8, Ordering};

/// Every verb `shell::execute` matches on. `tools/verbs.py` compares this
/// against the match arms, because Rust cannot enumerate a `match` and a
/// hand-kept list is the kind that goes quietly stale -- which is what
/// `KNOWN_COMMANDS` did.
pub const VERBS: &[&str] = &[
    "about", "abstract", "acpi", "act", "adapter", "agent", "agentlog", "answer", "app",
    "author", "autosnap", "batt", "battery", "bench", "brain", "bringup", "census", "checklist",
    "clear", "cont", "core", "cpu", "crypto", "ctx", "date", "deeptrain", "desk", "devices",
    "dhcp", "diag", "disk", "dmi", "dns", "ec", "echo", "edit", "enternet", "exec", "fat",
    "fault", "feature", "file", "firmware", "fit", "font", "forest", "fw", "gate", "gen", "godel",
    "godsays", "gpu", "grammar", "halt", "hda", "help", "http", "https", "if", "ifconfig",
    "initiative", "intent", "iwx", "kbd", "keys", "learn", "licence", "licences", "license",
    "licenses", "linux", "log", "logits", "mem", "mind", "mine", "mines", "minesweeper",
    "mixbench", "model", "mouse", "net", "netman", "network", "nvme", "open",
    "oracle", "outcome", "pagemap", "paint", "pci", "perf", "ping", "pkg", "plan", "port",
    "power", "poweroff", "probe", "racy", "reboot", "recall", "recovery", "redqueen", "refresh",
    "render", "repair", "repeat", "research", "rng", "route", "sandbox", "search", "serial",
    "shutdown", "simd", "skill", "smp", "sound", "splash", "store", "study", "sysinfo",
    "tasks", "tcp", "teach", "tensor", "think", "todo", "train", "trust", "typewriter", "uname",
    "update", "uptime", "usb", "vars", "version", "vi", "video", "voice", "wargame", "wifi", "win",
    "window", "wlan", "words", "work", "wpa2", "zeroshot",
    "vol", "volume",
];

/// How far plain language may reach on its own.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// Read-only tasks run; a task that would change things is described and
    /// not run. The default, because a sentence is a much looser instruction
    /// than a command and the cost of a misroute should be a wrong listing.
    ReadOnly = 0,
    /// A task that would change things is offered, and runs on `y`.
    Ask = 1,
    /// It runs.
    Full = 2,
}

/// `Ask` by default. It was `ReadOnly`, and a sentence asking for a folder
/// was refused with a line about `intent trust` -- correct, and of no use to
/// the person who asked: plain language that cannot do anything plain
/// language asks for is a paraphraser. Asking costs one `y`, and still never
/// runs a change nobody approved.
const DEFAULT_REACH: Reach = Reach::Ask;
static REACH: AtomicU8 = AtomicU8::new(DEFAULT_REACH as u8);
const REACH_PATH: &str = "/etc/intent";

pub fn reach() -> Reach {
    match REACH.load(Ordering::Acquire) {
        1 => Reach::Ask,
        2 => Reach::Full,
        _ => Reach::ReadOnly,
    }
}

pub fn set_reach(r: Reach) {
    REACH.store(r as u8, Ordering::Release);
    let word = match r {
        Reach::ReadOnly => "readonly",
        Reach::Ask => "ask",
        Reach::Full => "full",
    };
    crate::sysbox::write_blob(REACH_PATH, word.as_bytes().to_vec());
}

/// Read the stored setting, if any. Called once the namespace exists.
pub fn load() {
    if let Some(b) = crate::sysbox::read_blob(REACH_PATH) {
        let r = match core::str::from_utf8(&b).unwrap_or("").trim() {
            "readonly" => Reach::ReadOnly,
            "full" => Reach::Full,
            _ => Reach::Ask,
        };
        REACH.store(r as u8, Ordering::Release);
    }
}

/// What the shell should do with a line it did not recognise.
pub enum Class {
    /// Run it as Aiksi, as always.
    Code,
    /// It is language.
    Prose,
}

/// Code or language.
///
/// The parser alone cannot say, and it was the first thing tried: Aiksi
/// accepts a run of bare expressions, so `list the files in ai` *parses* --
/// as five variable reads -- and only fails at evaluation, as `undefined
/// variable 'list'`. What tells the two apart is the shape of the tokens. A
/// line is language when it reads at least one name nothing defines and
/// carries none of the marks code cannot do without: no assignment, no call,
/// no block, no string literal, no keyword. `2 + 2` is code (no unknown
/// name), `fib(10)` is code (a call), `what is 2 + 2` is language.
///
/// `known` answers whether the interpreter has a variable by this name.
pub fn classify(line: &str, known: impl Fn(&str) -> bool) -> Class {
    use crate::aiksi::lex::{self, Tok};
    const KEYWORDS: &[&str] = &["if", "else", "while", "fn", "return", "rec", "use", "nil"];
    let Ok(toks) = lex::lex(line) else { return Class::Prose };
    let mut unknown = false;
    for t in toks.iter() {
        match t {
            Tok::Assign | Tok::LParen | Tok::LBrace | Tok::Str(_) => return Class::Code,
            Tok::Ident(n) if KEYWORDS.contains(&n.as_str()) => return Class::Code,
            Tok::Ident(n) if !known(n) => unknown = true,
            _ => {}
        }
    }
    if unknown { Class::Prose } else { Class::Code }
}

/// The verb the operator probably meant, when the first word is one typo
/// away from a real one and the rest of the line looks like a command's
/// arguments rather than a sentence.
pub fn near_miss(line: &str) -> Option<&'static str> {
    let mut words = line.split_whitespace();
    let first = words.next()?;
    let rest: alloc::vec::Vec<&str> = words.collect();
    // `list the files in ai` is a sentence whose first word happens to be two
    // edits from `ls`; `lss /ai` is a command with a typo. What tells them
    // apart is the rest of the line.
    let commandish = rest.iter().all(|w| w.starts_with('/') || w.starts_with('-') || w.parse::<i64>().is_ok());
    if !commandish || first.len() < 3 {
        return None;
    }
    let limit = if first.len() <= 4 { 1 } else { 2 };
    let applets = crate::sysbox::APPLETS.iter().map(|a| a.name);
    let mut best: Option<(&'static str, usize)> = None;
    for v in VERBS.iter().copied().chain(applets) {
        let d = distance(first, v);
        if d == 0 {
            return None;
        }
        if d <= limit && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((v, d));
        }
    }
    best.map(|(v, _)| v)
}

fn distance(a: &str, b: &str) -> usize {
    let a: alloc::vec::Vec<char> = a.chars().collect();
    let b: alloc::vec::Vec<char> = b.chars().collect();
    let mut prev: alloc::vec::Vec<usize> = (0..=b.len()).collect();
    let mut cur = alloc::vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + (a[i - 1] != b[j - 1]) as usize;
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Verbs whose argument is prose by design: what follows them is the
/// operator's words for the command to take as they are, so a sentence
/// after one of these is not a reason to hand the line to the model.
const PROSE_VERBS: &[&str] = &[
    "agent", "think", "teach", "remember", "about", "echo", "author", "write", "open", "work", "route", "act",
    "search", "web", "find", "learn", "if", "repeat",
];

/// Words that are almost never a command's argument and almost always part
/// of a sentence.
const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "me", "my", "all", "some", "please", "for", "to", "of", "with", "is", "are", "what",
    "how", "why", "can", "could", "would", "you", "i", "it", "this", "that", "these", "those", "your",
];

/// A line that starts with a command's name and then carries on in English.
///
/// "help me find the big files" begins with `help`; "show me what is
/// running" might begin with a verb tomorrow. The shell would take either as
/// a command with odd arguments. The rule: the first word is a command, its
/// arguments are prose for no command (`PROSE_VERBS` take prose on purpose),
/// and the rest reads as a sentence -- a function word, and nothing shaped
/// like an argument: no path, no flag, no number.
pub fn verb_first_language(cmd: &str, rest: &str) -> bool {
    if rest.trim().is_empty() || PROSE_VERBS.contains(&cmd) {
        return false;
    }
    let is_command = VERBS.contains(&cmd) || crate::sysbox::APPLETS.iter().any(|a| a.name == cmd);
    if !is_command {
        return false;
    }
    let words: alloc::vec::Vec<&str> = rest.split_whitespace().collect();
    // One word after a command is an argument, whatever word it is: `diag
    // all`, `voice on`, `brain off`. "all" is a function word, and for the
    // life of this check the command that runs every test suite was being
    // handed to the chat model, which found it was not a command.
    if words.len() < 2 {
        return false;
    }
    let argish = words.iter().any(|w| w.starts_with('/') || w.starts_with('-') || w.parse::<i64>().is_ok() || w.contains('.'));
    let sentence = words.iter().any(|w| FUNCTION_WORDS.contains(&w.to_lowercase().as_str()));
    sentence && !argish
}

/// "start over", "new conversation", "forget this conversation": `ask new`,
/// said the way a person says it.
fn starts_over(line: &str) -> bool {
    let l = line.trim().trim_end_matches(|c: char| c.is_ascii_punctuation()).to_lowercase();
    matches!(
        l.as_str(),
        "start over" | "new conversation" | "new chat" | "forget this conversation" | "forget our conversation" | "let's start over"
    )
}

/// What a rescue goal starts with. `learn` refuses any goal carrying it: the
/// command after a rescue is the operator carrying on, not correcting a
/// routing decision, and the first live run recorded exactly that mistake --
/// `ls` filed as the right answer to "cat readme".
pub const RESCUE_MARK: &str = "rescue: ";

static RESCUE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

pub fn rescue_on() -> bool {
    RESCUE.load(Ordering::Acquire)
}

pub fn set_rescue(on: bool) {
    RESCUE.store(on, Ordering::Release);
}

/// A command the operator typed, and how it failed: the agent gets both and
/// tries to get the thing done another way. Read-only, three steps, and said
/// -- an operator who meant exactly what they typed turns it off with
/// `intent rescue off`.
pub fn rescue(line: &str, error: &str) -> Decision {
    if !super::engine_ready() {
        return Decision::Say(String::new());
    }
    Decision::Run(
        format!("agent -n 3 {}{} -- it said {}; find the right path or name first", RESCUE_MARK, line, error.trim()),
        String::from("that failed -- the agent will try to get it done another way"),
    )
}

/// Whether a line is addressed to the machine rather than asking it to do
/// something.
///
/// Not the router's question, and measured to be one it cannot answer:
/// `how are you today` routed to `stat` with all three cores agreeing. The
/// corpus has a class per applet and none for conversation, so every input
/// lands confidently on *some* task -- the 90.3% figure is accuracy among
/// tasks and says nothing about a line that is not one. So a small closed
/// vocabulary decides first, and it is deliberately conservative: a line
/// naming a path is a task whatever else it says.
///
/// The honest fix is a conversation class in `dataset.py`'s families, so the
/// probe can say "none of these"; until that is measured this is the rule.
pub fn conversational(line: &str) -> bool {
    if line.contains('/') {
        return false;
    }
    // An action verb outranks politeness. "please make a file", "can you
    // create a folder" carry "please" and "you", and were sent to the chat
    // model, which cannot touch a file and so said the request back in its
    // own words -- the operator's first afternoon with the machine, measured:
    // `please make a file named hello.txt that says hi` answered "hello.txt /
    // Hi!" and made nothing. A line asking for a thing to be done is a task,
    // whoever it is addressed to.
    if asks_for_action(line) {
        return false;
    }
    const TO_YOU: &[&str] = &["you", "your", "you're", "youre", "yourself", "glados"];
    const GREETING: &[&str] = &["hi", "hello", "hey", "thanks", "thank", "morning", "evening", "goodnight", "bye", "goodbye", "sorry", "please"];
    const OPINION: &[&str] = &["think", "feel", "feeling", "favourite", "favorite", "joke", "story", "opinion", "believe", "dream", "remember", "alive", "conscious"];
    let words: alloc::vec::Vec<String> = line
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let has = |set: &[&str]| words.iter().any(|w| set.contains(&w.as_str()));
    has(TO_YOU) || has(GREETING) || has(OPINION)
}

/// Verbs that ask for something to be done to the machine, rather than said.
const ACTION_VERBS: &[&str] = &[
    "create", "make", "delete", "remove", "rename", "move", "copy", "write", "save", "list", "show",
    "open", "find", "search", "read", "print", "count", "hash", "run", "install", "mkdir", "edit",
    "erase", "touch", "snapshot", "snap", "compare", "check", "display", "tell", "put", "add", "take",
];

/// Whether a line asks for an action: its first word after any politeness
/// ("please", "can you", "could you", "would you", "i want you to") is an
/// action verb. "tell me a joke" is the one shape this misfiles as a task,
/// so `tell` is an action only when the line names something to tell about
/// a file or the machine -- which is what the router then decides.
pub fn asks_for_action(line: &str) -> bool {
    const LEAD: &[&str] = &["please", "can", "could", "would", "will", "you", "kindly", "i", "want", "need", "to", "go", "ahead", "and", "now", "just", "glados", "hey"];
    let words: alloc::vec::Vec<String> = line
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let Some(first) = words.iter().find(|w| !LEAD.contains(&w.as_str())) else { return false };
    if first == "tell" {
        return words.iter().any(|w| matches!(w.as_str(), "file" | "files" | "folder" | "directory" | "memory" | "disk" | "time" | "date" | "uptime"));
    }
    ACTION_VERBS.contains(&first.as_str())
}

/// The exact command a plain request is asking for, when it is one of the
/// everyday shapes: make a folder, make a file (with what it says), delete,
/// rename or move, copy. `None` for anything else, which the agent takes.
///
/// Read off the sentence rather than decoded, because the measurement said
/// so: handed "please make a file named hello.txt that says hi", the agent on
/// the 0.6B re-decided the applet and ran `find 1`, twice, and "can you delete
/// hello.txt" dispatched nothing at all. The router had already said `write`
/// and the sentence had already said the name and the text. What is parsed
/// here is shown to the operator verbatim before it runs, so a misreading is
/// visible and costs one `n`.
pub fn command_for(line: &str) -> Option<String> {
    const LEAD: &[&str] = &["please", "can", "could", "would", "will", "you", "kindly", "i", "want", "need", "to", "go", "ahead", "and", "now", "just", "glados", "hey"];
    const ARTICLES: &[&str] = &["a", "an", "the", "new", "empty", "my"];
    let clean = |w: &str| -> String {
        String::from(
            w.trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c == ',' || c == '?' || c == '!')
                .trim_end_matches('.'),
        )
    };
    let words: alloc::vec::Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() && LEAD.contains(&words[i].to_lowercase().trim_matches(|c: char| !c.is_alphanumeric())) {
        i += 1;
    }
    let lower: alloc::vec::Vec<String> = words.iter().map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-').into()).collect();
    let verb = lower.get(i)?.as_str();
    let mut j = i + 1;
    let skip = |j: &mut usize, set: &[&str]| {
        while *j < lower.len() && set.contains(&lower[*j].as_str()) {
            *j += 1;
        }
    };
    // A name: one word, cleaned of quotes and a sentence's last full stop.
    let name_at = |k: usize| -> Option<String> {
        let w = clean(words.get(k)?);
        if w.is_empty() { None } else { Some(w) }
    };
    match verb {
        "create" | "make" | "add" | "mkdir" | "write" | "new" | "take" | "save" => {
            skip(&mut j, ARTICLES);
            let kind = lower.get(j).map(|s| s.as_str()).unwrap_or("");
            if matches!(kind, "snapshot" | "snap" | "checkpoint") {
                return Some(String::from("snap"));
            }
            if matches!(verb, "take" | "save") {
                return None;
            }
            let folder = matches!(kind, "folder" | "directory" | "dir");
            let file = matches!(kind, "file" | "note" | "textfile");
            if verb == "mkdir" {
                return Some(format!("mkdir {}", name_at(i + 1)?));
            }
            if !folder && !file {
                // "write hello to notes.txt"
                if verb == "write" {
                    let to = lower.iter().rposition(|w| w == "to" || w == "into")?;
                    if to <= i + 1 || to + 2 != words.len() {
                        return None;
                    }
                    let text: alloc::vec::Vec<&str> = words[i + 1..to].to_vec();
                    return Some(format!("write {} {}", name_at(to + 1)?, text.join(" ").trim_matches('"')));
                }
                return None;
            }
            j += 1;
            skip(&mut j, &["called", "named", "titled"]);
            let name = name_at(j)?;
            if folder {
                return (j + 1 == words.len()).then(|| format!("mkdir {}", name));
            }
            // What the file says: "that says hi", "saying hi", "containing
            // hi", "with the text hi", "with hi".
            let mut k = j + 1;
            skip(&mut k, &["that", "which", "and", "it"]);
            if k >= words.len() {
                return Some(format!("write {}", name));
            }
            if !matches!(lower[k].as_str(), "says" | "saying" | "say" | "containing" | "contains" | "with") {
                return None;
            }
            k += 1;
            skip(&mut k, &["the", "text", "words", "contents", "content"]);
            if k >= words.len() {
                return None;
            }
            let text = words[k..].join(" ");
            Some(format!("write {} {}", name, text.trim_matches(|c: char| c == '"' || c == '\'')))
        }
        "delete" | "remove" | "erase" | "rm" | "trash" => {
            skip(&mut j, ARTICLES);
            skip(&mut j, &["file", "folder", "directory", "dir", "called", "named"]);
            (j + 1 == words.len()).then(|| ()).and_then(|_| name_at(j)).map(|n| format!("rm {}", n))
        }
        "rename" | "move" | "mv" | "copy" | "cp" | "duplicate" => {
            skip(&mut j, ARTICLES);
            skip(&mut j, &["file", "folder", "directory", "dir", "called", "named"]);
            let a = name_at(j)?;
            let to = j + 1;
            if lower.get(to).map(|s| s.as_str()) != Some("to") && lower.get(to).map(|s| s.as_str()) != Some("as") {
                return None;
            }
            if to + 2 != words.len() {
                return None;
            }
            let b = name_at(to + 1)?;
            let applet = if matches!(verb, "copy" | "cp" | "duplicate") { "cp" } else { "mv" };
            Some(format!("{} {} {}", applet, a, b))
        }
        _ => None,
    }
}

/// `write` followed by prose rather than a path: "write a haiku about
/// kernels", "write me a poem". A path has a slash or a dot in it, and a
/// sentence has a function word.
pub fn write_is_prose(rest: &str) -> bool {
    let words: alloc::vec::Vec<&str> = rest.split_whitespace().collect();
    let Some(first) = words.first() else { return false };
    if first.contains('/') || first.contains('.') || words.len() < 2 {
        return false;
    }
    words.iter().any(|w| FUNCTION_WORDS.contains(&w.to_lowercase().as_str()) || matches!(w.to_lowercase().as_str(), "poem" | "haiku" | "story" | "song" | "letter" | "about"))
}

/// The word a sentence-shaped search is for: "files that mention godel" is
/// `godel`, "anything called notes" is `notes`, "the word cathedral" is
/// `cathedral`. `None` when the argument is already a plain search term.
pub fn find_keyword(rest: &str) -> Option<String> {
    const FRAME: &[&str] = &[
        "for", "a", "an", "the", "all", "any", "files", "file", "anything", "everything", "something",
        "that", "which", "mention", "mentions", "mentioning", "contain", "contains", "containing", "with",
        "named", "called", "about", "word", "text", "string", "me", "in", "where", "has", "have", "say", "says",
    ];
    let words: alloc::vec::Vec<&str> = rest.split_whitespace().collect();
    if words.len() < 2 || !words.iter().any(|w| FRAME.contains(&w.to_lowercase().as_str())) {
        return None;
    }
    let kept: alloc::vec::Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| !FRAME.contains(&w.to_lowercase().as_str()))
        .map(|w| w.trim_matches(|c: char| c == '"' || c == '\'' || c == '?' || c == '.' || c == ','))
        .filter(|w| !w.is_empty())
        .collect();
    (!kept.is_empty()).then(|| kept.join(" "))
}

/// A request for words rather than for an action: "write a haiku", "explain
/// how a kernel works", "tell me a story". The chat model's job, and the one
/// thing it is good at.
fn creative(line: &str) -> bool {
    const ASKS: &[&str] = &["write", "compose", "explain", "describe", "summarize", "summarise", "translate", "tell"];
    let first = line.split_whitespace().map(|w| w.to_lowercase()).find(|w| !matches!(w.as_str(), "please" | "can" | "could" | "would" | "you"));
    first.map_or(false, |w| ASKS.contains(&w.as_str())) && command_for(line).is_none() && !line.contains('/')
}

/// A question, rather than a request: wh- or yes/no shaped, or a question mark.
fn question(line: &str) -> bool {
    const Q: &[&str] = &["what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does", "did", "was", "were", "whats", "what's"];
    let first = line.split_whitespace().next().map(|w| w.to_lowercase()).unwrap_or_default();
    line.trim_end().ends_with('?') || Q.contains(&first.as_str())
}

/// Words that make a question about the file tree, which the applets answer
/// and the chat model would only invent.
const TREE_WORDS: &[&str] = &["file", "files", "folder", "folders", "directory", "directories", "dir", "snapshot", "snapshots", "disk", "stored", "tree"];

/// "my name is Lummy", "call me Ana", "my favourite colour is green": a fact
/// about the operator, kept with `remember`.
pub fn fact_about_me(line: &str) -> Option<String> {
    let l = line.trim().trim_end_matches(|c: char| c == '.' || c == '!');
    let lower = l.to_lowercase();
    if let Some(rest) = lower.strip_prefix("call me ") {
        let name = &l[l.len() - rest.len()..];
        return Some(format!("my name is {}", name.trim()));
    }
    if lower.starts_with("my ") && (lower.contains(" is ") || lower.contains(" are ")) && !lower.ends_with('?') && !l.contains('/') {
        return Some(String::from(l));
    }
    None
}

/// The paths a line names, cleaned of a sentence's punctuation.
fn paths_in(line: &str) -> alloc::vec::Vec<String> {
    line.split_whitespace()
        .filter(|w| w.starts_with('/'))
        .map(|w| String::from(w.trim_end_matches(|c: char| c == '.' || c == ',' || c == '?' || c == '!')))
        .collect()
}

/// A plain listing asked for in words: "go to /lib and list it", "show me
/// what is in /lib", "what's inside /ai". `ls` of the one path named, or `cd`
/// for "go to /x" alone. The agent was handed "go to /lib and list it",
/// ran `cd /` and declared the goal met.
pub fn listing_for(line: &str) -> Option<String> {
    let paths = paths_in(line);
    let [p] = paths.as_slice() else { return None };
    if changing_verb(line) {
        return None;
    }
    let lower = line.to_lowercase();
    let words: alloc::vec::Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let lists = words.iter().any(|w| matches!(*w, "list" | "contents" | "inside" | "show" | "ls"));
    if lists && crate::sysbox::is_dir(p) {
        return Some(format!("ls {}", p));
    }
    let goes = matches!(words.first().copied(), Some("go" | "cd" | "enter")) && words.len() <= 4;
    (goes && crate::sysbox::is_dir(p)).then(|| format!("cd {}", p))
}

/// A question about a thing the machine can show: the command that shows it.
/// "which file in /lib is the biggest" was handed to the chat model, which
/// cannot see a file and said so -- the machine could, with one `ls`.
pub fn grounding_for(line: &str) -> Option<String> {
    let paths = paths_in(line);
    let [p] = paths.as_slice() else { return None };
    let lower = line.to_lowercase();
    let asks = question(line)
        || ["biggest", "largest", "smallest", "newest", "oldest", "how many", "count", "summarise", "summarize", "explain"]
            .iter()
            .any(|w| lower.contains(w));
    if !asks || changing_verb(line) {
        return None;
    }
    if crate::sysbox::is_dir(p) {
        Some(format!("ls {}", p))
    } else if crate::sysbox::read_blob(p).is_some() {
        Some(format!("cat {}", p))
    } else {
        None
    }
}

/// A question about something the operator said about themselves, answered
/// from what they said. "what is my name", "who am i", "what is my favourite
/// colour". Looked up rather than generated: the 0.6B, given "their name is
/// Lummy" beside the question, answered "My name is Lummy", and once "I am
/// Lummy" -- the fact right, the person wrong, every time.
pub fn recall_about(line: &str, about: &str) -> Option<String> {
    let q = line.trim().trim_end_matches(|c: char| c == '?' || c == '.' || c == '!').to_lowercase();
    let subject = if matches!(q.as_str(), "who am i" | "do you know who i am" | "do you know my name") {
        String::from("name")
    } else {
        let rest = q
            .strip_prefix("what is my ")
            .or_else(|| q.strip_prefix("what's my "))
            .or_else(|| q.strip_prefix("whats my "))
            .or_else(|| q.strip_prefix("what are my "))?;
        String::from(rest.trim())
    };
    for l in about.lines().rev() {
        let t = l.trim();
        let lower = t.to_lowercase();
        let Some(after) = lower.strip_prefix("my ") else { continue };
        for verb in [" is ", " are "] {
            if let Some(at) = after.find(verb) {
                if after[..at].trim() == subject {
                    let value = &t[3 + at + verb.len()..];
                    return Some(format!("Your {}{}{}.", subject, verb, value.trim_end_matches('.')));
                }
            }
        }
    }
    None
}

/// Whether the action a line asks for changes the machine.
pub fn changing_verb(line: &str) -> bool {
    const CHANGES: &[&str] = &[
        "create", "make", "delete", "remove", "rename", "move", "copy", "write", "save", "mkdir", "edit",
        "erase", "touch", "put", "add", "install", "snapshot", "snap", "take",
    ];
    asks_for_action(line)
        && line
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.to_lowercase())
            .any(|w| CHANGES.contains(&w.as_str()))
}

/// What the shell runs, decided from a line of language.
pub enum Decision {
    /// A command line for `shell::execute`, and the one-line reason.
    Run(String, String),
    /// A turn of the conversation: the operator's own words, and why.
    Chat(String, String),
    /// A question about something the machine can show: run the read-only
    /// command, then answer the question from what it printed.
    Grounded(String, String, String),
    /// Forget the conversation and start a new one.
    Forget,
    /// Nothing runs; say this.
    Say(String),
}

/// A pending offer that is an exact command rather than a goal for the agent.
const EXACT_MARK: &str = "\u{1}exact ";

/// A task waiting for `y`, under `Reach::Ask`.
static PENDING: Racy<Option<String>> = Racy::new(None);

/// The answer to a pending offer, if this line is one.
pub fn answer(line: &str) -> Option<Decision> {
    let pending = unsafe { PENDING.get() };
    if pending.is_none() {
        // A yes or no with nothing asked. Handed to the model it was a chat
        // turn, and the model answered "n" by saying "n" back.
        return match line.trim() {
            "y" | "yes" | "n" | "no" => Some(Decision::Say(String::from("nothing is waiting for a yes or no"))),
            _ => None,
        };
    }
    match line.trim() {
        "y" | "yes" => {
            let goal = pending.take()?;
            if let Some(cmd) = goal.strip_prefix(EXACT_MARK) {
                return Some(Decision::Run(String::from(cmd), String::from("as you said")));
            }
            Some(Decision::Run(format!("agent --trust full {}", goal), String::from("as you said")))
        }
        "n" | "no" => {
            pending.take();
            Some(Decision::Say(String::from("left alone")))
        }
        _ => None,
    }
}

/// Decide what a line of language is for.
pub fn decide(line: &str) -> Decision {
    use super::harness::{self, Trust};
    if !super::engine_ready() {
        return Decision::Say(String::from("that is not a command, and no model is loaded to ask"));
    }
    let chat = |why: String| Decision::Chat(String::from(line), why);
    if starts_over(line) {
        return Decision::Forget;
    }
    if super::wargame::invited(line) {
        return Decision::Run(String::from("wargame"), String::from("an invitation to play"));
    }
    if let Some(answer) = crate::sysbox::read_blob(super::companion::ABOUT)
        .and_then(|b| recall_about(line, &String::from_utf8_lossy(&b)))
    {
        return Decision::Say(answer);
    }
    if let Some(fact) = fact_about_me(line) {
        return Decision::Run(format!("remember {}", fact), String::from("something about you, so it is kept"));
    }
    if creative(line) {
        return chat(String::from("a request for words, so it is a conversation"));
    }
    if conversational(line) {
        return chat(String::from("addressed to me, so it is a conversation"));
    }
    // A question that names no file and no path is for the conversation, which
    // is given the machine's own readings when the question is about them. The
    // router answers every line with *some* applet: "what is my name" went to
    // `pwd` with all three cores agreeing, and "how much memory is free" to
    // `du`, which measures the disk.
    let lower = line.to_lowercase();
    let about_tree = lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| TREE_WORDS.contains(&w) || crate::sysbox::APPLETS.iter().any(|a| a.name == w));
    if question(line) && !line.contains('/') && !about_tree && !asks_for_action(line) {
        return chat(String::from("a question, so it is a conversation"));
    }
    if let Some(cmd) = listing_for(line) {
        return Decision::Run(cmd.clone(), format!("'{}', read off what you said", cmd));
    }
    if let Some(cmd) = grounding_for(line) {
        return Decision::Grounded(String::from(line), cmd.clone(), format!("'{}', then an answer from what it shows", cmd));
    }
    if !harness::ensure_router() {
        return chat(String::from("no router yet, so it is a question"));
    }
    let Some((choice, v)) = harness::route_verdict(line, Trust::Full) else {
        return chat(format!("the router could not run ({}), so it is a question", super::engine_refusal()));
    };
    let names_target = line.contains('/')
        || line.split_whitespace().any(|w| crate::sysbox::APPLETS.iter().any(|a| a.name == w));
    // A line that asks for something to be done is a task whatever the
    // router's confidence: the agent decides the applet again for itself, and
    // a task sent to the chat model is answered by a paraphrase of it.
    let action = asks_for_action(line);
    let task = v.agreement == 3 || (v.agreement == 2 && names_target) || action;
    if !task {
        return chat(format!("{} of 3 agree on '{}' -- a conversation, not a task", v.agreement, choice.applet));
    }
    let why = format!("'{}', {} of 3 agree", choice.applet, v.agreement);
    // Whether it changes things is the verb's to say as much as the router's:
    // the router sent "delete hello.txt" to `write` and "remove" to `cd`, and
    // a request that is plainly a change must not run as a read-only one.
    let changes = choice.mutates || changing_verb(line);
    if !changes {
        // A read-only applet whose whole argument is one path, and a line
        // naming exactly one path: that is the command a person would have
        // typed, so it runs. Handing it to the agent instead decoded
        // `stat 1: 100% of the files in /ai` for "list the files in /ai"
        // and then searched for nothing twice -- the agent re-decides the
        // applet the router already chose, and decodes arguments the line
        // already says.
        let paths: alloc::vec::Vec<&str> = line.split_whitespace().filter(|w| w.starts_with('/')).collect();
        let shape = crate::sysbox::APPLETS.iter().find(|a| a.name == choice.applet).map(|a| a.args);
        match (shape, paths.as_slice()) {
            (Some("[path]") | Some("<path>"), [p]) => {
                let p = p.trim_end_matches(|c: char| c == '.' || c == ',' || c == '?' || c == '!');
                return Decision::Run(format!("{} {}", choice.applet, p), format!("{} -- run as said", why));
            }
            (Some(""), []) => return Decision::Run(String::from(choice.applet), format!("{} -- run as said", why)),
            _ => {}
        }
        return Decision::Run(format!("agent -n 3 {}", line), format!("a task for {} -- the agent takes it", why));
    }
    let exact = command_for(line);
    match reach() {
        Reach::Ask if exact.is_some() => {
            let cmd = exact.unwrap_or_default();
            unsafe { *PENDING.get() = Some(format!("{}{}", EXACT_MARK, cmd)) };
            Decision::Say(format!("that would run '{}' -- go ahead? y/n", cmd))
        }
        Reach::Full if exact.is_some() => {
            let cmd = exact.unwrap_or_default();
            Decision::Run(cmd.clone(), format!("'{}', read off what you said", cmd))
        }
        Reach::ReadOnly => Decision::Say(format!(
            "that would change things ({}). 'intent trust ask' lets me offer, or say it as 'agent --trust full ...'",
            why
        )),
        Reach::Ask => {
            unsafe { *PENDING.get() = Some(String::from(line)) };
            if choice.mutates {
                Decision::Say(format!("that would change things ({}) -- run it? y/n", why))
            } else {
                Decision::Say(String::from("that would change things -- run it? y/n"))
            }
        }
        Reach::Full => Decision::Run(format!("agent --trust full -n 3 {}", line), format!("a task for {}", why)),
    }
}

/// `diag intent`. The model-free half: what is code, what is a typo, what
/// may run without asking.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let none = |_: &str| false;
    let some = |n: &str| n == "x";
    let is_code = |l: &str, k: &dyn Fn(&str) -> bool| matches!(classify(l, k), Class::Code);
    // The offer and its answer, driven directly: the router on the measured
    // checkpoint sends "remove" and "rename" to `cd`, so a live run never
    // reaches this path, and a path nothing reaches is a path nothing checks.
    unsafe { *PENDING.get() = Some(String::from("remove /tmp/x")) };
    let yes = matches!(answer("y"), Some(Decision::Run(ref c, _)) if c == "agent --trust full remove /tmp/x");
    let spent = !matches!(answer("y"), Some(Decision::Run(..)));
    unsafe { *PENDING.get() = Some(String::from("remove /tmp/x")) };
    let no = matches!(answer("no"), Some(Decision::Say(_))) && !matches!(answer("y"), Some(Decision::Run(..)));
    let unrelated = answer("ls").is_none();
    let rows: [(&str, bool); 61] = [
        ("'help me find the big files' is a sentence, not the help command", verb_first_language("help", "me find the big files")),
        ("'ls /ai' is a command", !verb_first_language("ls", "/ai")),
        ("'cat the readme please' is a sentence", verb_first_language("cat", "the readme please")),
        ("a prose verb keeps its prose: 'remember my name is Ana'", !verb_first_language("remember", "my name is Ana")),
        ("an unknown first word is not this rule's business", !verb_first_language("hello", "there you")),
        ("'diag all' is a command with one argument, however function-word the argument", !verb_first_language("diag", "all") && !verb_first_language("voice", "on")),
        ("'start over' forgets the conversation", starts_over("Start over.")),
        ("an ordinary sentence does not", !starts_over("start over the download")),
        ("y runs the offered task, at full trust, verbatim", yes),
        ("and the offer is spent once taken", spent),
        ("n withdraws it, and a later y runs nothing", no),
        ("with nothing offered, a line is not an answer", unrelated),
        ("an assignment is code", is_code("x = 3", &none)),
        ("a call is code", is_code("fib(10)", &none)),
        ("arithmetic is code", is_code("2 + 2", &none)),
        ("a known variable is code", is_code("x", &some)),
        ("an unknown lone word is language", !is_code("hello", &none)),
        ("a sentence is language", !is_code("list the files in ai", &none)),
        ("a question is language", !is_code("what is in /ai", &none)),
        ("a sentence that parses as five reads is still language", !is_code("how are you today", &none)),
        ("arithmetic inside a question is language", !is_code("what is 2 + 2", &none)),
        ("a typo'd command with a path is caught", near_miss("lss /ai") == Some("ls")),
        ("a sentence is not a typo", near_miss("list the files in ai").is_none()),
        ("a real verb is not a typo", near_miss("diag ctx").is_none() && near_miss("ls /ai").is_none()),
        ("being asked how it is is conversation", conversational("how are you today")),
        ("a greeting is conversation", conversational("hello there")),
        ("a task is not conversation", !conversational("list the files in ai")),
        ("a path makes it a task whatever else it says", !conversational("could you list /ai for me")),
        ("the default reach asks before anything that mutates, and never just runs it",
            DEFAULT_REACH == Reach::Ask),
        ("'please make a file' is a task, not small talk", !conversational("please make a file named hello.txt that says hi")),
        ("'can you create a folder' is a task", !conversational("can you create a folder called notes")),
        ("'thank you' is still conversation", conversational("thank you")),
        ("'tell me a joke' is still conversation", conversational("tell me a joke")),
        ("'tell me what files are here' is a task", asks_for_action("tell me what files are here")),
        ("'delete hello.txt' changes things whatever the router guessed", changing_verb("can you delete hello.txt")),
        ("'list the files' changes nothing", !changing_verb("list the files in ai")),
        ("'please make a file named hello.txt that says hi' is write hello.txt hi",
            command_for("please make a file named hello.txt that says hi").as_deref() == Some("write hello.txt hi")),
        ("'create a folder called notes' is mkdir notes", command_for("create a folder called notes").as_deref() == Some("mkdir notes")),
        ("'can you delete hello.txt' is rm hello.txt", command_for("can you delete hello.txt").as_deref() == Some("rm hello.txt")),
        ("'rename a.txt to b.txt' is mv", command_for("rename a.txt to b.txt").as_deref() == Some("mv a.txt b.txt")),
        ("'copy /a to /b' is cp", command_for("copy /a to /b").as_deref() == Some("cp /a /b")),
        ("'write hello world to note.txt' is write", command_for("write hello world to note.txt").as_deref() == Some("write note.txt hello world")),
        ("anything else is the agent's", command_for("make the machine faster").is_none() && command_for("delete everything older than a week").is_none()),
        ("an exact offer runs exactly that on y", {
            unsafe { *PENDING.get() = Some(format!("{}rm /tmp/x", EXACT_MARK)) };
            matches!(answer("y"), Some(Decision::Run(ref c, _)) if c == "rm /tmp/x")
        }),
        ("'my name is Lummy' is kept", fact_about_me("my name is Lummy").as_deref() == Some("my name is Lummy")),
        ("'call me Ana' is kept as a name", fact_about_me("call me Ana").as_deref() == Some("my name is Ana")),
        ("'my files are gone?' is not a fact", fact_about_me("my files are gone?").is_none()),
        ("'write a haiku about kernels' is prose, not the write applet", write_is_prose("a haiku about kernels")),
        ("'write notes.txt hello' is the applet", !write_is_prose("notes.txt hello")),
        ("'files that mention godel' searches for godel", find_keyword("files that mention godel").as_deref() == Some("godel")),
        ("and a plain search term is left alone", find_keyword("godel").is_none()),
        ("'what is my name' is a question for the conversation", question("what is my name") && creative("write a haiku about kernels")),
        ("'what is my name' is answered from what was kept",
            recall_about("what is my name?", "my name is Lummy\nmy favourite colour is green").as_deref() == Some("Your name is Lummy.")),
        ("'who am i' too", recall_about("who am i", "my name is Lummy").as_deref() == Some("Your name is Lummy.")),
        ("and an unknown subject falls through to the conversation", recall_about("what is my age", "my name is Lummy").is_none()),
        ("'take a snapshot' is snap", command_for("take a snapshot").as_deref() == Some("snap")),
        ("'go to /lib and list it' is ls /lib", listing_for("go to /lib and list it").as_deref() == Some("ls /lib")),
        ("'which file in /lib is the biggest' is answered from ls /lib",
            grounding_for("which file in /lib is the biggest").as_deref() == Some("ls /lib")),
        ("a question naming no path is not grounded", grounding_for("which file is the biggest").is_none()),
        ("a yes with nothing asked is answered, not handed to the model", {
            unsafe { *PENDING.get() = None };
            matches!(answer("y"), Some(Decision::Say(_))) && answer("ls").is_none()
        }),
        ("no operator-only verb is an applet the grammar could spell", ["unlock", "trust", "update", "store", "skill", "work"]
            .iter()
            .all(|v| crate::sysbox::APPLETS.iter().all(|a| a.name != *v))),
    ];
    let mut ok = true;
    for (what, good) in rows.iter() {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
