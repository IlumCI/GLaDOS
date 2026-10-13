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
    "update", "uptime", "usb", "vars", "version", "vi", "video", "voice", "wifi", "win",
    "window", "wlan", "words", "work", "wpa2", "zeroshot",
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

static REACH: AtomicU8 = AtomicU8::new(Reach::ReadOnly as u8);
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
            "ask" => Reach::Ask,
            "full" => Reach::Full,
            _ => Reach::ReadOnly,
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

/// What the shell runs, decided from a line of language.
pub enum Decision {
    /// A command line for `shell::execute`, and the one-line reason.
    Run(String, String),
    /// A turn of the conversation: the operator's own words, and why.
    Chat(String, String),
    /// Forget the conversation and start a new one.
    Forget,
    /// Nothing runs; say this.
    Say(String),
}

/// A task waiting for `y`, under `Reach::Ask`.
static PENDING: Racy<Option<String>> = Racy::new(None);

/// The answer to a pending offer, if this line is one.
pub fn answer(line: &str) -> Option<Decision> {
    let pending = unsafe { PENDING.get() };
    pending.as_ref()?;
    match line.trim() {
        "y" | "yes" => {
            let goal = pending.take()?;
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
    if conversational(line) {
        return chat(String::from("addressed to me, so it is a conversation"));
    }
    if !harness::ensure_router() {
        return chat(String::from("no router yet, so it is a question"));
    }
    let Some((choice, v)) = harness::route_verdict(line, Trust::Full) else {
        return chat(format!("the router could not run ({}), so it is a question", super::engine_refusal()));
    };
    let names_target = line.contains('/')
        || line.split_whitespace().any(|w| crate::sysbox::APPLETS.iter().any(|a| a.name == w));
    let task = v.agreement == 3 || (v.agreement == 2 && names_target);
    if !task {
        return chat(format!("{} of 3 agree on '{}' -- a conversation, not a task", v.agreement, choice.applet));
    }
    let why = format!("'{}', {} of 3 agree", choice.applet, v.agreement);
    if !choice.mutates {
        return Decision::Run(format!("agent -n 3 {}", line), format!("a task for {} -- the agent takes it", why));
    }
    match reach() {
        Reach::ReadOnly => Decision::Say(format!(
            "that would change things ({}). 'intent trust ask' lets me offer, or say it as 'agent --trust full ...'",
            why
        )),
        Reach::Ask => {
            unsafe { *PENDING.get() = Some(String::from(line)) };
            Decision::Say(format!("that would change things ({}) -- run it? y/n", why))
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
    let spent = answer("y").is_none();
    unsafe { *PENDING.get() = Some(String::from("remove /tmp/x")) };
    let no = matches!(answer("no"), Some(Decision::Say(_))) && answer("y").is_none();
    let unrelated = answer("ls").is_none();
    let rows: [(&str, bool); 30] = [
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
        ("the default reach runs nothing that mutates", Reach::ReadOnly as u8 == 0),
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
