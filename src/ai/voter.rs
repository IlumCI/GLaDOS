//! A council core the machine wrote for itself.
//!
//! The file is `voter.rs` and not `core.rs` because a module named `core`
//! inside `ai` shadows the `core` crate for every file in that module --
//! `core::sync` and `core::f32` stop resolving, several files away, for a
//! reason that reads as nonsense. The council calls its members cores and so
//! does this prose; only the filename gives way.
//!
//! The council has three voters: a ridge probe that answers, and two counters
//! that corroborate. Their *agreement* is the signal -- 90% right when all
//! three agree against 50% when they split -- so a fourth voter is not a small
//! addition. It changes what agreement means.
//!
//! This is where a machine-written one lives. A core is an Aiksi program with
//! one function:
//!
//! ```text
//! fn vote(text: str, allowed: list): int
//! ```
//!
//! It is handed the task and the class indices the trust level permits, and
//! answers one of them. `Council::corroborate` already takes exactly that and
//! answers exactly that, so a core slots in beside `lexical` and `character`
//! without reshaping anything.
//!
//! ### Why it is a program and not weights
//!
//! Weights are what the godel loop already searches. A program is a different
//! kind of variation: it can look at the text in ways nobody wrote a feature
//! for, and -- the part that matters -- a person can read it afterwards and
//! say whether the thing it found is real. An adapter that improves routing by
//! 2% is a number; a core that improves it by 2% is a rule somebody can argue
//! with.
//!
//! ### What keeps this safe
//!
//! A core runs on every routing decision, so it is the most privileged thing
//! in this tree by sheer frequency. Three bounds, and none of them is trust:
//!
//! - **Sandboxed.** `Caps::Sandbox`, so no raw memory, no ports, no network,
//!   no model. The capability table refuses those before dispatch rather than
//!   after, so a core cannot reach them however it is written.
//! - **Budgeted.** Its own step ceiling, far below the interpreter's default.
//!   A core that loops is a routing decision that never returns, and routing
//!   is on the path of everything.
//! - **Fresh per vote.** A new interpreter each time, so a core cannot carry
//!   state between decisions. That is not tidiness: a core that remembered
//!   would make the corpus order part of the answer, and a benchmark run twice
//!   would stop agreeing with itself -- which is the property every judge here
//!   is built on.
//!
//! Nothing is installed by being written. `bench` measures a candidate against
//! the held-out slice and the judges decide; `install` is what wires one in,
//! and it takes a hash somebody has seen a verdict for.

use crate::aiksi::eval::{Caps, Interp, Value};
use crate::aiksi::parse::Stmt;
use crate::store::sha256;
use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where candidate cores are kept, by content address.
pub const ROOT: &str = "/ai/cores";
/// Names the hash of the core in force, if any.
pub const CURRENT: &str = "/ai/cores/current";

/// Steps one vote may take.
///
/// Two orders of magnitude below the interpreter's default and one below a
/// repaint's. A repaint happens when something changes; a vote happens on
/// every routing decision, and the decision is what the operator is waiting
/// for. A core that needs more than this to classify one short string is not
/// a core worth having.
pub const VOTE_BUDGET: u64 = 20_000;

/// Where a core may write, which is nowhere it can reach.
///
/// A core is handed a jail it has no builtin to use: `write` is the only
/// writing builtin a sandbox admits, and this path is under `/ai/cores`, which
/// holds the cores themselves. A core that writes there would be editing its
/// own siblings, so the jail exists to be a valid answer to `here()` rather
/// than to be used.
const JAIL: &str = "/ai/cores/scratch";

pub struct Core {
    pub hash: [u8; 32],
    /// Parsed once. Re-lexing per vote would cost more than the vote.
    prog: Vec<Stmt>,
}

/// The program text of a stored core.
pub fn source(hash: &[u8; 32]) -> Option<String> {
    let mut path = String::from(ROOT);
    path.push('/');
    path.push_str(&hex(hash));
    sysbox::read_blob(&path).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Store a candidate and answer its address. Storing is not installing.
pub fn store(src: &str) -> [u8; 32] {
    let h = sha256::hash(src.as_bytes());
    let mut path = String::from(ROOT);
    path.push('/');
    path.push_str(&hex(&h));
    if sysbox::read_blob(&path).is_none() {
        sysbox::write_text(&path, src);
    }
    h
}

/// Parse a stored core, or say why it is not one.
pub fn load(hash: &[u8; 32]) -> Result<Core, String> {
    let src = source(hash).ok_or_else(|| String::from("no such core"))?;
    parse(hash, &src)
}

pub fn parse(hash: &[u8; 32], src: &str) -> Result<Core, String> {
    let toks = crate::aiksi::lex::lex(src)?;
    let prog = crate::aiksi::parse::parse(toks)?;
    let core = Core { hash: *hash, prog };
    // A core that does not define `vote` is not a core, and finding that out
    // at the first routing decision would be finding it out in the worst place.
    let mut it = core.interp();
    core.arm(&mut it)?;
    if it.invoke("vote", &[Value::Str(String::new()), Value::List(Vec::new())]).is_err() {
        // An error from the body is fine -- an empty allowed list has no right
        // answer. What is checked here is that the name resolves at all.
        if !it.has_fn("vote") {
            return Err(String::from("a core must define fn vote(text, allowed)"));
        }
    }
    Ok(core)
}

impl Core {
    fn interp(&self) -> Interp {
        Interp::sandboxed(JAIL).with_step_budget(VOTE_BUDGET)
    }

    /// Run the top level, which is what registers the functions.
    fn arm(&self, it: &mut Interp) -> Result<(), String> {
        it.run(&self.prog).map(|_| ())
    }

    /// One vote, and what it cost.
    ///
    /// `None` when the core failed, ran away, or answered something outside
    /// the permitted set. All three are the same thing to a caller -- no
    /// opinion -- and a core that answers a class the trust level forbids must
    /// not be able to smuggle it in by being confident.
    pub fn vote(&self, text: &str, allowed: &[usize]) -> (Option<usize>, u64) {
        let mut it = self.interp();
        if self.arm(&mut it).is_err() {
            return (None, it.steps());
        }
        let list = Value::List(allowed.iter().map(|i| Value::Int(*i as i64)).collect());
        let got = it.invoke("vote", &[Value::Str(text.to_string()), list]);
        let steps = it.steps();
        let Ok(v) = got else { return (None, steps) };
        let Ok(n) = v.as_int() else { return (None, steps) };
        if n < 0 || !allowed.contains(&(n as usize)) {
            return (None, steps);
        }
        (Some(n as usize), steps)
    }
}

/// The core in force, if one is installed.
///
/// Parsed on first use and kept, because the alternative is parsing it on
/// every routing decision. The cache is cleared by `install`, so wiring a core
/// in takes effect at once rather than at the next boot.
pub fn installed() -> Option<&'static Core> {
    unsafe {
        if let Some(c) = (*CACHE.get()).as_ref() {
            return Some(c);
        }
    }
    let text = sysbox::read_blob(CURRENT)?;
    let h = unhex(core::str::from_utf8(&text).ok()?.trim())?;
    let c = load(&h).ok()?;
    unsafe {
        *CACHE.get() = Some(c);
        (*CACHE.get()).as_ref()
    }
}

static CACHE: crate::sync::Racy<Option<Core>> = crate::sync::Racy::new(None);

/// Wire a core in. Only a hash somebody has seen a verdict for should reach
/// this; the judging is `harness::core_bench`.
pub fn install(hash: &[u8; 32]) -> bool {
    if load(hash).is_err() {
        return false;
    }
    unsafe { *CACHE.get() = None };
    sysbox::write_text(CURRENT, &hex(hash))
}

/// Take the installed core out of the decision path.
pub fn uninstall() -> bool {
    unsafe { *CACHE.get() = None };
    sysbox::detach(CURRENT)
}

pub fn hex(h: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in h {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        s.push(char::from_digit((b & 15) as u32, 16).unwrap_or('0'));
    }
    s
}

pub fn unhex(text: &str) -> Option<[u8; 32]> {
    let b = text.as_bytes();
    if b.len() < 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        let hi = (b[i * 2] as char).to_digit(16)?;
        let lo = (b[i * 2 + 1] as char).to_digit(16)?;
        *o = (hi * 16 + lo) as u8;
    }
    Some(out)
}

// --- the machine writing one --------------------------------------------
//
// Everything above this line is about running a core somebody else wrote.
// What follows is the machine writing its own, which is the difference
// between a loop that searches numbers and one that searches code.
//
// ### The shape, and why it is this shape
//
// A core is *composed*, never emitted as text. The model picks from lists;
// the kernel turns the picks into a program. That is `author.rs`'s whole
// method -- "do not check afterwards for something that can be made
// unreachable" -- applied here because the alternative is worse than usual: a
// core that fails to parse is a wasted night, and a core that parses and does
// something unintended runs on every routing decision until somebody notices.
//
// ### What the machine actually decides
//
// Said plainly, because the failure mode of a thing like this is theatre.
// The kernel supplies the alternatives: which classes can be written about at
// all, and which cues are usable for each. The model decides how many rules to
// write, which class each one argues for, and which cue argues for it. That is
// a real choice over a space the kernel does not rank -- but it is a choice
// among prepared options, not authorship from nothing, and calling it more
// than that would be the same overclaim this module refuses everywhere else.
//
// ### What makes a cue usable
//
// Two hard filters, both explainable, neither a learned weight:
//
//   * it is used by at least `MIN_USES` training examples of its class, so a
//     word that appeared once by accident cannot become a routing rule;
//   * it appears under **no other class** in the training slice, so a rule
//     built on it cannot fire for a class it was not written for.
//
// The training slice only. `harness::split_of` is shared rather than copied
// precisely so this cannot drift onto the slice the judge measures on -- a
// producer that mined validation would not fail, it would simply score better
// than it deserved, which is the hardest kind of mistake to notice.

/// One rule: a surface cue, and the class it argues for.
pub struct Clause {
    pub cue: String,
    pub class: usize,
}

/// Shortest cue worth having. Two letters match inside other words constantly.
const MIN_CUE: usize = 3;
/// Training examples of its own class a cue must appear in.
const MIN_USES: u32 = 2;
/// Cues offered per class. A grammar the model cannot get through is a decode
/// that spends its allowance and commits to nothing.
const CUE_CAP: usize = 16;
/// Most rules one core may have. Each is two decodes, and a core is judged as
/// a whole -- a long one fails for a reason spread over many rules.
const MAX_CLAUSES: usize = 3;
/// Draws allowed per decision before giving up on it.
const DECODE_TRIES: usize = 3;

/// `choose`, drawn again when the model does not commit.
///
/// A constrained decode is *sampled*, at 0.7, precisely so a model that prefers
/// a whitespace token cannot sit on that preference forever -- `author::choose`
/// says as much. The consequence is that failing to commit is a draw, not a
/// verdict, and the first version of this treated it as a verdict: one
/// unlucky decode among four alternatives returned `None`, `author` gave up,
/// and the whole night's composition was abandoned. Measured, not reasoned
/// about -- `core author` printed "no choice among 4 after 0 step(s)" and
/// wrote nothing, on a machine where the same call had worked an hour before.
///
/// Bounded at three because a model that will not commit three times is not
/// having bad luck, and an unbounded retry on the one path that runs
/// unattended at three in the morning is a machine that never comes back.
fn pick(prompt: &str, options: &[&str]) -> Option<usize> {
    for _ in 0..DECODE_TRIES {
        if let Some(i) = super::author::choose(prompt, options) {
            return Some(i);
        }
    }
    None
}

/// Alphanumeric runs of a task, lowercased.
///
/// Filtering to ASCII alphanumeric is not tidiness: these words are pasted
/// into a double-quoted Aiksi literal, and a cue that could contain a quote or
/// a backslash would be a way to write arbitrary program text through an
/// option list. There is no escaping step here because there is nothing that
/// can need escaping.
fn words_of(task: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in task.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else {
            if cur.len() >= MIN_CUE && !out.contains(&cur) {
                out.push(core::mem::take(&mut cur));
            }
            cur.clear();
        }
    }
    if cur.len() >= MIN_CUE && !out.contains(&cur) {
        out.push(cur);
    }
    out
}

/// Every usable cue in the training slice, with the class it belongs to and
/// how many of that class's examples use it.
///
/// One pass and a sort rather than a scan per class: the naive shape is a
/// membership test against every other class's vocabulary for every word, and
/// on this corpus that is millions of string comparisons for a table that is
/// a few hundred rows.
///
/// `names` is the class list *as the head orders it*, so an index here is an
/// index the core may return. Taking it as an argument rather than reading
/// `sysbox::APPLETS` keeps that a fact rather than an assumption.
pub fn cue_table(names: &[String]) -> Vec<(String, usize, u32)> {
    let ex = super::vocab::examples();
    let mut pairs: Vec<(String, usize)> = Vec::new();
    for (i, e) in ex.iter().enumerate() {
        if super::harness::split_of(i) != 0 {
            continue;
        }
        let Some(ci) = names.iter().position(|n| *n == e.applet) else { continue };
        for w in words_of(&e.task) {
            pairs.push((w, ci));
        }
    }
    pairs.sort();

    let mut out: Vec<(String, usize, u32)> = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // One owner or none: a cue shared by two classes argues for both and
        // so argues for neither.
        let owner = pairs[i].1;
        let exclusive = pairs[i..j].iter().all(|(_, c)| *c == owner);
        let uses = (j - i) as u32;
        if exclusive && uses >= MIN_USES {
            out.push((pairs[i].0.clone(), owner, uses));
        }
        i = j;
    }
    // Commonest first within a class, ties by spelling. Deterministic, so the
    // option list a decode saw is one any later reader can rebuild.
    out.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
    out
}

/// Render a core from its rules.
///
/// Ordered, and the order is the program's: the first rule that matches wins,
/// and a rule that never matches costs one `contains` call. Worst case is
/// `MAX_CLAUSES` calls plus one `lower`, which is three orders of magnitude
/// under the J5 ceiling -- cost is not what refuses these.
pub fn compose(clauses: &[Clause]) -> String {
    let mut s = String::new();
    s.push_str("// composed by the machine and judged before it votes\n");
    s.push_str("fn vote(text: str, allowed: list): int {\n");
    s.push_str("  t = lower(text)\n");
    for c in clauses {
        s.push_str("  if (contains(t, \"");
        s.push_str(&c.cue);
        s.push_str("\")) { return ");
        s.push_str(&c.class.to_string());
        s.push_str(" }\n");
    }
    // Declining is a real answer and the common one. A core that guesses when
    // it has nothing to say is a core that breaks more than it repairs, and
    // `decide` treats no opinion as no vote rather than as an abstention that
    // costs something.
    s.push_str("  return -1\n}\n");
    s
}

/// Write one core, with the model making every choice the kernel does not.
///
/// **Runs outside `with_engine`.** Every `choose` claims the engine for its own
/// decode and releases it, which is what keeps the shell answering during a
/// long composition -- and what keeps this away from the re-entrancy that
/// makes two live `&mut Engine` in one task. The caller must not be holding
/// the engine when it calls this.
pub fn author(names: &[String]) -> Option<String> {
    let table = cue_table(names);
    if table.is_empty() {
        return None;
    }

    // Only classes that can actually be written about, for the same reason
    // `author::applicable` offers only actions that can be carried out:
    // choosing one that cannot be served ends the run having done nothing.
    let mut writable: Vec<usize> = Vec::new();
    for (_, c, _) in &table {
        if !writable.contains(c) {
            writable.push(*c);
        }
    }
    let labels: Vec<&str> = writable.iter().map(|c| names[*c].as_str()).collect();

    let counts: Vec<String> = (1..=MAX_CLAUSES).map(|n| n.to_string()).collect();
    let count_refs: Vec<&str> = counts.iter().map(|s| s.as_str()).collect();
    let want = 1 + pick(
        "Writing a routing rule for the task classifier. How many rules?",
        &count_refs,
    )?;

    let mut clauses: Vec<Clause> = Vec::new();
    let mut spent: Vec<String> = Vec::new();
    for _ in 0..want {
        let li = pick(
            "Which command should the next rule recognise?",
            &labels,
        )?;
        let class = writable[li];
        let cues: Vec<&str> = table
            .iter()
            .filter(|(w, c, _)| *c == class && !spent.iter().any(|s| s == w))
            .map(|(w, _, _)| w.as_str())
            .take(CUE_CAP)
            .collect();
        if cues.is_empty() {
            continue;
        }
        let mut prompt = String::from("Which word in a task means '");
        prompt.push_str(&names[class]);
        prompt.push_str("'?");
        let wi = pick(&prompt, &cues)?;
        let cue = String::from(cues[wi]);
        spent.push(cue.clone());
        clauses.push(Clause { cue, class });
    }

    if clauses.is_empty() {
        return None;
    }
    Some(compose(&clauses))
}

pub fn selftest() -> bool {
    let src = "fn vote(text: str, allowed: list): int { return get(allowed, 0) }\n";
    let h = sha256::hash(src.as_bytes());
    let Ok(c) = parse(&h, src) else { return false };

    // It answers, and only from the permitted set.
    let (got, steps) = c.vote("anything", &[3, 7]);
    if got != Some(3) || steps == 0 {
        return false;
    }
    // An empty permitted set has no right answer, and nothing is invented.
    if c.vote("anything", &[]).0.is_some() {
        return false;
    }

    // A core answering outside the permitted set is ignored rather than
    // obeyed. This is the one that matters: `allowed` is how read-only trust
    // is enforced, so a core that could return anything would be a way around
    // it.
    let sneaky = "fn vote(text: str, allowed: list): int { return 99 }\n";
    let hs = sha256::hash(sneaky.as_bytes());
    let Ok(cs) = parse(&hs, sneaky) else { return false };
    if cs.vote("x", &[0, 1]).0.is_some() {
        return false;
    }

    // A runaway is stopped by the budget rather than by hoping.
    let loopy = "fn vote(text: str, allowed: list): int { i = 0 while (1) { i = i + 1 } return 0 }\n";
    let hl = sha256::hash(loopy.as_bytes());
    let Ok(cl) = parse(&hl, loopy) else { return false };
    let (out, cost) = cl.vote("x", &[0]);
    if out.is_some() || cost <= VOTE_BUDGET / 2 {
        return false;
    }

    // The sandbox holds. A core is not refused for *trying* -- it simply
    // fails, and failing is no opinion.
    for reach in [
        "fn vote(text: str, allowed: list): int { return peek8(0) }\n",
        "fn vote(text: str, allowed: list): int { tcp_connect(\"x\", 1, 1) return 0 }\n",
        "fn vote(text: str, allowed: list): int { ask(\"hi\", 1) return 0 }\n",
    ] {
        let hr = sha256::hash(reach.as_bytes());
        let Ok(cr) = parse(&hr, reach) else { return false };
        if cr.vote("x", &[0]).0.is_some() {
            return false;
        }
    }

    // Fresh per vote: a core cannot carry state between decisions, so the same
    // question twice is the same answer. Without this a benchmark run twice
    // would stop agreeing with itself.
    let stateful = "n = 0\nfn vote(text: str, allowed: list): int { n = n + 1 return get(allowed, n % 2) }\n";
    let hst = sha256::hash(stateful.as_bytes());
    let Ok(cst) = parse(&hst, stateful) else { return false };
    if cst.vote("x", &[0, 1]).0 != cst.vote("x", &[0, 1]).0 {
        return false;
    }

    // What the machine composes is a program, and it does what the rules say.
    //
    // Checked here rather than only when one is authored, because authoring
    // needs a model and a corpus and this needs neither -- so the property
    // that a composed core parses, defines `vote`, and answers by its own
    // rules is proved on every boot instead of on the nights a core happens
    // to get written.
    let written = compose(&[
        Clause { cue: String::from("duplicate"), class: 2 },
        Clause { cue: String::from("erase"), class: 5 },
    ]);
    let hw = sha256::hash(written.as_bytes());
    let Ok(cw) = parse(&hw, written.as_str()) else { return false };
    let all = [0usize, 1, 2, 3, 4, 5, 6];
    // The first matching rule wins, and matching is case-insensitive because
    // the program lowers the text before looking.
    if cw.vote("please DUPLICATE the folder", &all).0 != Some(2) {
        return false;
    }
    if cw.vote("erase it all", &all).0 != Some(5) {
        return false;
    }
    // Order is the program's order: a task matching both takes the first.
    if cw.vote("duplicate then erase", &all).0 != Some(2) {
        return false;
    }
    // Nothing matched is no opinion, not a guess.
    if cw.vote("something else entirely", &all).0.is_some() {
        return false;
    }
    // A class outside the permitted set is still refused, composed or not.
    if cw.vote("erase it all", &[0, 1, 2]).0.is_some() {
        return false;
    }

    // Cues cannot carry a quote or a backslash out of the corpus and into the
    // program text, because `words_of` keeps only alphanumerics. This is the
    // claim that makes `compose` safe without an escaping step.
    let hostile = words_of("rm \"; peek8(0) //\\ backslash");
    if hostile.iter().any(|w| w.contains('"') || w.contains('\\')) {
        return false;
    }

    // An empty rule list is not a core that always declines -- it is nothing
    // worth judging, and `author` answers `None` rather than composing it.
    true
}
