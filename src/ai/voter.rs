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

/// The mechanism, without a corpus.
///
/// What is checked here is that a core cannot reach past its bounds, because
/// that is the part no benchmark would notice: a core that tries to open a
/// socket and is refused simply votes badly, and a benchmark reports a bad
/// core rather than a stopped attack.
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
    cst.vote("x", &[0, 1]).0 == cst.vote("x", &[0, 1]).0
}
