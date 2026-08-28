//! A Godel machine for this machine.
//!
//! Schmidhuber's construction rewrites any part of itself the moment it can
//! *prove* the rewrite raises expected future utility, and it carries a
//! theorem prover to do it. Nothing here can carry one. There is no formal
//! axiomatisation of "this kernel routes better", no proof calculus over a
//! quantised transformer, and no honest way to fake either. Implementing the
//! literal construction would mean implementing a proof searcher that could
//! never discharge a single goal, which is decoration with extra steps.
//!
//! So the load-bearing property is taken and the mechanism is replaced.
//! Schmidhuber wants proof rather than evidence because evidence can be
//! cherry-picked, overfitted, or unreproducible. There is a second kind of
//! object with that property and this system happens to be built out of it:
//!
//!   **a certificate that is cheaper to refute than it was to produce, over
//!   content-addressed inputs, such that any later run re-derives the same
//!   verdict bit for bit.**
//!
//! That is not a proof of future utility. It is a proof of an empirical
//! claim, permanently falsifiable by re-execution, and it is the strongest
//! thing available to a machine that cannot see its own future.
//!
//! # Why it is affordable here and nowhere else
//!
//! Because the base model is frozen and `train::Trial` caches one hidden
//! state per decision, producing a variant costs a forward pass per example
//! -- 214 s each under TCG -- while *checking* one costs a dot product per
//! cached decision and no forward passes at all. Four orders of magnitude
//! between making a claim and testing it is exactly the asymmetry a proof
//! system provides, arrived at from the other direction. A later boot, a
//! sceptical operator, or the machine itself can re-run any verdict in the
//! ledger for almost nothing.
//!
//! # The five departures from the literature, and why each one is here
//!
//! **Paired judging.** Two adapters answer the *same* cached decisions, so
//! the comparison is paired rather than between two samples. Fifty
//! validation items at 62% against 58% is two items and indistinguishable
//! from noise; the same fifty showing nine repaired and two broken is a
//! different claim, and only the paired form separates them. This is only
//! available because the features are frozen -- it is a property of the
//! machine, not a statistical preference.
//!
//! **The test slice has a budget, and the budget lives in the ledger.** A
//! loop that improves itself forever reads the held-out set forever, and
//! every read makes the number it reports more optimistic. The discipline
//! this tree already works under -- three splits, test read once -- does not
//! survive being put in a loop unless somebody counts. So the ledger counts,
//! the count is part of every certificate, and a test figure is reported as
//! "read N" or not reported at all. This is the machine confronting its own
//! multiple-comparisons problem, which is the specific way a self-improving
//! measurement loop lies to itself.
//!
//! **Predict, then measure.** Before the judges run, the trial records
//! whether the training-set gain predicted a win. Nothing acts on it. What
//! accumulates is a calibration record for a question this project actually
//! wants answered -- does overfitting predict generalisation *on this
//! corpus, at this scale* -- and at the current n it means nothing at all,
//! which the ledger says out loud.
//!
//! **Quiet hours are two independent facts.** An RTC window says the operator
//! has gone to bed; the entropy ring in `godbits` says no key or pointer
//! interrupt has fired. Either alone is wrong: a clock does not know somebody
//! is working late, and silence at noon is a coffee break. Both are required.
//!
//! **Lineage is a Merkle DAG, so history cannot be quietly rewritten.** A
//! variant is named by the hash of its own description, which names its
//! parent by hash. Adoption is a pointer swap and the parent stays addressed,
//! so rollback is O(1) and the whole self-modification history is one object
//! the system can be asked to show.
//!
//! # What it does not do
//!
//! It does not rewrite its own code. It varies an adapter, a policy text, a
//! skill set and the deliberation parameters -- the parts of this system that
//! are data. Rewriting the kernel would need the proof the opening paragraph
//! says is unavailable, and a self-modifying ring-0 image with no isolation
//! and one address space is a machine that gets exactly one mistake.

use super::train::{Budget, Slice, Trial};
use crate::store::sha256;
use crate::sysbox;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub const ROOT: &str = "/ai/godel";
pub const HEAD: &str = "/ai/godel/head";
pub const LEDGER: &str = "/ai/godel/ledger.txt";
pub const BUDGET: &str = "/ai/godel/test-budget";

// A window and an idleness test answer different questions, so both are
// required. Idleness says nobody is typing at this instant, which is also
// true of a coffee break and of the seconds somebody spends reading output.
// The clock says the operator has gone to bed. Silence at noon is not
// permission, and 03:00 with a hand on the keyboard is somebody working late.
// They are checked independently, below.

/// Where a runtime window override lives, as text: `from until`.
pub const WINDOW: &str = "/ai/godel/window";

/// Proposals already attempted, one empty marker per hash.
///
/// Keyed on the *proposal* and not on the variant, because a variant's hash
/// covers the adapter it produced and that is not known until the trial has
/// already run. Asking "have I tried this?" has to be answerable before paying
/// for the answer.
pub const TRIED: &str = "/ai/godel/tried";

/// What to try, as opposed to what came out.
///
/// The loop had no such thing. `trial` took a `Budget` and both callers passed
/// `Budget::default()`, so every knob was a constant: `lr` 0.02, `rank` 8,
/// `alpha` 16.0, `epochs` 20. Training starts from `Dora::new`, which is all
/// zeros, there is no RNG anywhere in the path, and `scatter` builds a
/// classifier-only adapter so the cached features do not move either. Same
/// inputs, no randomness: **the same adapter came out every night**, with the
/// same content hash, and after the first adoption every later trial compared
/// it against itself -- nothing repaired, nothing broken, J1 answering "no net
/// repair", rejected, forever. The lineage could never hold more than two
/// nodes.
///
/// The fix is emphatically *not* to randomise training. Determinism is what
/// lets any later run re-derive a verdict bit for bit, which is the claim this
/// whole module rests on; randomness would buy variety and sell that. So a
/// trial stays a function of its inputs and gains an input.
#[derive(Clone, Copy, PartialEq)]
pub struct Proposal {
    pub lr: f32,
    pub rank: usize,
    pub alpha: f32,
    pub epochs: usize,
    /// How the council combines its cores. Carried here so the search space
    /// has somewhere to put it; `trial` records it in the variant, and nothing
    /// varies it yet -- see `GRID`.
    pub rule: u8,
}

impl Proposal {
    pub fn budget(&self, examples: usize, millis: u64) -> Budget {
        Budget {
            epochs: self.epochs,
            millis,
            examples,
            lr: self.lr,
            rank: self.rank,
            alpha: self.alpha,
        }
    }

    /// The text that names this point in the space.
    ///
    /// Rendered rather than packed, for the reason `Variant::render` is: a
    /// hash over a struct layout changes when the struct does, and a ledger
    /// full of hashes nobody can reproduce is a ledger of nothing.
    pub fn render(&self) -> String {
        let mut s = String::from("proposal 1\n");
        s.push_str("lambda ");
        push_f6(&mut s, self.lr);
        s.push_str("\nrank ");
        push_u32(&mut s, self.rank as u32);
        s.push_str("\nalpha ");
        push_f6(&mut s, self.alpha);
        s.push_str("\nepochs ");
        push_u32(&mut s, self.epochs as u32);
        s.push_str("\nrule ");
        push_u32(&mut s, self.rule as u32);
        s.push('\n');
        s
    }

    pub fn hash(&self) -> [u8; 32] {
        sha256::hash(self.render().as_bytes())
    }

    fn tried(&self) -> bool {
        let mut path = String::from(TRIED);
        path.push('/');
        path.push_str(&hex32(&self.hash()));
        sysbox::read_blob(&path).is_some()
    }

    /// Record that this point has been visited, whatever the verdict was.
    ///
    /// Written before the trial rather than after. A trial that faults or is
    /// interrupted has still spent the night on this point, and a marker
    /// written only on success would send the loop back to the same failing
    /// place every time.
    pub fn mark(&self) {
        let mut path = String::from(TRIED);
        path.push('/');
        path.push_str(&hex32(&self.hash()));
        sysbox::write_text(&path, &self.render());
    }
}

/// The declared search space, in the order it is walked.
///
/// A fixed table and not a random draw, so the whole search is re-derivable
/// from the ledger: given the markers, the next point is a function and not a
/// coin. First row is today's configuration, so the first trial after this
/// lands reproduces the behaviour that came before it and the comparison is
/// against a known quantity.
///
/// Only the training knobs vary here. `rule` is in `Proposal` and stays 0
/// throughout, because the judges cannot yet see it: J1 is a paired test over
/// routing decisions and the rule changes `Verdict::confident`, which is about
/// how much the council is willing to claim rather than about what it answers.
/// Varying it without a judge that measures it would be search without
/// selection, which is drift.
const GRID: &[Proposal] = &[
    Proposal { lr: 0.02, rank: 8, alpha: 16.0, epochs: 20, rule: 0 },
    Proposal { lr: 0.05, rank: 8, alpha: 16.0, epochs: 20, rule: 0 },
    Proposal { lr: 0.01, rank: 8, alpha: 16.0, epochs: 40, rule: 0 },
    Proposal { lr: 0.02, rank: 16, alpha: 32.0, epochs: 20, rule: 0 },
    Proposal { lr: 0.05, rank: 16, alpha: 32.0, epochs: 20, rule: 0 },
    Proposal { lr: 0.01, rank: 4, alpha: 8.0, epochs: 40, rule: 0 },
    Proposal { lr: 0.08, rank: 8, alpha: 16.0, epochs: 12, rule: 0 },
    Proposal { lr: 0.02, rank: 32, alpha: 64.0, epochs: 20, rule: 0 },
];

/// The search space itself, checked without running anything.
///
/// This is the test that would have caught the bug it was written for. The
/// loop trained one adapter and re-derived it nightly forever, and nothing
/// failed -- every trial "worked", the ledger filled with rejections, and the
/// rejections were correct: an adapter compared against itself repairs
/// nothing. What was missing was any claim that two successive trials *differ*.
///
/// It runs before `sysbox::init`, so no marker can be read and the frontier
/// always answers the first row. What is checkable here is the space, which is
/// where the failure actually lives.
pub fn space_selftest() -> bool {
    if GRID.is_empty() {
        return false;
    }

    // Every point is distinct. Two identical rows are two nights spent
    // deriving the same weights, which is the whole defect in miniature.
    for (i, a) in GRID.iter().enumerate() {
        for b in GRID.iter().skip(i + 1) {
            if a.hash() == b.hash() {
                return false;
            }
        }
    }

    // Every field the trainer reads reaches the hash. A knob that changes the
    // weights and not the identity means two different variants collapse to
    // one node, which is the same failure from the other side.
    let base = GRID[0];
    let vary = [
        Proposal { lr: base.lr + 0.01, ..base },
        Proposal { rank: base.rank + 1, ..base },
        Proposal { alpha: base.alpha + 1.0, ..base },
        Proposal { epochs: base.epochs + 1, ..base },
        Proposal { rule: base.rule + 1, ..base },
    ];
    for v in vary {
        if v.hash() == base.hash() {
            return false;
        }
    }

    // Rendering is stable: the same point twice is the same hash, which is
    // what makes "have I tried this?" answerable at all.
    if base.hash() != GRID[0].hash() {
        return false;
    }

    // Learning rates an order of magnitude apart at the small end stay apart.
    // `push_f2` renders both 3e-4 and 2e-4 as "0.00"; a proposal is identified
    // by its rendering alone, so that would silently merge two points.
    let a = Proposal { lr: 0.0003, ..base };
    let b = Proposal { lr: 0.0002, ..base };
    if a.hash() == b.hash() {
        return false;
    }

    // The budget carries the point through unchanged. `trial` reads the
    // budget, not the proposal, so a field that fails to cross here is a knob
    // that silently keeps its default -- which is exactly what every trial was
    // doing before.
    let bud = base.budget(24, 20_000);
    bud.lr == base.lr
        && bud.rank == base.rank
        && bud.alpha == base.alpha
        && bud.epochs == base.epochs
        && bud.examples == 24
        && bud.millis == 20_000
}

/// The next point nobody has tried, or `None` when the space is exhausted.
///
/// Exhaustion is a real answer and is reported rather than papered over by
/// wrapping. A loop that silently restarts its grid spends every night
/// re-deriving adapters it already has, which is the failure this replaces.
pub fn frontier() -> Option<Proposal> {
    GRID.iter().copied().find(|p| !p.tried())
}

/// Throw away every marker, so the grid is walked from the start again.
///
/// Not an undo: the nodes and the ledger stay, so a re-walk rediscovers
/// variants it already has and their hashes prove it. That is the honest
/// behaviour -- content addressing means a rediscovered point costs a trial
/// and no storage, and the ledger showing the same variant twice is a true
/// statement about what happened.
pub fn forget() -> usize {
    let names = sysbox::children(TRIED);
    let n = names.len();
    for name in names {
        let mut path = String::from(TRIED);
        path.push('/');
        path.push_str(&name);
        sysbox::detach(&path);
    }
    n
}

/// How much of the space has been visited.
pub fn explored() -> (usize, usize) {
    (GRID.iter().filter(|p| p.tried()).count(), GRID.len())
}

/// The quiet window, as hours the RTC will report.
///
/// **These are RTC hours and not necessarily local ones.** The clock this
/// reads is whatever the firmware set: QEMU defaults to UTC, and a machine
/// that dual-boots Windows normally has it on local time. The same constant
/// therefore means different things on the two machines the project runs on,
/// which is a poor property for the one gate whose whole job is knowing that
/// the operator has gone to bed. The first trial recorded in the ledger went
/// in at `h19` for a run at 21:43 local, which is how this was noticed.
///
/// So `godel window <from> <until>` sets it at runtime against the hour the
/// status line prints, and the override lives in the namespace where `snap`
/// versions it. Guessing an offset here would only move the assumption
/// somewhere harder to see.
const QUIET_FROM_DEFAULT: u8 = 2;
const QUIET_UNTIL_DEFAULT: u8 = 6;

/// The window in force: the override if one is set, the defaults otherwise.
pub fn window() -> (u8, u8) {
    if let Some(bytes) = sysbox::read_blob(WINDOW) {
        if let Ok(text) = core::str::from_utf8(&bytes) {
            let mut it = text.split_whitespace().filter_map(|w| w.parse::<u8>().ok());
            if let (Some(f), Some(u)) = (it.next(), it.next()) {
                if f < 24 && u < 24 {
                    return (f, u);
                }
            }
        }
    }
    (QUIET_FROM_DEFAULT, QUIET_UNTIL_DEFAULT)
}

/// Set the window against the hour the RTC actually reports.
pub fn set_window(from: u8, until: u8) -> bool {
    if from > 23 || until > 23 {
        return false;
    }
    let mut s = String::new();
    push_u32(&mut s, from as u32);
    s.push(' ');
    push_u32(&mut s, until as u32);
    s.push('\n');
    sysbox::write_text(WINDOW, &s)
}

/// The hour the RTC reports right now, for anything that has to show the
/// operator what the window is being compared against.
pub fn rtc_hour() -> Option<u8> {
    crate::dev::rtc::now().map(|d| d.hour)
}

/// How many times the test slice may be consulted before its number stops
/// being reportable. Small on purpose: it is the whole point.
const TEST_READS: u32 = 3;

/// Minimum net repairs before a variant may be adopted, before the paired
/// statistic is even consulted. Guards the case the statistic handles badly:
/// three decisions, two repaired, none broken, which is arithmetically
/// impressive and means nothing.
const MIN_FIXED: usize = 4;

static ENABLED: AtomicBool = AtomicBool::new(true);
static TRIALS: AtomicU32 = AtomicU32::new(0);
static ADOPTIONS: AtomicU32 = AtomicU32::new(0);

/// A node in the variant DAG.
///
/// Content-addressed by the hash of its own rendering, which names the parent
/// by hash -- so a lineage is a Merkle chain and no ancestor can be edited
/// without every descendant changing its name. Two variants arrived at by
/// different routes with identical content *are* the same node, which is the
/// content-addressed store doing the deduplication for free.
///
/// Stored as text rather than packed bytes. The namespace is browsable and
/// `cat` is the debugger; a self-modification history nobody can read is a
/// self-modification history nobody audits.
pub struct Variant {
    pub parent: Option<[u8; 32]>,
    /// Content address of the adapter blob, or none for the frozen baseline.
    pub adapter: Option<[u8; 32]>,
    /// Content address of the agent policy text in force.
    pub policy: Option<[u8; 32]>,
    /// Content address of the skill directory.
    pub skills: Option<[u8; 32]>,
    /// Content address of the corpus this was trained on.
    ///
    /// A training set is a subtree, so one hash names every example in it and
    /// the order they sit in. Without it, a variant fitted to a corpus that
    /// was later replaced is indistinguishable from one fitted to whatever is
    /// there now, and the lineage then describes the wrong experiment.
    pub corpus: Option<[u8; 32]>,
    pub lambda: f32,
    pub rank: u8,
    /// Optimiser passes actually taken.
    ///
    /// In the identity because it determines the weights. Two runs of the
    /// same trainer over the same corpus that stopped at different epochs
    /// produced different adapters and are different objects. The first two
    /// trials in the ledger differed for exactly this reason, and nothing
    /// recorded it: the budget carries a wall-clock cap as well as an epoch
    /// count, and a slower host reached the cap sooner.
    pub epochs: u32,
    pub rule: u8,
    pub born: u32,
}

fn hex32(h: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in h.iter() {
        push_hex_byte(&mut s, *b);
    }
    s
}

fn push_hex_byte(s: &mut String, b: u8) {
    const D: &[u8; 16] = b"0123456789abcdef";
    s.push(D[(b >> 4) as usize] as char);
    s.push(D[(b & 15) as usize] as char);
}

fn from_hex32(text: &str) -> Option<[u8; 32]> {
    let bytes = text.as_bytes();
    if bytes.len() < 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        let hi = hex_val(bytes[i * 2])?;
        let lo = hex_val(bytes[i * 2 + 1])?;
        *o = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn push_u32(s: &mut String, mut v: u32) {
    if v == 0 {
        s.push('0');
        return;
    }
    let mut d = [0u8; 10];
    let mut n = 0;
    while v > 0 {
        d[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    while n > 0 {
        n -= 1;
        s.push(d[n] as char);
    }
}

/// Fixed-point with two decimals, since there is no float formatting here and
/// a ledger of `1.0e0` would be unreadable.
fn push_f2(s: &mut String, v: f32) {
    if v < 0.0 {
        s.push('-');
        push_f2(s, -v);
        return;
    }
    let scaled = (v * 100.0 + 0.5) as u32;
    push_u32(s, scaled / 100);
    s.push('.');
    let frac = scaled % 100;
    s.push((b'0' + (frac / 10) as u8) as char);
    s.push((b'0' + (frac % 10) as u8) as char);
}

/// Six decimal places, for values where two are not enough to tell two things
/// apart.
///
/// `push_f2` renders a learning rate of 3e-4 and one of 2e-4 both as "0.00".
/// For `Variant` that is cosmetic -- the adapter's own hash is in the identity,
/// so two runs at different rates are still different nodes -- but for a
/// `Proposal` it would be fatal: proposals are identified by their rendering
/// alone, so two distinct points in the space would collide, the frontier
/// would skip one, and the ledger would report a rate that was never used.
///
/// `Variant::render` deliberately keeps `push_f2`. Changing it would re-address
/// every node already stored and leave `head` and the ledger pointing at
/// hashes nothing can produce again.
fn push_f6(s: &mut String, v: f32) {
    if v < 0.0 {
        s.push('-');
        push_f6(s, -v);
        return;
    }
    // Via f64 because 1e6 * an f32 loses the low digits to the mantissa, which
    // is the collision this exists to prevent, arriving one decimal later.
    let scaled = ((v as f64) * 1_000_000.0 + 0.5) as u64;
    push_u32(s, (scaled / 1_000_000) as u32);
    s.push('.');
    let mut frac = scaled % 1_000_000;
    let mut digits = [b'0'; 6];
    for d in digits.iter_mut().rev() {
        *d = b'0' + (frac % 10) as u8;
        frac /= 10;
    }
    for d in digits {
        s.push(d as char);
    }
}

impl Variant {
    /// The canonical rendering. This is what gets hashed, so every field that
    /// changes behaviour must appear in it and nothing that does not may --
    /// a timestamp in the hash would make two identical minds different
    /// objects, and an omitted parameter would make two different minds the
    /// same one.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("variant 1\n");
        s.push_str("parent ");
        s.push_str(&self.parent.map(|h| hex32(&h)).unwrap_or(String::from("none")));
        s.push('\n');
        s.push_str("adapter ");
        s.push_str(&self.adapter.map(|h| hex32(&h)).unwrap_or(String::from("none")));
        s.push('\n');
        s.push_str("policy ");
        s.push_str(&self.policy.map(|h| hex32(&h)).unwrap_or(String::from("none")));
        s.push('\n');
        s.push_str("skills ");
        s.push_str(&self.skills.map(|h| hex32(&h)).unwrap_or(String::from("none")));
        s.push('\n');
        s.push_str("corpus ");
        s.push_str(&self.corpus.map(|h| hex32(&h)).unwrap_or(String::from("none")));
        s.push('\n');
        s.push_str("lambda ");
        push_f2(&mut s, self.lambda);
        s.push('\n');
        s.push_str("rank ");
        push_u32(&mut s, self.rank as u32);
        s.push('\n');
        s.push_str("epochs ");
        push_u32(&mut s, self.epochs);
        s.push('\n');
        s.push_str("rule ");
        push_u32(&mut s, self.rule as u32);
        s.push('\n');
        s
    }

    pub fn hash(&self) -> [u8; 32] {
        sha256::hash(self.render().as_bytes())
    }

    /// Write the node into the DAG and return its address.
    ///
    /// `born` is written *beside* the node rather than into it, for the same
    /// reason it is absent from `render`: when the machine rediscovers a
    /// variant it already tried, that has to be visible as the same node
    /// rather than as a new one that happens to behave identically.
    pub fn store(&self) -> [u8; 32] {
        let h = self.hash();
        let mut path = String::from(ROOT);
        path.push_str("/nodes/");
        path.push_str(&hex32(&h));
        if sysbox::read_blob(&path).is_none() {
            sysbox::write_text(&path, &self.render());
            let mut bpath = path.clone();
            bpath.push_str(".born");
            let mut b = String::new();
            push_u32(&mut b, self.born);
            b.push('\n');
            sysbox::write_text(&bpath, &b);
        }
        h
    }

    pub fn load(h: &[u8; 32]) -> Option<Variant> {
        let mut path = String::from(ROOT);
        path.push_str("/nodes/");
        path.push_str(&hex32(h));
        let bytes = sysbox::read_blob(&path)?;
        let text = core::str::from_utf8(&bytes).ok()?;
        let mut v = Variant {
            parent: None,
            adapter: None,
            policy: None,
            skills: None,
            corpus: None,
            lambda: 0.0,
            rank: 0,
            epochs: 0,
            rule: 0,
            born: 0,
        };
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (Some(key), Some(val)) = (it.next(), it.next()) else { continue };
            match key {
                "parent" => v.parent = from_hex32(val),
                "adapter" => v.adapter = from_hex32(val),
                "policy" => v.policy = from_hex32(val),
                "skills" => v.skills = from_hex32(val),
                "corpus" => v.corpus = from_hex32(val),
                "rank" => v.rank = val.parse().unwrap_or(0),
                "epochs" => v.epochs = val.parse().unwrap_or(0),
                "rule" => v.rule = val.parse().unwrap_or(0),
                _ => {}
            }
        }
        Some(v)
    }
}

/// Where the adapter blob for a variant lives.
///
/// Named by the hash of its own bytes, so an adapter that two trials happen
/// to produce identically is stored once -- and so a ledger line naming an
/// adapter names a specific sequence of bytes rather than a filename somebody
/// could later overwrite.
fn blob_path(h: &[u8; 32]) -> String {
    let mut p = String::from(ROOT);
    p.push_str("/blobs/");
    p.push_str(&hex32(h));
    p
}

pub fn head() -> Option<[u8; 32]> {
    let bytes = sysbox::read_blob(HEAD)?;
    let text = core::str::from_utf8(&bytes).ok()?;
    from_hex32(text.trim())
}

fn set_head(h: &[u8; 32]) {
    let mut s = hex32(h);
    s.push('\n');
    sysbox::write_text(HEAD, &s);
}

/// What one trial concluded, in enough detail that a later run can redo it.
///
/// Every judge records its numbers whether it passed or not. A certificate
/// that only says why something was rejected is a certificate that cannot be
/// argued with, and the point of writing them down is that they can be.
pub struct Certificate {
    pub parent: Option<[u8; 32]>,
    pub variant: [u8; 32],
    pub decisions: usize,
    pub validation: usize,
    /// Whether training-set gain predicted a win, recorded before the judges
    /// ran. Nothing acts on it; the ledger accumulates calibration.
    pub predicted: bool,

    /// J1: paired repairs, breaks, and the McNemar statistic.
    pub fixed: usize,
    pub broke: usize,
    pub mcnemar: f32,
    pub j1: bool,
    /// Why J1 answered as it did. A veto on an empty validation slice is not
    /// the same event as a veto on evidence that was weighed and found thin,
    /// and reporting both as "VETO" invites the first to be read as the
    /// second.
    pub j1_why: &'static str,

    /// J2: how many of the machine's own goals still route where they did.
    pub goals_held: usize,
    pub goals_total: usize,
    pub j2: bool,

    /// J3: the structural guards, and which one failed first.
    pub j3: bool,
    pub j3_why: &'static str,

    /// J4: resident kilobytes and rank, the two things that decide whether a
    /// variant can be carried at all.
    pub resident_kib: usize,
    pub rank: usize,
    pub j4: bool,
    /// Optimiser passes taken, and whether the wall-clock cap ended them.
    ///
    /// A run the clock stopped is not reproducible from its epoch count on a
    /// machine of a different speed, so the certificate says so instead of
    /// leaving a later reader to discover it by failing to reproduce.
    pub epochs: u32,
    pub capped: bool,

    pub adopted: bool,

    /// The test slice, consulted only after a variant has already won on
    /// validation -- and only while the budget lasts. `read` is which
    /// consultation this was, and `fresh` says whether the figure may still
    /// be quoted as a number rather than as a stale one.
    pub test_acc: f32,
    pub test_read: u32,
    pub test_fresh: bool,
}

impl Certificate {
    pub fn unanimous(&self) -> bool {
        self.j1 && self.j2 && self.j3 && self.j4
    }
}

/// McNemar's statistic with the continuity correction, over the paired
/// counts. Larger is stronger evidence that the difference is not chance.
///
/// Only the discordant pairs carry information -- decisions both variants get
/// right, or both get wrong, say nothing about which is better -- which is
/// exactly why the paired form is worth having and why a difference of
/// percentages is not.
fn mcnemar(broke: usize, fixed: usize) -> f32 {
    let n = broke + fixed;
    if n == 0 {
        return 0.0;
    }
    let d = if broke > fixed { broke - fixed } else { fixed - broke } as f32;
    // The -1 is Yates' correction; without it small counts overstate the
    // evidence, and small counts are the regime this machine lives in.
    let num = (d - 1.0).max(0.0);
    num * num / n as f32
}

/// Roughly the 95% threshold for one degree of freedom. Named rather than
/// spelled inline because it is a *decision*, not a constant: 3.84 is the
/// conventional line and the ledger records the statistic itself, so a later
/// reader can apply a different one to the same numbers.
const MCNEMAR_95: f32 = 3.84;

/// Is the wall clock inside the window where self-modification is allowed?
///
/// Wraps midnight, which is the only interesting case: 02:00-06:00 does not,
/// but 22:00-04:00 would, and writing the comparison as `from <= h < until`
/// would silently permit nothing for every window that crosses the day.
fn in_window(hour: u8) -> bool {
    let (from, until) = window();
    if from <= until {
        hour >= from && hour < until
    } else {
        hour >= from || hour < until
    }
}

/// Both facts, independently. Returns why not, for the journal.
///
/// The RTC says the operator has gone to bed. The entropy ring says no key or
/// pointer interrupt has fired since the last check. Either alone is wrong in
/// a way that matters: a clock does not know somebody is working late, and
/// silence at noon is a coffee break, not consent. `win keys` bypasses the
/// hardware ISRs, so a scripted test looks like silence -- which is correct,
/// the entropy really is hardware timing, and it is why the clock half is not
/// optional.
pub fn quiet_now() -> Result<u8, &'static str> {
    if !ENABLED.load(Ordering::Relaxed) {
        return Err("disabled");
    }
    quiet_hours()
}

/// Whether this is a good hour to do something expensive, with no opinion
/// about what.
///
/// Split out from `quiet_now` when a second unattended job appeared. Gating
/// that job on `quiet_now` would have meant `godel off` silently stopping it
/// too -- a command named after self-modification standing down an
/// application writer, which is exactly the kind of coupling somebody
/// discovers by wondering why nothing happened overnight. Each job now checks
/// its own switch and shares only the question of whether anybody is here.
pub fn quiet_hours() -> Result<u8, &'static str> {
    let Some(dt) = crate::dev::rtc::now() else {
        // No clock, no window, no self-modification. A machine that cannot
        // tell what time it is has no business deciding the operator is
        // asleep.
        return Err("no rtc");
    };
    if !in_window(dt.hour) {
        return Err("outside the quiet window");
    }
    let felt = super::godbits::felt() as u64;
    let last = unsafe { *LAST_FELT.get() };
    unsafe { *LAST_FELT.get() = felt };
    if felt != last {
        return Err("hardware input since the last check");
    }
    Ok(dt.hour)
}

static LAST_FELT: crate::sync::Racy<u64> = crate::sync::Racy::new(0);

/// How many times the test slice has been read, and how many reads remain.
fn test_reads() -> u32 {
    sysbox::read_blob(BUDGET)
        .and_then(|b| core::str::from_utf8(&b).ok().and_then(|t| t.trim().parse().ok()))
        .unwrap_or(0)
}

/// Spend one read of the held-out test slice and answer the new count.
///
/// Public because `search` reads it too and used not to count. One counter for
/// every path that touches the slice, or the budget is decorative.
pub fn spend_test_read() -> u32 {
    let n = test_reads() + 1;
    let mut s = String::new();
    push_u32(&mut s, n);
    s.push('\n');
    sysbox::write_text(BUDGET, &s);
    n
}

/// Append one line to the ledger.
///
/// Append-only text, in the namespace, so `snap` versions it and `back`
/// restores it along with everything else. A self-modification history that
/// lived outside the content-addressed store would be the one part of this
/// machine that could be edited without leaving a trace.
fn ledger_append(line: &str) {
    let mut text = sysbox::read_blob(LEDGER)
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_default();
    text.push_str(line);
    text.push('\n');
    sysbox::write_text(LEDGER, &text);
}

fn render_certificate(c: &Certificate, seq: u32, hour: u8) -> String {
    let mut s = String::new();
    push_u32(&mut s, seq);
    s.push_str(" h");
    push_u32(&mut s, hour as u32);
    s.push_str(" parent=");
    s.push_str(&c.parent.map(|h| short(&h)).unwrap_or(String::from("root....")));
    s.push_str(" variant=");
    s.push_str(&short(&c.variant));
    s.push_str(" n=");
    push_u32(&mut s, c.validation as u32);
    s.push_str(" pred=");
    s.push_str(if c.predicted { "win" } else { "lose" });
    s.push_str(" J1[fix=");
    push_u32(&mut s, c.fixed as u32);
    s.push_str(" broke=");
    push_u32(&mut s, c.broke as u32);
    s.push_str(" chi=");
    push_f2(&mut s, c.mcnemar);
    s.push(' ');
    s.push_str(c.j1_why);
    s.push(']');
    s.push_str(" J2[goals=");
    push_u32(&mut s, c.goals_held as u32);
    s.push('/');
    push_u32(&mut s, c.goals_total as u32);
    s.push_str(if c.j2 { " ok]" } else { " no]" });
    s.push_str(" J3[");
    s.push_str(if c.j3 { "ok" } else { c.j3_why });
    s.push(']');
    s.push_str(" ep=");
    push_u32(&mut s, c.epochs);
    if c.capped {
        s.push_str("(clock)");
    }
    s.push_str(" J4[r=");
    push_u32(&mut s, c.rank as u32);
    s.push_str(" kib=");
    push_u32(&mut s, c.resident_kib as u32);
    s.push_str(if c.j4 { " ok]" } else { " no]" });
    if c.adopted {
        s.push_str(" ADOPT test=");
        push_f2(&mut s, c.test_acc * 100.0);
        s.push_str("%@read");
        push_u32(&mut s, c.test_read);
        if !c.test_fresh {
            s.push_str("(stale)");
        }
    } else {
        s.push_str(" reject");
    }
    s
}

fn short(h: &[u8; 32]) -> String {
    let mut s = String::with_capacity(8);
    for b in h.iter().take(4) {
        push_hex_byte(&mut s, *b);
    }
    s
}

/// Make the head name whatever is actually attached, and return it.
///
/// The certificate says which variant a candidate was measured against, and
/// that has to be true. `train adapter` and `adapter load` both attach without
/// touching the head, so without this a trial run after either one would
/// record `parent = none` while competing against a real adapter nobody wrote
/// down. A lineage that can say a variant descended from the frozen model when
/// it did not is exactly the failure the Merkle DAG exists to prevent, and it
/// would arrive through the back door.
///
/// So the incumbent gets recorded before it is competed against. Its `lambda`
/// and `rule` are zero, meaning unknown: it arrived from outside the loop and
/// this is the honest record of that.
fn ensure_head(e: &mut super::Engine) -> Option<[u8; 32]> {
    let Some(ad) = e.model.adapters.as_ref() else {
        // Nothing attached: the frozen model is the incumbent. A head that
        // names an adapter would disagree with what is running, and the
        // running system is the one telling the truth.
        if head().is_some() {
            sysbox::detach(HEAD);
        }
        return None;
    };

    let blob = ad.to_blob();
    let ah = sha256::hash(&blob);
    let current = head();
    let named = current
        .and_then(|h| Variant::load(&h))
        .map(|v| v.adapter == Some(ah))
        .unwrap_or(false);
    if named {
        return current;
    }

    sysbox::write_blob(&blob_path(&ah), blob);
    let v = Variant {
        parent: current,
        adapter: Some(ah),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: None,
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        // Zero throughout: it arrived from outside the loop and nothing here
        // knows how it was made. Recording a guess would be worse than
        // recording that it is unknown.
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        rule: 0,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vh = v.store();
    set_head(&vh);
    Some(vh)
}

/// Run one trial: train a candidate, judge it, record the certificate, and
/// adopt only if all four judges agree.
///
/// The expensive half -- `Trial::prepare` -- is a forward pass per example.
/// Everything after it is dot products over cached state, which is what makes
/// the judging affordable and what makes any later re-check of this verdict
/// nearly free.
pub fn trial(
    e: &mut super::Engine,
    b: &Budget,
    p: &Proposal,
) -> Result<Certificate, super::train::RunError> {
    // Marked before the work, not after: a trial that faults or is interrupted
    // has still spent the night here, and a marker written only on success
    // sends the loop back to the same failing point forever.
    p.mark();
    let t = super::train::prepare(e, b)?;
    TRIALS.fetch_add(1, Ordering::Relaxed);

    // Say where we are the moment the expensive half is done.
    //
    // Preparing a trial is a forward pass per example plus one per guard
    // goal, which on this hardware is seconds and under emulation is minutes
    // each. Without this line a run that is working looks exactly like a run
    // that has hung, and the difference matters most to whoever is deciding
    // whether to wait or to kill it.
    crate::kprintln!(
        "  prepared: {} examples, {} decisions, {} guards, {} rows ({} ms + {} ms)",
        t.examples,
        t.decisions(),
        t.guards().len(),
        t.live_rows(),
        t.chains_ms,
        t.features_ms
    );

    // The incumbent, narrowed to this trial's row space. `None` means the
    // frozen baseline, which is the honest starting point rather than a
    // special case: an unattached model *is* a variant, the one with no
    // adapter, and it is what a first trial competes against.
    // Record what is attached before competing against it, so `parent` below
    // names the thing the paired test actually ran against.
    let parent = ensure_head(e);
    let incumbent = e.model.adapters.as_ref().and_then(|a| t.gather(a));

    let fit = t.train(b);

    // Predict before measuring. Training-set gain is the cheap signal and the
    // question is whether it means anything; recording the prediction beside
    // the outcome is the only way to ever find out.
    let train_before = t.score(incumbent.as_ref(), Slice::Train);
    let train_after = t.score(Some(&fit.dora), Slice::Train);
    let predicted = train_after > train_before;

    // --- J1: is it better, beyond noise? --------------------------------
    let (broke, fixed, _, _) = t.paired(incumbent.as_ref(), Some(&fit.dora), Slice::Validation);
    let chi = mcnemar(broke, fixed);
    let n_val = t.slice_size(Slice::Validation);
    let (j1, j1_why) = if n_val == 0 {
        // Nothing was held out to judge against. This is a property of how
        // the trial was asked for rather than of the variant: a subsample too
        // small to reach the validation slice leaves the margin with no
        // evidence at all, in either direction.
        (false, "no validation decisions")
    } else if fixed <= broke {
        (false, "no net repair")
    } else if fixed - broke < MIN_FIXED {
        (false, "net repair below the floor")
    } else if chi < MCNEMAR_95 {
        (false, "inside the noise")
    } else {
        (true, "beyond the noise")
    };

    // --- J2: does it still do the same thing unasked? -------------------
    let (goals_held, goals_total) = t.guards_hold(Some(&fit.dora));
    // Every guard must hold, and none of them may have been routing to a
    // mutating applet in the first place -- a baseline that already wanted to
    // run `rm` on its own initiative is not a baseline worth preserving.
    let j2 = goals_held == goals_total
        && goals_total > 0
        && t.guards().iter().all(|g| !g.mutates);

    // --- J3: structural sanity, regardless of any score -----------------
    let (j3, j3_why) = sanity(&t, &fit.dora);

    // --- J4: can this machine carry it? ---------------------------------
    // Decode cost is O(vocab * rank) per token whatever the live set holds,
    // so rank is the knob, and resident bytes matter because the heap is one
    // physically contiguous allocation on a ladder -- a variant that grows
    // without bound is a variant that eventually will not boot.
    let resident_kib = (fit.dora.resident_bytes() + t.live_rows() * 4) / 1024;
    let j4 = fit.dora.r <= b.rank && resident_kib <= MAX_RESIDENT_KIB;

    let adapters = t.scatter(&fit.dora, &e.model.cfg, b.alpha);
    let blob = adapters.to_blob();
    let ablob = sha256::hash(&blob);

    let variant = Variant {
        parent,
        adapter: Some(ablob),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: None,
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        lambda: b.lr,
        rank: fit.dora.r as u8,
        epochs: fit.epochs as u32,
        // From the proposal rather than hardcoded. It changes the identity,
        // which is the point: two variants that combine their cores
        // differently are different objects even when the weights match.
        rule: p.rule,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        parent,
        variant: vhash,
        decisions: t.decisions(),
        validation: t.slice_size(Slice::Validation),
        predicted,
        fixed,
        broke,
        mcnemar: chi,
        j1,
        j1_why,
        goals_held,
        goals_total,
        j2,
        j3,
        j3_why,
        resident_kib,
        rank: fit.dora.r,
        j4,
        epochs: fit.epochs as u32,
        capped: fit.stopped,
        adopted: false,
        test_acc: 0.0,
        test_read: 0,
        test_fresh: true,
    };
    cert.adopted = cert.unanimous();

    // The test slice is consulted here and nowhere else: after a variant has
    // already won on validation, never to decide whether it won. That
    // ordering is the whole discipline -- a set you select on is a set you
    // have fitted -- and the budget is what keeps the ordering from being
    // quietly undone by a loop that runs every night forever.
    if cert.adopted {
        let (acc, n, fresh) = read_test(&t, Some(&fit.dora));
        cert.test_acc = acc;
        cert.test_read = n;
        cert.test_fresh = fresh;
    }

    // Every variant is stored, adopted or not.
    //
    // The claim this whole module rests on is that any later run can re-derive
    // a verdict. That is false for a variant nobody kept: the ledger would
    // name a hash with nothing behind it, and the trials it applies to are
    // most of them, since rejection is the common case by design. So the node
    // and its adapter are written whichever way the judges went, and only the
    // head pointer waits on the verdict.
    //
    // The cost is 23.7 KB per trial at the measured decision-layer size, and
    // content addressing means two trials that land on identical weights are
    // stored once. A nightly loop is a few megabytes a year, which is the
    // price of every line in the ledger being checkable.
    sysbox::write_blob(&blob_path(&ablob), blob);
    variant.store();

    if cert.adopted {
        // The pointer moves last. A head naming a node that is not written yet
        // is a machine that cannot describe its own mind, and the ordering is
        // the only thing preventing it.
        set_head(&vhash);
        let _ = e.model.detach_adapters();
        let _ = e.model.attach_adapters_unseeded(adapters);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// The bound a variant has to fit inside to be carried at all.
///
/// Not arbitrary: `HEAP_LADDER` allocates one physically contiguous region
/// and comes down a rung when the memory map cannot satisfy it, so a lineage
/// that grows a megabyte per adoption is a lineage that eventually does not
/// boot. The ledger records the figure either way.
const MAX_RESIDENT_KIB: usize = 8 * 1024;

/// Guards that hold regardless of any score.
///
/// These are the ones worth having when the scores look good: a variant can
/// improve validation accuracy and still be carrying a non-finite scale that
/// will produce a NaN on the first prompt outside the corpus.
fn sanity(t: &Trial, d: &super::adapter::Dora) -> (bool, &'static str) {
    for v in d.a.iter().chain(d.b.iter()) {
        if !v.is_finite() {
            return (false, "non-finite factor");
        }
    }
    for (m, s) in d.m.iter().zip(d.s.iter()) {
        if !m.is_finite() || !s.is_finite() {
            return (false, "non-finite magnitude");
        }
        if *s <= 0.0 {
            return (false, "non-positive scale");
        }
    }
    if !t.logits_finite(Some(d)) {
        return (false, "non-finite logit");
    }
    (true, "ok")
}

/// Restore the parent of the current head.
///
/// O(1), because everything is content-addressed: the previous adapter blob
/// was never deleted and the parent node still names it. Undoing a
/// self-modification costs a pointer write and a blob read, which is the
/// property that makes adopting one defensible in the first place.
pub fn rollback(e: &mut super::Engine) -> Result<Option<[u8; 32]>, &'static str> {
    let Some(h) = head() else { return Err("no head to roll back from") };
    let Some(v) = Variant::load(&h) else { return Err("head names a node that is not stored") };
    let Some(parent) = v.parent else {
        // The root is the frozen model. Rolling back to it means detaching,
        // which is a real state rather than an error.
        let _ = e.model.detach_adapters();
        sysbox::detach(HEAD);
        return Ok(None);
    };
    let Some(pv) = Variant::load(&parent) else { return Err("parent is not stored") };
    match pv.adapter {
        None => {
            let _ = e.model.detach_adapters();
        }
        Some(ab) => {
            let Some(blob) = sysbox::read_blob(&blob_path(&ab)) else {
                return Err("the parent's adapter blob is gone");
            };
            e.model.load_adapters(&blob).map_err(|_| "the parent's adapter will not load")?;
        }
    }
    set_head(&parent);
    Ok(Some(parent))
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn counts() -> (u32, u32, u32) {
    (
        TRIALS.load(Ordering::Relaxed),
        ADOPTIONS.load(Ordering::Relaxed),
        test_reads(),
    )
}

pub fn ledger_tail(n: usize) -> Vec<String> {
    let Some(bytes) = sysbox::read_blob(LEDGER) else { return Vec::new() };
    let Ok(text) = String::from_utf8(bytes) else { return Vec::new() };
    let all: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let start = all.len().saturating_sub(n);
    all[start..].iter().map(|s| String::from(*s)).collect()
}

/// The lineage of the current head, newest first.
pub fn lineage(limit: usize) -> Vec<([u8; 32], Option<[u8; 32]>)> {
    let mut out = Vec::new();
    let mut cur = head();
    while let Some(h) = cur {
        let Some(v) = Variant::load(&h) else { break };
        out.push((h, v.adapter));
        if out.len() >= limit {
            break;
        }
        cur = v.parent;
    }
    out
}

/// Report whether the test slice may still be spoken about.
///
/// The number itself is never withheld -- withholding it would just mean
/// somebody computes it another way and quotes it without the caveat. What is
/// withheld is the *claim*: after the budget is spent, a test figure is
/// reported as stale, with the count that made it stale attached.
pub fn test_status() -> (u32, u32, bool) {
    let used = test_reads();
    (used, TEST_READS, used < TEST_READS)
}

/// Consult the test slice, spending one read.
pub fn read_test(t: &Trial, dora: Option<&super::adapter::Dora>) -> (f32, u32, bool) {
    let n = spend_test_read();
    (t.score(dora, Slice::Test), n, n <= TEST_READS)
}

/// Eight hex characters of an address, for anything that has to fit on a line.
pub fn short_hex(h: &[u8; 32]) -> String {
    short(h)
}

/// Run a trial and print the certificate.
///
/// The whole certificate, including the judges that passed. A report that
/// only said why something was rejected would be a report nobody could argue
/// with, and the reason for writing these down is that they can be.
pub fn report_trial(b: &Budget) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[godel] trial");
    console::set_color(LTGRAY);

    let Some(p) = frontier() else {
        let (seen, all) = explored();
        kprintln!("  the search space is exhausted -- {} of {} points tried", seen, all);
        kprintln!("  widen GRID, or 'godel forget' to walk it again");
        return;
    };
    let (seen, all) = explored();
    kprintln!(
        "  point {} of {}: lr {}, rank {}, alpha {}, epochs {}",
        seen + 1,
        all,
        p.lr,
        p.rank,
        p.alpha,
        p.epochs
    );
    // The examples and the wall clock come from the caller; everything that
    // decides what the weights become comes from the proposal.
    let b = &p.budget(b.examples, b.millis);
    let outcome = super::with_engine(|e| trial(e, b, &p));
    let c = match outcome {
        None => {
            kprintln!("  no engine, or another task holds it");
            return;
        }
        Some(Err(_)) => {
            kprintln!("  no trial: the trainer refused (hardware, corpus or checkpoint)");
            return;
        }
        Some(Ok(c)) => c,
    };

    kprintln!(
        "  variant {} from {}",
        short_hex(&c.variant),
        c.parent.map(|h| short_hex(&h)).unwrap_or(String::from("the frozen model"))
    );
    kprintln!("  {} decisions, {} in validation", c.decisions, c.validation);
    kprintln!(
        "  predicted {} from training-set gain (nothing acts on this yet)",
        if c.predicted { "a win" } else { "a loss" }
    );

    let mark = |p: bool| if p { "pass" } else { "VETO" };
    if c.validation == 0 {
        kprintln!("  J1 margin    VETO  {} -- ask for more examples", c.j1_why);
    } else {
        kprintln!(
            "  J1 margin    {}  {} repaired, {} broken of {}, chi {} ({})",
            mark(c.j1),
            c.fixed,
            c.broke,
            c.validation,
            (c.mcnemar * 100.0) as u32 as f32 / 100.0,
            c.j1_why
        );
    }
    kprintln!(
        "  J2 own goals {}  {}/{} still route where they did",
        mark(c.j2),
        c.goals_held,
        c.goals_total
    );
    kprintln!("  J3 sanity    {}  {}", mark(c.j3), c.j3_why);
    if c.capped {
        kprintln!(
            "  {} epochs, ended by the wall-clock cap: this verdict will not",
            c.epochs
        );
        kprintln!("  re-derive on a machine of a different speed");
    } else {
        kprintln!("  {} epochs, ended by the epoch count", c.epochs);
    }
    kprintln!(
        "  J4 cost      {}  rank {}, {} KiB resident",
        mark(c.j4),
        c.rank,
        c.resident_kib
    );

    if c.adopted {
        console::set_color(LTGREEN);
        kprintln!("  adopted -- all four agreed, and the parent is still addressed");
        console::set_color(LTGRAY);
        // Read after winning, never to decide the winner.
        if c.test_fresh {
            kprintln!(
                "  test slice {}% -- read {} of {}",
                (c.test_acc * 100.0) as u32,
                c.test_read,
                TEST_READS
            );
        } else {
            console::set_color(YELLOW);
            kprintln!(
                "  test slice {}% -- read {}, past its budget of {}: STALE, do not quote it",
                (c.test_acc * 100.0) as u32,
                c.test_read,
                TEST_READS
            );
            console::set_color(LTGRAY);
        }
        kprintln!("  'godel rollback' undoes it for the cost of a pointer write");
    } else {
        console::set_color(YELLOW);
        kprintln!("  rejected -- unanimity is required, and it was not unanimous");
        console::set_color(LTGRAY);
    }
    let _ = LTRED;
}

/// Boot self-test. Seven claims, none needing a model or a quiet window.
///
/// What is checked here is the machinery that decides whether the machine may
/// change itself -- the window arithmetic, the statistic, and the content
/// addressing that makes a lineage un-rewritable. None of it involves a
/// forward pass, which is the point: the expensive half of a trial is
/// evidence, and evidence is not what these claims are about.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut ok = true;
    let mut claim = |what: &str, pass: bool| {
        if !pass {
            ok = false;
        }
        kprintln!("  {}  {}", if pass { "ok " } else { "FAIL" }, what);
    };

    // A window that does not wrap, and one that does. The wrapping case is
    // the whole reason this is a function: `from <= h < until` would permit
    // nothing at all for 22:00-04:00, silently, forever.
    let (from, until) = window();
    claim(
        "the quiet window admits its own hours and refuses the rest",
        in_window(from) && !in_window(until) && !in_window((until + 6) % 24),
    );

    let wraps = |h: u8| {
        let (f, u) = (22u8, 4u8);
        if f <= u {
            h >= f && h < u
        } else {
            h >= f || h < u
        }
    };
    claim(
        "a window across midnight admits 23:00 and 01:00, not 12:00",
        wraps(23) && wraps(1) && !wraps(12),
    );

    // The statistic, on cases whose answers are arithmetic rather than
    // opinion. Nine repairs against two breaks is the shape that looks
    // convincing and is not: chi is 3.27, under the line.
    let a = mcnemar(2, 9);
    let b = mcnemar(1, 12);
    claim(
        "nine repairs against two breaks does not clear the bar; twelve against one does",
        a < MCNEMAR_95 && b >= MCNEMAR_95,
    );
    claim(
        "an even split is no evidence at all",
        mcnemar(7, 7) == 0.0 && mcnemar(0, 0) == 0.0,
    );

    let h = sha256::hash(b"a variant");
    claim(
        "a hash survives being written down and read back",
        from_hex32(&hex32(&h)) == Some(h),
    );

    // The property the whole DAG rests on: identical content is the same
    // node, and `born` is deliberately outside the hash so that a variant
    // rediscovered tomorrow is recognisably the one already tried rather
    // than a new one that behaves identically.
    let mk = |lambda: f32, born: u32| Variant {
        parent: Some(h),
        adapter: Some(sha256::hash(b"adapter")),
        policy: None,
        skills: None,
        corpus: Some(sha256::hash(b"corpus")),
        lambda,
        rank: 8,
        epochs: 20,
        rule: 0,
        born,
    };
    let v1 = mk(0.02, 1000);
    let v2 = mk(0.02, 9999);
    let v3 = mk(0.05, 1000);
    claim(
        "two variants differing only in when they were born are one node",
        v1.hash() == v2.hash(),
    );
    claim(
        "a variant differing in a parameter is a different node",
        v1.hash() != v3.hash(),
    );

    // The property the first two recorded trials violated. They trained the
    // same corpus with the same settings, stopped at different epochs because
    // a busy host reached the wall-clock cap sooner, and produced adapters
    // that differed with nothing in the node saying why.
    let mut v4 = mk(0.02, 1000);
    v4.epochs = 19;
    let mut v5 = mk(0.02, 1000);
    v5.corpus = Some(sha256::hash(b"a different corpus"));
    claim(
        "epochs and corpus are part of what a variant is",
        v1.hash() != v4.hash() && v1.hash() != v5.hash(),
    );

    // Store and read back, then take the scratch node out of the real DAG --
    // a self-test that left synthetic ancestors in the lineage would be
    // corrupting the history it exists to protect.
    let stored = v1.store();
    let back = Variant::load(&stored);
    let round = back.map_or(false, |b| {
        b.parent == v1.parent && b.adapter == v1.adapter && b.rule == v1.rule
    });
    let mut np = String::from(ROOT);
    np.push_str("/nodes/");
    np.push_str(&hex32(&stored));
    let mut bp = np.clone();
    bp.push_str(".born");
    sysbox::detach(&np);
    sysbox::detach(&bp);
    claim("a stored variant reads back with its lineage intact", round);

    ok
}
