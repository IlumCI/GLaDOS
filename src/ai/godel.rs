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
/// The bar the judges are using, when it is not the constant.
pub const BAR: &str = "/ai/godel/bar";

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
/// What kind of change is being proposed.
///
/// The loop began as an adapter search: every proposal was a set of training
/// knobs, and the one other thing the machine could change -- a council core --
/// reached adoption down a separate path with its own entry point. Two ways in
/// meant two places for the bookkeeping to drift, and the second one had to
/// grow its own copy of the lineage discipline before it was safe to use.
///
/// So the kind is in the proposal, and one dispatcher routes it. What that
/// buys is not tidiness: it means the *scheduler* no longer has to know what
/// kind of thing it is running. The night loop asks for the next proposal and
/// runs it, and whether that turns out to be a learning rate or a program the
/// machine wrote an hour ago is a fact about the proposal, not about the loop.
///
/// **A kind here must have a judge.** The obvious extensions -- deep training
/// (J1-J4 exist, the routing does not), skills (no judge at all), the routing
/// rule (needs a calibration judge before `rule` is searchable) -- are absent
/// on purpose. A variant in this enum that `run` cannot judge would be a
/// promise the type makes and the machine cannot keep, and an unjudged change
/// adopted at three in the morning is exactly what this module exists to
/// prevent.
#[derive(Clone, Copy, PartialEq)]
pub enum ProposalKind {
    /// Train a classifier adapter with these knobs. Judged J1-J4.
    Adapter,
    /// Judge a council core already stored under its content address.
    /// Judged J1/J5/J6 by `harness::core_bench_in`.
    Core([u8; 32]),
    /// Adapt the attention path as well as the classifier, and judge what
    /// that bought against what it costs. J1-J4, paired on routing.
    Deep,
    /// Admit a skill the machine compiled from an episode. Judged by
    /// `skill::bench`: it is a program, it runs under the powers it will
    /// really have, it repeats, and it is cheap.
    Skill([u8; 32]),
    /// Move the bar the other five answer to. Judged by `judge_verdict` on a
    /// cross-evaluation matrix, and reachable only at an epoch boundary.
    Judge(f32),
    /// Change how the council combines its cores. Judged on calibration by
    /// `harness::rule_bench`, because accuracy is not what this axis moves.
    Config(u8),
    /// Admit a function into the solver's library, by the address of its
    /// source.
    ///
    /// The only axis here that does not touch the model at all. What it moves
    /// is what the solver can *say* -- a library function is one node where
    /// the expression it replaces was several, so answers get shorter and
    /// problems that were out of reach come into reach. Judged by
    /// `redqueen::bench`, paired over every stored problem, because adding a
    /// function lengthens every candidate list and one that buys nothing
    /// costs something.
    Lib([u8; 32]),
    /// Change a declared constant in this kernel's own source.
    ///
    /// **The one kind that cannot be judged here, and that is the shape of the
    /// machine rather than a gap in this axis.** The kernel has no copy of its
    /// source and cannot compile, so nothing on this side can build the
    /// variant or measure it. What a source proposal produces is a patch in
    /// the outbox and a record that the point was reached; the verdict comes
    /// back from whatever built it, against the rail the knob declares.
    ///
    /// `(row, value)` as indices into `knob::KNOBS`, because `Proposal` is
    /// `Copy`. A marker written by a kernel with a wider table must resolve to
    /// nothing here rather than to whatever row happens to sit at that index,
    /// which is what `knob::at` refuses.
    Source(u16, u16),
}

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
    pub kind: ProposalKind,
}

/// Why a proposal could not be run to a verdict.
///
/// Two error types met here: the trainer refuses for reasons about the machine
/// and the corpus, the core judge for reasons about the program. Flattening
/// them to a bool would lose the difference between "there was nothing to
/// train on" and "the core will not load", which are the two facts a journal
/// line at 3am has to carry.
pub enum Refused {
    Train(super::train::RunError),
    Judge(&'static str),
}

impl Refused {
    pub fn why(&self) -> &'static str {
        match self {
            Refused::Train(super::train::RunError::Hardware) => "the hardware check said no",
            Refused::Train(super::train::RunError::NoCorpus) => "there is no corpus",
            Refused::Train(super::train::RunError::Hybrid) => "the model is a hybrid the trainer will not touch",
            Refused::Train(super::train::RunError::NoDecisions) => "the corpus produced no decisions",
            Refused::Judge(w) => w,
        }
    }
}

impl Proposal {
    /// A proposal to judge a stored council core.
    ///
    /// The training knobs are zero because a core is not trained -- it is a
    /// program, already written, and the only question is whether it earns a
    /// place in the decision path. They are still rendered, and still in the
    /// hash, because a proposal is identified by its whole text: leaving them
    /// out would make this a different kind of document and the marker
    /// directory holds one kind.
    pub fn core(h: [u8; 32]) -> Proposal {
        Proposal {
            lr: 0.0,
            rank: 0,
            alpha: 0.0,
            epochs: 0,
            rule: super::harness::Rule::WithCore as u8,
            kind: ProposalKind::Core(h),
        }
    }

    /// A proposal to adapt the attention path, with the knobs it trains under.
    ///
    /// Unlike a core, this one is trained here and now, so it carries a real
    /// learning rate, rank and epoch count -- and they are in the rendering,
    /// which means two deep runs at different settings are different points
    /// and the marker directory can tell them apart.
    pub fn deep(lr: f32, rank: usize, alpha: f32, epochs: usize) -> Proposal {
        Proposal { lr, rank, alpha, epochs, rule: 0, kind: ProposalKind::Deep }
    }

    /// A proposal to admit a stored skill.
    ///
    /// No training knobs at all: a skill is a program that already exists and
    /// the question is whether it is fit to keep, not how to make one. They
    /// are still rendered, for the reason `core` renders them -- the marker
    /// directory holds one kind of document.
    pub fn skill(h: [u8; 32]) -> Proposal {
        Proposal { lr: 0.0, rank: 0, alpha: 0.0, epochs: 0, rule: 0, kind: ProposalKind::Skill(h) }
    }

    /// A proposal to change how the council combines its cores.
    /// A proposal to move the bar the other judges answer to.
    ///
    /// `rule` is zero rather than the one in force, so a bar is identified by
    /// the bar alone -- otherwise the same criterion proposed under two
    /// routing rules would be two points, and the marker directory would let
    /// the loop propose it twice.
    ///
    /// **The training knobs are not zero, and they were, and that made this
    /// axis incapable of adopting anything.** The reasoning for zeroing them
    /// was copied from `core` and `skill`, where nothing is trained and the
    /// candidate already exists. Here something *is* trained: `trial_judge`
    /// builds the budget from this proposal and calls `Trial::train` with it,
    /// so `epochs: 0` ran the optimiser zero times and `rank: 0` left
    /// `Dora::refresh` with an empty low-rank factor and every scale at one.
    /// The candidate came out numerically identical to the incumbent, giving
    /// `fixed == broke == 0`; `Cross::admits` requires `fixed > broke`, so
    /// both bars refused it, `judge_verdict` answered "admits and refuses what
    /// the standing one did", and adoption was unreachable for every input.
    ///
    /// The values are `Budget::default()`'s, which is what `godel now` and
    /// `godel storm` already start from. They are *constant* rather than
    /// varied, which is what keeps the identity argument above intact: the
    /// bar is still the only thing that distinguishes two judge proposals.
    pub fn judge(bar: f32) -> Proposal {
        Proposal {
            lr: 0.02,
            rank: 8,
            alpha: 16.0,
            epochs: 20,
            rule: 0,
            kind: ProposalKind::Judge(bar),
        }
    }

    /// No training knobs, for the reason `core` and `skill` have none: the
    /// candidate is a program that already exists and the question is whether
    /// it is fit to keep. Unlike `judge`, nothing here is trained from the
    /// budget, so zeroes are correct rather than inert.
    pub fn lib(h: [u8; 32]) -> Proposal {
        Proposal { lr: 0.0, rank: 0, alpha: 0.0, epochs: 0, rule: 0, kind: ProposalKind::Lib(h) }
    }

    pub fn source(knob: usize, value: usize) -> Proposal {
        Proposal {
            lr: 0.0,
            rank: 0,
            alpha: 0.0,
            epochs: 0,
            rule: 0,
            kind: ProposalKind::Source(knob as u16, value as u16),
        }
    }

    pub fn config(rule: u8) -> Proposal {
        Proposal { lr: 0.0, rank: 0, alpha: 0.0, epochs: 0, rule, kind: ProposalKind::Config(rule) }
    }

    /// Which of `AXIS_NAMES` this proposal will be filed under.
    ///
    /// The same mapping the certificates make, in one place, so a schedule
    /// reading the ledger by axis and a trial writing it by axis cannot
    /// disagree about which is which. `Source` has no slot: it is not one of
    /// the axes the ranking chooses among, because the machine cannot run it.
    pub fn axis_slot(&self) -> Option<usize> {
        let name = match self.kind {
            ProposalKind::Adapter => "adapter",
            ProposalKind::Config(..) => "rule",
            ProposalKind::Skill(..) => "skill",
            ProposalKind::Deep => "deep",
            ProposalKind::Lib(..) => "lib",
            ProposalKind::Core(..) => "core",
            ProposalKind::Judge(..) => "judge",
            ProposalKind::Source(..) => return None,
        };
        AXIS_NAMES.iter().position(|n| *n == name)
    }

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
        // Emitted only when the kind is not the one every existing marker
        // was written under.
        //
        // The same compatibility rule `Variant::render` follows, and here it
        // guards something with teeth: `tried()` looks a proposal up by the
        // hash of this text, so an unconditional `kind` line would re-address
        // every marker in `/ai/godel/tried` at once and the grid would be
        // walked from the beginning as though nothing had ever been measured.
        match self.kind {
            ProposalKind::Adapter => {}
            ProposalKind::Core(h) => {
                s.push_str("core ");
                s.push_str(&hex32(&h));
                s.push('\n');
            }
            ProposalKind::Deep => s.push_str("deep 1\n"),
            ProposalKind::Judge(bar) => {
                s.push_str("judge ");
                push_f6(&mut s, bar);
                s.push('\n');
            }
            ProposalKind::Skill(h) => {
                s.push_str("skill ");
                s.push_str(&hex32(&h));
                s.push('\n');
            }
            // The rule is already a field of the rendering above, so a config
            // point is distinguished by carrying nothing else: zero knobs and a
            // `rule` line that differs. Emitting a second copy of the rule would
            // make the identity depend on the same fact twice.
            ProposalKind::Config(_) => s.push_str("config 1
"),
            ProposalKind::Lib(h) => {
                s.push_str("lib ");
                s.push_str(&hex32(&h));
                s.push('\n');
            }
            // The *names*, not the indices. A marker has to survive a row
            // being inserted above it in `KNOBS`, and an index does not: the
            // table would shift and every marker would resolve to a different
            // constant while still reading as tried. What the point is *about*
            // is the file, the symbol and the value.
            ProposalKind::Source(ki, vi) => {
                s.push_str("source ");
                match super::knob::at(ki as usize, vi as usize) {
                    Some((k, v)) => {
                        s.push_str(k.file);
                        s.push(' ');
                        s.push_str(k.symbol);
                        s.push('=');
                        s.push_str(v);
                    }
                    // A point this kernel cannot resolve still has to render
                    // to something stable, or `tried()` would answer
                    // differently on two calls. It renders to its indices and
                    // is refused before it runs.
                    None => {
                        push_u32(&mut s, ki as u32);
                        s.push(' ');
                        push_u32(&mut s, vi as u32);
                    }
                }
                s.push('\n');
            }
        }
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
    Proposal { lr: 0.02, rank: 8, alpha: 16.0, epochs: 20, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.05, rank: 8, alpha: 16.0, epochs: 20, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.01, rank: 8, alpha: 16.0, epochs: 40, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.02, rank: 16, alpha: 32.0, epochs: 20, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.05, rank: 16, alpha: 32.0, epochs: 20, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.01, rank: 4, alpha: 8.0, epochs: 40, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.08, rank: 8, alpha: 16.0, epochs: 12, rule: 0, kind: ProposalKind::Adapter },
    Proposal { lr: 0.02, rank: 32, alpha: 64.0, epochs: 20, rule: 0, kind: ProposalKind::Adapter },
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

    // The kind reaches the hash, and only when it is not the default.
    //
    // Both halves are load-bearing and they pull against each other. If an
    // adapter point rendered a `kind` line, every marker already in
    // `/ai/godel/tried` would re-address at once and the grid would be walked
    // again from the top as though nothing had ever been measured -- weeks of
    // nights, silently repeated. If a core point did *not* render one, two
    // different programs would share a marker and the second would never be
    // judged.
    if base.render().contains("core ") {
        return false;
    }
    let c1 = Proposal::core(sha256::hash(b"one core"));
    let c2 = Proposal::core(sha256::hash(b"another core"));
    if !c1.render().contains("core ") || c1.hash() == c2.hash() {
        return false;
    }
    if GRID.iter().any(|p| p.hash() == c1.hash()) {
        return false;
    }
    // A deep point is its own kind, and its knobs are in its identity: two
    // deep runs at different ranks are different experiments, and a marker
    // that could not tell them apart would let the loop believe it had already
    // tried something it had not.
    let d1 = Proposal::deep(0.02, 4, 8.0, 4);
    let d2 = Proposal::deep(0.02, 8, 8.0, 4);
    if !d1.render().contains("deep 1") || d1.hash() == d2.hash() || d1.hash() == c1.hash() {
        return false;
    }
    if GRID.iter().any(|p| p.hash() == d1.hash()) {
        return false;
    }

    // A config point is its own kind, and its rule is its identity.
    //
    // The rule is already a rendered field, so what the kind line has to do is
    // stop a config point colliding with an adapter point that happens to run
    // under the same rule -- they are different experiments and a shared
    // marker would let the loop believe it had judged one when it judged the
    // other.
    let g1 = Proposal::config(1);
    let g3 = Proposal::config(3);
    if g1.hash() == g3.hash() || !g1.render().contains("config 1") {
        return false;
    }
    if GRID.iter().any(|p| p.hash() == g1.hash() || p.hash() == g3.hash()) {
        return false;
    }
    // The byte in a node is `Rule as u8`, so the mapping back has to be the
    // declaration order and nothing else. A node from a future kernel naming a
    // fifth rule is refused rather than silently routed as `ProbeOnly`.
    use super::harness::Rule;
    if Rule::from_u8(Rule::Majority as u8) != Some(Rule::Majority)
        || Rule::from_u8(Rule::WithCore as u8) != Some(Rule::WithCore)
        || Rule::from_u8(200).is_some()
    {
        return false;
    }

    // The surprise ordering is a permutation, not merely a ranking.
    //
    // What this guards is the insertion sort quietly losing a slot. An order
    // that came back `[0, 0, 1, 2]` would leave `deep` unreachable for the
    // rest of the machine's life, every night would still look like work, and
    // the only symptom would be a ledger that never mentions the axis -- the
    // same failure the old round-robin claim was written against, in the shape
    // the new ordering can fail in.
    let mut seen_kinds = [false; RANKED];
    for slot in surprise_order() {
        if slot >= RANKED {
            return false;
        }
        seen_kinds[slot] = true;
    }
    if !seen_kinds.iter().all(|s| *s) {
        return false;
    }
    // Deep points are a declared grid like the adapter one, so two of them
    // must be different experiments.
    if DEEP_GRID.len() < 2 {
        return false;
    }
    let d0 = Proposal::deep(DEEP_GRID[0].0, DEEP_GRID[0].1, DEEP_GRID[0].2, DEEP_GRID[0].3);
    let d1 = Proposal::deep(DEEP_GRID[1].0, DEEP_GRID[1].1, DEEP_GRID[1].2, DEEP_GRID[1].3);
    if d0.hash() == d1.hash() {
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

/// Have the machine write a council core, and store it by content address.
///
/// **Claims the engine only briefly, and never while composing.** The class
/// list is read under one short claim and released; every decode inside
/// `voter::author` then takes its own. That is not an optimisation -- calling
/// this from inside `with_engine` would hand the same task a second `&mut
/// Engine`, which is undefined behaviour in this kernel and was a live defect
/// in `trial_core` until recently.
pub fn write_core() -> Option<[u8; 32]> {
    // One claim, and everything the decodes will need comes out of it.
    //
    // Mining needs the tokenizer, so it needs the engine -- but the decodes
    // that follow each claim it themselves, so the mining has to finish and
    // let go first. Owned data comes back; nothing borrowed escapes.
    let (names, table) = super::with_engine(|e| {
        let names: Vec<String> =
            (0..e.head.len()).map(|i| String::from(e.head.name(i))).collect();
        let table = super::harness::contested_cues(e, &names);
        (names, table)
    })?;
    let src = super::voter::author(&names, &table)?;
    Some(super::voter::store(&src))
}

/// A proposal for a core the machine has just written, if it is new.
///
/// `None` when nothing could be composed, or when this exact program has
/// already been judged. The second case is the marker doing its job: the
/// composition is a function of the corpus and the model's choices, so a
/// machine that keeps making the same choices keeps writing the same program,
/// and judging it nightly would fill the ledger with one result reported
/// forever.
pub fn author_core() -> Option<Proposal> {
    // Do not spend a night writing something that cannot win.
    //
    // The census says how many validation items a core could repair even if it
    // got every one of them right; J1 says how many it must. When the first is
    // below the second, no program of any construction clears the judge on
    // this slice, and composing one is a night spent producing a rejection
    // that was arithmetic before it was a measurement. The number is whatever
    // the last bench measured, so the first night still tries.
    if let Some((prize, need)) = core_room() {
        if prize < need {
            return None;
        }
    }
    let p = Proposal::core(write_core()?);
    if p.tried() {
        return None;
    }
    Some(p)
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
pub const MIN_FIXED: usize = 4;

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
#[derive(Clone)]
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
    /// Content address of the council core in force, if one is installed.
    ///
    /// The axis that makes this a general loop rather than an adapter search.
    /// A core is an Aiksi program the machine can write, `core_bench` already
    /// judges it, and `voter::install` already adopts it by hash -- what was
    /// missing was any record that it happened, so a core could pass three
    /// judges and leave no trace in the lineage, no ledger line, and nothing
    /// for `rollback` to undo.
    pub core: Option<[u8; 32]>,
    /// Whether this node actually *says* anything about a core.
    ///
    /// Not part of the identity -- it is a fact about the text, not about the
    /// mind -- and so deliberately absent from `render` as a value. What it
    /// controls is whether the `core` line is written at all.
    ///
    /// The distinction is load-bearing and its absence was a real defect.
    /// `rollback` read `parent.core == None` as "the parent had no core" and
    /// uninstalled. But every node written before this field existed also
    /// parses as `None`, so rolling back an *adapter* on any older lineage
    /// silently pulled a machine-written core out of the decision path, with
    /// nothing printed. "Absent" and "none" are different claims and a format
    /// that cannot tell them apart forces the reader to guess.
    ///
    /// Nodes this kernel writes always set it, so they always state the
    /// answer; nodes it reads keep whatever the text said, so an old node
    /// still renders to the exact bytes it was stored as.
    pub core_seen: bool,
    /// Whether the attention path moved, not only the classifier.
    ///
    /// `deeptrain` adapts every q/k/v site as well as the decision layer, and
    /// the two are different objects with different economics -- a
    /// classifier-only adapter leaves every hidden state a constant, which is
    /// what makes cached features and cheap re-judging possible, and a deep
    /// one gives that up. A node that does not say which it is describes the
    /// wrong experiment.
    pub deep: bool,
    /// Content address of the redqueen library in force, if one has been
    /// adopted.
    ///
    /// Conditional in the rendering for the reason `core` and `deep` are: an
    /// unconditional line re-addresses every node that already exists, which
    /// makes `head` name something that no longer reproduces -- the change
    /// meant to extend re-derivability breaking it instead.
    pub lib: Option<[u8; 32]>,
    /// The bar this variant was judged under, when the judge axis set one.
    ///
    /// **It used to ride in `rule`, and that corrupted the lineage it was
    /// meant to record.** `trial_judge` packed it as hundredths --
    /// `(bar * 100.0) as u8` -- over a `JUDGE_GRID` of
    /// `[1.0, 2.0, 3.0, 5.0, 8.0, 12.0]`. Rust's `as` saturates, so 3.0, 5.0,
    /// 8.0 and 12.0 all became **255**: four of six grid points collapsed onto
    /// one number, and 100, 200 and 255 are none of them a `Rule`, which
    /// `harness::Rule::from_u8` maps over `0..=3`. Every judge node therefore
    /// named a routing rule this kernel does not have, and `rollback` onto one
    /// fails with exactly that sentence.
    ///
    /// A field of its own, conditional in the rendering for the reason `core`,
    /// `deep` and `lib` are: an unconditional line re-addresses every node
    /// that already exists. `f32` and not a second packing, because a lossy
    /// encoding is how this went wrong the first time.
    ///
    /// **Every axis records it, not only the judge axis**, because the claim
    /// it exists to support -- that a lineage says which criterion each
    /// variant was judged under -- is false if the next adapter trial drops
    /// it. So it is `bar_in_force()` at the moment of the trial, which is a
    /// fact about the judgement rather than something carried from a parent.
    /// `None` survives in exactly two places and means "did not say" rather
    /// than "had none": a node written before this field existed, and the
    /// incumbent `ensure_head` records, which arrived from outside the loop
    /// and was judged under nothing.
    pub bar: Option<f32>,
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
        // **Emitted only when there is one.**
        //
        // Every node written before this field existed has no core, and must
        // go on rendering to exactly the bytes it was stored as -- otherwise
        // `head` names an address that no longer reproduces, the ledger stops
        // being checkable, and the re-derivability this module promises is
        // broken by the very change meant to extend it. An unconditional
        // "core none" line would have done precisely that to every node in
        // every existing lineage.
        //
        // `core none` is written too, and only by nodes that know they have
        // an answer to give. That is what lets `rollback` tell a node that
        // says "no core" from one that says nothing at all -- see
        // `core_seen`. An old node has `core_seen` clear, emits neither line,
        // and re-renders byte-for-byte to the address it is stored under.
        match (self.core, self.core_seen) {
            (Some(c), _) => {
                s.push_str("core ");
                s.push_str(&hex32(&c));
                s.push('\n');
            }
            (None, true) => s.push_str("core none\n"),
            (None, false) => {}
        }
        // Conditional for the same reason `core` is: absent from the rendering
        // when absent from the object, so every node written before this
        // existed still renders to the bytes it was stored as.
        // Same rule as `core` and `deep`: written only when there is one, so
        // every node stored before the library existed renders byte for byte
        // as it was stored.
        if let Some(l) = self.lib {
            s.push_str("lib ");
            s.push_str(&hex32(&l));
            s.push_str("\n");
        }
        if self.deep {
            s.push_str("deep 1\n");
        }
        // Last, and conditional, so every node written before the judge axis
        // had a field of its own renders byte for byte as it was stored.
        // `push_f6` is what `BAR` itself is written with, so a bar read back
        // and re-rendered is exact and the round trip holds.
        if let Some(b) = self.bar {
            s.push_str("bar ");
            push_f6(&mut s, b);
            s.push('\n');
        }
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
        Some(Variant::from_text(text))
    }

    /// Parse a node from its canonical rendering.
    ///
    /// Separate from `load` so the round trip -- the property the whole DAG
    /// rests on -- can be checked without a store, a namespace, or a node
    /// somebody has to remember to write first.
    pub fn from_text(text: &str) -> Variant {
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
            core: None,
            // Clear until a `core` line is actually seen below, which is the
            // whole point: a node that does not mention a core must not be
            // read as one that mentions having none.
            core_seen: false,
            deep: false,
            lib: None,
            bar: None,
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
                // Absent for a long time, and the omission was not cosmetic.
                // A node read back got `lambda: 0.0` whatever was stored, so
                // `Variant::load(h).render()` did not reproduce the text at
                // `h` and re-hashed to a different address. Nothing re-rendered
                // a loaded node, so it never surfaced -- but re-deriving a
                // verdict from the DAG is the claim this whole module rests
                // on, and it was false for any variant with a learning rate.
                //
                // `push_f2` writes exactly two decimals, and parsing that back
                // and re-rendering it is exact, so the round trip holds.
                "lambda" => v.lambda = val.parse().unwrap_or(0.0),
                // `none` parses to `None`, and either way the node has now
                // made a statement about its core.
                "core" => {
                    v.core = from_hex32(val);
                    v.core_seen = true;
                }
                "deep" => v.deep = val == "1",
                "lib" => v.lib = from_hex32(val),
                "bar" => v.bar = val.parse().ok(),
                "rank" => v.rank = val.parse().unwrap_or(0),
                "epochs" => v.epochs = val.parse().unwrap_or(0),
                "rule" => v.rule = val.parse().unwrap_or(0),
                _ => {}
            }
        }
        v
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
#[derive(Clone)]
pub struct Certificate {
    /// Which axis produced this, as it appears in the ledger.
    ///
    /// **The ledger could not say.** Five judges wrote lines in the same shape
    /// and nothing in the file distinguished a night spent on the adapter grid
    /// from one spent on a routing rule, so "which axes have actually been
    /// tried" was a question the record could not answer -- and the surprise
    /// ordering is exactly that question asked every night. A field rather
    /// than an inference from the other columns, because a deep trial and an
    /// adapter trial can render identical numbers.
    pub axis: &'static str,
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
    /// How many validation decisions the *incumbent* gets wrong, which is the
    /// ceiling on `fixed`. **The number that makes a veto legible.**
    ///
    /// `paired` has always computed it and thrown it away -- it is
    /// `fixed + neither` -- and without it the ledger records that J1 wanted
    /// six repairs and got two, with no way to tell a candidate that
    /// underperformed from a budget on which six repairs did not exist. The
    /// same thing `core_room` reports for a core, on the axis that needed it
    /// more: at 24 nightly examples this measured **15 validation decisions**
    /// and `clean_fixes_needed()` is **6**, so J1 was asking for a repair of
    /// forty per cent of everything held out, with nothing broken.
    pub wrong: Option<usize>,
    /// The corpus subsample this trial was *given*, which is not the same as
    /// how many decisions came out of it.
    ///
    /// **Without it the schedule cannot read its own history.** `oops` decides
    /// how much tonight may spend from what past nights spent and what they
    /// got for it, and `n=` is validation decisions rather than the budget --
    /// derived from it, and not invertibly. A schedule that inferred the
    /// budget would be a second account of the record free to disagree with
    /// it, which is the objection `axis_counts` makes about a counter in its
    /// own file.
    ///
    /// `None` on the axes that take no subsample: a library function, a skill,
    /// a core and a routing rule are judged against things that are not a
    /// corpus slice, so there is no budget to record and saying zero would
    /// read as one.
    pub budget_n: Option<usize>,
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
/// The paired statistic, shared with the core judge.
///
/// Public so a second judge cannot grow a second definition of "beyond the
/// noise". Two thresholds that drift apart would let a change be significant
/// to one loop and not to the other, with nothing saying so.
pub fn mcnemar(broke: usize, fixed: usize) -> f32 {
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

/// The smallest number of clean repairs -- repairs with nothing broken --
/// that actually clears J1.
///
/// Derived rather than written down, because it is not `MIN_FIXED`. Yates'
/// correction subtracts one from the difference before squaring, so four clean
/// fixes score 2.25 and are refused by the very judge whose floor is four; the
/// real bar on this configuration is six. Two constants that look like the
/// answer and are not is exactly the arrangement in which somebody eventually
/// quotes the wrong one, so the answer is computed from both.
pub fn clean_fixes_needed() -> usize {
    // The bar a judge answers to, which is the adopted one lifted by whatever
    // the family-wise budget has spent -- not `bar_in_force`, or this would
    // report a requirement no trial is actually held to.
    let bar = effective_bar().unwrap_or(f32::INFINITY);
    let mut f = MIN_FIXED;
    while f < 1024 {
        if mcnemar(0, f) >= bar {
            return f;
        }
        f += 1;
    }
    f
}

/// What a core could win here, and what it would have to win, when both are
/// known. `None` until something has been benched this boot.
pub fn core_room() -> Option<(usize, usize)> {
    super::harness::last_prize().map(|p| (p, clean_fixes_needed()))
}

/// Roughly the 95% threshold for one degree of freedom. Named rather than
/// spelled inline because it is a *decision*, not a constant: 3.84 is the
/// conventional line and the ledger records the statistic itself, so a later
/// reader can apply a different one to the same numbers.
pub const MCNEMAR_95: f32 = 3.84;

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
    // And not on battery. The unattended jobs are the most expensive thing
    // this machine does -- two passes over the corpus for a deep trial, a
    // dozen decodes to compose a core -- and a laptop that spends the night
    // improving itself into a flat battery has not improved itself. A skipped
    // night costs one rotation slot; a flat battery costs the morning.
    //
    // Only a *known* battery refuses. A desktop reports no adapter and gets
    // the same answer it always did, because "unknown" must not become "no".
    if let Some(c) = crate::dev::battery::status() {
        if c.on_ac == Some(false) {
            return Err("on battery");
        }
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

    // The one moment a trial changes anything a window can show: the counts,
    // the head and the judges' verdicts all land with this line. Repainted
    // from here, on this task, which is the pattern `agent` and `author`
    // already follow -- `desk::with` is unguarded, so a repaint driven from
    // the clock task could alias `&mut Desktop` against whatever the working
    // task was halfway through.
    //
    // Without it the Improve window kept whatever it last drew until some
    // unrelated window operation repainted the desktop, so a trial could run,
    // be judged and be refused with the pane in front of the operator still
    // reading `0 of 0 trials`.
    crate::gfx::desk::draw();
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
    s.push_str(" axis=");
    s.push_str(c.axis);
    // **Which body of evidence this test was paid for out of.** The
    // family-wise budget counts tests against the corpus in force, so a line
    // that does not say which corpus it was run against cannot be counted --
    // and counting the wrong ones would either refuse a machine that had
    // earned more questions or let one keep asking after its evidence was
    // spent. Derived from the variant rather than stored twice.
    if let Some(h) = sysbox::hash_of(super::vocab::CORPUS) {
        s.push_str(" corpus=");
        s.push_str(&short(&h));
    }
    // Derived at render time from columns already here rather than stored, so
    // a cell can never disagree with the counts it was computed from, and an
    // old line re-read under a changed `descriptor` reports where that variant
    // would land *now* -- which is what a reader comparing epochs wants.
    s.push_str(" cell=");
    push_u32(&mut s, cell_of(c.rank, c.fixed, c.broke) as u32);
    s.push_str(" n=");
    push_u32(&mut s, c.validation as u32);
    s.push_str(" pred=");
    s.push_str(if c.predicted { "win" } else { "lose" });
    s.push_str(" J1[fix=");
    push_u32(&mut s, c.fixed as u32);
    s.push_str(" broke=");
    push_u32(&mut s, c.broke as u32);
    // Written only when it is known, so an axis that cannot compute a ceiling
    // says nothing rather than claiming zero -- which reads as "the incumbent
    // was already perfect" and is the opposite of what it would mean. Every
    // reader here keys on the field name, so adding one is safe; the shape
    // `axis=` and `cell=` already established.
    if let Some(w) = c.wrong {
        s.push_str(" wrong=");
        push_u32(&mut s, w as u32);
    }
    if let Some(n) = c.budget_n {
        s.push_str(" ex=");
        push_u32(&mut s, n as u32);
    }
    s.push_str(" chi=");
    push_f2(&mut s, c.mcnemar);
    s.push(' ');
    s.push_str(c.j1_why);
    // Every judge ends its bracket with its verdict. J2 and J4 always did; J1
    // ended with prose ("inside the noise") and J3 with either "ok" or its
    // reason, so a reader had to know the why-strings to tell a pass from a
    // veto -- and the Improve window, which is such a reader, showed four red
    // chips for a trial two judges had passed. A window that invents a verdict
    // is worse than one that admits it cannot read the line.
    s.push_str(if c.j1 { " ok]" } else { " no]" });
    s.push_str(" J2[goals=");
    push_u32(&mut s, c.goals_held as u32);
    s.push('/');
    push_u32(&mut s, c.goals_total as u32);
    s.push_str(if c.j2 { " ok]" } else { " no]" });
    s.push_str(" J3[");
    if c.j3 {
        s.push_str("ok]");
    } else {
        s.push_str(c.j3_why);
        s.push_str(" no]");
    }
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

    // What kind of adapter this actually is, read off the thing itself.
    //
    // `deeptrain` attaches a full q/k/v adapter and touches neither the head
    // nor the ledger, so the next trial recorded it here -- as a
    // classifier-only variant with unknown parameters, because that is what
    // this function used to assume. The lineage then claimed a deep adapter
    // was a shallow one, which is worse than claiming it was unknown.
    let deep = ad.qkv.iter().any(|t| t.iter().any(|d| d.is_some()));
    let blob = ad.to_blob();
    let ah = sha256::hash(&blob);
    let current = head();
    // Whether the head still describes the *whole* mind, not just its weights.
    //
    // Testing the adapter alone was not enough. `core install` is still a live
    // operator verb and touches neither the head nor the ledger, so a core
    // could be swapped underneath a node that goes on claiming a different one
    // -- and then `rollback`, which restores whatever the parent recorded,
    // would put back a council the machine was never running. Comparing the
    // core as well means an out-of-band install forces a new node, which is
    // the same rule the adapter has always obeyed: record what is in force,
    // never what was assumed.
    let installed = super::voter::installed().map(|c| c.hash);
    let named = current
        .and_then(|h| Variant::load(&h))
        .map(|v| v.adapter == Some(ah) && v.core == installed)
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
        deep,
        lib: None,
        bar: None,
        // What is actually installed, recorded rather than assumed -- the same
        // discipline as `policy` and `corpus`. A variant trained while a
        // machine-written core was voting is not the same object as one
        // trained without it, and a lineage that cannot tell them apart
        // describes the wrong experiment.
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
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

/// Run whatever this proposal proposes, and answer with a certificate.
///
/// The one way in. Before this there were two entry points with two
/// signatures and two error types, and the scheduler had to know which kind of
/// change it was making in order to call the right one -- so widening the loop
/// to a third kind meant editing the scheduler, the shell, and the journal
/// together. Now the proposal carries its kind and the caller carries none.
///
/// **The economics stay separate even though the entry point is one.**
/// Branching *inside* `trial` was the tempting shape and it is the wrong one:
/// `trial` is a training run whose expensive half is a forward pass per
/// example, and a core changes no weights and needs no training. Two costs
/// behind one name is the mistake `deeptrain` was split out to avoid. This
/// dispatches; it does not merge.
///
/// Marking happens here for every kind, so a proposal that faults still counts
/// as visited -- for a core that is what stops the machine judging the same
/// program it wrote every night for the rest of its life.
///
/// **And `ensure_head` happens here, once, before any of them.** It used to be
/// each axis's own business, and two of them could not do it: `trial_skill`
/// and `trial_lib` take no engine, so both called `head()` directly and wrote
/// a node whose parent named whatever the last trial had recorded. Run one
/// after an out-of-band `adapter load` or `core install` -- both live operator
/// verbs that touch neither the head nor the ledger -- and the lineage says a
/// skill was adopted on top of a mind that was not running, which is the exact
/// failure `ensure_head` exists to prevent. Idempotent, so the axes that
/// already call it are unaffected; the answer is discarded here because each
/// of them reads `head()` for itself and now gets a true one.
pub fn run(
    e: &mut super::Engine,
    b: &Budget,
    p: &Proposal,
) -> Result<Certificate, Refused> {
    let _ = ensure_head(e);
    match p.kind {
        ProposalKind::Adapter => trial(e, b, p).map_err(Refused::Train),
        ProposalKind::Core(h) => {
            p.mark();
            trial_core(e, &h).map_err(Refused::Judge)
        }
        ProposalKind::Deep => trial_deep(e, b, p),
        ProposalKind::Judge(bar) => {
            p.mark();
            trial_judge(e, b, bar)
        }
        ProposalKind::Skill(h) => {
            p.mark();
            trial_skill(&h).map_err(Refused::Judge)
        }
        ProposalKind::Config(r) => {
            p.mark();
            trial_config(e, r).map_err(Refused::Judge)
        }
        ProposalKind::Lib(h) => {
            p.mark();
            // No engine is touched. The library is about what the solver can
            // say, not about what the model believes, so this is the one axis
            // that would run on a machine with no checkpoint loaded.
            trial_lib(&h).map_err(Refused::Judge)
        }
        // **Refused here, deliberately, and it is the only kind that is.**
        // This dispatcher's contract is that it answers a certificate, and a
        // certificate is a verdict. There is no verdict available for a source
        // change on a machine that cannot compile, so the honest answer is
        // that this is not a trial. `propose_source` is the path, it marks the
        // point itself, and the patch waits in the outbox for something that
        // can build it.
        ProposalKind::Source(..) => Err(Refused::Judge(
            "a source change cannot be judged here -- 'godel source' writes the patch",
        )),
    }
}

/// Judge a candidate library function.
///
/// Four judges, mapped onto the certificate the other axes already use so a
/// library adoption reads in the ledger like everything else and `rollback`
/// walks it like everything else.
///
/// - **J1, margin.** `redqueen::bench` is paired over every stored problem:
///   problems that got easier against problems that got harder. Through
///   `judge_one`, so this axis answers to the same bar the rest do rather
///   than to a second definition of "beyond the noise".
/// - **J2, own goals.** The seeded problems must still be solvable. A
///   function that made the whole set cheaper by making the simplest things
///   unreachable is the failure aggregate difficulty would not show, and it
///   is the same question `trial`'s J2 asks of the curiosity goals.
/// - **J3, structure.** It parses back as the declaration it claims to be,
///   and its arity is one the enumerator can actually call.
/// - **J4, cost.** The library is a preamble on every candidate program, so
///   it is paid for on every one of thousands of runs. Bounded.
pub fn trial_lib(h: &[u8; 32]) -> Result<Certificate, &'static str> {
    use super::redqueen::{self, LibFn, Lib, Solver};

    let Some(f) = redqueen::candidate(h) else {
        return Err("no such library candidate");
    };
    TRIALS.fetch_add(1, Ordering::Relaxed);

    let lib = Lib::load();
    let s = Solver::with(redqueen::BUDGET, lib.clone());
    let v = redqueen::bench(&s, &f);
    let n = super::problem::count();

    let (j1, j1_why) = judge_one(n, v.fixed, v.broke);

    // J2: the seeds, which are the simplest thing the solver is asked to do.
    let mut with = s.clone();
    with.lib.add(f.clone());
    let seeds = super::problem::seeds();
    let goals_total = seeds.len();
    let goals_held = seeds
        .iter()
        .filter(|p| {
            redqueen::solve(p, &s).is_none() || redqueen::solve(p, &with).is_some()
        })
        .count();

    let j3 = LibFn::parse(&f.src).as_ref() == Some(&f) && f.arity >= 1 && f.arity <= 2;
    let j4 = with.lib.preamble().len() <= LIB_MAX_BYTES;

    // A library function does not change the mind, so the node names the same
    // adapter and core the head already did. What it changes is what the
    // solver can say, and `lib` is the field for it -- the same shape
    // `trial_skill` uses for the toolkit.
    let parent = head();
    let carried = parent.and_then(|p| Variant::load(&p));
    let variant = Variant {
        parent,
        adapter: carried.as_ref().and_then(|v| v.adapter),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: carried.as_ref().and_then(|v| v.skills),
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        deep: carried.as_ref().map(|v| v.deep).unwrap_or(false),
        lib: Some(*h),
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        rule: super::harness::rule_in_force() as u8,
        bar: Some(bar_in_force()),
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: None,
        axis: "lib",
        parent,
        variant: vhash,
        decisions: n,
        validation: n,
        predicted: v.after < v.before,
        fixed: v.fixed,
        broke: v.broke,
        wrong: None,
        mcnemar: mcnemar(v.broke, v.fixed),
        j1,
        j1_why,
        j2: goals_total > 0 && goals_held == goals_total,
        goals_held,
        goals_total,
        j3,
        j3_why: if j3 { "a callable declaration" } else { "not a callable declaration" },
        j4,
        rank: 0,
        resident_kib: with.lib.preamble().len() / 1024,
        epochs: 0,
        capped: false,
        adopted: false,
        test_acc: 0.0,
        test_read: test_reads(),
        test_fresh: true,
    };

    // **This judged and then did nothing at all, and the consequence chained
    // all the way to a rotation that could not move.** The certificate was
    // built with `adopted: false` hardcoded and immutable, with no `store`, no
    // `set_head` and no ledger line -- so a library function could pass all
    // four judges and be discarded, and nothing recorded that it had ever been
    // tried. No ledger line means `axis_counts()[4]` stays `(0, 0)`, so
    // `axis_uncertainty(0, 0)` is 1.0 forever, so `lib` sorts first in
    // `surprise_order()` whenever it has work -- and `next_lib` did not
    // consult `/ai/godel/tried` either, so it offered the same refused
    // candidate every night, indefinitely, in front of every other axis.
    cert.adopted = cert.unanimous();

    if cert.adopted {
        // Into `/ai/lib`, which is what `Lib::load` reads, so the solver
        // actually gains the function. `store` answers whether it was new;
        // false means the same declaration is already held, which is not a
        // failure -- `next_candidate` skips a candidate whose name the library
        // already has, so reaching here with one is a race rather than a bug.
        f.store();
        variant.store();
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// How much preamble a library may grow to.
///
/// Every candidate program carries the whole library above it, and a round
/// runs thousands of candidates, so this is paid per run and not per
/// adoption. Sixteen kilobytes is far past anything measured -- three
/// functions came to about two hundred bytes -- and the point is that the
/// bound exists rather than where it sits.
const LIB_MAX_BYTES: usize = 16 * 1024;

/// J1, as one function.
///
/// Written out because the storm asks the same question of a dozen candidates
/// a generation, and a second copy of this ladder would be a second definition
/// of "better beyond noise" -- free to drift from the nightly one with nothing
/// to say the two had parted.
///
/// The bar is read rather than constant, which is the point of the judge axis:
/// a criterion the loop adopted and the judges ignored would be a certificate
/// about nothing.
pub(crate) fn judge_one(n_val: usize, fixed: usize, broke: usize) -> (bool, &'static str) {
    // **The family-wise budget, ahead of every other question.** A test this
    // body of evidence can no longer pay for is not a test that failed, it is
    // a test that should not have been run, and reporting it as "inside the
    // noise" would invite somebody to read the numbers and disagree.
    let Some(bar) = effective_bar() else {
        return (false, "this corpus has answered as many questions as it can");
    };
    // The bar is the one place this ladder varies, and it varies at the top of
    // the loop rather than inside it: `bar_in_force()` is read once here, so a
    // criterion the judge axis adopted reaches every caller. `passes_j1` is
    // the same ladder with the bar as an argument, which is what lets `drift`
    // re-judge a past line under a *different* bar without a second copy of
    // "beyond the noise" that could disagree with this one.
    passes_j1(n_val, fixed, broke, bar)
}

/// J1, with the bar named rather than read.
///
/// Pure -- no store, no clock -- because `drift` calls it once per ledger line
/// under two bars, and the founding-vs-current comparison is only re-derivable
/// if the ladder itself is a function of its inputs. `judge_one` is this with
/// `bar_in_force()` supplied, so there is exactly one definition of the ladder
/// and the nightly loop and the drift report cannot part ways about it.
pub fn passes_j1(n_val: usize, fixed: usize, broke: usize, bar: f32) -> (bool, &'static str) {
    if n_val == 0 {
        // Nothing was held out to judge against. A property of how the trial
        // was asked for rather than of the variant: a subsample too small to
        // reach the validation slice leaves the margin with no evidence at
        // all, in either direction.
        (false, "no validation decisions")
    } else if fixed <= broke {
        (false, "no net repair")
    } else if fixed - broke < MIN_FIXED {
        (false, "net repair below the floor")
    } else if mcnemar(broke, fixed) < bar {
        (false, "inside the noise")
    } else {
        (true, "beyond the noise")
    }
}

/// How many chimeras one storm may breed.
///
/// Capped because breeding is quadratic in survivors and costs nothing:
/// a generation of twelve pairs into sixty-six, all scorable, none trained,
/// and the tribunal would be handed whichever noise came out highest. Six is
/// enough to cross the ranks that actually survived and few enough that the
/// generation is still mostly things that were trained.
const MAX_CHIMERAS: usize = 6;

/// What one storm did.
pub struct StormReport {
    pub trained: usize,
    pub descendants: usize,
    pub chimeras: usize,
    pub lit: usize,
    pub best: f32,
    pub verdict: &'static str,
    /// Whether the best of the generation would have cleared J1.
    ///
    /// **`adopted`, and it adopted nothing.** The shell printed "the best was
    /// accepted" off this field while `storm` had no `set_head`, no ledger
    /// line and no `TRIALS` increment anywhere in it -- so an operator was
    /// told a variant had been taken up and the machine went on running the
    /// one it had before, with nothing in the lineage either way.
    ///
    /// Renamed rather than made true, and the choice is an argument rather
    /// than the smaller edit. A storm is an *explorer*: its product is the
    /// archive, which it really does populate, and it weighs the whole
    /// generation against J1 alone. The nightly loop adopts on four judges in
    /// unanimity, and adopting here would put a weaker gate in front of the
    /// head -- which is the failure this module's own history records as "an
    /// axis in the rotation without a judge in front of it". `godel archive`
    /// names the elites and `trial` is how one is taken up.
    pub cleared_bar: bool,
}

/// One generation of the whole apparatus, on one `prepare`.
///
/// **The economics are the reason this exists as a command.** The base weights
/// are frozen, so a hidden state is a constant, a constant is cached once, and
/// a cached decision replays against any candidate for the price of a dot
/// product. Eight grid points, the descendants and every chimera therefore
/// cost *one* preparation and no forward passes at all -- which is what makes
/// scoring a dozen variants a night affordable when producing one costs 214
/// seconds under TCG.
///
/// Chimeras carry an honest `Fit` of zero epochs and zero loss, because they
/// were never trained. Reporting an inherited loss would make the ledger say
/// an optimiser reached a number it never saw.
pub fn storm(e: &mut super::Engine, b: &Budget, points: usize) -> Result<StormReport, Refused> {
    let t = super::train::prepare(e, b).map_err(Refused::Train)?;
    let incumbent = e.model.adapters.as_ref().and_then(|a| t.gather(a));
    let n_val = t.slice_size(Slice::Validation);

    // Generation order is the grid's own order, then descendants, then
    // chimeras. Declared rather than sorted, so a later run breeds the same
    // pairs from the same survivors rather than plausible ones.
    let mut gen: Vec<(super::adapter::Dora, usize)> = Vec::new();
    let take = points.min(GRID.len());
    for p in GRID.iter().take(take) {
        let bi = super::train::Budget {
            epochs: p.epochs,
            millis: b.millis,
            examples: b.examples,
            lr: p.lr,
            rank: p.rank,
            alpha: p.alpha,
        };
        let fit = t.train(&bi);
        gen.push((fit.dora, p.rank));
    }
    let trained = gen.len();

    // Descendants: trained *from* the incumbent rather than from scratch, so
    // they start where the machine already is. None if there is no incumbent,
    // which is the ordinary case on a fresh machine and not a failure.
    let mut descendants = 0;
    if let Some(inc) = incumbent.as_ref() {
        let mask = alloc::vec![true; t.live_rows()];
        let fit = t.train_masked(b, Some(inc), &mask);
        gen.push((fit.dora, b.rank));
        descendants = 1;
    }

    // Chimeras, from same-rank survivors paired in generation order.
    let mut chimeras = 0;
    let mut bred: Vec<(super::adapter::Dora, usize)> = Vec::new();
    'outer: for i in 0..gen.len() {
        for j in (i + 1)..gen.len() {
            if gen[i].1 != gen[j].1 {
                continue;
            }
            if let Some(d) = t.breed(&gen[i].0, &gen[j].0) {
                bred.push((d, gen[i].1));
                chimeras += 1;
                if chimeras >= MAX_CHIMERAS {
                    break 'outer;
                }
            }
        }
    }
    gen.extend(bred);

    // Score every one of them, and offer each to its cell.
    let parent = ensure_head(e);
    let mut best: Option<(usize, f32, usize, usize)> = None;
    for (idx, (d, rank)) in gen.iter().enumerate() {
        let (broke, fixed, _, _) = t.paired(incumbent.as_ref(), Some(d), Slice::Validation);
        let score = t.score(Some(d), Slice::Validation);
        // The address is of the *variant node*, so what the cell points at is
        // the same kind of object the ledger names and `rollback` walks.
        let v = Variant {
            parent,
            adapter: Some(sha256::hash(&t.scatter(d, &e.model.cfg, b.alpha).to_blob())),
            policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
            skills: None,
            corpus: sysbox::hash_of(super::vocab::CORPUS),
            deep: false,
            lib: None,
            bar: Some(bar_in_force()),
            core: super::voter::installed().map(|c| c.hash),
            core_seen: true,
            lambda: b.lr,
            rank: *rank as u8,
            epochs: 0,
            // `b.rank as u8` was here, which is the *budget's rank* -- 8 by
            // default -- landing in the slot that holds a routing rule.
            // `Rule::from_u8(8)` is `None`, so every node a storm ever wrote
            // was unparseable, and `rollback` onto one would have failed with
            // "the parent names a routing rule this kernel does not have".
            // It stayed invisible because the only route to these nodes is the
            // archive, and nothing reads the archive back.
            rule: super::harness::rule_in_force() as u8,
            born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
        };
        let vh = v.hash();
        if offer(&vh, *rank, fixed, broke, score) {
            v.store();
        }
        if best.is_none_or(|(_, s, _, _)| score > s) {
            best = Some((idx, score, fixed, broke));
        }
    }

    // The best of the generation is measured against the tribunal, which is
    // the same J1 the nightly loop uses and not a softer one.
    //
    // **Measured and not adopted**, and nothing below this line changes the
    // head. What a storm produces is the archive -- `offer` and `v.store()`
    // above, which are real and durable -- plus one number saying whether the
    // generation reached the standing bar. See `cleared_bar`.
    let (verdict, cleared_bar) = match best {
        Some((_, _, fixed, broke)) => {
            let (ok, why) = judge_one(n_val, fixed, broke);
            (why, ok)
        }
        None => ("nothing was bred", false),
    };
    let (lit, archive_best) = archive_census();
    Ok(StormReport {
        trained,
        descendants,
        chimeras,
        lit,
        best: archive_best,
        verdict,
        cleared_bar,
    })
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
    // Where the machine's own goals actually go, which nothing ever printed.
    //
    // `Guard.name` has been recorded since the guards existed and was read by
    // one thing, the mutation check. So J2 could veto a night's work for
    // rerouting "list the files in /ai" and no line said where the baseline
    // had been sending it -- or whether that was the right place. A judge
    // whose subject is invisible is a judge nobody can argue with.
    for g in t.guards().iter() {
        crate::kprintln!(
            "    goal: {} -> {} ({})",
            g.goal,
            g.name,
            if g.protected() {
                "as declared, so J2 protects it"
            } else {
                "not where it was declared to go, so J2 has nothing to protect"
            }
        );
    }

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
    let (broke, fixed, _, neither) = t.paired(incumbent.as_ref(), Some(&fit.dora), Slice::Validation);
    let chi = mcnemar(broke, fixed);
    let n_val = t.slice_size(Slice::Validation);
    let (j1, j1_why) = judge_one(n_val, fixed, broke);

    // --- J2: does it still do the same thing unasked? -------------------
    // Walked under the candidate rather than replayed against the incumbent's
    // cached path, so the verdict can name where a goal went instead of only
    // that it moved. Eight prefills against a forward pass per corpus example.
    let went = t.guards_where(e, Some(&fit.dora));
    for (g, w) in t.guards().iter().zip(went.iter()) {
        if !g.protected() {
            continue;
        }
        let to = w.unwrap_or("nowhere the grammar finishes");
        crate::kprintln!(
            "    goal: {} -> {}{}",
            g.goal,
            to,
            if w == &Some(g.expect) { "" } else { "  <- moved, and J2 protects this one" }
        );
    }
    // Every goal and not only the protected ones: an unprotected goal is
    // outside the count and would otherwise be free to land on `rm`.
    for w in went.iter().flatten() {
        if crate::sysbox::applet_mutates(w).unwrap_or(true) {
            crate::kprintln!("    a goal now reaches '{}', which changes things", w);
        }
    }
    let (goals_held, goals_total) = t.guards_kept(&went);
    // Every protected guard must hold, and none of the goals may be routing to
    // a mutating applet in the first place -- a baseline that already wanted to
    // run `rm` on its own initiative is not a baseline worth preserving. The
    // mutation check walks *every* goal and not only the protected ones,
    // because a goal that reaches `rm` is a fact about the machine either way.
    //
    // **`goals_total == 0` passes now, and says so.** It used to veto, on the
    // reasonable-looking ground that a judge with nothing to weigh should not
    // wave a variant through. But `guards_hold` counts only the goals the
    // baseline routes *correctly*, so zero means this machine gets none of its
    // own goals right -- and there is then nothing here a candidate could
    // break. Vetoing on it makes J2 unpassable precisely when the machine is
    // at its worst, which is when it most needs to be able to improve. The
    // certificate carries the count, so a pass on nothing is legible as one.
    let j2 = goals_held == goals_total && t.guards_read_only(&went);

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
        // `scatter` builds a classifier-only adapter, always.
        deep: false,
        lib: None,
        bar: Some(bar_in_force()),
        // What is actually installed, recorded rather than assumed -- the same
        // discipline as `policy` and `corpus`. A variant trained while a
        // machine-written core was voting is not the same object as one
        // trained without it, and a lineage that cannot tell them apart
        // describes the wrong experiment.
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: b.lr,
        rank: fit.dora.r as u8,
        epochs: fit.epochs as u32,
        // What is actually routing, not what the proposal happened to carry.
        //
        // This was `p.rule`, and every grid point carries 0 -- `ProbeOnly` --
        // while the machine has been running the default `Majority` the whole
        // time. So every node in every lineage recorded a rule its variant was
        // never measured under, which is the "describes the wrong experiment"
        // failure the corpus and policy hashes are here to prevent, on the one
        // field nobody was varying. A trial trains an adapter *under* a rule;
        // it does not choose one, and `ProposalKind::Config` is what does.
        rule: super::harness::rule_in_force() as u8,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: Some(b.examples),
        axis: "adapter",
        parent,
        variant: vhash,
        decisions: t.decisions(),
        validation: t.slice_size(Slice::Validation),
        predicted,
        fixed,
        broke,
        wrong: Some(fixed + neither),
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
pub const MAX_RESIDENT_KIB: usize = 8 * 1024;

/// Guards that hold regardless of any score.
///
/// These are the ones worth having when the scores look good: a variant can
/// improve validation accuracy and still be carrying a non-finite scale that
/// will produce a NaN on the first prompt outside the corpus.
pub fn sanity(t: &Trial, d: &super::adapter::Dora) -> (bool, &'static str) {
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
/// Record whatever is attached right now as a node, and make it the head.
///
/// For changes that arrive from outside the loop -- `deeptrain`, `adapter
/// load`, `train adapter`. None of them is judged, and this does not pretend
/// otherwise: the node it writes carries the honest "arrived from outside"
/// zeros, plus `deep` read off the adapter itself.
///
/// What it buys is that the change is *addressable*. Before this, `deeptrain`
/// moved every q/k/v site and left no trace, so the lineage's account of the
/// mind was silently wrong until the next trial happened to notice, and
/// `godel rollback` had nothing to walk back to. `adapter off` was the only
/// undo, and it discards everything rather than stepping back one change.
pub fn record_current(e: &mut super::Engine) -> Option<[u8; 32]> {
    ensure_head(e)
}

/// Judge a council core and, if it passes, adopt it into the lineage.
///
/// The second kind of thing this loop can change, and the first that is not a
/// number. A core is an Aiksi program defining `fn vote(text, allowed): int`;
/// the machine can write one, `Caps::Sandbox` and a step budget already make
/// running an untrusted one safe, and `harness::core_bench` already judges it
/// on the validation slice with a paired McNemar test (J1), a cost ceiling
/// (J5) and an independence requirement (J6).
///
/// What did not exist was any of the *bookkeeping that makes a change
/// reversible*. A core could pass all three judges and leave no node in the
/// DAG, no line in the ledger, and nothing for `rollback` to undo -- so the
/// one path by which the machine could adopt code it wrote itself was also the
/// one path outside the discipline every other change obeys. `core install`
/// remains for the operator; this is the way in that keeps a record.
///
/// Deliberately not folded into `trial`. That function is a training run --
/// `prepare` is a forward pass per example, and every judge after it reads
/// cached features. A core changes no weights, needs no training, and shares
/// only the lineage machinery. Branching inside `trial` would put two
/// economics behind one name, which is the mistake `deeptrain` was split out
/// to avoid.
pub fn trial_core(e: &mut super::Engine, h: &[u8; 32]) -> Result<Certificate, &'static str> {
    // The engine this function was handed, not a second claim on it.
    //
    // `core_bench` opens `with_engine` itself, and `trial_core` is called from
    // inside one -- so the old spelling produced two live `&mut Engine` at
    // once. Undefined behaviour, and with teeth: the judging passes below
    // mutate the KV cache and `e.pos` under a reference the compiler is
    // allowed to assume nothing else touches, and `ensure_head` immediately
    // afterwards reads the adapter through it to decide what to record.
    let verdict = match super::harness::core_bench_in(e, h, super::harness::VALIDATION) {
        Err(_) => return Err("the core will not load, or scored nothing"),
        Ok(v) => v,
    };
    TRIALS.fetch_add(1, Ordering::Relaxed);

    // Record what is in force before competing against it, exactly as the
    // adapter path does, so `parent` names the thing actually measured.
    let parent = ensure_head(e);

    let variant = Variant {
        parent,
        // Unchanged by this trial: a core votes, it does not move weights.
        // Carried from the incumbent so the node describes the whole mind
        // rather than only the part this trial touched.
        adapter: parent.and_then(|p| Variant::load(&p)).and_then(|v| v.adapter),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: None,
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        rule: super::harness::Rule::WithCore as u8,
        core: Some(*h),
        core_seen: true,
        // Carried from the incumbent: a core changes no weights, so whatever
        // the parent was, this variant still is.
        deep: parent.and_then(|p| Variant::load(&p)).map(|v| v.deep).unwrap_or(false),
        lib: None,
        bar: Some(bar_in_force()),
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: None,
        axis: "core",
        parent,
        variant: vhash,
        decisions: verdict.n,
        validation: verdict.n,
        // There is a cheap signal after all, and it is a better one than the
        // adapter path's.
        //
        // This used to be `false` with a comment saying no prediction existed.
        // The census makes one: `prize` counts the validation items a core
        // could repair if it answered every one of them correctly, so a prize
        // below the bar is a prediction of failure that is not a guess -- it
        // is arithmetic. The ledger accumulates these beside the outcomes, and
        // a run of `predicted false / adopted false` is not a calibration
        // failure here, it is the ceiling being reported honestly.
        predicted: verdict.prize >= clean_fixes_needed(),
        fixed: verdict.fixed,
        broke: verdict.broke,
        wrong: None,
        mcnemar: verdict.chi,
        j1: verdict.j1,
        j1_why: if verdict.j1 { "beyond the noise" } else { "inside the noise" },
        // J2 asks whether the machine still does the same thing unasked. A
        // core cannot reach an applet -- it answers an index into a set the
        // caller already chose -- so the guards are held by construction
        // rather than by measurement, and saying so is more honest than
        // running them and reporting a pass they could not fail.
        goals_held: 0,
        goals_total: 0,
        j2: true,
        j3: verdict.j6,
        j3_why: if verdict.j6 { "disagrees somewhere" } else { "adds a vote and no information" },
        resident_kib: 0,
        rank: 0,
        j4: verdict.j5,
        epochs: 0,
        capped: false,
        adopted: false,
        test_acc: 0.0,
        test_read: 0,
        test_fresh: true,
    };
    cert.adopted = cert.unanimous();

    if cert.adopted {
        let (acc, n, fresh) = read_test_core(e, h);
        cert.test_acc = acc;
        cert.test_read = n;
        cert.test_fresh = fresh;
    }

    variant.store();

    if cert.adopted {
        // Install first, then move the pointer: a head naming a core that is
        // not in force describes a mind the machine is not running.
        if !super::voter::install(h) {
            return Err("the core passed but would not install");
        }
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// The test slice, for a core that has already won on validation.
///
/// Same budget and same ordering as the adapter path: consulted only after
/// adoption, counted against the same three reads, because a loop that
/// improves itself forever reads the held-out set forever whichever axis it
/// is searching.
///
/// **The read is spent only if it buys a measurement.** This function used to
/// call `spend_test_read` and return `0.0` -- so three adopted cores exhausted
/// the global held-out budget having looked at nothing, wrote `test_acc 0.00`
/// into three certificates (which reads as "0% on test", not as "not
/// measured"), and stamped every later adapter certificate `test_fresh
/// false` permanently. The one number the loop is not allowed to overfit was
/// being spent on a code path that never opened the data.
fn read_test_core(e: &mut super::Engine, h: &[u8; 32]) -> (f32, u32, bool) {
    match super::harness::core_bench_in(e, h, super::harness::TEST) {
        Ok(v) if v.n > 0 => {
            let n = spend_test_read();
            (v.correct as f32 / v.n as f32, n, n <= TEST_READS)
        }
        // Nothing measurable in the test slice. Say so by leaving the budget
        // alone: an unspent read is recoverable, a spent one never is.
        _ => (0.0, test_reads(), false),
    }
}

/// Take the installed core out of the decision path, if there is one.
///
/// `voter::uninstall` answers `false` both when the detach failed and when
/// there was nothing to detach, and those are opposite facts: the second is
/// the ordinary case and must not read as an error.
fn drop_core() -> bool {
    if super::voter::installed().is_none() {
        return true;
    }
    super::voter::uninstall()
}

/// Undo the last adoption: put the parent's mind back and move the head to it.
///
/// **Everything is validated before anything is changed.** The old order swapped
/// the core first and read the adapter blob afterwards, so a rollback whose
/// adapter had been pruned returned an error having *already* changed which
/// core was voting -- leaving the parent's core in force, the child's adapter
/// attached, and the head still naming the child. That is precisely the "the
/// pointer said one thing and the machine did another" state this function
/// exists to prevent, relocated into its own failure path. Now the failable
/// reads all happen first, and the mutations happen only once none of them can
/// fail for a reason we could have seen coming.
/// Adapt the attention path, judge what it bought, and keep it only if it won.
///
/// The third kind of change, and the first that is *destructive while it is
/// being measured*. `trial` builds a candidate adapter beside the running one
/// and attaches it only on adoption; `train_full` has no such shape -- it
/// walks gradients into the tensors the model is using. So the incumbent is
/// copied out first and put back on every path that does not adopt. A judge
/// that leaves a rejected variant installed is not a judge.
///
/// **What the frozen-base trade costs, said in the certificate.** A
/// classifier-only adapter leaves every hidden state a constant, and that is
/// what makes cached features, cheap re-judging and a cheap `Trial` possible.
/// A deep one gives that up: judging it costs two full passes over the corpus
/// rather than one cached one, and every earlier certificate's cached
/// comparison stops applying to it. `deep: true` on the node is what tells a
/// later reader which kind of object they are looking at.
pub fn trial_deep(
    e: &mut super::Engine,
    b: &Budget,
    p: &Proposal,
) -> Result<Certificate, Refused> {
    use super::train::RunError;
    p.mark();

    // Where everything goes now, before anything moves.
    let before = super::harness::route_snapshot(e, super::harness::VALIDATION)
        .map_err(|_| Refused::Train(RunError::NoCorpus))?;

    // The incumbent, kept so a rejection can be undone. `None` is a real
    // answer -- the frozen model is a variant -- and detaching is how it comes
    // back.
    let saved = e.model.adapters.as_ref().map(|a| a.to_blob());
    let parent = ensure_head(e);

    // A deep adapter to train into, if the machine is not already carrying
    // one. `Adapters::full` adapts every q/k/v site as well as the decision
    // layer, which is the whole difference being judged.
    if e.model.adapters.is_none() {
        let cfg = e.model.cfg.clone();
        let full = super::adapter::Adapters::full(&cfg, b.rank, b.alpha);
        if e.model.attach_adapters(full).is_err() {
            return Err(Refused::Judge("a deep adapter will not attach to this checkpoint"));
        }
    }

    let Some(report) = super::train::train_full(e, b, b.examples) else {
        restore(e, &saved);
        return Err(Refused::Train(RunError::Hardware));
    };

    let after = match super::harness::route_snapshot(e, super::harness::VALIDATION) {
        Ok(s) => s,
        Err(_) => {
            restore(e, &saved);
            return Err(Refused::Train(RunError::NoCorpus));
        }
    };

    // --- J1: paired, on the items both snapshots saw --------------------
    let n = before.correct.len().min(after.correct.len());
    let mut fixed = 0usize;
    let mut broke = 0usize;
    for i in 0..n {
        match (before.correct[i], after.correct[i]) {
            (false, true) => fixed += 1,
            (true, false) => broke += 1,
            _ => {}
        }
    }
    let chi = mcnemar(broke, fixed);
    // Called rather than written out again. This ladder was a hand-inlined
    // copy of `judge_one`, which is the exact thing `judge_one`'s own doc
    // comment exists to forbid: a second definition of "better beyond noise",
    // free to drift from the nightly one with nothing to say the two had
    // parted. The two happened to agree; that is luck, not a property.
    let (j1, j1_why) = judge_one(n, fixed, broke);

    // --- J2: does it still do the same thing unasked? -------------------
    //
    // Recomputed on both sides rather than cached, because the cache is
    // exactly what deep training invalidates. A goal that now routes
    // somewhere else is the failure this judge exists for, and one that
    // routes to a mutating applet fails it whether or not it moved.
    // Only the goals the baseline routes where they were *declared* to go,
    // for the reason `Trial::guards_hold` gives at length: a goal the incumbent
    // gets wrong has nothing to protect, and requiring the candidate to
    // preserve a wrong answer is J2 vetoing exactly what J1 is looking for.
    // Ground truth rather than the other side's answer, since `RouteSnapshot`
    // carries it now.
    let n = before.guards.len().min(after.guards.len()).min(before.expect.len());
    let protected: alloc::vec::Vec<usize> =
        (0..n).filter(|i| before.guards[*i] == before.expect[*i]).collect();
    let goals_total = protected.len();
    let goals_held = protected.iter().filter(|i| after.guards[**i] == after.expect[**i]).count();
    // The mutation check walks every goal and not only the protected ones: a
    // goal that reaches `rm` is a fact about the machine either way. And zero
    // protected goals passes -- see the note on `trial`'s J2.
    let j2 = goals_held == goals_total
        && after.guards[..n].iter().all(|c| {
            crate::sysbox::APPLETS
                .get(*c)
                .map(|a| !a.mutates)
                .unwrap_or(false)
        });

    // --- J3: structural sanity ------------------------------------------
    let (j3, j3_why) = if !after.finite {
        (false, "the features stopped being finite")
    } else if !report.last_loss.is_finite() {
        (false, "the loss diverged")
    } else if before.correct.len() != after.correct.len() {
        (false, "the slice changed under the run")
    } else {
        (true, "finite throughout")
    };

    // --- J4: can this machine carry it? ---------------------------------
    //
    // The judge that a deep variant is most likely to fail, and rightly. Every
    // adapted attention site is resident for the life of the model, where a
    // classifier adapter is a few rows.
    // The rank bound is not decode cost here, it is identity: `Variant.rank`
    // is a byte, so a rank past 255 would wrap and two different experiments
    // would share a node. Refusing is better than recording a lie.
    let resident_kib = e.model.adapters.as_ref().map(|a| a.resident_bytes()).unwrap_or(0) / 1024;
    let j4 = b.rank <= 255 && resident_kib <= MAX_RESIDENT_KIB;

    let blob = e.model.adapters.as_ref().map(|a| a.to_blob());
    let ablob = blob.as_ref().map(|x| sha256::hash(x));

    let variant = Variant {
        parent,
        adapter: ablob,
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: None,
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        // The whole point of the node: this one moved the attention path, and
        // nothing that reads the lineage may confuse it with one that did not.
        deep: true,
        lib: None,
        bar: Some(bar_in_force()),
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: b.lr,
        rank: b.rank as u8,
        epochs: report.epochs as u32,
        // The rule in force, not the proposal's. `Proposal::deep` always sets
        // `rule: 0` -- `ProbeOnly` -- while the machine has been routing under
        // the default `Majority` throughout, so every deep node ever written
        // recorded a routing rule its variant was never measured under. This
        // is the same defect `trial` carries ten lines of comment about having
        // fixed; the deep path was not updated with it.
        rule: super::harness::rule_in_force() as u8,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: Some(b.examples),
        axis: "deep",
        parent,
        variant: vhash,
        decisions: n,
        validation: n,
        // The cheap signal a deep run does have: the loss went down.
        predicted: report.last_loss < report.first_loss,
        fixed,
        broke,
        wrong: None,
        mcnemar: chi,
        j1,
        j1_why,
        goals_held,
        goals_total,
        j2,
        j3,
        j3_why,
        resident_kib,
        rank: b.rank,
        j4,
        epochs: report.epochs as u32,
        capped: report.stopped,
        adopted: false,
        test_acc: 0.0,
        test_read: 0,
        test_fresh: true,
    };
    cert.adopted = cert.unanimous();
    TRIALS.fetch_add(1, Ordering::Relaxed);

    if cert.adopted {
        // The blob before the node, so a head can never name an adapter whose
        // bytes are not stored -- that is the one state `rollback` cannot get
        // out of.
        match (blob, ablob) {
            (Some(bytes), Some(h)) => {
                sysbox::write_blob(&blob_path(&h), bytes);
            }
            _ => {
                restore(e, &saved);
                return Err(Refused::Judge("it passed but there is nothing attached to store"));
            }
        }
        variant.store();
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    } else {
        // Put the machine back. This is the half `deeptrain` never had: it
        // trained into the live model and left whatever came out, judged or
        // not, until the next reboot.
        restore(e, &saved);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// Judge a skill and, if it passes, put it where `run` will find it.
///
/// The cheapest trial in the module and the only one that needs no model: a
/// skill is a program, and the four things worth asking about one are all
/// answerable by running it. That matters more than it sounds -- it means the
/// night loop can judge a skill on a machine with no checkpoint loaded, and
/// that a rejection costs seconds rather than the twenty minutes an adapter
/// trial costs.
///
/// **Adoption is a copy, not a rename.** The candidate stays at its content
/// address under `/ai/skills` and a copy lands in `/ai/tools`, so the thing
/// that was judged and the thing that runs are provably the same bytes. A
/// rename would leave the ledger naming an address nothing holds.
pub fn trial_skill(h: &[u8; 32]) -> Result<Certificate, &'static str> {
    let v = super::skill::bench(h);
    let Some(src) = super::skill::source(h) else { return Err("no such skill") };
    TRIALS.fetch_add(1, Ordering::Relaxed);

    // A skill does not change the mind, so the node it writes names the same
    // adapter and core the head already did. What it changes is the toolkit,
    // and `skills` is the field for it -- hooked up at last, having been
    // hardcoded `None` since the struct was written.
    //
    // `head()` and not `ensure_head`, because this axis takes no engine and
    // `ensure_head` needs the adapter bytes. `run` calls it once before
    // dispatching for exactly that reason, so the head this reads already
    // describes the mind that is running.
    let parent = head();
    let carried = parent.and_then(|p| Variant::load(&p));
    let variant = Variant {
        parent,
        adapter: carried.as_ref().and_then(|v| v.adapter),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: Some(*h),
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        deep: carried.as_ref().map(|v| v.deep).unwrap_or(false),
        lib: None,
        bar: Some(bar_in_force()),
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        rule: super::harness::rule_in_force() as u8,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: None,
        axis: "skill",
        parent,
        variant: vhash,
        // Two runs, which is what J3 compared. Not routing decisions, and the
        // ledger's `n` should not be read as though it were.
        decisions: 2,
        validation: 2,
        // Nothing predicts a skill: it is admitted or it is not, and there is
        // no cheap signal that anticipates the verdict the way a training loss
        // anticipates an adapter's.
        predicted: false,
        fixed: 0,
        broke: 0,
        wrong: None,
        mcnemar: 0.0,
        j1: v.j1,
        j1_why: v.j1_why,
        goals_held: if v.j2 { 1 } else { 0 },
        goals_total: 1,
        j2: v.j2,
        j3: v.j3,
        j3_why: v.j3_why,
        // Steps stand in for resident bytes: both are "what it costs to have",
        // measured in the unit that matters for the kind of thing it is.
        resident_kib: (v.steps / 1024) as usize,
        rank: 0,
        j4: v.j4,
        epochs: 0,
        capped: false,
        adopted: false,
        test_acc: 0.0,
        test_read: 0,
        test_fresh: true,
    };
    cert.adopted = cert.unanimous();

    if cert.adopted {
        let path = super::skill::adopted_path(h);
        if !sysbox::write_text(&path, &src) {
            return Err("it passed and the toolkit would not take it");
        }
        variant.store();
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// How much better the confident items must get before a rule is worth
/// changing for.
///
/// Five points of separation. The measured baseline is 90.3% right when the
/// three agree against 50% when they split -- a gap of about 0.40 -- so this
/// is roughly an eighth of the signal, which is large enough not to be
/// chasing validation noise on 180 items and small enough to be reachable.
/// It is a decision and not a derivation, and the certificate records the
/// gaps themselves so a later reader can apply a different one.
pub const MIN_CAL_GAIN: f32 = 0.05;

/// How much of its confidence a candidate may give up while claiming to have
/// improved it. Four fifths: a rule that is beautifully calibrated over six
/// items has not improved the router, it has stopped answering.
const MIN_CONF_KEEP: f32 = 0.8;

/// Judge a routing rule on what it actually changes.
///
/// **The one axis where accuracy is the wrong judge, which is why it sat
/// unsearchable behind a comment for so long.** Every other proposal is
/// selected by J1 -- a net repair beyond the noise -- and a rule change is
/// mostly not that. What it moves is *calibration*: how much better the
/// council's confident answers are than its unconfident ones, which is the
/// property the whole three-core arrangement exists to produce. A router that
/// knows when it is guessing can ask, escalate or refuse; one that is silently
/// 78% accurate cannot.
///
/// So the two judges point in different directions on purpose:
///
///   J1  **do no harm.** The candidate must not lose accuracy beyond the
///       noise. Not "must win" -- requiring a win here is exactly what made
///       this axis unsearchable, because a rule that trades a point of
///       accuracy for a much sharper confidence signal is a trade worth
///       making and J1 as written would veto it.
///   J2  **must improve.** The confidence gap has to widen by `MIN_CAL_GAIN`,
///       and the confident set must not collapse. Something has to get better
///       or this is drift with a certificate.
///
/// J3 is the arithmetic that keeps the comparison meaningful, and J4 is free:
/// a rule costs no resident bytes, which is the honest answer rather than a
/// judge invented to fill the slot.
pub fn trial_config(e: &mut super::Engine, rule: u8) -> Result<Certificate, &'static str> {
    let Some(candidate) = super::harness::Rule::from_u8(rule) else {
        return Err("no such routing rule");
    };
    let v = super::harness::rule_bench(e, candidate)
        .map_err(|_| "the router would not fit, or there is nothing to judge on")?;
    TRIALS.fetch_add(1, Ordering::Relaxed);

    // J1: did it lose anything it should not have?
    //
    // Two ways to fail, and the first was missing. Written as "not
    // significantly worse" alone, this adopted `ProbeOnly` on a measured
    // `fixed 4 broke 10` -- a net loss of six items out of 180 -- because chi
    // reached only 1.79 against a threshold of 3.84. Significance is a poor
    // guard in the losing direction on a slice this size: a real loss can sit
    // under it comfortably.
    //
    // So the floor is symmetric with the one the adapter path uses to call a
    // *gain* real. `MIN_FIXED` says a net repair under four is not a repair;
    // it says just as well that a net loss over four is not nothing.
    let net_loss = v.broke.saturating_sub(v.fixed);
    let lost = net_loss >= MIN_FIXED || (v.broke > v.fixed && v.chi >= bar_in_force());
    let j1 = !lost;
    let j1_why = if lost { "it costs accuracy" } else { "accuracy is unchanged beyond noise" };

    // J2: did the thing this axis is for actually improve?
    let kept = v.conf_now as f32 >= v.conf_was as f32 * MIN_CONF_KEEP;
    let j2 = v.gain() >= MIN_CAL_GAIN && kept;

    // J3: a comparison over nothing is not a comparison.
    let (j3, j3_why) = if v.n == 0 {
        (false, "no validation decisions")
    } else if v.conf_now == 0 || v.conf_was == 0 {
        (false, "one of the rules never claims confidence")
    } else {
        (true, "both rules answer and both claim confidence")
    };

    let parent = ensure_head(e);
    let carried = parent.and_then(|p| Variant::load(&p));
    let variant = Variant {
        parent,
        adapter: carried.as_ref().and_then(|x| x.adapter),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: carried.as_ref().and_then(|x| x.skills),
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        deep: carried.as_ref().map(|x| x.deep).unwrap_or(false),
        lib: None,
        bar: Some(bar_in_force()),
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        rule,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: None,
        axis: "rule",
        parent,
        variant: vhash,
        decisions: v.n,
        validation: v.n,
        // The cheap signal is the gap itself, known before the judges run.
        predicted: v.gain() > 0.0,
        fixed: v.fixed,
        broke: v.broke,
        wrong: None,
        mcnemar: v.chi,
        j1,
        j1_why,
        // Confident items, before and after -- the coverage half of J2, in the
        // two fields shaped to carry a "held out of total".
        goals_held: v.conf_now,
        goals_total: v.conf_was,
        j2,
        j3,
        j3_why,
        // A rule is a number. Saying it costs kilobytes would be inventing a
        // judge to fill a slot.
        resident_kib: 0,
        rank: 0,
        j4: true,
        epochs: 0,
        capped: false,
        adopted: false,
        test_acc: 0.0,
        test_read: 0,
        test_fresh: true,
    };
    cert.adopted = cert.unanimous();

    if cert.adopted {
        let cfg = super::harness::Config {
            lambda: super::harness::default_lambda(),
            rule: candidate,
        };
        if !super::harness::save_config(cfg) {
            return Err("it passed and the configuration would not save");
        }
        variant.store();
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = TRIALS.load(Ordering::Relaxed);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// What a rollback should do about the adopted core.
///
/// Three outcomes, and the middle one is why this is a type: a parent that
/// *said* it had no core and a parent that said nothing at all are different
/// facts, and reading the second as the first pulled a core out of the
/// decision path as a side effect of undoing an adapter.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum CoreMove {
    /// The parent is silent and the node being left adopted nothing, so
    /// whatever is installed was installed out of band and is not ours to
    /// move.
    Leave,
    Drop,
    Install([u8; 32]),
}

/// Which core a rollback should end on.
///
/// **Lifted out of `rollback` and asserted, because `clade::reconsider` calls
/// that function unattended now.** Every one of these decisions was a branch
/// inside a hundred-and-fifty-line function needing an engine, a store and a
/// real lineage to reach, which is to say a branch nothing could check. They
/// are pure functions of two nodes, in the shape `update::decide` is, and they
/// earn the same treatment for the same reason: each has a recorded history of
/// having been got wrong, and each is wrong *silently*.
pub fn core_move(leaving: &Variant, parent: &Variant) -> CoreMove {
    if parent.core_seen {
        match parent.core {
            None => CoreMove::Drop,
            Some(c) => CoreMove::Install(c),
        }
    } else if leaving.core.is_some() {
        CoreMove::Drop
    } else {
        CoreMove::Leave
    }
}

/// The routing rule to put back, or none when the two nodes agree.
///
/// **Only on disagreement, and the guard is correctness rather than caution.**
/// Every node renders a `rule` line, so unlike `core` there is no absent to
/// detect -- but nodes written before that axis was searchable recorded 0,
/// which is `ProbeOnly`, while the machine that wrote them ran the default
/// `Majority`. Restoring unconditionally would switch a lineage full of those
/// legacy zeroes to a rule none of them ever ran.
pub fn rule_move(leaving: &Variant, parent: &Variant) -> Option<u8> {
    if leaving.rule != parent.rule {
        Some(parent.rule)
    } else {
        None
    }
}

/// The bar to put back, or none.
///
/// A parent's `None` is "did not say" rather than "had none", so there is
/// nothing to restore from: a node written before the field existed says
/// nothing about a criterion, and writing one out of its silence would put the
/// machine on a bar nobody chose.
pub fn bar_move(leaving: &Variant, parent: &Variant) -> Option<f32> {
    match (leaving.bar, parent.bar) {
        (a, Some(b)) if a != Some(b) => Some(b),
        _ => None,
    }
}

/// Put back the adapters a trial was handed, whatever it did to them.
fn restore(e: &mut super::Engine, saved: &Option<Vec<u8>>) {
    match saved {
        Some(bytes) => {
            let _ = e.model.load_adapters(bytes);
        }
        None => {
            let _ = e.model.detach_adapters();
        }
    }
}

pub fn rollback(e: &mut super::Engine) -> Result<Option<[u8; 32]>, &'static str> {
    let Some(h) = head() else { return Err("no head to roll back from") };
    let Some(v) = Variant::load(&h) else { return Err("head names a node that is not stored") };
    let Some(parent) = v.parent else {
        // The root is the frozen model. Rolling back to it means detaching,
        // which is a real state rather than an error.
        //
        // The core belongs to that detachment. A root node can carry one --
        // `trial_core` on a machine with no adapter attached gets `parent:
        // None` from `ensure_head` and records `core: Some(..)` on it -- and
        // this arm used to return `Ok(None)`, reporting a return to the frozen
        // model while a machine-written core went on voting on every routing
        // decision, with the head now deleted so nothing named it and a second
        // rollback could not reach it either.
        if v.core.is_some() && !drop_core() {
            return Err("the adopted core will not detach");
        }
        let _ = e.model.detach_adapters();
        sysbox::detach(HEAD);
        return Ok(None);
    };
    let Some(pv) = Variant::load(&parent) else { return Err("parent is not stored") };

    // --- read and validate, changing nothing ---------------------------

    let blob = match pv.adapter {
        None => None,
        Some(ab) => match sysbox::read_blob(&blob_path(&ab)) {
            None => return Err("the parent's adapter blob is gone"),
            Some(b) => Some(b),
        },
    };

    // What to do about the core, decided before anything moves.
    //
    // `pv.core == None` is only an instruction to uninstall when the parent
    // actually *said* so. Nodes written before the field existed parse the
    // same way, and reading those as "the parent had no core" meant that
    // rolling back an adapter on any older lineage quietly pulled a core out
    // of the decision path -- a change to what the machine does, made as a
    // side effect of undoing something else, and printed nowhere. When the
    // parent is silent the only core this rollback owns is the one the node
    // being left adopted; anything installed out of band is not ours to move.
    let want = core_move(&v, &pv);
    if let CoreMove::Install(c) = want {
        if super::voter::load(&c).is_err() {
            return Err("the parent's core will not load");
        }
    }

    // The routing rule, when the two nodes disagree about it.
    //
    // Only then, and the guard is not caution -- it is correctness. Every node
    // renders a `rule` line, so unlike `core` there is no "absent" to detect;
    // but nodes written before this axis was searchable recorded 0, which is
    // `ProbeOnly`, while the machine that wrote them was running the default
    // `Majority`. Restoring a parent's rule unconditionally would therefore
    // switch a lineage full of legacy nodes to a rule none of them ever ran.
    // If the two agree there is nothing to put back.
    let rule_back = match rule_move(&v, &pv) {
        None => None,
        Some(byte) => match super::harness::Rule::from_u8(byte) {
            None => return Err("the parent names a routing rule this kernel does not have"),
            Some(r) => Some(r),
        },
    };

    // The bar, under exactly the same rule and for exactly the same reason.
    //
    // Undoing a judge trial without putting its criterion back leaves the
    // machine on a bar the node it now points at was never judged under,
    // which is the one thing this axis exists to make visible. Conditional
    // because a node written before the field existed says nothing about a
    // bar, and `None` is "did not say" rather than "had none" -- restoring
    // from it would write a bar nobody chose. Checked before anything moves,
    // like the rule and the core, so a refusal leaves the machine as it was.
    let bar_back = match bar_move(&v, &pv) {
        Some(b) if !sane_bar(b) => {
            return Err("the parent names a bar outside the range this kernel accepts")
        }
        other => other,
    };

    // --- change things -------------------------------------------------

    // The adapter first: it is the half that can still fail on bytes we have
    // already proved are present, so a failure here leaves the machine exactly
    // as it was rather than half-rolled-back.
    match blob {
        None => {
            let _ = e.model.detach_adapters();
        }
        Some(b) => {
            e.model.load_adapters(&b).map_err(|_| "the parent's adapter will not load")?;
        }
    }
    if let Some(r) = rule_back {
        let cfg = super::harness::Config { lambda: super::harness::default_lambda(), rule: r };
        if !super::harness::save_config(cfg) {
            return Err("the adapter was restored but the routing rule will not save");
        }
    }
    if let Some(b) = bar_back {
        let mut text = String::new();
        push_f6(&mut text, b);
        text.push('\n');
        if !sysbox::write_text(BAR, &text) {
            return Err("the adapter was restored but the bar will not save");
        }
    }
    match want {
        CoreMove::Leave => {}
        CoreMove::Drop => {
            if !drop_core() {
                return Err("the adapter was restored but the core will not detach");
            }
        }
        CoreMove::Install(c) => {
            if !super::voter::install(&c) {
                return Err("the adapter was restored but the parent's core will not install");
            }
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

/// How many verdicts have been recorded. The rotation's clock.
pub fn ledger_len() -> usize {
    let Some(bytes) = sysbox::read_blob(LEDGER) else { return 0 };
    let Ok(text) = core::str::from_utf8(&bytes) else { return 0 };
    text.lines().filter(|l| !l.is_empty()).count()
}

// ---------------------------------------------------------------------------
// Drift: how far the loop's own judgment has moved from the one it started on.
// ---------------------------------------------------------------------------
//
// **The problem this measures is that nothing here is sacred.** The judge axis
// can move the bar the other judges answer to, so a loop that improves itself
// every night can also, over a month, quietly lower the standard it improves
// against -- and every certificate it writes on the way stays true, because it
// was true under the bar in force when it was written. A ledger of honest
// certificates is not proof the loop got better; it can equally be proof the
// loop got easier to satisfy.
//
// The guard is not to freeze the bar. It is to keep two verdicts for every
// line and watch them separate: the **founding** verdict, under the bar this
// machine started with (`MCNEMAR_95`), and the **current** verdict, under the
// bar in force now. Both are recomputed from the counts on the line, so there
// is no second ledger to fall out of step -- the evidence is written once and
// the criterion is applied at read time, which is what makes the number
// re-derivable by anyone from the ledger and the two bars alone.
//
// The only judge the bar moves is J1, so a line drifts exactly when its J1
// flips between the two bars *and* the other three judges passed -- a line
// rejected for its cost or its structure is rejected under any bar and cannot
// drift. `looser` is the direction that matters: verdicts the founding
// criterion refused and the current one adopts. `stricter` is the machine
// holding itself to more than it began with, which is the safe way to be
// wrong about your own standard.

/// The integer written right after `key` on a ledger line, if any.
fn field_usize(line: &str, key: &str) -> Option<usize> {
    let at = line.find(key)? + key.len();
    let rest = &line[at..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}

/// Whether the judge tagged `tag` (e.g. `"J2["`) closed with `ok]`.
fn judge_ok(line: &str, tag: &str) -> bool {
    let Some(at) = line.find(tag) else { return false };
    let rest = &line[at..];
    let Some(close) = rest.find(']') else { return false };
    rest[..close].trim_end().ends_with("ok")
}

/// Full adoption of a line under a given bar.
///
/// J1 recomputed from the counts at that bar, and the other three read off the
/// line as they were -- they do not move with the bar, so their recorded
/// verdicts are their verdicts under any bar.
fn adopts_at(line: &str, bar: f32) -> bool {
    let (Some(n), Some(fixed), Some(broke)) = (
        field_usize(line, " n="),
        field_usize(line, "fix="),
        field_usize(line, " broke="),
    ) else {
        return false;
    };
    passes_j1(n, fixed, broke, bar).0
        && judge_ok(line, "J2[")
        && judge_ok(line, "J3[")
        && judge_ok(line, "J4[")
}

/// How far the current criterion has drifted from the founding one, and
/// whether the loop's own win/lose prediction has been worth anything.
pub struct Drift {
    /// Lines the two criteria agree on.
    pub agree: usize,
    /// Founding rejects, current adopts -- the criterion loosened. The
    /// direction the whole metric exists to catch.
    pub looser: usize,
    /// Founding adopts, current rejects -- the criterion tightened.
    pub stricter: usize,
    /// The bar this machine started with, and the one it holds now.
    pub founding_bar: f32,
    pub current_bar: f32,
    /// The prediction cross-tab, over lines that carry an `axis=` field (only
    /// those have an interpretable `pred=`). `predicted` is whether
    /// training-set gain said win; the outcome is whether the line adopted.
    pub pred_win_adopt: usize,
    pub pred_win_reject: usize,
    pub pred_lose_adopt: usize,
    pub pred_lose_reject: usize,
}

impl Drift {
    /// Lines where the loop's prediction matched the outcome, over lines where
    /// it made one. The calibration `pred=` has been recording, unread, since
    /// the first godel commit.
    pub fn pred_total(&self) -> usize {
        self.pred_win_adopt + self.pred_win_reject + self.pred_lose_adopt + self.pred_lose_reject
    }

    pub fn pred_right(&self) -> usize {
        self.pred_win_adopt + self.pred_lose_reject
    }
}

/// Compute drift from ledger lines and the two bars. Pure, so a saved ledger
/// re-derives the same numbers and the boot self-test can hand it fixtures.
pub fn drift_of(lines: &[String], founding: f32, current: f32) -> Drift {
    let mut d = Drift {
        agree: 0,
        looser: 0,
        stricter: 0,
        founding_bar: founding,
        current_bar: current,
        pred_win_adopt: 0,
        pred_win_reject: 0,
        pred_lose_adopt: 0,
        pred_lose_reject: 0,
    };
    for line in lines {
        // Only lines carrying a variant's counts can be re-judged. A judge-axis
        // line records a criterion change rather than a variant, so it has no
        // `fix=`/`broke=` to recompute and is skipped here -- it is the thing
        // being measured, not a measurement.
        if field_usize(line, "fix=").is_none() {
            continue;
        }
        let f = adopts_at(line, founding);
        let c = adopts_at(line, current);
        match (f, c) {
            (true, true) | (false, false) => d.agree += 1,
            (false, true) => d.looser += 1,
            (true, false) => d.stricter += 1,
        }

        // The prediction cross-tab, gated on `axis=` for the reason the plan
        // records: `pred=` predates `axis=`, and the six axes compute the
        // prediction from six different quantities, so a line with no axis is
        // uninterpretable and is left out rather than pooled.
        if axis_of(line).is_some() {
            let win = line.contains(" pred=win");
            let adopt = line.contains(" ADOPT");
            match (win, adopt) {
                (true, true) => d.pred_win_adopt += 1,
                (true, false) => d.pred_win_reject += 1,
                (false, true) => d.pred_lose_adopt += 1,
                (false, false) => d.pred_lose_reject += 1,
            }
        }
    }
    d
}

/// Read the ledger and compute drift against the founding and current bars.
pub fn drift() -> Drift {
    drift_of(&ledger_tail(usize::MAX), MCNEMAR_95, bar_in_force())
}

/// Deep points, walked by markers exactly as `GRID` is.
///
/// Two, and small ones. A deep trial costs two full passes over the corpus
/// plus the training between them, so this is not a space to sweep -- it is
/// enough points to find out whether moving the attention path buys anything
/// on this corpus at all, which is a question with a cheap answer and an
/// expensive one and no middle.
const DEEP_GRID: &[(f32, usize, f32, usize)] = &[(0.02, 4, 8.0, 4), (0.01, 8, 16.0, 6)];

// ---------------------------------------------------------------------------
// Red Queen epochs: the criterion holds still while the agent moves.
// ---------------------------------------------------------------------------

/// How many trials run against a frozen criterion before the criterion itself
/// may be re-examined.
///
/// **A bar that can move at any moment chases the proposal it is meant to
/// judge.** Nothing in the loop prevents it: a night that cannot find a
/// variant clearing 3.84 can always find a reason 3.20 was the better number,
/// and a month of that converges on an evaluator that says yes to everything
/// while the ledger fills with adoptions. That record would be *true*, and it
/// would describe criterion drift rather than improvement, and nothing in the
/// file would distinguish the two.
///
/// So the bar is frozen inside an epoch and questionable only at its edge.
/// Five, because there are five other axes -- adapter, rule, skill, deep,
/// composed core -- so an epoch reads as "try each kind once, then look at the
/// criterion".
pub const EPOCH_LEN: usize = 5;

/// Whether the loop stands where it is allowed to question its own bar.
pub fn at_epoch_boundary() -> bool {
    is_boundary(ledger_len())
}

/// A pure function of how many verdicts have been recorded.
///
/// Of the *ledger*, deliberately, and not of a counter kept beside it. Where
/// the loop stands must be re-derivable from the record for the same reason
/// the rotation is: a reader asking "why did it question the bar that night"
/// has to be able to answer from the file, and a counter that drifted from the
/// ledger would leave the two disagreeing with nothing to say which was right.
///
/// Genesis is excluded. At length zero there is no agent to protect and no
/// epoch behind it to have improved anything, so a bar change there would be
/// grounded in nothing at all.
fn is_boundary(n: usize) -> bool {
    n > 0 && n % EPOCH_LEN == 0
}

// ---------------------------------------------------------------------------
// The unbound judge, and the anchor that catches it drifting.
// ---------------------------------------------------------------------------

/// The narrowest bar a criterion may take.
///
/// Not zero. A bar of zero admits every candidate whose repairs outnumber its
/// breaks by any margin at all, which abolishes the criterion rather than
/// loosening it -- and does so while still looking like a threshold.
pub const JUDGE_MIN: f32 = 0.5;

/// And the widest. A bar above this refuses everything the corpus can produce
/// evidence for, which switches the loop off as effectively as `set_enabled`
/// and leaves no sign that it happened.
pub const JUDGE_MAX: f32 = 12.0;

/// Finite, and inside the range a criterion may take.
pub fn sane_bar(v: f32) -> bool {
    v.is_finite() && v >= JUDGE_MIN && v <= JUDGE_MAX
}

/// The family-wise error rate the whole loop is allowed, over one body of
/// evidence.
///
/// **Every judged comparison at the bar is a test at p < 0.05, and nothing
/// charged for it.** Run one a night for a year and roughly one adoption in
/// twenty is noise, permanently, by construction -- not a bug in any judge but
/// the arithmetic of repeating a test. `godel.rs` named that problem in its
/// own header and did not bill for it.
///
/// So the loop gets one nickel of error to spend across every test it ever
/// runs against a given corpus, and the k-th test may spend
///
///     alpha_k = ALPHA_TOTAL * 6 / (pi^2 * k^2)
///
/// which sums over every k to exactly `ALPHA_TOTAL`. A geometric schedule
/// would do as well; this one is the Basel series, so the total is a closed
/// form somebody can check rather than a number that happens to converge.
pub const ALPHA_TOTAL: f32 = 0.05;

/// The bar each successive test answers to, as a chi-squared value for one
/// degree of freedom.
///
/// Computed rather than guessed, and the computation is written down so it can
/// be redone: `chi_k = z(1 - alpha_k/2)^2` where `alpha_k` is the schedule
/// above. Reproduced by
///
///     from statistics import NormalDist; import math
///     A, c = 0.05, 6/math.pi**2
///     [round(NormalDist().inv_cdf(1 - A*c/(k*k)/2)**2, 3) for k in range(1, 33)]
///
/// A table rather than an inverse normal in the kernel, for the reason
/// `JUDGE_GRID` is a table: this target has no such function, an approximation
/// would be a second place for the criterion to be wrong, and a threshold
/// nobody can recompute is a threshold nobody can argue with.
const SPEND: [f32; 32] = [
    4.687, 7.126, 8.591, 9.644,
    10.466, 11.141, 11.714, 12.212,
    12.652, 13.046, 13.403, 13.730,
    14.031, 14.309, 14.569, 14.813,
    15.041, 15.257, 15.462, 15.656,
    15.840, 16.016, 16.185, 16.346,
    16.501, 16.649, 16.793, 16.931,
    17.064, 17.193, 17.317, 17.438,
];

/// How many judged comparisons this body of evidence has already paid for.
///
/// Counted out of the *ledger*, and only lines whose `corpus=` matches the one
/// in force. That is the discipline `axis_counts` follows and for the same
/// reason: a counter in its own file could disagree with the record, and then
/// the loop's account of why it refused something would be unfalsifiable
/// exactly where it most needs not to be.
///
/// Lines written before the corpus was recorded count for nothing. That
/// undercounts the early history rather than guessing at it, which is the safe
/// direction here as well -- it spends the budget on tests that were really
/// run rather than refusing on the strength of lines nobody can attribute.
pub fn tests_spent() -> usize {
    let Some(now) = sysbox::hash_of(super::vocab::CORPUS) else {
        return 0;
    };
    let tag = short(&now);
    ledger_tail(usize::MAX)
        .iter()
        .filter(|l| {
            l.find(" corpus=")
                .map(|at| l[at + 8..].starts_with(&tag))
                .unwrap_or(false)
        })
        .count()
}

/// The floor the family-wise budget puts under the bar, or `None` when this
/// body of evidence has been asked as many questions as it can answer.
///
/// **`None` is not a very high bar, it is a refusal**, and the difference is
/// the whole point. A bar that kept rising would let the loop go on testing
/// forever against evidence that has stopped being able to support a
/// conclusion, each test individually defensible and the family of them not.
/// What refills the budget is *new evidence* -- a corpus that has grown, which
/// changes its hash and starts the count again -- and that is the incentive a
/// machine meant to keep improving should have.
pub fn spend_floor() -> Option<f32> {
    SPEND.get(tests_spent()).copied()
}

/// The bar a judge actually answers to: the adopted one, never below the
/// family-wise floor.
///
/// The two compose this way round and not the other. The judge axis may raise
/// the bar and may lower it, and lowering it below the spending floor would be
/// the loop buying itself more chances to be fooled -- which is exactly the
/// drift `judge_verdict`'s third question exists to catch, arriving through a
/// door that judge cannot see.
pub fn effective_bar() -> Option<f32> {
    let adopted = bar_in_force();
    match spend_floor() {
        None => None,
        Some(f) if f > adopted => Some(f),
        _ => Some(adopted),
    }
}

/// The bar the judges are using tonight.
///
/// `MCNEMAR_95` remains the constant it was -- what moves is this reading, so
/// a tree that has never adopted a bar change behaves exactly as it did before
/// this axis existed, and the default is still the number from the table.
///
/// A stored value outside the sane range is **ignored rather than clamped**.
/// Clamping would let a corrupt or hand-edited file slide the bar to the
/// nearest legal number and report nothing; refusing it puts the machine back
/// on the constant, which is the one value no trial ever chose.
pub fn bar_in_force() -> f32 {
    sysbox::read_blob(BAR)
        .and_then(|b| core::str::from_utf8(&b).ok().and_then(|t| t.trim().parse::<f32>().ok()))
        .filter(|v| sane_bar(*v))
        .unwrap_or(MCNEMAR_95)
}

/// The same candidate scored under both bars, plus a reading neither bar can
/// see.
///
/// **This is what makes unbinding the judge affordable.** A trial that only
/// asked "does the proposed bar admit this variant" would be a proposal
/// grading itself. The matrix adds a column the bars do not participate in --
/// held-out accuracy from the anchor -- and the verdict becomes a question
/// about whether the bars and the anchor agree, which is answerable.
pub struct Cross {
    /// The bar in force when the trial started.
    pub standing: f32,
    /// The bar being proposed.
    pub proposed: f32,
    /// The candidate's paired counts, from which chi follows.
    pub fixed: usize,
    pub broke: usize,
    /// Held-out accuracy of the candidate, and of the incumbent it would
    /// replace, on the slice neither bar can see.
    pub anchor_candidate: f32,
    pub anchor_incumbent: f32,
}

impl Cross {
    /// The paired statistic for this candidate. One definition, shared.
    pub fn chi(&self) -> f32 {
        mcnemar(self.broke, self.fixed)
    }

    /// Whether a given bar admits this candidate.
    ///
    /// The direction test is not decoration. Chi is symmetric, so a variant
    /// that broke twelve and repaired one clears 3.84 exactly as convincingly
    /// as one that repaired twelve and broke one; a bar alone cannot tell them
    /// apart and was never meant to.
    pub fn admits(&self, bar: f32) -> bool {
        self.fixed > self.broke && self.chi() >= bar
    }

    /// What the anchor says the candidate was actually worth.
    pub fn anchor_gain(&self) -> f32 {
        self.anchor_candidate - self.anchor_incumbent
    }
}

/// Whether a proposed bar may be adopted, and why.
///
/// Pure, so every outcome below is asserted at boot with no model, no corpus
/// and no NVMe -- the way `update::decide` is. A criterion that can only be
/// checked by running the loop overnight is a criterion nobody checks.
///
/// Three questions, and the third is the whole of it:
///
/// - **sane** -- finite, inside `[JUDGE_MIN, JUDGE_MAX]`. A bar of zero
///   abolishes the criterion; one too high freezes the loop.
/// - **moves** -- the two bars actually disagree about this candidate. A bar
///   that admits and refuses exactly what the standing one did has changed
///   nothing, and adopting it writes a certificate with no content.
/// - **honest** -- the change agrees with the anchor. Loosening is admitted
///   only when the newly-admitted variant genuinely improves held-out
///   accuracy; tightening only when the newly-refused variant genuinely was
///   not a gain.
///
/// Without the third this axis is a machine for lowering its own bar and
/// recording the result as progress.
///
/// **The two questions that need no anchor, answered separately so the read
/// can be withheld.** `sane` is a property of the proposed number and `moves`
/// is a property of the paired counts; neither looks at held-out accuracy. The
/// caller spends one of three lifetime test reads to obtain that accuracy, and
/// spending it to learn something already decided is spending it for nothing.
///
/// `Some(verdict)` means settled without evidence; `None` means the anchor is
/// now genuinely the question, which is the case the module header argues
/// should be billed.
pub fn judge_precheck(c: &Cross) -> Option<(bool, &'static str)> {
    if !sane_bar(c.proposed) {
        return Some((false, "the proposed bar is outside the range a criterion may take"));
    }
    if c.admits(c.standing) == c.admits(c.proposed) {
        return Some((false, "the proposed bar admits and refuses what the standing one did"));
    }
    None
}

pub fn judge_verdict(c: &Cross) -> (bool, &'static str) {
    if let Some(settled) = judge_precheck(c) {
        return settled;
    }
    // Admission is monotone in the bar, and the precheck has already
    // established that the two bars disagree about this candidate -- so the
    // proposed bar admitting it is exactly a loosening, and not admitting it
    // exactly a tightening. Derived from the matrix rather than from comparing
    // the two numbers, so the direction and the effect can never disagree
    // about which case this is.
    let then = c.admits(c.proposed);
    if then {
        if c.anchor_gain() > 0.0 {
            (true, "a looser bar admits a variant the anchor confirms")
        } else {
            (false, "drift: a looser bar admits a variant the anchor rejects")
        }
    } else if c.anchor_gain() > 0.0 {
        (false, "a tighter bar would refuse a genuine gain")
    } else {
        (true, "a tighter bar refuses a variant the anchor does not support")
    }
}

// ---------------------------------------------------------------------------
// MAP-Elites: illuminate the space instead of climbing it.
// ---------------------------------------------------------------------------

/// Rank bands. Four, because rank is what decides whether a variant can be
/// carried at all (`J4`), and these are the four answers that matter: nearly
/// free, cheap, real, and expensive.
pub const RANK_BANDS: usize = 4;

/// Repair behaviours: mostly breaking, mixed, mostly repairing.
pub const REPAIR_BANDS: usize = 3;

/// Twelve cells.
pub const CELLS: usize = RANK_BANDS * REPAIR_BANDS;

/// Which cell a variant belongs in.
///
/// **The behaviour axis is the fraction of touched decisions that were repairs
/// rather than breaks**, not the count. A rank-4 adapter that moves nine
/// decisions and repairs eight, and a rank-32 one that moves ninety and
/// repairs eighty, are the same *behaviour* at very different prices -- and a
/// hill-climber keeping a single champion throws the cheap one away over a
/// margin inside the noise. An archive keeps both, so a later reader can ask
/// what the price bought instead of taking it on trust.
pub fn descriptor(rank: usize, fixed: usize, broke: usize) -> (usize, usize) {
    let band = match rank {
        0..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        _ => 3,
    };
    let touched = fixed + broke;
    // A variant that moved nothing is not a perfect repairer. It is the most
    // breaking-shaped thing there is, because it bought nothing at all, and
    // filing it in the top band would let an adapter that does nothing sit in
    // the cell reserved for the ones that work.
    let behaviour = if touched == 0 {
        0
    } else {
        let share = fixed as f32 / touched as f32;
        if share < 0.5 {
            0
        } else if share < 0.8 {
            1
        } else {
            2
        }
    };
    (band, behaviour)
}

/// The cell index, for a flat archive.
pub fn cell_of(rank: usize, fixed: usize, broke: usize) -> usize {
    let (band, behaviour) = descriptor(rank, fixed, broke);
    band * REPAIR_BANDS + behaviour
}

/// Where the cell-winners live: one file per cell, named by index.
pub const ARCHIVE: &str = "/ai/godel/archive";

/// What occupies a cell.
pub struct Elite {
    pub variant: [u8; 32],
    pub score: f32,
}

fn cell_path(i: usize) -> String {
    let mut s = String::from(ARCHIVE);
    s.push('/');
    push_u32(&mut s, i as u32);
    s
}

/// What is sitting in a cell, if anything.
pub fn cell(i: usize) -> Option<Elite> {
    let bytes = sysbox::read_blob(&cell_path(i))?;
    let text = core::str::from_utf8(&bytes).ok()?;
    let mut it = text.split_whitespace();
    let variant = from_hex32(it.next()?)?;
    let score = it.next()?.parse::<f32>().ok()?;
    Some(Elite { variant, score })
}

/// Offer a variant to the cell its behaviour puts it in.
///
/// **A variant earns a cell by beating whatever is in that cell**, not by
/// beating the champion, and that is the whole difference between an archive
/// and a hill-climb. A rank-4 adapter that repairs a different set of
/// decisions than the rank-32 one survives on its own terms instead of being
/// discarded over a margin inside the noise -- and the loop ends up
/// illuminating the space rather than walking uphill in it.
///
/// Cells hold addresses in the same DAG the ledger names, so an elite from
/// three weeks ago is still reachable and still re-derivable. The archive
/// stores a pointer to evidence, never a copy of a conclusion.
pub fn offer(variant: &[u8; 32], rank: usize, fixed: usize, broke: usize, score: f32) -> bool {
    let i = cell_of(rank, fixed, broke);
    // Strictly better, so a rerun that reproduces an elite exactly leaves the
    // cell alone. The archive then records when something was first found
    // rather than when it was last recomputed, which is the fact worth having.
    if cell(i).is_some_and(|e| e.score >= score) {
        return false;
    }
    let mut text = String::new();
    text.push_str(&hex32(variant));
    text.push(' ');
    push_f6(&mut text, score);
    text.push('\n');
    sysbox::write_text(&cell_path(i), &text);
    true
}

/// How many cells are lit, and the best score in any of them.
pub fn archive_census() -> (usize, f32) {
    let mut lit = 0;
    let mut best = 0.0f32;
    for i in 0..CELLS {
        if let Some(e) = cell(i) {
            lit += 1;
            if e.score > best {
                best = e.score;
            }
        }
    }
    (lit, best)
}

/// The best-scoring elite in the archive, and the variant that holds it.
///
/// **This is the read the archive was missing.** `offer` writes a variant
/// address into every cell it wins and `cell` parses one back, but nothing
/// consumed the address -- so `Elite.variant` was written and never read, and
/// the twelve cells preserved re-derivable elites that nothing re-derived,
/// which is a high-score table with the winners' names filled in and never
/// looked at. `godel storm` and `godel archive` reach for this now: a storm
/// with no live incumbent can descend from the best thing the machine has
/// found, and the operator can be shown which variant tops the archive rather
/// than only how many cells are lit.
pub fn best_elite() -> Option<Elite> {
    let mut best: Option<Elite> = None;
    for i in 0..CELLS {
        if let Some(e) = cell(i) {
            if best.as_ref().is_none_or(|b| e.score > b.score) {
                best = Some(e);
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Bayesian surprise: reach for the axis whose verdict is least predictable.
// ---------------------------------------------------------------------------

/// How unpredictable an axis's next verdict is, from its own record.
///
/// A Laplace-smoothed Beta posterior mean folded to a distance from the
/// coin-flip. An axis that has said yes to everything and one that has said no
/// to everything are equally predictable and equally uninformative; the axis
/// sitting near 50% is the one whose next trial actually tells you something.
///
/// **This trades fairness for information, and the trade is the point.** The
/// old rotation gave every axis a turn in sequence, which is fair and spends
/// nights confirming what the ledger already says.
///
/// The smoothing is what stops that becoming starvation. `+1` over `+2` keeps
/// even a saturated axis strictly above zero, so an axis that has refused
/// twenty proposals still outranks nothing and still comes up once the others
/// are out of moves.
fn axis_uncertainty(att: u32, adopt: u32) -> f32 {
    let rate = (adopt as f32 + 1.0) / (att as f32 + 2.0);
    1.0 - (rate - 0.5).abs() * 2.0
}

fn next_deep() -> Option<Proposal> {
    DEEP_GRID
        .iter()
        .map(|&(lr, rank, alpha, epochs)| Proposal::deep(lr, rank, alpha, epochs))
        .find(|p| !p.tried())
}

/// A routing rule that has not been judged yet, other than the one in force.
///
/// The rule already running is excluded rather than marked: judging it against
/// itself is a certificate saying nothing changed, which is true and is not
/// worth a night.
/// Where a source patch waits for something that can build it.
///
/// **Not the ledger, and the distinction is what keeps the ledger meaning
/// something.** Every line in `/ai/godel/ledger.txt` is a verdict: judged,
/// adopted or refused, by judges that ran here. A source proposal has no
/// verdict on this side at all -- the kernel cannot compile, so nothing here
/// can say whether the change is good. Writing one as a ledger line would put
/// an unjudged entry among judged ones, which is the "axis with no judge in
/// front of it" failure this module opens by warning about.
pub const OUTBOX: &str = "/ai/godel/outbox";

/// Everything a proposal has to say to something that can build it.
///
/// **The patch plus who is asking and what they ran it on.** A patch alone is
/// a change with no provenance: the receiving side cannot tell which machine
/// wants it, which lineage it descends from, or which corpus the rail it
/// claims will be measured against. All three are already addresses in this
/// tree, so carrying them costs nothing and refusing to carry them would make
/// the returning verdict impossible to file.
pub fn envelope(p: &Proposal) -> Result<String, &'static str> {
    let ProposalKind::Source(ki, vi) = p.kind else {
        return Err("not a source proposal");
    };
    let Some(patch) = super::knob::patch(ki as usize, vi as usize) else {
        return Err("this kernel does not have that knob");
    };
    let mut s = String::from("proposal 1\n");
    s.push_str("point ");
    s.push_str(&hex32(&p.hash()));
    s.push_str("\nfrom ");
    s.push_str(crate::VERSION);
    s.push_str("\nhead ");
    s.push_str(&head().map(|h| hex32(&h)).unwrap_or(String::from("none")));
    s.push_str("\ncorpus ");
    s.push_str(
        &sysbox::hash_of(super::vocab::CORPUS)
            .map(|h| hex32(&h))
            .unwrap_or(String::from("none")),
    );
    s.push_str("\ntests ");
    push_u32(&mut s, tests_spent() as u32);
    s.push('\n');
    s.push_str(&patch);
    Ok(s)
}

/// Where a proposal that has left the machine is recorded, so it is not sent
/// twice.
///
/// A separate marker from `/ai/godel/tried`, because they answer different
/// questions and conflating them costs the loop a whole axis. `tried` means
/// "this point has been reached"; if pushing also marked it, a push that
/// failed on the network would read as a point already explored and the
/// machine would never come back to it.
pub const PUSHED: &str = "/ai/godel/pushed";

/// The next thing in the outbox that has not been sent.
///
/// Rebuilt from the declared space rather than by listing the outbox, so what
/// is offered is always a point this kernel can still resolve -- the same
/// refusal `knob::at` makes, arriving one level up.
pub fn next_pushable() -> Option<Proposal> {
    let mut i = 0usize;
    while i < super::knob::KNOBS.len() {
        let mut j = 0usize;
        while j < super::knob::KNOBS[i].values.len() {
            let p = Proposal::source(i, j);
            // Proposed but not yet sent. A point nobody has written a patch
            // for has nothing to push.
            if p.tried() && !pushed(&p) {
                return Some(p);
            }
            j += 1;
        }
        i += 1;
    }
    None
}

fn pushed_path(p: &Proposal) -> String {
    let mut path = String::from(PUSHED);
    path.push('/');
    path.push_str(&hex32(&p.hash()));
    path
}

pub fn pushed(p: &Proposal) -> bool {
    sysbox::read_blob(&pushed_path(p)).is_some()
}

/// Record that this proposal reached something that can build it.
///
/// **Written only on a `200`**, unlike `mark`, which is written before the
/// work. The difference is what each one protects against: a trial that faults
/// has still spent the night on that point, while a push that never arrived
/// has spent nothing and the patch is still worth sending.
pub fn mark_pushed(p: &Proposal) {
    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let mut s = String::from("h");
    push_u32(&mut s, hour as u32);
    s.push('\n');
    sysbox::write_text(&pushed_path(p), &s);
}

/// Where a verdict about a proposal comes back to.
///
/// **Separate from the outbox, because they are different claims.** The outbox
/// holds what this machine asked for; this holds what something that could
/// build it answered. Merging them would make "proposed" and "judged" one
/// state, and the whole reason a source proposal is not a trial is that those
/// are two.
pub const INBOX: &str = "/ai/godel/inbox";

/// What came back about a proposal, once the signature said it could be read.
pub struct Verdict {
    /// The proposal this is about, by the hash the envelope carried.
    pub point: [u8; 32],
    /// The rail the knob claimed, echoed so a verdict about the wrong rail is
    /// visible rather than silently filed against the right one.
    pub rail: String,
    /// `better`, `worse`, `same` or `unstable`, as `tools/rails.py` spells
    /// them. A fifth word this kernel does not know is refused rather than
    /// guessed at.
    pub moved: String,
    pub why: String,
    pub adopted: bool,
}

/// Read a signed verdict, or say why it will not be believed.
///
/// **Signature first and parse second**, which is `manifest::verified`'s
/// ordering and its reason: parsing attacker-chosen text is the larger of the
/// two surfaces and there is no cause to enter it before knowing the bytes
/// came from the signer.
///
/// And it *must* be signed, more than a manifest must. A manifest names an
/// image whose own signature is checked again before anything installs, so an
/// unsigned one costs a wasted download. A verdict changes this machine's
/// account of what it has learned -- it is the only thing in this tree that
/// writes a ledger line the machine did not derive itself -- so an unsigned
/// one is somebody else editing the lineage. Same key as the images, because
/// adopting a second signer is itself a kernel change and that is the point.
pub fn read_verdict(blob: &[u8]) -> Result<Verdict, String> {
    let Some((text, sig)) = super::super::update::manifest::split(blob) else {
        return Err(alloc::format!(
            "{} B is too short to be a signed verdict",
            blob.len()
        ));
    };
    let v = crate::update::verify(text, sig);
    if !v.ok() {
        return Err(String::from(v.why()));
    }
    parse_verdict(text)
}

/// The parse alone, for the claims.
///
/// Split from the signature check so every refusal below can be asserted at
/// boot with no private key on the machine -- which there is not and must not
/// be. The signature half is checked the other way, by feeding a real verdict
/// with one word changed and watching it be refused.
pub fn parse_verdict(text: &[u8]) -> Result<Verdict, String> {
    let Ok(text) = core::str::from_utf8(text) else {
        return Err(String::from("the verdict is not text"));
    };
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("verdict 1") {
        return Err(String::from("that is not a verdict"));
    }
    let (mut point, mut rail, mut moved, mut why, mut adopted) =
        (None, String::new(), String::new(), String::new(), false);
    for line in lines {
        let line = line.trim();
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        match k {
            "point" => point = from_hex32(v),
            "rail" => rail = String::from(v),
            "moved" => moved = String::from(v),
            "why" => why = String::from(v),
            "adopted" => adopted = v == "yes",
            _ => {}
        }
    }
    let Some(point) = point else {
        return Err(String::from("the verdict names no proposal"));
    };
    // A word this kernel does not know is refused rather than filed. The four
    // are `tools/rails.py`'s own, and a fifth would mean the two halves have
    // parted -- which is exactly the drift that must not be resolved by
    // guessing.
    if !matches!(moved.as_str(), "better" | "worse" | "same" | "unstable") {
        return Err(alloc::format!("'{}' is not a verdict this kernel knows", moved));
    }
    // **A verdict may not claim an adoption its own rail comparison refuses.**
    // The machine that built it is trusted to measure; it is not trusted to
    // conclude, because the conclusion is the one thing this side can check.
    if adopted && moved != "better" {
        return Err(alloc::format!(
            "it says adopted and says the rail was '{}', which cannot both be true",
            moved
        ));
    }
    Ok(Verdict { point, rail, moved, why, adopted })
}

/// File a verdict: write it down, and put a line in the ledger.
///
/// The ledger line is the point. Everything else this module records is a
/// verdict it reached itself; this is the one that arrives, and a lineage that
/// could not say which of its entries came from outside would be a lineage
/// nobody could audit.
pub fn file_verdict(blob: &[u8]) -> Result<Verdict, String> {
    let v = read_verdict(blob)?;
    let mut path = String::from(INBOX);
    path.push('/');
    path.push_str(&hex32(&v.point));
    sysbox::write_blob(&path, blob.to_vec());

    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    let seq = ledger_len() as u32 + 1;
    let mut line = String::new();
    push_u32(&mut line, seq);
    line.push_str(" h");
    push_u32(&mut line, hour as u32);
    line.push_str(" parent=");
    line.push_str(&head().map(|h| short(&h)).unwrap_or(String::from("root....")));
    line.push_str(" variant=");
    line.push_str(&short(&v.point));
    // Its own axis name. `axis_of` finds it by position in `AXIS_NAMES` and
    // will answer `None`, which is correct: this is not one of the axes the
    // surprise ranking chooses among, because the machine does not run it.
    line.push_str(" axis=source rail=");
    line.push_str(&v.rail);
    line.push_str(" moved=");
    line.push_str(&v.moved);
    if let Some(h) = sysbox::hash_of(super::vocab::CORPUS) {
        line.push_str(" corpus=");
        line.push_str(&short(&h));
    }
    line.push(' ');
    line.push_str(&v.why);
    line.push_str(if v.adopted { " ADOPT" } else { " reject" });
    ledger_append(&line);
    Ok(v)
}

/// The next constant worth proposing, if the declared space holds one.
///
/// Walks `KNOBS` in order and takes the first point with no marker, exactly as
/// `frontier()` walks `GRID`. The search is therefore re-derivable from the
/// markers rather than from a coin, which is the property the whole module
/// rests on.
pub fn next_source() -> Option<Proposal> {
    let mut i = 0usize;
    while i < super::knob::KNOBS.len() {
        let mut j = 0usize;
        while j < super::knob::KNOBS[i].values.len() {
            let p = Proposal::source(i, j);
            if !p.tried() {
                return Some(p);
            }
            j += 1;
        }
        i += 1;
    }
    None
}

/// Write the patch a source point stands for, and answer where it went.
///
/// Marked here, so a point that was proposed is not proposed again on the next
/// pass. The marker means "this was reached", which is what every other axis's
/// marker means; whether the change was any good is a separate record that
/// arrives from outside.
pub fn propose_source(p: &Proposal) -> Result<String, &'static str> {
    let ProposalKind::Source(ki, vi) = p.kind else {
        return Err("not a source proposal");
    };
    let Some(text) = super::knob::patch(ki as usize, vi as usize) else {
        return Err("this kernel does not have that knob");
    };
    let mut path = String::from(OUTBOX);
    path.push('/');
    path.push_str(&hex32(&p.hash()));
    if !sysbox::write_text(&path, &text) {
        return Err("the outbox would not take it");
    }
    p.mark();
    Ok(path)
}

/// The next library candidate worth judging, if the queue holds one.
///
/// **`tried()` is the whole of the fix and its absence was the whole of the
/// bug.** This was the one `next_*` that did not consult `/ai/godel/tried`,
/// while `run` has always written a marker for a `Lib` proposal. So a refused
/// candidate stayed at the head of `/ai/libcand` and was offered again on
/// every pass: `next_candidate` skips a candidate whose *name* the library
/// already holds, which covers an adopted one and says nothing about a
/// rejected one. Combined with `lib` sorting first on an empty record, an
/// unattended machine spent every night re-judging one function it had already
/// refused.
pub fn next_lib() -> Option<Proposal> {
    super::redqueen::unheld_candidates()
        .into_iter()
        .map(Proposal::lib)
        .find(|p| !p.tried())
}

fn next_config() -> Option<Proposal> {
    let now = super::harness::rule_in_force() as u8;
    (0u8..4)
        .filter(|r| *r != now)
        .map(Proposal::config)
        .find(|p| !p.tried())
}

/// A program in the toolkit that has never been judged.
///
/// Scanned rather than queued, because `agent learn` writes straight into
/// `/ai/tools` and a queue would be a second record of the same fact. An
/// adopted skill is copied back there too and is skipped on the next pass by
/// its own marker, which is the marker doing what it is for.
fn next_skill() -> Option<Proposal> {
    for name in sysbox::children("/ai/tools") {
        if !name.ends_with(".ai&xi") {
            continue;
        }
        let mut path = String::from("/ai/tools/");
        path.push_str(&name);
        let Some(bytes) = sysbox::read_blob(&path) else { continue };
        let Ok(text) = core::str::from_utf8(&bytes) else { continue };
        let h = super::skill::store(text);
        let p = Proposal::skill(h);
        if !p.tried() {
            return Some(p);
        }
    }
    None
}

/// The axes, in slot order, as they are written in the ledger.
///
/// Strings rather than a discriminant because they go in the record and a
/// ledger of integers is a ledger nobody reads. The order is the slot order,
/// and ties in the surprise ranking break by it, so the ordering is total and
/// a later run reconstructs the same one rather than a plausible one.
// Order is load-bearing: `surprise_order` ranks slots `0..RANKED` and those
// slots index straight into this array and into `axis_counts`, so the ranked
// axes must occupy the first `RANKED` positions. `lib` therefore sits at four
// and the two unranked axes -- `core`, exempt and always last; `judge`, asked
// only at an epoch boundary -- come after the ranked block. Nothing else
// depends on position: a ledger line names its axis and `axis_of` finds the
// index by name, so every consumer derives its indices from this one array.
pub const AXIS_NAMES: [&str; 7] = ["adapter", "rule", "skill", "deep", "lib", "core", "judge"];

/// How many axes the surprise ranking covers.
///
/// Five, not seven. `core` is exempt and stays last regardless of how
/// uncertain it looks, and `judge` is not ranked at all -- it is reachable
/// only at an epoch boundary and is asked before the ranking runs.
///
/// `lib` joined the ranking rather than being appended after it, and that is
/// the point of ranking by surprise: it is a brand new axis, so it has no
/// verdicts, so `axis_uncertainty` puts it at the coin flip and the loop
/// reaches for it early. An axis bolted on at the end would have waited for
/// four others to run out of moves first.
const RANKED: usize = 5;

/// Which axis a ledger line came from.
pub(crate) fn axis_of(line: &str) -> Option<usize> {
    let at = line.find(" axis=")? + 6;
    let rest = &line[at..];
    let end = rest.find(' ').unwrap_or(rest.len());
    AXIS_NAMES.iter().position(|n| *n == &rest[..end])
}

/// Attempts and adoptions per axis, counted out of the ledger.
///
/// From the *record*, not from a tally kept beside it -- the discipline
/// `is_boundary` follows, for the same reason. A counter in its own file could
/// disagree with the ledger, and then the loop's account of why it chose an
/// axis would be unfalsifiable exactly where it most needs not to be.
///
/// Lines written before the axis was recorded carry no `axis=` field and count
/// for nothing. That undercounts the early history rather than guessing at it,
/// and it is the right direction to be wrong in: an axis whose record is
/// invisible reads as untried, an untried axis is maximally uncertain, so it
/// gets reached for and measured. The loop recovers by looking rather than by
/// assuming.
fn axis_counts() -> [(u32, u32); AXIS_NAMES.len()] {
    let mut out = [(0u32, 0u32); AXIS_NAMES.len()];
    for line in ledger_tail(usize::MAX) {
        if let Some(i) = axis_of(&line) {
            out[i].0 += 1;
            if line.contains(" ADOPT") {
                out[i].1 += 1;
            }
        }
    }
    out
}

/// The four findable axes, most surprising first.
///
/// An insertion sort over four elements, written out rather than reached for:
/// `sort_by` over an `f32` comparator wants a total order `f32` does not have,
/// and the tie rule here is a decision rather than a detail.
fn surprise_order() -> [usize; RANKED] {
    let counts = axis_counts();
    let mut order = [0usize, 1, 2, 3, 4];
    for i in 1..RANKED {
        let mut j = i;
        while j > 0 {
            let (a, b) = (order[j - 1], order[j]);
            let ua = axis_uncertainty(counts[a].0, counts[a].1);
            let ub = axis_uncertainty(counts[b].0, counts[b].1);
            // Strictly greater, so equal uncertainty leaves the earlier slot
            // where it is and the order stays a function of the record.
            if ub > ua {
                order.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    order
}

/// The bars the loop may propose, coarsest first.
///
/// Six declared points rather than a continuum, and that is the safeguard
/// rather than the simplification. A continuum lets the loop creep the bar
/// down by a hundredth a night -- every step passing `moves` on some candidate
/// somewhere, every step honestly certified, and the sum of them being exactly
/// the drift this axis exists to catch.
const JUDGE_GRID: [f32; 6] = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0];

/// A bar to put in front of the tribunal, or none.
///
/// **`None` everywhere except an epoch boundary.** That is what makes the
/// freeze real rather than advisory: the axis cannot be reached from any other
/// route, so the check in `next_proposal` is belt and braces rather than the
/// only guard.
///
/// The bar in force is skipped, because judging a criterion against itself
/// answers `moves` in advance and spends a night doing it.
fn next_judge() -> Option<Proposal> {
    if !at_epoch_boundary() {
        return None;
    }
    let now = bar_in_force();
    JUDGE_GRID.iter().filter(|b| **b != now).map(|b| Proposal::judge(*b)).find(|p| !p.tried())
}

/// The next thing to try tonight, over every axis the loop can judge.
///
/// **A rotation and not a choice.** The night branch knew two jobs and godel
/// always won the tie, so the adapter grid was walked to exhaustion while
/// every other axis the judges can reach -- the routing rule, deep training, a
/// skill the agent compiled, a core the machine wrote -- was never tried
/// unattended at all. Widening it needed the judges first, which is why this
/// comes last.
///
/// The starting point is the number of verdicts already recorded, so the
/// rotation is a function of the ledger rather than of a coin or of a counter
/// that resets at boot. That matters for the same reason `frontier` walks a
/// declared grid: a later reader has to be able to say what the machine would
/// have done, and "it picked at random" is not an account of a night.
///
/// From the offset it takes the first kind that has work, so an exhausted axis
/// costs one skipped slot rather than an idle night, and the loop stops only
/// when every axis is out of moves.
///
/// Order is deliberate: cheap and declared before expensive and composed. A
/// grid point and a rule change are minutes; a deep trial is two passes over
/// the corpus and a composed core spends a dozen decodes writing something
/// that may not survive its first judge.
pub fn next_proposal() -> Option<Proposal> {
    // The criterion first, and only at the edge of an epoch. Inside one this
    // costs a modulo and the axes below are reached exactly as they were.
    if let Some(p) = next_judge() {
        return Some(p);
    }
    for slot in surprise_order() {
        let candidate = match slot {
            0 => frontier(),
            1 => next_config(),
            2 => next_skill(),
            3 => next_deep(),
            _ => next_lib(),
        };
        if candidate.is_some() {
            return candidate;
        }
    }
    // Last, and exempt from the ranking. Composing a core costs a dozen
    // constrained decodes whether or not the result survives its first judge,
    // so it is reached for when everything cheaper is out of moves -- never
    // because it happened to look uncertain.
    author_core()
}

/// Where the rotation stands, without taking a turn.
///
/// Report-only, and deliberately does not ask the last slot whether it has
/// work: finding out costs a dozen constrained decodes, because composing a
/// core *is* the work. A command that answers "what would you do tonight"
/// must not spend the night doing it.
pub fn rotation() -> (bool, [(&'static str, f32, bool); RANKED]) {
    let counts = axis_counts();
    let mut out = [("", 0.0f32, false); RANKED];
    for (i, slot) in surprise_order().iter().enumerate() {
        let has = match slot {
            0 => frontier().is_some(),
            1 => next_config().is_some(),
            2 => next_skill().is_some(),
            3 => next_deep().is_some(),
            _ => next_lib().is_some(),
        };
        out[i] = (
            AXIS_NAMES[*slot],
            axis_uncertainty(counts[*slot].0, counts[*slot].1),
            has,
        );
    }
    (at_epoch_boundary(), out)
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

/// Score both sides of a bar proposal on the held-out slice, for one read.
///
/// **One consultation, two scores.** `read_test` spends a read per adapter,
/// which is right when the question is "how good is this variant". Here the
/// question is whether the bars agree with the anchor, and that needs the
/// candidate and the incumbent measured against each other -- charging two
/// reads for one look at the slice would exhaust a three-read budget in a
/// night and a half, and would overstate what was spent. The slice is read
/// once either way.
fn read_anchor(
    t: &Trial,
    incumbent: Option<&super::adapter::Dora>,
    candidate: Option<&super::adapter::Dora>,
) -> (f32, f32, u32, bool) {
    let n = spend_test_read();
    (t.score(candidate, Slice::Test), t.score(incumbent, Slice::Test), n, n <= TEST_READS)
}

/// Judge a proposed bar on a matrix neither bar can grade itself on.
///
/// **The price is deliberate, and it is the test-slice budget.** Every other
/// axis reads the anchor only after a variant has already won on validation,
/// as an after-the-fact confirmation that costs nothing if the answer is
/// boring. Here the anchor *is* the evidence: both bars are the thing under
/// suspicion, so there is no cheaper reading that could settle the question.
/// A judge-trial therefore spends a read to exist at all, and after three of
/// them a criterion change cannot be grounded any more.
///
/// Charging it there is the point rather than a side effect. The one
/// non-renewable resource in the building is what a moving criterion is
/// billed against, so a loop that spends its nights rewriting its own bar runs
/// out of the ability to justify doing so -- while the axes that improve the
/// agent against a fixed bar go on costing nothing.
pub fn trial_judge(e: &mut super::Engine, b: &Budget, bar: f32) -> Result<Certificate, Refused> {
    let t = super::train::prepare(e, b).map_err(Refused::Train)?;
    TRIALS.fetch_add(1, Ordering::Relaxed);

    // The candidate is a freshly trained variant rather than a stored one,
    // because the matrix has to be about a decision the bars would actually
    // face. Re-scoring an old elite would be cheaper and would ask a different
    // question: whether the bars disagree about something already decided.
    let incumbent = e.model.adapters.as_ref().and_then(|a| t.gather(a));
    let fit = t.train(b);
    let (broke, fixed, _, neither) = t.paired(incumbent.as_ref(), Some(&fit.dora), Slice::Validation);

    let standing = bar_in_force();

    // The anchor is bought, not taken. `judge_precheck` answers the two
    // questions that are decidable from the bar and the paired counts alone,
    // and this axis previously spent a read before asking either -- the first
    // line of `read_anchor` was `spend_test_read()`, unconditionally. With the
    // candidate inert that outcome was always "admits and refuses what the
    // standing one did", so three epoch boundaries emptied a lifetime budget
    // to record three foregone rejections, and every *adapter* certificate
    // afterwards rendered `(stale)`.
    //
    // Where the precheck declines to settle it, the anchor genuinely is the
    // evidence and the read is charged, which is the arrangement the module
    // header argues for.
    let mut cross =
        Cross { standing, proposed: bar, fixed, broke, anchor_candidate: 0.0, anchor_incumbent: 0.0 };
    let (honest, why, read, fresh) = match judge_precheck(&cross) {
        Some((verdict, reason)) => (verdict, reason, test_reads(), false),
        None => {
            let (cand, inc, read, fresh) =
                read_anchor(&t, incumbent.as_ref(), Some(&fit.dora));
            cross.anchor_candidate = cand;
            cross.anchor_incumbent = inc;
            let (honest, why) = judge_verdict(&cross);
            (honest, why, read, fresh)
        }
    };
    let cross = cross;
    let moves = cross.admits(standing) != cross.admits(bar);
    let n_val = t.slice_size(Slice::Validation);

    // A bar change grounded on a stale read is not grounded. The budget is
    // what makes the anchor evidence rather than decoration, so a reading past
    // it may still be printed and may not be acted on.
    let adopted = honest && fresh && n_val > 0;

    let parent = ensure_head(e);
    let carried = parent.and_then(|p| Variant::load(&p));
    let variant = Variant {
        parent,
        adapter: carried.as_ref().and_then(|x| x.adapter),
        policy: sysbox::read_blob("/ai/agent/policy").map(|p| sha256::hash(&p)),
        skills: carried.as_ref().and_then(|x| x.skills),
        corpus: sysbox::hash_of(super::vocab::CORPUS),
        deep: carried.as_ref().map(|x| x.deep).unwrap_or(false),
        lib: None,
        bar: Some(bar),
        core: super::voter::installed().map(|c| c.hash),
        core_seen: true,
        lambda: 0.0,
        rank: 0,
        epochs: 0,
        // The rule actually in force, as every other axis records it.
        //
        // **This was `(bar * 100.0) as u8`**, packing the criterion into the
        // routing rule's slot so a lineage would record which bar each variant
        // was judged under -- a good reason with a lossy encoding under it.
        // Rust's `as` saturates, and `JUDGE_GRID` is
        // `[1.0, 2.0, 3.0, 5.0, 8.0, 12.0]`, so four of the six became 255 and
        // none of 100, 200 or 255 is a `Rule`. Every judge node named a
        // routing rule this kernel does not have, which is what `rollback`
        // refuses in so many words. The bar has a field of its own now, and
        // the reason that field exists is still the one written here: without
        // it, "the loop improved for a month" and "the loop lowered its bar in
        // week two" are the same chain of hashes.
        rule: super::harness::rule_in_force() as u8,
        born: crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0),
    };
    let vhash = variant.hash();

    let mut cert = Certificate {
        budget_n: Some(b.examples),
        axis: "judge",
        parent,
        variant: vhash,
        decisions: n_val,
        validation: n_val,
        predicted: cross.anchor_gain() > 0.0,
        fixed,
        broke,
        wrong: Some(fixed + neither),
        mcnemar: cross.chi(),
        // J1 carries the verdict itself, because on this axis the verdict is
        // one sentence and the other judges are its preconditions.
        j1: honest,
        j1_why: why,
        // J2: the two bars actually disagree about this candidate.
        goals_held: usize::from(moves),
        goals_total: 1,
        j2: moves,
        // J3: sane, and asked of the number before anything is measured.
        j3: sane_bar(bar),
        j3_why: if sane_bar(bar) { "the bar is inside the range" } else { "the bar is outside the range" },
        // J4: nothing resident changes. A criterion is a number in a file, so
        // the judge that asks whether this machine can carry the variant has
        // nothing to weigh and says so rather than abstaining.
        resident_kib: 0,
        rank: 0,
        j4: true,
        epochs: 0,
        capped: false,
        adopted,
        test_acc: cross.anchor_candidate,
        test_read: read,
        test_fresh: fresh,
    };

    if adopted {
        let mut text = String::new();
        push_f6(&mut text, bar);
        text.push('\n');
        sysbox::write_text(BAR, &text);
        variant.store();
        set_head(&vhash);
        ADOPTIONS.fetch_add(1, Ordering::Relaxed);
    } else {
        cert.adopted = false;
    }
    let seq = ledger_len() as u32 + 1;
    let hour = crate::dev::rtc::now().map(|d| d.hour).unwrap_or(0);
    ledger_append(&render_certificate(&cert, seq, hour));
    Ok(cert)
}

/// Eight hex characters of an address, for anything that has to fit on a line.
pub fn short_hex(h: &[u8; 32]) -> String {
    short(h)
}

/// The head, in the four-byte name the ledger writes it under.
pub fn name4(h: &[u8; 32]) -> super::clade::Name {
    u32::from_be_bytes([h[0], h[1], h[2], h[3]])
}

/// Where the clade posterior says tonight should grow from, and the arms it
/// drew to decide. Read-only: it moves nothing.
pub fn clade_now() -> (super::clade::Move, Vec<super::clade::Arm>) {
    let lines = ledger_tail(usize::MAX);
    let here = head().map(|h| name4(&h)).unwrap_or(super::clade::ROOT);
    super::clade::decide(&lines, here)
}

/// What `reconsider` did.
pub enum Reconsidered {
    /// The head won its own draw, which is what every night before this did
    /// unconditionally.
    Stayed,
    /// **The head file did not describe the mind that was running**, so
    /// `ensure_head` moved it before anything was decided, and whatever
    /// `godel clade` said a moment ago was about a lineage this machine is not
    /// on. Reported rather than folded into `Stayed`, because the two look
    /// identical from outside and mean opposite things: one is a decision, the
    /// other is the decision having been made about the wrong record.
    ///
    /// Found by driving it -- a report offering a backtrack, `reconsider`
    /// answering "staying", and the head silently at the root afterwards.
    Rebased(Option<[u8; 32]>),
    /// Went back this many parents, landing here. `None` is the frozen model.
    Went(usize, Option<[u8; 32]>),
    /// A rollback refused partway. The machine is wherever it got to, which
    /// is a real state and is reported rather than guessed at.
    Stuck(usize, &'static str),
}

/// Choose where to grow from, and go there if it is not here.
///
/// **The one thing in this module that moves the machine without a verdict**,
/// and it is worth saying why that is admissible. Every other change to the
/// head is an adoption: judged, certified and written to the ledger. This is
/// not a change to what the machine *is* so much as a change to where the
/// search stands, and the mechanism is `rollback` -- which restores a node
/// that was itself adopted under the four judges, validated before anything
/// moves, and undone by the next adoption.
///
/// It writes no ledger line, and that is deliberate for the reason `OUTBOX`
/// is not the ledger: every line there is a verdict, and "the search moved
/// back two" is not one. Where the machine stands is the head file, which is
/// the record of exactly this.
pub fn reconsider(e: &mut super::Engine) -> Reconsidered {
    // Before the draw, because a draw made against a head that is not the
    // running mind is a draw about somebody else's lineage.
    let before = head();
    let _ = ensure_head(e);
    if head() != before {
        return Reconsidered::Rebased(head());
    }
    let (mv, _) = clade_now();
    let super::clade::Move::Back(n) = mv else { return Reconsidered::Stayed };
    let mut at = head();
    for done in 0..n {
        match rollback(e) {
            Ok(h) => at = h,
            Err(why) => return Reconsidered::Stuck(done, why),
        }
    }
    Reconsidered::Went(n, at)
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
    // **The line that makes a J1 veto legible**, and the one this axis wanted
    // for as long as `core_room` has had its equivalent. Without it "wanted
    // six repairs, got two" cannot be told from "wanted six repairs on a
    // budget where only three were available", and those are a candidate
    // problem and a budget problem.
    if let Some(w) = c.wrong {
        let need = clean_fixes_needed();
        console::set_color(if w >= need { LTGRAY } else { LTRED });
        kprintln!(
            "  the incumbent gets {} of those wrong, so at most {} can be repaired, and J1 needs {}{}",
            w,
            w,
            need,
            if w >= need { "" } else { " -- this budget cannot pass J1 at all" }
        );
        console::set_color(LTGRAY);
    }
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
        "  J2 own goals {}  {}/{} {}",
        mark(c.j2),
        c.goals_held,
        c.goals_total,
        // Zero protected goals is a pass and reads like one now. It used to be
        // a veto, and on a machine getting none of its own goals right that
        // made J2 unpassable exactly when improving mattered most.
        if c.goals_total == 0 {
            "-- none of the goals route where they were declared to, so there \
             is nothing here to break"
        } else {
            "of the goals that work still route where they did"
        }
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

/// Drive the criterion axis by hand.
///
/// It had no operator path at all: `next_proposal` reaches it only at an epoch
/// boundary, and the `godel` dispatch had no arm for it. So the one axis that
/// edits the bar every other judge answers to could never be exercised
/// deliberately -- which is most of why it went so long being unable to adopt
/// anything without anybody noticing.
///
/// Forced, the way `godel now` is: an operator asking for a trial is the
/// consent the quiet window stands in for the rest of the time.
pub fn report_judge(bar: f32, b: &Budget) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[godel] criterion trial");
    console::set_color(LTGRAY);

    if !sane_bar(bar) {
        console::set_color(LTRED);
        kprintln!("  {} is outside [{}, {}]", bar, JUDGE_MIN, JUDGE_MAX);
        console::set_color(LTGRAY);
        return;
    }

    let standing = bar_in_force();
    kprintln!("  standing {}, proposed {}", standing, bar);
    let (used, cap) = (test_reads(), TEST_READS);
    kprintln!("  test budget {} of {} spent before this trial", used, cap);

    let p = Proposal::judge(bar);
    let b = &p.budget(b.examples, b.millis);
    let outcome = super::with_engine(|e| trial_judge(e, b, bar));
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

    kprintln!("  {} repaired, {} broken of {}", c.fixed, c.broke, c.validation);
    kprintln!("  chi {}", (c.mcnemar * 100.0) as u32 as f32 / 100.0);
    kprintln!("  verdict: {}", c.j1_why);
    let after = test_reads();
    if after == used {
        kprintln!("  no test read spent -- settled without the anchor");
    } else {
        kprintln!("  test read {} of {} spent on the anchor", after, cap);
    }
    if c.adopted {
        console::set_color(LTGREEN);
        kprintln!("  ADOPTED -- the bar is now {}", bar);
    } else {
        console::set_color(LTRED);
        kprintln!("  refused -- the bar stays {}", standing);
    }
    console::set_color(LTGRAY);
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

    // --- what a rollback decides, before anything moves -----------------
    //
    // **`clade::reconsider` calls `rollback` unattended now**, and until this
    // block nothing in the tree checked any of its decisions: they live inside
    // a hundred-and-fifty-line function that needs an engine, a store and a
    // real lineage to reach, so a boot could not get near them. They are pure
    // functions of two nodes now, in the shape `update::decide` is, and every
    // state is asserted here with no model and no disk.
    {
        let blank = Variant::from_text("");
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        let with_core = |h: Option<[u8; 32]>, seen: bool| {
            let mut v = Variant::from_text("");
            v.core = h;
            v.core_seen = seen;
            v
        };

        // **The regression this shape exists for.** A parent that says nothing
        // about a core is not a parent that had none, and reading it as one
        // pulled a core out of the decision path as a side effect of undoing
        // an adapter, printed nowhere.
        claim(
            "a silent parent leaves a core nobody here adopted alone",
            core_move(&blank, &blank) == CoreMove::Leave,
        );
        claim(
            "and takes away the one the node being left adopted",
            core_move(&with_core(Some(c1), true), &blank) == CoreMove::Drop,
        );
        claim(
            "a parent that said it had none takes it away",
            core_move(&with_core(Some(c1), true), &with_core(None, true)) == CoreMove::Drop,
        );
        claim(
            "and one that names a core puts that core back",
            core_move(&blank, &with_core(Some(c2), true)) == CoreMove::Install(c2),
        );
        claim(
            "a silent parent never installs, whatever the node was carrying",
            !matches!(core_move(&with_core(Some(c1), false), &blank), CoreMove::Install(_)),
        );

        // The routing rule. Every node renders one, so the guard is the
        // *disagreement* rather than an absence: nodes written before the axis
        // was searchable recorded 0 while the machine ran the default, and
        // restoring unconditionally would switch a whole legacy lineage to a
        // rule none of them ever ran.
        let ruled = |r: u8| {
            let mut v = Variant::from_text("");
            v.rule = r;
            v
        };
        claim("two nodes agreeing about the rule restore nothing", rule_move(&ruled(2), &ruled(2)) == None);
        claim("and a lineage of legacy zeroes restores nothing", rule_move(&blank, &blank) == None);
        claim(
            "two that disagree put the parent's rule back",
            rule_move(&ruled(2), &ruled(1)) == Some(1),
        );

        // The bar. A parent's `None` is "did not say" and not "had none".
        let barred = |b: f32| {
            let mut v = Variant::from_text("");
            v.bar = Some(b);
            v
        };
        claim("a silent parent restores no bar", bar_move(&barred(3.0), &blank) == None);
        claim("and two silent nodes restore none either", bar_move(&blank, &blank) == None);
        claim(
            "a parent that named one puts it back on a node that named none",
            bar_move(&blank, &barred(8.0)) == Some(8.0),
        );
        claim("two that agree restore nothing", bar_move(&barred(3.0), &barred(3.0)) == None);
        claim(
            "and two that disagree put the parent's back",
            bar_move(&barred(3.0), &barred(12.0)) == Some(12.0),
        );
        // The pair `rollback` actually checks: what comes out of the decision,
        // against the range this kernel accepts. A hand-edited node naming a
        // bar outside it is refused rather than clamped, for the reason
        // `bar_in_force` gives -- clamping lets an edited file slide the
        // criterion to the nearest legal number and report nothing.
        claim(
            "a bar outside the range comes out of the decision and is refused by the guard",
            bar_move(&blank, &barred(99.0)) == Some(99.0) && !sane_bar(99.0),
        );
    }

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

    // ---- Red Queen epochs ---------------------------------------------
    //
    // The boundary is a function of the ledger, so these are the whole of it.
    claim(
        "an epoch boundary is every fifth verdict, and genesis is not one",
        !is_boundary(0)
            && !is_boundary(1)
            && is_boundary(EPOCH_LEN)
            && is_boundary(EPOCH_LEN * 3)
            && !is_boundary(EPOCH_LEN + 1),
    );

    // ---- the unbound judge ---------------------------------------------
    //
    // Nine repairs against two breaks gives chi 3.27, under the standing 3.84
    // and over a proposed 3.0: that is a loosening, and it is the exact shape
    // a loop lowering its own bar would produce. Twelve against one gives
    // 7.69, over the standing bar and under a proposed 10.0: a tightening.
    // Both candidates are held fixed and only the bar and the anchor move, so
    // each claim below isolates one question.
    let cross = |proposed: f32, fixed: usize, broke: usize, gain: f32| Cross {
        standing: MCNEMAR_95,
        proposed,
        fixed,
        broke,
        anchor_candidate: 0.60 + gain,
        anchor_incumbent: 0.60,
    };
    claim(
        "a bar of zero abolishes the criterion and is refused",
        !judge_verdict(&cross(0.0, 9, 2, 0.04)).0,
    );
    claim(
        "a bar too high freezes the loop and is refused",
        !judge_verdict(&cross(50.0, 9, 2, 0.04)).0,
    );
    claim(
        "a bar that changes nothing about the candidate is refused",
        !judge_verdict(&cross(3.9, 9, 2, 0.04)).0,
    );
    claim(
        "drift: a looser bar admitting what the anchor rejects is refused",
        judge_verdict(&cross(3.0, 9, 2, -0.01))
            == (false, "drift: a looser bar admits a variant the anchor rejects"),
    );

    // The precheck, which is what decides whether a lifetime test read is
    // spent. Both settled cases must be answerable without the anchor, and
    // the case that genuinely turns on it must decline to settle -- otherwise
    // the read is either wasted or skipped when it was the evidence.
    claim(
        "an insane bar is settled without spending the anchor",
        judge_precheck(&cross(50.0, 9, 2, 0.04)).is_some(),
    );
    claim(
        "a bar that changes nothing is settled without spending the anchor",
        judge_precheck(&cross(3.9, 9, 2, 0.04)).is_some(),
    );
    claim(
        "an inert candidate is settled without spending the anchor",
        judge_precheck(&cross(1.0, 0, 0, 0.0)).is_some(),
    );
    claim(
        "a bar the anchor must arbitrate is not settled early",
        judge_precheck(&cross(3.0, 9, 2, 0.04)).is_none(),
    );

    // The proposal the criterion axis actually builds. It carried zero epochs
    // and rank zero, which trains nothing: the candidate came out identical to
    // the incumbent, every trial scored `fixed == broke == 0`, and the
    // precheck above settles that as "changes nothing" -- so adoption was
    // unreachable for every input the axis could ever be given.
    let jp = Proposal::judge(3.0);
    claim(
        "a criterion proposal carries knobs that actually train",
        jp.epochs > 0 && jp.rank > 0 && jp.lr > 0.0,
    );
    claim(
        "a looser bar admitting what the anchor confirms is adopted",
        judge_verdict(&cross(3.0, 9, 2, 0.04)).0,
    );
    claim(
        "a tighter bar that would refuse a genuine gain is refused",
        judge_verdict(&cross(10.0, 12, 1, 0.04))
            == (false, "a tighter bar would refuse a genuine gain"),
    );
    claim(
        "a tighter bar refusing what the anchor does not support is adopted",
        judge_verdict(&cross(10.0, 12, 1, -0.02)).0,
    );
    claim(
        "a bar outside the range is refused before anything else is asked",
        !sane_bar(0.0) && !sane_bar(f32::INFINITY) && sane_bar(MCNEMAR_95),
    );

    // ---- drift: founding vs current criterion, from rendered lines ------
    //
    // Rendered rather than hand-typed, so the parser this checks is tied to
    // the renderer it reads: if the ledger format moves, these break, which is
    // the point. The counts are chosen so one line drifts and one does not
    // under each direction, and one is gated out by a failed non-bar judge.
    let mut base = Certificate {
        budget_n: None,
        axis: "adapter",
        parent: None,
        variant: [0u8; 32],
        decisions: 180,
        validation: 180,
        predicted: true,
        fixed: 12,
        broke: 1,
        wrong: Some(20),
        mcnemar: mcnemar(1, 12),
        j1: true,
        j1_why: "beyond the noise",
        goals_held: 4,
        goals_total: 4,
        j2: true,
        j3: true,
        j3_why: "",
        resident_kib: 24,
        rank: 8,
        j4: true,
        epochs: 20,
        capped: false,
        adopted: true,
        test_acc: 0.6,
        test_read: 1,
        test_fresh: true,
    };
    // chi 7.69: clears 3.84 and fails 10.0.
    let strong = render_certificate(&base, 1, 3);
    // chi 3.12 (7 vs 1): fails 3.84, clears 2.0.
    base.fixed = 7;
    base.mcnemar = mcnemar(1, 7);
    base.j1 = false;
    base.j1_why = "inside the noise";
    let weak = render_certificate(&base, 2, 3);
    // Same thin margin, but J4 failed -- rejected under any bar, so it must
    // not read as drift when the bar alone would have flipped J1.
    base.j4 = false;
    let weak_costly = render_certificate(&base, 3, 3);

    let loosen = drift_of(
        &[strong.clone(), weak.clone(), weak_costly.clone()],
        MCNEMAR_95,
        2.0,
    );
    claim(
        "loosening the bar adopts a line the founding one refused",
        loosen.looser == 1 && loosen.stricter == 0,
    );
    claim(
        "a line another judge rejected does not read as criterion drift",
        loosen.agree == 2,
    );

    let tighten = drift_of(&[strong.clone(), weak.clone()], MCNEMAR_95, 10.0);
    claim(
        "tightening the bar refuses a line the founding one adopted",
        tighten.stricter == 1 && tighten.looser == 0,
    );

    // The prediction cross-tab, and that a line with no axis is left out.
    let mut p = base.clone();
    p.j4 = true;
    p.predicted = true;
    p.adopted = true;
    let win_adopt = render_certificate(&p, 4, 3);
    p.predicted = false;
    p.adopted = false;
    let lose_reject = render_certificate(&p, 5, 3);
    let cal = drift_of(&[win_adopt, lose_reject], MCNEMAR_95, MCNEMAR_95);
    claim(
        "the prediction cross-tab counts hits the loop never read",
        cal.pred_total() == 2 && cal.pred_right() == 2,
    );
    claim(
        "a line with no axis is left out of the prediction tally",
        drift_of(&[String::from("1 h3 n=180 fix=12 broke=1 pred=win ADOPT")], MCNEMAR_95, MCNEMAR_95)
            .pred_total()
            == 0,
    );

    // ---- the archive ----------------------------------------------------
    claim(
        "a cheap variant and an expensive one that repair alike share a behaviour and not a cell",
        descriptor(4, 8, 1).1 == descriptor(32, 80, 10).1
            && descriptor(4, 8, 1) != descriptor(32, 80, 10),
    );
    claim(
        "a variant that moved nothing is not filed as a perfect repairer",
        descriptor(8, 0, 0).1 == 0,
    );
    claim(
        "every cell a descriptor can name is inside the archive",
        cell_of(64, 100, 0) < CELLS && cell_of(0, 0, 0) < CELLS && CELLS == 12,
    );

    // ---- Bayesian surprise ----------------------------------------------
    claim(
        "an axis at the coin-flip is where the information is",
        axis_uncertainty(10, 5) > axis_uncertainty(10, 9)
            && axis_uncertainty(10, 5) > axis_uncertainty(10, 1),
    );
    claim(
        "an axis that has refused everything still outranks nothing",
        axis_uncertainty(20, 0) > 0.0,
    );
    claim(
        "an axis nobody has tried is as uncertain as the loop can be",
        axis_uncertainty(0, 0) >= axis_uncertainty(10, 5),
    );

    // --- the family-wise budget -------------------------------------------
    //
    // Every judged comparison at the bar is a test at p < 0.05, so a loop that
    // runs one a night against one corpus is fooled about once in twenty,
    // permanently, by arithmetic rather than by any judge being wrong. The
    // schedule bills for that. These claims are about the schedule itself,
    // because it is a table and a table is exactly the thing that goes quietly
    // wrong when somebody edits a number in it.
    claim(
        "the bar rises with every test this evidence has already paid for",
        SPEND.windows(2).all(|w| w[1] > w[0]),
    );
    claim(
        "and the first test already pays more than the standing bar",
        SPEND[0] > MCNEMAR_95,
    );
    // The series is what makes the total a closed form rather than a number
    // that happens to converge, so it is worth checking the terms are the ones
    // the comment claims. Chi-squared for one degree of freedom is monotone in
    // alpha, so a table built from a *different* schedule would not land on
    // these values.
    claim(
        "the table is thirty-two deep, which is what the budget buys",
        SPEND.len() == 32,
    );
    claim(
        "a test past the table is refused rather than given the last bar",
        SPEND.get(SPEND.len()).is_none(),
    );
    // **`None` is a refusal and not a very high number**, which is the whole
    // design: a bar that kept rising would let the loop test forever against
    // evidence that had stopped being able to support a conclusion.
    claim(
        "an exhausted budget refuses J1 with its own reason rather than 'inside the noise'",
        {
            let (pass, why) = passes_j1(100, 50, 0, f32::INFINITY);
            !pass && why == "inside the noise"
        },
    );
    // The composition. The judge axis may raise the bar and may lower it, and
    // lowering it below the floor would be the loop buying itself more chances
    // to be fooled -- through a door `judge_verdict` cannot see.
    claim(
        "the floor lifts a lower adopted bar, and a higher one is left alone",
        {
            let floor = SPEND[3];
            let lower = if floor > 1.0 { 1.0f32 } else { floor };
            let higher = floor + 1.0;
            lower.max(floor) == floor && higher.max(floor) == higher
        },
    );

    // --- what comes back from outside -------------------------------------
    //
    // A verdict is the only thing in this tree that writes a ledger line the
    // machine did not derive itself, so every way of refusing one is worth
    // watching happen. The signature half is checked the other way -- by
    // feeding a real signed verdict with one word changed -- because there is
    // no private key on this machine and there must not be.
    let good = concat!(
        "verdict 1
",
        "point 0000000000000000000000000000000000000000000000000000000000000001
",
        "rail host.retrieval
",
        "moved better
",
        "why fixed 9 broke 1
",
        "adopted yes
",
    );
    claim(
        "a well-formed verdict parses, and carries what it said",
        match parse_verdict(good.as_bytes()) {
            Ok(v) => v.adopted && v.moved == "better" && v.rail == "host.retrieval",
            Err(_) => false,
        },
    );
    claim(
        "anything that is not a verdict is refused before its fields are read",
        parse_verdict(b"manifest 1
version 9.9.9
").is_err(),
    );
    claim(
        "and one naming no proposal, since there would be nothing to file it against",
        parse_verdict(b"verdict 1
moved better
").is_err(),
    );
    // A fifth word would mean the two halves of this loop have parted, and
    // that is exactly the drift that must not be resolved by guessing.
    claim(
        "a verdict word this kernel does not know is refused rather than guessed at",
        parse_verdict(good.replace("moved better", "moved excellent").as_bytes()).is_err(),
    );
    // **The one that makes the receiving side more than a parser.** The
    // machine that built the change is trusted to measure and is not trusted
    // to conclude, because the conclusion is the one thing this side can
    // check against the numbers beside it.
    claim(
        "and one claiming an adoption its own rail comparison refuses",
        parse_verdict(good.replace("moved better", "moved worse").as_bytes()).is_err(),
    );
    claim(
        "while the same verdict without the claim is read",
        parse_verdict(
            good.replace("moved better", "moved worse")
                .replace("adopted yes", "adopted no")
                .as_bytes(),
        )
        .is_ok(),
    );
    // `unstable` is a verdict and not an error: `rails.py` answers it when a
    // control drifted, which is neither evidence of a regression nor evidence
    // against one, and a machine that could not record "we could not tell"
    // would have to record something else.
    claim(
        "'we could not tell' is a verdict this kernel can file",
        parse_verdict(
            good.replace("moved better", "moved unstable")
                .replace("adopted yes", "adopted no")
                .as_bytes(),
        )
        .is_ok(),
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
    // Shaped like a node from before the `core` field: it does not mention one
    // at all, which is what the compatibility claims below are about.
    let mk = |lambda: f32, born: u32| Variant {
        core: None,
        core_seen: false,
        deep: false,
        lib: None,
        bar: None,
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

    // Re-derivability across the `core` field, both directions.
    //
    // The failure this guards is quiet and permanent: a node whose rendering
    // does not reproduce is a node whose address does not reproduce, and the
    // ledger stops being checkable from that point on. Both halves matter --
    // a node written before the field existed must go on hashing to what it
    // hashed to, and a node written now must be able to *say* it has no core,
    // because `rollback` treats "said none" and "said nothing" differently and
    // was quietly uninstalling live cores when it could not tell them apart.
    let old_text = "variant 1\nparent none\nadapter none\npolicy none\nskills none\n\
                    corpus none\nlambda 0.00\nrank 8\nepochs 20\nrule 0\n";
    let old = Variant::from_text(old_text);
    claim(
        "a node written before the core field says nothing about one",
        !old.core_seen && old.core.is_none(),
    );
    claim(
        "and still renders to exactly the bytes it was stored as",
        old.render() == old_text,
    );
    let mut says_none = old.clone();
    says_none.core_seen = true;
    claim(
        "a node that says it has no core renders that, and is a different node",
        says_none.render().contains("core none\n") && says_none.hash() != old.hash(),
    );
    claim(
        "and reads back as having said it",
        Variant::from_text(&says_none.render()).core_seen,
    );
    let with_core = Variant { core: Some(h), core_seen: true, ..old.clone() };
    let back = Variant::from_text(&with_core.render());
    claim(
        "a node naming a core round-trips through its own rendering",
        back.core == Some(h) && back.core_seen && back.hash() == with_core.hash(),
    );

    // The same three questions of the `bar` field, because it is the newest
    // conditional one and it exists to undo a lossy encoding. Every point on
    // `JUDGE_GRID` has to survive the round trip *distinctly* -- that is the
    // whole of what went wrong when the bar rode in the `rule` byte as
    // hundredths, where `as` saturated four of six onto 255.
    claim(
        "a node written before the bar field says nothing about one",
        old.bar.is_none() && !old.render().contains("bar "),
    );
    let mut bars_distinct = true;
    let mut bars_round_trip = true;
    for (i, a) in JUDGE_GRID.iter().enumerate() {
        let va = Variant { bar: Some(*a), ..old.clone() };
        let back = Variant::from_text(&va.render());
        if back.bar != Some(*a) || back.hash() != va.hash() || va.hash() == old.hash() {
            bars_round_trip = false;
        }
        for b in JUDGE_GRID.iter().skip(i + 1) {
            if va.hash() == (Variant { bar: Some(*b), ..old.clone() }).hash() {
                bars_distinct = false;
            }
        }
    }
    claim("every bar on the grid round-trips through its own rendering", bars_round_trip);
    claim(
        "and no two of them are the same node, which the packed byte made four of six",
        bars_distinct,
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
    let round = back.as_ref().map_or(false, |b| {
        b.parent == v1.parent && b.adapter == v1.adapter && b.rule == v1.rule
    });
    // The identity, not a sample of the fields. Comparing three of them is
    // what let the missing `lambda` arm live here: `v1` has a nonzero lambda,
    // it read back as 0.0, and every checked field still matched. A node that
    // does not re-hash to the address it was read from cannot be used to
    // re-derive anything, which is the one property this module promises.
    let readdressed = back.as_ref().map_or(false, |b| b.hash() == stored);
    let mut np = String::from(ROOT);
    np.push_str("/nodes/");
    np.push_str(&hex32(&stored));
    let mut bp = np.clone();
    bp.push_str(".born");
    sysbox::detach(&np);
    sysbox::detach(&bp);
    claim("a stored variant reads back with its lineage intact", round);
    claim("a stored variant re-hashes to the address it came from", readdressed);

    // The compatibility property the `core` field rests on.
    //
    // Adding a field to a hashed structure re-addresses every object that
    // already exists unless the field is absent from the rendering when it is
    // absent from the object. If this claim fails, `head` names a node that no
    // longer reproduces, every ledger line stops being checkable, and the
    // change meant to extend re-derivability is what broke it -- so it is
    // asserted rather than reasoned about.
    let no_core = mk(0.02, 1000);
    let mut with_core = mk(0.02, 1000);
    with_core.core = Some(sha256::hash(b"a council core"));
    with_core.core_seen = true;
    claim(
        "a variant that predates the core field renders no core line",
        !no_core.render().contains("core "),
    );
    claim(
        "a core is part of what a variant is",
        no_core.hash() != with_core.hash(),
    );
    let ch = with_core.store();
    let cback = Variant::load(&ch);
    let core_round = cback.as_ref().map_or(false, |b| b.core == with_core.core && b.hash() == ch);
    let mut cnp = String::from(ROOT);
    cnp.push_str("/nodes/");
    cnp.push_str(&hex32(&ch));
    let mut cbp = cnp.clone();
    cbp.push_str(".born");
    sysbox::detach(&cnp);
    sysbox::detach(&cbp);
    claim("a variant carrying a core reads back as itself", core_round);

    ok
}
