//! How much a night may spend, and the argument that it cannot waste much.
//!
//! **`GODEL_EXAMPLES` was a constant chosen by measurement, and the
//! measurement was about one axis.** 24 examples made J1 arithmetically
//! impossible -- five wrong answers on the held-out slice against a
//! requirement of six clean repairs -- so it became 96, where the headroom is
//! 26 against a need of 6. That fixed the adapter axis and charged every other
//! axis four times over for it: `lib` judges a library function against a
//! solver and needs no corpus subsample at all to reach a verdict, and it pays
//! 321 seconds of prepare either way.
//!
//! A constant cannot be right for both. The honest answer is not a better
//! constant; it is to not know, and to search.
//!
//! ### The schedule, and the guarantee that makes it safe to not know
//!
//! Schmidhuber's Optimal Ordered Problem Solver (2004) answers exactly this
//! shape of question: given a search whose right time limit is unknown, how do
//! you allocate so that no more than a constant factor is wasted? The answer
//! is **doubling**, and the arithmetic is the whole of it. Reaching level `L`
//! by trying every level below it costs
//!
//!     base * (1 + 2 + 4 + ... + 2^L)  =  base * (2^(L+1) - 1)  <  2 * base * 2^L
//!
//! which is under twice what an oracle that already knew `L` would have spent.
//! Not knowing costs a factor of two, forever, whatever `L` turns out to be.
//!
//! The other half of OOPS is that the schedule never commits: half the budget
//! extends the current best prefix and half starts fresh. Here that is nights
//! rather than instruction steps -- a night at the axis's raised level, then a
//! night at the base -- which bounds the *average* spend at `(base + raised) /
//! 2` however high the raised level has climbed. Multiply the two and the
//! whole scheme wastes at most four times an oracle's budget, which is the
//! bounded-waste constant this module exists to be able to state.
//!
//! ### What raises a level, and what must not
//!
//! **Only a trial the budget could not have decided.** `Certificate.wrong`
//! carries the ceiling on how many decisions there were to repair, and
//! `clean_fixes_needed()` carries how many the bar asks for; a trial where the
//! ceiling is below the requirement did not fail, it was never asked. That is
//! a budget fact and doubling is the right response.
//!
//! A trial that had the evidence and was refused anyway is a *candidate*
//! failure, and doubling for it would be the loop spending more and more on
//! the same ground because it did not like the answer. So a verdict of any
//! kind, reached on sufficient evidence, drops the axis back to the base.
//!
//! And a line that says nothing about its ceiling -- every line written before
//! `wrong=` existed -- is not starved. "Did not say" is not "could not pass",
//! and reading it as the latter would raise every axis on the strength of the
//! early history being silent.

use alloc::string::{String, ToString};

/// The subsample a night starts from.
///
/// 24, which is where the measurement that condemned it as a *constant* was
/// taken. It is the right **floor** for the same reason it was the wrong
/// constant: an axis that can reach a verdict here should not be charged for
/// one that cannot.
///
/// Measured under QEMU on the seeded 717-example corpus, both figures from the
/// trial report's own lines:
///
///     examples   validation decisions   incumbent wrong   J1 needs
///     24         15                     5                 6
///     96         56                     26                6
///
/// `clean_fixes_needed()` is six, not `MIN_FIXED`, because Yates' correction
/// subtracts one before squaring. At 24 the baseline gets **five** validation
/// decisions wrong, so five is the ceiling on `fixed` and six is the floor on
/// passing: the judge was asking for more repairs than there were wrong
/// answers to repair. Not a hard trial, an arithmetically impossible one --
/// which is precisely the condition `starved` detects and this schedule
/// answers by doubling, rather than by somebody noticing a year later.
///
/// The price of the raised level is the prepare half, a forward pass per
/// example and *not* what the millisecond ceiling bounds: 63 s at 24 and 321 s
/// at 96 under WHPX with SmolLM2. Up to five minutes of held engine on a night
/// when nobody is there, which is what `godbits::felt()` is checked for, and
/// it is why the alternation matters as much as the doubling.
pub const BASE_EXAMPLES: usize = 24;

/// The optimiser ceiling a night starts from, doubling alongside the examples.
///
/// Both halves double together because they bound the same trial: raising the
/// subsample without raising the time gives a run that prepares more features
/// and then stops mid-descent, which is a different trial rather than a bigger
/// one.
pub const BASE_MS: u64 = 5_000;

/// How far the doubling may go. Level 3 is 192 examples and 40 s.
///
/// A cap rather than a limit the arithmetic needs: the guarantee above holds
/// for any `L`. What it bounds is the engine being held on a night when
/// somebody might come back, which `godbits::felt()` checks for at the start
/// and cannot re-check in the middle of a prepare.
pub const MAX_LEVEL: usize = 3;

/// Which half of the schedule a night is on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Half {
    /// Spend at the level this axis's record justifies.
    Extend,
    /// Spend at the base, whatever the record says. This is the half that
    /// makes the waste bounded rather than merely finite.
    Fresh,
}

/// What a night may spend.
#[derive(Clone, Copy)]
pub struct Plan {
    pub level: usize,
    pub half: Half,
    pub examples: usize,
    pub ms: u64,
}

pub fn examples(level: usize) -> usize {
    BASE_EXAMPLES << level.min(MAX_LEVEL)
}

pub fn ms(level: usize) -> u64 {
    BASE_MS << level.min(MAX_LEVEL) as u32
}

/// Could this trial have reached a verdict at all?
///
/// `wrong` is the ceiling on repairs available and `need` is what the bar
/// asks. `None` is a line that did not say, which is not a claim that it could
/// not pass.
pub fn starved(wrong: Option<usize>, need: usize) -> bool {
    matches!(wrong, Some(w) if w < need)
}

/// The `wrong=` field of a ledger line, if it carries one.
pub fn wrong_of(line: &str) -> Option<usize> {
    let at = line.find(" wrong=")? + 7;
    let rest = &line[at..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The `ex=` field: the subsample the trial was *given*.
pub fn ex_of(line: &str) -> Option<usize> {
    let at = line.find(" ex=")? + 4;
    let rest = &line[at..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The `corpus=` field, which says which body of evidence a line was paid for
/// out of.
pub fn corpus_of(line: &str) -> Option<&str> {
    let at = line.find(" corpus=")? + 8;
    let rest = &line[at..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Which level a budget is, as a count of doublings from the base.
pub fn level_of(examples: usize) -> usize {
    let mut level = 0usize;
    let mut at = BASE_EXAMPLES;
    while at < examples && level < MAX_LEVEL {
        at *= 2;
        level += 1;
    }
    level
}

/// How many doublings this axis's own record justifies.
///
/// **The largest budget at which it starved, and the smallest at which it
/// decided.** If it has decided something at a budget above every starvation,
/// use that: the search has found a level that works and there is nothing left
/// to double for. Otherwise go one above the highest starvation, which is the
/// doubling.
///
/// The first version counted the *trailing run* of starved trials, which
/// converges only by a coincidence: it works because the `Fresh` nights keep
/// injecting starved lines at the base, and it would stop working the day they
/// stopped failing. A schedule whose convergence depends on something
/// continuing to go wrong is a schedule that fails silently, which is the
/// class of defect this module was written to remove rather than to join.
///
/// Reading `ex=` is what makes the explicit rule possible, and that field had
/// to be added: `n=` is validation decisions, derived from the budget and not
/// invertibly, so a schedule that inferred the budget would be a second
/// account of the record free to disagree with it.
///
/// **Scoped to one corpus.** Without that it is a ratchet -- an axis that
/// starved at 192 once stays at 192 forever, even after the corpus grew enough
/// that 24 would do. The ledger already records which body of evidence each
/// line was paid for out of, for the family-wise budget's sake, and the same
/// field answers this. A corpus this machine has no lines for starts at the
/// base, which is the right place to start.
///
/// Another axis's record is invisible here. They are different questions asked
/// of different judges, and one being unanswerable at 24 examples says nothing
/// about the next.
pub fn level_for(lines: &[String], axis: usize, need: usize, corpus: Option<&str>) -> usize {
    let (mut starved_at, mut decided_at) = (0usize, None::<usize>);
    for line in lines {
        if super::godel::axis_of(line) != Some(axis) {
            continue;
        }
        if let Some(c) = corpus {
            if corpus_of(line) != Some(c) {
                continue;
            }
        }
        // A line with no `ex=` was written before the field existed and says
        // nothing about what it was given. Skipped rather than assumed, the
        // same way a line with no `wrong=` is not read as starved.
        let Some(ex) = ex_of(line) else { continue };
        if starved(wrong_of(line), need) {
            starved_at = starved_at.max(ex);
        } else {
            decided_at = Some(decided_at.map_or(ex, |d| d.min(ex)));
        }
    }
    if let Some(d) = decided_at {
        if d > starved_at {
            return level_of(d);
        }
    }
    if starved_at == 0 {
        return 0;
    }
    (level_of(starved_at) + 1).min(MAX_LEVEL)
}

/// Which half tonight is, from the record rather than from a counter.
///
/// Ledger length, so a later reader can say which half any past night was on.
/// A counter in its own file could disagree with the ledger, and then the
/// loop's account of what it spent would be unfalsifiable exactly where it
/// most needs not to be -- the objection `axis_counts` already makes.
pub fn half_of(ledger_len: usize) -> Half {
    if ledger_len % 2 == 0 {
        Half::Extend
    } else {
        Half::Fresh
    }
}

/// The whole allocation for one night on one axis.
pub fn plan(lines: &[String], axis: usize, need: usize, corpus: Option<&str>) -> Plan {
    let half = half_of(lines.len());
    let level = match half {
        Half::Extend => level_for(lines, axis, need, corpus),
        Half::Fresh => 0,
    };
    Plan { level, half, examples: examples(level), ms: ms(level) }
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        ok &= good;
    };

    claim(examples(0) == BASE_EXAMPLES, "level zero is the base");
    claim(
        examples(1) == 48 && examples(2) == 96 && ms(1) == 10_000,
        "and each level doubles both halves of the budget together",
    );
    claim(
        examples(99) == examples(MAX_LEVEL) && ms(99) == ms(MAX_LEVEL),
        "a level past the cap is the cap rather than an overflow",
    );

    // **The guarantee, as arithmetic rather than as a citation.** Trying every
    // level up to L costs less than twice what knowing L would have cost. This
    // is the whole reason not knowing is acceptable.
    for l in 0..=MAX_LEVEL {
        let tried: usize = (0..=l).map(examples).sum();
        let oracle = examples(l);
        if tried >= 2 * oracle {
            claim(false, "doubling costs under twice an oracle's budget");
            break;
        }
        if l == MAX_LEVEL {
            claim(true, "doubling costs under twice an oracle's budget");
        }
    }
    // And the alternation bounds the average, which is the half of OOPS that
    // stops a raised level becoming a permanent commitment.
    let raised = examples(MAX_LEVEL);
    let pair = raised + examples(0);
    claim(
        pair < 2 * raised,
        "and alternating with the base keeps two nights under two raised ones",
    );

    // Starvation is a fact about the budget, never about the candidate.
    claim(starved(Some(5), 6), "a ceiling below the requirement is a starved trial");
    claim(!starved(Some(6), 6), "a ceiling that just meets it is not");
    claim(!starved(Some(26), 6), "and one far above it is not, however the trial went");
    claim(
        !starved(None, 6),
        "a line that did not say its ceiling is not read as having failed to pass",
    );

    claim(wrong_of("x wrong=5 chi=1.00 no]") == Some(5), "the ceiling is read off a line");
    claim(wrong_of("x fix=0 broke=0 chi=1.00") .is_none(), "and a line without one says nothing");

    // The level, from a record. `axis=adapter` is slot 0 and `axis=lib` is a
    // different axis, so one starving must not raise the other.
    let line = |axis: &str, ex: usize, wrong: usize, corpus: &str| {
        alloc::format!(
            "1 h3 parent=root.... variant=000000aa axis={} corpus={} cell=0 n=24              J1[fix=0 broke=0 wrong={} ex={} chi=0.00 no] reject",
            axis, corpus, wrong, ex
        )
    };
    let starved_at = |ex: usize| line("adapter", ex, 5, "aaaa1111");
    let decided_at = |ex: usize| line("adapter", ex, 26, "aaaa1111");

    let a0 = super::godel::axis_of(&starved_at(24));
    claim(a0.is_some(), "the axis names in these fixtures are ones the ledger uses");
    let a0 = a0.unwrap_or(0);
    claim(
        ex_of(&starved_at(48)) == Some(48) && wrong_of(&starved_at(48)) == Some(5),
        "a line gives up the budget it was given and the ceiling it reached",
    );
    claim(
        corpus_of(&starved_at(24)) == Some("aaaa1111"),
        "and which body of evidence it was paid for out of",
    );
    claim(
        level_of(BASE_EXAMPLES) == 0 && level_of(BASE_EXAMPLES * 4) == 2,
        "a budget maps back to the level that produced it",
    );
    claim(
        level_of(BASE_EXAMPLES * 64) == MAX_LEVEL && level_of(1) == 0,
        "and one off the end of the schedule clamps rather than looping",
    );

    let none: [String; 0] = [];
    claim(level_for(&none, a0, 6, None) == 0, "an empty ledger justifies no doubling");

    let one = [starved_at(24)];
    claim(level_for(&one, a0, 6, None) == 1, "starving at the base raises the level by one");

    // **The case the first version got right only by luck.** It counted the
    // trailing run, so a decided trial anywhere reset it -- and the schedule
    // converged because the `Fresh` nights kept failing at the base. This asks
    // the question directly: it starved at 24 and decided at 48, so 48 is what
    // works and there is nothing left to double for.
    let settled = [starved_at(24), decided_at(48)];
    claim(
        level_for(&settled, a0, 6, None) == 1,
        "a level it decided at is kept, even with a starvation below it",
    );
    let settled_then_base = [starved_at(24), decided_at(48), starved_at(24)];
    claim(
        level_for(&settled_then_base, a0, 6, None) == 1,
        "and a later failure at the base does not undo what the higher level proved",
    );

    let worse = [decided_at(24), starved_at(48)];
    claim(
        level_for(&worse, a0, 6, None) == 2,
        "while a starvation above a lucky cheap verdict doubles past it",
    );

    let cheap = [decided_at(24)];
    claim(
        level_for(&cheap, a0, 6, None) == 0,
        "an axis that decides at the base stays at the base",
    );

    // The one that keeps a cheap axis cheap. `lib` starving says nothing about
    // `adapter`, and a shared counter would have charged both.
    let other = super::godel::axis_of(&line("lib", 24, 5, "aaaa1111")).unwrap_or(a0 + 1);
    let cross = [line("lib", 96, 5, "aaaa1111")];
    claim(
        other != a0
            && level_for(&cross, a0, 6, None) == 0
            && level_for(&cross, other, 6, None) == MAX_LEVEL,
        "one axis starving does not raise another's budget",
    );

    // **Scoped to one corpus, or it is a ratchet.** An axis that starved at
    // 192 once would otherwise stay at 192 forever, including after the corpus
    // grew enough that the base would do.
    let older = [starved_at(96)];
    claim(
        level_for(&older, a0, 6, Some("aaaa1111")) == MAX_LEVEL
            && level_for(&older, a0, 6, Some("bbbb2222")) == 0,
        "a record paid for out of another corpus does not bind this one",
    );

    // A line written before `ex=` existed says nothing about what it was given.
    let silent = ["1 h3 parent=root.... variant=000000aa axis=adapter cell=0 n=24 reject".to_string()];
    claim(
        level_for(&silent, a0, 6, None) == 0,
        "and a line that never recorded its budget is skipped rather than assumed",
    );

    // The halves, and that they are a function of the record.
    claim(
        half_of(0) == Half::Extend && half_of(1) == Half::Fresh && half_of(2) == Half::Extend,
        "the halves alternate with the ledger's length",
    );
    let p = plan(&settled, a0, 6, None);
    claim(
        p.half == Half::Extend && p.level == 1 && p.examples == 48,
        "an extending night spends what the axis's record justifies",
    );
    let three = [starved_at(24), starved_at(48), starved_at(96)];
    let p = plan(&three, a0, 6, None);
    claim(
        p.half == Half::Fresh && p.level == 0 && p.examples == BASE_EXAMPLES,
        "and a fresh night spends the base however high the record has climbed",
    );

    ok
}
