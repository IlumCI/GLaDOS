//! Which coin each slice is working on.
//!
//! The miner had one job, one algorithm and one connection, and every slice
//! hashed the same header over a disjoint slice of the nonce space. That is the
//! right shape for mining one coin faster and the wrong shape for what
//! `design/mining.md` actually asks for, which is a machine taking a meaningful
//! share of a dozen small networks at once. This is the table that makes the
//! second thing expressible: `MAX_COINS` slots, each with its own algorithm and
//! its own job, and an assignment saying which slice works which slot.
//!
//! ### The aggregate hashrate stops meaning anything, and that is the point
//!
//! One `HASHES` counter was a fair summary while every slice computed the same
//! function. It is not one now: a yespower hash is roughly a thousand times the
//! work of a sha256d hash, so two slices on different algorithms produce a
//! combined H/s figure dominated by whichever algorithm is cheap, and it says
//! nothing about either. Every slot therefore carries its own counter and its
//! own start time, and the report prints a row per coin rather than a total.
//! `client::HASHES` survives as the *sweep's* counter, where every slice is on
//! one algorithm by construction.
//!
//! ### Assignment is sticky, and that is a memory decision rather than a policy
//!
//! A slice holds its hasher across batches because `Yespower` owns up to 8 MiB
//! of working set -- 8,296 KiB measured at N=2048, r=32 -- and a batch at that
//! setting is eight hashes. Rotating a slice between coins per batch would
//! throw that allocation away and take it again several times a second, which
//! spends more time in this kernel's locking allocator than in the algorithm.
//! So a slice is assigned to a slot and stays there until the table changes,
//! and `assign` is called when it does rather than from the hash loop.
//!
//! ### A share knows which slot it came from
//!
//! Only a slot with a pool behind it can submit anything. A fixture slot exists
//! to be measured and its target is one nothing meets -- but the day one does,
//! through a mistyped difficulty or a target of all ones, submitting it would
//! send the pool a share for a header it never issued, which is how a worker
//! gets banned. `Source` is on the slot and the miner checks it before queueing.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::sync::{Guard, Spin};

use super::algo::{Algo, Bound};
use super::client::{Template, MAX_SLICES};

/// How many coins may be worked at once.
///
/// Four, matching `MAX_SLICES`, because a slot with no slice on it produces
/// nothing -- and the ceiling on slices is the task table rather than anything
/// about coins. More slots than slices would let the operator build a table
/// where some coins are silently never mined, which reads from the report
/// exactly like a coin whose pool has gone quiet.
pub const MAX_COINS: usize = 4;

/// Where a slot's work comes from.
#[derive(Clone, Copy, PartialEq)]
pub enum Source {
    /// The Stratum connection. Shares from here are submitted.
    Pool,
    /// A synthetic job for measurement. Shares from here are dropped.
    Fixture,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Pool => "pool",
            Source::Fixture => "fixture",
        }
    }
}

/// One coin being worked.
pub struct Coin {
    /// What to call it in the report. Free text: the kernel has no way to check
    /// that a label and an algorithm describe the same network, and inventing a
    /// per-coin preset table would be this tree asserting parameters about
    /// somebody else's chain without having read their source -- the objection
    /// `Algo` already makes about presets.
    pub label: String,
    pub algo: Algo,
    pub source: Source,
    /// `None` until the source has issued work.
    pub template: Option<Template>,
}

/// Slot 0 is the Stratum connection's. Everything else is installed by hand.
///
/// One `Spin` per slot rather than a `Spin<[Option<Coin>; N]>`: the socket task
/// rebuilds slot 0 on every `mining.notify` while three unpinned slices read
/// their own slots, and one lock over the whole table would put them all behind
/// a job update they have nothing to do with.
static COINS: [Spin<Option<Coin>>; MAX_COINS] = [
    Spin::new(None),
    Spin::new(None),
    Spin::new(None),
    Spin::new(None),
];

/// Hashes computed against each slot, when it started counting, and shares
/// found. TSC milliseconds, never `lapic::ticks` -- see `client::now_ms`.
static C_HASHES: [AtomicU64; MAX_COINS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static C_SINCE: [AtomicU64; MAX_COINS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static C_FOUND: [AtomicU64; MAX_COINS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Which slot each slice works. `NONE` when there is nothing for it to do.
const NONE: u32 = u32::MAX;
// Written out rather than `[const { .. }; MAX_SLICES]`, to match the table
// above it -- and long enough now that the next change to `MAX_SLICES` will be
// a compile error here rather than a silent mismatch, which is what it was.
static ASSIGN: [AtomicU32; MAX_SLICES] = [
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
    AtomicU32::new(NONE),
];

/// Borrow one slot. The caller holds a lock, so do not hash under it.
pub fn coin(slot: usize) -> Guard<'static, Option<Coin>> {
    COINS[slot.min(MAX_COINS - 1)].lock_irq()
}

/// The algorithm a slot computes, or `None` when the slot is empty.
pub fn algo(slot: usize) -> Option<Algo> {
    COINS.get(slot)?.lock_irq().as_ref().map(|c| c.algo.clone())
}

pub fn label(slot: usize) -> Option<String> {
    COINS.get(slot)?.lock_irq().as_ref().map(|c| c.label.clone())
}

pub fn occupied() -> usize {
    COINS.iter().filter(|s| s.lock_irq().is_some()).count()
}

/// Put a coin in a slot, keeping the job already there when the algorithm has
/// not changed.
///
/// Keeping it matters for slot 0: `mine algo` while connected would otherwise
/// discard the pool's current job and the miner would idle until the next
/// `mining.notify`, which on a slow pool is a minute of nothing happening for
/// no reason the operator can see. When the algorithm *has* changed the job
/// goes, because a header assembled for one chain hashed under another
/// algorithm produces shares that are valid arithmetic and belong to nothing.
pub fn install(slot: usize, label: &str, algo: Algo, source: Source) -> bool {
    let Some(cell) = COINS.get(slot) else {
        return false;
    };
    let mut changed = true;
    {
        let mut g = cell.lock_irq();
        let keep = match g.take() {
            Some(c) if c.algo == algo && c.source == source => {
                changed = false;
                c.template
            }
            _ => None,
        };
        *g = Some(Coin {
            label: String::from(label),
            algo,
            source,
            template: keep,
        });
    }
    if changed {
        reset(slot);
    }
    assign();
    true
}

pub fn clear(slot: usize) -> bool {
    let Some(cell) = COINS.get(slot) else {
        return false;
    };
    *cell.lock_irq() = None;
    reset(slot);
    assign();
    true
}

/// Install a job into a slot. Answers false when the slot holds no coin, which
/// is what a `mining.notify` arriving after `mine coin 0 off` looks like.
pub fn set_template(slot: usize, t: Template) -> bool {
    let Some(cell) = COINS.get(slot) else {
        return false;
    };
    let mut g = cell.lock_irq();
    let ok = match g.as_mut() {
        Some(c) => {
            c.template = Some(t);
            true
        }
        None => false,
    };
    drop(g);
    if ok {
        // A slot only becomes workable when it has a job, so the supervisor has
        // to hear about it. Cheap in steady state: `assign` resets nothing
        // unless a slot's share of the slices actually moved, and a job
        // replacing a job moves nothing.
        assign();
    }
    ok
}

pub fn drop_template(slot: usize) {
    if let Some(cell) = COINS.get(slot) {
        if let Some(c) = cell.lock_irq().as_mut() {
            c.template = None;
        }
    }
    // The slot stops being workable, so its slices go to coins that are. A
    // disconnected pool must not hold a quarter of the machine idle.
    assign();
}

/// Everything the hash loop needs, taken under the lock in one go.
///
/// Returned by value rather than as a guard because the loop hashes for a
/// millisecond and a half afterwards, and holding a slot's lock across that
/// would block the socket task's job update for the length of every batch.
pub struct Snapshot {
    pub algo: Algo,
    pub serial: u64,
    pub header: [u8; 80],
    pub target: super::u256::U256,
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime_be: Vec<u8>,
    pub echo: super::proto::Echo,
    pub verified: bool,
    pub submits: bool,
}

pub fn snapshot(slot: usize) -> Option<Snapshot> {
    let g = COINS.get(slot)?.lock_irq();
    let c = g.as_ref()?;
    let t = c.template.as_ref()?;
    Some(Snapshot {
        algo: c.algo.clone(),
        serial: t.serial,
        header: t.header,
        target: t.target,
        job_id: t.job_id.clone(),
        extranonce2: t.extranonce2.clone(),
        ntime_be: t.ntime_be.clone(),
        echo: t.echo.clone(),
        verified: t.verified,
        submits: c.source == Source::Pool,
    })
}

/// Spread the wanted slices over the occupied slots, round robin.
///
/// Deterministic and re-derivable from the table alone, so two runs of the same
/// commands assign identically -- the property `godel::frontier` wants of its
/// grid walk, for the same reason: a rate that depends on which slice happened
/// to get which coin is not a measurement.
///
/// Slices beyond the wanted count are cleared rather than left pointing at a
/// slot, because a parked slice that still reads as assigned makes the report
/// claim two slices on a coin that one is working.
/// How many slices the last-level cache can actually carry, given what the
/// workable slots are running.
///
/// **Slices run at once, so their working sets are resident at once.** A
/// memory-bound algorithm exists to exceed a core's private cache -- that is
/// the mechanism, not a side effect -- so concurrent jobs contend in the last
/// level and upstream's own PERFORMANCE file says eight threads is already
/// "substantial slowdown". Handing out more slices than the cache holds is
/// therefore not neutral: it buys thrashing, and the report shows a rate that
/// went *down* when more of the machine was given to it.
///
/// An arithmetic-bound algorithm costs nothing here. sha256d's whole state is
/// a few hundred bytes, so a slice on one is free however many are running,
/// which is the practical form of `Algo::bound` -- and the reason a mixed
/// table is worth more than a uniform one.
///
/// **A cache this cannot read is not a reason to refuse to mine.** With no
/// answer from CPUID the request stands unchanged, which is exactly the
/// behaviour every build before this one had. A budget that silently throttled
/// a machine it could not measure would be worse than no budget.
fn cache_budget(slots: &[usize], want: usize) -> usize {
    // The largest working set among the workable coins, because slices are
    // handed out round-robin and any of them may land on the worst one. Taking
    // the mean would be right on average and wrong exactly when it matters.
    let mut worst = 0usize;
    for &i in slots {
        if let Some(c) = COINS[i].lock_irq().as_ref() {
            if matches!(c.algo.bound(), Bound::Memory) {
                let w = c.algo.working_set();
                if w > worst {
                    worst = w;
                }
            }
        }
    }
    budget_from(crate::cpu::last_level_cache(), worst, want)
}

/// The arithmetic, pure and therefore assertable.
///
/// Separated from the gathering above for the reason `update::decide` and
/// `code::locate` are: every branch here is a decision about how much of the
/// machine to use, two of them are refusals, and a suite can walk all four
/// without a processor, a coin table or a slice.
pub fn budget_from(cache: Option<usize>, worst: usize, want: usize) -> usize {
    // No answer from CPUID means the request stands, which is exactly what
    // every build before this one did. A budget that silently throttled a
    // machine it could not measure would be worse than no budget.
    let Some(cache) = cache else {
        return want;
    };
    // Nothing memory-bound is being worked, so the cache is not the resource
    // in question and the core count is.
    if worst == 0 {
        return want;
    }
    // Never zero: one slice thrashing is still strictly better than a coin
    // nobody works, and a budget that could refuse every slice would turn a
    // large-parameter coin into a silent no-op that reads like a dead pool.
    let fits = (cache / worst).max(1);
    want.min(fits)
}

pub fn assign() {
    let before: [u32; MAX_COINS] = core::array::from_fn(|i| slices_on(i));

    // Only slots that can actually be worked. A coin with no job yet is a real
    // coin and shows in the report as one, but a slice given to it parks --
    // measured: `mine algo blake2s` with no pool created slot 0, `assign` gave
    // it two of three slices on the strength of it existing, and those two did
    // nothing at all while a coin with work sat on one. Workability is having a
    // job, so this is re-run when a job arrives and when one goes.
    let mut slots = [0usize; MAX_COINS];
    let mut n = 0;
    for (i, cell) in COINS.iter().enumerate() {
        if cell.lock_irq().as_ref().map(|c| c.template.is_some()).unwrap_or(false) {
            slots[n] = i;
            n += 1;
        }
    }
    let want = cache_budget(&slots[..n], super::client::slices() as usize);
    for (i, a) in ASSIGN.iter().enumerate() {
        if n == 0 || i >= want {
            a.store(NONE, Ordering::Relaxed);
        } else {
            a.store(slots[i % n] as u32, Ordering::Relaxed);
        }
    }

    // A slot whose slice count moved starts counting again, for exactly the
    // reason a slot whose algorithm changed does: the figure would otherwise
    // average two different machines. Measured, and this is why it is here
    // rather than argued -- clearing a second coin took slot 1 from two slices
    // to three and it reported 792 H/s, which is neither the two-slice rate
    // (710) nor the three-slice one, and looks like a plausible number for
    // either.
    for i in 0..MAX_COINS {
        if slices_on(i) != before[i] {
            reset(i);
        }
    }
}

pub fn slot_for(slice: u32) -> Option<usize> {
    let v = ASSIGN.get(slice as usize)?.load(Ordering::Relaxed);
    if v == NONE {
        None
    } else {
        Some(v as usize)
    }
}

/// How many slices are on a slot. For the report, and for saying out loud when
/// the answer is zero.
pub fn slices_on(slot: usize) -> u32 {
    ASSIGN
        .iter()
        .filter(|a| a.load(Ordering::Relaxed) == slot as u32)
        .count() as u32
}

pub fn count(slot: usize, hashes: u64) {
    if let Some(c) = C_HASHES.get(slot) {
        c.fetch_add(hashes, Ordering::Relaxed);
    }
    if let Some(s) = C_SINCE.get(slot) {
        if s.load(Ordering::Relaxed) == 0 {
            s.store(super::client::now_ms(), Ordering::Relaxed);
        }
    }
}

pub fn found(slot: usize) {
    if let Some(c) = C_FOUND.get(slot) {
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// Hashes, milliseconds and shares found for a slot.
pub fn rate(slot: usize) -> (u64, u64, u64) {
    let h = C_HASHES
        .get(slot)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0);
    let t0 = C_SINCE
        .get(slot)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0);
    let f = C_FOUND
        .get(slot)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0);
    let ms = if t0 == 0 {
        0
    } else {
        super::client::now_ms().saturating_sub(t0)
    };
    (h, ms, f)
}

/// Forget a slot's figures. Called whenever its coin changes, because a rate
/// spanning two algorithms is not a rate for either.
pub fn reset(slot: usize) {
    if let Some(c) = C_HASHES.get(slot) {
        c.store(0, Ordering::Relaxed);
    }
    if let Some(c) = C_SINCE.get(slot) {
        c.store(0, Ordering::Relaxed);
    }
    if let Some(c) = C_FOUND.get(slot) {
        c.store(0, Ordering::Relaxed);
    }
}

/// A job nothing will ever solve, for measuring a slot without a pool.
///
/// The target is zero rather than merely hard: measuring hashing is the whole
/// purpose, and a target that is difficult still spends time building and
/// queueing shares on a slow machine, which the curve then reports as hashing.
///
/// `seed` differs per slot so two fixture slots do not hash identical headers.
/// Nothing depends on that for correctness -- the rate does not care which
/// bytes go in -- but identical inputs across slots is exactly the shape that
/// would let a caching bug read as concurrency.
pub fn fixture_template(seed: u8) -> Template {
    let header: [u8; 80] = core::array::from_fn(|i| ((i as u32 * 3) as u8) ^ seed);
    Template {
        serial: super::client::bump_serial(),
        job_id: String::from("fixture"),
        extranonce2: alloc::vec![0u8; 4],
        ntime_be: alloc::vec![0u8; 4],
        header,
        target: super::u256::U256::ZERO,
        nbits: 0x1d00_ffff,
        coin_value: None,
        coinbase_len: 0,
        coinbase_head: [0u8; 8],
        echo: alloc::vec::Vec::new(),
        // A fixture is this machine's own invention and says so. Claiming a
        // verified job here would put "checked" beside a header nobody issued.
        verified: false,
    }
}

/// Claims. Pure over the table, so none of this needs a pool or a task.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // Saved and restored, because `diag` may run on a machine that is mining.
    // The templates are not saved: a job is re-sent within seconds by any live
    // pool, and copying one here would mean this suite carried a second way to
    // build a `Template` that could drift from the first.
    let mut saved: Vec<Option<(String, Algo, Source)>> = Vec::new();
    for i in 0..MAX_COINS {
        saved.push(
            coin(i)
                .as_ref()
                .map(|c| (c.label.clone(), c.algo.clone(), c.source)),
        );
    }
    let saved_assign: Vec<u32> = ASSIGN.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    for i in 0..MAX_COINS {
        clear(i);
    }

    out.push((
        "an empty table assigns no slice a slot",
        (0..MAX_SLICES as u32).all(|s| slot_for(s).is_none()),
    ));

    let want = super::client::slices();
    install(0, "a", Algo::Sha256d, Source::Fixture);
    // A coin with no job is a coin in the report and not a coin to work.
    out.push((
        "a coin with no job gets no slice",
        (0..MAX_SLICES as u32).all(|s| slot_for(s).is_none()),
    ));
    set_template(0, fixture_template(0));
    // Every slice on the only coin there is. The alternative -- one slice on it
    // and the rest idle -- is what a naive one-to-one mapping gives, and it
    // would leave a single-coin machine three quarters idle without saying so.
    out.push((
        "one coin takes every wanted slice",
        (0..want).all(|s| slot_for(s) == Some(0)) && slices_on(0) == want,
    ));

    install(1, "b", Algo::Sha256d, Source::Fixture);
    set_template(1, fixture_template(1));
    out.push((
        "two coins alternate across slices",
        slot_for(0) == Some(0) && (want < 2 || slot_for(1) == Some(1)),
    ));

    // Losing the job is not losing the coin, and the slices must move anyway.
    drop_template(1);
    out.push((
        "a coin that loses its job loses its slices",
        (0..MAX_SLICES as u32).all(|s| slot_for(s) != Some(1)),
    ));
    set_template(1, fixture_template(1));

    // The one that matters. A slice told to work a slot that was emptied hashes
    // against whatever lands there next, which produces perfectly valid shares
    // for a network nobody asked it to mine.
    clear(0);
    out.push((
        "clearing a coin moves its slices off it",
        (0..MAX_SLICES as u32).all(|s| slot_for(s) != Some(0)),
    ));

    install(2, "c", Algo::Sha256d, Source::Fixture);
    set_template(2, fixture_template(0));
    out.push((
        "a fixture slot refuses to submit",
        snapshot(2).map(|s| !s.submits).unwrap_or(false),
    ));
    install(2, "c", Algo::Sha256d, Source::Pool);
    set_template(2, fixture_template(0));
    out.push((
        "a pool slot submits",
        snapshot(2).map(|s| s.submits).unwrap_or(false),
    ));

    // A slot with no template answers nothing rather than a zeroed job: an
    // all-zero snapshot hashes happily and every share it finds is against a
    // header no network issued.
    clear(3);
    install(3, "d", Algo::Sha256d, Source::Pool);
    out.push(("a coin with no job yields no snapshot", snapshot(3).is_none()));

    set_template(3, fixture_template(3));
    count(3, 1000);
    install(3, "e", Algo::Yespower { v10: true, n: 2048, r: 8, pers: None }, Source::Pool);
    out.push(("changing the algorithm forgets the rate", rate(3).0 == 0));

    // And the other half of that: renaming a coin without changing what it
    // computes must *not* forget, or `mine coin 0 <label>` on a running miner
    // silently resets the figure it was about to be read for.
    count(3, 1000);
    install(3, "f", Algo::Yespower { v10: true, n: 2048, r: 8, pers: None }, Source::Pool);
    out.push(("relabelling the same algorithm keeps the rate", rate(3).0 == 1000));

    // Adding a coin re-spreads the slices, so every slot whose share of them
    // moved has to start counting again. Without this the report averages a
    // two-slice epoch and a three-slice one into a rate that was never true.
    clear(1);
    clear(2);
    clear(3);
    install(0, "a", Algo::Sha256d, Source::Fixture);
    set_template(0, fixture_template(0));
    count(0, 5000);
    let moved = want > 1;
    install(1, "b", Algo::Sha256d, Source::Fixture);
    set_template(1, fixture_template(1));
    out.push((
        "re-spreading the slices forgets the rates it changed",
        !moved || rate(0).0 == 0,
    ));

    for i in 0..MAX_COINS {
        clear(i);
    }
    for (i, s) in saved.into_iter().enumerate() {
        if let Some((l, a, src)) = s {
            install(i, &l, a, src);
        }
    }
    for (i, v) in saved_assign.into_iter().enumerate() {
        ASSIGN[i].store(v, Ordering::Relaxed);
    }
    out
}
