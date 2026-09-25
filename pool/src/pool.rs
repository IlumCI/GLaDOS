//! What the pool knows: which coins it serves, and which shares were good.
//!
//! Deliberately free of sockets. Everything here is a pure function of state
//! plus a message, so the whole of share validation can be exercised without a
//! listener, a miner, or a network -- the same separation `update::decide` and
//! `code::locate` get, and for the same reason: the interesting failures are
//! decisions, and a decision that needs a TCP connection to reproduce is one
//! nobody reproduces.

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::mine::algo::{Algo, Hasher};
use crate::mine::hash::below_target;
use crate::mine::header;
use crate::mine::proto;
use crate::mine::u256::U256;

/// A coin this pool serves.
pub struct Coin {
    pub label: String,
    /// Which traded asset this coin is, as `tools/prices.py` names it.
    ///
    /// Separate from `label` because a label is the operator's shorthand --
    /// `btc`, `xvg`, `zeny` -- and a price file has to be keyed by something
    /// two independent sources agree on. Defaulting it to the label is what
    /// makes it optional; naming it is what makes `btc` findable.
    pub asset: String,
    pub algo: Algo,
    /// Where a *new* miner starts, in leading zero bits. Only a starting
    /// point: `server.rs` moves each connection from here, per coin, so this
    /// is the operator's guess rather than a setting anybody has to get right.
    pub share_bits: u32,
    /// The same thing as a target, kept so a coin with no connection on it
    /// still reports something meaningful.
    pub share_target: U256,
    /// The network's own target, when a chain is behind this coin. `None` for
    /// a coin with no upstream, which is every coin today -- and it is `None`
    /// rather than a plausible constant, so `ev` prints "cannot say" instead of
    /// an expected value derived from a number nobody measured.
    pub network_target: Option<U256>,
    /// Where the header comes from. See `Source`.
    pub source: Source,
    /// The latest `mining.notify` from upstream. `None` for a local coin, and
    /// also for an upstream one that has not connected yet -- which is why
    /// `make_job` answers `None` rather than inventing a header.
    pub work: Option<Work>,
    /// Counter feeding extranonce2, so two jobs from one `mining.notify` search
    /// different coinbases. Monotonic and never reset within a connection: a
    /// repeat would hand two miners the same space and pay one of them for the
    /// other's work.
    pub e2: u64,
}

#[derive(Clone)]
pub enum Source {
    /// The pool builds the header itself. No chain, so a share that beat the
    /// network target would still be worth nothing -- which is why a coin in
    /// this state is reported as such rather than quietly served.
    Local,
    /// A real Stratum V1 pool upstream. Work comes from its `mining.notify`
    /// and qualifying shares go back as `mining.submit`.
    Upstream {
        host: String,
        port: u16,
        user: String,
        pass: String,
    },
}

impl Source {
    pub fn name(&self) -> &'static str {
        match self {
            Source::Local => "local",
            Source::Upstream { .. } => "upstream",
        }
    }
}

/// What an upstream pool last told us, before any of it becomes a header.
///
/// Held verbatim rather than pre-assembled, because the extranonce2 changes per
/// job handed downstream and the coinbase has to be rebuilt around it -- which
/// is the whole mechanism by which two miners search different spaces.
#[derive(Clone)]
pub struct Work {
    pub job_id: String,
    pub prev_wire: [u8; 32],
    pub coinb1: Vec<u8>,
    pub coinb2: Vec<u8>,
    pub branch: Vec<[u8; 32]>,
    pub version: u32,
    pub ntime: u32,
    pub nbits: u32,
    pub extranonce1: Vec<u8>,
    pub extranonce2_size: usize,
    /// What upstream will actually credit, from `mining.set_difficulty`. A
    /// share beating our own target is worth counting; a share beating this one
    /// is worth *sending*, and the two are different numbers on purpose.
    pub up_target: U256,
}

/// A job that has been handed out and may still have shares arriving for it.
#[derive(Clone)]
pub struct Issued {
    pub slot: u32,
    pub coin: String,
    pub algo: Algo,
    pub header: [u8; 80],
    pub target: U256,
    pub echo: proto::Echo,
    /// The difficulty this job went out at, in leading zero bits.
    ///
    /// Kept on the job rather than read off the coin, because with per
    /// -connection retargeting the coin no longer has one: two miners hold two
    /// jobs on the same coin at two difficulties, and crediting either from the
    /// coin's own figure would pay one of them for the other's work.
    pub bits: u32,
    /// Everything a `mining.submit` needs, kept from when the header was built.
    ///
    /// Stored rather than recomputed, for the reason `client::Template` gives
    /// about the same fields: formatting them a second time at submit is how a
    /// client sends a share for a header it never built. The miner's `echo`
    /// carries them too and is deliberately not trusted for this -- it comes
    /// back over the network and this did not.
    pub up: Option<UpstreamRef>,
}

/// The parts of an issued job that only matter if it goes back upstream.
#[derive(Clone)]
pub struct UpstreamRef {
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime_be: Vec<u8>,
    pub up_target: U256,
}

/// A share good enough to be worth sending upstream.
#[derive(Clone)]
pub struct Forward {
    pub slot: u32,
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime_be: Vec<u8>,
    pub nonce_be: Vec<u8>,
}

/// What happened to a submitted share. One enum so a caller cannot invent a
/// state, and so "stale" and "wrong" stay apart -- they mean completely
/// different things about a miner and folding them loses the difference.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    /// Good, and counted.
    Accepted,
    /// Correct arithmetic against a job we no longer hold. The miner did the
    /// work; the chain moved. Not the miner's fault and not counted against it.
    Stale,
    /// The hash does not meet the target it was issued. Either a bug at one
    /// end or a miner sending noise, and the pool cannot tell which.
    Bad,
    /// This exact nonce was already credited for this job.
    Duplicate,
    /// Parameters the algorithm refuses. The pool issued them, so this is the
    /// pool's own bug and says so rather than blaming the share.
    Unhashable,
}

impl Verdict {
    pub fn name(&self) -> &'static str {
        match self {
            Verdict::Accepted => "accepted",
            Verdict::Stale => "stale",
            Verdict::Bad => "bad",
            Verdict::Duplicate => "duplicate",
            Verdict::Unhashable => "unhashable",
        }
    }
}

/// One worker's record against one coin.
///
/// **`work` is the number that decides a payout, and `accepted` is not.** That
/// was true the moment difficulty stopped being fixed: a miner retargeted to 12
/// bits produces 256 times as many shares as one at 20 for the same effort, so
/// paying by share count would pay the slow device several hundred times per
/// unit of work what the fast one gets. The count stays because it is what an
/// operator reads to see whether a miner is alive.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Tally {
    /// Expected hashes behind the accepted shares: `2^bits` each, summed.
    ///
    /// Saturating rather than wrapping. At the 40-bit ceiling it takes about
    /// sixteen million shares to reach `u64::MAX`, which at one every ten
    /// seconds is five years -- but a wrap would silently reset somebody's
    /// entire record, and a clamp at least stops rising visibly.
    pub work: u64,
    pub accepted: u64,
    pub stale: u64,
    pub bad: u64,
    pub duplicate: u64,
}

/// One accepted share, kept only long enough to be paid for.
#[derive(Clone)]
pub struct Contribution {
    pub worker: String,
    /// `2^bits`, the same figure the cumulative tally credits. Denominated in
    /// work rather than in shares because VarDiff makes a share meaningless as
    /// a unit -- measured on one sweep, 394 shares at 10 bits against 5 at 16,
    /// for the same effort.
    pub work: u64,
}

/// The last N units of work on one coin, and what each worker did of it.
///
/// **This is the difference between a share log and something that can pay.**
/// The cumulative tally answers "how much has this worker ever done", which is
/// the wrong question at the moment a block is found: the coins in that block
/// were produced by the hashing that happened *recently*, and paying them out
/// against all-time work pays a miner who left last year out of a block they
/// had nothing to do with.
///
/// It is also the standard defence against pool-hopping, and the reason PPLNS
/// exists rather than proportional payment. Under proportional, a miner who
/// mines only early in a round -- when the expected shares to a block are
/// still low -- collects a larger slice per unit of work than one who stays,
/// and the difference comes out of everybody who stayed. Under PPLNS the
/// window keeps moving whether or not you are in it, so leaving costs you the
/// window and there is nothing to game.
///
/// **That last sentence is true under an assumption this pool does not get to
/// make, and the assumption is worth naming.** Rosenfeld's analysis of pooled
/// reward systems (arXiv:1112.4980, §3.3) proves the hopping-proofness of this
/// *simple* variant only while difficulty and block reward are constant. He
/// then drops it: a participant's contribution is fixed by the difficulty when
/// the share was submitted, while the reward it earns is set by the difficulty
/// when the block is found, so a hopper who knows a retarget is coming joins
/// before a decrease and leaves before an increase. Small chains retarget
/// often, and `payrate.py` selects for exactly those.
///
/// The hopping-proof form is **unit-PPLNS**, which stores two more things per
/// share: the value of `p` at submission -- share difficulty over *network*
/// difficulty -- and the block reward then in force. `Contribution.work` is
/// `2^bits`, which is the miner's own VarDiff target and says nothing about
/// the network. So this is not a scheme to swap in; it is blocked on the same
/// missing number `market.rs` is blocked on, which is the sharpest reason yet
/// to want it.
///
/// **And the window size is a dial, not an optimum.** Rosenfeld again: reward
/// variance goes as `pB^2/N` and mean time to being paid as `pN/2`, so their
/// **product is fixed at `(pB)^2/2` whatever N is**. There is no best window;
/// there is a choice between miners paid smoothly and miners paid soon, and
/// `--window` being an operator constant is right. What is missing is telling
/// the operator which end they picked, and that needs `p`, which needs the
/// network target. At the classic `N = D` the payouts per share are Poisson
/// with mean one, so **37% of shares are never paid at all** -- a fact a miner
/// who does not know it reads as the pool cheating.
pub struct Window {
    /// Oldest first. A `Vec` and not a `VecDeque` because the whole thing is
    /// walked to compute a payout anyway, and the front is popped a handful of
    /// times per share rather than in a loop.
    pub(crate) shares: Vec<Contribution>,
    /// Kept alongside rather than summed on demand: a payout report over a
    /// window of tens of thousands of shares would otherwise re-add every one
    /// of them on every call, and the pool publishes on a timer.
    total: u64,
}

impl Window {
    /// Add a share and drop whatever has fallen out of the back.
    ///
    /// The window holds **at least** `limit` work, and the smallest suffix
    /// that does. Dropping down to exactly `limit` would mean discarding part
    /// of a share, and a share is indivisible -- it is one miner's one
    /// discovery.
    pub fn push(&mut self, worker: &str, work: u64, limit: u64) {
        self.shares.push(Contribution { worker: String::from(worker), work });
        self.total = self.total.saturating_add(work);
        let mut drop_to = 0usize;
        let mut running = self.total;
        for c in &self.shares {
            if running.saturating_sub(c.work) < limit {
                break;
            }
            running -= c.work;
            drop_to += 1;
        }
        if drop_to > 0 {
            self.shares.drain(..drop_to);
            self.total = running;
        }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn len(&self) -> usize {
        self.shares.len()
    }

    /// Each worker's work in the window, largest first, ties by name.
    ///
    /// Sorted totally rather than by share alone, so two runs over one history
    /// print the same order -- the property `ledger` already has and for the
    /// same reason: a published document whose row order moved would look
    /// edited every time it was regenerated.
    pub fn shares_by_worker(&self) -> Vec<(String, u64)> {
        let mut by: BTreeMap<String, u64> = BTreeMap::new();
        for c in &self.shares {
            *by.entry(c.worker.clone()).or_insert(0) += c.work;
        }
        let mut v: Vec<(String, u64)> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }
}

impl Default for Window {
    fn default() -> Window {
        Window { shares: Vec::new(), total: 0 }
    }
}

pub struct Pool {
    pub coins: Vec<Coin>,
    /// Where to append a recomputable record of every accepted share, if
    /// anywhere.
    ///
    /// **The published ledger is a tally and a tally cannot be re-verified.**
    /// `design/pool.md` calls the share log "the only thing standing in for
    /// trust", and what it publishes is work/accepted/stale/bad per worker --
    /// numbers a miner can read and nobody can check. This is the other half:
    /// one line per accepted share carrying the header, the nonce and the
    /// target, so a third party recomputes the hash with code the operator
    /// never ran.
    ///
    /// `None` by default, because it is unbounded by construction -- a share a
    /// few seconds forever -- and an operator should turn that on deliberately.
    pub sharelog: Option<std::path::PathBuf>,
    /// Jobs still accepting shares, newest last. Bounded, because a miner that
    /// never submits would otherwise grow this forever -- and because a job old
    /// enough to fall off is a job whose shares are stale by definition.
    issued: Vec<(String, Issued)>,
    /// `(job, nonce)` already credited. The duplicate check is per job rather
    /// than global: two coins can legitimately produce the same nonce, and a
    /// global set would refuse the second as a duplicate of work it is not.
    seen: Vec<(String, u32)>,
    tallies: HashMap<(String, String), Tally>,
    /// Shares that beat upstream's target, waiting for the upstream thread.
    ///
    /// A queue rather than a call, because `Pool` holds no sockets -- the same
    /// split `server.rs` has, and the same shape the kernel's own `SHARES`
    /// queue uses between its hash loop and its socket task.
    forwards: Vec<Forward>,
    /// The payout window per coin. Per coin because a unit of btc work and a
    /// unit of zeny work are not the same thing and never become comparable --
    /// the same reason the tally is keyed by coin.
    windows: HashMap<String, Window>,
    /// How much work a window holds before its oldest shares fall out.
    ///
    /// **Set rather than derived, and that is stated rather than hidden.** The
    /// textbook figure is a multiple of the expected shares to a block, which
    /// needs the network difficulty -- available for an upstream coin and
    /// simply absent for a local one, which has no chain behind it at all. So
    /// this is the operator's number. `payout_report` prints the window
    /// against a coin's own network target where one is known, which is the
    /// figure needed to choose it well; inventing a default that looked
    /// derived would be worse than asking.
    /// **A `u64` reaches difficulty 4.3e9 and no further, which is a real
    /// ceiling and is not the one it looks like.** A block is `D * 2^32`
    /// expected hashes, `u64::MAX` is 1.845e19, so the largest window this
    /// type can express is one block only while `D < 4.295e9`.
    ///
    /// Against Bitcoin that fails by **21,432x** -- one block at the `nbits`
    /// this pool took off `solo.ckpool.org` is 3.95e23 hashes, so *no* value
    /// of `--window` would have been well chosen and PPLNS degenerates to
    /// paying the most recent shares whatever is passed. Against everything
    /// this pool is actually pointed at it is not close: Feathercoin fits
    /// 14,000 blocks of work, Bitzeny 210,000, Yenten 860,000.
    ///
    /// So the ceiling bites exactly one chain, and it is the chain `payrate.py`
    /// exists to steer away from -- the whole selection criterion is
    /// ASIC-free algorithms, which is to say small networks. Recorded rather
    /// than fixed, because widening this to `u128` touches the ledger format,
    /// its digest and every stored window, and buying that with a coin nobody
    /// will serve is the wrong trade. The day an upstream has a difficulty
    /// over 4.3e9, this is the line that explains the symptom.
    window_work: u64,
    next_job: u64,
    /// Where each worker's difficulty had got to, per `(worker, slot)`.
    ///
    /// **VarDiff cannot converge on a connection shorter than it.** Its idle
    /// path fires at `WINDOW_SECS`, sixty seconds, and its share path wants
    /// eight accepted shares -- so a miner reconnecting every forty-five
    /// seconds restarts at `start_bits` every time and never retargets at all.
    /// Found by a soak that deliberately churned: nineteen consecutive
    /// forty-five-second connections, zero shares accepted, while a
    /// thirty-minute one on the identical miner eased from 24 bits to 8 and
    /// was credited normally. Nothing about the miner differed but how long it
    /// stayed.
    ///
    /// A flaky link or a phone is exactly that miner, so this is not a corner
    /// case -- it is the device class the pool exists to serve.
    ///
    /// **A worker name is unauthenticated and it does not need to be.** The
    /// worst somebody can do by claiming another worker's name is start at a
    /// difficulty that does not suit them, and credit is denominated in work,
    /// so an easier start pays proportionally less per share and buys nothing.
    /// A harder one only hurts the claimant.
    ///
    /// Bounded, for the reason `tallies` is: the key is attacker-chosen.
    converged: HashMap<(String, usize), u32>,
    /// Where the counters go for a worker the tally has no room for. See
    /// `tally`. Never read; it exists so the refusal costs nothing and needs
    /// no second signature.
    discard: Tally,
    /// What `--window` was when the loaded ledger was written. See
    /// `load_ledger`; recorded rather than applied.
    loaded_window_work: Option<u64>,
    /// How many verdicts went to `discard`. Printed, because a cap that bites
    /// silently is a cap nobody knows about.
    untallied: u64,
}

/// How many `(worker, slot)` difficulties are remembered across reconnects.
///
/// A ceiling rather than an eviction policy: the value is an optimisation, so
/// losing one costs a miner one convergence and costs the pool nothing, while
/// the machinery to decide *which* to lose is state of its own on a path that
/// runs under load. Past this, a reconnecting worker starts where it always
/// did, which is the behaviour that existed before this map.
const MAX_REMEMBERED: usize = 4096;

/// How many `(worker, coin)` records the tally will hold.
///
/// **The key is a worker name and a worker name is unauthenticated**, so
/// without a bound this map is a stranger's write primitive against a machine
/// that is not ours. Measured: sixty seconds of ordinary flooding with a fresh
/// name per connection left 248 records, 104 KiB of resident memory and 31 KiB
/// of ledger JSON -- which is then re-serialised and rewritten every sixty
/// seconds, forever, and published. At four connections a second that is
/// fourteen thousand records an hour.
///
/// Not caught by any of the four connection limits, and it could not be: every
/// one of them held perfectly while this grew underneath them. It is the same
/// shape of failure `budget.rs` was written about -- each bound correct, and
/// the thing they were all bounding not bounded at all.
///
/// **A cap turns unbounded growth into a ceiling and does not make the attack
/// free, which is worth measuring rather than assuming.** The same six thousand
/// names from one connection: 641 KiB of published ledger uncapped against 204
/// capped, and 988 KiB of resident growth against 848. `retain` clears the map
/// when it fills and it refills, so the ceiling is 4,096 rows -- about half a
/// megabyte -- however many names arrive, where before it was linear in them.
/// Resident memory does not come back either; the allocator keeps the
/// high-water mark. What actually removed the cost was refusing a second
/// `hello` under a different name, in `server.rs`, which takes the rate from
/// nineteen a second per connection to one ever.
pub const MAX_TALLIES: usize = 4096;

/// A window of 2^32, which is the work in one difficulty-1 share.
///
/// Chosen so a pool nobody has configured still behaves like a pool rather
/// than like a payout of the last share, and small enough that a test can fill
/// it. It is a starting point and `--window` is how it stops being one.
pub const DEFAULT_WINDOW_WORK: u64 = 1u64 << 32;

/// Bumped when a change makes an older document unreadable rather than merely
/// different. Recording `work` did exactly that, and so did the payout window.
///
/// **Version 2 is still loaded, with its consequence said out loud.** A v2
/// file has tallies and no window, so the tallies restore and the window
/// starts empty -- which is a real loss (everybody's recent contribution goes
/// to zero) and is much better than refusing to start. Refusing was the old
/// behaviour for v1 and was right there: a v1 row has no `work` at all, so
/// loading it would credit every past share as zero effort and rewrite the
/// record the file exists to preserve. Missing data and *wrong* data are
/// different, and only one of them is worth refusing over.
const LEDGER_VERSION: u32 = 3;

/// How many issued jobs to remember. Sixty-four is four coins' worth of a
/// couple of minutes at a thirty-second job cadence, which is comfortably
/// longer than any honest share takes to arrive.
const KEEP_JOBS: usize = 64;
/// Nonces remembered for the duplicate check. Larger than `KEEP_JOBS` because
/// one job can yield many shares, and a forgotten nonce is credited twice.
const KEEP_NONCES: usize = 4096;

impl Pool {
    pub fn new(coins: Vec<Coin>) -> Pool {
        Pool {
            coins,
            sharelog: None,
            issued: Vec::new(),
            seen: Vec::new(),
            tallies: HashMap::new(),
            forwards: Vec::new(),
            windows: HashMap::new(),
            window_work: DEFAULT_WINDOW_WORK,
            next_job: 1,
            converged: HashMap::new(),
            discard: Tally::default(),
            loaded_window_work: None,
            untallied: 0,
        }
    }

    /// The difficulty a worker should resume at on this slot, falling back to
    /// the coin's configured start when it has never converged here.
    pub fn resume_bits(&self, worker: &str, slot: usize) -> u32 {
        self.converged
            .get(&(String::from(worker), slot))
            .copied()
            .unwrap_or_else(|| self.start_bits(slot))
    }

    /// Record where a worker's difficulty settled, so its next connection does
    /// not start over. Silently declines past `MAX_REMEMBERED`.
    pub fn remember_bits(&mut self, worker: &str, slot: usize, bits: u32) {
        let key = (String::from(worker), slot);
        if self.converged.contains_key(&key) || self.converged.len() < MAX_REMEMBERED {
            self.converged.insert(key, bits);
        }
    }

    /// How many difficulties are being remembered. For the operator report and
    /// for the claim that the map is bounded.
    pub fn remembered(&self) -> usize {
        self.converged.len()
    }

    /// How much work the payout window holds. Zero is refused rather than
    /// stored: a window of nothing pays only the miner who found the last
    /// share, which is not a payout scheme but a lottery with one ticket.
    pub fn set_window(&mut self, work: u64) -> bool {
        if work == 0 {
            return false;
        }
        self.window_work = work;
        true
    }

    pub fn window_work(&self) -> u64 {
        self.window_work
    }

    /// What each worker is owed of one coin, as a fraction of the window.
    ///
    /// Answers `None` for a coin nobody has mined, which is different from a
    /// coin everybody has mined equally and must not print as zeroes.
    pub fn payouts(&self, coin: &str) -> Option<Vec<(String, u64, f64)>> {
        let w = self.windows.get(coin)?;
        let total = w.total();
        if total == 0 {
            return None;
        }
        Some(
            w.shares_by_worker()
                .into_iter()
                .map(|(name, work)| (name, work, work as f64 / total as f64))
                .collect(),
        )
    }

    /// Which algorithm a job wants, without validating anything.
    ///
    /// The budget needs this *before* the share is checked, because what a
    /// check costs is a property of the algorithm and the decision is whether
    /// to pay it at all.
    pub fn algo_of_job(&self, job: &str) -> Option<String> {
        self.issued
            .iter()
            .find(|(id, _)| id == job)
            .map(|(_, j)| String::from(j.algo.name()))
    }

    /// Every configured coin's label, in slot order.
    pub fn coin_labels(&self) -> Vec<String> {
        self.coins.iter().map(|c| c.label.clone()).collect()
    }

    /// The window for a coin, for the report and for the claims.
    pub fn window(&self, coin: &str) -> Option<&Window> {
        self.windows.get(coin)
    }

    /// Build the next job for a coin.
    ///
    /// The header is assembled here and the miner never sees a coinbase, which
    /// is the whole shape of the protocol and its whole trust cost. See
    /// `design/pool.md`.
    /// Build the next job for a coin, at the difficulty this caller wants.
    ///
    /// **The difficulty is an argument and not a property of the coin**, which
    /// is what makes per-connection retargeting possible at all: every call
    /// takes a fresh job id, so two miners on one coin simply hold two
    /// `Issued` records with two targets and nothing has to be shared between
    /// them.
    pub fn make_job(&mut self, slot: u32, bits: u32) -> Option<proto::Job> {
        let coin = self.coins.get(slot as usize)?;
        let label = coin.label.clone();
        let algo = coin.algo.clone();
        let target = target_with_leading_zeros(bits);
        let work = coin.work.clone();
        let upstream = matches!(coin.source, Source::Upstream { .. });

        // An upstream coin with no work yet yields no job. Building one anyway
        // would mean inventing a header, and a miner would then spend real time
        // on a search that can never pay -- which from `mine coins` looks
        // exactly like a coin that is working.
        if upstream && work.is_none() {
            return None;
        }

        let id = self.next_job;
        self.next_job += 1;
        let job = format!("{id:08x}");

        let mut proof = None;
        let (header, up) = match &work {
            Some(w) => {
                // The extranonce2 is what makes two jobs from one `notify` into
                // two different searches, so it advances per job and not per
                // notify.
                let c = self.coins.get_mut(slot as usize)?;
                c.e2 = c.e2.wrapping_add(1);
                let counter = c.e2;

                let mut e2 = Vec::with_capacity(w.extranonce2_size);
                // Big-endian, so a hex dump reads in order. Which encoding is
                // used does not matter for validity; what matters is that the
                // same bytes reach the coinbase and the submit, which is why
                // they are stored below rather than formatted twice.
                for i in (0..w.extranonce2_size).rev() {
                    e2.push((counter >> (8 * (i % 8))) as u8);
                }

                let coinbase = header::coinbase(&w.coinb1, &w.extranonce1, &e2, &w.coinb2);
                let root = header::merkle_root(&coinbase, &w.branch);
                // The miner is shown the working. `extranonce1 || extranonce2`
                // goes as one field because the split is Stratum's answer to a
                // problem the miner does not have here: the pool varies the
                // second half, so where the boundary falls is nothing a miner
                // can use.
                let mut spliced = w.extranonce1.clone();
                spliced.extend_from_slice(&e2);
                if LIE.load(core::sync::atomic::Ordering::Relaxed) {
                    // One byte. The point is that a miner must refuse a proof
                    // that is *almost* right, not only one that is obviously
                    // malformed -- a check that only caught garbage would pass
                    // on every interesting lie.
                    spliced[0] ^= 0x01;
                }
                proof = Some(proto::Proof {
                    coinb1: w.coinb1.clone(),
                    extranonce: spliced,
                    coinb2: w.coinb2.clone(),
                    branch: w.branch.clone(),
                });
                let h = header::assemble(w.version, &w.prev_wire, &root, w.ntime, w.nbits, 0);
                let (ntime_be, _) = header::submit_hex(&h);
                (
                    h,
                    Some(UpstreamRef {
                        job_id: w.job_id.clone(),
                        extranonce2: e2,
                        ntime_be,
                        up_target: w.up_target,
                    }),
                )
            }
            None => {
                // A local coin has no chain, so the header is this pool's own:
                // a version, an id where the previous hash goes, and the clock.
                // Deliberately *not* a fixed fixture -- an unchanging header
                // means every job is the same search, and a nonce found once
                // would be a share forever.
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);
                let mut h = [0u8; 80];
                h[0..4].copy_from_slice(&1u32.to_le_bytes());
                h[36..44].copy_from_slice(&id.to_le_bytes());
                h[68..72].copy_from_slice(&now.to_le_bytes());
                h[72..76].copy_from_slice(&0x1d00_ffffu32.to_le_bytes());
                (h, None)
            }
        };

        self.issued.push((
            job.clone(),
            Issued {
                slot,
                coin: label.clone(),
                algo: algo.clone(),
                header,
                target,
                echo: Vec::new(),
                bits,
                up,
            },
        ));
        if self.issued.len() > KEEP_JOBS {
            self.issued.remove(0);
        }

        Some(proto::Job {
            slot,
            coin: label,
            job,
            algo,
            header,
            target,
            echo: Vec::new(),
            clean: true,
            // Absent for a local coin, deliberately. There is no chain behind
            // one, so its "coinbase" would be a fabrication -- and a proof that
            // verifies against an invented header is worse than none, because
            // it looks like evidence.
            proof,
        })
    }

    /// Install what upstream just sent.
    pub fn set_work(&mut self, slot: usize, w: Work) -> bool {
        let Some(coin) = self.coins.get_mut(slot) else {
            return false;
        };
        // **The network target arrives in every `mining.notify` and was being
        // thrown away.** `Coin::network_target` has existed since the struct
        // did, `U256::from_nbits` has existed and been claimed since `u256`
        // did, and nothing ever joined them -- so three separate things were
        // blocked on a number that was on the wire the whole time: expected
        // value per coin, the payout window having units, and telling simple
        // PPLNS from the hopping-proof variant.
        //
        // `None` on a malformed `nbits` rather than a clamp, which is
        // `from_nbits`'s own rule: a clamped target is a legal-looking number
        // that is not the one the network asked for.
        coin.network_target = U256::from_nbits(w.nbits);
        coin.work = Some(w);
        true
    }

    /// The payout window as a fraction of one block's expected work, where a
    /// chain is known.
    ///
    /// **This is the unit `--window` never had.** Rosenfeld's analysis gives
    /// reward variance as `pB^2/N` and mean time to payment as `pN/2`, whose
    /// product is fixed whatever `N` is -- so the window is not an optimisation
    /// with a right answer, it is a dial between paying smoothly and paying
    /// soon. An operator cannot pick a point on a dial with no markings, and
    /// the marking needs the network target.
    ///
    /// A block is `2^256 / target` expected hashes and the window is denominated
    /// in expected hashes already, so the ratio is the whole of it.
    pub fn window_in_blocks(&self, coin: &str) -> Option<f64> {
        let c = self.coins.iter().find(|c| c.label == coin)?;
        let t = c.network_target.as_ref()?;
        let per_block = approx_hashes_per_block(t)?;
        Some(self.window_work as f64 / per_block)
    }

    /// Which coin an issued job belongs to.
    ///
    /// Asked of the pool rather than tracked a second time in the connection,
    /// because the pool is already the one thing that knows -- and two records
    /// of which job is which coin is the arrangement that eventually
    /// disagrees.
    pub fn slot_of_job(&self, job: &str) -> Option<usize> {
        self.issued
            .iter()
            .find(|(id, _)| id == job)
            .map(|(_, j)| j.slot as usize)
    }

    /// Where a new connection should start on this coin.
    pub fn start_bits(&self, slot: usize) -> u32 {
        self.coins.get(slot).map(|c| c.share_bits).unwrap_or(20)
    }

    /// Whether a coin has upstream work in hand.
    pub fn has_work(&self, slot: usize) -> bool {
        self.coins
            .get(slot)
            .map(|c| c.work.is_some())
            .unwrap_or(false)
    }

    /// Take the shares that beat upstream's target.
    ///
    /// Drains, so a caller that fails to send them loses them -- which is
    /// correct rather than careless: a share is only worth anything on the
    /// connection whose extranonce1 it was found under, so holding one for a
    /// reconnection would be keeping something already worthless.
    pub fn take_forwards(&mut self) -> Vec<Forward> {
        core::mem::take(&mut self.forwards)
    }

    /// Validate a submitted share by computing the hash the miner computed.
    ///
    /// This is the one thing the pool cannot delegate and the reason it shares
    /// the kernel's source. Everything else here is bookkeeping.
    pub fn submit(&mut self, worker: &str, sh: &proto::Share) -> Verdict {
        let Some((_, job)) = self.issued.iter().find(|(id, _)| *id == sh.job) else {
            // Not counted against the worker. The arithmetic may have been
            // perfect; the job simply aged out.
            self.tally(worker, "?").stale += 1;
            return Verdict::Stale;
        };
        let job = job.clone();

        if self.seen.iter().any(|(j, n)| *j == sh.job && *n == sh.nonce) {
            self.tally(worker, &job.coin).duplicate += 1;
            return Verdict::Duplicate;
        }

        let Some(mut hasher) = Hasher::new(&job.algo, &job.header) else {
            // The pool issued these parameters, so this is the pool's bug.
            // Not tallied against the worker at all.
            return Verdict::Unhashable;
        };
        let digest = hasher.hash(&job.header, sh.nonce);

        if !below_target(&digest, &job.target) {
            self.tally(worker, &job.coin).bad += 1;
            return Verdict::Bad;
        }

        self.seen.push((sh.job.clone(), sh.nonce));
        if self.seen.len() > KEEP_NONCES {
            self.seen.remove(0);
        }

        // Appended here rather than by the caller, because this is the only
        // place that holds all four things a verifier needs at once: the
        // assembled header, the nonce, the target this pool set, and the
        // algorithm. `submit`'s own job table is gone thirty seconds later.
        //
        // A failed write is ignored on purpose. The share is already valid and
        // already credited; refusing it because a log could not be appended
        // would make a full disk into lost work, and the tally -- which is what
        // pays people -- is written elsewhere and does report its failures.
        if let Some(path) = &self.sharelog {
            use std::io::Write;
            let rec = crate::record::Record {
                algo: job.algo.clone(),
                header: job.header,
                nonce: sh.nonce,
                target: job.target,
                worker: String::from(worker),
            };
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{}", rec.render());
            }
        }

        // Two targets, and the difference between them is the whole of being a
        // proxy. Ours decides what a miner is *credited* for and is set low
        // enough that a laptop reports in regularly; upstream's decides what is
        // worth *sending*, and most accepted shares do not meet it. One number
        // for both would either flood upstream with work it rejects or leave a
        // miner silent for hours.
        if let Some(up) = &job.up {
            if below_target(&digest, &up.up_target) {
                self.forwards.push(Forward {
                    slot: job.slot,
                    job_id: up.job_id.clone(),
                    extranonce2: up.extranonce2.clone(),
                    ntime_be: up.ntime_be.clone(),
                    nonce_be: sh.nonce.to_be_bytes().to_vec(),
                });
            }
        }

        let credit = if job.bits >= 64 { u64::MAX } else { 1u64 << job.bits };
        // Both records, and they answer different questions. The tally is
        // all-time and is what a miner checks their own history against; the
        // window is recent and is what a payout comes from. Keeping one and
        // deriving the other is not available -- a cumulative total cannot be
        // walked backwards into a window, and a window has forgotten the past
        // by construction.
        let limit = self.window_work;
        self.windows
            .entry(job.coin.clone())
            .or_default()
            .push(worker, credit, limit);
        let t = self.tally(worker, &job.coin);
        t.accepted += 1;
        t.work = t.work.saturating_add(credit);
        Verdict::Accepted
    }

    /// The record for one worker on one coin, creating it if there is room.
    ///
    /// **Room is made by dropping records that are owed nothing, and never by
    /// dropping one that is.** A record with `work > 0` was paid for in hashes
    /// that actually met a target, so it cannot be manufactured cheaply and it
    /// is the only kind a payout ever reads. A record with zero work is a
    /// stranger's bad and stale shares, which cost nothing to produce in any
    /// quantity. Evicting the second class to protect the first turns the cap
    /// from a denial of service into a proof-of-work admission rule: a flood
    /// of invented names can fill this map and can never crowd out a miner
    /// who has done something.
    ///
    /// A refused verdict goes to `discard` rather than being an `Option` the
    /// four call sites have to unwrap. It is counted, and the count is printed,
    /// because the operator's question when a worker is missing from the ledger
    /// is exactly whether this happened.
    fn tally(&mut self, worker: &str, coin: &str) -> &mut Tally {
        let key = (String::from(worker), String::from(coin));
        if !self.tallies.contains_key(&key) && self.tallies.len() >= MAX_TALLIES {
            self.tallies.retain(|_, t| t.work > 0);
            if self.tallies.len() >= MAX_TALLIES {
                self.untallied += 1;
                self.discard = Tally::default();
                return &mut self.discard;
            }
        }
        self.tallies.entry(key).or_default()
    }

    /// How many verdicts were dropped because the tally was full of records
    /// that are owed something. Nonzero means the cap is genuinely binding
    /// rather than merely present.
    pub fn untallied(&self) -> u64 {
        self.untallied
    }

    /// How many `(worker, coin)` records are held. For the operator report and
    /// for the claim that the map is bounded.
    pub fn tally_len(&self) -> usize {
        self.tallies.len()
    }

    /// The share log, which is the whole product of a non-custodial pool and
    /// the only thing standing in for trust. Sorted, so two runs of the same
    /// history print the same report rather than a `HashMap`'s order.
    pub fn ledger(&self) -> Vec<(String, String, Tally)> {
        let mut v: Vec<(String, String, Tally)> = self
            .tallies
            .iter()
            .map(|((w, c), t)| (w.clone(), c.clone(), t.clone()))
            .collect();
        v.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        v
    }

    pub fn slots(&self) -> u32 {
        self.coins.len() as u32
    }

    /// Read a ledger back, so a restart does not begin at zero.
    ///
    /// Only the *tallies* are restored. The coins come from the command line,
    /// which is authoritative: a file that disagreed about which algorithm a
    /// label means would otherwise silently change what the pool serves, and
    /// the operator would be reading their own config to find out why.
    ///
    /// Parsed with `crate::json`, which is the kernel's parser reading this
    /// program's own output -- one parser at both ends, for the reason the
    /// protocol gives.
    ///
    /// **What the digest can and cannot catch.** It is over the rows, so a
    /// truncated or corrupted file is refused rather than half-loaded. It is
    /// not a signature and proves nothing about *who* wrote the file: anybody
    /// who can edit it can recompute the digest. That is acceptable because
    /// this is the operator's own record on the operator's own disk, and it is
    /// written down because a digest is easy to mistake for more than it is.
    pub fn load_ledger(&mut self, text: &str) -> Result<usize, String> {
        let doc = crate::json::Json::parse(text.trim()).ok_or("not JSON")?;
        // Absent means version 1, which is what the pool wrote before it
        // recorded work. Refused rather than half-read: those rows have no
        // `work` field, so loading them would credit every past share as zero
        // effort and quietly rewrite the record the file exists to preserve.
        let v = doc.get("v").and_then(|x| x.as_i64()).unwrap_or(1);
        if v != LEDGER_VERSION as i64 && v != 2 {
            return Err(format!(
                "written by an older pool (format {v}, this one writes {LEDGER_VERSION})"
            ));
        }
        let rows = match doc.get("shares") {
            Some(crate::json::Json::Arr(items)) => items,
            _ => return Err(String::from("no shares array")),
        };

        let mut restored: Vec<((String, String), Tally)> = Vec::new();
        let mut canon = String::new();
        for r in rows.iter() {
            let worker = r.get("worker").and_then(|x| x.as_str()).ok_or("row has no worker")?;
            let coin = r.get("coin").and_then(|x| x.as_str()).ok_or("row has no coin")?;
            let n = |k: &str| -> Result<u64, String> {
                match r.get(k).and_then(|x| x.as_i64()) {
                    // Negative is refused rather than clamped. A count below
                    // zero is a file that has been edited or corrupted, and
                    // saturating it to zero would load the damage silently.
                    Some(v) if v >= 0 => Ok(v as u64),
                    Some(v) => Err(format!("{k} is {v}")),
                    None => Err(format!("row has no {k}")),
                }
            };
            let t = Tally {
                work: n("work")?,
                accepted: n("accepted")?,
                stale: n("stale")?,
                bad: n("bad")?,
                duplicate: n("duplicate")?,
            };
            canon.push_str(&format!(
                "{worker}\t{coin}\t{}\t{}\t{}\t{}\t{}\n",
                t.work, t.accepted, t.stale, t.bad, t.duplicate
            ));
            restored.push(((String::from(worker), String::from(coin)), t));
        }

        // The window, share by share and in order, because the order is what
        // decides who falls out next. Restoring an aggregate instead would put
        // back the right payout today and evict the wrong worker tomorrow.
        let mut windows: HashMap<String, Window> = HashMap::new();
        if let Some(crate::json::Json::Arr(cw)) = doc.get("windows") {
            for entry in cw.iter() {
                let coin = entry.get("coin").and_then(|x| x.as_str()).ok_or("window has no coin")?;
                let Some(crate::json::Json::Arr(items)) = entry.get("shares") else {
                    return Err(format!("window for {coin} has no shares array"));
                };
                let mut w = Window::default();
                for it in items.iter() {
                    let name = it.get("worker").and_then(|x| x.as_str())
                        .ok_or("a window share has no worker")?;
                    let work = match it.get("work").and_then(|x| x.as_i64()) {
                        Some(v) if v >= 0 => v as u64,
                        Some(v) => return Err(format!("a window share has work {v}")),
                        None => return Err(String::from("a window share has no work")),
                    };
                    canon.push_str(&format!("w\t{coin}\t{name}\t{work}\n"));
                    // `u64::MAX` as the limit, so restoring never evicts: the
                    // file already holds a window that was bounded when it was
                    // written, and re-applying the bound here would trim it
                    // again against a limit the operator may since have raised.
                    w.push(name, work, u64::MAX);
                }
                windows.insert(String::from(coin), w);
            }
        }

        // The rows are canonicalised the same way `ledger_json` does, so this
        // check is against the writer rather than against a second idea of what
        // the document says.
        let want = doc.get("digest").and_then(|x| x.as_str()).unwrap_or("");
        let got = crate::mine::stratum::hex(&crate::store::sha256::hash(canon.as_bytes()));
        if want != got {
            return Err(format!("digest {want} does not match the rows ({got})"));
        }

        let n = restored.len();
        for (k, v) in restored {
            self.tallies.insert(k, v);
        }
        self.windows = windows;
        // **Recorded and not applied.** The file's window is what the restored
        // shares were accumulated under; the command line is what the operator
        // is asking for now, and the command line wins for the reason the coin
        // list does -- a file that silently overrode it would change what the
        // pool serves and leave the operator reading their own config to find
        // out why.
        //
        // But a difference is not nothing. PPLNS has no round boundaries, so
        // there is no safe moment to change the window: Rosenfeld's fix is to
        // rescale every stored share by the ratio, and nothing here does that,
        // so raising or lowering `--window` across a restart moves every
        // outstanding miner's pending reward without saying so. Kept so the
        // caller can say so.
        self.loaded_window_work = doc.get("window_work").and_then(|x| x.as_i64()).map(|v| v as u64);
        Ok(n)
    }

    /// The window the restored shares were accumulated under, when the file
    /// recorded one. `None` for a format that predates the field.
    pub fn loaded_window_work(&self) -> Option<u64> {
        self.loaded_window_work
    }

    /// The share log as a publishable document.
    ///
    /// This is the whole product of a non-custodial pool. Layer 1 never holds
    /// a miner's coins, so there is nothing to audit by looking at a wallet;
    /// what a miner has instead is this record and whatever can be checked
    /// against it. `design/pool.md` says that out loud and it constrains the
    /// format rather than only the paperwork.
    ///
    /// **Canonical, so the digest means something.** Rows are sorted and every
    /// field is written in one fixed order, so two runs over the same history
    /// produce identical bytes -- a document whose hash depended on a
    /// `HashMap`'s iteration order would have a different digest every time it
    /// was regenerated, which is indistinguishable from a record that changed.
    ///
    /// The digest is over the rows and not over the whole file, because
    /// `generated_at` moves on every write and would otherwise make an
    /// unchanged log look edited.
    ///
    /// **This is not a Merkle root and does not claim to be.** A distributor
    /// needs a tree whose leaves are per-address payouts and whose proofs a
    /// contract can verify; this is a flat digest over a tally. It exists so
    /// the published record is fixed to a value now, and so the day the tree
    /// is built there is something to check it against.
    pub fn ledger_json(&self, epoch: u64, generated_at: u64) -> String {
        let rows = self.ledger();

        let mut canon = String::new();
        for (w, c, t) in &rows {
            // Tab-separated and newline-terminated rather than JSON, because
            // the digest must not depend on how a JSON writer spaces or
            // escapes. Two encoders that agree about a document can still
            // disagree about its bytes.
            //
            // `work` leads, because it is the field a payout comes from.
            canon.push_str(&format!(
                "{w}\t{c}\t{}\t{}\t{}\t{}\t{}\n",
                t.work, t.accepted, t.stale, t.bad, t.duplicate
            ));
        }
        // The window is inside the digest too. A payout comes from it, so a
        // document that fixed only the tallies would leave the number anybody
        // is actually paid on unattested -- which is the wrong half to leave
        // open on the one record standing in for trust.
        let mut coins_sorted: Vec<&String> = self.windows.keys().collect();
        coins_sorted.sort();
        for c in &coins_sorted {
            if let Some(w) = self.windows.get(*c) {
                for sh in &w.shares {
                    canon.push_str(&format!("w\t{c}\t{}\t{}\n", sh.worker, sh.work));
                }
            }
        }
        let digest = crate::store::sha256::hash(canon.as_bytes());

        let mut s = String::from("{\n");
        // The format's own version, so a document written by an older pool is
        // refused by name rather than as a digest mismatch. Those are different
        // problems -- one is a format change and one is a damaged file -- and a
        // single confusing error for both is how somebody concludes their
        // ledger was corrupted when it was merely old.
        s.push_str(&format!("  \"v\": {LEDGER_VERSION},\n"));
        s.push_str(&format!("  \"epoch\": {epoch},\n"));
        s.push_str(&format!("  \"generated_at\": {generated_at},\n"));
        s.push_str(&format!(
            "  \"digest\": \"{}\",\n",
            crate::mine::stratum::hex(&digest)
        ));
        s.push_str("  \"coins\": [\n");
        for (i, c) in self.coins.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"slot\": {i}, \"label\": \"{}\", \"algo\": \"{}\", \"source\": \"{}\"}}{}\n",
                c.label,
                c.algo.detail(),
                c.source.name(),
                if i + 1 == self.coins.len() { "" } else { "," }
            ));
        }
        s.push_str(&format!("  ],\n  \"window_work\": {},\n", self.window_work));
        // Every share in the window, in order, and the payout each implies.
        // **Verbose on purpose.** A PPLNS payout that cannot be checked share
        // by share is exactly the opacity a non-custodial pool has nothing else
        // to offer against -- there is no wallet to audit, so the arithmetic
        // has to be reproducible from the published document alone.
        s.push_str("  \"windows\": [\n");
        for (i, c) in coins_sorted.iter().enumerate() {
            let Some(w) = self.windows.get(*c) else { continue };
            s.push_str(&format!(
                "    {{\"coin\": \"{c}\", \"total\": {}, \"shares\": [\n",
                w.total()
            ));
            for (j, sh) in w.shares.iter().enumerate() {
                s.push_str(&format!(
                    "      {{\"worker\": \"{}\", \"work\": {}}}{}\n",
                    sh.worker,
                    sh.work,
                    if j + 1 == w.shares.len() { "" } else { "," }
                ));
            }
            s.push_str("    ], \"payout\": [\n");
            let by = w.shares_by_worker();
            for (j, (name, work)) in by.iter().enumerate() {
                s.push_str(&format!(
                    "      {{\"worker\": \"{name}\", \"work\": {work}, \"share\": {:.9}}}{}\n",
                    *work as f64 / w.total().max(1) as f64,
                    if j + 1 == by.len() { "" } else { "," }
                ));
            }
            s.push_str(&format!(
                "    ]}}{}\n",
                if i + 1 == coins_sorted.len() { "" } else { "," }
            ));
        }
        s.push_str("  ],\n  \"shares\": [\n");
        for (i, (w, c, t)) in rows.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"worker\": \"{w}\", \"coin\": \"{c}\", \"work\": {}, \"accepted\": {}, \"stale\": {}, \"bad\": {}, \"duplicate\": {}}}{}\n",
                t.work,
                t.accepted,
                t.stale,
                t.bad,
                t.duplicate,
                if i + 1 == rows.len() { "" } else { "," }
            ));
        }
        s.push_str("  ]\n}\n");
        s
    }
}

/// Set by `--bad-proof`. Off unless an operator deliberately asked for it.
///
/// A global rather than a field on `Pool`, because it is a testing switch and
/// not a property of a coin: threading it through every constructor would put a
/// "tell lies" argument in the signature of ordinary code.
static LIE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn lie_about_proofs() {
    LIE.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// A target easy enough that a laptop finds shares in seconds.
///
/// `leading` is how many leading zero *bits* a digest must have. Expressed in
/// bits rather than as a difficulty because a difficulty is a float and the
/// conversion is exactly where `stratum::decimal` records that `Json::as_i64`
/// reads `0.001` as zero.
/// `2^256 / target`, as a float, for reporting only.
///
/// Float on purpose and nowhere near a validation: what this feeds is a line an
/// operator reads, and the answers span forty orders of magnitude, so what is
/// wanted is the exponent rather than the units digit. Every place the number
/// actually decides something -- `below_target`, `target_for` -- stays in
/// `U256` for the reason `u256` exists.
fn approx_hashes_per_block(t: &U256) -> Option<f64> {
    let b = t.to_be_bytes();
    // Where the target's leading one sits, and the value of the bytes from
    // there. 2^256 / t is then 2^(256 - pos) / (mantissa scaled to that).
    let first = b.iter().position(|x| *x != 0)?;
    let mut mant = 0f64;
    for i in 0..8usize.min(32 - first) {
        mant = mant * 256.0 + b[first + i] as f64;
    }
    if mant == 0.0 {
        return None;
    }
    // t = mant * 256^(32 - first - taken), so 2^256 / t = 2^256 / mant / 256^k.
    let taken = 8usize.min(32 - first);
    let k = (32 - first - taken) as i32;
    let v = 2f64.powi(256) / mant / 256f64.powi(k);
    if v.is_finite() && v > 0.0 {
        Some(v)
    } else {
        None
    }
}

pub fn target_with_leading_zeros(leading: u32) -> U256 {
    let mut b = [0xffu8; 32];
    let full = (leading / 8) as usize;
    for x in b.iter_mut().take(full.min(32)) {
        *x = 0;
    }
    if full < 32 {
        b[full] = 0xffu8 >> (leading % 8);
    }
    U256::from_be_bytes(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_pool() -> Pool {
        Pool::new(vec![Coin {
            label: String::from("test"),
            asset: String::from("test"),
            algo: Algo::Sha256d,
            // Eight bits, so a share turns up within a few hundred nonces.
            share_bits: 8,
            share_target: target_with_leading_zeros(8),
            network_target: None,
            source: Source::Local,
            work: None,
            e2: 0,
        }])
    }

    /// The check the whole pool rests on: a share is credited only when the
    /// pool can reproduce the hash the miner claims to have found.
    #[test]
    fn a_real_share_is_accepted_and_a_forged_one_is_not() {
        let mut p = a_pool();
        let job = p.make_job(0, 8).unwrap();
        let mut h = Hasher::new(&job.algo, &job.header).unwrap();

        let mut good = None;
        for n in 0..200_000u32 {
            if below_target(&h.hash(&job.header, n), &job.target) {
                good = Some(n);
                break;
            }
        }
        let n = good.expect("no share inside 200k nonces at 8 leading bits");

        assert_eq!(
            p.submit("w1", &proto::Share { job: job.job.clone(), nonce: n, echo: vec![] }),
            Verdict::Accepted
        );
        // The same nonce again is a duplicate rather than a second credit.
        assert_eq!(
            p.submit("w1", &proto::Share { job: job.job.clone(), nonce: n, echo: vec![] }),
            Verdict::Duplicate
        );
        // A nonce that does not meet the target is refused however confidently
        // it is sent.
        //
        // **The bad nonce is found rather than assumed, and that is a fix.**
        // This was `n + 1` under a comment saying it is "overwhelmingly not a
        // solution", which at an eight-bit target is true 255 times in 256 --
        // so the suite failed about once every few hundred runs, on correct
        // code, and did exactly that during this session. It is the same shape
        // as the flake already fixed in `credit_follows_work_and_not_share_count`
        // and it gets the same treatment: make the case deterministic rather
        // than make the bound wider, because a test that is right 99.6% of the
        // time teaches everybody to rerun the suite instead of reading it.
        let bad = (0..200_000u32)
            .find(|n| !below_target(&h.hash(&job.header, *n), &job.target))
            .expect("some nonce misses an eight-bit target");
        assert_eq!(
            p.submit("w1", &proto::Share { job: job.job.clone(), nonce: bad, echo: vec![] }),
            Verdict::Bad
        );
        // And a job the pool never issued is stale, not bad -- the distinction
        // decides whether a miner is misbehaving or merely late.
        assert_eq!(
            p.submit("w1", &proto::Share { job: String::from("ffffffff"), nonce: n, echo: vec![] }),
            Verdict::Stale
        );

        let led = p.ledger();
        let t = &led.iter().find(|(_, c, _)| c == "test").unwrap().2;
        assert_eq!((t.accepted, t.duplicate, t.bad), (1, 1, 1));
    }

    /// A share found under one algorithm must not be credited under another.
    /// This is what would break if `proto` dropped the algorithm field, and it
    /// would break silently: every share simply stops being accepted.
    #[test]
    fn a_share_is_validated_under_its_own_algorithm() {
        let mut p = Pool::new(vec![
            Coin {
                label: String::from("a"),
            asset: String::from("a"),
                algo: Algo::Sha256d,
                share_bits: 8,
                share_target: target_with_leading_zeros(8),
                network_target: None,
                source: Source::Local,
                work: None,
                e2: 0,
            },
            Coin {
                label: String::from("b"),
            asset: String::from("b"),
                algo: Algo::Blake2s,
                share_bits: 8,
                share_target: target_with_leading_zeros(8),
                network_target: None,
                source: Source::Local,
                work: None,
                e2: 0,
            },
        ]);
        let ja = p.make_job(0, 8).unwrap();
        let mut ha = Hasher::new(&ja.algo, &ja.header).unwrap();
        let n = (0..200_000u32)
            .find(|n| below_target(&ha.hash(&ja.header, *n), &ja.target))
            .expect("no sha256d share found");

        // Under blake2s the same header and nonce give a different digest, so
        // this nonce is almost certainly not a solution there.
        let jb = p.make_job(1, 8).unwrap();
        let mut hb = Hasher::new(&jb.algo, &jb.header).unwrap();
        assert_ne!(ha.hash(&ja.header, n), hb.hash(&jb.header, n));

        assert_eq!(
            p.submit("w", &proto::Share { job: ja.job, nonce: n, echo: vec![] }),
            Verdict::Accepted
        );
    }

    #[test]
    fn the_target_helper_means_what_it_says() {
        let t = target_with_leading_zeros(8);
        let b = t.to_be_bytes();
        assert_eq!(b[0], 0x00);
        assert_eq!(b[1], 0xff);
        // Zero leading bits is every digest, which is what a pool with the
        // difficulty turned all the way down should mean rather than an error.
        assert_eq!(target_with_leading_zeros(0).to_be_bytes()[0], 0xff);
    }

    /// The digest is a function of the record and of nothing else.
    ///
    /// The property that makes a published log worth publishing: regenerating
    /// it must not change it. A digest that moved with the clock, or with a
    /// map's iteration order, would make every routine republish look like an
    /// edit -- and a record nobody can tell has changed is not evidence.
    #[test]
    fn the_ledger_digest_depends_on_the_record_and_not_the_run() {
        let mut p = a_pool();
        let job = p.make_job(0, 8).unwrap();
        let mut h = Hasher::new(&job.algo, &job.header).unwrap();
        let n = (0..200_000u32)
            .find(|n| below_target(&h.hash(&job.header, *n), &job.target))
            .expect("no share found");
        p.submit("w1", &proto::Share { job: job.job, nonce: n, echo: vec![] });

        let a = p.ledger_json(1, 1_000);
        let b = p.ledger_json(1, 9_999);
        let da = a.lines().find(|l| l.contains("digest")).unwrap();
        let db = b.lines().find(|l| l.contains("digest")).unwrap();
        assert_eq!(da, db, "the clock moved the digest");
        assert_ne!(a, b, "generated_at should still be recorded");

        // And a record that genuinely changed must change it, or the digest is
        // decoration rather than evidence.
        let job2 = p.make_job(0, 8).unwrap();
        let mut h2 = Hasher::new(&job2.algo, &job2.header).unwrap();
        let n2 = (0..200_000u32)
            .find(|n| below_target(&h2.hash(&job2.header, *n), &job2.target))
            .expect("no second share found");
        p.submit("w2", &proto::Share { job: job2.job, nonce: n2, echo: vec![] });
        let c = p.ledger_json(1, 1_000);
        let dc = c.lines().find(|l| l.contains("digest")).unwrap();
        assert_ne!(da, dc, "a new share did not move the digest");
    }

    fn some_work() -> Work {
        Work {
            job_id: String::from("job1"),
            prev_wire: [0x11; 32],
            coinb1: vec![0x01, 0x00, 0x00, 0x00],
            coinb2: vec![0xff, 0xff, 0xff, 0xff],
            branch: vec![[0xaa; 32], [0xbb; 32]],
            version: 1,
            ntime: 0x4dd7_f5c7,
            nbits: 0x1a44_b9f2,
            extranonce1: vec![0xde, 0xad, 0xbe, 0xef],
            extranonce2_size: 4,
            up_target: target_with_leading_zeros(24),
        }
    }

    fn upstream_pool() -> Pool {
        Pool::new(vec![Coin {
            label: String::from("chain"),
            asset: String::from("chain"),
            algo: Algo::Sha256d,
            share_bits: 8,
            share_target: target_with_leading_zeros(8),
            network_target: None,
            source: Source::Upstream {
                host: String::from("nowhere"),
                port: 1,
                user: String::from("w"),
                pass: String::from("x"),
            },
            work: None,
            e2: 0,
        }])
    }

    /// An upstream coin with no work must not invent a header.
    ///
    /// The alternative is a miner spending real time on a search that can never
    /// pay, which from the report looks exactly like a coin that is working.
    #[test]
    fn an_upstream_coin_with_no_work_yields_no_job() {
        let mut p = upstream_pool();
        assert!(p.make_job(0, 8).is_none());
        assert!(!p.has_work(0));
        p.set_work(0, some_work());
        assert!(p.has_work(0));
        assert!(p.make_job(0, 8).is_some());
    }

    /// Two jobs from one `mining.notify` must be two different searches.
    ///
    /// The extranonce2 is the only thing separating them, so a counter that
    /// failed to advance would hand two miners identical work and pay one of
    /// them for the other's -- with both looking busy while it happened.
    #[test]
    fn two_jobs_from_one_notify_use_different_extranonces() {
        let mut p = upstream_pool();
        p.set_work(0, some_work());
        let a = p.make_job(0, 8).unwrap();
        let b = p.make_job(0, 8).unwrap();
        assert_ne!(a.header, b.header, "same header means the same search");
        // And the difference is in the merkle root, since that is the only part
        // of the header an extranonce reaches. Everything else is upstream's,
        // so a job differing anywhere else would mean something was invented.
        assert_eq!(a.header[..36], b.header[..36]);
        assert_ne!(a.header[36..68], b.header[36..68]);
        assert_eq!(a.header[68..], b.header[68..]);
    }

    /// A job carries its own working, and the working checks out.
    ///
    /// This is the pool verifying itself with the *miner's* function --
    /// `proto::proves` is the same code the kernel runs -- so a header assembly
    /// that drifted from the proof beside it would fail here rather than in a
    /// miner's log a week later.
    #[test]
    fn an_upstream_job_proves_its_own_header() {
        let mut p = upstream_pool();
        p.set_work(0, some_work());
        let job = p.make_job(0, 8).unwrap();
        let proof = job.proof.expect("an upstream job carried no proof");
        assert!(
            proto::proves(&proof, &job.header),
            "the pool's own proof does not produce the header it sent"
        );

        // A coinbase changed by one byte must not. Otherwise the check passes
        // on anything and is worse than absent, because it looks like evidence.
        let bent = proto::Proof {
            coinb1: proof.coinb1.clone(),
            extranonce: proof.extranonce.clone(),
            coinb2: {
                let mut c = proof.coinb2.clone();
                c.push(0x00);
                c
            },
            branch: proof.branch.clone(),
        };
        assert!(!proto::proves(&bent, &job.header));

        // And so must a branch in the wrong order, which is the mistake that
        // produces thirty-two perfectly plausible bytes.
        if proof.branch.len() >= 2 {
            let mut rev = proof.branch.clone();
            rev.reverse();
            let flipped = proto::Proof {
                coinb1: proof.coinb1.clone(),
                extranonce: proof.extranonce.clone(),
                coinb2: proof.coinb2.clone(),
                branch: rev,
            };
            assert!(!proto::proves(&flipped, &job.header));
        }
    }

    /// A local coin sends no proof, because it has no chain to prove anything
    /// about. A proof that verified against an invented header would be worse
    /// than none: it would look like evidence.
    #[test]
    fn a_local_coin_offers_no_proof() {
        let mut p = a_pool();
        assert!(p.make_job(0, 8).unwrap().proof.is_none());
    }

    /// The window is meaningless until it can be said in blocks, and against a
    /// real chain the number a `u64` can hold turns out to be far too small.
    ///
    /// (The first three lines of this comment used to be the tail of another
    /// test's prose, ending mid-sentence at "exactly what". An edit ate the
    /// middle and nothing reads a doc comment, so it sat there.)
    ///
    /// Two claims in one because the second only has weight beside the first.
    /// A synthetic target with a known exponent checks the arithmetic; then
    /// Bitcoin's own `nbits` from a real `mining.notify` checks the ceiling
    /// `window_work` argues for, and what the shipped configuration is worth
    /// underneath it.
    #[test]
    fn the_window_can_be_said_in_blocks_and_the_deployed_one_is_a_rounding_error() {
        let mut p = a_pool();

        // 0x1d00ffff is difficulty 1: 2^32 expected hashes to a block. A window
        // of exactly that is one block of work by construction, which is the
        // cheapest possible check of the arithmetic and would catch an
        // off-by-a-factor-of-256 in the mantissa placement.
        p.coins[0].network_target = U256::from_nbits(0x1d00_ffff);
        p.set_window(1u64 << 32);
        let b = p.window_in_blocks(&p.coins[0].label.clone()).expect("a target is known");
        assert!(
            (b - 1.0).abs() < 0.01,
            "a difficulty-1 chain and a 2^32 window is one block of work, got {b}"
        );

        // And with the window halved it is half a block, which is what
        // separates a real ratio from a constant that happens to be 1.
        p.set_window(1u64 << 31);
        let h = p.window_in_blocks(&p.coins[0].label.clone()).unwrap();
        assert!((h - 0.5).abs() < 0.01, "half the window is half a block, got {h}");

        // Now the real one. `0x17030ecd` is Bitcoin's `nbits` as carried by a
        // live `mining.notify` this pool took from solo.ckpool.org.
        p.coins[0].network_target = U256::from_nbits(0x1703_0ecd);

        // **The ceiling first**, because it is the claim `window_work` makes
        // and the only one here that does not move when a deploy script does:
        // a `u64` cannot express one Bitcoin block at all, so this is a limit
        // of the type rather than a setting somebody got wrong. Pinning the
        // largest expressible window is what says "no value would have been
        // right" -- an assertion about one deployed number cannot say that,
        // and the previous version of this test tried to.
        p.set_window(u64::MAX);
        let most = p.window_in_blocks(&p.coins[0].label.clone()).unwrap();
        assert!(
            most < 1e-4,
            "even the largest u64 window is a rounding error against Bitcoin, got {most}"
        );

        // And what `run-pool.wg.sh` actually passes, so the two stay in step.
        // It is sized for Feathercoin -- the coin on that listener whose blocks
        // a `u64` can hold -- and against Bitcoin it is nine orders of
        // magnitude short for the reason above rather than for want of a
        // larger number.
        p.set_window(1u64 << 50);
        let real = p.window_in_blocks(&p.coins[0].label.clone()).unwrap();
        assert!(
            real < 1e-8,
            "the deployed window against Bitcoin is a rounding error, got {real}"
        );

        // A coin with no chain behind it answers nothing rather than a
        // plausible number, which is the rule the whole file runs on.
        p.coins[0].network_target = None;
        assert!(p.window_in_blocks(&p.coins[0].label.clone()).is_none());
    }

    /// A stranger cycling worker names must not be able to push a miner who
    /// has actually done work out of the ledger.
    ///
    /// This is the claim the cap exists for, and the cap alone does not make
    /// it true -- a plain ceiling would refuse the *real* miner whenever the
    /// flood got there first, which is the same denial of service arriving one
    /// step later. What makes it hold is that eviction only ever takes records
    /// with zero credited work, and credited work costs hashes that met a
    /// target.
    ///
    /// Driven the wrong way round on purpose: the flood fills the map *first*,
    /// so the honest worker arrives to a map that is already full. A test that
    /// registered the miner first would pass on a naive ceiling and prove
    /// nothing.
    #[test]
    fn invented_names_cannot_crowd_out_a_worker_with_credited_work() {
        let mut p = a_pool();
        for i in 0..(MAX_TALLIES + 500) {
            p.tally("flood-{i}".replace("{i}", &i.to_string()).as_str(), "t").bad += 1;
        }
        assert!(p.tally_len() <= MAX_TALLIES, "the map is bounded");

        // An accepted share is what "did work" means, and it is the one thing
        // in this test that cannot be faked cheaply.
        let t = p.tally("real", "t");
        t.accepted += 1;
        t.work += 1 << 20;
        assert_eq!(p.tally("real", "t").work, 1 << 20, "the record was created");

        // Now flood again, harder than the whole map, and it must still be there.
        for i in 0..(MAX_TALLIES * 2) {
            p.tally("later-{i}".replace("{i}", &i.to_string()).as_str(), "t").stale += 1;
        }
        assert_eq!(
            p.tally("real", "t").work,
            1 << 20,
            "a worker with credited work survives a flood of invented names"
        );
        assert!(p.tally_len() <= MAX_TALLIES, "still bounded after the second flood");
        // And nothing was refused, which is the interesting half: the cap did
        // not have to turn anybody away, because everything it dropped was a
        // record owed nothing. `untallied` is for the case below.
        assert_eq!(p.untallied(), 0, "eviction found room without refusing anyone");
    }

    /// The other end of the same rule: when the map really is full of workers
    /// who are owed something, there is nothing safe to drop and the pool says
    /// so instead of dropping one of them.
    ///
    /// This is the branch that would be dead code if only the flood case were
    /// tested, and a bound whose refusal path has never run is a bound that
    /// panics the first time it matters.
    #[test]
    fn a_tally_full_of_paid_workers_refuses_rather_than_evicting_one() {
        let mut p = a_pool();
        for i in 0..MAX_TALLIES {
            let t = p.tally("paid-{i}".replace("{i}", &i.to_string()).as_str(), "t");
            t.accepted += 1;
            t.work += 1 << 10;
        }
        assert_eq!(p.tally_len(), MAX_TALLIES, "full, and every record is owed something");

        p.tally("newcomer", "t").bad += 1;
        assert_eq!(p.tally_len(), MAX_TALLIES, "nobody was evicted to make room");
        assert_eq!(p.untallied(), 1, "and the refusal was counted rather than silent");
        assert_eq!(
            p.tally("paid-0", "t").work,
            1 << 10,
            "the record that would have been evicted is intact"
        );
    }

    /// A difficulty is remembered across reconnects, and the map that
    /// remembers it is bounded by the same argument.
    #[test]
    fn a_difficulty_survives_a_reconnect_and_the_map_is_bounded() {
        let mut p = a_pool();
        let fresh = p.resume_bits("nobody", 0);
        assert_eq!(fresh, p.start_bits(0), "an unknown worker starts where it always did");

        p.remember_bits("phone", 0, 11);
        assert_eq!(p.resume_bits("phone", 0), 11, "and a known one does not start over");
        assert_eq!(p.resume_bits("phone", 1), p.start_bits(1), "per slot, not per worker");

        for i in 0..(MAX_REMEMBERED + 100) {
            p.remember_bits("w-{i}".replace("{i}", &i.to_string()).as_str(), 0, 12);
        }
        assert!(p.remembered() <= MAX_REMEMBERED, "bounded");
        // Already present, so it is an update rather than an insertion and the
        // cap must not refuse it. A cap that refused updates would freeze every
        // remembered difficulty the moment the map filled.
        p.remember_bits("phone", 0, 9);
        assert_eq!(p.resume_bits("phone", 0), 9, "an existing entry is still updated when full");
    }

    /// VarDiff broke about a raw share count: a miner retargeted to 12 bits
    /// finds 256 times as many shares as one at 20 for the same work. Counting
    /// shares would pay it 256 times as much.
    ///
    /// So: mine the *same nonce range* twice, once at an easy target and once
    /// at a hard one, and require the credited work to land close. The share
    /// counts will differ by orders of magnitude, which is the point.
    #[test]
    fn credit_follows_work_and_not_share_count() {
        const SPAN: u32 = 400_000;
        // **Several jobs, not one, and that is a fix rather than thoroughness.**
        //
        // A local coin's header carries `SystemTime::now()`, so every run of
        // this test sweeps a *different* header and finds its solutions in
        // different places. One job at 16 bits expects six shares over this
        // span, and six is small enough that the ratio below swings past a
        // factor of three by luck alone -- which it did, once, in a suite run,
        // and passed on every re-run afterwards.
        //
        // That is the worst shape a check can have: it fails rarely enough to
        // be re-run rather than read, which is how a real regression gets
        // dismissed as "the flaky one". Eight jobs is eight times the shares
        // and cuts the spread by about the root of that, without widening the
        // bound -- widening it would have hidden the variance instead of
        // reducing it, and the bound is the thing being asserted.
        const JOBS: u32 = 8;

        fn run(bits: u32) -> Tally {
            let mut p = a_pool();
            for _ in 0..JOBS {
                let job = p.make_job(0, bits).unwrap();
                let mut h = Hasher::new(&job.algo, &job.header).unwrap();
                for n in 0..SPAN {
                    if below_target(&h.hash(&job.header, n), &job.target) {
                        p.submit(
                            "w",
                            &proto::Share {
                                job: job.job.clone(),
                                nonce: n,
                                echo: vec![],
                            },
                        );
                    }
                }
            }
            p.ledger().into_iter().next().map(|(_, _, t)| t).unwrap_or_default()
        }

        let easy = run(10);
        let hard = run(16);
        // Measured, and worth keeping in the record: 394 shares against 5 for
        // the same nonce sweep, with the work landing within 1.2x. That ratio
        // is what a share count would have paid out on.

        // The counts differ enormously -- that is the whole hazard.
        assert!(
            easy.accepted > hard.accepted * 8,
            "the two difficulties were not far enough apart to test anything: \
             {} vs {}",
            easy.accepted,
            hard.accepted
        );

        // The work does not. Both swept the same nonces, so the expected work
        // is the same span; a factor of two either way is the luck of where the
        // solutions fell, and is generous rather than tight because share
        // intervals are exponentially distributed.
        assert!(easy.work > 0 && hard.work > 0);
        let (lo, hi) = if easy.work < hard.work {
            (easy.work, hard.work)
        } else {
            (hard.work, easy.work)
        };
        assert!(
            hi <= lo * 3,
            "credited work diverged with difficulty: {} at 10 bits against {} at 16 \
             (over {} jobs of {} nonces; {} shares against {})",
            easy.work,
            hard.work,
            JOBS,
            SPAN,
            easy.accepted,
            hard.accepted
        );
    }

    /// A ledger from before work was recorded is refused by name.
    ///
    /// Refusing it as a *digest mismatch* would be the wrong error entirely: a
    /// format change and a damaged file are different problems, and one
    /// confusing message for both is how somebody concludes their record was
    /// corrupted when it was merely old.
    #[test]
    fn an_older_ledger_format_is_refused_for_the_right_reason() {
        let old = r#"{
  "epoch": 1,
  "generated_at": 1,
  "digest": "00",
  "coins": [],
  "shares": [
    {"worker": "w", "coin": "t", "accepted": 1, "stale": 0, "bad": 0, "duplicate": 0}
  ]
}"#;
        let mut p = a_pool();
        let err = p.load_ledger(old).expect_err("an old ledger was accepted");
        assert!(
            err.contains("older pool"),
            "refused for the wrong reason: {err}"
        );
        assert!(p.ledger().is_empty());
    }

    /// A share is forwarded only when it beats *upstream's* target, not ours.
    ///
    /// One number for both would either flood upstream with work it rejects or
    /// leave a miner silent for hours, which is why there are two.
    #[test]
    fn only_shares_beating_the_upstream_target_are_forwarded() {
        let mut p = upstream_pool();
        // Ours easy, upstream's hard enough that almost nothing passes it.
        p.set_work(
            0,
            Work {
                up_target: target_with_leading_zeros(28),
                ..some_work()
            },
        );
        let job = p.make_job(0, 8).unwrap();
        let mut h = Hasher::new(&job.algo, &job.header).unwrap();

        let mut accepted = 0;
        for n in 0..200_000u32 {
            if below_target(&h.hash(&job.header, n), &job.target)
                && p.submit(
                    "w",
                    &proto::Share {
                        job: job.job.clone(),
                        nonce: n,
                        echo: vec![],
                    },
                ) == Verdict::Accepted
            {
                accepted += 1;
            }
        }
        assert!(accepted > 0, "no share met even the easy target");
        let fwd = p.take_forwards();
        assert!(
            fwd.len() < accepted,
            "every accepted share was forwarded, so the two targets are not distinct"
        );
        // Whatever was forwarded carries exactly what a `mining.submit` needs.
        for f in &fwd {
            assert_eq!(f.job_id, "job1");
            assert_eq!(f.extranonce2.len(), 4);
            assert_eq!(f.ntime_be.len(), 4);
            assert_eq!(f.nonce_be.len(), 4);
        }
        // Draining is draining: a share is only worth anything on the
        // connection whose extranonce1 it was found under.
        assert!(p.take_forwards().is_empty());
    }

    /// A restart must not lose a miner's record.
    ///
    /// The write and the read are the same canonicalisation, so this is a
    /// check that they agree rather than a check that a file exists -- and a
    /// disagreement is exactly what a digest over the rows is for.
    #[test]
    fn a_ledger_survives_being_written_and_read_back() {
        let mut p = a_pool();
        let job = p.make_job(0, 8).unwrap();
        let mut h = Hasher::new(&job.algo, &job.header).unwrap();
        let n = (0..200_000u32)
            .find(|n| below_target(&h.hash(&job.header, *n), &job.target))
            .expect("no share found");
        p.submit(
            "w1",
            &proto::Share {
                job: job.job.clone(),
                nonce: n,
                echo: vec![],
            },
        );
        // A refusal too, so the row carries more than one non-zero field and a
        // reader that restored only `accepted` would be caught.
        p.submit(
            "w1",
            &proto::Share {
                job: String::from("ffffffff"),
                nonce: n,
                echo: vec![],
            },
        );
        let doc = p.ledger_json(1, 1_000);

        let mut fresh = Pool::new(vec![Coin {
            label: String::from("test"),
            asset: String::from("test"),
            algo: Algo::Sha256d,
            share_bits: 8,
            share_target: target_with_leading_zeros(8),
            network_target: None,
            source: Source::Local,
            work: None,
            e2: 0,
        }]);
        let rows = fresh.load_ledger(&doc).expect("a ledger we just wrote was refused");
        assert!(rows > 0);
        assert_eq!(fresh.ledger(), p.ledger(), "the record changed on the way back");
    }

    /// A damaged ledger is refused whole rather than loaded in part.
    ///
    /// Half a record is worse than none: the counts would be wrong in a way
    /// nothing downstream could detect, where an empty tally is at least
    /// obviously empty and is announced.
    #[test]
    fn a_tampered_ledger_is_refused() {
        let mut p = a_pool();
        let job = p.make_job(0, 8).unwrap();
        let mut h = Hasher::new(&job.algo, &job.header).unwrap();
        let n = (0..200_000u32)
            .find(|n| below_target(&h.hash(&job.header, *n), &job.target))
            .expect("no share found");
        p.submit(
            "w1",
            &proto::Share {
                job: job.job,
                nonce: n,
                echo: vec![],
            },
        );
        let doc = p.ledger_json(1, 1_000);

        // Somebody gives themselves credit without recomputing the digest,
        // which is the whole shape this check catches.
        let doctored = doc.replace("\"accepted\": 1", "\"accepted\": 9999");
        assert_ne!(doctored, doc, "the test did not actually change anything");

        let mut fresh = a_pool();
        let err = fresh
            .load_ledger(&doctored)
            .expect_err("an edited ledger was accepted");
        assert!(err.contains("digest"), "refused for the wrong reason: {err}");
        // And nothing was loaded from it.
        assert!(fresh.ledger().is_empty());

        // Truncation is the accidental version of the same thing.
        let cut = &doc[..doc.len() / 2];
        assert!(fresh.load_ledger(cut).is_err());
    }

    /// Two jobs must not be the same search. A fixed header would make every
    /// job identical, so a nonce found once is a share forever.
    #[test]
    fn two_jobs_are_two_different_searches() {
        let mut p = a_pool();
        let a = p.make_job(0, 8).unwrap();
        let b = p.make_job(0, 8).unwrap();
        assert_ne!(a.job, b.job);
        assert_ne!(a.header, b.header);
    }
}
