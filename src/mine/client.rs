//! The task that owns the pool connection.
//!
//! One task, one socket, and nothing else in the kernel ever touches that
//! handle. `mine off` sets an atomic and the task closes on its next wake;
//! closing it from the shell would put two tasks in the TCP state machine,
//! which is the thing `net`'s `StackGuard` and `tcp::at`'s masking exist to
//! prevent.
//!
//! ### Why the loop blocks rather than polls
//!
//! `tcp::service()` has one call site, the shell's idle loop, so the stack does
//! not advance while a command runs. But `tcp::wait_until` polls the NIC
//! sixteen times and runs `pump()` on every iteration, and `pump` services all
//! sixteen connection slots. **So a task blocked inside `recv_at` is driving
//! the whole stack**, and a task that polled `pending()` and yielded would
//! drive nothing and watch its own connection die.
//!
//! That is also why there is no `yield_now` here. `task::yield_now`'s own doc
//! records the hang that a hundred-times-a-second yield loop in `net::tcp`
//! caused, and `initiative` states the house rule. The waiting happens inside
//! `wait_until`'s `hlt`; the parked case is a plain spin that preemption shares
//! out.
//!
//! ### Waiting for a job and sending a share are the same wait
//!
//! There is one thread of control on the socket, so there is no conflict to
//! resolve, only a latency to bound. `RECV_MS` is that bound: a share found
//! while the task is blocked waits at most that long before the queue is
//! drained. Two hundred milliseconds against a share found once an hour is not
//! a trade worth widening `recv_at`'s signature for, and if it ever is, the
//! wake-predicate form is written up in the plan.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::net::tcp;
use crate::sync::{Racy, Spin};

use super::stratum::{self, Message};

/// How long a blocked read waits before the submit queue is drained.
const RECV_MS: u64 = 200;
/// How long to wait for the TCP handshake.
const CONNECT_MS: u64 = 8_000;
/// Reconnect backoff, doubling, in milliseconds.
const BACKOFF_MIN: u64 = 1_000;
const BACKOFF_MAX: u64 = 60_000;
/// Journal depth for `mine log`.
const JOURNAL: usize = 32;

/// Whether a slice stands down while the model is working.
///
/// **On by default, and the default is the argument.** This kernel's reason for
/// existing is the model in it; mining is a side job that pays for the token.
/// A miner that quietly takes a third of the machine's arithmetic away from
/// inference has inverted that, and it does so invisibly -- the model does not
/// get slower in a way anybody can see, it just is slower.
///
/// Measured under WHPX at `-smp 4`, three decodes of twelve tokens per
/// condition, in one boot on SmolLM2:
///
///     no mining              692  751  686 ms/token    median  692
///     4 slices, no yield    1538 1452 1347             median 1452
///     4 slices, yielding    1074 1109 1089             median 1089
///
/// So mining **doubles** the cost of a token, and standing down gives back
/// about half of that: `(1452 - 1089) / (1452 - 692)` is 48%.
///
/// **It does not give back all of it, and the reason is structural.**
/// `with_engine` claims the engine for the length of one call, so a decode
/// releases it between tokens and a slice legitimately works in those gaps.
/// Parked slices also keep their working sets resident, and on a
/// memory-bound algorithm that costs the model bandwidth whether or not
/// anybody is hashing. Recovering the rest means a claim that spans a whole
/// generation, which is a change to the engine rather than to the miner.
///
/// The absolute figures are a fact about the host, a hypervisor being present.
/// The *ratios* between conditions in one boot are the part worth quoting,
/// which is the argument `video bench` makes about its own control -- and it
/// had to be made here too: a first attempt at this comparison with one sample
/// per condition put idle anywhere between 2.64 and 5.30 GFLOP/s on the matmul
/// bench, which is a 2x spread across a measurement of nothing changing.
static YIELD_TO_MODEL: AtomicBool = AtomicBool::new(true);

pub fn yield_to_model() -> bool {
    YIELD_TO_MODEL.load(Ordering::Relaxed)
}

pub fn set_yield_to_model(on: bool) {
    YIELD_TO_MODEL.store(on, Ordering::Relaxed);
}

/// How many batches stood down, so the cost is visible rather than assumed.
///
/// A miner that yields is a miner whose rate is lower than the hardware could
/// deliver, and an operator comparing this machine against a hashrate
/// calculator deserves to know why rather than to conclude the kernel is slow.
pub static YIELDED: AtomicU64 = AtomicU64::new(0);

/// Set by `mine on`, cleared by `mine off`. The task never exits.
pub static ENABLED: AtomicBool = AtomicBool::new(false);
/// Serial of the current template. Bumped on every job and on every
/// disconnection, so a hash loop can tell that its work is now worthless.
pub static JOB_SERIAL: AtomicU64 = AtomicU64::new(0);
/// Whether the handshake got all the way through.
pub static LIVE: AtomicBool = AtomicBool::new(false);
static SPAWNED: AtomicBool = AtomicBool::new(false);
static NEXT_ID: AtomicU64 = AtomicU64::new(3);
/// Share difficulty as the pool last set it, mantissa and scale.
static DIFF_M: AtomicU64 = AtomicU64::new(1);
static DIFF_S: AtomicU32 = AtomicU32::new(0);

/// Which dialect the far end speaks.
///
/// A setting rather than something negotiated. A pool that answered `hello`
/// with a Stratum error and a pool that answered a `subscribe` with nothing
/// look identical from here -- a silent connection -- so probing for it would
/// turn a typo in an address into a minute of waiting with no reason given.
#[derive(Clone, Copy, PartialEq)]
pub enum Protocol {
    /// Somebody else's pool, one coin, no algorithm on the wire.
    StratumV1,
    /// Ours. Several coins down one connection, each with its algorithm. See
    /// `design/pool.md` and `mine::proto`.
    Glados,
}

impl Protocol {
    pub fn name(self) -> &'static str {
        match self {
            Protocol::StratumV1 => "stratum v1",
            Protocol::Glados => "glados",
        }
    }
}

/// Where to connect and as whom. Not the connection itself.
pub struct Config {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub pass: String,
    pub proto: Protocol,
}

pub static CONFIG: Spin<Option<Config>> = Spin::new(None);

/// The job in force, already reduced to what a hash loop needs.
///
/// Carries the midstate rather than only the header, so the hash loop clones a
/// forty-byte state per batch instead of re-absorbing sixty-four bytes. And it
/// carries `extranonce2` and `ntime_be` because the submit has to send the
/// *same* bytes that were hashed: formatting them a second time at submit is
/// how a client ends up submitting a share for a header it never built.
pub struct Template {
    pub serial: u64,
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime_be: Vec<u8>,
    /// The assembled header with a zero nonce. The miner substitutes into it.
    ///
    /// Not a midstate any more: that is a SHA-256d-only optimisation, since it
    /// rests on the first 64 bytes being absorbable once, and yespower puts all
    /// eighty through every nonce. The midstate is rebuilt on the miner's side
    /// for the algorithms that can use one.
    pub header: [u8; 80],
    pub target: super::u256::U256,
    /// The network's own target, for the expected-value block. On the wire,
    /// so it moves with the job and is never a constant.
    pub nbits: u32,
    /// What the pool is paying itself for this block, summed from the coinbase
    /// outputs. `None` when the transaction did not parse exactly, in which
    /// case the report omits every line that depends on it.
    pub coin_value: Option<u64>,
    /// How long the assembled coinbase was, and its first bytes. Printed when
    /// the parse fails, because "could not read it" without saying what was
    /// read is a diagnostic that sends you to the wrong file -- it cost a run
    /// working out whether the stub or the assembly was wrong.
    pub coinbase_len: usize,
    pub coinbase_head: [u8; 8],
    /// Whatever the pool wants handed back at submit time, stored without
    /// being read. Empty under Stratum V1, where `extranonce2` and `ntime_be`
    /// above are the same idea with the fields named.
    pub echo: super::proto::Echo,
    /// Whether this job arrived with a proof that checked out.
    ///
    /// False covers two different things and the report says which: a pool that
    /// sent no proof, and Stratum V1, where the miner assembled the header
    /// itself and there was never anything to take on trust.
    pub verified: bool,
}

/// The slot the Stratum connection fills.
///
/// Fixed at zero rather than allocated. There is one connection, and a slot
/// number that moved would mean a share queued before a reconnection could be
/// submitted against a different coin's job after it.
pub const POOL_SLOT: usize = 0;

/// Which proof-of-work to compute.
///
/// A setting rather than something the pool says, because Stratum V1 has no
/// field for it -- the pool and the miner are simply assumed to agree, which is
/// true when a pool serves one coin and is the whole reason our own pool wants
/// a protocol that says so out loud. See `design/mining.md`.
///
/// It lives on the *slot* now rather than being one global, because the whole
/// point of the work table is that two slices may be computing different
/// functions at the same instant.
pub fn algo_in_force() -> super::algo::Algo {
    super::work::algo(POOL_SLOT).unwrap_or(super::algo::Algo::Sha256d)
}

/// Set the pool slot's algorithm, creating the slot if the operator has not.
pub fn set_pool_algo(a: super::algo::Algo) {
    let label = super::work::label(POOL_SLOT).unwrap_or_else(|| {
        CONFIG
            .lock_irq()
            .as_ref()
            .map(|c| c.host.clone())
            .unwrap_or_else(|| String::from("pool"))
    });
    super::work::install(POOL_SLOT, &label, a, super::work::Source::Pool);
}

/// A share, waiting for the socket task to send it.
pub struct Share {
    /// Which coin it belongs to. Checked before submitting, because only the
    /// pool slot has an upstream that issued the header it was found against.
    pub slot: usize,
    pub serial: u64,
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime_be: Vec<u8>,
    pub nonce_be: Vec<u8>,
    pub echo: super::proto::Echo,
}

pub static SHARES: Spin<Vec<Share>> = Spin::new(Vec::new());

/// Set by `mine hash on|off`, separately from the connection.
pub static MINING: AtomicBool = AtomicBool::new(true);
pub static HASHES: AtomicU64 = AtomicU64::new(0);
pub static FOUND: AtomicU64 = AtomicU64::new(0);
pub static ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static REJECTED: AtomicU64 = AtomicU64::new(0);
/// Best leading-zero count this run. A display figure, never a decision.
pub static BEST: AtomicU32 = AtomicU32::new(0);
/// When hashing started, for the rate. TSC milliseconds, never `lapic::ticks`.
static HASH_SINCE: AtomicU64 = AtomicU64::new(0);

/// How many mining slices may ever exist.
///
/// Bounded by task slots rather than by cores: `MAX_TASKS` is 24, every
/// application processor takes one through `adopt_idle`, and slots are never
/// reclaimed -- so asking for more than the table can spare would spawn until
/// the machine could not spawn anything else, ever.
///
/// **Eight rather than four, and the reason is arithmetic rather than a
/// benchmark.** Four was sized for a full boot: on the GF63's sixteen logical
/// processors, sixteen slots go to idle tasks and the shell, clock, compositor,
/// mind, agent and initiative take six more, leaving about two. A miner image
/// runs *none* of those six -- no model, so no mind, agent or initiative; no
/// desktop, so no clock or compositor -- and draws its screen from the socket
/// loop rather than a task of its own. That is six slots back, and six is what
/// makes eight sensible where four was the honest ceiling before.
///
/// `set_slices` still clamps to what `task::spawn` will actually give it and
/// reports the number it got, so this is a ceiling and not a promise: a full
/// boot asking for eight gets whatever the table can spare and says so.
///
/// The nonce stride divides by this, so raising it narrows each slice's range:
/// at eight that is 536,870,912 nonces each, which a slice at a quarter of a
/// megahash exhausts in half an hour against jobs that change every thirty
/// seconds. Not a constraint, but it is the thing that would become one.
pub const MAX_SLICES: usize = 8;

/// How many slices are wanted. Slices above this park.
static SLICES: AtomicU32 = AtomicU32::new(1);
/// How many have actually been spawned. Only ever rises, because a task that
/// returns is not reclaimed -- so slices are spawned lazily and then reused
/// rather than spawned per `mine on`.
static SPAWNED_SLICES: AtomicU32 = AtomicU32::new(0);
/// Claimed once by each slice task at entry, since `task::spawn` takes a bare
/// `fn()` and there is nowhere to pass an index.
static NEXT_SLICE: AtomicU32 = AtomicU32::new(0);

pub fn slices() -> u32 {
    SLICES.load(Ordering::Relaxed)
}

pub fn spawned_slices() -> u32 {
    SPAWNED_SLICES.load(Ordering::Relaxed)
}

/// Ask for `n` concurrent slices, spawning any that do not exist yet.
///
/// Answers how many are actually available, which can be fewer than asked when
/// the task table is full -- and saying so is the point, since the alternative
/// is a sweep that reports a flat curve because half its slices were never
/// created.
pub fn set_slices(n: u32) -> u32 {
    let n = n.clamp(1, MAX_SLICES as u32);
    while SPAWNED_SLICES.load(Ordering::Relaxed) < n {
        match crate::task::spawn("mine slice", mine_task) {
            Some(i) => {
                SPAWNED_SLICES.fetch_add(1, Ordering::Relaxed);
                // **This kernel's first `unpin` caller outside the selftest.**
                // The claim it makes is the audit above: everything this task
                // touches is an atomic, a `Spin`, or its own stack. Without it
                // every slice shares core 0 and the concurrency curve would be
                // measuring round-robin overhead rather than cache contention,
                // which is the one thing it exists to find.
                if !crate::task::unpin(i) {
                    note("a slice could not be unpinned; it stays on core 0");
                }
            }
            None => {
                note("no task slot for another slice");
                break;
            }
        }
    }
    let have = SPAWNED_SLICES.load(Ordering::Relaxed).min(n);
    SLICES.store(have, Ordering::Relaxed);
    // The supervisor runs here rather than in the hash loop, because assignment
    // is sticky by design -- see `work`'s header. Changing the slice count
    // without re-spreading would leave a newly wanted slice pointing at nothing
    // and reading, from the report, exactly like a slice that could not spawn.
    super::work::assign();
    have
}

pub fn hash_ms() -> u64 {
    let t0 = HASH_SINCE.load(Ordering::Relaxed);
    if t0 == 0 {
        return 0;
    }
    now_ms().saturating_sub(t0)
}

/// Connection and share history, newest last.
///
/// **A `Spin` and not a `Racy`, and that conversion is the whole of the unpin
/// audit.** `note` is called from the socket task, from every mining slice, and
/// from the shell, so the day a slice stops being pinned to core 0 this is two
/// cores in one `Vec`. It was the only thing the miner touched that was not
/// already an atomic or a real lock; everything else it reads or writes is
/// the work table, `SHARES` or a counter.
///
/// `lock_irq` rather than `lock`, for the reason the heap and console take it:
/// this is reachable from a path that can be preempted while holding it.
static LOG: Spin<Vec<String>> = Spin::new(Vec::new());

pub fn note(s: &str) {
    let mut v = LOG.lock_irq();
    if v.len() == JOURNAL {
        v.remove(0);
    }
    v.push(String::from(s));
}

pub fn journal() -> Vec<String> {
    LOG.lock_irq().clone()
}

pub fn difficulty() -> (u64, u32) {
    (DIFF_M.load(Ordering::Relaxed), DIFF_S.load(Ordering::Relaxed))
}

/// Where the session is. One enum so the report cannot invent a state.
#[derive(Clone, Copy, PartialEq)]
pub enum Phase {
    Off,
    Resolving,
    Connecting,
    Subscribed,
    Authorized,
    Live,
}

static PHASE: Racy<Phase> = Racy::new(Phase::Off);

pub fn phase() -> Phase {
    unsafe { *PHASE.get() }
}

fn set_phase(p: Phase) {
    unsafe { *PHASE.get() = p };
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Off => "off",
            Phase::Resolving => "resolving",
            Phase::Connecting => "connecting",
            Phase::Subscribed => "subscribed",
            Phase::Authorized => "authorized",
            Phase::Live => "live",
        }
    }
}

/// Start the connection task, once. A second `mine on` re-arms the atomic
/// rather than spawning again: `MAX_TASKS` is 24, slots are never reclaimed,
/// and on the GF63 most of them are already spoken for by idle tasks.
pub fn start() -> Result<(), &'static str> {
    if CONFIG.lock_irq().is_none() {
        return Err("no pool set -- try `mine pool <host:port>` first");
    }
    ENABLED.store(true, Ordering::Release);
    if SPAWNED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    match crate::task::spawn("stratum", stratum_task) {
        Some(_) => {}
        None => {
            SPAWNED.store(false, Ordering::Release);
            ENABLED.store(false, Ordering::Release);
            return Err("no task slot free");
        }
    }
    // The hash loop is separate tasks and not a branch of the first. The socket
    // task spends its life blocked in `recv_at`, which is what keeps the TCP
    // stack alive; hashing inside that loop would stop it doing so for the
    // length of every batch.
    if set_slices(slices()) == 0 {
        note("no task slot for a hash loop -- connected, but not mining");
    }
    Ok(())
}

pub fn stop() {
    ENABLED.store(false, Ordering::Release);
}

/// What a slice with nothing to do does.
///
/// `hlt` and not the spin `idle()` uses, and the difference only started
/// mattering when slices were unpinned. A parked task on core 0 spinning is a
/// task the scheduler hands its quantum to and takes it back from -- annoying
/// and bounded. A parked task on a core of its own spinning is that whole core
/// held at full power computing nothing, for as long as the table is empty,
/// which on this laptop is a fan that never stops after a `mine coin ... off`.
///
/// Safe because a task runs with interrupts enabled and the timer fires at
/// 100 Hz, so the longest this can sleep is one tick. The socket task keeps
/// `idle()`, since its spin is what drives the TCP stack.
fn park() {
    unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
}

/// A short spin, for the parked case. Not `yield_now`: see the module header.
fn idle() {
    for _ in 0..2000 {
        core::hint::spin_loop();
    }
}

/// Milliseconds since boot, off the TSC.
///
/// **Never `lapic::ticks()`.** That counter is the timer-interrupt count and
/// only the bootstrap processor advances it now, but the whole class of bug is
/// worth staying away from in code whose entire output is a rate.
pub fn now_ms() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return 0;
    }
    crate::time::rdtsc() / (mhz * 1000)
}

/// The next template serial.
///
/// One counter across every slot, not one per slot. A slice compares the serial
/// it snapshotted against the one it holds to decide whether to retarget, and
/// with per-slot counters a slice moved from one coin to another would see two
/// unrelated sequences and could read a *lower* number as no change at all.
pub fn bump_serial() -> u64 {
    JOB_SERIAL.fetch_add(1, Ordering::AcqRel) + 1
}

fn sleep_ms(ms: u64) {
    let deadline = now_ms() + ms;
    while now_ms() < deadline && ENABLED.load(Ordering::Acquire) {
        idle();
    }
}

struct Session {
    proto: Protocol,
    h: tcp::Handle,
    buf: Vec<u8>,
    e1: Vec<u8>,
    e2_size: usize,
    /// The job as the pool last described it, before a template is built.
    job: Option<stratum::Job>,
    /// Counter feeding extranonce2.
    e2: u64,
    /// Submit ids we are still waiting on.
    pending: Vec<u64>,
}

fn stratum_task() {
    let mut backoff = BACKOFF_MIN;
    loop {
        if !ENABLED.load(Ordering::Acquire) {
            if phase() != Phase::Off {
                set_phase(Phase::Off);
                LIVE.store(false, Ordering::Release);
                JOB_SERIAL.fetch_add(1, Ordering::AcqRel);
                super::work::drop_template(POOL_SLOT);
                note("disconnected on request");
            }
            idle();
            continue;
        }
        match connect() {
            Some(mut s) => {
                backoff = BACKOFF_MIN;
                run(&mut s);
                tcp::abort_at(s.h);
            }
            None => {
                sleep_ms(backoff);
                backoff = core::cmp::min(backoff * 2, BACKOFF_MAX);
            }
        }
        LIVE.store(false, Ordering::Release);
        // A share found against the old connection cannot be submitted on the
        // next one: `extranonce1` is per-connection, so the work is not merely
        // stale, it is unsubmittable.
        JOB_SERIAL.fetch_add(1, Ordering::AcqRel);
        // Only the pool slot's job. A fixture slot's is still perfectly good --
        // it never depended on this connection's extranonce.
        super::work::drop_template(POOL_SLOT);
        if ENABLED.load(Ordering::Acquire) {
            set_phase(Phase::Connecting);
        }
    }
}

/// Resolve, connect, subscribe, authorize. `None` on any refusal.
fn connect() -> Option<Session> {
    let (host, port, user, pass, proto) = {
        let g = CONFIG.lock_irq();
        let c = g.as_ref()?;
        (c.host.clone(), c.port, c.user.clone(), c.pass.clone(), c.proto)
    };

    set_phase(Phase::Resolving);
    let ip = match crate::net::dns::lookup(&host) {
        Ok(ip) => ip,
        Err(_) => {
            note("could not resolve the pool host");
            return None;
        }
    };

    set_phase(Phase::Connecting);
    let h = match tcp::open(ip, port, CONNECT_MS) {
        Ok(h) => h,
        Err(_) => {
            note("connection refused or timed out");
            return None;
        }
    };
    let mut s = Session {
        proto,
        h,
        buf: Vec::new(),
        e1: Vec::new(),
        e2_size: 4,
        job: None,
        e2: 0,
        pending: Vec::new(),
    };

    if proto == Protocol::Glados {
        return greet(s, &user);
    }

    if tcp::send_at(s.h, stratum::subscribe(1).as_bytes(), 5_000).is_err() {
        tcp::abort_at(s.h);
        return None;
    }
    if !await_id(&mut s, 1, |s, body| match stratum::subscribe_result(body) {
        Some((e1, size)) => {
            s.e1 = e1;
            s.e2_size = size;
            true
        }
        None => false,
    }) {
        note("the pool's subscribe reply could not be read");
        tcp::abort_at(s.h);
        return None;
    }
    set_phase(Phase::Subscribed);

    if tcp::send_at(s.h, stratum::authorize(2, &user, &pass).as_bytes(), 5_000).is_err() {
        tcp::abort_at(s.h);
        return None;
    }
    if !await_id(&mut s, 2, |_, _| true) {
        note("the pool refused this worker");
        tcp::abort_at(s.h);
        return None;
    }
    set_phase(Phase::Authorized);
    note("subscribed and authorized");
    Some(s)
}

/// The `glados.hello` handshake. One round trip and no subscribe.
///
/// There is no extranonce to negotiate, because there is no coinbase for the
/// miner to put one in: the pool assembles the header. That is the whole
/// simplification the protocol buys on this side, and it is why this function
/// is a tenth of the Stratum path above.
fn greet(mut s: Session, user: &str) -> Option<Session> {
    let hello = super::proto::encode_hello(1, user, concat!("glados/", env!("CARGO_PKG_VERSION")));
    if tcp::send_at(s.h, hello.as_bytes(), 5_000).is_err() {
        tcp::abort_at(s.h);
        return None;
    }
    set_phase(Phase::Subscribed);

    let mut welcomed = false;
    if !await_id(&mut s, 1, |_, body| {
        // `classify` hands back the whole message rather than the `result`
        // field -- `subscribe_result` unwraps it too, and reading past the
        // wrapper here cost a full end-to-end run: the pool logged the hello
        // and the answer, and the kernel reported that nothing had answered.
        let Some(result) = body.get("result") else {
            return false;
        };
        match super::proto::parse_welcome(result) {
            Some(w) => {
                // Refused rather than adapted. Two ends disagreeing about the
                // wire have nothing to say to each other, and a client that
                // guessed at an older shape would be a second code path that
                // only ever runs against a pool nobody is running.
                if w.v != super::proto::VERSION {
                    return false;
                }
                welcomed = true;
                true
            }
            None => false,
        }
    }) {
        note(if welcomed {
            "the pool speaks a different version of this protocol"
        } else {
            "the pool did not answer the greeting"
        });
        tcp::abort_at(s.h);
        return None;
    }
    set_phase(Phase::Authorized);
    note("greeted, and the pool answered");
    Some(s)
}

/// A `glados.job`: install it as a coin in the work table.
///
/// **The pool's slot number is used directly**, which is what lets one
/// connection feed several slices at once. It also means a pool job displaces
/// whatever `mine coin` put in that slot, and that is the right way round: a
/// fixture exists because there was no real work, and there is now.
fn handle_glados_job(params: &crate::json::Json) {
    let Some(j) = super::proto::parse_job(params) else {
        // A job this build cannot hash is declined and said out loud. Silence
        // here reads exactly like a pool that has stopped sending.
        note("a job was refused: unknown algorithm or a malformed field");
        return;
    };
    let slot = j.slot as usize;
    if slot >= super::work::MAX_COINS {
        note("the pool offered a slot beyond this build's coin table");
        return;
    }

    // **What the pool is paid, checked rather than taken on trust.**
    //
    // A job is a finished header, so the coinbase is inside a merkle root and
    // a root is a hash: there is nothing to read. When the pool sends its
    // working, rebuilding it and requiring the same root is a complete check --
    // a matching root proves the coinbase shown is the coinbase committed to,
    // and a pool cannot show one and mine another without breaking it.
    //
    // Once per job against several billion hashes, so it costs nothing.
    let mut verified = false;
    let mut value = None;
    let mut cb_len = 0usize;
    let mut cb_head = [0u8; 8];
    if let Some(p) = &j.proof {
        if !super::proto::proves(p, &j.header) {
            // Refused, and this is the one refusal in the file that is about
            // honesty rather than capability. The job would hash perfectly
            // well; what fails is the pool's account of what it pays. Mining
            // it anyway would make the check decorative.
            note("a job's proof does not match its header -- refused");
            return;
        }
        verified = true;
        let cb = super::proto::coinbase_of(p);
        value = super::ev::coinbase_value(&cb);
        cb_len = cb.len();
        for (i, b) in cb.iter().take(8).enumerate() {
            cb_head[i] = *b;
        }
    }

    // `install` keeps the template it finds when the coin has not changed, so
    // this is a no-op on every job after the first for a slot -- and it resets
    // the slot's rate when the algorithm does change, which is what stops a
    // figure spanning two different functions.
    super::work::install(slot, &j.coin, j.algo, super::work::Source::Pool);

    let serial = bump_serial();
    let t = Template {
        serial,
        job_id: j.job,
        // Under this protocol these two are the pool's problem: it assembled
        // the header, so it holds whatever went into it. They stay empty
        // rather than being filled with something plausible.
        extranonce2: Vec::new(),
        ntime_be: Vec::new(),
        header: j.header,
        target: j.target,
        nbits: 0,
        coin_value: value,
        coinbase_len: cb_len,
        coinbase_head: cb_head,
        echo: j.echo,
        verified,
    };
    super::work::set_template(slot, t);

    // Shares for the previous job on this slot cannot be submitted: the job id
    // is gone. Dropped here rather than filtered at submit, and only for this
    // slot, because another coin's queued shares are still perfectly current.
    if j.clean {
        SHARES
            .lock_irq()
            .retain(|sh| sh.slot != slot || sh.serial == serial);
    }
}

/// Read until the response with `id` arrives, handling notifications on the way.
///
/// The notifications are not a distraction to be skipped: a pool sends
/// `set_difficulty` and the first `notify` *before* it answers the authorize on
/// several implementations, so a reader that discarded anything that was not
/// the awaited id would throw away the first job of every session.
fn await_id(
    s: &mut Session,
    id: u64,
    mut on_ok: impl FnMut(&mut Session, &crate::json::Json) -> bool,
) -> bool {
    let deadline = now_ms() + 15_000;
    while now_ms() < deadline {
        if !ENABLED.load(Ordering::Acquire) {
            return false;
        }
        if !fill(s) {
            return false;
        }
        loop {
            match stratum::take_line(&mut s.buf) {
                Ok(Some(line)) => match stratum::classify(&line) {
                    Ok(Message::Response { id: got, ok, body }) if got == id => {
                        return ok && on_ok(s, &body);
                    }
                    // Dispatched by dialect for the same reason `run` does. A
                    // pool sends work immediately after the welcome, and a
                    // handshake that routed those to the Stratum handler would
                    // drop the first job of every session -- the exact failure
                    // this function's own doc comment warns about, arriving
                    // through the other protocol.
                    Ok(Message::Notify { method, params }) => match s.proto {
                        Protocol::StratumV1 => handle_notify(s, &method, &params),
                        Protocol::Glados if method == "glados.job" => handle_glados_job(&params),
                        Protocol::Glados => {}
                    },
                    // Somebody else's id, or a line we cannot read. Neither is
                    // fatal: pools send things this client does not implement.
                    Ok(_) => {}
                    Err(_) => {}
                },
                Ok(None) => break,
                Err(_) => return false,
            }
        }
    }
    false
}

/// One blocking read. `false` when the peer has gone.
///
/// `recv_at` answers an empty vector for both a timeout and end of file, and
/// its own doc says so -- a handle has no `LAST_DATA` cushion to tell them
/// apart. So `alive` is asked every time round, or a dead pool reads as a quiet
/// one forever and the miner keeps hashing a job it can never submit.
fn fill(s: &mut Session) -> bool {
    let data = tcp::recv_at(s.h, RECV_MS);
    if data.is_empty() && !tcp::alive(s.h) {
        note("the pool closed the connection");
        return false;
    }
    s.buf.extend_from_slice(&data);
    true
}

fn run(s: &mut Session) {
    set_phase(Phase::Live);
    LIVE.store(true, Ordering::Release);
    while ENABLED.load(Ordering::Acquire) {
        // **The miner's screen is drawn from here, and not from a task of its
        // own.** This kernel has no sleep: a task that wants to act once a
        // second can only spin on `yield_now`, which leaves it permanently
        // runnable and taking a share of every quantum. Measured -- with a
        // one-second dashboard on its own task, the concurrency curve went
        // 100%, 196%, 286% and then *back down* to 273% at four slices on eight
        // cores, because the painter was competing for a core with the thing it
        // was describing.
        //
        // This loop already wakes regularly: `recv_at` blocks for `RECV_MS` and
        // returns, so hanging the redraw off it costs one comparison per
        // iteration and no task at all. A frame is about 600 us against a 200 ms
        // wait, so mining does not notice, and the screen cannot outlive the
        // miner it is reporting on -- which is the right coupling anyway.
        super::screen::tick();
        if !drain_shares(s) {
            return;
        }
        if !fill(s) {
            return;
        }
        loop {
            match stratum::take_line(&mut s.buf) {
                Ok(Some(line)) => match stratum::classify(&line) {
                    // Shape, not order. `classify` sorts a notification from a
                    // response and both dialects are JSON-RPC, so the only
                    // thing that differs is which method names mean something.
                    Ok(Message::Notify { method, params }) => match s.proto {
                        Protocol::StratumV1 => handle_notify(s, &method, &params),
                        Protocol::Glados if method == "glados.job" => handle_glados_job(&params),
                        Protocol::Glados => {}
                    },
                    Ok(Message::Response { id, ok, body }) => on_submit_reply(s, id, ok, &body),
                    Err(_) => {}
                },
                Ok(None) => break,
                Err(_) => {
                    note("the pool sent a line too long to be a message");
                    return;
                }
            }
        }
    }
}

/// Send whatever the hash loop found. `false` if the socket has gone.
fn drain_shares(s: &mut Session) -> bool {
    loop {
        let Some(sh) = SHARES.lock_irq().pop() else {
            return true;
        };
        // Belt and braces against the one thing that gets a worker banned. The
        // miner already declines to queue a fixture slot's share; this is the
        // second gate, at the only point where bytes actually leave, because
        // the cost of being wrong here is not a bad measurement -- it is the
        // pool refusing this address afterwards.
        //
        // Asked of the *slot* rather than compared against `POOL_SLOT`, which
        // was right only while one connection meant one coin. Under the glados
        // protocol every slot the pool filled has an upstream and a fixture
        // beside them still must not be sent.
        let submits = super::work::coin(sh.slot)
            .as_ref()
            .map(|c| c.source == super::work::Source::Pool)
            .unwrap_or(false);
        if !submits {
            continue;
        }
        let user = {
            let g = CONFIG.lock_irq();
            match g.as_ref() {
                Some(c) => c.user.clone(),
                None => return true,
            }
        };
        let id = next_id();
        let msg = match s.proto {
            Protocol::StratumV1 => stratum::submit(
                id,
                &user,
                &sh.job_id,
                &sh.extranonce2,
                &sh.ntime_be,
                &sh.nonce_be,
            ),
            Protocol::Glados => {
                // Big-endian on the wire either way, and taken from the same
                // bytes the miner hashed rather than reformatted here -- the
                // reason `submit_hex` exists at all.
                let nonce = u32::from_be_bytes([
                    sh.nonce_be[0],
                    sh.nonce_be[1],
                    sh.nonce_be[2],
                    sh.nonce_be[3],
                ]);
                super::proto::encode_submit(
                    id,
                    &super::proto::Share {
                        job: sh.job_id.clone(),
                        nonce,
                        echo: sh.echo.clone(),
                    },
                )
            }
        };
        if tcp::send_at(s.h, msg.as_bytes(), 5_000).is_err() {
            note("could not send a share; the connection is going");
            return false;
        }
        // Bounded. A pool that never answers must not grow this forever, and
        // what an unanswered submit means is a half-dead connection rather
        // than a rejection -- which is why it is not counted as one.
        if s.pending.len() >= 32 {
            s.pending.remove(0);
        }
        s.pending.push(id);
    }
}

/// A reply to one of our submits.
///
/// The reason string is printed verbatim and that is the instrument, not
/// decoration. A pool saying "job not found" is telling you about staleness;
/// "low difficulty share" is telling you the target arithmetic is wrong; and
/// "invalid nonce" is telling you the byte order is. Folding those into a
/// counter throws away the only feedback loop that can tell them apart, and
/// nothing in this tree has yet seen one from a real server.
fn on_submit_reply(s: &mut Session, id: u64, ok: bool, body: &crate::json::Json) {
    let Some(pos) = s.pending.iter().position(|&p| p == id) else {
        return;
    };
    s.pending.remove(pos);
    if ok {
        ACCEPTED.fetch_add(1, Ordering::Relaxed);
        note("share accepted");
        return;
    }
    REJECTED.fetch_add(1, Ordering::Relaxed);
    let why = body
        .get("error")
        .and_then(|e| e.idx(1))
        .and_then(|m| m.as_str())
        .unwrap_or("no reason given");
    let mut line = String::from("share rejected: ");
    line.push_str(why);
    note(&line);
}

fn handle_notify(s: &mut Session, method: &str, params: &crate::json::Json) {
    match method {
        "mining.set_difficulty" => {
            let Some(first) = params.idx(0) else { return };
            // The raw token, not `as_i64`. See `stratum::decimal`.
            let text = match first {
                crate::json::Json::Num(t) => t.as_str(),
                _ => return,
            };
            match stratum::decimal(text) {
                Some((m, sc)) => {
                    DIFF_M.store(m, Ordering::Relaxed);
                    DIFF_S.store(sc, Ordering::Relaxed);
                    rebuild(s);
                }
                // Never silently. Keeping the previous difficulty leaves the
                // miner hashing against a target the pool did not set and
                // submitting nothing, which looks exactly like a miner that is
                // broken. Found by a stub whose difficulty Python serialised
                // as `1e-05`.
                None => note("could not read the difficulty the pool sent"),
            }
        }
        "mining.notify" => {
            if let Some(job) = stratum::parse_job(params) {
                s.job = Some(job);
                s.e2 = s.e2.wrapping_add(1);
                rebuild(s);
            }
        }
        "mining.set_extranonce" => {
            // Some altcoin pools rotate this. A client that ignored it would
            // submit into a stale extranonce space and have every share
            // rejected with nothing in the message saying why.
            let Some(e1) = params.idx(0).and_then(|j| j.as_str()) else { return };
            let Some(bytes) = stratum::unhex(e1) else { return };
            s.e1 = bytes;
            if let Some(size) = params.idx(1).and_then(|j| j.as_i64()) {
                if (0..=64).contains(&size) {
                    s.e2_size = size as usize;
                }
            }
            rebuild(s);
        }
        "client.reconnect" => {
            // Refused rather than followed. Redirecting to an address no human
            // typed is not something this kernel should do quietly.
            note("the pool asked us to reconnect elsewhere; refused");
        }
        _ => {}
    }
}

/// Turn the current job, extranonce and difficulty into something hashable.
fn rebuild(s: &mut Session) {
    let Some(job) = s.job.as_ref() else { return };
    let (m, sc) = difficulty();
    let Some(target) = super::u256::target_for(m, sc) else { return };

    let mut e2 = Vec::with_capacity(s.e2_size);
    // Big-endian, so a hex dump reads in order. Which encoding does not matter
    // for validity; what matters is that the same bytes reach the coinbase and
    // the submit, which is why they are stored rather than formatted twice.
    for i in (0..s.e2_size).rev() {
        e2.push((s.e2 >> (8 * (i % 8))) as u8);
    }
    let coinbase = super::header::coinbase(&job.coinb1, &s.e1, &e2, &job.coinb2);
    let root = super::header::merkle_root(&coinbase, &job.branch);
    let header = super::header::assemble(
        job.version,
        &job.prev_wire,
        &root,
        job.ntime,
        job.nbits,
        0,
    );
    let (ntime_be, _) = super::header::submit_hex(&header);
    let serial = bump_serial();
    let t = Template {
        serial,
        job_id: job.id.clone(),
        extranonce2: e2,
        ntime_be,
        header,
        target,
        nbits: job.nbits,
        coin_value: super::ev::coinbase_value(&coinbase),
        coinbase_len: coinbase.len(),
        coinbase_head: {
            let mut h = [0u8; 8];
            for (i, b) in coinbase.iter().take(8).enumerate() {
                h[i] = *b;
            }
            h
        },
        // Stratum V1 names its round-trip fields, so there is nothing opaque
        // to carry. Empty rather than a copy of the named ones, which would be
        // two places holding one fact.
        echo: Vec::new(),
        // Nothing to verify, because nothing was taken on trust: under Stratum
        // the miner built this header out of the coinbase halves itself.
        verified: true,
    };
    // The slot has to exist before the job goes in, and creating it here rather
    // than at `mine on` is deliberate: an operator who never ran `mine coin`
    // still gets one, named after the host, the moment the pool sends work.
    // `install` keeps the template it finds when nothing about the coin
    // changed, so this is a no-op on every job after the first.
    if super::work::algo(POOL_SLOT).is_none() {
        set_pool_algo(super::algo::Algo::Sha256d);
    }
    super::work::set_template(POOL_SLOT, t);

    // Shares for a template nobody holds any more cannot be submitted: the job
    // id is gone and the extranonce2 has moved. Dropping them here is cheaper
    // than filtering at submit and cannot leave one behind.
    //
    // **Scoped to the pool slot**, which it did not have to be while there was
    // one template. A bare `serial ==` filter now discards every other coin's
    // queued share every time this pool sends a job -- shares for work that is
    // still perfectly current, thrown away by a rebuild they have nothing to
    // do with.
    SHARES
        .lock_irq()
        .retain(|sh| sh.slot != POOL_SLOT || sh.serial == serial);
}

/// The hash loop. One task per slice, unpinned, each on one coin.
///
/// Deliberately the slow version. `smp::parallel_split` is the obvious reach
/// and it is the wrong instrument: it allows one job system-wide and its only
/// other callers are the model's forward and backward passes, so a mining batch
/// in flight makes every projection go serial and vice versa. Worse, it *spins*
/// on the bootstrap processor, and its `count * width >= 2^19` floor forces
/// batches large enough that the spin freezes the shell, the clock and this
/// connection for the whole of one. There is no batch size that is both
/// accepted and short.
///
/// So: one task, taking its round-robin share, which on a machine with seven
/// runnable tasks is about a seventh of one core. `mine` prints the measured
/// rate and the task count beside it rather than a flat-out figure, because the
/// flat-out figure is not one this machine ever delivers.
fn mine_task() {
    // Claimed once, because `task::spawn` takes a bare `fn()` and there is
    // nowhere to pass an index.
    let slice = NEXT_SLICE.fetch_add(1, Ordering::Relaxed);

    // The hasher lives here rather than in the template, because it owns a
    // working set that must not be shared with the socket task -- and with
    // several slices there is one per slice, which is the memory the budget in
    // `design/mining.md` is really spending.
    let mut held: Option<(super::algo::Algo, super::algo::Hasher)> = None;
    let mut held_slot: Option<usize> = None;
    let mut held_serial = 0u64;

    loop {
        // A slice above the wanted count parks rather than exiting, because a
        // task that returns is never reclaimed and `mine slices` would then be
        // a one-way door.
        //
        // **The work table is the switch, not `ENABLED`.** That gate was a fact
        // about the *connection*, which is the right question only while there
        // is one coin and it comes from a pool: a fixture coin installed for
        // measurement is real work with no connection behind it, and a slice
        // that waited for `mine on` would never touch it. What actually decides
        // is whether the supervisor gave this slice a slot and whether that
        // slot has a job -- both of which go away on their own when the
        // connection does, because `stratum_task` drops the pool slot's job.
        if !MINING.load(Ordering::Acquire) || slice >= SLICES.load(Ordering::Relaxed) {
            park();
            continue;
        }
        // Which coin this slice is on. The supervisor decides, and it decides
        // when the table changes rather than here -- see `work::assign`.
        let Some(slot) = super::work::slot_for(slice) else {
            park();
            continue;
        };
        // Snapshot under the lock and hash outside it. Holding it across a
        // batch would block a job update for the length of one every time.
        let Some(w) = super::work::snapshot(slot) else {
            park();
            continue;
        };

        // Rebuilt when the algorithm changes *or* when the slice moves to a
        // different coin. The second condition is the new one and it is not
        // optional: two coins can share an algorithm and still be different
        // chains, and a hasher carrying the wrong slot's midstate hashes a
        // header that never existed while looking perfectly healthy -- the
        // failure `Hasher::hash` already warns about, arriving by a new route.
        if held_slot != Some(slot) || held.as_ref().map(|(a, _)| a != &w.algo).unwrap_or(true) {
            match super::algo::Hasher::new(&w.algo, &w.header) {
                Some(h) => {
                    held = Some((w.algo.clone(), h));
                    held_slot = Some(slot);
                    held_serial = w.serial;
                }
                None => {
                    // Parameters the algorithm refuses, or a working set that
                    // will not fit. The *coin* is stood down rather than all
                    // mining: one bad slot must not stop the other three, which
                    // is what `MINING.store(false)` used to do here.
                    note("a coin's parameters are refused by its algorithm; dropping it");
                    super::work::clear(slot);
                    held = None;
                    held_slot = None;
                    continue;
                }
            }
        } else if w.serial != held_serial {
            if let Some((_, h)) = held.as_mut() {
                h.retarget(&w.header);
            }
            held_serial = w.serial;
        }
        let Some((_, hasher)) = held.as_mut() else {
            park();
            continue;
        };

        // **Stand down while the model is working.**
        //
        // Checked here rather than at the top of the loop on purpose: every
        // `continue` above is a slice with nothing to do, which parks anyway,
        // and putting the check before them would spend an atomic load on a
        // path that does not hash. This is the last gate before arithmetic.
        //
        // `engine_holder` is a claim rather than a lock, and it is held for a
        // whole episode by the mind and the agent, and for the length of one
        // call by a foreground `ask`. So a decode makes this flap on and off
        // per call, which is exactly right: the slice sleeps through the
        // forward passes and works in the gaps between them.
        //
        // `park()` and not `idle()`, for the reason `park` gives: an unpinned
        // slice spinning is a whole core held at full power computing nothing.
        if yield_to_model() && crate::ai::engine_holder().is_some() {
            YIELDED.fetch_add(1, Ordering::Relaxed);
            park();
            continue;
        }

        if HASH_SINCE.load(Ordering::Relaxed) == 0 {
            HASH_SINCE.store(now_ms(), Ordering::Relaxed);
        }

        // Each slice owns a disjoint quarter of the nonce space, so two slices
        // on the same coin never hash the same header twice -- which would burn
        // a core to find a share somebody else already found and would make the
        // concurrency curve read as scaling when it is duplicating.
        let batch = w.algo.batch();
        let stride = (u32::MAX / MAX_SLICES as u32).wrapping_add(1);
        let base = slice
            .wrapping_mul(stride)
            .wrapping_add((HASHES.load(Ordering::Relaxed) & 0xffff_ffff) as u32);
        for i in 0..batch {
            let nonce = base.wrapping_add(i);
            let d = hasher.hash(&w.header, nonce);
            let z = super::hash::leading_zero_bits(&d);
            if z > BEST.load(Ordering::Relaxed) {
                BEST.store(z, Ordering::Relaxed);
            }
            if super::hash::below_target(&d, &w.target) {
                FOUND.fetch_add(1, Ordering::Relaxed);
                super::work::found(slot);
                // A fixture slot has no upstream that issued this header, so
                // its share is counted and dropped. Submitting it would send
                // the pool work for a job it never sent, which is a ban.
                if !w.submits {
                    continue;
                }
                let mut q = SHARES.lock_irq();
                // Bounded: a misconfigured difficulty of nearly zero would
                // otherwise queue faster than the socket can drain, and the
                // heap is the thing that runs out.
                if q.len() < 64 {
                    q.push(Share {
                        slot,
                        serial: w.serial,
                        job_id: w.job_id.clone(),
                        extranonce2: w.extranonce2.clone(),
                        ntime_be: w.ntime_be.clone(),
                        nonce_be: nonce.to_be_bytes().to_vec(),
                        echo: w.echo.clone(),
                    });
                }
            }
        }
        super::work::count(slot, batch as u64);
        HASHES.fetch_add(batch as u64, Ordering::Relaxed);
    }
}

/// Hash as fast as this machine can for `ms`, with no pool and no network.
///
/// Bounded, and that is not a nicety: `drive.py` sends the next command when it
/// sees a prompt, so an unbounded measurement leaves the harness with commands
/// unsent -- the lesson `port bars <ms>` records. Runs on the caller's task.
///
/// The header is a fixture rather than a real job, because the rate does not
/// depend on which bytes go in and requiring a pool to measure a hash rate
/// would make the number impossible to take before the pool exists.
pub fn bench(algo: &super::algo::Algo, ms: u64) -> Option<(u64, u64, usize)> {
    let header: [u8; 80] = core::array::from_fn(|i| (i as u32 * 3) as u8);
    let mut h = super::algo::Hasher::new(algo, &header)?;
    let foot = h.footprint();
    let t0 = now_ms();
    let deadline = t0 + ms;
    let mut n = 0u64;
    let mut nonce = 0u32;
    // In chunks, so the clock is read once per chunk rather than once per hash
    // -- at sha256d speed the read would otherwise be a measurable part of what
    // is being measured.
    let chunk = algo.batch();
    while now_ms() < deadline {
        for _ in 0..chunk {
            core::hint::black_box(h.hash(&header, nonce));
            nonce = nonce.wrapping_add(1);
        }
        n += chunk as u64;
    }
    Some((n, now_ms().saturating_sub(t0), foot))
}

/// Wait, on `hlt` rather than a spin.
///
/// The sweep must not compete with what it is measuring: a shell task spinning
/// through the measurement is one more runnable task on the core, and the whole
/// question being asked is how several runnable tasks interact.
fn rest_ms(ms: u64) {
    let deadline = now_ms() + ms;
    while now_ms() < deadline {
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

/// What the sweep displaced, so it can be put back.
///
/// The *templates* are not saved. A live pool re-sends a job within seconds, so
/// the cost of dropping one is a few seconds of idle at worst -- against
/// carrying a second way to construct a `Template` here, which is exactly the
/// duplicate-writer arrangement `v4.py` and `tokenizer.py --verify` exist to
/// avoid. What is saved is the table's *shape*, which nothing else can restore.
pub struct SweepState {
    mining: bool,
    slices: u32,
    table: alloc::vec::Vec<Option<(String, super::algo::Algo, super::work::Source)>>,
}

/// One point of the concurrency curve: `n` slices for `ms`, aggregate H/s.
///
/// The aggregate is meaningful here and nowhere else: `sweep_begin` puts one
/// coin in the table, so every slice is computing the same function and the
/// objection `work`'s header makes about summing across algorithms does not
/// apply. That is the reason the sweep clears the table rather than measuring
/// whatever happens to be in it.
pub fn sweep_point(n: u32, ms: u64) -> (u32, u64, u64) {
    let have = set_slices(n);
    HASHES.store(0, Ordering::Relaxed);
    HASH_SINCE.store(0, Ordering::Relaxed);
    let t0 = now_ms();
    rest_ms(ms);
    let took = now_ms().saturating_sub(t0);
    (have, HASHES.load(Ordering::Relaxed), took)
}

/// Clear the table down to one fixture coin and start the curve.
///
/// A sweep with no pool, deliberately. The concurrency curve is a fact about
/// this machine's caches and not about anybody's network, and requiring a live
/// pool to measure it would mean the one measurement that has to be taken on
/// the GF63 could only be taken with the GF63 online.
pub fn sweep_begin(algo: super::algo::Algo) -> SweepState {
    let mut table = alloc::vec::Vec::new();
    for i in 0..super::work::MAX_COINS {
        table.push(
            super::work::coin(i)
                .as_ref()
                .map(|c| (c.label.clone(), c.algo.clone(), c.source)),
        );
    }
    let state = SweepState {
        mining: MINING.load(Ordering::Relaxed),
        slices: SLICES.load(Ordering::Relaxed),
        table,
    };
    for i in 0..super::work::MAX_COINS {
        super::work::clear(i);
    }
    super::work::install(0, "sweep", algo, super::work::Source::Fixture);
    super::work::set_template(0, super::work::fixture_template(0));
    MINING.store(true, Ordering::Release);
    state
}

pub fn sweep_end(state: SweepState) {
    MINING.store(state.mining, Ordering::Release);
    // The fixture coin is not a real one and must not outlive the sweep, or the
    // next `mine` would report a job the pool never sent.
    for i in 0..super::work::MAX_COINS {
        super::work::clear(i);
    }
    for (i, c) in state.table.into_iter().enumerate() {
        if let Some((l, a, src)) = c {
            super::work::install(i, &l, a, src);
            // A pool slot gets its job back from the pool within seconds, which
            // is the whole reason templates are not saved. A **fixture** slot
            // has nobody to send it one, so not rebuilding it here leaves the
            // coin in the table reading "no job yet" for the rest of the boot
            // -- a sweep quietly killing every coin the operator installed to
            // measure. Regenerating is exactly right rather than a workaround:
            // a fixture template is generated and not received, so this is the
            // same bytes the slot had.
            if src == super::work::Source::Fixture {
                super::work::set_template(i, super::work::fixture_template(i as u8));
            }
        }
    }
    SLICES.store(state.slices, Ordering::Relaxed);
    super::work::assign();
    JOB_SERIAL.fetch_add(1, Ordering::AcqRel);
}

pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Connect, subscribe, listen for a few seconds, print what a real server said,
/// and disconnect. No hashing, no worker address, no background task.
///
/// This exists because the stub cannot produce the two things that matter most:
/// a **non-empty merkle branch** and a real coinbase. `merkle_root`'s fold has
/// never run against real data -- the fixture's branch is empty, so the loop
/// body has literally never executed outside a one-element synthetic case --
/// and `from_nbits` has never met a live network target. Both are decoded and
/// printed here so a person can check them against a block explorer.
///
/// Runs on the caller's task and blocks, the way `https` does. Bounded, because
/// `drive.py` sends the next command when it sees a prompt and an unbounded
/// full-screen command deadlocks it -- the lesson `port bars <ms>` records.
pub fn probe(host: &str, port: u16, worker: &str, seconds: u64) {
    use crate::kprintln;

    kprintln!("  resolving {}", host);
    let ip = match crate::net::dns::lookup(host) {
        Ok(ip) => ip,
        Err(e) => {
            kprintln!("  could not resolve: {:?}", e);
            return;
        }
    };
    kprintln!("  {}.{}.{}.{}:{}", ip[0], ip[1], ip[2], ip[3], port);

    let h = match tcp::open(ip, port, CONNECT_MS) {
        Ok(h) => h,
        Err(e) => {
            kprintln!("  could not connect: {:?}", e);
            return;
        }
    };
    let mut s = Session {
        // Stratum, always. This command exists to read what a *real* pool
        // sends -- a non-empty merkle branch and a live network target, which
        // our own pool by construction never produces -- so pointing it at the
        // glados dialect would defeat its whole purpose.
        proto: Protocol::StratumV1,
        h,
        buf: Vec::new(),
        e1: Vec::new(),
        e2_size: 4,
        job: None,
        e2: 0,
        pending: Vec::new(),
    };

    if tcp::send_at(s.h, stratum::subscribe(1).as_bytes(), 5_000).is_err() {
        kprintln!("  could not send subscribe");
        tcp::abort_at(s.h);
        return;
    }
    // Authorize as well, because many pools send no work until a worker is
    // named. A refusal is printed and the probe carries on: the subscribe
    // result is worth having either way, and what the pool says when it says no
    // is itself one of the things nothing here has ever seen.
    if !worker.is_empty() {
        let _ = tcp::send_at(s.h, stratum::authorize(2, worker, "x").as_bytes(), 5_000);
    }

    let deadline = now_ms() + seconds * 1000;
    let mut jobs = 0usize;
    while now_ms() < deadline {
        let data = tcp::recv_at(s.h, 500);
        if data.is_empty() && !tcp::alive(s.h) {
            kprintln!("  the pool closed the connection");
            break;
        }
        s.buf.extend_from_slice(&data);
        loop {
            match stratum::take_line(&mut s.buf) {
                Ok(Some(line)) => {
                    match stratum::classify(&line) {
                        Ok(Message::Response { id, ok, body }) => {
                            kprintln!("  <- response id {} {}", id, if ok { "ok" } else { "error" });
                            if id == 1 {
                                match stratum::subscribe_result(&body) {
                                    Some((e1, size)) => {
                                        kprintln!(
                                            "     extranonce1 {} ({} byte(s)), extranonce2 size {}",
                                            stratum::hex(&e1),
                                            e1.len(),
                                            size
                                        );
                                        s.e1 = e1;
                                        s.e2_size = size;
                                    }
                                    None => kprintln!("     could not read the subscribe result"),
                                }
                            }
                            if !ok {
                                let why = body
                                    .get("error")
                                    .and_then(|e| e.idx(1))
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("no reason given");
                                kprintln!("     reason: {}", why);
                            }
                        }
                        Ok(Message::Notify { method, params }) => {
                            kprintln!("  <- {}", method);
                            if method == "mining.set_difficulty" {
                                if let Some(crate::json::Json::Num(t)) = params.idx(0) {
                                    match stratum::decimal(t) {
                                        Some((m, sc)) => {
                                            kprintln!("     difficulty {} / 10^{}", m, sc)
                                        }
                                        None => kprintln!("     unreadable difficulty '{}'", t),
                                    }
                                }
                            } else if method == "mining.notify" {
                                match stratum::parse_job(&params) {
                                    Some(j) => {
                                        jobs += 1;
                                        dump_job(&s, &j);
                                    }
                                    None => kprintln!("     could not parse the job"),
                                }
                            }
                        }
                        Err(_) => kprintln!("  <- unreadable line"),
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    kprintln!("  the pool sent a line too long to be a message");
                    break;
                }
            }
        }
    }
    tcp::close_at(s.h, 2_000);
    kprintln!("  done -- {} job(s) seen", jobs);
}

/// Print a real job in enough detail to check it against a block explorer.
fn dump_job(s: &Session, j: &stratum::Job) {
    use crate::kprintln;
    kprintln!("     job      {}", j.id);
    kprintln!("     prevhash {}", stratum::hex(&j.prev_wire));
    kprintln!(
        "     coinbase {} + {} byte(s) around a {}-byte extranonce",
        j.coinb1.len(),
        j.coinb2.len(),
        s.e1.len() + s.e2_size
    );
    kprintln!(
        "     version {:08x}  nbits {:08x}  ntime {:08x}  clean {}",
        j.version,
        j.nbits,
        j.ntime,
        j.clean
    );
    match super::u256::U256::from_nbits(j.nbits) {
        Some(t) => {
            let b = t.to_be_bytes();
            kprintln!(
                "     network target {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}..",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
            );
        }
        None => kprintln!("     nbits {:08x} is not a target this can express", j.nbits),
    }
    // The part the stub could not reach. A real branch is several levels deep,
    // and until now the fold has only ever run over an empty one.
    kprintln!("     merkle branch {} level(s)", j.branch.len());
    let mut e2 = alloc::vec![0u8; s.e2_size];
    if let Some(last) = e2.last_mut() {
        *last = 1;
    }
    let cb = super::header::coinbase(&j.coinb1, &s.e1, &e2, &j.coinb2);
    let root = super::header::merkle_root(&cb, &j.branch);
    kprintln!("     coinbase {} byte(s), merkle root {}", cb.len(), stratum::hex(&root));
    let header = super::header::assemble(j.version, &j.prev_wire, &root, j.ntime, j.nbits, 0);
    kprintln!("     header   {}", stratum::hex(&header[..16]));
}
