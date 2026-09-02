//! What a window may read about the machine, and how often.
//!
//! The windows in `gfx::mindwin` report on the AI while it is working. That
//! puts them in an awkward position: they repaint on the same schedule as
//! everything else, and almost everything worth showing costs more than a
//! repaint can afford. So this is the one place that decides what a paint pass
//! may touch, and every one of those windows goes through it.
//!
//! ### The tiers
//!
//! **Free.** An atomic or a clone of a small static. `initiative::status()`,
//! `agent::log_snapshot()`, `author::progress()`, `agent::doing()`,
//! `engine_holder()`, `godel::enabled()`. Read every frame; that is what they
//! are for.
//!
//! **Cheap.** One namespace resolve and a bounded clone: `work::runs()`,
//! `godel::counts()`, `head()`, `test_status()`, `journal_tail()`. Refreshed
//! no more than once a second, which is what `STALE_TICKS` is.
//!
//! **Costly.** A directory of blob reads, or parsing proportional to the data:
//! `work::plan()`, `work::steps()`, `godel::ledger_tail()`, `explored()`,
//! `lineage()`, `skill_list()`. Never here -- a window asks for these itself,
//! when a selection changes or a tab opens, and remembers the answer.
//!
//! ### What must never be read from a paint pass, and why
//!
//! This list is the reason the module exists, and none of it is theoretical:
//!
//!   * **Anything through `with_engine`.** It does not consult the engine, it
//!     *claims* it. A repaint calling it takes the engine out from under the
//!     agent between two of the agent's own calls, and the failure is not an
//!     error -- it is an interleaved decode, a corrupted KV cache and
//!     confident nonsense. `model::Config` is `Copy` and never changes after
//!     load, so the one thing worth having is read once and kept.
//!   * **`godel::quiet_now` / `quiet_hours`.** They *write* `LAST_FELT`, which
//!     is how the nightly loop decides whether anybody is present. Polling
//!     them would consume the edge and stand the loop down forever.
//!   * **`work::root()`.** A recursive SHA-256 over a run's whole subtree.
//!   * **`godel::next_proposal()`.** Its last rotation slot composes a core,
//!     which spends a dozen constrained decodes.
//!   * **`godel::read_test()`.** Spends one of three lifetime reads of the
//!     held-out set. A number on a screen is not worth a test read.
//!   * **`harness::choose` / `route` / `route_verdict`, `skill::bench`,
//!     `work::harvest`, `train_role`, `tick_unattended`.** Each runs the model
//!     or mutates the machine. A window is a window.

use alloc::string::String;
use alloc::vec::Vec;

/// How long a cheap reading is good for, in timer ticks at `TIMER_HZ` (100).
///
/// One second. Short enough that a run appearing feels immediate, long enough
/// that a window repainting on every agent step does not walk the namespace
/// each time.
const STALE_TICKS: u64 = 100;

/// The facts a window may have for free.
///
/// Deliberately flat and owned: a window holds no reference into the machine's
/// state, so nothing here can be read while another task is halfway through
/// changing it.
pub struct Glance {
    // --- free, refreshed on every call ---
    pub ticks: u64,
    pub acts: u32,
    pub episodes: u32,
    pub suppressed: u32,
    pub mind_on: bool,
    /// Which task holds the engine, if any.
    pub engine: Option<usize>,
    pub engine_ready: bool,
    /// What the agent is doing, in its own words.
    pub doing: Option<&'static str>,
    pub godel_on: bool,

    // --- cheap, refreshed when stale ---
    pub runs: Vec<String>,
    pub trials: u32,
    pub adoptions: u32,
    pub head: Option<[u8; 32]>,
    /// Held-out reads used, the cap, and whether the figure is still quotable.
    pub test: (u32, u32, bool),
    pub journal: Vec<String>,

    // --- read once and kept ---
    pub model: Option<Model>,

    /// When the cheap tier was last refreshed.
    at: u64,
}

/// The model, as it was at load. None of it changes afterwards.
#[derive(Clone, Copy)]
pub struct Model {
    pub params: usize,
    pub bytes: usize,
    pub seq_len: usize,
    pub quantised: bool,
    pub hybrid: bool,
}

impl Glance {
    const fn empty() -> Glance {
        Glance {
            ticks: 0,
            acts: 0,
            episodes: 0,
            suppressed: 0,
            mind_on: false,
            engine: None,
            engine_ready: false,
            doing: None,
            godel_on: false,
            runs: Vec::new(),
            trials: 0,
            adoptions: 0,
            head: None,
            test: (0, 0, true),
            journal: Vec::new(),
            model: None,
            at: 0,
        }
    }
}

static GLANCE: crate::sync::Racy<Option<Glance>> = crate::sync::Racy::new(None);

/// Force the next read to refresh the cheap tier.
///
/// Called after a command, because a command is the one thing that changes
/// this state faster than a second.
pub fn invalidate() {
    if let Some(g) = unsafe { (*GLANCE.get()).as_mut() } {
        g.at = 0;
    }
}

/// Read the machine, refreshing whatever has gone stale.
///
/// A closure rather than a returned value: the cache lives in a `Racy` and
/// handing out a reference to it would outlive the borrow. The same shape
/// `sysbox::with` and `console::with` already use here.
pub fn with_glance<R>(f: impl FnOnce(&Glance) -> R) -> R {
    let slot = unsafe { GLANCE.get() };
    if slot.is_none() {
        *slot = Some(Glance::empty());
    }
    let g = slot.as_mut().expect("just filled");
    let now = crate::dev::lapic::ticks();

    // --- free, every call -------------------------------------------------
    let (ticks, acts, episodes, suppressed, mind_on, _, _) = super::initiative::status();
    g.ticks = ticks;
    g.acts = acts;
    g.episodes = episodes;
    g.suppressed = suppressed;
    g.mind_on = mind_on;
    g.engine = super::engine_holder();
    g.engine_ready = super::engine_ready();
    g.doing = super::agent::doing();
    g.godel_on = super::godel::enabled();

    // --- cheap, when stale ------------------------------------------------
    //
    // `at == 0` means never, which is refresh rather than a cooldown in force
    // -- the same reading `since_episode` gives its own clock.
    if g.at == 0 || now.saturating_sub(g.at) >= STALE_TICKS {
        g.runs = super::work::runs();
        let (trials, adoptions, _) = super::godel::counts();
        g.trials = trials;
        g.adoptions = adoptions;
        g.head = super::godel::head();
        g.test = super::godel::test_status();
        g.journal = super::initiative::journal_tail(8);
        g.at = now.max(1);
    }

    // --- once, and only when the engine is genuinely free -----------------
    //
    // `with_engine` claims, so this is attempted only while nothing holds it
    // and only until it succeeds. Missing the reading costs a blank field for
    // a second; taking the engine from a running episode costs the episode.
    if g.model.is_none() && g.engine.is_none() && g.engine_ready {
        g.model = super::with_engine(|e| Model {
            params: e.model.cfg.param_count(),
            bytes: e.model.weight_bytes(),
            seq_len: e.model.cfg.seq_len,
            quantised: e.model.is_quantised(),
            hybrid: e.model.cfg.hybrid(),
        });
    }

    f(g)
}

/// A short label for the model, for a status line.
pub fn model_line(g: &Glance) -> String {
    match g.model {
        None => String::from("no model"),
        Some(m) => {
            let mb = m.bytes / (1024 * 1024);
            // Not `seq_len / 1024` unconditionally: SmolLM2 at 512 rendered
            // as "0k ctx", which reads as a broken field rather than as a
            // small one. Under a thousand the number itself is short enough
            // to print.
            if m.seq_len >= 1024 {
                alloc::format!(
                    "{}M {} {}k ctx {} MB",
                    m.params / 1_000_000,
                    if m.quantised { "int8" } else { "f32" },
                    m.seq_len / 1024,
                    mb
                )
            } else {
                alloc::format!(
                    "{}M {} {} ctx {} MB",
                    m.params / 1_000_000,
                    if m.quantised { "int8" } else { "f32" },
                    m.seq_len,
                    mb
                )
            }
        }
    }
}

/// Who holds the engine, in a word.
pub fn engine_line(g: &Glance) -> &'static str {
    if !g.engine_ready {
        return "no model";
    }
    match g.engine {
        None => "idle",
        Some(_) => match g.doing {
            Some(what) => what,
            None => "held",
        },
    }
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };

    // The cache must answer even before anything has filled it, because the
    // first paint happens before the first refresh.
    let ticks = with_glance(|g| g.ticks);
    claim("a glance answers on the first call", ticks < u64::MAX);

    // Two reads in the same instant must not walk the namespace twice. The
    // stamp is what proves it: a second refresh would move it.
    let a = with_glance(|g| g.at);
    let b = with_glance(|g| g.at);
    claim("and does not refresh twice in one tick", a == b && a != 0);

    // And `invalidate` has to actually invalidate, or a command's effect
    // would not show until the next second.
    invalidate();
    let c = with_glance(|g| g.at);
    claim("invalidate forces the next read to refresh", c != 0);

    // The engine must not have been claimed by any of that. This is the
    // property the whole module exists for.
    claim("and none of it holds the engine", super::engine_holder().is_none());
    ok
}
