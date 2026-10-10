//! A nervous system: a real animal's connectome, run inside the kernel.
//!
//! The wiring is read off electron micrographs -- C. elegans compiled in,
//! a larger animal staged at `\GLADOS\brain.bin` -- and stepped as leaky
//! integrate-and-fire neurons with the constants Shiu et al. 2024 used for
//! the whole fly brain (`lif.rs`, and `tools/connectome.py` for the oracle
//! it must agree with exactly).
//!
//! **What the OS gives it are senses, and what it gives back is affect.**
//! Being touched (keyboard and mouse), being spoken to (a chat turn), a fault,
//! a failed or corrected action, heat, a good outcome: each is a named sensor
//! population driven with Poisson input for a while (`Sense`). What comes back
//! is read from declared readout populations -- approach, avoidance, arousal
//! -- and folded into two slow numbers, valence and arousal (`Affect`), which
//! is the only thing the rest of the kernel reads.
//!
//! **What this claims and what it does not.** It is real wiring run with
//! published dynamics, and the dynamics are where the honesty runs out: a
//! connectome gives which paths exist and not how strongly they act
//! (Pospisil et al. 2024), so the constants are one set for every neuron,
//! gap junctions are dropped, and sign is a per-neuron guess from its
//! transmitter. It reproduces the textbook avoidance reflex -- the nociceptor
//! drives the reversal command neurons, and `diag brain` requires it to --
//! and it does not feel anything this file can show. Affect here is a
//! readout, named for what it modulates.
//!
//! **It modulates and never grants.** Nothing read from here decides whether
//! an action is allowed. And it is pinned to neutral for every judge
//! (`affect::now` in the judging context, or under `pin_neutral`), or every
//! verdict the machine reaches about itself would depend on its mood when the
//! measurement was taken.

pub mod lif;

use crate::sync::Spin;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use lif::{Connectome, Role, Sim};

/// The worm, compiled in so there is always a nervous system: Cook et al.
/// 2019's hermaphrodite, built by `tools/connectome.py build worm`.
static WORM: &[u8] = include_bytes!("../../assets/brain/worm.bin");

/// The OS's senses, a closed table. Each names a sensor population in the
/// connectome file, so the same sense reaches different cells in different
/// animals without anything here knowing which.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sense {
    /// Hands on the keyboard or the mouse.
    Touch,
    /// Being addressed: a chat turn arrived.
    Voice,
    /// Something went well: an episode met its goal.
    Food,
    /// Something hurt: a fault, a refused or corrected action.
    Noxious,
    /// The package running hot.
    Thermal,
}

impl Sense {
    pub const ALL: [Sense; 5] = [Sense::Touch, Sense::Voice, Sense::Food, Sense::Noxious, Sense::Thermal];

    pub fn name(self) -> &'static str {
        match self {
            Sense::Touch => "touch",
            Sense::Voice => "voice",
            Sense::Food => "food",
            Sense::Noxious => "noxious",
            Sense::Thermal => "thermal",
        }
    }

    pub fn parse(s: &str) -> Option<Sense> {
        Sense::ALL.iter().copied().find(|x| x.name() == s)
    }
}

/// Input rate for a driven sensor, Shiu's 150 Hz.
const RATE_HZ: u64 = 150;
/// Simulated steps per timer tick. 100 is real time at dt = 0.1 ms and a
/// 100 Hz tick; a larger animal is throttled below it rather than allowed to
/// take a core (`init` sets the ratio from the size of what loaded).
static STEPS_PER_TICK: AtomicUsize = AtomicUsize::new(100);
/// Readout window, in steps (100 ms).
const WINDOW: u64 = 1000;

struct Live {
    c: Connectome,
    sim: Sim,
    /// Driven sensors and the step each drive ends at.
    drive: Vec<(usize, u64)>,
    /// Spike counts per readout in the current window.
    window: Vec<u32>,
    window_start: u64,
    fired: Vec<u32>,
    /// Which neuron belongs to which readouts, flattened for the hot loop.
    member: Vec<Vec<u8>>,
    /// Smoothed readout rates in millihertz.
    rate_mhz: Vec<i64>,
    source: &'static str,
}

static LIVE: Spin<Option<Live>> = Spin::new(None);
static ON: AtomicBool = AtomicBool::new(true);
/// Affect, published as atomics so a reader never takes the brain's lock.
/// Both in thousandths: valence -1000..1000, arousal 0..1000.
static VALENCE: AtomicI32 = AtomicI32::new(0);
static AROUSAL: AtomicI32 = AtomicI32::new(0);
static SPIKES_LAST: AtomicU32 = AtomicU32::new(0);
static PINNED: AtomicUsize = AtomicUsize::new(0);

pub fn init(staged: Option<&'static [u8]>) {
    use crate::gfx::console::{self, LTGRAY, LTRED, YELLOW};
    console::set_color(YELLOW);
    crate::kprintln!("\n[brain]");
    console::set_color(LTGRAY);
    let (c, source) = match staged.map(Connectome::parse) {
        Some(Ok(c)) => (c, "staged"),
        Some(Err(why)) => {
            console::set_color(LTRED);
            crate::kprintln!("  {} refused: {} -- falling back to the worm", crate::BRAIN_PATH, why);
            console::set_color(LTGRAY);
            match Connectome::parse(WORM) {
                Ok(c) => (c, "compiled-in worm"),
                Err(why) => {
                    crate::kprintln!("  the compiled-in worm is broken too: {}", why);
                    return;
                }
            }
        }
        None => match Connectome::parse(WORM) {
            Ok(c) => (c, "compiled-in worm"),
            Err(why) => {
                console::set_color(LTRED);
                crate::kprintln!("  the compiled-in worm will not parse: {}", why);
                console::set_color(LTGRAY);
                return;
            }
        },
    };
    // Real time for anything the size of a worm or a larva; a whole fly is
    // throttled to a tenth, which is still a hundred milliseconds of brain a
    // second and keeps the fans off.
    let per_tick = if c.n <= 10_000 { 100 } else { 10 };
    STEPS_PER_TICK.store(per_tick, Ordering::Release);
    crate::kprintln!(
        "  {} neurons, {} connections, {} synapses ({}), {}% of real time",
        c.n,
        c.col.len(),
        c.synapses(),
        source,
        per_tick
    );
    let mut member = alloc::vec![Vec::new(); c.n];
    for (r, ro) in c.readouts.iter().enumerate() {
        for &i in ro.ids.iter() {
            member[i as usize].push(r as u8);
        }
    }
    let nr = c.readouts.len();
    let sim = Sim::new(&c, crate::time::rdtsc() | 1);
    *LIVE.lock() = Some(Live {
        c,
        sim,
        drive: Vec::new(),
        window: alloc::vec![0; nr],
        window_start: 0,
        fired: Vec::new(),
        member,
        rate_mhz: alloc::vec![0; nr],
        source,
    });
    if crate::task::spawn("brain", brain_task).is_none() {
        crate::kprintln!("  no task slot for the brain -- it is loaded and will not run");
    }
}

/// Drive a sense for `ms` milliseconds of simulated time. Extends a drive
/// already running rather than stacking a second one on the same cells.
pub fn stimulate(s: Sense, ms: u64) {
    let mut g = LIVE.lock();
    let Some(l) = g.as_mut() else { return };
    let Some(idx) = l.c.sensor(s.name()) else { return };
    let until = l.sim.t + ms * 1000 / l.c.dt_us;
    match l.drive.iter_mut().find(|d| d.0 == idx) {
        Some(d) => d.1 = d.1.max(until),
        None => l.drive.push((idx, until)),
    }
}

fn brain_task() {
    let mut last = crate::dev::lapic::ticks();
    let mut felt = crate::ai::godbits::felt();
    let mut faults = crate::cpu::recover::caught();
    loop {
        let now = crate::dev::lapic::ticks();
        if now == last {
            crate::task::sleep_until(now + 1);
            continue;
        }
        last = now;

        // Senses the brain polls for itself; the rest arrive by `stimulate`.
        let f = crate::ai::godbits::felt();
        if f != felt {
            felt = f;
            stimulate(Sense::Touch, 50);
        }
        let k = crate::cpu::recover::caught();
        if k != faults {
            faults = k;
            stimulate(Sense::Noxious, 300);
        }
        if now % 100 == 0 {
            if let Some(r) = crate::dev::power::package_temp() {
                if r.throttling || r.headroom < 10 {
                    stimulate(Sense::Thermal, 1000);
                }
            }
        }

        if ON.load(Ordering::Acquire) {
            run(STEPS_PER_TICK.load(Ordering::Acquire));
        }
    }
}

/// Step the live brain `steps` times and fold the readouts.
fn run(steps: usize) {
    let mut g = LIVE.lock();
    let Some(l) = g.as_mut() else { return };
    let mut driven: Vec<(u32, u64)> = Vec::new();
    for _ in 0..steps {
        let t = l.sim.t;
        l.drive.retain(|d| d.1 > t);
        driven.clear();
        for &(s, _) in l.drive.iter() {
            let thr = lif::threshold(RATE_HZ, l.c.dt_us);
            for &i in l.c.sensors[s].ids.iter() {
                driven.push((i, thr));
            }
        }
        let mut fired = core::mem::take(&mut l.fired);
        l.sim.step(&l.c, &driven, &mut fired);
        for &i in fired.iter() {
            for &r in l.member[i as usize].iter() {
                l.window[r as usize] += 1;
            }
        }
        l.fired = fired;
        if l.sim.t - l.window_start >= WINDOW {
            fold(l);
        }
    }
}

/// Close a readout window: rates per neuron, smoothed over about two
/// seconds, then valence and arousal.
fn fold(l: &mut Live) {
    let secs_x1000 = (l.sim.t - l.window_start) * l.c.dt_us / 1000; // ms
    let mut total = 0u32;
    for (r, ro) in l.c.readouts.iter().enumerate() {
        let n = ro.ids.len().max(1) as i64;
        let hz_m = l.window[r] as i64 * 1_000_000 / (n * secs_x1000.max(1) as i64);
        // An exponential average with a time constant of twenty windows.
        l.rate_mhz[r] += (hz_m - l.rate_mhz[r]) / 20;
        total += l.window[r];
        l.window[r] = 0;
    }
    SPIKES_LAST.store(total, Ordering::Release);
    l.window_start = l.sim.t;

    let mut approach = 0i64;
    let mut avoid = 0i64;
    let mut arousal = 0i64;
    for (r, ro) in l.c.readouts.iter().enumerate() {
        match ro.role {
            Role::Approach => approach += l.rate_mhz[r],
            Role::Avoid => avoid += l.rate_mhz[r],
            Role::Arousal => arousal += l.rate_mhz[r],
        }
    }
    // A floor of 5 Hz under the denominator, so two silent populations read
    // as neutral rather than as whichever is a millihertz ahead.
    let v = (approach - avoid) * 1000 / (approach + avoid + 5_000);
    let a = arousal * 1000 / (arousal + 20_000);
    VALENCE.store(v.clamp(-1000, 1000) as i32, Ordering::Release);
    AROUSAL.store(a.clamp(0, 1000) as i32, Ordering::Release);
}

/// What the rest of the kernel reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Affect {
    /// -1000 (avoidance) .. 1000 (approach).
    pub valence: i32,
    /// 0 .. 1000.
    pub arousal: i32,
}

impl Affect {
    pub const NEUTRAL: Affect = Affect { valence: 0, arousal: 0 };

    /// In words, for the system turn: a reading, said as one.
    pub fn words(self) -> &'static str {
        match (self.valence, self.arousal) {
            (v, a) if v < -300 && a > 300 => "agitated",
            (v, _) if v < -300 => "uneasy",
            (v, a) if v > 300 && a > 300 => "eager",
            (v, _) if v > 300 => "content",
            (_, a) if a > 500 => "alert",
            _ => "calm",
        }
    }
}

/// Hold affect at neutral for a scope. Every judge does, so no verdict is a
/// function of the mood it was measured in.
pub struct Pinned(());

impl Drop for Pinned {
    fn drop(&mut self) {
        PINNED.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn pin_neutral() -> Pinned {
    PINNED.fetch_add(1, Ordering::AcqRel);
    Pinned(())
}

pub fn now() -> Affect {
    if PINNED.load(Ordering::Acquire) > 0 || crate::ai::ctx::current() == crate::ai::ctx::Ctx::Judge {
        return Affect::NEUTRAL;
    }
    Affect { valence: VALENCE.load(Ordering::Acquire), arousal: AROUSAL.load(Ordering::Acquire) }
}

pub fn set_on(on: bool) {
    ON.store(on, Ordering::Release);
}

/// `brain`, one block.
pub fn report() -> Vec<String> {
    let mut out = Vec::new();
    let g = LIVE.lock();
    let Some(l) = g.as_ref() else {
        out.push(String::from("  no brain loaded"));
        return out;
    };
    out.push(alloc::format!(
        "  {}: {} neurons, {} connections, {} synapses",
        l.source,
        l.c.n,
        l.c.col.len(),
        l.c.synapses()
    ));
    out.push(alloc::format!(
        "  {} ms simulated, {}% of real time, {}",
        l.sim.t * l.c.dt_us / 1000,
        STEPS_PER_TICK.load(Ordering::Acquire),
        if ON.load(Ordering::Acquire) { "running" } else { "stopped ('brain on')" }
    ));
    for (r, ro) in l.c.readouts.iter().enumerate() {
        let role = match ro.role {
            Role::Approach => "approach",
            Role::Avoid => "avoid",
            Role::Arousal => "arousal",
        };
        out.push(alloc::format!(
            "    {:10} {:8} {:>4}.{} Hz",
            ro.name,
            role,
            l.rate_mhz[r] / 1000,
            (l.rate_mhz[r] % 1000) / 100
        ));
    }
    let driving: Vec<&str> = l.drive.iter().map(|d| l.c.sensors[d.0].name.as_str()).collect();
    if !driving.is_empty() {
        out.push(alloc::format!("  driven now: {}", driving.join(", ")));
    }
    drop(g);
    let a = now();
    out.push(alloc::format!(
        "  affect: valence {:+}, arousal {} -- {}",
        a.valence,
        a.arousal,
        a.words()
    ));
    out.push(String::from("  a readout of real wiring under published dynamics, not a feeling"));
    out
}

// --- diag brain ---------------------------------------------------------------

// connectome.py sim assets/brain/worm.bin --poke noxious --rate 150 --ms 500 --seed 1 --emit
const ORACLE_STEPS: u64 = 5000;
const ORACLE_TOTAL: u64 = 2157;
const ORACLE_DIGEST: u64 = 0x8930_01e0_06ec_52b6;

fn readout_hz(c: &Connectome, sim: &Sim, name: &str) -> u64 {
    let Some(r) = c.readouts.iter().find(|r| r.name == name) else { return 0 };
    let n: u64 = r.ids.iter().map(|&i| sim.counts[i as usize] as u64).sum();
    n * 1_000_000 / (r.ids.len().max(1) as u64 * sim.t * c.dt_us)
}

fn run_fixed(c: &Connectome, poke: &[&str], steps: u64, seed: u64) -> Sim {
    let mut sim = Sim::new(c, seed);
    let mut driven = Vec::new();
    for p in poke {
        if let Some(s) = c.sensor(p) {
            for &i in c.sensors[s].ids.iter() {
                driven.push((i, lif::threshold(RATE_HZ, c.dt_us)));
            }
        }
    }
    let mut fired = Vec::new();
    for _ in 0..steps {
        sim.step(c, &driven, &mut fired);
    }
    sim
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut rows: Vec<(String, bool)> = Vec::new();
    let mut claim = |w: &str, ok: bool| rows.push((String::from(w), ok));

    let mut r = lif::Rng::new(1);
    claim("splitmix64 matches its published first output", r.next() == 0x910A_2DEC_8902_5CC1);
    claim("truncating division agrees with the oracle on negatives", -7i64 / 2 == -3);

    match Connectome::parse(WORM) {
        Err(why) => claim(why, false),
        Ok(c) => {
            claim("the compiled-in worm parses to its last byte, CSR intact", c.n > 250);
            claim("every sense the kernel drives names a population in it", Sense::ALL.iter().all(|s| c.sensor(s.name()).is_some()));
            let silent = run_fixed(&c, &[], 2000, 1);
            claim("with no input nothing fires -- no basal activity, as in Shiu", silent.counts.iter().all(|&x| x == 0));
            let ns = run_fixed(&c, &["noxious"], ORACLE_STEPS, 1);
            let total: u64 = ns.counts.iter().map(|&x| x as u64).sum();
            claim("noxious drive: the same total spike count as the Python oracle", total == ORACLE_TOTAL);
            claim("and the same count for every one of the neurons (FNV digest)", lif::digest(&ns.counts) == ORACLE_DIGEST);
            // The canary, Shiu's argument in worm form: an instrument has to
            // reproduce a known effect before anything read off it is believed.
            // ASH, the polymodal nociceptor, drives the reversal command
            // interneurons (Kaplan & Horvitz 1993; Hilliard et al. 2005).
            //
            // Over, not "twice over": that was this claim's first form, and it
            // held only under looser dynamics than Shiu's. With g frozen while
            // refractory, as model.py has it, the command layer saturates and
            // reverse reads 173 Hz against forward's 130. The direction is the
            // animal's; the margin is the model's, and the model is not the
            // animal.
            let rev = readout_hz(&c, &ns, "reverse");
            let fwd = readout_hz(&c, &ns, "forward");
            claim("canary: the nociceptor drives reversal over forward, as in the animal", rev > fwd && rev > 10);
            let again = run_fixed(&c, &["noxious"], ORACLE_STEPS, 1);
            claim("the same seed gives the same spikes", again.counts == ns.counts);
        }
    }
    {
        let _p = pin_neutral();
        claim("a pinned reading is neutral, whatever the brain is doing", now() == Affect::NEUTRAL);
    }
    {
        let _j = crate::ai::ctx::scoped(crate::ai::ctx::Ctx::Judge);
        claim("the judging context always reads neutral", now() == Affect::NEUTRAL);
    }

    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
