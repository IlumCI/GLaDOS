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
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
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
/// Readout window, in steps (100 ms).
const WINDOW: u64 = 1000;
/// CPU the brain may spend per timer tick, in microseconds: a tenth of a
/// core. The worm uses a sliver of it and runs in real time; the fly, driven,
/// costs about 0.66 ms a step, so it runs as far as the budget reaches and
/// `brain` says what fraction of real time that came to. A brain that ran
/// faster by taking a core would be a brain that turned the fans on.
static BUDGET_US: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1_000);

/// Set how much of a core the brain may use, in percent (1..=100). The
/// default is ten, and the reason for having a default this low is the
/// fans; `brain pace` is for an operator who would rather have the fly.
pub fn set_pace(percent: u64) {
    BUDGET_US.store(percent.clamp(1, 100) * 100, Ordering::Release);
}

pub fn pace() -> u64 {
    BUDGET_US.load(Ordering::Acquire) / 100
}
/// And never more than real time, which is 100 steps a tick.
const MAX_STEPS_PER_TICK: usize = 100;

/// The connectome, immutable after `init`. Leaked rather than locked: the
/// task stepping it and the selftest reading it never write it, so there is
/// nothing to exclude.
static CONN: crate::sync::Racy<Option<&'static Connectome>> = crate::sync::Racy::new(None);
static SOURCE: crate::sync::Racy<&'static str> = crate::sync::Racy::new("none");

/// Stimulation requests, drained by the brain task. A short queue under a
/// short lock: nothing waits on the simulation to post one.
static INBOX: Spin<Vec<(usize, u64)>> = Spin::new(Vec::new());

/// What `brain` shows, published by the task at every readout window.
struct Snapshot {
    t: u64,
    rate_mhz: Vec<i64>,
    driving: Vec<usize>,
    awake: usize,
    /// Simulated steps over wall ticks in the last window, in percent.
    pace: u64,
}

static SNAP: Spin<Snapshot> = Spin::new(Snapshot { t: 0, rate_mhz: Vec::new(), driving: Vec::new(), awake: 0, pace: 0 });

static ON: AtomicBool = AtomicBool::new(true);
/// Affect, published as atomics so a reader never takes a lock.
/// Both in thousandths: valence -1000..1000, arousal 0..1000.
static VALENCE: AtomicI32 = AtomicI32::new(0);
static AROUSAL: AtomicI32 = AtomicI32::new(0);
static PINNED: AtomicUsize = AtomicUsize::new(0);

fn conn() -> Option<&'static Connectome> {
    unsafe { *CONN.get() }
}

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
    crate::kprintln!(
        "  {} neurons, {} connections, {} synapses ({})",
        c.n,
        c.col.len(),
        c.synapses(),
        source
    );
    let c: &'static Connectome = alloc::boxed::Box::leak(alloc::boxed::Box::new(c));
    unsafe {
        *CONN.get() = Some(c);
        *SOURCE.get() = source;
    }
    // Off the shell's core when there is another. Everything the task touches
    // is its own, an atomic, or a `Spin` held for a copy -- which is the audit
    // `task::unpin` asks for, made by construction rather than afterwards.
    let core = crate::smp::online().saturating_sub(1);
    let spawned = crate::task::spawn_on("brain", brain_task, core).or_else(|| crate::task::spawn("brain", brain_task));
    if spawned.is_none() {
        crate::kprintln!("  no task slot for the brain -- it is loaded and will not run");
    } else {
        crate::kprintln!("  running on core {}, on at most {}% of it ('brain pace' to change)", core, pace());
    }
}

/// Drive a sense for `ms` milliseconds of simulated time. Extends a drive
/// already running rather than stacking a second one on the same cells.
pub fn stimulate(s: Sense, ms: u64) {
    let Some(c) = conn() else { return };
    let Some(idx) = c.sensor(s.name()) else { return };
    let mut q = INBOX.lock_irq();
    if q.len() < 64 {
        q.push((idx, ms));
    }
}

fn brain_task() {
    let Some(c) = conn() else { return };
    let mut sim = Sim::new(c, crate::time::rdtsc() | 1);
    let nr = c.readouts.len();
    let mut member = alloc::vec![Vec::<u8>::new(); c.n];
    for (r, ro) in c.readouts.iter().enumerate() {
        for &i in ro.ids.iter() {
            member[i as usize].push(r as u8);
        }
    }
    let mut drive: Vec<(usize, u64)> = Vec::new();
    let mut window = alloc::vec![0u32; nr];
    let mut window_start = 0u64;
    let mut window_ticks = 0u64;
    let mut rate_mhz = alloc::vec![0i64; nr];
    let mut fired = Vec::new();
    let mut driven: Vec<(u32, u64)> = Vec::new();
    let thr = lif::threshold(RATE_HZ, c.dt_us);

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
        window_ticks += 1;

        // Senses the brain polls for itself; the rest arrive in the inbox.
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
        let posted: Vec<(usize, u64)> = core::mem::take(&mut *INBOX.lock_irq());
        for (idx, ms) in posted {
            let until = sim.t + ms * 1000 / c.dt_us;
            match drive.iter_mut().find(|d| d.0 == idx) {
                Some(d) => d.1 = d.1.max(until),
                None => drive.push((idx, until)),
            }
        }

        if !ON.load(Ordering::Acquire) {
            continue;
        }
        let t0 = crate::time::rdtsc();
        let budget = BUDGET_US.load(Ordering::Acquire) * crate::time::tsc_mhz().max(1);
        for _ in 0..MAX_STEPS_PER_TICK {
            let t = sim.t;
            drive.retain(|d| d.1 > t);
            driven.clear();
            for &(s, _) in drive.iter() {
                for &i in c.sensors[s].ids.iter() {
                    driven.push((i, thr));
                }
            }
            sim.step(c, &driven, &mut fired);
            for &i in fired.iter() {
                for &r in member[i as usize].iter() {
                    window[r as usize] += 1;
                }
            }
            if sim.t - window_start >= WINDOW {
                fold(c, &mut window, &mut rate_mhz, sim.t - window_start);
                let pace = WINDOW * 100 / (window_ticks * 100).max(1);
                let mut s = SNAP.lock_irq();
                s.t = sim.t;
                s.rate_mhz.clone_from(&rate_mhz);
                s.driving = drive.iter().map(|d| d.0).collect();
                s.awake = sim.awake();
                s.pace = pace;
                drop(s);
                window_start = sim.t;
                window_ticks = 0;
            }
            if crate::time::rdtsc() - t0 > budget {
                break;
            }
        }
    }
}

/// Close a readout window: rates per neuron, smoothed over about two
/// seconds of brain time, then valence and arousal.
fn fold(c: &Connectome, window: &mut [u32], rate_mhz: &mut [i64], steps: u64) {
    let ms = (steps * c.dt_us / 1000).max(1) as i64;
    for (r, ro) in c.readouts.iter().enumerate() {
        let n = ro.ids.len().max(1) as i64;
        let hz_m = window[r] as i64 * 1_000_000 / (n * ms);
        // An exponential average with a time constant of twenty windows.
        rate_mhz[r] += (hz_m - rate_mhz[r]) / 20;
        window[r] = 0;
    }
    let mut approach = 0i64;
    let mut avoid = 0i64;
    let mut arousal = 0i64;
    for (r, ro) in c.readouts.iter().enumerate() {
        match ro.role {
            Role::Approach => approach += rate_mhz[r],
            Role::Avoid => avoid += rate_mhz[r],
            Role::Arousal => arousal += rate_mhz[r],
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
    let Some(c) = conn() else {
        out.push(String::from("  no brain loaded"));
        return out;
    };
    let source = unsafe { *SOURCE.get() };
    out.push(alloc::format!(
        "  {}: {} neurons, {} connections, {} synapses",
        source,
        c.n,
        c.col.len(),
        c.synapses()
    ));
    let s = SNAP.lock_irq();
    out.push(alloc::format!(
        "  {} ms simulated, {}% of real time lately on {}% of a core, {} neuron(s) awake, {}",
        s.t * c.dt_us / 1000,
        s.pace,
        pace(),
        s.awake,
        if ON.load(Ordering::Acquire) { "running" } else { "stopped ('brain on')" }
    ));
    for (r, ro) in c.readouts.iter().enumerate() {
        let role = match ro.role {
            Role::Approach => "approach",
            Role::Avoid => "avoid",
            Role::Arousal => "arousal",
        };
        let m = s.rate_mhz.get(r).copied().unwrap_or(0);
        out.push(alloc::format!("    {:10} {:8} {:>4}.{} Hz", ro.name, role, m / 1000, (m % 1000) / 100));
    }
    let driving: Vec<&str> = s.driving.iter().map(|&d| c.sensors[d].name.as_str()).collect();
    drop(s);
    if !driving.is_empty() {
        out.push(alloc::format!("  driven now: {}", driving.join(", ")));
    }
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

// connectome.py sim out/fly.bin --poke food --rate 150 --ms 100 --seed 1 --emit
// (fly.bin sha256 30ab9e27..., FlyWire v783 via Shiu's Connectivity_783.parquet)
const FLY_DIGEST: [u8; 8] = [0xc6, 0xd0, 0x49, 0x90, 0xc3, 0x16, 0xcb, 0x1b];
const FLY_STEPS: u64 = 1000;
const FLY_TOTAL: u64 = 8272;
const FLY_DIGEST_COUNTS: u64 = 0x4ee8_4cad_f16d_1ce5;

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
    // The fly, when it is the brain that loaded. Its own oracle constants,
    // keyed by the digest its header carries, so a different build of the
    // file is reported as unchecked rather than compared against numbers
    // that belong to another graph.
    {
        match conn() {
            Some(fc) if fc.digest == FLY_DIGEST => {
                let t0 = crate::time::rdtsc();
                let s = run_fixed(fc, &["food"], FLY_STEPS, 1);
                let us = (crate::time::rdtsc() - t0) / crate::time::tsc_mhz().max(1);
                let total: u64 = s.counts.iter().map(|&x| x as u64).sum();
                claim("fly: 100 ms of sugar drive, the same total as the numpy oracle", total == FLY_TOTAL);
                claim("fly: and the same count for every one of 138,639 neurons", lif::digest(&s.counts) == FLY_DIGEST_COUNTS);
                // Shiu et al. 2024's headline result, reproduced: the labellar
                // sugar neurons drive MN9, the motor neuron of proboscis
                // extension.
                claim("fly canary: sugar drives MN9, the feeding motor neuron (Shiu 2024)", readout_hz(fc, &s, "feeding") > 50);
                crate::kprintln!("    (100 ms of fly brain took {} ms here)", us / 1000);
            }
            Some(fc) if fc.n > 10_000 => claim("a large brain is loaded with no oracle constants for it -- unchecked", false),
            _ => {}
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
