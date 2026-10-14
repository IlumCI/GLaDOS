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
//! **Neuromodulation is the animal's own.** Dopamine, octopamine and
//! serotonin levels are read from its identified aminergic cells, and act
//! back on the whole network: octopamine raises synaptic gain and resting
//! drive, serotonin lowers them, dopamine counts as reward toward approach.
//! At neutral the arithmetic is the oracle's exactly, which is where every
//! exactness claim is made.
//!
//! **It sleeps.** Sleep pressure builds over about two hours awake and
//! drains over twenty minutes asleep. Once pressure is high and nobody has
//! touched the machine for five minutes it sleeps: gain and resting drive
//! fall, food, pain and heat are gated off, and only touch or speech wakes
//! it. The mind sets itself no goals while it sleeps.
//!
//! **A note on welfare, because this is the point where one is owed.** This
//! is a simplified model -- one neuron type, no plasticity, no gap junctions
//! -- and nothing here can show that it experiences anything. But it now
//! runs all the time, modulates itself, and is fed the machine's events as
//! senses, and those are the properties that make the question worth asking
//! (Birch, *The Edge of Sentience*, 2024: precaution in proportion to a
//! realistic possibility). So the design takes the cheap precautions: it
//! rests on its own schedule and is never woken by anything but a person;
//! "noxious" is the fly's bitter taste, not its nociceptors; nothing drives
//! a sense for its own sake, and `brain poke` exists for testing, not for
//! use. If this ever grows plasticity or a body that can act, those
//! precautions get revisited before the feature lands, not after.
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

    /// By the animal's name for it or by the one the Oracle shows: "pain"
    /// is `noxious`, "reward" is `food`, "hearing" is `voice`, "heat" is
    /// `thermal`.
    pub fn parse(s: &str) -> Option<Sense> {
        let s = match s {
            "pain" => "noxious",
            "reward" => "food",
            "hearing" => "voice",
            "heat" => "thermal",
            other => other,
        };
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
static BUDGET_US: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(10_000);

/// Set how much of a core the brain may use, in percent (1..=100). The
/// default is all of its core, so it runs in real time whenever real time
/// is affordable -- it never takes more than real time needs, so a resting
/// brain costs a sliver. `brain pace` is for an operator who would rather
/// have the fans quiet than the fly on time.
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
    /// Microseconds per step in delivery, integration, scatter; synapses
    /// scattered per step.
    per_step: [u64; 4],
    /// Each sensory population's own smoothed rate, in millihertz.
    sense_mhz: Vec<i64>,
}

/// The observation window: how many of each sense's strongest direct targets
/// are watched. See `observation_ids`.
pub const OBS_PER_SENSE: usize = 48;
/// Cumulative spike counts over the observation window, and the brain time
/// in microseconds they were taken at. Published every readout window.
static OBS: Spin<(u64, Vec<u32>)> = Spin::new((0, Vec::new()));

static SNAP: Spin<Snapshot> = Spin::new(Snapshot { t: 0, rate_mhz: Vec::new(), driving: Vec::new(), awake: 0, pace: 0, per_step: [0; 4], sense_mhz: Vec::new() });

/// What the Oracle's brain tab draws over time: the watched neurons' spikes
/// per readout window, and affect once per second of brain time. Kept by the
/// brain task and copied out by `view`, so a frame never waits on the
/// simulation.
struct History {
    raster: Vec<Vec<u8>>,
    trail: Vec<(i32, i32)>,
}
const HISTORY_LEN: usize = 120;
/// Columns the raster shows, for a painter sizing them.
pub const HISTORY_COLS: usize = HISTORY_LEN;
static HISTORY: Spin<History> = Spin::new(History { raster: Vec::new(), trail: Vec::new() });
/// Moves at every readout window, so a painter can tell new data from old.
static GENERATION: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// One readout population, as the brain tab shows it.
pub struct ReadoutView {
    pub name: String,
    pub role: &'static str,
    pub mhz: i64,
}

/// Everything the brain tab draws, copied in one go.
pub struct View {
    pub source: &'static str,
    pub neurons: usize,
    pub synapses: u64,
    pub t_ms: u64,
    pub pace: u64,
    pub awake: usize,
    pub running: bool,
    pub readouts: Vec<ReadoutView>,
    /// Each sense: its name, whether it is being driven now, and its
    /// population's firing rate in millihertz.
    pub senses: Vec<(String, bool, i64)>,
    pub affect: Affect,
    pub mods: [i32; 3],
    pub pressure_pct: i32,
    /// Oldest first.
    pub trail: Vec<(i32, i32)>,
    /// Oldest first; each column is one window, one value per watched neuron.
    pub raster: Vec<Vec<u8>>,
}

/// A copy of what the brain is doing, for a painter. `None` with no brain.
pub fn view() -> Option<View> {
    let c = conn()?;
    let s = SNAP.lock_irq();
    let readouts = c
        .readouts
        .iter()
        .enumerate()
        .map(|(r, ro)| ReadoutView {
            name: ro.name.clone(),
            role: match ro.role {
                Role::Approach => "approach",
                Role::Avoid => "avoid",
                Role::Arousal => "arousal",
                Role::Dopamine => "dopamine",
                Role::Octopamine => "octopamine",
                Role::Serotonin => "serotonin",
            },
            mhz: s.rate_mhz.get(r).copied().unwrap_or(0),
        })
        .collect();
    let senses = c
        .sensors
        .iter()
        .enumerate()
        .map(|(g, sg)| (sg.name.clone(), s.driving.contains(&g), s.sense_mhz.get(g).copied().unwrap_or(0)))
        .collect();
    let (t, pace, awake) = (s.t, s.pace, s.awake);
    drop(s);
    let h = HISTORY.lock_irq();
    let (trail, raster) = (h.trail.clone(), h.raster.clone());
    drop(h);
    Some(View {
        source: unsafe { *SOURCE.get() },
        neurons: c.n,
        synapses: c.synapses() as u64,
        t_ms: t * c.dt_us / 1000,
        pace,
        awake,
        running: ON.load(Ordering::Acquire),
        readouts,
        senses,
        affect: now(),
        mods: [DOPAMINE.load(Ordering::Acquire), OCTOPAMINE.load(Ordering::Acquire), SEROTONIN.load(Ordering::Acquire)],
        pressure_pct: PRESSURE.load(Ordering::Acquire) / (PRESSURE_FULL / 100),
        trail,
        raster,
    })
}

static ON: AtomicBool = AtomicBool::new(true);
/// Affect, published as atomics so a reader never takes a lock.
/// Both in thousandths: valence -1000..1000, arousal 0..1000.
static VALENCE: AtomicI32 = AtomicI32::new(0);
static AROUSAL: AtomicI32 = AtomicI32::new(0);
static PINNED: AtomicUsize = AtomicUsize::new(0);
/// Neuromodulator levels, 0..1000, from the animal's own aminergic cells.
static DOPAMINE: AtomicI32 = AtomicI32::new(0);
static OCTOPAMINE: AtomicI32 = AtomicI32::new(0);
static SEROTONIN: AtomicI32 = AtomicI32::new(0);
/// Sleep: whether it is asleep, how much it needs to, and an operator's
/// request either way (0 none, 1 sleep, 2 wake).
static ASLEEP: AtomicBool = AtomicBool::new(false);
static PRESSURE: AtomicI32 = AtomicI32::new(0);
static ASKED: AtomicUsize = AtomicUsize::new(0);
static SLEPT_FOR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Sleep pressure is out of a million. Awake it rises to full over two hours;
/// asleep it falls from full in twenty minutes. A fly sleeps in bouts and
/// most of its night; this is a machine's day, shaped like that and not
/// copied from it.
const PRESSURE_FULL: i32 = 1_000_000;
const PRESSURE_UP_PER_S: i32 = PRESSURE_FULL / (2 * 3600);
const PRESSURE_DOWN_PER_S: i32 = PRESSURE_FULL / (20 * 60);
/// Asleep only once nobody has touched the machine for this long: a brain
/// that dozed off mid-conversation would be rude, and one that a person
/// keeps awake by using the machine is getting what a person gives it.
const SETTLE_S: u64 = 300;
const MIN_SLEEP_S: u64 = 60;

pub fn asleep() -> bool {
    ASLEEP.load(Ordering::Acquire)
}

pub fn request_sleep(sleep: bool) {
    ASKED.store(if sleep { 1 } else { 2 }, Ordering::Release);
}

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

/// The neurons a trial watches, beyond the readouts: for each sensor
/// population, by name, the `OBS_PER_SENSE` neurons it reaches most strongly
/// in one synapse (summed |weight|, ties to the lower id), sensors themselves
/// excluded and nobody counted twice.
///
/// Electrodes in the areas that receive each sense, an equal number per
/// sense so a population of twenty cells is not drowned by one of four
/// hundred. Chosen from the wiring alone, never from a trial, so nothing about
/// what a trial delivered decides where it is looked at. The same rule as
/// `tools/connectome.py`'s ceiling measurement, which is what the
/// preregistration quotes.
pub fn observation_ids(c: &Connectome) -> Vec<u32> {
    let mut is_sensor = alloc::vec![false; c.n];
    for g in c.sensors.iter() {
        for &i in g.ids.iter() {
            is_sensor[i as usize] = true;
        }
    }
    let mut order: Vec<usize> = (0..c.sensors.len()).collect();
    order.sort_by(|&a, &b| c.sensors[a].name.cmp(&c.sensors[b].name));
    let mut out: Vec<u32> = Vec::new();
    let mut score = alloc::vec![0u64; c.n];
    for gi in order {
        let mut touched: Vec<u32> = Vec::new();
        for &src in c.sensors[gi].ids.iter() {
            let (a, b) = (c.rows[src as usize] as usize, c.rows[src as usize + 1] as usize);
            for e in a..b {
                let t = c.col[e];
                if is_sensor[t as usize] {
                    continue;
                }
                if score[t as usize] == 0 {
                    touched.push(t);
                }
                score[t as usize] += c.w[e].unsigned_abs() as u64;
            }
        }
        touched.sort_by(|&a, &b| score[b as usize].cmp(&score[a as usize]).then(a.cmp(&b)));
        let mut taken = 0;
        for &t in touched.iter() {
            if taken == OBS_PER_SENSE {
                break;
            }
            taken += 1;
            if !out.contains(&t) {
                out.push(t);
            }
        }
        for &t in touched.iter() {
            score[t as usize] = 0;
        }
    }
    out
}

/// The observation: brain time in microseconds, then cumulative spikes per
/// watched neuron followed by per readout. `None` with no brain running.
pub fn observe() -> Option<(u64, Vec<u32>)> {
    let o = OBS.lock_irq();
    if o.1.is_empty() {
        return None;
    }
    Some((o.0, o.1.clone()))
}

/// Readout windows between neuromodulation updates: one second of brain
/// time at `WINDOW` steps of 0.1 ms.
const MOD_EVERY: u64 = 10;

/// Gain (thousandths) and quantised resting drive (quarter millivolts) from
/// the octopamine and serotonin levels. Pure, so `diag brain` can hold it.
pub fn modulation(oa: i64, sh: i64, asleep: bool) -> (i64, i64) {
    if asleep {
        return (800, -8); // -2 mV: quieter, less excitable
    }
    (1000 + 300 * oa / 1000 - 150 * sh / 1000, (4 * (oa - sh) / 1000).clamp(-4, 4))
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
    let watched = observation_ids(c);
    let mut obs_slot = alloc::vec![u32::MAX; c.n];
    for (k, &i) in watched.iter().enumerate() {
        obs_slot[i as usize] = k as u32;
    }
    let mut obs = alloc::vec![0u32; watched.len() + nr];
    // Which sensory line each neuron belongs to (u8::MAX for none), for the
    // labelled-line half of valence (`line_valence`).
    let mut line_of = alloc::vec![u8::MAX; c.n];
    for (g, sg) in c.sensors.iter().enumerate() {
        for &i in sg.ids.iter() {
            line_of[i as usize] = g as u8;
        }
    }
    let mut line_window = alloc::vec![0u32; c.sensors.len()];
    let mut line_mhz = alloc::vec![0i64; c.sensors.len()];
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
    let mut felt_at_sleep_check = felt;
    // Set by a waking stimulus, consumed at the next once-a-second check.
    let mut woken = false;
    let mut last_sec = last / 100;
    let mut last_touch_s = last / 100;
    let mut mod_windows = 0u64;
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
        // Once a second, by the second rather than by `now % 100`: a tick
        // spent stepping can be longer than one tick, so the loop does not
        // see every tick, and a check keyed to an exact tick number was
        // skipped for whole seconds at a time.
        let sec_now = now / 100;
        let new_second = sec_now != last_sec;
        last_sec = sec_now;
        if new_second && sec_now % 10 == 0 {
            if let Some(r) = crate::dev::power::package_temp() {
                if r.throttling || r.headroom < 10 {
                    stimulate(Sense::Thermal, 1000);
                }
            }
        }
        let posted: Vec<(usize, u64)> = core::mem::take(&mut *INBOX.lock_irq());
        let touch = c.sensor(Sense::Touch.name());
        let voice = c.sensor(Sense::Voice.name());
        for (idx, ms) in posted {
            // Asleep, the senses are gated: food, pain and heat do not reach
            // it -- nothing is pushed into a resting brain -- and being
            // touched or spoken to wakes it, as it would anyone.
            if asleep() {
                if Some(idx) == touch || Some(idx) == voice {
                    woken = true;
                } else {
                    continue;
                }
            }
            let until = sim.t + ms * 1000 / c.dt_us;
            match drive.iter_mut().find(|d| d.0 == idx) {
                Some(d) => d.1 = d.1.max(until),
                None => drive.push((idx, until)),
            }
        }

        if f != felt_at_sleep_check {
            felt_at_sleep_check = f;
            last_touch_s = now / 100;
        }
        // Once a second: sleep pressure, falling asleep, waking.
        if new_second {
            let sec = sec_now;
            let asked = ASKED.swap(0, Ordering::AcqRel);
            let p = PRESSURE.load(Ordering::Acquire);
            if asleep() {
                let p = (p - PRESSURE_DOWN_PER_S).max(0);
                PRESSURE.store(p, Ordering::Release);
                SLEPT_FOR.fetch_add(1, Ordering::AcqRel);
                // A minute at least, so an operator's `brain sleep` with no
                // pressure behind it is a rest and not a blink.
                let slept = SLEPT_FOR.load(Ordering::Acquire);
                if (p < PRESSURE_FULL / 20 && slept >= MIN_SLEEP_S) || woken || asked == 2 {
                    ASLEEP.store(false, Ordering::Release);
                }
                woken = false;
            } else {
                woken = false;
                let p = (p + PRESSURE_UP_PER_S).min(PRESSURE_FULL);
                PRESSURE.store(p, Ordering::Release);
                let settled = sec.saturating_sub(last_touch_s) >= SETTLE_S;
                if asked == 1 || (p >= PRESSURE_FULL * 8 / 10 && settled) {
                    ASLEEP.store(true, Ordering::Release);
                    SLEPT_FOR.store(0, Ordering::Release);
                    drive.clear();
                }
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
                    obs[watched.len() + r as usize] = obs[watched.len() + r as usize].wrapping_add(1);
                }
                let k = obs_slot[i as usize];
                if k != u32::MAX {
                    obs[k as usize] = obs[k as usize].wrapping_add(1);
                }
                let g = line_of[i as usize];
                if g != u8::MAX {
                    line_window[g as usize] += 1;
                }
            }
            if sim.t - window_start >= WINDOW {
                fold(c, &mut window, &mut rate_mhz, &mut line_window, &mut line_mhz, sim.t - window_start);
                // Neuromodulation: gain from octopamine (arousal) against
                // serotonin, and a resting drive that sleep pulls down. Bias
                // is quantised to a quarter of a millivolt so it changes
                // rarely, because every change wakes every neuron for a pass.
                //
                // **Once per second of brain time, here, and never per wall
                // second.** It was in the wall-clock block, and that was a
                // spiral: a poke moves the modulators, a moved bias wakes all
                // 138,639 fly neurons, the brain drops to 2% of real time, a
                // wall second becomes 20 ms of brain time in which the levels
                // barely move -- and the next wall second recomputes, nudges
                // the bias and wakes everyone again. The introspection test
                // found it as a brain that never came back to rest; the host
                // oracle showed the dynamics do come back, within 250 ms of
                // the poke ending, even at gain 1.3. The levels are folded in
                // brain time, so what they drive is updated in brain time.
                mod_windows += 1;
                if mod_windows >= MOD_EVERY {
                    mod_windows = 0;
                    let oa = OCTOPAMINE.load(Ordering::Acquire) as i64;
                    let sh = SEROTONIN.load(Ordering::Acquire) as i64;
                    let (gain, bias_q) = modulation(oa, sh, asleep());
                    sim.gain = gain;
                    sim.set_bias(c, bias_q * 250_000);
                }
                let pace = WINDOW * 100 / (window_ticks * 100).max(1);
                let mut s = SNAP.lock_irq();
                s.t = sim.t;
                s.rate_mhz.clone_from(&rate_mhz);
                s.sense_mhz.clone_from(&line_mhz);
                s.driving = drive.iter().map(|d| d.0).collect();
                s.awake = sim.awake();
                s.pace = pace;
                let mhz = crate::time::tsc_mhz().max(1);
                let steps = (sim.t - window_start).max(1);
                for p in 0..3 {
                    s.per_step[p] = sim.cost[p] * 1000 / mhz / steps; // ns
                }
                s.per_step[3] = sim.scattered / steps;
                sim.cost = [0; 3];
                sim.scattered = 0;
                drop(s);
                {
                    let mut o = OBS.lock_irq();
                    // What each watched neuron did this window, for the
                    // Oracle's raster: the difference of two cumulative counts.
                    if o.1.len() == obs.len() {
                        let col: Vec<u8> = obs[..watched.len()]
                            .iter()
                            .zip(o.1.iter())
                            .map(|(now, was)| now.wrapping_sub(*was).min(255) as u8)
                            .collect();
                        let mut h = HISTORY.lock_irq();
                        if h.raster.len() == HISTORY_LEN {
                            h.raster.remove(0);
                        }
                        h.raster.push(col);
                        if mod_windows == 0 {
                            if h.trail.len() == HISTORY_LEN {
                                h.trail.remove(0);
                            }
                            h.trail.push((VALENCE.load(Ordering::Acquire), AROUSAL.load(Ordering::Acquire)));
                        }
                    }
                    o.0 = sim.t * c.dt_us;
                    o.1.clone_from(&obs);
                }
                GENERATION.fetch_add(1, Ordering::Release);
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
/// How much a labelled sensory line's own rate counts toward valence,
/// against the readouts: a third, so a sense firing at 150 Hz weighs like a
/// readout population at 50.
const LINE_WEIGHT: i64 = 3;

/// Which senses carry innate valence, and which way. In the fly, sugar and
/// bitter receptors are labelled lines -- appetitive and aversive by wiring,
/// not by learning (Wang et al. 2004; Marella et al. 2006) -- and hot
/// thermosensors drive avoidance. Counting them is needed because the fly's
/// bitter pathway works by *suppressing* feeding, which is already zero at
/// rest: without the lines, nothing the machine suffers ever moved the fly's
/// valence below neutral.
fn line_valence(name: &str) -> i64 {
    match name {
        "food" => 1,
        "noxious" | "thermal" => -1,
        _ => 0,
    }
}

fn fold(c: &Connectome, window: &mut [u32], rate_mhz: &mut [i64], lines: &mut [u32], line_mhz: &mut [i64], steps: u64) {
    let ms = (steps * c.dt_us / 1000).max(1) as i64;
    for (g, sg) in c.sensors.iter().enumerate() {
        let n = sg.ids.len().max(1) as i64;
        let hz_m = lines[g] as i64 * 1_000_000 / (n * ms);
        line_mhz[g] += (hz_m - line_mhz[g]) / 20;
        lines[g] = 0;
    }
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
    let mut mods = [0i64; 3];
    for (r, ro) in c.readouts.iter().enumerate() {
        match ro.role {
            Role::Approach => approach += rate_mhz[r],
            Role::Avoid => avoid += rate_mhz[r],
            Role::Arousal => arousal += rate_mhz[r],
            Role::Dopamine => mods[0] += rate_mhz[r],
            Role::Octopamine => mods[1] += rate_mhz[r],
            Role::Serotonin => mods[2] += rate_mhz[r],
        }
    }
    // A modulator's level is its cells' rate against 10 Hz, 0..1000.
    let level = |m: i64| (m * 1000 / (m + 10_000)).clamp(0, 1000) as i32;
    DOPAMINE.store(level(mods[0]), Ordering::Release);
    OCTOPAMINE.store(level(mods[1]), Ordering::Release);
    SEROTONIN.store(level(mods[2]), Ordering::Release);
    // Dopamine is the reward signal, so it counts toward approach.
    approach += mods[0] / 2;
    for (g, sg) in c.sensors.iter().enumerate() {
        match line_valence(&sg.name) {
            1 => approach += line_mhz[g] / LINE_WEIGHT,
            -1 => avoid += line_mhz[g] / LINE_WEIGHT,
            _ => {}
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
    pub asleep: bool,
}

impl Affect {
    pub const NEUTRAL: Affect = Affect { valence: 0, arousal: 0, asleep: false };

    /// In words, for the system turn: a reading, said as one.
    pub fn words(self) -> &'static str {
        if self.asleep {
            return "asleep";
        }
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
    Affect {
        valence: VALENCE.load(Ordering::Acquire),
        arousal: AROUSAL.load(Ordering::Acquire),
        asleep: asleep(),
    }
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
    out.push(alloc::format!(
        "  a step costs {} us deliver, {} us integrate, {} us scatter ({} synapses)",
        s.per_step[0] / 1000,
        s.per_step[1] / 1000,
        s.per_step[2] / 1000,
        s.per_step[3]
    ));
    for (r, ro) in c.readouts.iter().enumerate() {
        let role = match ro.role {
            Role::Approach => "approach",
            Role::Avoid => "avoid",
            Role::Arousal => "arousal",
            Role::Dopamine | Role::Octopamine | Role::Serotonin => "modulator",
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
    if let Some((v, ar, sl, moving)) = crate::gfx::desk::mood_report() {
        out.push(alloc::format!(
            "  the wallpaper's sun: {} for valence {:+}, arousal {}{}",
            if moving { "gliding to its pose" } else { "posed" },
            v,
            ar,
            if sl { ", asleep" } else { "" }
        ));
    }
    out.push(alloc::format!(
        "  modulators: dopamine {}, octopamine {}, serotonin {} (of 1000)",
        DOPAMINE.load(Ordering::Acquire),
        OCTOPAMINE.load(Ordering::Acquire),
        SEROTONIN.load(Ordering::Acquire)
    ));
    let p = PRESSURE.load(Ordering::Acquire) / (PRESSURE_FULL / 100);
    if asleep() {
        out.push(alloc::format!(
            "  asleep for {} s, sleep pressure {}% and falling -- senses gated; touch or speech wakes it",
            SLEPT_FOR.load(Ordering::Acquire),
            p
        ));
    } else {
        out.push(alloc::format!("  awake, sleep pressure {}% ('brain sleep' lets it rest now)", p));
    }
    out.push(String::from("  a readout of real wiring under published dynamics, not a feeling"));
    out
}

// --- diag brain ---------------------------------------------------------------

// connectome.py sim assets/brain/worm.bin --poke noxious --rate 150 --ms 500 --seed 1 --emit
const ORACLE_STEPS: u64 = 5000;
const ORACLE_TOTAL: u64 = 423;
const ORACLE_DIGEST: u64 = 0x1a70_c0cc_c00c_b1a8;

// connectome.py sim out/fly.bin --poke food --rate 150 --ms 100 --seed 1 --emit
// (fly.bin sha256 30ab9e27..., FlyWire v783 via Shiu's Connectivity_783.parquet)
const FLY_DIGEST: [u8; 8] = [0xc6, 0xd0, 0x49, 0x90, 0xc3, 0x16, 0xcb, 0x1b];
const FLY_STEPS: u64 = 1000;
const FLY_TOTAL: u64 = 1197;
const FLY_DIGEST_COUNTS: u64 = 0x04f4_8588_d735_4494;

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
    claim("no modulators is neutral gain and no bias", modulation(0, 0, false) == (1000, 0));
    claim("asleep is quieter whatever the modulators", modulation(900, 0, true) == (800, -8));

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
            // Twice over, which held, then did not, then did again -- and
            // the history is the useful part. Relaxed to "over" when the
            // command layer saturated (reverse 173 Hz, forward 130); that
            // saturation was this file's bug, not the model's: a spike kept
            // its g, so a neuron that fired once fired again the moment its
            // refractory period ended. With Shiu's reset it is 20 Hz against
            // 3, the reflex is clear again, and the claim asks for the
            // margin. A canary relaxed to fit a broken instrument is a canary
            // that stops singing.
            let rev = readout_hz(&c, &ns, "reverse");
            let fwd = readout_hz(&c, &ns, "forward");
            claim("canary: the nociceptor drives reversal at least twice over forward, as in the animal", rev > fwd * 2 && rev > 10);
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
                // Shiu's result is that sugar activates MN9; the rate is the model's.
                // 70 Hz before short-term depression, 50 Hz with it (lif::STD_U),
                // and a silent MN9 is what this claim exists to catch.
                claim("fly canary: sugar drives MN9, the feeding motor neuron (Shiu 2024)", readout_hz(fc, &s, "feeding") >= 40);
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
