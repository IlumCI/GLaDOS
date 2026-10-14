//! A connectome, loaded, and stepped with leaky integrate-and-fire neurons.
//!
//! The arithmetic is `tools/connectome.py`'s `simulate`, line for line, in
//! the same integers: nanovolts, steps of 0.1 ms, division that truncates
//! toward zero. That is not a stylistic choice. `diag brain` requires this
//! file's spike counts to equal the Python oracle's with `==`, the way
//! `differ.rs` requires two routes through one program to agree bit for bit,
//! and a float simulation would agree to "within noise" -- which is where a
//! wrong delay or a sign read backwards hides.
//!
//! One step, in exactly this order (the oracle's docstring says the same):
//!   1. deliver the delay ring's slot for this step into g
//!   2. Poisson input to driven sensors, in drive order, one draw each: a
//!      kick to v (`PoissonInput(target_var='v')`)
//!   3. a refractory neuron is held at reset with g frozen; any other
//!      integrates v and then decays g
//!   4. a neuron strictly over threshold spikes: v to reset and g to zero
//!      (`eq_rst`), refractory unless it is being driven (model.py gives
//!      Poisson targets `rfc = 0`), and its row's weights enter the ring
//!      `delay` steps ahead
//!
//! The first version had three of those wrong and the fly ran away: with g
//! kept across a spike, every neuron that fired once fired again the moment
//! its refractory period ended, and a second of sugar became three million
//! spikes. Shiu's notebook reports about 13,000; this reads 15,298.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

pub const MAGIC: &[u8; 8] = b"GLADOSB1";
const HEADER: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Approach,
    Avoid,
    Arousal,
    /// The animal's own aminergic cells, read as neuromodulator levels.
    Dopamine,
    Octopamine,
    Serotonin,
}

pub struct Group {
    pub name: String,
    pub ids: Vec<u32>,
}

pub struct Readout {
    pub name: String,
    pub role: Role,
    pub ids: Vec<u32>,
}

/// A GLADOSB1 file, parsed. The arrays are copied out because the compiled-in
/// worm is `include_bytes!` data with no alignment promise, and the fly is
/// read once at boot -- neither is on a path where a copy costs anything.
pub struct Connectome {
    pub n: usize,
    pub rows: Vec<u32>,
    pub col: Vec<u32>,
    pub w: Vec<i16>,
    pub v0: i64,
    pub vth: i64,
    pub vreset: i64,
    pub tau_m: i64,
    pub tau_syn: i64,
    pub refr: u16,
    pub delay: usize,
    pub w_syn: i64,
    pub w_poi: i64,
    pub dt_us: u64,
    pub sensors: Vec<Group>,
    pub readouts: Vec<Readout>,
    pub labels: Vec<[u8; 8]>,
    pub digest: [u8; 8],
}

fn u32_at(d: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(p..p + 4)?.try_into().ok()?))
}

fn i32_at(d: &[u8], p: usize) -> Option<i32> {
    Some(i32::from_le_bytes(d.get(p..p + 4)?.try_into().ok()?))
}

fn name_at(d: &[u8], p: usize) -> Option<String> {
    let raw = d.get(p..p + 16)?;
    let end = raw.iter().position(|&b| b == 0).unwrap_or(16);
    Some(String::from(core::str::from_utf8(&raw[..end]).ok()?))
}

impl Connectome {
    /// Parse by walking, never seeking, and refuse a file whose walk does not
    /// land on its last byte -- `v4.py`'s bargain, for its reason: a reader
    /// and writer that disagree about one length read valid integers forever
    /// after it.
    pub fn parse(d: &[u8]) -> Result<Connectome, &'static str> {
        if d.len() < HEADER || &d[..8] != MAGIC {
            return Err("not a GLADOSB1 file");
        }
        let g = |p| u32_at(d, p).ok_or("truncated header");
        if g(8)? != 1 {
            return Err("a GLADOSB1 version this kernel does not read");
        }
        let n = g(12)? as usize;
        let e = g(16)? as usize;
        let v0 = i32_at(d, 20).ok_or("truncated")? as i64;
        let vth = i32_at(d, 24).ok_or("truncated")? as i64;
        let vreset = i32_at(d, 28).ok_or("truncated")? as i64;
        let tau_m = g(32)? as i64;
        let tau_syn = g(36)? as i64;
        let refr = g(40)? as u16;
        let delay = g(44)? as usize;
        let w_syn = i32_at(d, 48).ok_or("truncated")? as i64;
        let w_poi = i32_at(d, 52).ok_or("truncated")? as i64;
        let ns = g(56)? as usize;
        let nr = g(60)? as usize;
        let mut digest = [0u8; 8];
        digest.copy_from_slice(&d[64..72]);
        let dt_us = g(76)? as u64;
        if tau_m == 0 || tau_syn == 0 || delay == 0 || dt_us == 0 {
            return Err("a time constant of zero");
        }

        let mut p = HEADER;
        let mut rows = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            rows.push(u32_at(d, p).ok_or("truncated rows")?);
            p += 4;
        }
        let mut col = Vec::with_capacity(e);
        for _ in 0..e {
            col.push(u32_at(d, p).ok_or("truncated targets")?);
            p += 4;
        }
        let mut w = Vec::with_capacity(e);
        for _ in 0..e {
            let b = d.get(p..p + 2).ok_or("truncated weights")?;
            w.push(i16::from_le_bytes([b[0], b[1]]));
            p += 2;
        }
        p = (p + 3) & !3;
        let mut sensors = Vec::with_capacity(ns);
        for _ in 0..ns {
            let name = name_at(d, p).ok_or("truncated sensor")?;
            p += 16;
            let k = u32_at(d, p).ok_or("truncated sensor")? as usize;
            p += 4;
            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                ids.push(u32_at(d, p).ok_or("truncated sensor")?);
                p += 4;
            }
            sensors.push(Group { name, ids });
        }
        let mut readouts = Vec::with_capacity(nr);
        for _ in 0..nr {
            let name = name_at(d, p).ok_or("truncated readout")?;
            p += 16;
            let role = match u32_at(d, p).ok_or("truncated readout")? {
                0 => Role::Approach,
                1 => Role::Avoid,
                2 => Role::Arousal,
                3 => Role::Dopamine,
                4 => Role::Octopamine,
                5 => Role::Serotonin,
                _ => return Err("a readout role this kernel does not know"),
            };
            let k = u32_at(d, p + 4).ok_or("truncated readout")? as usize;
            p += 8;
            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                ids.push(u32_at(d, p).ok_or("truncated readout")?);
                p += 4;
            }
            readouts.push(Readout { name, role, ids });
        }
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let mut l = [0u8; 8];
            l.copy_from_slice(d.get(p..p + 8).ok_or("truncated labels")?);
            labels.push(l);
            p += 8;
        }
        if p != d.len() {
            return Err("the walk did not land on the last byte");
        }
        // CSR invariants: monotone, ending at the edge count, every target in
        // range, every named neuron in range. A file failing any of these
        // would index out of bounds somewhere inside a step.
        if rows[n] as usize != e || rows.windows(2).any(|r| r[0] > r[1]) {
            return Err("row pointers are not a CSR index");
        }
        if col.iter().any(|&c| c as usize >= n) {
            return Err("a synapse targets a neuron that does not exist");
        }
        let named = sensors.iter().flat_map(|s| s.ids.iter()).chain(readouts.iter().flat_map(|r| r.ids.iter()));
        if named.into_iter().any(|&i| i as usize >= n) {
            return Err("a sensor or readout names a neuron that does not exist");
        }
        Ok(Connectome {
            n, rows, col, w, v0, vth, vreset, tau_m, tau_syn, refr, delay, w_syn, w_poi, dt_us,
            sensors, readouts, labels, digest,
        })
    }

    pub fn sensor(&self, name: &str) -> Option<usize> {
        self.sensors.iter().position(|s| s.name == name)
    }

    pub fn synapses(&self) -> u64 {
        self.w.iter().map(|w| w.unsigned_abs() as u64).sum()
    }
}

/// Truncating division, which is what Rust's `/` on `i64` already is. Named
/// so a reader comparing this file with the oracle's `tdiv` finds the pair.
#[inline]
fn tdiv(a: i64, b: i64) -> i64 {
    a / b
}

/// splitmix64, the oracle's generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A Poisson input to one neuron, as a threshold on a 64-bit draw: the
/// oracle's `(rate * dt * 2^32 / 1e6) << 32`, so a rate maps to the same
/// threshold on both sides.
pub fn threshold(rate_hz: u64, dt_us: u64) -> u64 {
    ((rate_hz * dt_us * (1u64 << 32)) / 1_000_000) << 32
}

/// Slices of the neuron range the dense pass is split into.
const PARTS: usize = 64;

/// Short-term synaptic depression (Tsodyks & Markram 1997), in
/// `tools/connectome.py` as STD_U and STD_TAU_MS and identical there. Shiu's
/// model has none, and without it the fly has a self-sustaining state: a
/// heat or odour input ignites about 10,000 neurons that fire for good,
/// anchored in a 477-neuron antennal-lobe core (design/attractor.md). With
/// U = 0.1 and 1 s recovery, the state ends within 100 ms of its input and
/// sugar still drives MN9 at 50 Hz (70 without).
pub const STD_U: i64 = 100_000;
pub const STD_TAU_MS: i64 = 1000;
pub const PPM: i64 = 1_000_000;

/// One neuron's step: Shiu's update, then whether it spiked and whether it is
/// now at a fixed point. The one implementation the serial walk, the list
/// walk and the parallel slices all call, so they cannot drift apart.
#[inline(always)]
fn neuron<const TM: i64, const TS: i64>(
    c: &Connectome,
    bias: i64,
    v: &mut i64,
    g: &mut i64,
    r: &mut u16,
    count: &mut u32,
    nore: bool,
) -> (bool, bool) {
    let dm = |x: i64| if TM != 0 { x / TM } else { x / c.tau_m };
    let ds = |x: i64| if TS != 0 { x / TS } else { x / c.tau_syn };
    let mut spiked = false;
    if *r > 0 {
        // Shiu's model.py marks both equations `(unless refractory)`: v is
        // held and g does not decay.
        *r -= 1;
        *v = c.vreset;
    } else {
        *v += dm(c.v0 - *v + *g + bias);
        *g -= ds(*g);
        if *v > c.vth {
            *v = c.vreset;
            *g = 0;
            *r = if nore { 0 } else { c.refr };
            *count = count.wrapping_add(1);
            spiked = true;
        }
    }
    // A fixed point: the dense loop would leave this neuron exactly as it is
    // until input lands, so skipping it is exact. "Would this step change
    // anything" and not "is g zero" -- truncating decay leaves a residue under
    // tau_syn that never reaches zero, and the first version kept 125,772 fly
    // neurons awake forever on it.
    let inert = *r == 0 && ds(*g) == 0 && dm(c.v0 - *v + *g + bias) == 0 && *v <= c.vth;
    (spiked, inert)
}

pub struct Sim {
    pub v: Vec<i64>,
    pub g: Vec<i64>,
    pub refr: Vec<u16>,
    ring: Vec<i64>,
    /// Which neurons each ring slot holds anything for, so delivery visits
    /// those and not all of them.
    touched: Vec<Vec<u32>>,
    slots: usize,
    /// Neurons that can change this step. Everything else is at a fixed
    /// point of the arithmetic -- see `inert` -- and skipping it is exact.
    active: Vec<u32>,
    /// How many neurons are awake, whichever way they are being tracked.
    awake: usize,
    /// Walking the awake list or scanning everything in order. See `integrate`.
    dense: bool,
    on: Vec<bool>,
    /// Driven this step, and so never refractory -- set and cleared around
    /// `integrate` from the drive list.
    nore: Vec<bool>,
    pub t: u64,
    /// TSC cycles spent in delivery, integration and scatter, for `brain`.
    pub cost: [u64; 3],
    /// Neuromodulation, as two global terms. `gain` scales every synapse, in
    /// thousandths; `bias` shifts every neuron's resting drive, in
    /// nanovolts. Neutral is 1000 and 0, where the arithmetic is the
    /// oracle's exactly -- `w * w_syn * 1000 / 1000` is `w * w_syn` in
    /// integers -- so every exactness claim is made at neutral and still
    /// means what it meant.
    pub gain: i64,
    bias: i64,
    /// Synapses scattered, for the same.
    pub scattered: u64,
    pub rng: Rng,
    pub counts: Vec<u32>,
    /// The parallel dense pass's per-slice fired lists, kept between steps.
    parts: Vec<Vec<u32>>,
    /// Short-term depression: resources available per presynaptic neuron,
    /// in parts per million, and the step they were last brought up to date
    /// (`resources`). Sensor neurons are exempt.
    xr: Vec<i64>,
    xr_t: Vec<u64>,
    exempt: Vec<bool>,
    tau_rec: i64,
}

impl Sim {
    pub fn new(c: &Connectome, seed: u64) -> Sim {
        let slots = c.delay + 1;
        Sim {
            v: vec![c.v0; c.n],
            g: vec![0; c.n],
            refr: vec![0; c.n],
            ring: vec![0; slots * c.n],
            touched: (0..slots).map(|_| Vec::new()).collect(),
            slots,
            active: Vec::new(),
            awake: 0,
            dense: false,
            on: vec![false; c.n],
            nore: vec![false; c.n],
            t: 0,
            cost: [0; 3],
            gain: 1000,
            bias: 0,
            scattered: 0,
            rng: Rng::new(seed),
            counts: vec![0; c.n],
            parts: Vec::new(),
            xr: vec![PPM; c.n],
            xr_t: vec![0; c.n],
            exempt: {
                let mut e = vec![false; c.n];
                for g in c.sensors.iter() {
                    for &i in g.ids.iter() {
                        e[i as usize] = true;
                    }
                }
                e
            },
            tau_rec: (STD_TAU_MS * 1000 / c.dt_us as i64).max(1),
        }
    }

    /// Neuron `i`'s resources at this step, with every recovery since its
    /// last spike applied. The oracle recovers every neuron at the end of
    /// every step, `x += (1e6 - x) / tau_rec`; this replays exactly that
    /// recurrence for the steps that passed, stopping where it stops moving,
    /// so nothing is touched for a neuron that is not firing.
    #[inline]
    fn resources(&mut self, i: usize) -> i64 {
        let mut x = self.xr[i];
        let mut k = self.t - self.xr_t[i];
        while k > 0 && PPM - x >= self.tau_rec {
            x += (PPM - x) / self.tau_rec;
            k -= 1;
        }
        self.xr[i] = x;
        self.xr_t[i] = self.t;
        x
    }

    #[inline]
    fn wake(&mut self, i: usize) {
        if !self.on[i] {
            self.on[i] = true;
            self.awake += 1;
            if !self.dense {
                self.active.push(i as u32);
            }
        }
    }

    /// One step. `driven` is (neuron, threshold) in drive order; `fired` is
    /// cleared and filled with this step's spikes, in no particular order --
    /// what is done with them is a sum, and integer addition does not care.
    pub fn step(&mut self, c: &Connectome, driven: &[(u32, u64)], fired: &mut Vec<u32>) {
        let n = c.n;
        let slot = self.t as usize % self.slots;
        let base = slot * n;
        let c0 = crate::time::rdtsc();
        // Delivery. With most of the brain awake it happens inside the dense
        // pass instead, each slice reading its own stretch of this slot in
        // order -- a walk of the touched list there is a random walk over a
        // megabyte, and it measured as large as the integration itself.
        // Exact either way: delivery adds to g, the Poisson kicks below add
        // to v, and nothing reads one before the other is done.
        let due = core::mem::take(&mut self.touched[slot]);
        if !self.dense {
            for &i in due.iter() {
                let i = i as usize;
                let x = self.ring[base + i];
                if x != 0 {
                    self.g[i] += x;
                    self.ring[base + i] = 0;
                    self.wake(i);
                }
            }
        }
        let mut due = due;
        due.clear();
        self.touched[slot] = due;
        for &(i, thr) in driven {
            self.nore[i as usize] = true;
            if self.rng.next() < thr {
                self.v[i as usize] += c.w_poi;
                self.wake(i as usize);
            }
        }
        let c1 = crate::time::rdtsc();
        fired.clear();
        // Shiu's constants get a step the compiler can divide in: `/ 200` by
        // a constant is a multiply and a shift that truncates exactly as the
        // instruction does, where `/ c.tau_m` is an `idiv` of forty-odd
        // cycles -- five of them per neuron per step, which measured 6 ms of
        // a 7.6 ms step on the fly. Any other constants take the general path,
        // which computes the same thing slowly.
        if c.tau_m == 200 && c.tau_syn == 50 {
            self.integrate::<200, 50>(c, base, fired);
        } else {
            self.integrate::<0, 0>(c, base, fired);
        }
        for &(i, _) in driven {
            self.nore[i as usize] = false;
        }
        let c2 = crate::time::rdtsc();
        if !fired.is_empty() {
            let os = (self.t as usize + c.delay) % self.slots;
            let out = os * n;
            for &i in fired.iter() {
                let iu = i as usize;
                let x = self.resources(iu);
                let (a, b) = (c.rows[iu] as usize, c.rows[iu + 1] as usize);
                for k in a..b {
                    let j = c.col[k] as usize;
                    if self.ring[out + j] == 0 {
                        self.touched[os].push(j as u32);
                    }
                    let base = c.w[k] as i64 * c.w_syn * self.gain / 1000;
                    self.ring[out + j] += if x == PPM { base } else { base * x / PPM };
                }
                if !self.exempt[iu] {
                    self.xr[iu] -= x * STD_U / PPM;
                }
                self.scattered += (b - a) as u64;
            }
        }
        let c3 = crate::time::rdtsc();
        self.cost[0] += c1 - c0;
        self.cost[1] += c2 - c1;
        self.cost[2] += c3 - c2;
        self.t += 1;
    }

    /// Step 3 of the five: every awake neuron, and retirement of the ones
    /// this step left at a fixed point. `TM`/`TS` are the time constants
    /// when known at compile time and zero when they are not.
    ///
    /// Two ways to visit the awake set, and which is cheaper depends on how
    /// much of the animal is awake. A few hundred neurons: walk the list. Most
    /// of the brain -- which is what a fly under sustained sugar drive turned
    /// out to be, 125,000 of 138,639 receiving input every few milliseconds
    /// -- and a list in the order wakes arrived is a random walk through three
    /// arrays, 35 ns a neuron measured. Then the step scans every neuron in
    /// order and skips the sleeping ones, which is sequential and the
    /// prefetcher's favourite thing. The same updates either way, so the
    /// switch changes no spike, which `diag brain` checks by digest.
    #[inline(always)]
    fn integrate<const TM: i64, const TS: i64>(&mut self, c: &Connectome, base: usize, fired: &mut Vec<u32>) {
        let bias = self.bias;
        if self.dense {
            if !self.integrate_parallel::<TM, TS>(c, base, fired) {
                for i in 0..c.n {
                    let x = self.ring[base + i];
                    if x != 0 {
                        self.g[i] += x;
                        self.ring[base + i] = 0;
                        if !self.on[i] {
                            self.on[i] = true;
                            self.awake += 1;
                        }
                    }
                    if self.on[i] {
                        let (spiked, inert) = neuron::<TM, TS>(
                            c,
                            bias,
                            &mut self.v[i],
                            &mut self.g[i],
                            &mut self.refr[i],
                            &mut self.counts[i],
                            self.nore[i],
                        );
                        if spiked {
                            fired.push(i as u32);
                        }
                        if inert {
                            self.on[i] = false;
                            self.awake -= 1;
                        }
                    }
                }
            }
            // Back to the list once the brain has quietened, rebuilt in order.
            if self.awake * 16 < c.n {
                self.dense = false;
                self.active.clear();
                for i in 0..c.n {
                    if self.on[i] {
                        self.active.push(i as u32);
                    }
                }
            }
        } else {
            let mut k = 0;
            while k < self.active.len() {
                let i = self.active[k] as usize;
                let (spiked, inert) = neuron::<TM, TS>(
                    c,
                    bias,
                    &mut self.v[i],
                    &mut self.g[i],
                    &mut self.refr[i],
                    &mut self.counts[i],
                    self.nore[i],
                );
                if spiked {
                    fired.push(i as u32);
                }
                if inert {
                    self.on[i] = false;
                    self.awake -= 1;
                    self.active.swap_remove(k);
                } else {
                    k += 1;
                }
            }
            if self.awake * 8 > c.n {
                self.dense = true;
                self.active.clear();
            }
        }
    }

    /// The dense pass across every core `smp::parallel_split` can lend, in
    /// `PARTS` fixed slices of the neuron range, each with its own fired list
    /// and retire count, merged in slice order afterwards.
    ///
    /// Exact, which is the only reason it is allowed: a neuron's update reads
    /// and writes its own state and nothing else, and everything done with the
    /// fired list afterwards is a sum, which order cannot change -- the
    /// oracle digests in `diag brain` say so on every run. This was the
    /// bottleneck: 1,950 us of a 2,134 us step on the fly once a poke had
    /// woken most of it, one core walking 138,639 neurons every 0.1 ms.
    ///
    /// Answers false having done nothing when there is no help to be had --
    /// no other core, another job in flight -- and the caller walks serially.
    fn integrate_parallel<const TM: i64, const TS: i64>(&mut self, c: &Connectome, base: usize, fired: &mut Vec<u32>) -> bool {
        if self.parts.len() != PARTS {
            self.parts = (0..PARTS).map(|_| Vec::new()).collect();
        }
        for p in self.parts.iter_mut() {
            p.clear();
        }
        let mut retired = [0usize; PARTS];
        let mut woke = [0usize; PARTS];
        struct Job {
            c: *const Connectome,
            bias: i64,
            len: usize,
            n: usize,
            v: *mut i64,
            g: *mut i64,
            refr: *mut u16,
            counts: *mut u32,
            on: *mut bool,
            nore: *const bool,
            parts: *mut Vec<u32>,
            retired: *mut usize,
            woke: *mut usize,
            ring: *mut i64,
        }
        unsafe fn run<const TM: i64, const TS: i64>(ctx: usize, lo: usize, hi: usize) {
            let j = unsafe { &*(ctx as *const Job) };
            let c = unsafe { &*j.c };
            for p in lo..hi {
                // Slices never overlap, so no two cores touch one neuron.
                let (a, b) = (p * j.len, ((p + 1) * j.len).min(j.n));
                let out = unsafe { &mut *j.parts.add(p) };
                let (mut gone, mut up) = (0, 0);
                for i in a..b {
                    unsafe {
                        let x = *j.ring.add(i);
                        if x != 0 {
                            *j.g.add(i) += x;
                            *j.ring.add(i) = 0;
                            if !*j.on.add(i) {
                                *j.on.add(i) = true;
                                up += 1;
                            }
                        }
                        if !*j.on.add(i) {
                            continue;
                        }
                        let (spiked, inert) = neuron::<TM, TS>(
                            c,
                            j.bias,
                            &mut *j.v.add(i),
                            &mut *j.g.add(i),
                            &mut *j.refr.add(i),
                            &mut *j.counts.add(i),
                            *j.nore.add(i),
                        );
                        if spiked {
                            out.push(i as u32);
                        }
                        if inert {
                            *j.on.add(i) = false;
                            gone += 1;
                        }
                    }
                }
                unsafe {
                    *j.retired.add(p) = gone;
                    *j.woke.add(p) = up;
                }
            }
        }
        let job = Job {
            c: c as *const Connectome,
            bias: self.bias,
            len: c.n.div_ceil(PARTS),
            n: c.n,
            v: self.v.as_mut_ptr(),
            g: self.g.as_mut_ptr(),
            refr: self.refr.as_mut_ptr(),
            counts: self.counts.as_mut_ptr(),
            on: self.on.as_mut_ptr(),
            nore: self.nore.as_ptr(),
            parts: self.parts.as_mut_ptr(),
            retired: retired.as_mut_ptr(),
            woke: woke.as_mut_ptr(),
            ring: unsafe { self.ring.as_mut_ptr().add(base) },
        };
        // `width` is a cost hint, not a count: about sixteen operations a
        // neuron, so the fly's 138,639 clear the fabric's floor and the
        // worm's 300 stay on one core where the handshake would cost more.
        let width = job.len * 16;
        if !crate::smp::parallel_split(&job as *const Job as usize, run::<TM, TS>, PARTS, width) {
            return false;
        }
        for (p, out) in self.parts.iter().enumerate() {
            fired.extend_from_slice(out);
            self.awake = self.awake + woke[p] - retired[p];
        }
        true
    }

    /// Change the resting drive. Every neuron's fixed point moves with it, so
    /// the ones being skipped as inert are not inert any more: all of them
    /// wake, and the next steps settle them at the new rest. Rare by
    /// construction -- the caller quantises the bias -- so the one dense pass
    /// it costs is nothing.
    pub fn set_bias(&mut self, c: &Connectome, nv: i64) {
        if nv == self.bias {
            return;
        }
        self.bias = nv;
        for i in 0..c.n {
            self.wake(i);
        }
    }

    pub fn bias(&self) -> i64 {
        self.bias
    }

    /// Neurons awake right now, for `brain`.
    pub fn awake(&self) -> usize {
        self.awake
    }
}

/// FNV-1a over every neuron's count as a little-endian u32 -- the oracle's
/// `oracle_digest`, so one number says two runs agreed about every neuron.
pub fn digest(counts: &[u32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for c in counts {
        for b in c.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}
