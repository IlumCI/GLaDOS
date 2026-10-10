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
//!   2. Poisson input to driven sensors, in drive order, one draw each
//!   3. a refractory neuron is held at reset with g frozen; any other
//!      integrates v and then decays g
//!   4. a neuron strictly over threshold spikes, resets, goes refractory,
//!      and its row's weights enter the ring `delay` steps ahead

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
    on: Vec<bool>,
    pub t: u64,
    pub rng: Rng,
    pub counts: Vec<u32>,
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
            on: vec![false; c.n],
            t: 0,
            rng: Rng::new(seed),
            counts: vec![0; c.n],
        }
    }

    #[inline]
    fn wake(&mut self, i: usize) {
        if !self.on[i] {
            self.on[i] = true;
            self.active.push(i as u32);
        }
    }

    /// A neuron the dense loop would leave exactly as it is, this step and
    /// every step until something arrives: not refractory, a conductance the
    /// integer decay no longer moves, and a voltage the integer update no
    /// longer moves either. Nothing about it changes, so nothing about it
    /// will until input lands, and it is below threshold so it cannot spike.
    ///
    /// The first version asked for g == 0 and was exact but nearly useless:
    /// truncating decay leaves a residue under `tau_syn` nanovolts that never
    /// reaches zero, so after one sugar drive 125,772 of the fly's neurons
    /// stayed awake forever doing nothing, and the brain ran at 1% of real
    /// time stepping them. "Would this step change anything" is the right
    /// question, and it is still exact.
    ///
    /// This is the whole of the optimisation and it is exact rather than
    /// approximate -- which `diag brain` checks by requiring the same digest
    /// as the oracle's dense loop. With no basal firing, almost every neuron
    /// of a fly is inert almost all of the time, so a step costs what the
    /// activity costs and not what the animal costs.
    #[inline]
    fn inert(&self, c: &Connectome, i: usize) -> bool {
        self.refr[i] == 0
            && tdiv(self.g[i], c.tau_syn) == 0
            && tdiv(c.v0 - self.v[i] + self.g[i], c.tau_m) == 0
            && self.v[i] <= c.vth
    }

    /// One step. `driven` is (neuron, threshold) in drive order; `fired` is
    /// cleared and filled with this step's spikes, in no particular order --
    /// what is done with them is a sum, and integer addition does not care.
    pub fn step(&mut self, c: &Connectome, driven: &[(u32, u64)], fired: &mut Vec<u32>) {
        let n = c.n;
        let slot = self.t as usize % self.slots;
        let base = slot * n;
        let due = core::mem::take(&mut self.touched[slot]);
        for &i in due.iter() {
            let i = i as usize;
            let x = self.ring[base + i];
            if x != 0 {
                self.g[i] += x;
                self.ring[base + i] = 0;
                self.wake(i);
            }
        }
        let mut due = due;
        due.clear();
        self.touched[slot] = due;
        for &(i, thr) in driven {
            if self.rng.next() < thr {
                self.g[i as usize] += c.w_poi;
                self.wake(i as usize);
            }
        }
        fired.clear();
        let mut k = 0;
        while k < self.active.len() {
            let i = self.active[k] as usize;
            if self.refr[i] > 0 {
                // Shiu's model.py marks both equations `(unless refractory)`:
                // v is held and g does not decay.
                self.refr[i] -= 1;
                self.v[i] = c.vreset;
            } else {
                self.v[i] += tdiv(c.v0 - self.v[i] + self.g[i], c.tau_m);
                self.g[i] -= tdiv(self.g[i], c.tau_syn);
                if self.v[i] > c.vth {
                    self.v[i] = c.vreset;
                    self.refr[i] = c.refr;
                    self.counts[i] = self.counts[i].wrapping_add(1);
                    fired.push(i as u32);
                }
            }
            if self.inert(c, i) {
                self.on[i] = false;
                self.active.swap_remove(k);
            } else {
                k += 1;
            }
        }
        if !fired.is_empty() {
            let os = (self.t as usize + c.delay) % self.slots;
            let out = os * n;
            for &i in fired.iter() {
                let (a, b) = (c.rows[i as usize] as usize, c.rows[i as usize + 1] as usize);
                for k in a..b {
                    let j = c.col[k] as usize;
                    if self.ring[out + j] == 0 {
                        self.touched[os].push(j as u32);
                    }
                    self.ring[out + j] += c.w[k] as i64 * c.w_syn;
                }
            }
        }
        self.t += 1;
    }

    /// Neurons awake right now, for `brain`.
    pub fn awake(&self) -> usize {
        self.active.len()
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
