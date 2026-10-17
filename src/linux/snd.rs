//! `/dev/snd`: the ALSA kernel interface, over `dev::hda::pcm`.
//!
//! **Why ALSA and not something smaller.** The SDL2 every Steam game carries
//! is built with PulseAudio, PipeWire and ALSA backends and nothing else -- no
//! OSS -- and PulseAudio and PipeWire are both servers that end in ALSA
//! anyway. So this is the one interface that makes a game's own audio code
//! work unmodified, and libasound (userspace, staged beside the game) does
//! format and rate conversion through its `plug` plugin, which is why the
//! device itself can offer one format and two rates.
//!
//! One card, one playback device, one subdevice. No mixer elements, no
//! capture, no `mmap` of the status page -- refused, so libasound falls back
//! to `SYNC_PTR`, which it has always supported for exactly that case.
//!
//! **`HW_REFINE` is the part that is silent when wrong.** libasound narrows
//! a configuration space by asking the kernel to refine it repeatedly, and the
//! kernel's half is a fixpoint over the relations between parameters -- period
//! size, bytes and time; buffer likewise; buffer = period x periods. A refine
//! that answered each parameter on its own would hand back a space with no
//! consistent point in it, and `snd_pcm_hw_params` then fails with nothing
//! saying which parameter. The interval arithmetic here is the kernel's own
//! (`snd_interval_mul`, `div`, `muldivk`, `mulkdiv`), open bounds included,
//! because libasound's `set_near` builds open intervals and closing them
//! naively turns `(x, x+1)` into nothing.
//!
//! **Struct layouts are per ABI.** `snd_pcm_uframes_t` is `unsigned long`, so
//! `hw_params`, `sw_params`, `sync_ptr`, `status` and `xferi` all differ
//! between an i386 guest and an x86-64 one. `abi32()` picks.

use super::syscall::{abi32, reachable};
use crate::sync::Spin;

const EINVAL: u64 = (-22i64) as u64;
const ENOTTY: u64 = (-25i64) as u64;
const EFAULT: u64 = (-14i64) as u64;
const EAGAIN: u64 = (-11i64) as u64;
const ENOENT: u64 = (-2i64) as u64;
const EBADFD: u64 = (-77i64) as u64;
const EIO: u64 = (-5i64) as u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Node {
    Control,
    Playback,
}

// Parameter indices, `SNDRV_PCM_HW_PARAM_*`.
const ACCESS: usize = 0;
const FORMAT: usize = 1;
const SUBFORMAT: usize = 2;
const SAMPLE_BITS: usize = 8;
const FRAME_BITS: usize = 9;
const CHANNELS: usize = 10;
const RATE: usize = 11;
const PERIOD_TIME: usize = 12;
const PERIOD_SIZE: usize = 13;
const PERIOD_BYTES: usize = 14;
const PERIODS: usize = 15;
const BUFFER_TIME: usize = 16;
const BUFFER_SIZE: usize = 17;
const BUFFER_BYTES: usize = 18;
const TICK_TIME: usize = 19;

const ACCESS_RW_INTERLEAVED: u32 = 3;
const FORMAT_S16_LE: u32 = 2;
const RATES: [u32; 2] = [44100, 48000];

// States, `SNDRV_PCM_STATE_*`.
const OPEN: u32 = 0;
const SETUP: u32 = 1;
const PREPARED: u32 = 2;
const RUNNING: u32 = 3;
const PAUSED: u32 = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Interval {
    pub min: u32,
    pub max: u32,
    pub openmin: bool,
    pub openmax: bool,
    pub integer: bool,
    pub empty: bool,
}

impl Interval {
    fn closed(min: u32, max: u32, integer: bool) -> Interval {
        Interval { min, max, integer, ..Default::default() }
    }
    fn any() -> Interval {
        Interval::closed(0, u32::MAX, false)
    }
    fn read(b: &[u8]) -> Interval {
        let w = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let f = w(8);
        Interval { min: w(0), max: w(4), openmin: f & 1 != 0, openmax: f & 2 != 0, integer: f & 4 != 0, empty: f & 8 != 0 }
    }
    fn write(&self, b: &mut [u8]) {
        b[0..4].copy_from_slice(&self.min.to_le_bytes());
        b[4..8].copy_from_slice(&self.max.to_le_bytes());
        let f = self.openmin as u32 | (self.openmax as u32) << 1 | (self.integer as u32) << 2 | (self.empty as u32) << 3;
        b[8..12].copy_from_slice(&f.to_le_bytes());
    }
    fn checkempty(&self) -> bool {
        self.min > self.max || (self.min == self.max && (self.openmin || self.openmax))
    }
    fn single(&self) -> bool {
        !self.empty && (self.min == self.max || (self.min + 1 == self.max && (self.openmin || self.openmax)))
    }
    fn value(&self) -> u32 {
        if self.openmin && !self.openmax { self.max } else { self.min }
    }
    /// `snd_interval_refine`: intersect with `v`, answering whether anything
    /// moved, or `Err` for an empty result.
    pub fn refine(&mut self, v: &Interval) -> Result<bool, ()> {
        if self.empty {
            return Err(());
        }
        let mut changed = false;
        if self.min < v.min {
            self.min = v.min;
            self.openmin = v.openmin;
            changed = true;
        } else if self.min == v.min && !self.openmin && v.openmin {
            self.openmin = true;
            changed = true;
        }
        if self.max > v.max {
            self.max = v.max;
            self.openmax = v.openmax;
            changed = true;
        } else if self.max == v.max && !self.openmax && v.openmax {
            self.openmax = true;
            changed = true;
        }
        if !self.integer && v.integer {
            self.integer = true;
            changed = true;
        }
        if self.integer {
            if self.openmin {
                self.min = self.min.saturating_add(1);
                self.openmin = false;
            }
            if self.openmax {
                self.max = self.max.saturating_sub(1);
                self.openmax = false;
            }
        } else if !self.openmin && !self.openmax && self.min == self.max {
            self.integer = true;
        }
        if self.checkempty() {
            self.empty = true;
            return Err(());
        }
        Ok(changed)
    }
}

fn mul(a: &Interval, b: &Interval) -> Interval {
    let m = |x: u32, y: u32| (x as u64 * y as u64).min(u32::MAX as u64) as u32;
    Interval {
        min: m(a.min, b.min),
        openmin: a.openmin || b.openmin,
        max: m(a.max, b.max),
        openmax: a.openmax || b.openmax,
        integer: a.integer && b.integer,
        empty: false,
    }
}

fn div(a: &Interval, b: &Interval) -> Interval {
    let mut c = Interval::default();
    if b.max == 0 {
        return Interval { empty: true, ..c };
    }
    c.min = a.min / b.max;
    c.openmin = a.min % b.max != 0 || a.openmin || b.openmax;
    if b.min > 0 {
        let r = a.max % b.min;
        c.max = a.max / b.min;
        if r != 0 {
            c.max += 1;
            c.openmax = true;
        } else {
            c.openmax = a.openmax || b.openmin;
        }
    } else {
        c.max = u32::MAX;
        c.openmax = false;
    }
    c
}

/// `a * b / k`.
fn muldivk(a: &Interval, b: &Interval, k: u32) -> Interval {
    let mut c = Interval::default();
    let lo = a.min as u64 * b.min as u64;
    c.min = (lo / k as u64).min(u32::MAX as u64) as u32;
    c.openmin = lo % k as u64 != 0 || a.openmin || b.openmin;
    let hi = a.max as u64 * b.max as u64;
    let q = hi / k as u64;
    if hi % k as u64 != 0 {
        c.max = (q + 1).min(u32::MAX as u64) as u32;
        c.openmax = true;
    } else {
        c.max = q.min(u32::MAX as u64) as u32;
        c.openmax = a.openmax || b.openmax;
    }
    c
}

/// `a * k / b`.
fn mulkdiv(a: &Interval, k: u32, b: &Interval) -> Interval {
    let mut c = Interval::default();
    if b.max == 0 {
        return Interval { empty: true, ..c };
    }
    let lo = a.min as u64 * k as u64;
    c.min = (lo / b.max as u64).min(u32::MAX as u64) as u32;
    c.openmin = lo % b.max as u64 != 0 || a.openmin || b.openmax;
    if b.min > 0 {
        let hi = a.max as u64 * k as u64;
        let q = hi / b.min as u64;
        if hi % b.min as u64 != 0 {
            c.max = (q + 1).min(u32::MAX as u64) as u32;
            c.openmax = true;
        } else {
            c.max = q.min(u32::MAX as u64) as u32;
            c.openmax = a.openmax || b.openmin;
        }
    } else {
        c.max = u32::MAX;
        c.openmax = false;
    }
    c
}

/// A configuration space, as `snd_pcm_hw_params` carries it.
#[derive(Clone)]
pub struct Space {
    pub masks: [[u32; 8]; 3],
    pub iv: [Interval; 12],
}

impl Space {
    fn i(&self, p: usize) -> &Interval {
        &self.iv[p - SAMPLE_BITS]
    }
    fn set(&mut self, p: usize, v: &Interval) -> Result<bool, ()> {
        self.iv[p - SAMPLE_BITS].refine(v)
    }
    fn mask_refine(&mut self, p: usize, allowed: &[u32]) -> Result<bool, ()> {
        let mut m = [0u32; 8];
        for &a in allowed {
            m[a as usize / 32] |= 1 << (a % 32);
        }
        let old = self.masks[p];
        let mut any = false;
        for k in 0..8 {
            self.masks[p][k] &= m[k];
            any |= self.masks[p][k] != 0;
        }
        if !any {
            return Err(());
        }
        Ok(old != self.masks[p])
    }

    /// The kernel's refine: this device's constraints, then the rules
    /// between parameters until nothing moves.
    pub fn refine(&mut self) -> Result<u32, ()> {
        let before = self.clone();
        self.mask_refine(ACCESS, &[ACCESS_RW_INTERLEAVED])?;
        self.mask_refine(FORMAT, &[FORMAT_S16_LE])?;
        self.mask_refine(SUBFORMAT, &[0])?;
        self.set(SAMPLE_BITS, &Interval::closed(16, 16, true))?;
        self.set(FRAME_BITS, &Interval::closed(32, 32, true))?;
        self.set(CHANNELS, &Interval::closed(2, 2, true))?;
        self.set(RATE, &Interval::closed(RATES[0], RATES[RATES.len() - 1], true))?;
        self.set(PERIOD_BYTES, &Interval::closed(256, 32768, true))?;
        self.set(PERIODS, &Interval::closed(2, 64, true))?;
        self.set(BUFFER_BYTES, &Interval::closed(512, 32768, true))?;
        self.set(PERIOD_SIZE, &Interval { integer: true, ..Interval::any() })?;
        self.set(BUFFER_SIZE, &Interval { integer: true, ..Interval::any() })?;
        for _ in 0..64 {
            let mut moved = false;
            // The rate is a list, not a range.
            let r = *self.i(RATE);
            let lo = RATES.iter().copied().find(|&x| x > r.min || (x == r.min && !r.openmin));
            let hi = RATES.iter().rev().copied().find(|&x| x < r.max || (x == r.max && !r.openmax));
            match (lo, hi) {
                (Some(l), Some(h)) if l <= h => moved |= self.set(RATE, &Interval::closed(l, h, true))?,
                _ => return Err(()),
            }
            let (sb, fb, ch, rate) = (*self.i(SAMPLE_BITS), *self.i(FRAME_BITS), *self.i(CHANNELS), *self.i(RATE));
            moved |= self.set(FRAME_BITS, &mul(&sb, &ch))?;
            moved |= self.set(SAMPLE_BITS, &div(&fb, &ch))?;
            moved |= self.set(CHANNELS, &div(&fb, &sb))?;
            let fb = *self.i(FRAME_BITS);
            for (size, bytes, time) in [(PERIOD_SIZE, PERIOD_BYTES, PERIOD_TIME), (BUFFER_SIZE, BUFFER_BYTES, BUFFER_TIME)] {
                let (s, b, t) = (*self.i(size), *self.i(bytes), *self.i(time));
                moved |= self.set(size, &mulkdiv(&b, 8, &fb))?;
                moved |= self.set(size, &muldivk(&t, &rate, 1_000_000))?;
                moved |= self.set(bytes, &muldivk(&s, &fb, 8))?;
                moved |= self.set(time, &mulkdiv(&s, 1_000_000, &rate))?;
                let _ = t;
            }
            let (ps, pn, bs) = (*self.i(PERIOD_SIZE), *self.i(PERIODS), *self.i(BUFFER_SIZE));
            moved |= self.set(BUFFER_SIZE, &mul(&ps, &pn))?;
            moved |= self.set(PERIOD_SIZE, &div(&bs, &pn))?;
            moved |= self.set(PERIODS, &div(&bs, &ps))?;
            if !moved {
                break;
            }
        }
        let mut cmask = 0u32;
        for p in 0..3 {
            if self.masks[p] != before.masks[p] {
                cmask |= 1 << p;
            }
        }
        for p in SAMPLE_BITS..=TICK_TIME {
            if self.iv[p - SAMPLE_BITS] != before.iv[p - SAMPLE_BITS] {
                cmask |= 1 << p;
            }
        }
        Ok(cmask)
    }
}

/// Byte offsets in `snd_pcm_hw_params`.
const HW_MASKS: usize = 4;
const HW_IV: usize = 260;
const HW_RMASK: usize = 512;
const HW_CMASK: usize = 516;
const HW_INFO: usize = 520;
const HW_MSBITS: usize = 524;
const HW_RATE_NUM: usize = 528;
const HW_RATE_DEN: usize = 532;
const HW_FIFO: usize = 536;

fn hw_len() -> usize {
    if abi32() { 604 } else { 608 }
}

fn load(b: &[u8]) -> Space {
    let mut masks = [[0u32; 8]; 3];
    for (p, m) in masks.iter_mut().enumerate() {
        for k in 0..8 {
            let o = HW_MASKS + p * 32 + k * 4;
            m[k] = u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        }
    }
    let mut iv = [Interval::default(); 12];
    for (k, v) in iv.iter_mut().enumerate() {
        *v = Interval::read(&b[HW_IV + k * 12..HW_IV + k * 12 + 12]);
    }
    Space { masks, iv }
}

fn store(s: &Space, b: &mut [u8], cmask: u32) {
    for p in 0..3 {
        for k in 0..8 {
            let o = HW_MASKS + p * 32 + k * 4;
            b[o..o + 4].copy_from_slice(&s.masks[p][k].to_le_bytes());
        }
    }
    for k in 0..12 {
        s.iv[k].write(&mut b[HW_IV + k * 12..HW_IV + k * 12 + 12]);
    }
    let put = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    put(b, HW_RMASK, 0);
    put(b, HW_CMASK, cmask);
    // SNDRV_PCM_INFO_INTERLEAVED | BLOCK_TRANSFER | PAUSE | RESUME? Not resume.
    put(b, HW_INFO, 0x100 | 0x10000 | 0x80000);
    let sb = s.i(SAMPLE_BITS);
    put(b, HW_MSBITS, if sb.single() { sb.value() } else { 0 });
    let r = s.i(RATE);
    if r.single() {
        put(b, HW_RATE_NUM, r.value());
        put(b, HW_RATE_DEN, 1);
    }
    put(b, HW_FIFO, 0);
}

struct Pcm {
    state: u32,
    rate: u32,
    buffer: u64,
    period: u64,
    /// Frames the guest has written, ever: `appl_ptr`.
    appl: u64,
    avail_min: u64,
    boundary: u64,
}

static PCM: Spin<Pcm> = Spin::new(Pcm { state: OPEN, rate: 48000, buffer: 4096, period: 1024, appl: 0, avail_min: 1, boundary: 0 });

const FRAME: u64 = 4;

/// Counters for `linux snd`: ioctls seen, writes, frames taken, and the last
/// request number, so a stream that never starts says where it stopped.
static IOCTLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static WRITES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static FRAMES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static LAST: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static WAITS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn report() {
    use core::sync::atomic::Ordering::Relaxed;
    let p = PCM.lock_irq();
    crate::kprintln!(
        "  pcm state {} rate {} buffer {} period {} appl {} queued {} frames",
        p.state, p.rate, p.buffer, p.period, p.appl, queued()
    );
    drop(p);
    crate::kprintln!(
        "  {} ioctl(s), last {:#x}; {} write(s), {} frame(s) taken, {} wait(s) for room",
        IOCTLS.load(Relaxed), LAST.load(Relaxed), WRITES.load(Relaxed), FRAMES.load(Relaxed), WAITS.load(Relaxed)
    );
    crate::kprintln!(
        "  sink {} position {:?}",
        if crate::dev::hda::pcm::is_open() { "open" } else { "closed" },
        crate::dev::hda::with(|h| h.position()).flatten()
    );
}

/// Frames written and not yet played.
fn queued() -> u64 {
    crate::dev::hda::pcm::queued() / FRAME
}

/// `hw_ptr`: how far the "hardware" has got, which is the guest's own count
/// less what the ring still holds.
fn hw_ptr(p: &Pcm) -> u64 {
    p.appl.saturating_sub(queued())
}

fn avail(p: &Pcm) -> u64 {
    p.buffer.saturating_sub(queued().min(p.buffer))
}

/// Whether a write would not block, for `poll`.
pub fn writable() -> bool {
    let p = PCM.lock_irq();
    p.state != PREPARED && p.state != RUNNING || avail(&p) >= p.avail_min.max(1)
}

pub fn opened(n: Node) {
    if n == Node::Playback {
        let mut p = PCM.lock_irq();
        p.state = OPEN;
        p.appl = 0;
    }
}

pub fn closed(n: Node) {
    if n == Node::Playback {
        crate::dev::hda::pcm::close();
        PCM.lock_irq().state = OPEN;
    }
}

fn put_u32(at: u64, off: usize, v: u32) {
    unsafe { core::ptr::write_unaligned((at + off as u64) as *mut u32, v) }
}
fn get_u32(at: u64, off: usize) -> u32 {
    unsafe { core::ptr::read_unaligned((at + off as u64) as *const u32) }
}
/// An `unsigned long` at `off`, whose width is the guest's.
fn put_ul(at: u64, off: usize, v: u64) {
    if abi32() {
        put_u32(at, off, v as u32)
    } else {
        unsafe { core::ptr::write_unaligned((at + off as u64) as *mut u64, v) }
    }
}
fn get_ul(at: u64, off: usize) -> u64 {
    if abi32() {
        get_u32(at, off) as u64
    } else {
        unsafe { core::ptr::read_unaligned((at + off as u64) as *const u64) }
    }
}
fn put_str(at: u64, off: usize, cap: usize, s: &str) {
    let b = s.as_bytes();
    let n = b.len().min(cap - 1);
    unsafe {
        core::ptr::write_bytes((at + off as u64) as *mut u8, 0, cap);
        core::ptr::copy_nonoverlapping(b.as_ptr(), (at + off as u64) as *mut u8, n);
    }
}

fn fill_pcm_info(at: u64) {
    unsafe { core::ptr::write_bytes(at as *mut u8, 0, 288) };
    put_str(at, 16, 64, "GLaDOS HDA");
    put_str(at, 80, 80, "GLaDOS HDA Analog");
    put_str(at, 160, 32, "subdevice #0");
    put_u32(at, 200, 1); // subdevices_count
    put_u32(at, 204, 1); // subdevices_avail
}

pub fn ioctl(n: Node, req: u64, arg: u64, nonblock: bool) -> u64 {
    IOCTLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    LAST.store(req, core::sync::atomic::Ordering::Relaxed);
    let kind = (req >> 8) & 0xff;
    let nr = req & 0xff;
    let size = ((req >> 16) & 0x3fff) as usize;
    let dir = (req >> 30) & 3;
    // PAUSE carries its argument as the value itself, not a pointer to one.
    let by_value = n == Node::Playback && kind == 0x41 && nr == 0x45;
    if size > 0 && !by_value && !reachable(arg, size, dir & 2 != 0) {
        return EFAULT;
    }
    match (n, kind, nr) {
        // SNDRV_CTL_IOCTL_PVERSION
        (Node::Control, 0x55, 0x00) => {
            put_u32(arg, 0, 2 << 16 | 7);
            0
        }
        // CARD_INFO
        (Node::Control, 0x55, 0x01) => {
            unsafe { core::ptr::write_bytes(arg as *mut u8, 0, 376) };
            put_str(arg, 8, 16, "GLaDOS");
            put_str(arg, 24, 16, "HDA-Intel");
            put_str(arg, 40, 32, "GLaDOS HDA");
            put_str(arg, 72, 80, "GLaDOS High Definition Audio");
            put_str(arg, 168, 80, "GLaDOS Mixer");
            0
        }
        // ELEM_LIST: no elements. offset, space, used, count, pids.
        (Node::Control, 0x55, 0x10) => {
            put_u32(arg, 8, 0);
            put_u32(arg, 12, 0);
            0
        }
        // SUBSCRIBE_EVENTS, PCM_PREFER_SUBDEVICE
        (Node::Control, 0x55, 0x16) | (Node::Control, 0x55, 0x32) => 0,
        // PCM_NEXT_DEVICE: 0 after -1, then none.
        (Node::Control, 0x55, 0x30) => {
            let d = get_u32(arg, 0) as i32;
            put_u32(arg, 0, if d < 0 { 0 } else { u32::MAX });
            0
        }
        // PCM_INFO: playback on device 0 only.
        (Node::Control, 0x55, 0x31) => {
            let (dev, sub, stream) = (get_u32(arg, 0), get_u32(arg, 4), get_u32(arg, 8));
            if dev != 0 || sub != 0 || stream != 0 {
                return ENOENT;
            }
            fill_pcm_info(arg);
            0
        }
        // POWER_STATE
        (Node::Control, 0x55, 0xd1) => {
            if size >= 4 {
                put_u32(arg, 0, 0);
            }
            0
        }
        (Node::Control, _, _) => ENOTTY,

        // PVERSION
        (Node::Playback, 0x41, 0x00) => {
            put_u32(arg, 0, 2 << 16 | 12);
            0
        }
        (Node::Playback, 0x41, 0x01) => {
            fill_pcm_info(arg);
            0
        }
        // TSTAMP, TTSTAMP, USER_PVERSION
        (Node::Playback, 0x41, 0x02..=0x04) => 0,
        // HW_REFINE, HW_PARAMS
        (Node::Playback, 0x41, 0x10) | (Node::Playback, 0x41, 0x11) => {
            let len = hw_len();
            if !reachable(arg, len, true) {
                return EFAULT;
            }
            let b = unsafe { core::slice::from_raw_parts_mut(arg as *mut u8, len) };
            let mut s = load(b);
            let Ok(cmask) = s.refine() else { return EINVAL };
            if nr == 0x11 {
                // Choose: every interval to its minimum, which is what the
                // kernel's `snd_pcm_hw_params_choose` does for the ones
                // libasound left open, then refine once more so the
                // dependants follow.
                for p in [RATE, PERIOD_TIME, PERIOD_SIZE, PERIODS, BUFFER_SIZE] {
                    let v = s.i(p).value();
                    let iv = &mut s.iv[p - SAMPLE_BITS];
                    if !iv.single() {
                        *iv = Interval::closed(v, v, true);
                        if s.refine().is_err() {
                            return EINVAL;
                        }
                    }
                }
                let mut p = PCM.lock_irq();
                p.rate = s.i(RATE).value();
                p.period = s.i(PERIOD_SIZE).value() as u64;
                p.buffer = s.i(BUFFER_SIZE).value() as u64;
                p.state = SETUP;
                p.appl = 0;
            }
            store(&s, b, cmask);
            0
        }
        // HW_FREE
        (Node::Playback, 0x41, 0x12) => {
            crate::dev::hda::pcm::close();
            PCM.lock_irq().state = OPEN;
            0
        }
        // SW_PARAMS: tstamp_mode, period_step, sleep_min, then unsigned
        // longs from offset 12 (16 on x86-64): avail_min, xfer_align,
        // start_threshold, stop_threshold, silence_threshold, silence_size,
        // boundary.
        (Node::Playback, 0x41, 0x13) => {
            let base = if abi32() { 12 } else { 16 };
            let w = if abi32() { 4 } else { 8 };
            let mut p = PCM.lock_irq();
            p.avail_min = get_ul(arg, base).max(1);
            p.boundary = get_ul(arg, base + 6 * w);
            0
        }
        // STATUS, STATUS_EXT
        (Node::Playback, 0x41, 0x20) | (Node::Playback, 0x41, 0x24) => {
            unsafe { core::ptr::write_bytes(arg as *mut u8, 0, size) };
            let p = PCM.lock_irq();
            put_u32(arg, 0, p.state);
            let (appl, hw, delay, av) = if abi32() { (20, 24, 28, 32) } else { (40, 48, 56, 64) };
            put_ul(arg, appl, p.appl);
            put_ul(arg, hw, hw_ptr(&p));
            put_ul(arg, delay, queued());
            put_ul(arg, av, avail(&p));
            0
        }
        // DELAY
        (Node::Playback, 0x41, 0x21) => {
            put_ul(arg, 0, queued());
            0
        }
        // HWSYNC
        (Node::Playback, 0x41, 0x22) => 0,
        // SYNC_PTR: flags, then status (state, pad, hw_ptr, ...) and control
        // (appl_ptr, avail_min), each in a 64-byte union.
        (Node::Playback, 0x41, 0x23) => {
            let (s, c) = if abi32() { (4usize, 68usize) } else { (8, 72) };
            let flags = get_u32(arg, 0);
            let mut p = PCM.lock_irq();
            // SNDRV_PCM_SYNC_PTR_APPL: the guest keeps its own appl_ptr.
            if flags & 2 == 0 {
                let a = get_ul(arg, c);
                if a >= p.appl {
                    p.appl = a;
                }
            }
            if flags & 4 == 0 {
                p.avail_min = get_ul(arg, c + if abi32() { 4 } else { 8 }).max(1);
            }
            put_u32(arg, s, p.state);
            put_ul(arg, s + 8, hw_ptr(&p));
            put_ul(arg, c, p.appl);
            put_ul(arg, c + if abi32() { 4 } else { 8 }, p.avail_min);
            0
        }
        // PREPARE, RESET
        (Node::Playback, 0x41, 0x40) | (Node::Playback, 0x41, 0x41) => {
            let mut p = PCM.lock_irq();
            if p.state == OPEN {
                return EBADFD;
            }
            if !crate::dev::hda::pcm::is_open() && !crate::dev::hda::pcm::open(p.rate, 2, 16) {
                return EIO;
            }
            p.state = PREPARED;
            0
        }
        // START
        (Node::Playback, 0x41, 0x42) => {
            PCM.lock_irq().state = RUNNING;
            0
        }
        // DROP: what is queued goes, now.
        (Node::Playback, 0x41, 0x43) => {
            crate::dev::hda::pcm::close();
            let mut p = PCM.lock_irq();
            if p.state != OPEN {
                p.state = SETUP;
            }
            0
        }
        // DRAIN: wait for the ring to empty.
        (Node::Playback, 0x41, 0x44) => {
            while queued() > 0 {
                if super::syscall::overran(crate::dev::lapic::ticks()) {
                    unsafe { super::syscall::kill_blocked() }
                }
                super::syscall::wait_turn();
            }
            crate::dev::hda::pcm::close();
            PCM.lock_irq().state = SETUP;
            0
        }
        // PAUSE
        (Node::Playback, 0x41, 0x45) => {
            let mut p = PCM.lock_irq();
            p.state = if arg != 0 { PAUSED } else { RUNNING };
            0
        }
        // REWIND, FORWARD: nothing can be taken back once it is in the ring.
        (Node::Playback, 0x41, 0x46) | (Node::Playback, 0x41, 0x49) => {
            put_ul(arg, 0, 0);
            0
        }
        // RESUME, XRUN
        (Node::Playback, 0x41, 0x47) | (Node::Playback, 0x41, 0x48) => 0,
        // WRITEI_FRAMES: result, buf, frames.
        (Node::Playback, 0x41, 0x50) => writei(arg, nonblock),
        (Node::Playback, _, _) => ENOTTY,
    }
}

fn writei(arg: u64, nonblock: bool) -> u64 {
    let w = if abi32() { 4 } else { 8 };
    let buf = get_ul(arg, w);
    let frames = get_ul(arg, 2 * w);
    {
        let mut p = PCM.lock_irq();
        match p.state {
            PREPARED | RUNNING => {}
            _ => return EBADFD,
        }
        if !crate::dev::hda::pcm::is_open() && !crate::dev::hda::pcm::open(p.rate, 2, 16) {
            return EIO;
        }
        p.state = RUNNING;
    }
    if !reachable(buf, (frames * FRAME) as usize, false) {
        return EFAULT;
    }
    WRITES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let mut done = 0u64;
    while done < frames {
        let room = {
            let p = PCM.lock_irq();
            avail(&p).min(frames - done)
        };
        let n = if room > 0 {
            let src = unsafe { core::slice::from_raw_parts((buf + done * FRAME) as *const u8, (room * FRAME) as usize) };
            crate::dev::hda::pcm::write(src) as u64 / FRAME
        } else {
            0
        };
        if n > 0 {
            FRAMES.fetch_add(n, core::sync::atomic::Ordering::Relaxed);
            done += n;
            PCM.lock_irq().appl += n;
            continue;
        }
        if nonblock {
            break;
        }
        WAITS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if super::syscall::overran(crate::dev::lapic::ticks()) {
            unsafe { super::syscall::kill_blocked() }
        }
        super::syscall::wait_turn();
    }
    if done == 0 && frames > 0 {
        return EAGAIN;
    }
    put_ul(arg, 0, done);
    0
}

pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    let any = || {
        let mut iv = [Interval::any(); 12];
        for v in iv.iter_mut() {
            v.integer = false;
        }
        Space { masks: [[u32::MAX; 8]; 3], iv }
    };
    let mut s = any();
    out.push(("an unconstrained space refines", s.refine().is_ok()));
    out.push(("to S16_LE stereo", s.i(FRAME_BITS).single() && s.i(FRAME_BITS).value() == 32 && s.i(CHANNELS).value() == 2));
    out.push(("at 44.1 or 48 kHz", s.i(RATE).min == 44100 && s.i(RATE).max == 48000));
    // A period of 1024 frames at 48 kHz: the bytes and the time follow.
    let mut s = any();
    s.iv[RATE - SAMPLE_BITS] = Interval::closed(48000, 48000, true);
    s.iv[PERIOD_SIZE - SAMPLE_BITS] = Interval::closed(1024, 1024, true);
    s.iv[PERIODS - SAMPLE_BITS] = Interval::closed(4, 4, true);
    let ok = s.refine().is_ok();
    out.push(("a period of 1024 frames is 4096 bytes", ok && s.i(PERIOD_BYTES).single() && s.i(PERIOD_BYTES).value() == 4096));
    out.push(("and four of them are a 4096-frame buffer", ok && s.i(BUFFER_SIZE).value() == 4096 && s.i(BUFFER_SIZE).single()));
    out.push(("whose time is 85333 us, open above", ok && s.i(BUFFER_TIME).min == 85333));
    // A rate between the two offered is no rate at all.
    let mut s = any();
    s.iv[RATE - SAMPLE_BITS] = Interval::closed(45000, 46000, true);
    out.push(("a rate between 44.1 and 48 kHz is refused", s.refine().is_err()));
    // 22050 is not offered either; plug converts it in userspace.
    let mut s = any();
    s.iv[RATE - SAMPLE_BITS] = Interval::closed(22050, 22050, true);
    out.push(("and so is 22.05 kHz, which libasound's plug converts", s.refine().is_err()));
    // Open bounds survive a refine: (x, x+1) is not empty for a time.
    let mut a = Interval { min: 10, max: 11, openmin: true, openmax: true, ..Default::default() };
    out.push(("an open time interval (10, 11) is not empty", a.refine(&Interval::any()).is_ok()));
    out
}
