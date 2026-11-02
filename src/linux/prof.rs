//! Where a guest's time goes: `linux prof on`, run it, `linux prof`.
//!
//! Two instruments, because one cannot see the other's half.
//!
//! **Kernel time is timed, not sampled.** A syscall runs with `IF` clear --
//! `FMASK` takes it on entry -- so the timer cannot land inside one, and a
//! sampler would see nothing of the kernel but the instruction after each
//! `int 0x80`. So the dispatcher reads the TSC either side of every call and
//! bills the difference to the call's number. That includes the display
//! server's turn, which runs on the way out of every call and is billed to a
//! row of its own.
//!
//! **Guest time is sampled.** The timer interrupt records the interrupted
//! `rip` whenever it arrived from ring 3, on every core. Code mapped from a
//! file is named `file+offset` at report time, for `nm` on the host; anything
//! else is anonymous memory, which for a GL program is overwhelmingly
//! llvmpipe's JIT output. And a tick that landed on a core with no guest
//! running at all is counted as idle, per core, because "the second core sat
//! idle the whole time" is the answer to a different question from "the
//! first one was busy".

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

pub static ON: AtomicBool = AtomicBool::new(false);

/// Call numbers, i386 at 0..512 and x86-64 at 512..1024; the last row is the
/// display server's pump.
const ROWS: usize = 1025;
pub const PUMP: usize = 1024;
static CYCLES: [AtomicU64; ROWS] = [const { AtomicU64::new(0) }; ROWS];
static CALLS: [AtomicU64; ROWS] = [const { AtomicU64::new(0) }; ROWS];

const SAMPLES: usize = 1 << 16;
static RIPS: [AtomicU64; SAMPLES] = [const { AtomicU64::new(0) }; SAMPLES];
static NEXT: AtomicUsize = AtomicUsize::new(0);

const CORES: usize = 16;
/// Ring-3 ticks per task: which thread the time is.
static BY_TASK: [AtomicU64; crate::task::MAX_TASKS] = [const { AtomicU64::new(0) }; crate::task::MAX_TASKS];
/// Per core: ticks at ring 3, ticks in a guest's task at ring 0 (a wait with
/// interrupts on), and ticks with no guest on that core.
static RING3: [AtomicU64; CORES] = [const { AtomicU64::new(0) }; CORES];
static WAIT: [AtomicU64; CORES] = [const { AtomicU64::new(0) }; CORES];
static OTHER: [AtomicU64; CORES] = [const { AtomicU64::new(0) }; CORES];

pub fn row(nr: u64, i386: bool) -> usize {
    let n = (nr as usize).min(511);
    if i386 { n } else { 512 + n }
}

#[inline]
pub fn start() -> u64 {
    if ON.load(Ordering::Relaxed) { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 }
}

#[inline]
pub fn bill(row: usize, t0: u64) {
    if t0 == 0 {
        return;
    }
    let dt = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0);
    CYCLES[row].fetch_add(dt, Ordering::Relaxed);
    CALLS[row].fetch_add(1, Ordering::Relaxed);
}

/// From the timer handler, every core. Allocates nothing and takes no lock.
#[inline]
pub fn sample(cs: u64, rip: u64) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let core = if crate::cpu::percpu::armed() { crate::cpu::percpu::cpu_id() as usize } else { 0 }.min(CORES - 1);
    if cs & 3 == 3 {
        RING3[core].fetch_add(1, Ordering::Relaxed);
        let t = crate::task::current().min(crate::task::MAX_TASKS - 1);
        BY_TASK[t].fetch_add(1, Ordering::Relaxed);
        let i = NEXT.fetch_add(1, Ordering::Relaxed);
        if i < SAMPLES {
            RIPS[i].store(rip, Ordering::Relaxed);
        }
    } else if super::syscall::running() {
        WAIT[core].fetch_add(1, Ordering::Relaxed);
    } else {
        OTHER[core].fetch_add(1, Ordering::Relaxed);
    }
}

/// Sleeps by what they asked for, in three bands -- under a tick, under ten,
/// and longer -- with nanoseconds asked and nanoseconds taken. A sleep may run
/// long and must not run short; this says how long, because a waiter that
/// polls in 160 us sleeps and gets 10 ms each is a frame rate set by the
/// scheduler's quantum rather than by the work.
static SLEEP_N: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static SLEEP_ASK: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static SLEEP_GOT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

pub fn note_sleep(asked_ns: u64, got_ns: u64) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let tick = 1_000_000_000 / crate::TIMER_HZ as u64;
    let b = if asked_ns < tick { 0 } else if asked_ns < 10 * tick { 1 } else { 2 };
    SLEEP_N[b].fetch_add(1, Ordering::Relaxed);
    SLEEP_ASK[b].fetch_add(asked_ns, Ordering::Relaxed);
    SLEEP_GOT[b].fetch_add(got_ns, Ordering::Relaxed);
}

/// Waits for the guest lock that found it taken: how many, the cycles spent,
/// and the worst. A ticket lock whose waiters yield can convoy -- the waiter
/// whose turn comes up may be off its core, and everyone behind it waits for
/// it to be scheduled again -- and nothing else here would show that.
static BIG_N: AtomicU64 = AtomicU64::new(0);
static BIG_CYC: AtomicU64 = AtomicU64::new(0);
static BIG_WORST: AtomicU64 = AtomicU64::new(0);

/// How long each holder kept the guest lock, by the syscall it was serving:
/// a lock found taken 50,000 times in twenty seconds is a fact about whoever
/// held it, and the waits alone do not say who that was.
static HOLD: [AtomicU64; ROWS] = [const { AtomicU64::new(0) }; ROWS];
static HOLD_N: [AtomicU64; ROWS] = [const { AtomicU64::new(0) }; ROWS];
static HOLD_WORST: [AtomicU64; ROWS] = [const { AtomicU64::new(0) }; ROWS];
static HELD_AT: AtomicU64 = AtomicU64::new(0);

/// The lock was just taken.
#[inline]
pub fn held() {
    if ON.load(Ordering::Relaxed) {
        HELD_AT.store(unsafe { core::arch::x86_64::_rdtsc() }, Ordering::Relaxed);
    }
}

/// The lock is about to be let go: bill the hold to the call in flight on
/// this task, or to row 0 when the holder is not inside a syscall (a guest
/// thread's own bookkeeping, a kernel path).
#[inline]
pub fn releasing() {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let at = HELD_AT.swap(0, Ordering::Relaxed);
    if at == 0 {
        return;
    }
    let dt = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(at);
    let t = crate::task::current().min(crate::task::MAX_TASKS - 1);
    let v = INFLIGHT[t][0].load(Ordering::Relaxed);
    let r = if v & 1 << 62 == 0 {
        0
    } else {
        let nr = v & !(1 << 62);
        if nr & super::i386::TAG != 0 { row(nr & !super::i386::TAG, true) } else { row(nr, false) }
    };
    HOLD[r].fetch_add(dt, Ordering::Relaxed);
    HOLD_N[r].fetch_add(1, Ordering::Relaxed);
    HOLD_WORST[r].fetch_max(dt, Ordering::Relaxed);
    // An ioctl is many calls under one number: split by request, whose low
    // byte and type byte name it (DRM is type 'd', ALSA 'A').
    if r == row(54, true) || r == row(16, false) {
        let req = INFLIGHT[t][2].load(Ordering::Relaxed);
        let k = (((req >> 8) & 0xff) as usize * 7 + (req & 0xff) as usize) % IOCTLS;
        IOCTL_REQ[k].store(req & 0xffff, Ordering::Relaxed);
        IOCTL_HOLD[k].fetch_add(dt, Ordering::Relaxed);
        IOCTL_N[k].fetch_add(1, Ordering::Relaxed);
    }
}

const IOCTLS: usize = 64;
static IOCTL_REQ: [AtomicU64; IOCTLS] = [const { AtomicU64::new(0) }; IOCTLS];
static IOCTL_HOLD: [AtomicU64; IOCTLS] = [const { AtomicU64::new(0) }; IOCTLS];
static IOCTL_N: [AtomicU64; IOCTLS] = [const { AtomicU64::new(0) }; IOCTLS];

pub fn note_big_wait(cycles: u64) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    BIG_N.fetch_add(1, Ordering::Relaxed);
    BIG_CYC.fetch_add(cycles, Ordering::Relaxed);
    BIG_WORST.fetch_max(cycles, Ordering::Relaxed);
}

/// When profiling began, and the client frame count then.
static SINCE: AtomicU64 = AtomicU64::new(0);
static FRAMES_AT: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    SINCE.store(unsafe { core::arch::x86_64::_rdtsc() }, Ordering::Relaxed);
    FRAMES_AT.store(crate::sky::surface::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    for r in 0..ROWS {
        CYCLES[r].store(0, Ordering::Relaxed);
        CALLS[r].store(0, Ordering::Relaxed);
    }
    for c in 0..CORES {
        RING3[c].store(0, Ordering::Relaxed);
        WAIT[c].store(0, Ordering::Relaxed);
        OTHER[c].store(0, Ordering::Relaxed);
    }
    NEXT.store(0, Ordering::Relaxed);
    for r in 0..ROWS {
        HOLD[r].store(0, Ordering::Relaxed);
        HOLD_N[r].store(0, Ordering::Relaxed);
        HOLD_WORST[r].store(0, Ordering::Relaxed);
    }
    for k in 0..IOCTLS {
        IOCTL_HOLD[k].store(0, Ordering::Relaxed);
        IOCTL_N[k].store(0, Ordering::Relaxed);
    }
    BIG_N.store(0, Ordering::Relaxed);
    BIG_CYC.store(0, Ordering::Relaxed);
    BIG_WORST.store(0, Ordering::Relaxed);
    for i in 0..3 {
        SLEEP_N[i].store(0, Ordering::Relaxed);
        SLEEP_ASK[i].store(0, Ordering::Relaxed);
        SLEEP_GOT[i].store(0, Ordering::Relaxed);
    }
    for t in 0..crate::task::MAX_TASKS {
        BY_TASK[t].store(0, Ordering::Relaxed);
    }
}

pub fn report() {
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec::Vec;
    let mhz = crate::time::tsc_mhz().max(1);
    match super::syscall::big_holder() {
        Some(t) => crate::kprintln!("  the guest lock is held by task {}", t),
        None => crate::kprintln!("  the guest lock is free"),
    }
    let secs = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(SINCE.load(Ordering::Relaxed)) / mhz / 1_000_000;
    let frames = crate::sky::surface::FRAMES.load(Ordering::Relaxed) - FRAMES_AT.load(Ordering::Relaxed);
    crate::kprintln!(
        "  {} client frame(s) in {} s, {}.{} per second",
        frames, secs, frames / secs.max(1), frames * 10 / secs.max(1) % 10
    );
    let bn = BIG_N.load(Ordering::Relaxed);
    if bn > 0 {
        crate::kprintln!(
            "  guest lock found taken {} times, {} ms waiting in all, {} us each, worst {} us",
            bn,
            BIG_CYC.load(Ordering::Relaxed) / mhz / 1000,
            BIG_CYC.load(Ordering::Relaxed) / mhz / bn,
            BIG_WORST.load(Ordering::Relaxed) / mhz
        );
    }
    {
        let mut h: Vec<(u64, usize)> = (0..ROWS).map(|r| (HOLD[r].load(Ordering::Relaxed), r)).filter(|e| e.0 > 0).collect();
        h.sort_by(|a, b| b.0.cmp(&a.0));
        if !h.is_empty() {
            crate::kprintln!("  guest lock held, by the call holding it:");
        }
        for &(cy, r) in h.iter().take(8) {
            let n = HOLD_N[r].load(Ordering::Relaxed).max(1);
            let name = if r == 0 {
                String::from("(outside a call)")
            } else if r < 512 {
                alloc::format!("{} (i386)", super::syscall::name_of(r as u64 | super::i386::TAG))
            } else {
                String::from(super::syscall::name_of((r - 512) as u64))
            };
            crate::kprintln!(
                "    {:<24} {:>6} ms {:>7} holds {:>5} us each, worst {:>6} us",
                name, cy / mhz / 1000, n, cy / mhz / n, HOLD_WORST[r].load(Ordering::Relaxed) / mhz
            );
        }
    }
    {
        let mut v: Vec<(u64, usize)> = (0..IOCTLS).map(|k| (IOCTL_HOLD[k].load(Ordering::Relaxed), k)).filter(|e| e.0 > 0).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        for &(cy, k) in v.iter().take(6) {
            let n = IOCTL_N[k].load(Ordering::Relaxed).max(1);
            let req = IOCTL_REQ[k].load(Ordering::Relaxed);
            crate::kprintln!(
                "      ioctl '{}' {:#04x}  {:>6} ms {:>7} holds {:>5} us each",
                (((req >> 8) & 0xff) as u8 as char),
                req & 0xff,
                cy / mhz / 1000,
                n,
                cy / mhz / n
            );
        }
    }
    for (i, band) in ["under a tick", "1 to 10 ticks", "10 ticks or more"].iter().enumerate() {
        let n = SLEEP_N[i].load(Ordering::Relaxed);
        if n > 0 {
            let (ask, got) = (SLEEP_ASK[i].load(Ordering::Relaxed), SLEEP_GOT[i].load(Ordering::Relaxed));
            crate::kprintln!(
                "  sleeps {:<16} {:>6}, asked {:>6} us each, took {:>6} us each",
                band, n, ask / n / 1000, got / n / 1000
            );
        }
    }
    crate::kprintln!("  ticks per core: ring 3 / guest waiting / no guest");
    for c in 0..CORES {
        let (a, b, o) = (RING3[c].load(Ordering::Relaxed), WAIT[c].load(Ordering::Relaxed), OTHER[c].load(Ordering::Relaxed));
        if a + b + o > 0 {
            crate::kprintln!("    cpu{c}  {a:>7} {b:>7} {o:>7}");
        }
    }
    let mut tasks: Vec<(u64, usize)> = (0..crate::task::MAX_TASKS)
        .map(|t| (BY_TASK[t].load(Ordering::Relaxed), t))
        .filter(|&(n, _)| n > 0)
        .collect();
    tasks.sort_by(|a, b| b.0.cmp(&a.0));
    crate::kprintln!(
        "  ring-3 ticks by task: {}",
        tasks.iter().take(12).map(|(n, t)| alloc::format!("{t}:{n}")).collect::<Vec<_>>().join(" ")
    );
    let mut rows: Vec<(u64, u64, usize)> = (0..ROWS)
        .map(|r| (CYCLES[r].load(Ordering::Relaxed), CALLS[r].load(Ordering::Relaxed), r))
        .filter(|&(c, _, _)| c > 0)
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    let total: u64 = rows.iter().map(|r| r.0).sum();
    crate::kprintln!("  kernel time by call, {} ms in all", total / mhz / 1000);
    for &(cy, n, r) in rows.iter().take(24) {
        let name = if r == PUMP {
            "(display server)"
        } else if r < 512 {
            super::syscall::name_of(r as u64 | (1 << 31))
        } else {
            super::syscall::name_of((r - 512) as u64)
        };
        // A call the name table lacks still says which it is.
        let num = if r < 512 { r } else { r - 512 };
        let shown = if name == "?" { alloc::format!("? nr {}{}", num, if r < 512 { " (i386)" } else { "" }) } else { String::from(name) };
        crate::kprintln!("    {:<18} {:>8} ms {:>9} calls {:>7} us each", shown, cy / mhz / 1000, n, cy / mhz / n.max(1));
    }
    let n = NEXT.load(Ordering::Relaxed).min(SAMPLES);
    let mut by_file: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_spot: BTreeMap<String, u64> = BTreeMap::new();
    for i in 0..n {
        let rip = RIPS[i].load(Ordering::Relaxed);
        let (file, spot) = match super::syscall::name_code(rip) {
            Some(s) => {
                let file = String::from(s.split('+').next().unwrap_or("?"));
                // Buckets of 256 bytes, which is a function or a loop.
                let off = s.split('+').nth(1).and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
                (file.clone(), alloc::format!("{}+{:#x}", file, off & !0xff))
            }
            None => (String::from("(anonymous: JIT or heap)"), alloc::format!("anon {:#x}", rip & !0xfff)),
        };
        *by_file.entry(file).or_insert(0) += 1;
        *by_spot.entry(spot).or_insert(0) += 1;
    }
    crate::kprintln!("  {} ring-3 sample(s), by file", n);
    let mut f: Vec<_> = by_file.into_iter().collect();
    f.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in f.iter().take(16) {
        crate::kprintln!("    {:>6} {:>5.1}%  {}", v, *v as f32 * 100.0 / n.max(1) as f32, k);
    }
    crate::kprintln!("  hottest 256-byte spots");
    let mut s: Vec<_> = by_spot.into_iter().collect();
    s.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in s.iter().take(30) {
        crate::kprintln!("    {:>6}  {}", v, k);
    }
}

/// The call each task is inside, by task: number and three arguments, or a
/// zero number for none. Set and cleared by both dispatchers, so a guest that
/// has stopped making progress can be asked what each of its threads is
/// waiting on -- which no trace can say, since a call in progress has not
/// been recorded yet.
static INFLIGHT: [[AtomicU64; 4]; crate::task::MAX_TASKS] =
    [const { [const { AtomicU64::new(0) }; 4] }; crate::task::MAX_TASKS];

pub fn enter(nr: u64, a: &[u64]) {
    let t = crate::task::current().min(crate::task::MAX_TASKS - 1);
    INFLIGHT[t][1].store(a[0], Ordering::Relaxed);
    INFLIGHT[t][2].store(a[1], Ordering::Relaxed);
    INFLIGHT[t][3].store(a[2], Ordering::Relaxed);
    INFLIGHT[t][0].store(nr | 1 << 62, Ordering::Release);
}

pub fn leave() {
    let t = crate::task::current().min(crate::task::MAX_TASKS - 1);
    INFLIGHT[t][0].store(0, Ordering::Release);
}

pub fn inflight() {
    stacks();
    for t in 0..crate::task::MAX_TASKS {
        let nr = INFLIGHT[t][0].load(Ordering::Acquire);
        if nr == 0 {
            continue;
        }
        let nr = nr & !(1 << 62);
        crate::kprintln!(
            "  task {:>2} in {:<16} {:#x} {:#x} {:#x}",
            t,
            super::syscall::name_of(nr),
            INFLIGHT[t][1].load(Ordering::Relaxed),
            INFLIGHT[t][2].load(Ordering::Relaxed),
            INFLIGHT[t][3].load(Ordering::Relaxed)
        );
    }
}

/// The named return addresses on each task's user stack, taken when it
/// last blocked in `futex` -- what a stuck thread was waiting *for*, which
/// its call alone cannot say. Only while profiling: it costs a scan of the
/// stack and a lookup per word on every wait.
static STACKS: crate::sync::Racy<[alloc::vec::Vec<alloc::string::String>; crate::task::MAX_TASKS]> =
    crate::sync::Racy::new([const { alloc::vec::Vec::new() }; crate::task::MAX_TASKS]);

pub fn note_stack(sp: u64, rip: u64) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let t = crate::task::current().min(crate::task::MAX_TASKS - 1);
    let mut out = alloc::vec::Vec::new();
    if let Some(n) = super::syscall::name_code(rip) {
        out.push(alloc::format!("at {}", n));
    }
    for i in 0..768u64 {
        let at = sp + i * 4;
        if !super::syscall::reachable(at, 4, false) {
            break;
        }
        let w = unsafe { core::ptr::read_unaligned(at as *const u32) } as u64;
        if let Some(n) = super::syscall::name_code(w) {
            out.push(n);
            if out.len() >= 14 {
                break;
            }
        }
    }
    unsafe { STACKS.get()[t] = out };
}

pub fn stacks() {
    for t in 0..crate::task::MAX_TASKS {
        let s = unsafe { &STACKS.get()[t] };
        if s.is_empty() || INFLIGHT[t][0].load(Ordering::Acquire) == 0 {
            continue;
        }
        crate::kprintln!("  task {:>2}: {}", t, s.join(" < "));
    }
}
