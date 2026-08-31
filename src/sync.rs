//! Interior mutability, and a real lock for the things that now need one.
//!
//! Two types live here and the difference between them is the whole point.
//!
//! `Racy<T>` is not a lock. It is single-core interior mutability with a safety
//! argument that a human makes and the compiler cannot check: one core, and
//! interrupts masked where it matters. It stays because most kernel globals are
//! genuinely only ever touched by the bootstrap processor, and converting them
//! would be ceremony that hides which ones actually matter.
//!
//! `Spin<T>` is a lock. It is for state that more than one core reaches, and
//! every conversion from `Racy` to `Spin` is a claim that a second core can get
//! there. Those claims are made one at a time and each one is verified, because
//! converting everything at once would produce a kernel where nothing is known
//! to be right rather than one where a few things are.
//!
//! **`lock_irq` exists because a spinlock alone is not enough here.** A lock
//! taken by ordinary code and also by an interrupt handler on the same core
//! deadlocks against itself: the handler spins for a lock the code it
//! interrupted is holding, and that code cannot run to release it. The
//! allocator and the console are both in that position, so both take the
//! interrupt-masking form. A plain `lock` on such a structure is a latent hang
//! that appears under load and never in a test.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub struct Racy<T> {
    inner: UnsafeCell<T>,
    /// Which task last touched this, plus one. Zero means nobody yet.
    ///
    /// Only consulted while `audit::on()`. See `audit` for what it is for.
    owner: AtomicU32,
}

// The safety argument is "one core, and we mask interrupts around the users",
// not anything the compiler can verify.
unsafe impl<T> Sync for Racy<T> {}

impl<T> Racy<T> {
    pub const fn new(value: T) -> Self {
        Self { inner: UnsafeCell::new(value), owner: AtomicU32::new(0) }
    }

    /// # Safety
    /// Caller must ensure no other live reference exists. In practice: do not
    /// call this from an interrupt handler that can preempt another caller.
    #[allow(clippy::mut_from_ref)]
    #[track_caller]
    pub unsafe fn get(&self) -> &mut T {
        if audit::on() {
            audit::visit(&self.owner, core::panic::Location::caller());
        }
        unsafe { &mut *self.inner.get() }
    }
}

/// Finding out which of these are actually shared, by watching.
///
/// There are eighty-six of them. Reading all eighty-six and declaring which are
/// reachable from two tasks is exactly the kind of judgement that is wrong
/// somewhere and gives no sign of where, so this asks the running system
/// instead: every `Racy::get` records which task made it, and a second task
/// touching the same one is written down along with the line that did it.
///
/// **Off by default, and the cost is why.** A relaxed load and a branch on a
/// path this hot is affordable; the recording is not. The flag is checked first
/// and everything else hangs off it.
///
/// What this proves and what it does not. A site reported here is definitely
/// shared, which is a fact and is the useful direction. A site *not* reported
/// is unshared only along the paths that actually ran, so silence is evidence
/// rather than proof, which is why the report prints how much it saw.
pub mod audit {
    use core::panic::Location;
    use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    static ON: AtomicBool = AtomicBool::new(false);
    static VISITS: AtomicUsize = AtomicUsize::new(0);
    static SHARED: AtomicUsize = AtomicUsize::new(0);

    /// How many distinct offending lines are kept. Small on purpose: the same
    /// handful recur, and a report nobody can read is a report nobody reads.
    const SLOTS: usize = 24;

    #[allow(clippy::declare_interior_mutable_const)]
    const ZU: AtomicUsize = AtomicUsize::new(0);
    #[allow(clippy::declare_interior_mutable_const)]
    const Z32: AtomicU32 = AtomicU32::new(0);

    /// The file name's pointer. It is a string literal with static lifetime, so
    /// a pointer and a length are enough and keep a slot small.
    static FILES: [AtomicUsize; SLOTS] = [ZU; SLOTS];
    static NAMELEN: [AtomicU32; SLOTS] = [Z32; SLOTS];
    static LINES: [AtomicU32; SLOTS] = [Z32; SLOTS];
    static HITS: [AtomicUsize; SLOTS] = [ZU; SLOTS];
    static FIRST: [AtomicU32; SLOTS] = [Z32; SLOTS];
    static THEN: [AtomicU32; SLOTS] = [Z32; SLOTS];

    #[inline(always)]
    pub fn on() -> bool {
        ON.load(Ordering::Relaxed)
    }

    pub fn start() {
        VISITS.store(0, Ordering::Relaxed);
        SHARED.store(0, Ordering::Relaxed);
        for i in 0..SLOTS {
            FILES[i].store(0, Ordering::Relaxed);
            LINES[i].store(0, Ordering::Relaxed);
            HITS[i].store(0, Ordering::Relaxed);
        }
        ON.store(true, Ordering::Release);
    }

    pub fn stop() {
        ON.store(false, Ordering::Release);
    }

    /// Record a touch, and note it when the toucher changed.
    pub fn visit(owner: &AtomicU32, at: &'static Location<'static>) {
        // The task comes from per-core storage rather than from asking the
        // interrupt controller which core this is. An audit costing a
        // memory-mapped read per touch would change the timing it exists to
        // observe.
        let Some(task) = crate::cpu::percpu::billed() else { return };
        let me = task as u32 + 1;
        VISITS.fetch_add(1, Ordering::Relaxed);
        let was = owner.swap(me, Ordering::Relaxed);
        if was == 0 || was == me {
            return;
        }
        SHARED.fetch_add(1, Ordering::Relaxed);
        note(at, was - 1, me - 1);
    }

    fn note(at: &'static Location<'static>, first: u32, then: u32) {
        let f = at.file().as_ptr() as usize;
        let n = at.file().len() as u32;
        let l = at.line();
        for i in 0..SLOTS {
            let have = FILES[i].load(Ordering::Relaxed);
            if have == f && LINES[i].load(Ordering::Relaxed) == l {
                HITS[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            if have == 0
                && FILES[i].compare_exchange(0, f, Ordering::AcqRel, Ordering::Relaxed).is_ok()
            {
                NAMELEN[i].store(n, Ordering::Relaxed);
                LINES[i].store(l, Ordering::Relaxed);
                FIRST[i].store(first, Ordering::Relaxed);
                THEN[i].store(then, Ordering::Relaxed);
                HITS[i].store(1, Ordering::Relaxed);
                return;
            }
        }
    }

    pub fn report() {
        use crate::kprintln;
        let v = VISITS.load(Ordering::Relaxed);
        let s = SHARED.load(Ordering::Relaxed);
        kprintln!("  {} touches seen, {} of them by a second task", v, s);
        if v == 0 {
            kprintln!("  nothing ran while auditing. 'racy on', use the machine, then 'racy'");
            return;
        }
        let mut any = false;
        for i in 0..SLOTS {
            let f = FILES[i].load(Ordering::Relaxed);
            if f == 0 {
                continue;
            }
            any = true;
            let name = unsafe {
                core::str::from_utf8_unchecked(core::slice::from_raw_parts(
                    f as *const u8,
                    NAMELEN[i].load(Ordering::Relaxed) as usize,
                ))
            };
            kprintln!(
                "  {}:{}  tasks {} and {}, {} time(s)",
                name,
                LINES[i].load(Ordering::Relaxed),
                FIRST[i].load(Ordering::Relaxed),
                THEN[i].load(Ordering::Relaxed),
                HITS[i].load(Ordering::Relaxed)
            );
        }
        if !any {
            kprintln!("  no site was touched by two tasks on the paths that ran");
        }
    }

    /// Feed the recorder two different owners and check that it notices.
    ///
    /// A silent audit and a correct system look identical from outside, so the
    /// recorder has to be shown catching something before its silence is worth
    /// anything. Same argument the differential harness's canary makes.
    pub fn selftest_detects() -> bool {
        let probe = AtomicU32::new(0);
        let before = SHARED.load(Ordering::Relaxed);
        let was_on = on();
        ON.store(true, Ordering::Release);
        // A prior owner that is deliberately not this task. Storing 1 here
        // was the first attempt and it failed for a reason worth keeping: 1 is
        // task 0 plus one, and the suite runs on task 0, so `visit` correctly
        // saw no change of owner and the check called that a miss. The value
        // has to be a task this cannot be.
        probe.store(u32::MAX, Ordering::Relaxed);
        visit(&probe, Location::caller());
        let seen = SHARED.load(Ordering::Relaxed) > before;
        if !was_on {
            ON.store(false, Ordering::Release);
        }
        seen
    }
}

/// A spinlock, for state a second core can reach.
pub struct Spin<T> {
    locked: AtomicBool,
    inner: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for Spin<T> {}
unsafe impl<T: Send> Send for Spin<T> {}

/// How long to spin before deciding this is a deadlock rather than contention.
///
/// A held lock in this kernel covers a few hundred instructions at most, so any
/// wait this long is a bug. Reporting it beats hanging: a machine that stops
/// and says which core was waiting on which lock can be fixed, and one that
/// simply stops cannot.
///
/// Nothing about the holder is recorded, deliberately. Identifying a core means
/// reading the local controller over memory-mapped I/O, which costs far more
/// than the lock it would be describing, and the allocator takes one of these
/// on every allocation. The waiter and the lock's address are enough to find
/// the pair.
const PATIENCE: u32 = 200_000_000;

impl<T> Spin<T> {
    pub const fn new(value: T) -> Self {
        Self { locked: AtomicBool::new(false), inner: UnsafeCell::new(value) }
    }

    /// Take the lock, leaving interrupts as they are.
    ///
    /// Only correct for state no interrupt handler touches. Where a handler
    /// does, use `lock_irq`.
    pub fn lock(&self) -> Guard<'_, T> {
        self.acquire();
        Guard { lock: self, restore: false }
    }

    /// Take the lock with interrupts masked for the length of the critical
    /// section, restoring them on drop only if they were on to begin with.
    pub fn lock_irq(&self) -> Guard<'_, T> {
        let flags: u64;
        unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(preserves_flags)) };
        let was = flags & (1 << 9) != 0;
        crate::cpu::disable_interrupts();
        self.acquire();
        Guard { lock: self, restore: was }
    }

    /// Take it if it is free. Never spins, so it is safe from anywhere,
    /// including a fault handler that must not hang whatever else is true.
    pub fn try_lock(&self) -> Option<Guard<'_, T>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Some(Guard { lock: self, restore: false });
        }
        None
    }

    fn acquire(&self) {
        let mut spun: u32 = 0;
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // `pause` on every iteration, because a bare spin on a contended
            // line saturates the interconnect and starves the holder, making
            // the wait longer the more cores are waiting.
            core::hint::spin_loop();
            spun = spun.wrapping_add(1);
            if spun == PATIENCE {
                panic!(
                    "core {} deadlocked waiting on the lock at {:p}",
                    crate::smp::this_cpu(),
                    self
                );
            }
        }
    }
}

pub struct Guard<'a, T> {
    lock: &'a Spin<T>,
    restore: bool,
}

impl<T> core::ops::Deref for Guard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T> core::ops::DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.inner.get() }
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
        if self.restore {
            crate::cpu::enable_interrupts();
        }
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let l: Spin<u32> = Spin::new(7);
    {
        let mut g = l.lock();
        *g += 1;
        claim(&mut ok, l.try_lock().is_none(), "a held lock refuses a second taker");
    }
    claim(&mut ok, *l.lock() == 8, "and the value written under it survives");
    claim(&mut ok, l.try_lock().is_some(), "a released lock is free again");

    // Interrupt state is restored to what it was and not to what somebody
    // assumed it was. A lock that unconditionally enabled interrupts would
    // silently break every caller that had already masked them.
    crate::cpu::without_interrupts(|| {
        {
            let _g = l.lock_irq();
        }
        let flags: u64;
        unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(preserves_flags)) };
        claim(
            &mut ok,
            flags & (1 << 9) == 0,
            "lock_irq leaves interrupts masked when it found them masked",
        );
    });

    claim(
        &mut ok,
        audit::selftest_detects(),
        "the sharing audit notices a second toucher, so its silence means something",
    );
    ok
}
