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
use core::sync::atomic::{AtomicBool, Ordering};

pub struct Racy<T> {
    inner: UnsafeCell<T>,
}

// The safety argument is "one core, and we mask interrupts around the users",
// not anything the compiler can verify.
unsafe impl<T> Sync for Racy<T> {}

impl<T> Racy<T> {
    pub const fn new(value: T) -> Self {
        Self { inner: UnsafeCell::new(value) }
    }

    /// # Safety
    /// Caller must ensure no other live reference exists. In practice: do not
    /// call this from an interrupt handler that can preempt another caller.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &mut T {
        unsafe { &mut *self.inner.get() }
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
/// Nothing about the holder is recorded, deliberately. Identifying a core
/// means reading the local interrupt controller over memory-mapped I/O, which
/// is far more expensive than the lock it would be describing, and the
/// allocator takes one of these on every allocation. The waiter and the lock's
/// address are enough to find the pair.
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
    ok
}
