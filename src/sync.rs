//! Interior mutability for kernel globals.
//!
//! This kernel is single-core and ring-0 by design (see the M4 notes about
//! staying on the BSP), so there is genuinely no second thread of execution to
//! race with yet -- interrupts are the only reentrancy, and we mask them where
//! it matters. `Racy` says exactly that and nothing more.
//!
//! When SMP arrives this type is the thing that has to go. Every use of it is a
//! place that will need a real lock, so it is deliberately easy to grep for.

use core::cell::UnsafeCell;

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
