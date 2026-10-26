//! `eventfd` and `timerfd`: a counter and a clock, each behind a descriptor.
//!
//! **Neither existed, and modern event loops are built on both.** An eventfd
//! is how one thread wakes another's `poll` (Qt, Mesa, libinput, GLib when it
//! can); a timerfd is how a loop waits for time to pass alongside its sockets
//! (foot's key repeat, every Wayland compositor and terminal, systemd's
//! libraries). GLib falls back to a pipe without the first, which is why
//! LibreOffice never noticed; foot does not fall back, refused to create its
//! seat and exited.
//!
//! Both are readiness objects first: `poll` and `epoll` ask `readable`, and a
//! read is what consumes the state. Time is `syscall::mono_ns`, the clock the
//! guest reads as `CLOCK_MONOTONIC`, so a deadline a program computed from
//! that clock lands where it expects.

use alloc::vec::Vec;

/// A 64-bit counter: writes add, a read takes it.
pub struct EventFd {
    pub count: u64,
    /// `EFD_SEMAPHORE`: a read takes one rather than everything.
    pub semaphore: bool,
    pub nonblock: bool,
}

impl EventFd {
    pub fn readable(&self) -> bool {
        self.count > 0
    }

    /// Take what a read takes, or nothing if there is nothing to take.
    pub fn take(&mut self) -> Option<u64> {
        if self.count == 0 {
            return None;
        }
        if self.semaphore {
            self.count -= 1;
            Some(1)
        } else {
            Some(core::mem::take(&mut self.count))
        }
    }

    /// Add to the counter. The largest value is `u64::MAX - 1`, and a write
    /// that would pass it is refused -- Linux blocks a writer there, and no
    /// program in practice waits on a counter that full.
    pub fn add(&mut self, v: u64) -> Result<(), ()> {
        if v == u64::MAX {
            return Err(());
        }
        let next = self.count.checked_add(v).ok_or(())?;
        if next == u64::MAX {
            return Err(());
        }
        self.count = next;
        Ok(())
    }
}

/// A deadline with an optional period.
pub struct TimerFd {
    /// The guest clock this timer was made against; absolute times are read
    /// on it. Kept so `settime` with `TFD_TIMER_ABSTIME` means what it says.
    pub clock: u64,
    /// Monotonic nanoseconds of the next expiry, or zero for disarmed.
    pub next: u64,
    pub interval: u64,
    /// Expirations not yet read.
    pub pending: u64,
    pub nonblock: bool,
}

impl TimerFd {
    pub fn new(clock: u64, nonblock: bool) -> TimerFd {
        TimerFd { clock, next: 0, interval: 0, pending: 0, nonblock }
    }

    /// Count expirations up to `now`. A periodic timer that was not read for
    /// several periods counts every one, which is what a reader adds up.
    pub fn advance(&mut self, now: u64) {
        if self.next == 0 || now < self.next {
            return;
        }
        if self.interval == 0 {
            self.pending += 1;
            self.next = 0;
        } else {
            let periods = (now - self.next) / self.interval + 1;
            self.pending += periods;
            self.next += periods * self.interval;
        }
    }

    pub fn readable(&mut self, now: u64) -> bool {
        self.advance(now);
        self.pending > 0
    }

    pub fn take(&mut self, now: u64) -> Option<u64> {
        self.advance(now);
        (self.pending > 0).then(|| core::mem::take(&mut self.pending))
    }

    /// Arm or disarm. `value` is relative to `now` unless `abs` gives it as
    /// the absolute deadline already converted to monotonic nanoseconds.
    /// Answers the old setting as (remaining, interval).
    pub fn set(&mut self, now: u64, value: u64, interval: u64, abs: Option<u64>) -> (u64, u64) {
        let old = self.get(now);
        self.pending = 0;
        self.interval = interval;
        self.next = match (value, abs) {
            (0, _) => 0,
            (_, Some(at)) => at.max(1),
            (v, None) => now.saturating_add(v).max(1),
        };
        old
    }

    /// (remaining until the next expiry, interval), as `timerfd_gettime`.
    pub fn get(&mut self, now: u64) -> (u64, u64) {
        self.advance(now);
        let left = if self.next == 0 { 0 } else { self.next.saturating_sub(now).max(1) };
        (left, self.interval)
    }
}

/// What `diag linux` asks of both.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let mut e = EventFd { count: 0, semaphore: false, nonblock: true };
    out.push(("an empty eventfd is not readable and a read takes nothing", !e.readable() && e.take().is_none()));
    let _ = e.add(3);
    let _ = e.add(4);
    out.push(("writes add up, and a read takes the whole count", e.take() == Some(7) && !e.readable()));
    let mut s = EventFd { count: 0, semaphore: true, nonblock: true };
    let _ = s.add(2);
    out.push(("a semaphore eventfd gives one at a time", s.take() == Some(1) && s.take() == Some(1) && s.take().is_none()));
    out.push(("u64::MAX is refused as a value", e.add(u64::MAX).is_err()));

    let mut t = TimerFd::new(1, true);
    out.push(("a disarmed timer never fires", !t.readable(u64::MAX / 2)));
    t.set(1_000, 500, 0, None);
    out.push(("a one-shot timer is not due early", !t.readable(1_499)));
    out.push(("and fires once, then disarms", t.take(1_500) == Some(1) && !t.readable(10_000)));
    t.set(0, 100, 100, None);
    out.push(("a periodic timer counts every period it was not read for", t.take(1_000) == Some(10)));
    out.push(("and keeps its phase", t.get(1_050) == (50, 100)));
    t.set(0, 0, 0, None);
    out.push(("a zero value disarms", t.get(5_000) == (0, 0)));
    t.set(0, 1, 0, Some(2_000));
    out.push(("an absolute deadline lands where it was given", !t.readable(1_999) && t.readable(2_000)));
    out
}
