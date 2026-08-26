//! The boot log: everything the machine printed, kept where it can be read
//! back.
//!
//! The console holds one screen. `Console::cells` is the visible grid and
//! there is no scrollback, so by the time anybody thinks to look at a boot
//! line it has already scrolled off and is gone from the machine entirely.
//! The ToDo runbook records what that costs: one of its steps is "read the
//! boot log (it scrolls past)" followed by an instruction to write a number
//! down by hand, because there was no other way to get it off the screen.
//!
//! So every byte that goes to the console and the serial port comes here as
//! well. A fixed array rather than a heap allocation, because this has to
//! work from the first `kprintln` of boot, which happens long before there is
//! an allocator to ask.
//!
//! When it wraps it keeps the newest bytes and says how many it dropped. The
//! alternative is stopping at the cap, which would lose the end of a long
//! session, and the end is where the interesting thing usually is.

use crate::sync::Racy;
use core::fmt;

/// Bytes retained. A full boot prints about ten kilobytes, so this holds a
/// boot plus a long session, and it is 0.02% of the smallest heap rung.
const CAP: usize = 64 * 1024;

struct Ring {
    buf: [u8; CAP],
    /// Where the next byte goes.
    head: usize,
    /// Total bytes ever written, so wrap and loss are both derivable.
    total: usize,
}

static RING: Racy<Ring> = Racy::new(Ring { buf: [0; CAP], head: 0, total: 0 });

/// Append one byte. Called for every character the machine prints.
#[inline]
fn put(r: &mut Ring, b: u8) {
    r.buf[r.head] = b;
    r.head = (r.head + 1) % CAP;
    r.total += 1;
}

struct Sink<'a>(&'a mut Ring);

impl fmt::Write for Sink<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.as_bytes() {
            put(self.0, *b);
        }
        Ok(())
    }
}

/// The third sink behind `kprint!`, beside the console and the serial port.
///
/// Deliberately outside the console's capture stack. A capture redirects
/// applet output to a caller that asked for it, and the log wants what the
/// machine actually said, whoever it was said to.
pub fn _record(args: fmt::Arguments) {
    use fmt::Write;
    let r = unsafe { &mut *RING.get() };
    let _ = Sink(r).write_fmt(args);
}

/// Bytes held, bytes ever printed, and bytes lost to wrapping.
pub fn stats() -> (usize, usize, usize) {
    let r = unsafe { &*RING.get() };
    let held = r.total.min(CAP);
    (held, r.total, r.total.saturating_sub(CAP))
}

/// The log in order, oldest retained byte first.
///
/// Copied out rather than handed over as two slices: every caller wants one
/// contiguous thing, and the copy is the price of not making each of them
/// reassemble a ring correctly.
pub fn contents() -> alloc::vec::Vec<u8> {
    let r = unsafe { &*RING.get() };
    let held = r.total.min(CAP);
    let mut out = alloc::vec::Vec::with_capacity(held);
    if r.total <= CAP {
        out.extend_from_slice(&r.buf[..r.head]);
    } else {
        out.extend_from_slice(&r.buf[r.head..]);
        out.extend_from_slice(&r.buf[..r.head]);
    }
    out
}

/// Where a saved log lands in the namespace.
///
/// The namespace and not a file on the ESP, because after `ExitBootServices`
/// there is no filesystem and the ESP on this machine is on the USB stick the
/// firmware booted from, which no driver here can reach. Getting it off the
/// machine is a separate problem from getting it off the screen, and this
/// solves the second one; `snap` and a provisioned store region solve the
/// first.
pub const PATH: &str = "/sys/boot.log";

/// Write the log into the namespace. Returns the byte count.
pub fn save(path: &str) -> Option<usize> {
    let bytes = contents();
    let n = bytes.len();
    if crate::sysbox::write_blob(path, bytes) {
        Some(n)
    } else {
        None
    }
}
