//! `poll` and `select`: which descriptors are ready, and waiting until one is.
//!
//! Every program that talks to more than one thing at once is built on this.
//! Wine's server loop, SDL's event pump and every Wayland client sit inside a
//! `poll` waiting for whichever of their descriptors speaks first, so this is
//! the one call all three are blocked on, and it is why it comes before the
//! compositor rather than after.
//!
//! ### Most descriptors here are always ready, and that is correct
//!
//! It reads like a shortcut and it is what Linux does. `poll` on a regular
//! file always returns ready, because there is nothing to wait for: a read at
//! end of file answers zero immediately, and zero is an answer. The same goes
//! for `/dev/null`, `/dev/zero`, the framebuffer and the standard descriptors.
//!
//! The descriptors that can genuinely make a program wait are few, and they
//! are the whole reason this module exists: a Unix socket with no bytes in it,
//! a listening socket with nobody connected, an input device with no events,
//! and a TCP connection with nothing received.
//!
//! ### `poll` and `select` count differently
//!
//! `poll` answers **how many descriptors** have a non-zero `revents`, so one
//! that is ready to read and to write counts once. `select` answers **how many
//! bits** are set across its three sets, so that same descriptor counts twice.
//! Both numbers look perfectly plausible, and a program handed the wrong one
//! either loops on work it has already done or stops with work outstanding.
//! `count_poll` and `count_select` are separate functions for that reason, and
//! there is a claim below holding them apart on one descriptor.
//!
//! ### Two smaller asymmetries, both of which programs rely on
//!
//! A **negative** fd in a `pollfd` is skipped with its `revents` zeroed. That
//! is how a program switches one entry off without rebuilding its array, and a
//! server that treated it as an error would break every such caller.
//!
//! A descriptor that is **not open** is `POLLNVAL` in that entry rather than a
//! failed call: `poll` never fails because of one bad descriptor, it reports
//! which one. `select` genuinely does fail, with `EBADF`, because its sets have
//! nowhere to put a per-descriptor answer. Both halves are what a libc expects.
//!
//! ### The tick is the resolution
//!
//! Timeouts are measured against the timer interrupt at `TIMER_HZ`, which is
//! 100, so the granularity is ten milliseconds. A one-millisecond `poll` waits
//! up to ten. Saying so is better than the alternative of rounding it to zero,
//! which turns every short timed wait into a spin.

use alloc::vec::Vec;

use super::fs::Fd;

/// There is data to read.
pub const POLLIN: u16 = 0x001;
/// There is urgent data. Nothing here ever raises it.
pub const POLLPRI: u16 = 0x002;
/// A write would proceed.
pub const POLLOUT: u16 = 0x004;
/// Something went wrong on the descriptor itself.
pub const POLLERR: u16 = 0x008;
/// The far end has gone.
pub const POLLHUP: u16 = 0x010;
/// The descriptor is not open.
pub const POLLNVAL: u16 = 0x020;

/// One entry of a `poll` array, in Linux's layout.
///
/// Eight bytes: a signed 32-bit descriptor, then two 16-bit masks. The guest
/// writes the first two and the kernel writes only the third, which is what
/// lets a program keep one array across many calls.
pub const PFD_LEN: usize = 8;

/// What a descriptor can do this instant.
///
/// A pure question about the descriptor, asked again on every turn of the wait
/// loop rather than cached, because the whole point is that the answer changes
/// while the guest is not running.
pub fn ready(fd: &Fd) -> u16 {
    match fd {
        // Nothing types at a guest, so stdin is permanently at end of file --
        // and end of file is *ready*, since a read returns zero at once. A
        // server that reported it unready would hang a program waiting for
        // input that is already, definitively, not coming.
        Fd::Stdin => POLLIN,
        Fd::Stdout | Fd::Stderr => POLLOUT,
        Fd::File(f) => {
            let b = f.borrow();
            if b.writable {
                POLLIN | POLLOUT
            } else {
                POLLIN
            }
        }
        Fd::Dir(_) => POLLIN,
        Fd::Socket(s) => {
            let b = s.borrow();
            match b.conn {
                Some(h) => {
                    let mut m = POLLOUT;
                    if crate::net::tcp::pending(h) > 0 {
                        m |= POLLIN;
                    }
                    m
                }
                // A socket with no connection can do neither. Reporting it
                // writable would send a program into a `write` that cannot
                // work.
                None => 0,
            }
        }
        Fd::Unix(s) => {
            let b = s.borrow();
            match &*b {
                // A listening socket is *readable* when somebody is waiting,
                // which is the protocol `accept` is written against: a server
                // polls the listener and calls accept when it says ready.
                super::unix::Sock::Listening { waiting, .. } => {
                    if waiting.is_empty() {
                        0
                    } else {
                        POLLIN
                    }
                }
                super::unix::Sock::Stream { pipe, side } => {
                    let mut m = 0;
                    if super::unix::readable(pipe, *side) > 0 {
                        m |= POLLIN;
                    }
                    if super::unix::room(pipe, *side) > 0 {
                        m |= POLLOUT;
                    }
                    if !super::unix::peer_open(pipe, *side) {
                        // Buffered bytes survive the peer leaving, so `POLLIN`
                        // and `POLLHUP` together is the ordinary state at the
                        // end of a connection rather than a contradiction: a
                        // program is expected to drain what is left.
                        m |= POLLHUP;
                    }
                    m
                }
                // Made but not connected to anything.
                _ => 0,
            }
        }
        Fd::Dev(d) => {
            let b = d.borrow();
            match b.node {
                super::dev::Node::Input(dev) => {
                    if super::input::pending(dev, b.at as u64) {
                        POLLIN
                    } else {
                        0
                    }
                }
                super::dev::Node::Fb => POLLOUT,
                super::dev::Node::Null => POLLIN | POLLOUT,
                super::dev::Node::Zero => POLLIN | POLLOUT,
                super::dev::Node::Random => POLLIN,
            }
        }
    }
}

/// Narrow a readiness mask to what this entry asked about.
///
/// `POLLERR`, `POLLHUP` and `POLLNVAL` are reported whether or not they were
/// requested, which is in the specification and is load-bearing: a program
/// polling only for `POLLIN` still has to find out that the connection ended,
/// and masking those off would leave it waiting on a descriptor that will
/// never speak again.
pub fn wanted(state: u16, events: u16) -> u16 {
    (state & events) | (state & (POLLERR | POLLHUP | POLLNVAL))
}

/// How many descriptors `poll` should report.
///
/// One per entry with anything set, however many bits that entry carries.
pub fn count_poll(revents: &[u16]) -> usize {
    revents.iter().filter(|r| **r != 0).count()
}

/// How many bits `select` should report.
///
/// One per set bit across all three sets, so a descriptor ready to read and to
/// write counts twice. This is the other rule, and the pair of them is why
/// these are two functions.
pub fn count_select(per_fd: &[u16]) -> usize {
    per_fd
        .iter()
        .map(|m| {
            (*m & POLLIN != 0) as usize
                + (*m & POLLOUT != 0) as usize
                + (*m & (POLLERR | POLLPRI) != 0) as usize
        })
        .sum()
}

/// How many bytes of `fd_set` cover descriptors `0..nfds`.
///
/// Rounded up to a whole byte. The guest's own `fd_set` is 128 bytes whatever
/// `nfds` says, and reading or writing more than `nfds` covers would touch
/// bits the program did not offer.
pub fn set_bytes(nfds: usize) -> usize {
    nfds.div_ceil(8)
}

/// Whether descriptor `i` is in the set.
///
/// Bit `i` is bit `i % 8` of byte `i / 8`, which is the layout `FD_SET`
/// builds and the only one a libc will recognise.
pub fn set_get(set: &[u8], i: usize) -> bool {
    match set.get(i / 8) {
        Some(b) => b & (1 << (i % 8)) != 0,
        None => false,
    }
}

/// Put descriptor `i` in the set.
pub fn set_put(set: &mut [u8], i: usize) {
    if let Some(b) = set.get_mut(i / 8) {
        *b |= 1 << (i % 8);
    }
}

/// A timeout in timer ticks, or `None` for no limit.
///
/// Milliseconds arrive from `poll` and a `timeval` from `select`, and both end
/// up here. Rounded **up**, because a timeout is a floor on how long to wait:
/// rounding a one-millisecond wait down to zero ticks turns a timed poll into
/// a busy loop, which is the failure the granularity note above is about.
pub fn ticks_for_ms(ms: i64) -> Option<u64> {
    if ms < 0 {
        return None;
    }
    let hz = crate::TIMER_HZ as i64;
    Some(((ms * hz) + 999) as u64 / 1000)
}

/// What `diag linux` asks of readiness.
pub fn checks() -> Vec<(&'static str, bool)> {
    use super::unix::{self, Side};
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // ---- the standard descriptors ----
    out.push((
        "stdout can be written and not read",
        ready(&Fd::Stdout) == POLLOUT,
    ));
    out.push((
        "stdin reads ready, because end of file is an answer rather than a wait",
        ready(&Fd::Stdin) == POLLIN,
    ));

    // ---- a Unix socket, which is the one that can actually block ----
    let (a, b) = unix::pair();
    let sa = alloc::rc::Rc::new(core::cell::RefCell::new(unix::Sock::Stream {
        pipe: a.clone(),
        side: Side::A,
    }));
    let sb = alloc::rc::Rc::new(core::cell::RefCell::new(unix::Sock::Stream {
        pipe: b.clone(),
        side: Side::B,
    }));
    out.push((
        "a fresh socket pair can be written and has nothing to read",
        ready(&Fd::Unix(sa.clone())) == POLLOUT,
    ));
    unix::write(&a, Side::A, b"hello").ok();
    out.push((
        "and the far end reads ready once bytes are in it",
        ready(&Fd::Unix(sb.clone())) & POLLIN != 0,
    ));
    out.push((
        "while the sending end still has nothing of its own to read",
        ready(&Fd::Unix(sa.clone())) & POLLIN == 0,
    ));
    unix::close(&a, Side::A);
    let hung = ready(&Fd::Unix(sb.clone()));
    out.push((
        "a departed peer shows as a hangup, so a program learns of it without attempting a read",
        hung & POLLHUP != 0,
    ));
    out.push((
        "and the bytes it left behind are still readable, which is the ordinary end of a connection rather than a contradiction",
        hung & POLLIN != 0,
    ));

    // ---- a listening socket ----
    let listener = alloc::rc::Rc::new(core::cell::RefCell::new(unix::Sock::Listening {
        path: alloc::string::String::from("/tmp/p.sock"),
        waiting: alloc::collections::VecDeque::new(),
    }));
    out.push((
        "a listening socket with nobody waiting is not ready",
        ready(&Fd::Unix(listener.clone())) == 0,
    ));
    if let unix::Sock::Listening { waiting, .. } = &mut *listener.borrow_mut() {
        waiting.push_back(unix::pair().0);
    }
    out.push((
        "and reads ready once a connection is queued, which is the signal accept is written against",
        ready(&Fd::Unix(listener.clone())) == POLLIN,
    ));

    // ---- what an entry is told ----
    out.push((
        "an entry hears only about what it asked for",
        wanted(POLLIN | POLLOUT, POLLIN) == POLLIN,
    ));
    out.push((
        "except a hangup, which arrives whether or not it was asked for, or a program polling only to read never learns the connection ended",
        wanted(POLLIN | POLLHUP, POLLIN) == (POLLIN | POLLHUP)
            && wanted(POLLHUP, POLLOUT) == POLLHUP,
    ));

    // ---- the two counting rules ----
    let both = [POLLIN | POLLOUT];
    out.push((
        "poll counts one descriptor once however many ways it is ready",
        count_poll(&both) == 1,
    ));
    out.push((
        "select counts the bits, so the same descriptor counts twice, and the two numbers are both plausible",
        count_select(&both) == 2,
    ));
    out.push((
        "a descriptor ready for nothing is counted by neither",
        count_poll(&[0]) == 0 && count_select(&[0]) == 0,
    ));

    // ---- fd_set arithmetic ----
    let mut set = [0u8; 16];
    set_put(&mut set, 0);
    set_put(&mut set, 7);
    set_put(&mut set, 8);
    set_put(&mut set, 63);
    out.push((
        "a descriptor sits at its own bit of its own byte, the layout FD_SET builds",
        set[0] == 0b1000_0001 && set[1] == 0b0000_0001 && set[7] == 0b1000_0000,
    ));
    out.push((
        "and reads back where it was put",
        set_get(&set, 0) && set_get(&set, 7) && set_get(&set, 8) && set_get(&set, 63),
    ));
    out.push((
        "a bit nobody set is clear",
        !set_get(&set, 1) && !set_get(&set, 9) && !set_get(&set, 62),
    ));
    out.push((
        "a descriptor past the end of the set reads clear rather than faulting",
        !set_get(&set, 4096),
    ));
    out.push((
        "the set covers whole bytes, rounded up, so descriptor 8 needs two",
        set_bytes(1) == 1 && set_bytes(8) == 1 && set_bytes(9) == 2 && set_bytes(0) == 0,
    ));

    // ---- timeouts ----
    out.push((
        "a negative timeout is no limit at all",
        ticks_for_ms(-1).is_none(),
    ));
    out.push((
        "a zero timeout is zero ticks, which is a look rather than a wait",
        ticks_for_ms(0) == Some(0),
    ));
    out.push((
        "a timeout shorter than a tick rounds up to one, since rounding it down would make a timed poll a busy loop",
        ticks_for_ms(1) == Some(1) && ticks_for_ms(9) == Some(1),
    ));
    out.push((
        "and a whole second is the tick rate",
        ticks_for_ms(1000) == Some(crate::TIMER_HZ as u64),
    ));

    out
}
