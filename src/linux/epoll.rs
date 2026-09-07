//! `epoll`: the same question as `poll`, asked of a set the kernel keeps.
//!
//! `poll` hands the kernel its whole list on every call, which is fine for the
//! handful of descriptors a program usually watches and is why it came first.
//! `epoll` exists for the case where the list is thousands long and barely
//! changes: the set lives in the kernel, `epoll_ctl` edits it, and
//! `epoll_wait` returns only what is ready. Everything modern reaches for it,
//! which is why a machine with `poll` alone still fails a lot of software.
//!
//! Underneath it is the same `poll::ready` and nothing else. That is not a
//! shortcut -- Linux's own event bits were chosen to be the poll bits, so
//! `EPOLLIN` *is* `POLLIN`, and the readiness question has one answer however
//! it is asked.
//!
//! ### `struct epoll_event` is twelve bytes, and that is the trap
//!
//! It is a `uint32_t` and a 64-bit union, which every ordinary structure rule
//! says should be padded to sixteen. On x86-64 the header declares it
//! **packed**, so `data` sits at offset 4 and the whole thing is 12. A server
//! that laid it out the natural way would hand back a `data` field read from
//! the wrong place -- a pointer the program never registered, which it will
//! then dereference. There is a claim below pinning the size and the offset.
//!
//! ### Level-triggered, including for `EPOLLET`
//!
//! Edge triggering is accepted and behaves level-triggered, which is a
//! deviation in cost rather than in correctness and is worth being exact
//! about. An edge-triggered program reads until `EAGAIN` and then waits; under
//! level triggering it is told again whenever anything is left, so it reads,
//! finds nothing, and waits again. More wakeups than it asked for, never
//! fewer, and never a missed event. Refusing the flag instead would stop the
//! program dead, which is a worse answer to the same gap.
//!
//! `EPOLLONESHOT` **is** honoured, because that one changes what is correct: a
//! thread pool hands one event to one worker on the strength of it, and an
//! entry reported twice would be worked twice.

use alloc::vec::Vec;

/// `EPOLL_CTL_*`.
pub const CTL_ADD: u64 = 1;
pub const CTL_DEL: u64 = 2;
pub const CTL_MOD: u64 = 3;

/// The bits that mean what they mean in `poll`.
pub const EPOLLIN: u32 = 0x001;
pub const EPOLLOUT: u32 = 0x004;
pub const EPOLLERR: u32 = 0x008;
pub const EPOLLHUP: u32 = 0x010;

/// Report this entry once, then stop until the program re-arms it.
pub const EPOLLONESHOT: u32 = 1 << 30;
/// Edge triggering. Accepted, and level-triggered underneath. See above.
pub const EPOLLET: u32 = 1 << 31;

/// `sizeof(struct epoll_event)` on x86-64, where the header packs it.
pub const EV_LEN: usize = 12;
/// Where the 64-bit `data` sits inside one. Four, and not eight.
pub const EV_DATA_AT: usize = 4;

/// One watched descriptor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Watch {
    pub fd: u64,
    /// What the program asked to hear about, plus its flag bits.
    pub events: u32,
    /// The program's own cookie, handed back untouched. Usually a pointer.
    pub data: u64,
    /// Whether a one-shot entry has already fired.
    pub spent: bool,
}

/// One `epoll` instance.
///
/// A `Vec` rather than a map, and the reason is the same one `mem::fixed`
/// gives: the sets a program actually registers here are tens of entries, a
/// map would be a second structure to keep in step, and the scan happens once
/// per wait rather than once per message.
#[derive(Default)]
pub struct Epoll {
    watch: Vec<Watch>,
}

/// Errors, as the numbers Linux answers with.
pub const EEXIST: i64 = -17;
pub const ENOENT: i64 = -2;
pub const EINVAL: i64 = -22;

impl Epoll {
    pub fn new() -> Epoll {
        Epoll { watch: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.watch.len()
    }

    pub fn is_empty(&self) -> bool {
        self.watch.is_empty()
    }

    pub fn entries(&self) -> &[Watch] {
        &self.watch
    }

    fn at(&self, fd: u64) -> Option<usize> {
        self.watch.iter().position(|w| w.fd == fd)
    }

    /// `EPOLL_CTL_ADD`, which refuses a descriptor already in the set.
    ///
    /// `EEXIST` rather than a quiet replacement: a program adding twice has
    /// lost track of its own set, and silently taking the second registration
    /// would hand it one event where it is expecting to manage two.
    pub fn add(&mut self, fd: u64, events: u32, data: u64) -> Result<(), i64> {
        if self.at(fd).is_some() {
            return Err(EEXIST);
        }
        self.watch.push(Watch { fd, events, data, spent: false });
        Ok(())
    }

    /// `EPOLL_CTL_MOD`, which is also how a one-shot entry is re-armed.
    ///
    /// Clearing `spent` here is the whole of that mechanism, and it is why
    /// `MOD` on an unchanged mask is a meaningful call rather than a no-op.
    pub fn modify(&mut self, fd: u64, events: u32, data: u64) -> Result<(), i64> {
        match self.at(fd) {
            Some(i) => {
                self.watch[i].events = events;
                self.watch[i].data = data;
                self.watch[i].spent = false;
                Ok(())
            }
            None => Err(ENOENT),
        }
    }

    pub fn remove(&mut self, fd: u64) -> Result<(), i64> {
        match self.at(fd) {
            Some(i) => {
                self.watch.remove(i);
                Ok(())
            }
            None => Err(ENOENT),
        }
    }

    /// Mark a one-shot entry as fired.
    pub fn spend(&mut self, fd: u64) {
        if let Some(i) = self.at(fd) {
            if self.watch[i].events & EPOLLONESHOT != 0 {
                self.watch[i].spent = true;
            }
        }
    }
}

/// What one entry should report, given what its descriptor can do.
///
/// `None` means say nothing about it this time round. The flag bits are masked
/// out of the answer: `EPOLLET` and `EPOLLONESHOT` are instructions to the
/// server rather than conditions, and handing them back would have the program
/// test its own request against readiness it never asked about.
pub fn report(w: &Watch, state: u16) -> Option<u32> {
    if w.spent {
        return None;
    }
    let state = state as u32;
    // Errors and hangups arrive whether or not they were requested, exactly as
    // in `poll`, or a program watching only for readable never learns the far
    // end has gone.
    let hit = (state & w.events & !(EPOLLET | EPOLLONESHOT)) | (state & (EPOLLERR | EPOLLHUP));
    if hit == 0 {
        None
    } else {
        Some(hit)
    }
}

/// Lay one `epoll_event` into a buffer, in the packed layout.
pub fn put_event(out: &mut [u8], i: usize, events: u32, data: u64) -> bool {
    let at = match i.checked_mul(EV_LEN) {
        Some(v) => v,
        None => return false,
    };
    if at + EV_LEN > out.len() {
        return false;
    }
    out[at..at + 4].copy_from_slice(&events.to_le_bytes());
    out[at + EV_DATA_AT..at + EV_LEN].copy_from_slice(&data.to_le_bytes());
    true
}

/// Read one back, which the claims use and `epoll_ctl` needs.
pub fn get_event(buf: &[u8]) -> Option<(u32, u64)> {
    if buf.len() < EV_LEN {
        return None;
    }
    let events = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let mut d = [0u8; 8];
    d.copy_from_slice(&buf[EV_DATA_AT..EV_LEN]);
    Some((events, u64::from_le_bytes(d)))
}

/// What `diag linux` asks of the event set.
pub fn checks() -> Vec<(&'static str, bool)> {
    use super::poll::{POLLHUP, POLLIN, POLLOUT};
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // ---- the layout, which is the thing most easily got wrong ----
    out.push((
        "an epoll_event is twelve bytes, because x86-64's header packs it where every ordinary rule would pad it to sixteen",
        EV_LEN == 12,
    ));
    out.push((
        "and its data sits at four, so a server that padded would hand back a cookie read from the wrong place",
        EV_DATA_AT == 4,
    ));
    let mut buf = [0u8; EV_LEN * 2];
    put_event(&mut buf, 0, EPOLLIN, 0xDEAD_BEEF_1234_5678);
    put_event(&mut buf, 1, EPOLLOUT, 0x1122_3344_5566_7788);
    out.push((
        "two events sit back to back with no gap",
        get_event(&buf) == Some((EPOLLIN, 0xDEAD_BEEF_1234_5678))
            && get_event(&buf[EV_LEN..]) == Some((EPOLLOUT, 0x1122_3344_5566_7788)),
    ));
    out.push((
        "an event past the end of the buffer is refused rather than written over whatever follows",
        !put_event(&mut buf, 2, EPOLLIN, 0),
    ));

    // ---- the bits are poll's bits, which is why one readiness answer serves ----
    out.push((
        "the event bits are poll's own, so the readiness question has one answer however it is asked",
        EPOLLIN as u16 == POLLIN && EPOLLOUT as u16 == POLLOUT && EPOLLHUP as u16 == POLLHUP,
    ));

    // ---- the set ----
    let mut e = Epoll::new();
    out.push(("a new set is empty", e.is_empty()));
    out.push(("a descriptor can be added", e.add(3, EPOLLIN, 0xAA).is_ok()));
    out.push((
        "adding it twice is refused, since a program doing that has lost track of its own set",
        e.add(3, EPOLLIN, 0xBB) == Err(EEXIST),
    ));
    out.push((
        "modifying one that is not there is refused",
        e.modify(9, EPOLLIN, 0) == Err(ENOENT),
    ));
    out.push((
        "and so is removing one",
        e.remove(9) == Err(ENOENT),
    ));
    out.push(("removing a watched descriptor works", e.remove(3).is_ok()));
    out.push(("and the set is empty again", e.is_empty()));

    // ---- what an entry reports ----
    let w = Watch { fd: 3, events: EPOLLIN, data: 7, spent: false };
    out.push((
        "an entry hears only about what it asked for",
        report(&w, POLLIN | POLLOUT) == Some(EPOLLIN),
    ));
    out.push((
        "and nothing at all when its descriptor is not ready that way",
        report(&w, POLLOUT).is_none(),
    ));
    out.push((
        "a hangup arrives whether or not it was requested, or a reader never learns the far end went",
        report(&w, POLLHUP) == Some(EPOLLHUP),
    ));
    let flagged = Watch { fd: 3, events: EPOLLIN | EPOLLET | EPOLLONESHOT, data: 7, spent: false };
    out.push((
        "the flag bits are instructions rather than conditions, so they are not handed back as readiness",
        report(&flagged, POLLIN) == Some(EPOLLIN),
    ));

    // ---- one-shot, which genuinely changes what is correct ----
    let mut e = Epoll::new();
    e.add(4, EPOLLIN | EPOLLONESHOT, 0x55).ok();
    let first = report(&e.entries()[0], POLLIN);
    e.spend(4);
    let second = report(&e.entries()[0], POLLIN);
    out.push((
        "a one-shot entry reports once",
        first == Some(EPOLLIN),
    ));
    out.push((
        "and then stays quiet, so a pool cannot hand one event to two workers",
        second.is_none(),
    ));
    e.modify(4, EPOLLIN | EPOLLONESHOT, 0x55).ok();
    out.push((
        "re-arming through MOD is what wakes it again, which is why MOD on an unchanged mask is a real call",
        report(&e.entries()[0], POLLIN) == Some(EPOLLIN),
    ));
    let mut e = Epoll::new();
    e.add(5, EPOLLIN, 0).ok();
    e.spend(5);
    out.push((
        "an entry without the flag is never spent, however often it fires",
        report(&e.entries()[0], POLLIN) == Some(EPOLLIN),
    ));

    out
}
