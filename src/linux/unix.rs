//! Unix domain sockets, which is what wineserver talks over.
//!
//! Every Wine process opens `$WINEPREFIX/server/<id>/socket` and speaks to
//! wineserver across it; nothing about Wine starts until that connection is
//! made. It is also the first socket in this tree that is not a network
//! socket -- `net::tcp` carries one control block per connection and answers
//! `connect` by putting a segment on a wire, and none of that applies to two
//! processes on one machine passing bytes.
//!
//! **A connection is two byte queues and nothing else.** No addressing, no
//! sequence numbers, no retransmission, no MTU. That is not a simplification
//! of TCP, it is what a Unix socket *is*: the kernel is both endpoints, so the
//! only real questions are how much may be in flight and what happens when one
//! end goes away.
//!
//! ### What crossing a guest means here
//!
//! wineserver is a separate process from the program talking to it, so these
//! have to work between *guests* -- which is why the bound-name table is
//! global rather than living in a guest's entry. That is safe for the reason
//! `Fd` gives about `Rc`: one core, guests pinned to it, and two tasks that
//! interleave at switch points rather than running at once. It stops being
//! safe the day a guest is allowed onto a second core, and `smp.rs` already
//! says what that costs.
//!
//! ### What is deliberately not here yet
//!
//! `SCM_RIGHTS`, which is how a process passes a file descriptor over one of
//! these. Wine uses it -- wineserver hands out handles to shared memory that
//! way -- so this is not finished for Wine's purposes, and saying so is
//! cheaper than discovering it inside a failed startup.

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::sync::Racy;

/// How much may sit in one direction before a write blocks.
///
/// Linux's default is 208 KiB and nothing here needs that. What the number
/// has to be is *larger than any single message a client sends before reading
/// a reply*, or two ends both waiting to write deadlock -- which is the one
/// failure mode a buffer size can cause on its own.
const CAPACITY: usize = 64 * 1024;

/// One connection: a queue each way, and whether each end is still there.
pub struct Pipe {
    to_b: VecDeque<u8>,
    to_a: VecDeque<u8>,
    a_open: bool,
    b_open: bool,
}

impl Pipe {
    fn new() -> Pipe {
        Pipe { to_b: VecDeque::new(), to_a: VecDeque::new(), a_open: true, b_open: true }
    }
}

/// Which end of a `Pipe` a descriptor is.
///
/// A bool rather than two types, because every operation is the same one with
/// the two queues swapped -- and writing it twice is how they come to disagree.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

impl Side {
    fn other(self) -> Side {
        match self {
            Side::A => Side::B,
            Side::B => Side::A,
        }
    }
}

/// What a `AF_UNIX` descriptor currently is.
pub enum Sock {
    /// `socket()` and nothing more.
    Fresh,
    /// `bind()` without `listen()`. Holds the name so `listen` knows it.
    Bound(String),
    /// Accepting. The queue is connections `connect` left waiting.
    Listening { path: String, waiting: VecDeque<Rc<RefCell<Pipe>>> },
    /// Connected, either by `connect`, by `accept`, or by `socketpair`.
    Stream { pipe: Rc<RefCell<Pipe>>, side: Side },
}

/// Every listening name on the machine.
///
/// Global rather than per-guest, because the whole point is that two guests
/// find each other by it. A `Vec` because a handful of names is what a Wine
/// prefix has, and a map would be a second structure to keep in step for no
/// measurable gain.
static BOUND: Racy<Vec<(String, Rc<RefCell<Sock>>)>> = Racy::new(Vec::new());

/// Forget every bound name. Called when the last guest goes.
pub fn reset() {
    unsafe { BOUND.get() }.clear();
}

pub fn bound_names() -> usize {
    unsafe { BOUND.get() }.len()
}

/// Bind a name, or say why not.
pub fn bind(sock: &Rc<RefCell<Sock>>, path: &str) -> Result<(), i64> {
    const EINVAL: i64 = -22;
    const EADDRINUSE: i64 = -98;
    if path.is_empty() {
        return Err(EINVAL);
    }
    {
        let b = sock.borrow();
        if !matches!(*b, Sock::Fresh) {
            // Already bound, listening or connected. Linux answers `EINVAL`
            // for a second bind rather than `EADDRINUSE`, which is about the
            // socket rather than about the name.
            return Err(EINVAL);
        }
    }
    let table = unsafe { BOUND.get() };
    if table.iter().any(|(p, _)| p == path) {
        return Err(EADDRINUSE);
    }
    *sock.borrow_mut() = Sock::Bound(String::from(path));
    table.push((String::from(path), Rc::clone(sock)));
    Ok(())
}

/// Start accepting on a bound name.
pub fn listen(sock: &Rc<RefCell<Sock>>) -> Result<(), i64> {
    const EINVAL: i64 = -22;
    let mut b = sock.borrow_mut();
    let path = match &*b {
        Sock::Bound(p) => p.clone(),
        // Linux allows `listen` on an already-listening socket and this
        // follows, because a server that re-listens to change its backlog is
        // ordinary and refusing would be inventing a rule.
        Sock::Listening { .. } => return Ok(()),
        _ => return Err(EINVAL),
    };
    *b = Sock::Listening { path, waiting: VecDeque::new() };
    Ok(())
}

/// Connect to a bound, listening name. Answers this end of the new pipe.
///
/// **Completes immediately rather than waiting for `accept`.** Linux does the
/// same for a listening socket with room in its backlog: the connection is
/// made and the server picks it up later, which is what lets a client write
/// its first request before the server has looked.
pub fn connect(path: &str) -> Result<(Rc<RefCell<Pipe>>, Side), i64> {
    const ECONNREFUSED: i64 = -111;
    let table = unsafe { BOUND.get() };
    let Some((_, target)) = table.iter().find(|(p, _)| p == path) else {
        return Err(ECONNREFUSED);
    };
    let mut t = target.borrow_mut();
    let Sock::Listening { waiting, .. } = &mut *t else {
        // Bound but not listening. `ECONNREFUSED` is what Linux answers, and
        // it is the right one: the name exists and nothing is behind it.
        return Err(ECONNREFUSED);
    };
    let pipe = Rc::new(RefCell::new(Pipe::new()));
    waiting.push_back(Rc::clone(&pipe));
    Ok((pipe, Side::A))
}

/// Take the next waiting connection, if there is one.
pub fn accept(sock: &Rc<RefCell<Sock>>) -> Result<Option<(Rc<RefCell<Pipe>>, Side)>, i64> {
    const EINVAL: i64 = -22;
    let mut b = sock.borrow_mut();
    let Sock::Listening { waiting, .. } = &mut *b else {
        return Err(EINVAL);
    };
    // The server is the B end, because `connect` took A. Which is which does
    // not matter as long as the two never agree -- both ends taking the same
    // side would make every write land in the queue it was about to read.
    Ok(waiting.pop_front().map(|p| (p, Side::B)))
}

/// Two ends of one connection, for `socketpair`.
pub fn pair() -> (Rc<RefCell<Pipe>>, Rc<RefCell<Pipe>>) {
    let p = Rc::new(RefCell::new(Pipe::new()));
    (Rc::clone(&p), p)
}

/// Read from this end. `Ok(0)` means the far end is gone, as a socket has it.
pub fn read(pipe: &Rc<RefCell<Pipe>>, side: Side, out: &mut [u8]) -> Result<usize, i64> {
    const EAGAIN: i64 = -11;
    let mut p = pipe.borrow_mut();
    // Read the flag out before taking the queue, since the borrow checker is
    // right that one `&mut` and one `&` into the same struct is two borrows.
    let far_open = match side {
        Side::A => p.b_open,
        Side::B => p.a_open,
    };
    let q = match side {
        Side::A => &mut p.to_a,
        Side::B => &mut p.to_b,
    };
    if q.is_empty() {
        // **Empty and the far end open is not the same as empty and closed**,
        // and a reader cannot tell them apart from the byte count alone. Zero
        // is end of stream; a caller that got zero from a live peer would
        // conclude the connection had gone.
        return if far_open { Err(EAGAIN) } else { Ok(0) };
    }
    let n = out.len().min(q.len());
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = q.pop_front().unwrap_or(0);
        let _ = i;
    }
    Ok(n)
}

/// Write to this end. Short writes are real: a caller must look at the count.
pub fn write(pipe: &Rc<RefCell<Pipe>>, side: Side, data: &[u8]) -> Result<usize, i64> {
    const EPIPE: i64 = -32;
    const EAGAIN: i64 = -11;
    let mut p = pipe.borrow_mut();
    let far_open = match side {
        Side::A => p.b_open,
        Side::B => p.a_open,
    };
    if !far_open {
        // Linux also raises `SIGPIPE` here. This answers the error and does
        // not raise it, which is the half a program checking its return value
        // sees -- and the half that does not check would be killed by the
        // signal on Linux and merely lose the write here. Written down rather
        // than left to be discovered.
        return Err(EPIPE);
    }
    let q = match side {
        Side::A => &mut p.to_b,
        Side::B => &mut p.to_a,
    };
    let room = CAPACITY.saturating_sub(q.len());
    if room == 0 {
        return Err(EAGAIN);
    }
    let n = data.len().min(room);
    for b in &data[..n] {
        q.push_back(*b);
    }
    Ok(n)
}

/// How many bytes are readable at this end, for `poll`-shaped questions.
pub fn readable(pipe: &Rc<RefCell<Pipe>>, side: Side) -> usize {
    let p = pipe.borrow();
    match side {
        Side::A => p.to_a.len(),
        Side::B => p.to_b.len(),
    }
}

/// Mark one end closed. The other end reads what is left, then end of stream.
///
/// The queue is deliberately **not** cleared: bytes already written are the
/// far end's, and dropping them on close would lose a reply a server had
/// finished sending before it hung up.
pub fn close(pipe: &Rc<RefCell<Pipe>>, side: Side) {
    let mut p = pipe.borrow_mut();
    match side {
        Side::A => p.a_open = false,
        Side::B => p.b_open = false,
    }
}

/// Unbind a name when its listening socket goes.
pub fn unbind(path: &str) {
    unsafe { BOUND.get() }.retain(|(p, _)| p != path);
}

/// What `diag linux` asks of this.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    let listener = Rc::new(RefCell::new(Sock::Fresh));
    out.push(("a fresh socket binds a name", bind(&listener, "/tmp/t.sock").is_ok()));
    out.push((
        "and the same name refused a second time",
        bind(&Rc::new(RefCell::new(Sock::Fresh)), "/tmp/t.sock").err() == Some(-98),
    ));
    out.push((
        "connecting before listen is refused, since nothing is behind the name",
        connect("/tmp/t.sock").is_err(),
    ));
    out.push(("it listens", listen(&listener).is_ok()));
    out.push((
        "a name nobody bound is refused",
        connect("/tmp/nope.sock").err() == Some(-111),
    ));

    let Ok((cp, cs)) = connect("/tmp/t.sock") else {
        out.push(("a connection could be made", false));
        return out;
    };
    out.push(("a connection could be made", true));
    let Ok(Some((sp, ss))) = accept(&listener) else {
        out.push(("and accepted", false));
        return out;
    };
    out.push(("and accepted", true));
    // The two ends must disagree about which side they are, or every write
    // lands in the queue its own reader is about to drain.
    out.push(("the two ends take opposite sides", cs != ss));

    out.push(("a write reports what it took", write(&cp, cs, b"ping").is_ok_and(|n| n == 4)));
    let mut buf = [0u8; 8];
    out.push((
        "and the far end reads exactly it",
        read(&sp, ss, &mut buf).is_ok_and(|n| n == 4) && &buf[..4] == b"ping",
    ));
    out.push((
        "an empty queue with the peer alive is EAGAIN, not end of stream",
        read(&sp, ss, &mut buf) == Err(-11),
    ));
    // The reply goes the other way, which is the check that the queues are
    // two and not one: a single shared queue passes everything above this.
    out.push(("the reply travels the other way", write(&sp, ss, b"pong").is_ok()));
    out.push((
        "and does not come back to its sender",
        read(&sp, ss, &mut buf) == Err(-11),
    ));
    out.push((
        "while the other end has it",
        read(&cp, cs, &mut buf).is_ok_and(|n| n == 4) && &buf[..4] == b"pong",
    ));

    write(&cp, cs, b"last").ok();
    close(&cp, cs);
    out.push((
        "bytes written before a close are still delivered",
        read(&sp, ss, &mut buf).is_ok_and(|n| n == 4),
    ));
    out.push((
        "and then it reads as end of stream rather than blocking",
        read(&sp, ss, &mut buf) == Ok(0),
    ));
    out.push(("writing to a closed peer is EPIPE", write(&sp, ss, b"x") == Err(-32)));

    unbind("/tmp/t.sock");
    out.push(("the name is free once unbound", bound_names() == 0));
    out
}
