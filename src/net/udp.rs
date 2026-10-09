//! UDP: eight bytes of header and no state at all.
//!
//! Worth having not for itself but for what sits on it. DNS turns a name into
//! an address, and DHCP turns a network into a configuration -- between them
//! they are the difference between a machine you must describe by hand and one
//! that finds its own way onto a network. Neither needs a byte stream, an
//! acknowledgement, or a retransmission timer, which is why they were built on
//! this and not on TCP.
//!
//! Delivery is the same shape as TCP's: `deliver` only queues, and callers
//! drain the queue. See the re-entrancy note in `net`.
//!
//! One port is bound at a time. Every user of this so far is a
//! request/response exchange that runs to completion before the next one
//! starts, and a table of bound ports would be structure without a purpose.
//!
//! **That sentence was a premise nobody enforced, and a second task broke it.**
//! "Runs to completion before the next one starts" is true of one task. A miner
//! image resolves its pool's name in its own connect loop, and a shell that
//! resolves anything at the same moment called `bind` over the top of it: each
//! cleared the other's inbox, the shell received the miner's answer, and the
//! lookup failed as `malformed answer` -- then the miner's `unbind` took the port
//! from under the shell and everything after timed out. `BOUND` and `INBOX` are
//! `Racy`, which is interior mutability and not a lock.
//!
//! So the design stays and its precondition is now a claim: `session` holds the
//! one port for a whole exchange, another task yields until it is free, and the
//! port is given back when the `Session` drops. A table of ports would also fix
//! it, and would be the structure this note declined -- serialising costs nothing
//! for two protocols that are rare and fast, and keeps "one port" true rather
//! than merely usual.

use super::{send_ipv4, send_ipv4_from, transport_checksum, Ipv4, PROTO_UDP};
use crate::sync::Racy;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Bound on the queue: these are request/response protocols, so a reply that
/// arrives while nothing is listening is not worth keeping.
const MAX_QUEUED: usize = 16;

pub struct Datagram {
    pub src: Ipv4,
    pub src_port: u16,
    pub data: Vec<u8>,
    /// The interface it arrived on. DHCP asks one link for an address and must
    /// not take another link's answer.
    pub iface: usize,
}

static BOUND: Racy<Option<u16>> = Racy::new(None);
static INBOX: Racy<Vec<Datagram>> = Racy::new(Vec::new());

/// Which task holds the port, as `task::current() + 1`, so that zero is nobody.
static HOLDER: AtomicUsize = AtomicUsize::new(0);

/// The one UDP port, held for a whole request/response exchange.
///
/// Given back on drop, which is what makes it safe to hold across early
/// returns: the old pairing of `bind` and `unbind` relied on every path out of
/// a function reaching the `unbind`, and `?` is a path out.
pub struct Session {
    _private: (),
}

impl Drop for Session {
    fn drop(&mut self) {
        unbind();
        HOLDER.store(0, Ordering::Release);
    }
}

/// Take the one UDP port for an exchange, waiting while another task has it.
///
/// **It yields and never spins.** A holder can sit in `recv` for seconds
/// waiting on a resolver, and a `Spin` held that long reaches its patience
/// limit and panics -- which here would be one slow DNS server halting the
/// machine.
///
/// `None` if *this* task already holds it. Waiting would be waiting on itself
/// forever, and granting it would be the clobbering this exists to stop, so
/// nesting is refused and said so. Nothing nests today; the refusal is for the
/// day something does.
pub fn session(port: u16) -> Option<Session> {
    let me = crate::task::current() + 1;
    loop {
        match HOLDER.compare_exchange(0, me, Ordering::Acquire, Ordering::Relaxed) {
            Ok(_) => {
                bind(port);
                return Some(Session { _private: () });
            }
            Err(h) if h == me => return None,
            Err(_) => crate::task::yield_now(),
        }
    }
}

/// Whether some task holds the port. For the claims in `dns::selftest`.
pub fn held() -> bool {
    HOLDER.load(Ordering::Acquire) != 0
}

fn bind(port: u16) {
    unsafe {
        *BOUND.get() = Some(port);
        (*INBOX.get()).clear();
    }
}

fn unbind() {
    unsafe {
        *BOUND.get() = None;
        (*INBOX.get()).clear();
    }
}

/// Queue a datagram addressed to the bound port. Called from `net::poll`.
pub fn deliver(iface: usize, src: Ipv4, dst: Ipv4, segment: &[u8]) {
    if segment.len() < 8 {
        return;
    }
    let src_port = u16::from_be_bytes([segment[0], segment[1]]);
    let dst_port = u16::from_be_bytes([segment[2], segment[3]]);
    let length = u16::from_be_bytes([segment[4], segment[5]]) as usize;
    if length < 8 || length > segment.len() {
        return;
    }
    let segment = &segment[..length];

    // A zero checksum means the sender did not compute one, which IPv4 allows.
    // Anything else has to be right.
    let sent = u16::from_be_bytes([segment[6], segment[7]]);
    if sent != 0 && transport_checksum(src, dst, PROTO_UDP, segment) != 0 {
        return;
    }

    let Some(port) = (unsafe { *BOUND.get() }) else { return };
    if dst_port != port {
        return;
    }
    // Masked, and paired with the take in `recv`. This runs inside `poll`, so
    // `StackGuard` keeps another task out of *the stack* -- but `recv` reads
    // this same vector without entering the stack at all, so the guard does not
    // separate them. A tick landing mid-push, with `recv` then removing an
    // element, is two tasks in one `Vec`. Same argument as `tcp::at`.
    let data = segment[8..].to_vec();
    crate::cpu::without_interrupts(|| {
        let inbox = unsafe { &mut *INBOX.get() };
        if inbox.len() < MAX_QUEUED {
            inbox.push(Datagram { src, src_port, data, iface });
        }
    });
}

fn datagram(src: Ipv4, dst: Ipv4, src_port: u16, dst_port: u16, payload: &[u8]) -> Vec<u8> {
    let len = 8 + payload.len();
    let mut d = Vec::with_capacity(len);
    d.extend_from_slice(&src_port.to_be_bytes());
    d.extend_from_slice(&dst_port.to_be_bytes());
    d.extend_from_slice(&(len as u16).to_be_bytes());
    d.extend_from_slice(&[0, 0]); // checksum, filled below
    d.extend_from_slice(payload);

    let c = transport_checksum(src, dst, PROTO_UDP, &d);
    // A computed checksum of zero is transmitted as all ones. Zero on the wire
    // is reserved to mean "not computed", so sending it would tell the peer to
    // skip the check that just succeeded. The two are equal in one's
    // complement, which is what makes the substitution legal.
    let c = if c == 0 { 0xFFFF } else { c };
    d[6..8].copy_from_slice(&c.to_be_bytes());
    d
}

pub fn send(dst: Ipv4, dst_port: u16, src_port: u16, payload: &[u8]) -> bool {
    // The source is whichever interface routing picks, not "the" address --
    // with more than one interface those differ, and a checksum computed over
    // the wrong one is rejected silently at the far end.
    let src = super::local_addr_for(dst);
    let d = datagram(src, dst, src_port, dst_port, payload);
    send_ipv4(dst, PROTO_UDP, &d)
}

/// Send with an explicit source address, for DHCP.
pub fn send_from(src: Ipv4, dst: Ipv4, dst_port: u16, src_port: u16, payload: &[u8]) -> bool {
    let d = datagram(src, dst, src_port, dst_port, payload);
    send_ipv4_from(src, dst, PROTO_UDP, &d)
}

/// Send out of one named interface, for DHCP. See `net::send_ipv4_on`.
pub fn send_on(iface: usize, src: Ipv4, dst: Ipv4, dst_port: u16, src_port: u16, payload: &[u8]) -> bool {
    let d = datagram(src, dst, src_port, dst_port, payload);
    super::send_ipv4_on(iface, src, dst, PROTO_UDP, &d)
}

/// Wait for a datagram on the bound port.
///
/// Idles on `hlt` for the same reason `tcp::wait_until` does: there is nothing
/// to do until a packet or a tick arrives.
pub fn recv(timeout_ms: u64) -> Option<Datagram> {
    let deadline =
        crate::dev::lapic::ticks() + (timeout_ms * crate::TIMER_HZ as u64) / 1000 + 1;
    loop {
        for _ in 0..16 {
            if matches!(super::poll(), super::Event::None) {
                break;
            }
        }
        // The other half of the pair in `deliver`. `remove(0)` shifts the whole
        // vector, which is the worst possible thing to be interrupted in the
        // middle of while another task is pushing to it.
        let got = crate::cpu::without_interrupts(|| {
            let inbox = unsafe { &mut *INBOX.get() };
            if inbox.is_empty() {
                None
            } else {
                Some(inbox.remove(0))
            }
        });
        if got.is_some() {
            return got;
        }
        if crate::dev::lapic::ticks() >= deadline {
            return None;
        }
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

/// An ephemeral source port, drawn from the TSC.
pub fn ephemeral_port() -> u16 {
    49152 + (crate::time::rdtsc() as u16 % 16384)
}
