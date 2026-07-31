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

use super::{send_ipv4, send_ipv4_from, transport_checksum, Ipv4, PROTO_UDP};
use crate::sync::Racy;
use alloc::vec::Vec;

/// Bound on the queue: these are request/response protocols, so a reply that
/// arrives while nothing is listening is not worth keeping.
const MAX_QUEUED: usize = 16;

pub struct Datagram {
    pub src: Ipv4,
    pub src_port: u16,
    pub data: Vec<u8>,
}

static BOUND: Racy<Option<u16>> = Racy::new(None);
static INBOX: Racy<Vec<Datagram>> = Racy::new(Vec::new());

pub fn bind(port: u16) {
    unsafe {
        *BOUND.get() = Some(port);
        (*INBOX.get()).clear();
    }
}

pub fn unbind() {
    unsafe {
        *BOUND.get() = None;
        (*INBOX.get()).clear();
    }
}

/// Queue a datagram addressed to the bound port. Called from `net::poll`.
pub fn deliver(src: Ipv4, dst: Ipv4, segment: &[u8]) {
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
    let inbox = unsafe { &mut *INBOX.get() };
    if inbox.len() < MAX_QUEUED {
        inbox.push(Datagram {
            src,
            src_port,
            data: segment[8..].to_vec(),
        });
    }
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
    let src = super::config().ip;
    let d = datagram(src, dst, src_port, dst_port, payload);
    send_ipv4(dst, PROTO_UDP, &d)
}

/// Send with an explicit source address, for DHCP.
pub fn send_from(src: Ipv4, dst: Ipv4, dst_port: u16, src_port: u16, payload: &[u8]) -> bool {
    let d = datagram(src, dst, src_port, dst_port, payload);
    send_ipv4_from(src, dst, PROTO_UDP, &d)
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
        let inbox = unsafe { &mut *INBOX.get() };
        if !inbox.is_empty() {
            return Some(inbox.remove(0));
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
