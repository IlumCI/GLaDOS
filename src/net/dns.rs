//! DNS: enough of RFC 1035 to turn a name into an address.
//!
//! Queries A records only, over UDP, one question per message. No cache beyond
//! a single entry, no recursion of our own -- the configured server is asked to
//! do the walking, which is what the recursion-desired bit means.
//!
//! ### Name compression is the part that bites
//!
//! A name in a DNS message is a sequence of length-prefixed labels ending in a
//! zero byte -- except that any label may instead be a two-byte pointer, marked
//! by its top two bits being set, giving an offset from the *start of the
//! message* where the rest of the name lives. Answers almost always use one,
//! because the name being answered was already written out in the question.
//!
//! So a parser cannot walk an answer by adding up label lengths; it has to
//! recognise pointers, and it has to bound how many it will follow. A message
//! whose pointer points at itself is a loop, and it costs one byte to write.

use super::udp;
use super::Ipv4;
use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

const PORT: u16 = 53;
const TYPE_A: u16 = 1;
const CLASS_IN: u16 = 1;

/// A pointer may point at a name containing another pointer. Legal, and also
/// how a malicious message tries to make the parser loop forever.
const MAX_POINTERS: usize = 8;

/// One entry, like the ARP cache. A resolver that never evicts is a resolver
/// that eventually hands out an address that has moved.
static CACHE: Racy<Option<(String, Ipv4)>> = Racy::new(None);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    NoNic,
    Timeout,
    Refused,
    NotFound,
    Malformed,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::NoNic => "no NIC",
            Error::Timeout => "no answer from the resolver",
            Error::Refused => "the resolver refused",
            Error::NotFound => "no such name",
            Error::Malformed => "malformed answer",
        }
    }
}

/// Encode "example.com" as `7example3com0`.
fn encode_name(name: &str, out: &mut Vec<u8>) -> bool {
    for label in name.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
    true
}

/// Step over a name, returning the offset just past it.
///
/// Only the *length* is wanted, never the text: the question is echoed back
/// and the answer's name is whatever we asked about. Following the pointer to
/// read it would be work in service of a comparison already made.
fn skip_name(msg: &[u8], mut at: usize) -> Option<usize> {
    let mut hops = 0;
    loop {
        let len = *msg.get(at)? as usize;
        if len == 0 {
            return Some(at + 1);
        }
        if len & 0xC0 == 0xC0 {
            // A pointer ends the name here, however long the thing it points
            // at turns out to be.
            msg.get(at + 1)?;
            return Some(at + 2);
        }
        if len > 63 {
            return None;
        }
        at += 1 + len;
        hops += 1;
        if hops > MAX_POINTERS * 8 {
            return None;
        }
    }
}

fn build_query(name: &str, id: u16) -> Option<Vec<u8>> {
    let mut q = Vec::with_capacity(32 + name.len());
    q.extend_from_slice(&id.to_be_bytes());
    // Recursion desired. We are a stub resolver; the server does the walking.
    q.extend_from_slice(&0x0100u16.to_be_bytes());
    q.extend_from_slice(&1u16.to_be_bytes()); // one question
    q.extend_from_slice(&0u16.to_be_bytes()); // no answers
    q.extend_from_slice(&0u16.to_be_bytes()); // no authority
    q.extend_from_slice(&0u16.to_be_bytes()); // no additional
    if !encode_name(name, &mut q) {
        return None;
    }
    q.extend_from_slice(&TYPE_A.to_be_bytes());
    q.extend_from_slice(&CLASS_IN.to_be_bytes());
    Some(q)
}

fn parse_answer(msg: &[u8], id: u16) -> Result<Ipv4, Error> {
    if msg.len() < 12 {
        return Err(Error::Malformed);
    }
    if u16::from_be_bytes([msg[0], msg[1]]) != id {
        // Someone else's answer, or an off-path guess at ours.
        return Err(Error::Malformed);
    }
    let flags = u16::from_be_bytes([msg[2], msg[3]]);
    match flags & 0x000F {
        0 => {}
        3 => return Err(Error::NotFound), // NXDOMAIN
        5 => return Err(Error::Refused),
        _ => return Err(Error::Malformed),
    }
    let qdcount = u16::from_be_bytes([msg[4], msg[5]]) as usize;
    let ancount = u16::from_be_bytes([msg[6], msg[7]]) as usize;

    let mut at = 12;
    for _ in 0..qdcount {
        at = skip_name(msg, at).ok_or(Error::Malformed)?;
        at += 4; // question type and class
        if at > msg.len() {
            return Err(Error::Malformed);
        }
    }

    for _ in 0..ancount {
        at = skip_name(msg, at).ok_or(Error::Malformed)?;
        if at + 10 > msg.len() {
            return Err(Error::Malformed);
        }
        let rtype = u16::from_be_bytes([msg[at], msg[at + 1]]);
        let rclass = u16::from_be_bytes([msg[at + 2], msg[at + 3]]);
        let rdlen = u16::from_be_bytes([msg[at + 8], msg[at + 9]]) as usize;
        at += 10;
        if at + rdlen > msg.len() {
            return Err(Error::Malformed);
        }
        // Skip anything that is not the A record asked for -- a CNAME chain
        // usually arrives first, with the address record behind it.
        if rtype == TYPE_A && rclass == CLASS_IN && rdlen == 4 {
            return Ok([msg[at], msg[at + 1], msg[at + 2], msg[at + 3]]);
        }
        at += rdlen;
    }
    Err(Error::NotFound)
}

/// Resolve a name to an address, asking the configured server.
pub fn resolve(name: &str) -> Result<Ipv4, Error> {
    if !super::ready() {
        return Err(Error::NoNic);
    }
    if let Some((cached, ip)) = unsafe { (*CACHE.get()).clone() } {
        if cached == name {
            return Ok(ip);
        }
    }

    let id = crate::time::rdtsc() as u16;
    let query = build_query(name, id).ok_or(Error::Malformed)?;
    let port = udp::ephemeral_port();
    udp::bind(port);

    let server = super::config().dns;
    let mut result = Err(Error::Timeout);
    // Two attempts: UDP has no retransmission of its own, and a lost query is
    // indistinguishable from a slow one.
    for _ in 0..2 {
        if !udp::send(server, PORT, port, &query) {
            result = Err(Error::Timeout);
            continue;
        }
        match udp::recv(2000) {
            None => continue,
            Some(d) => {
                // An answer has to come from the server we asked, from the
                // port we asked on. Together with the transaction id being
                // drawn from the TSC, that is three things an off-path forger
                // has to guess -- which is the whole of a stub resolver's
                // defence, and the reason the id is not a counter.
                if d.src != server || d.src_port != PORT {
                    continue;
                }
                result = parse_answer(&d.data, id);
                break;
            }
        }
    }
    udp::unbind();

    if let Ok(ip) = result {
        unsafe { *CACHE.get() = Some((String::from(name), ip)) };
    }
    result
}

/// Accept either a dotted address or a name.
///
/// Everything that takes a host goes through here, so `ping example.com` and
/// `ping 10.0.2.2` are the same command.
pub fn lookup(host: &str) -> Result<Ipv4, Error> {
    match super::parse_ip(host) {
        Some(ip) => Ok(ip),
        None => resolve(host),
    }
}

pub fn cached() -> Option<(String, Ipv4)> {
    unsafe { (*CACHE.get()).clone() }
}
