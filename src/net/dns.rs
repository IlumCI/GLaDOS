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
    /// This task is already mid-exchange on the one UDP port. See `udp::session`.
    Busy,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::NoNic => "no NIC",
            Error::Timeout => "no answer from the resolver",
            Error::Refused => "the resolver refused",
            Error::NotFound => "no such name",
            Error::Malformed => "malformed answer",
            Error::Busy => "this task is already using the UDP port",
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

/// How long one attempt listens for its answer.
const TRY_MS: u64 = 2000;

/// What one received datagram means to a lookup in progress.
#[derive(Debug, PartialEq)]
enum Heard {
    /// Not the answer to this question -- another server, another port, or
    /// another transaction id. Discarded, and the lookup keeps listening.
    Ignore,
    /// The answer to this question, whether it is good or not.
    Answer(Result<Ipv4, Error>),
}

/// Decide whether a datagram is the answer this lookup is waiting for.
///
/// An answer has to come from the server we asked, from port 53, carrying the
/// id we sent. The id is drawn from the TSC rather than counted, so together
/// those are three things an off-path forger has to guess, which is the whole
/// of a stub resolver's defence. A datagram failing any of them is somebody
/// else's, and is *ignored* -- `Malformed` is reserved for a reply that passes
/// all three and still does not parse, which is a real fault in the answer
/// rather than a stranger's packet.
///
/// Pure, so each of its answers is a claim in `selftest` with no network.
fn classify(d: &udp::Datagram, server: Ipv4, id: u16) -> Heard {
    if d.src != server || d.src_port != PORT {
        return Heard::Ignore;
    }
    // Too short to carry an id is too short to be known as ours. Ignoring it
    // rather than calling it malformed means a one-byte packet cannot end a
    // lookup either.
    if d.data.len() < 2 || u16::from_be_bytes([d.data[0], d.data[1]]) != id {
        return Heard::Ignore;
    }
    Heard::Answer(parse_answer(&d.data, id))
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
    // Held until this function returns, on every path, so nothing else can bind
    // over the top of a lookup in flight. See `udp::session`.
    let Some(_udp) = udp::session(port) else {
        return Err(Error::Busy);
    };

    // The resolver of the interface the default route uses, which is the
    // network a query actually leaves on -- not whichever interface `primary`
    // prefers by index.
    let server = super::dns_server();
    let mut result = Err(Error::Timeout);
    // Two attempts: UDP has no retransmission of its own, and a lost query is
    // indistinguishable from a slow one.
    'tries: for _ in 0..2 {
        if !udp::send(server, PORT, port, &query) {
            result = Err(Error::Timeout);
            continue;
        }
        // **Keep listening until this attempt's time is up, whatever arrives.**
        // The loop used to take one datagram per attempt: a packet from anywhere
        // else consumed the attempt and resent, and a packet from the right
        // server with the wrong id ended the whole lookup as `Malformed`. That is
        // backwards for a stub resolver -- an answer to somebody else's question
        // is exactly what an off-path forger sends, and the defence against one
        // is to ignore it, not to give up. Giving up hands anybody who can reach
        // the port a way to make every lookup fail.
        let deadline = crate::dev::lapic::ticks() + (TRY_MS * crate::TIMER_HZ as u64) / 1000 + 1;
        loop {
            let now = crate::dev::lapic::ticks();
            if now >= deadline {
                break;
            }
            let left_ms = ((deadline - now) * 1000 / crate::TIMER_HZ as u64).max(1);
            match udp::recv(left_ms) {
                None => break,
                Some(d) => match classify(&d, server, id) {
                    Heard::Ignore => continue,
                    Heard::Answer(r) => {
                        result = r;
                        break 'tries;
                    }
                },
            }
        }
    }
    drop(_udp);

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

/// Drop the cached answer. A name resolved on one network may be another
/// address -- a captive portal's, or a private one -- on the next.
pub fn forget() {
    unsafe { *CACHE.get() = None };
}

pub fn cached() -> Option<(String, Ipv4)> {
    unsafe { (*CACHE.get()).clone() }
}

fn check(ok: &mut bool, what: &str, good: bool) {
    if !good {
        *ok = false;
    }
    crate::kprintln!("  {}  {}", if good { "ok  " } else { "FAIL" }, what);
}

/// Which datagrams a lookup accepts, and the one UDP port being held.
///
/// **DNS, UDP and DHCP had no claims at all**, and what found that was a miner
/// image where every lookup the shell made failed -- first `malformed answer`,
/// then `no answer from the resolver` -- while the same query from the host
/// resolved in a quarter of a second. Two defects stacked. The shell and the
/// miner's connect loop both bound the one UDP port, each clearing the other's
/// inbox; and a lookup that received the other's answer treated a wrong
/// transaction id as a fatal `Malformed` rather than as a stranger's packet.
pub fn selftest() -> bool {
    const SERVER: Ipv4 = [10, 0, 2, 3];
    const ID: u16 = 0x1234;
    // A real answer to "a.b": one question, one A record pointing at the
    // question's name by compression, 10.0.2.99.
    let answer = |id: u16, flags: u16| -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&id.to_be_bytes());
        m.extend_from_slice(&flags.to_be_bytes());
        m.extend_from_slice(&[0, 1, 0, 1, 0, 0, 0, 0]);
        m.extend_from_slice(&[1, b'a', 1, b'b', 0, 0, 1, 0, 1]);
        m.extend_from_slice(&[0xC0, 0x0C, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 10, 0, 2, 99]);
        m
    };
    let from = |src: Ipv4, src_port: u16, data: Vec<u8>| udp::Datagram { src, src_port, data, iface: 0 };

    let mut ok = true;
    check(
        &mut ok,
        "our server's answer with our id gives the address it carries",
        classify(&from(SERVER, PORT, answer(ID, 0x8180)), SERVER, ID) == Heard::Answer(Ok([10, 0, 2, 99])),
    );
    check(
        &mut ok,
        "the same answer from another server is ignored",
        classify(&from([10, 0, 2, 4], PORT, answer(ID, 0x8180)), SERVER, ID) == Heard::Ignore,
    );
    check(
        &mut ok,
        "and from our server on a port other than 53",
        classify(&from(SERVER, 5353, answer(ID, 0x8180)), SERVER, ID) == Heard::Ignore,
    );
    // The packet that broke it: right server, right port, somebody else's id.
    check(
        &mut ok,
        "an answer to another question is ignored, and not called malformed",
        classify(&from(SERVER, PORT, answer(ID ^ 1, 0x8180)), SERVER, ID) == Heard::Ignore,
    );
    check(
        &mut ok,
        "a datagram too short to carry an id cannot end a lookup",
        classify(&from(SERVER, PORT, alloc::vec![0x12]), SERVER, ID) == Heard::Ignore,
    );
    // Ours by server, port and id, promising a question it does not contain.
    // Ignoring this would hide a genuinely broken resolver behind a timeout.
    let mut truncated = answer(ID, 0x8180);
    truncated.truncate(12);
    check(
        &mut ok,
        "but a reply that is ours and does not parse is still reported malformed",
        classify(&from(SERVER, PORT, truncated), SERVER, ID) == Heard::Answer(Err(Error::Malformed)),
    );
    check(
        &mut ok,
        "and NXDOMAIN from our server is no such name",
        classify(&from(SERVER, PORT, answer(ID, 0x8183)), SERVER, ID) == Heard::Answer(Err(Error::NotFound)),
    );

    // The session. Taken on this task, so the claims are about this task; the
    // cross-task case is the one driven on a miner image, where a second task
    // genuinely contends.
    match udp::session(40_000) {
        Some(s) => {
            check(&mut ok, "a session takes the one UDP port", udp::held());
            // If nesting waited instead of refusing, this line would never be
            // reached: the task would yield forever to itself.
            check(
                &mut ok,
                "the same task asking again is refused rather than waiting on itself",
                udp::session(40_001).is_none(),
            );
            drop(s);
            let again = udp::session(40_002);
            check(&mut ok, "and dropping it gives the port back", again.is_some());
        }
        None => check(&mut ok, "a session takes the one UDP port", false),
    }
    ok
}
