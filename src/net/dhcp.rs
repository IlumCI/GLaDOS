//! DHCP: DISCOVER, OFFER, REQUEST, ACK.
//!
//! The four-message exchange of RFC 2131, which is what turns a machine that
//! has to be told its address into one that asks. Everything the rest of the
//! stack needs comes back in one reply: address, mask, router, and resolver.
//!
//! ### Sending before you have an address
//!
//! This is the protocol's one genuine oddity, and it shapes the code. The
//! client must transmit an IP packet in order to obtain an IP address, so
//! DISCOVER goes out from 0.0.0.0 to 255.255.255.255 -- which is why `net`
//! grew `send_ipv4_from` and why `resolve` short-circuits broadcast rather
//! than trying to ARP for it.
//!
//! The reply has the same problem in reverse: the server is answering a client
//! that cannot yet receive unicast, so we set the broadcast flag and accept
//! anything addressed to us while our own address is still unspecified.
//!
//! ### What is not here
//!
//! No lease renewal. The lease time is read and reported, and then nothing
//! watches it. A machine left running past its lease keeps using an address
//! the server believes is free -- fine for a session at a desk, wrong for
//! anything long-lived, and worth fixing before this is trusted on a network
//! it does not own.

use super::udp;
use super::{Config, Ipv4, BROADCAST_IP, UNSPECIFIED};
use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use alloc::vec::Vec;

const SERVER_PORT: u16 = 67;
const CLIENT_PORT: u16 = 68;

const OP_REQUEST: u8 = 1;
const OP_REPLY: u8 = 2;
const HTYPE_ETHERNET: u8 = 1;

const MAGIC: [u8; 4] = [99, 130, 83, 99];

const OPT_SUBNET_MASK: u8 = 1;
const OPT_ROUTER: u8 = 3;
const OPT_DNS: u8 = 6;
const OPT_REQUESTED_IP: u8 = 50;
const OPT_LEASE_TIME: u8 = 51;
const OPT_MESSAGE_TYPE: u8 = 53;
const OPT_SERVER_ID: u8 = 54;
const OPT_PARAM_LIST: u8 = 55;
const OPT_END: u8 = 255;

const DISCOVER: u8 = 1;
const OFFER: u8 = 2;
const REQUEST: u8 = 3;
const ACK: u8 = 5;
const NAK: u8 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    NoNic,
    NoOffer,
    NoAck,
    Refused,
    /// This task is already mid-exchange on the one UDP port.
    Busy,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::NoNic => "no NIC",
            Error::NoOffer => "no server offered a lease",
            Error::NoAck => "the server never confirmed",
            Error::Refused => "the server refused the request",
            Error::Busy => "this task is already using the UDP port",
        }
    }
}

#[derive(Default)]
struct Reply {
    kind: u8,
    yiaddr: Ipv4,
    mask: Option<Ipv4>,
    router: Option<Ipv4>,
    dns: Option<Ipv4>,
    server_id: Option<Ipv4>,
    lease: Option<u32>,
}

/// Build a BOOTP/DHCP message. The fixed part is 236 bytes whatever is in it.
fn message(kind: u8, xid: u32, mac: [u8; 6], requested: Option<Ipv4>, server: Option<Ipv4>, broadcast: bool) -> Vec<u8> {
    let mut m = Vec::with_capacity(300);
    m.push(OP_REQUEST);
    m.push(HTYPE_ETHERNET);
    m.push(6); // hardware address length
    m.push(0); // hops
    m.extend_from_slice(&xid.to_be_bytes());
    m.extend_from_slice(&0u16.to_be_bytes()); // seconds elapsed
    // Broadcast flag: a client that cannot take a unicast reply before it has
    // the address in it asks for broadcast. This stack can -- `addressed_to_us`
    // admits anything while the address is unspecified -- so it asks only on a
    // wired link. Over the air a broadcast goes at the lowest basic rate, under
    // the group key, and unacknowledged, so one lost is an offer lost; unicast
    // is retried by the access point until the station has it.
    m.extend_from_slice(&(if broadcast { 0x8000u16 } else { 0 }).to_be_bytes());
    m.extend_from_slice(&UNSPECIFIED); // ciaddr
    m.extend_from_slice(&UNSPECIFIED); // yiaddr
    m.extend_from_slice(&UNSPECIFIED); // siaddr
    m.extend_from_slice(&UNSPECIFIED); // giaddr
    m.extend_from_slice(&mac);
    m.extend_from_slice(&[0u8; 10]); // chaddr padding to 16
    m.extend_from_slice(&[0u8; 64]); // sname
    m.extend_from_slice(&[0u8; 128]); // file
    m.extend_from_slice(&MAGIC);

    m.push(OPT_MESSAGE_TYPE);
    m.push(1);
    m.push(kind);

    if let Some(ip) = requested {
        m.push(OPT_REQUESTED_IP);
        m.push(4);
        m.extend_from_slice(&ip);
    }
    if let Some(ip) = server {
        m.push(OPT_SERVER_ID);
        m.push(4);
        m.extend_from_slice(&ip);
    }

    m.push(OPT_PARAM_LIST);
    m.push(3);
    m.push(OPT_SUBNET_MASK);
    m.push(OPT_ROUTER);
    m.push(OPT_DNS);

    m.push(OPT_END);
    // Some servers ignore a BOOTP message shorter than the legacy 300 bytes.
    while m.len() < 300 {
        m.push(0);
    }
    m
}

fn parse(msg: &[u8], xid: u32) -> Option<Reply> {
    if msg.len() < 240 || msg[0] != OP_REPLY {
        return None;
    }
    if u32::from_be_bytes([msg[4], msg[5], msg[6], msg[7]]) != xid {
        // Another client's exchange on the same broadcast domain.
        return None;
    }
    if msg[236..240] != MAGIC {
        return None;
    }

    let mut r = Reply {
        yiaddr: [msg[16], msg[17], msg[18], msg[19]],
        ..Default::default()
    };

    let mut at = 240;
    while at < msg.len() {
        let code = msg[at];
        if code == OPT_END {
            break;
        }
        if code == 0 {
            at += 1; // pad
            continue;
        }
        if at + 1 >= msg.len() {
            break;
        }
        let len = msg[at + 1] as usize;
        let val = msg.get(at + 2..at + 2 + len)?;
        match code {
            OPT_MESSAGE_TYPE if len == 1 => r.kind = val[0],
            OPT_SUBNET_MASK if len == 4 => r.mask = Some([val[0], val[1], val[2], val[3]]),
            // Only the first router and the first resolver are kept; there is
            // one slot for each in Config.
            OPT_ROUTER if len >= 4 => r.router = Some([val[0], val[1], val[2], val[3]]),
            OPT_DNS if len >= 4 => r.dns = Some([val[0], val[1], val[2], val[3]]),
            OPT_SERVER_ID if len == 4 => r.server_id = Some([val[0], val[1], val[2], val[3]]),
            OPT_LEASE_TIME if len == 4 => {
                r.lease = Some(u32::from_be_bytes([val[0], val[1], val[2], val[3]]))
            }
            _ => {}
        }
        at += 2 + len;
    }
    Some(r)
}

/// Wait for a reply of the wanted type, ignoring anything else on the port.
fn await_reply(iface: usize, xid: u32, want: u8, ms: u64) -> Option<Reply> {
    let deadline = crate::dev::lapic::ticks() + (ms * crate::TIMER_HZ as u64) / 1000 + 1;
    loop {
        let remaining = deadline.saturating_sub(crate::dev::lapic::ticks());
        if remaining == 0 {
            return None;
        }
        let d = udp::recv(remaining * 1000 / crate::TIMER_HZ as u64)?;
        // Another link's server answering another link's client is not an
        // answer to this one, whatever its transaction id says.
        if d.src_port != SERVER_PORT || d.iface != iface {
            continue;
        }
        if let Some(r) = parse(&d.data, xid) {
            if r.kind == want {
                return Some(r);
            }
            if r.kind == NAK {
                return Some(r);
            }
        }
    }
}

/// Run the exchange and adopt whatever comes back.
pub fn configure_on(n: usize) -> Result<Config, Error> {
    let mac = match super::ifaces()[n].mac() {
        Some(m) => m,
        None => return Err(Error::NoNic),
    };
    let broadcast = !super::ifaces()[n].wireless;
    let xid = crate::time::rdtsc() as u32;

    // **The port first, and the address second.** A claim that is refused has
    // to leave the interface as it found it, so it is taken before the address
    // is given up rather than after -- the other order would return an error
    // with the machine unaddressed. See `udp::session`.
    let Some(udp_session) = udp::session(CLIENT_PORT) else {
        return Err(Error::Busy);
    };

    // Give up our address for the duration. It is not ours until the server
    // says so, and `addressed_to_us` lets everything through while it is
    // unspecified -- which is exactly what receiving the reply requires.
    let previous = super::config_of(n);
    let mut blank = previous;
    blank.ip = UNSPECIFIED;
    super::set_config_of(n, blank);

    let outcome = (|| {
        let discover = message(DISCOVER, xid, mac, None, None, broadcast);
        if !udp::send_on(n, UNSPECIFIED, BROADCAST_IP, SERVER_PORT, CLIENT_PORT, &discover) {
            return Err(Error::NoOffer);
        }
        let offer = await_reply(n, xid, OFFER, 4000).ok_or(Error::NoOffer)?;
        if offer.kind == NAK {
            return Err(Error::Refused);
        }

        let request = message(REQUEST, xid, mac, Some(offer.yiaddr), offer.server_id, broadcast);
        if !udp::send_on(n, UNSPECIFIED, BROADCAST_IP, SERVER_PORT, CLIENT_PORT, &request) {
            return Err(Error::NoAck);
        }
        let ack = await_reply(n, xid, ACK, 4000).ok_or(Error::NoAck)?;
        if ack.kind == NAK {
            return Err(Error::Refused);
        }

        // Anything the server did not supply keeps whatever was configured
        // before, rather than becoming zero.
        Ok((
            Config {
                ip: ack.yiaddr,
                netmask: ack.mask.unwrap_or(previous.netmask),
                gateway: ack.router.unwrap_or(previous.gateway),
                dns: ack.dns.unwrap_or(previous.dns),
            },
            ack.lease,
        ))
    })();

    drop(udp_session);

    match outcome {
        Ok((cfg, lease)) => {
            super::set_config_of(n, cfg);
            // The address was blanked for the exchange, so the setter cannot see
            // a move from the one held before; this can.
            if previous.ip != UNSPECIFIED && previous.ip != cfg.ip {
                super::tcp::abort_from(previous.ip);
                let i = &mut super::ifaces()[n];
                i.link_gen = i.link_gen.wrapping_add(1);
            }
            unsafe { LAST_LEASE = lease };
            Ok(cfg)
        }
        Err(e) => {
            // Put back what was working before rather than leaving the machine
            // with no address because a server did not answer.
            super::set_config_of(n, previous);
            Err(e)
        }
    }
}

static mut LAST_LEASE: Option<u32> = None;

/// The broadcast flag, which is the one field that differs by link.
pub fn checks() -> Vec<(&'static str, bool)> {
    let wired = message(DISCOVER, 1, [2; 6], None, None, true);
    let air = message(DISCOVER, 1, [2; 6], None, None, false);
    alloc::vec![(
        "DHCP asks for a broadcast reply on a wire and a unicast one over the air",
        wired[10..12] == [0x80, 0] && air[10..12] == [0, 0] && wired[..10] == air[..10],
    )]
}

pub fn report() {
    report_on(super::primary())
}

pub fn report_on(n: usize) {
    console::set_color(YELLOW);
    kprintln!("[dhcp] {}", super::ifaces()[n].name);
    console::set_color(LTGRAY);
    match configure_on(n) {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  {}", e.name());
            console::set_color(LTGRAY);
        }
        Ok(c) => {
            console::set_color(LTGREEN);
            kprintln!("  ip   {}.{}.{}.{}", c.ip[0], c.ip[1], c.ip[2], c.ip[3]);
            console::set_color(LTGRAY);
            kprintln!("  mask {}.{}.{}.{}", c.netmask[0], c.netmask[1], c.netmask[2], c.netmask[3]);
            kprintln!("  gw   {}.{}.{}.{}", c.gateway[0], c.gateway[1], c.gateway[2], c.gateway[3]);
            kprintln!("  dns  {}.{}.{}.{}", c.dns[0], c.dns[1], c.dns[2], c.dns[3]);
            match unsafe { LAST_LEASE } {
                Some(s) => kprintln!("  lease {} s  (not renewed -- see the module note)", s),
                None => kprintln!("  no lease time offered"),
            }
        }
    }
}
